"""Training, checkpoint evaluation, and paired-source experiment collection."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from .data import WindowDataset, fit_scaling, prepare_cache, verify_cache, window_loader
from .metrics import evaluate, source_bootstrap
from .model import EmbeddingForecaster
from .context_mixture import build_forecaster, call_forecaster, mixture_nll
from .augmentation import augment_history
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import write_metric_table
from .runtime import execution_settings, forecast_loader, implementation_hashes, check_resume_implementation
from src.project_runtime.paths import load_json, resolve_config, portable_config, resolve_path, machine


def forecast_directory(root):
    root = resolve_path(root)
    return root if (root / "config.json").is_file() else root / "technical"


def datasets(config, manifest):
    from .spatial import SpatialWindowDataset
    cls = SpatialWindowDataset if 'spatial_cache' in config['data'] else WindowDataset
    extra = {'spatial_root': config['data']['spatial_cache']} if cls is SpatialWindowDataset else {}
    return {split: cls(config['data']['cache'], manifest, split,
        config['history_ps'], config['horizons_ps'][-1], config['stride_ps'], config['anchor_history_ps'], **extra)
        for split in ('train', 'val', 'test')}


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.building')
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_scores(directory, model, metrics, cadence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = np.arange(1, model.output_steps + 1) * cadence
    if model.target == 'bin_means':
        times = np.array(model.edges[1:]) * cadence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [('mse_by_step', 'forecast'), ('persistence_mse_by_step', 'persistence'),
                       ('history_mean_mse_by_step', 'history mean'), ('linear_trend_mse_by_step', 'linear trend')]:
        axes[0].plot(times, metrics['curves'][key], marker='.', label=label)
    axes[0].set(xlabel='Future time / bin end (ps)', ylabel='Standardized embedding MSE')
    axes[0].legend()
    sources = list(metrics['per_source'].values())
    axes[1].bar(np.arange(len(sources)), [s['persistence_mse'] - s['mse'] for s in sources])
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set(xlabel='Held-out source', ylabel='Persistence MSE minus forecast MSE')
    fig.tight_layout()
    fig.savefig(directory / 'forecast_scores.png', dpi=160)
    plt.close(fig)


def evaluate_checkpoint(directory, device, *, runtime=None):
    runtime = execution_settings(runtime)
    directory = forecast_directory(directory)
    result_root = directory.parent if directory.name == 'technical' else directory
    result_folders(result_root)
    payload = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    config = resolve_config(payload['config'])
    root = Path(config['data']['cache'])
    if sha256(root / 'manifest.json') != payload['cache_manifest_sha256']:
        raise ValueError(f'Checkpoint embedding cache identity changed: {root}')
    manifest = verify_cache(root)
    data = datasets(config, manifest)
    model = build_forecaster(manifest['embedding_dim'], data['train'].history_steps,
        manifest['cadence_ps'], config['horizons_ps'], payload['variant']).to(device)
    model.load_state_dict(payload['model'], strict=True)
    mean, scale = payload['mean'].to(device), payload['scale'].to(device)
    settings = config['training']
    loader = forecast_loader(data['test'], settings, False, payload['seed'], device, runtime)
    # Sampling for probabilistic scores is reproducible and cannot affect any fitting RNG.
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(payload['seed'])
        metrics, rows, examples = evaluate(model, loader, mean, scale, device, sample_paths=True)
    metrics['selected_epoch'] = payload['epoch']
    metrics['criterion'] = payload['criterion']
    metrics['validation_selection_score'] = payload['selection_score']
    write_json(directory / 'test_metrics.json', metrics)
    np.savez_compressed(directory / 'test_errors.npz', **rows)
    np.savez_compressed(directory / 'test_examples.npz', **examples)
    del rows, examples
    interventions = {}
    for name in ('repeat_anchor', 'reverse_past'):
        values, _, _ = evaluate(model, loader, mean, scale, device, intervention=name, retain_rows=False)
        interventions[name] = values
    write_json(directory / 'history_interventions.json', interventions)
    plot_scores(result_root / 'plots', model, metrics, manifest['cadence_ps'])
    write_metric_table(metrics, result_root, family='forecast', name='forecast-scores')
    (result_root/'README.md').write_text('# Embedding forecast\n\n[Scores](tables/forecast-scores.csv) · '
        '[Metric definitions](tables/METRICS.md) · [Forecast plot](plots/forecast_scores.png)\n')
    write_json(directory / 'status.json', dict(state='complete', selected_epoch=payload['epoch']))
    return metrics


def train(config, variant, seed, device, *, resume=False, epochs_per_invocation=None,
          runtime=None, resume_transition=None):
    from .model import forecast_loss

    config = resolve_config(config)
    settings = config['training']
    runtime = execution_settings(runtime)
    if settings['epochs'] < 1 or settings['patience'] < 1:
        raise ValueError('epochs and patience must be positive.')
    if any(w < 0 for w in variant['loss'].values()) or sum(variant['loss'].values()) <= 0:
        raise ValueError(f"Forecast loss needs nonnegative weights and at least one active term: {variant['name']}")
    if epochs_per_invocation is not None and epochs_per_invocation < 1:
        raise ValueError('epochs_per_invocation must be positive.')
    if 'augmentation' in settings:
        augmentation = settings['augmentation']
        if augmentation['noise_std'] < 0 or not 0 <= augmentation['frame_dropout'] <= 1:
            raise ValueError(f'Invalid training history augmentation: {augmentation}')
    torch.set_num_threads(settings['cpu_threads'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    root = Path(config['data']['cache'])
    manifest = verify_cache(root)
    data = datasets(config, manifest)
    directory = Path(config['output']) / f"{variant['name']}-seed{seed}"
    if directory.exists() and not resume:
        raise FileExistsError(f'Fresh forecast fit would overwrite {directory}; evaluate it or choose a new output.')
    run_root = directory
    directory = forecast_directory(run_root)
    checkpoint = None
    if resume:
        saved = json.loads((directory / 'config.json').read_text())
        if portable_config(saved) != portable_config(dict(config=config, variant=variant, seed=seed)):
            raise ValueError(f'Resume must retain the exact scientific config, variant and seed: {directory}')
        if json.loads((directory / 'status.json').read_text())['state'] == 'complete':
            return json.loads((directory / 'test_metrics.json').read_text())
        checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if 'resume_state' not in checkpoint:
            raise ValueError(f'Checkpoint predates resumable forecast training: {directory / "last.pt"}')
        if checkpoint['cache_manifest_sha256'] != sha256(root / 'manifest.json'):
            raise ValueError(f'Resume cache identity changed: {root}')
    else:
        result_folders(run_root)
        write_json(directory / 'config.json', dict(config=config, variant=variant, seed=seed))
    write_json(directory / 'status.json', dict(state='training'))
    if checkpoint is None:
        if 'normalization_checkpoint' in config:
            normalization = torch.load(config['normalization_checkpoint'], map_location='cpu', weights_only=False)
            if normalization['cache_manifest_sha256'] != sha256(root / 'manifest.json'):
                raise ValueError('Normalization checkpoint uses a different embedding cache.')
            mean, scale = normalization['mean'].to(device), normalization['scale'].to(device)
            write_json(directory / 'normalization_provenance.json', dict(
                checkpoint=config['normalization_checkpoint'], sha256=sha256(Path(config['normalization_checkpoint']))))
            del normalization
        else:
            mean_np, scale_np = fit_scaling(data['train'], settings['scale_floor_fraction'])
            mean, scale = torch.from_numpy(mean_np).to(device), torch.from_numpy(scale_np).to(device)
    else:
        mean, scale = checkpoint['mean'].to(device), checkpoint['scale'].to(device)
    model = build_forecaster(manifest['embedding_dim'], data['train'].history_steps,
        manifest['cadence_ps'], config['horizons_ps'], variant).to(device)
    train_loader = forecast_loader(data['train'], settings, True, seed, device, runtime)
    validation_device = 'cpu' if runtime.get('validation_residency', 'device') == 'host' else device
    val_loader = forecast_loader(data['val'], settings, False, seed, validation_device, runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    if 'warmup_epochs' in settings:
        warmup = settings['warmup_epochs']
        if not 0 < warmup < settings['epochs']:
            raise ValueError('warmup_epochs must be positive and less than the total epoch budget.')
        def multiplier(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / (settings['epochs'] - warmup)
            return settings['minimum_lr_fraction'] + (1 - settings['minimum_lr_fraction']) * (1 + math.cos(math.pi * progress)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'],
            eta_min=settings['learning_rate'] * settings['minimum_lr_fraction'])
    criterion = 'nll' if model.distribution in ('low_rank_gaussian', 'trajectory_mixture') else 'mse'
    best, stale, step = float('inf'), 0, 0
    start_epoch = 0
    elapsed_before = 0.0
    started = time.perf_counter()
    cache_digest = sha256(root / 'manifest.json')
    implementation = implementation_hashes()
    if checkpoint is not None:
        check_resume_implementation(checkpoint, implementation, resume_transition, directory)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        state = checkpoint['resume_state']
        best, stale, step = state['best'], state['stale'], checkpoint['step']
        start_epoch = checkpoint['epoch'] + 1
        elapsed_before = state['elapsed_s']
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        train_loader.sampler.sampler.generator.set_state(checkpoint['sampler_rng'])
        # A crash can leave a log line newer than the last atomic checkpoint.
        lines = [line for line in (directory / 'training.jsonl').read_text().splitlines()
                 if json.loads(line)['epoch'] < start_epoch]
        (directory / 'training.jsonl').write_text(''.join(line + '\n' for line in lines))
    else:
        write_json(directory / 'data_summary.json', dict(windows={s: len(d) for s, d in data.items()},
            parameters=sum(p.numel() for p in model.parameters()), cadence_ps=manifest['cadence_ps'],
            history_steps=data['train'].history_steps, future_steps=data['train'].future_steps,
            cache_manifest_sha256=cache_digest, implementation_sha256=implementation,
            checkpoint_selection=f'lowest source-mean validation {criterion}',
            normalization='unique training embeddings; per-channel std with configured RMS-relative floor'))
    stop_epoch = settings['epochs'] if epochs_per_invocation is None else min(settings['epochs'], start_epoch + epochs_per_invocation)
    completed_epochs = start_epoch
    write_json(directory / 'runtime.json', dict(settings=runtime, started_from_epoch=start_epoch,
               implementation_sha256=implementation))
    write_json(directory / 'storage_locations.json', dict(machine=machine(), portable_config=portable_config(config)))
    with (directory / 'training.jsonl').open('a' if resume else 'x') as log:
        for epoch in range(start_epoch, stop_epoch):
            if stale >= settings['patience']:
                break
            model.train()
            totals, seen = {}, 0
            lr = optimizer.param_groups[0]['lr']
            epoch_started = time.perf_counter()
            for epoch_step, batch in enumerate(train_loader, 1):
                history = (batch['history'].to(device, non_blocking=True) - mean) / scale
                future = (batch['future'].to(device, non_blocking=True) - mean) / scale
                if 'augmentation' in settings:
                    history = augment_history(history, settings['augmentation'])
                optimizer.zero_grad(set_to_none=True)
                if (model.kind == 'autoregressive_gru' and
                        variant['autoregressive']['training'] == 'teacher_forcing'):
                    output = model.teacher_forced(history, future)
                else:
                    output = call_forecaster(model, history, batch, mean, scale)
                if model.distribution == 'trajectory_mixture':
                    if variant['loss']['bin_mse'] or variant['loss']['increment_mse']:
                        raise ValueError('Trajectory-mixture protocol uses explicit mean MSE and joint NLL terms.')
                    terms = dict(mse=(output['mean']-future).square().mean(), nll=mixture_nll(output, future).mean())
                    loss = variant['loss']['mse']*terms['mse']+variant['loss']['nll']*terms['nll']
                    terms['loss'] = loss
                else:
                    loss, terms = forecast_loss(model, output, future, history[:, -1], variant['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite forecast loss: {variant['name']}, seed={seed}, epoch={epoch}, step={step}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                for key, value in dict(terms, gradient_norm=gradient).items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(history)
                seen += len(history)
                step += 1
                if runtime['log_every_steps'] and epoch_step % runtime['log_every_steps'] == 0:
                    progress = dict(state='training', epoch=epoch, epoch_step=epoch_step,
                        epoch_steps=len(train_loader), step=step, total_epochs=settings['epochs'],
                        elapsed_epoch_s=time.perf_counter() - epoch_started, loader=runtime['loader'])
                    write_json(directory / 'progress.json', progress)
                    print(f"{variant['name']} epoch={epoch} batch={epoch_step}/{len(train_loader)} "
                          f"elapsed={progress['elapsed_epoch_s']:.1f}s loader={runtime['loader']}", flush=True)
            train_seconds = time.perf_counter() - epoch_started
            write_json(directory / 'progress.json', dict(state='validation', epoch=epoch, step=step,
                       completed_epochs=epoch, total_epochs=settings['epochs'], loader=runtime['loader']))
            validation_started = time.perf_counter()
            validation, _, _ = evaluate(model, val_loader, mean, scale, device, retain_rows=False)
            validation_seconds = time.perf_counter() - validation_started
            score = validation['source_mean'][criterion]
            scheduler.step()
            improved = score < best
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            elapsed = elapsed_before + time.perf_counter() - started
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                config=config, variant=variant, seed=seed, epoch=epoch, step=step,
                criterion=criterion, selection_score=score, mean=mean.cpu(), scale=scale.cpu(),
                cache_manifest_sha256=cache_digest, torch_rng=torch.get_rng_state(),
                implementation_sha256=implementation, runtime=runtime,
                sampler_rng=train_loader.sampler.sampler.generator.get_state(),
                resume_state=dict(best=best, stale=stale, elapsed_s=elapsed,
                    numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else []))
            if improved:
                save_checkpoint(directory / 'best.pt', payload)
            save_checkpoint(directory / 'last.pt', payload)
            record = dict(epoch=epoch, step=step, lr=lr, train={k: v/seen for k, v in totals.items()},
                validation=validation['source_mean'], validation_curves=validation['curves'],
                best=best, elapsed_s=elapsed, train_s=train_seconds, validation_s=validation_seconds)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            print(f"{variant['name']} seed={seed} epoch={epoch} val_{criterion}={score:.6g} best={best:.6g}", flush=True)
            completed_epochs = epoch + 1
            write_json(directory / 'status.json', dict(state='training', completed_epochs=epoch + 1,
                total_epochs=settings['epochs'], step=step, best_validation=best))
            if stale >= settings['patience']:
                break
    # Release worker pools before the independently reconstructed test evaluation.
    del train_loader, val_loader
    # The checkpoint is authoritative if an early-stop decision ended this invocation.
    if stop_epoch < settings['epochs'] and stale < settings['patience']:
        status = dict(state='training_paused', completed_epochs=completed_epochs,
                      total_epochs=settings['epochs'], next_action='Resume the same config with --resume.')
        write_json(directory / 'status.json', status)
        return status
    write_json(directory / 'progress.json', dict(state='test_evaluation', completed_epochs=completed_epochs,
               total_epochs=settings['epochs'], loader=runtime['loader']))
    return evaluate_checkpoint(run_root, device, runtime=runtime)


def collect(config):
    """Matched rows and seed averaging, then resample independent sources once."""
    root = Path(config['output'])
    runs, summary = {}, {}
    cache_digests = set()
    for variant in config['variants']:
        records = []
        for seed in config['seeds']:
            directory = forecast_directory(root / f"{variant['name']}-seed{seed}")
            if json.loads((directory / 'status.json').read_text())['state'] != 'complete':
                raise ValueError(f'Cannot collect incomplete forecast: {directory}')
            declared = json.loads((directory / 'config.json').read_text())
            if declared['variant'] != variant or declared['seed'] != seed:
                raise ValueError(f'Collected forecast differs from declared variant/seed: {directory}')
            for key in ('history_ps', 'anchor_history_ps', 'horizons_ps', 'stride_ps'):
                if declared['config'][key] != config[key]:
                    raise ValueError(f'Collected forecast differs in {key}: {directory}')
            cache_digests.add(json.loads((directory / 'data_summary.json').read_text())['cache_manifest_sha256'])
            if len(cache_digests) != 1:
                raise ValueError('Forecast comparisons require the same frozen embedding cache and target space.')
            with np.load(directory / 'test_errors.npz') as archive:
                records.append({key: archive[key] for key in archive.files})
        reference = records[0]
        for rows in records[1:]:
            for key in ('source', 'atom_id', 'anchor_frame'):
                np.testing.assert_array_equal(rows[key], reference[key], err_msg='Forecast seeds require identical held-out windows')
        errors = np.stack([r['mse'] for r in records])
        bins = np.stack([r['bin_mse'].mean(1) for r in records]).mean(0)
        means = np.array([np.mean([r['mse'][r['source'] == s].mean() for s in np.unique(r['source'])]) for r in records])
        runs[variant['name']] = dict(rows=reference, errors=errors.mean(0), bin_errors=bins)
        summary[variant['name']] = dict(target=variant['target'], seeds=config['seeds'],
            source_mean_mse=float(means.mean()), seed_std=float(means.std()) if len(means) > 1 else None,
            source_mean_bin_mse=float(np.mean([bins[reference['source'] == s].mean() for s in np.unique(reference['source'])])),
            persistence=source_bootstrap(errors.mean(0), reference['persistence_mse'], reference['source'], 20260911))
    comparisons = {}
    for candidate, baseline in config['comparisons']:
        a, b = runs[candidate], runs[baseline]
        for key in ('source', 'atom_id', 'anchor_frame'):
            np.testing.assert_array_equal(a['rows'][key], b['rows'][key])
        if summary[candidate]['target'] != summary[baseline]['target']:
            raise ValueError(f'Cannot compare different forecast targets: {candidate}, {baseline}')
        comparisons[f'{candidate}_vs_{baseline}'] = source_bootstrap(a['errors'], b['errors'], a['rows']['source'], 20260911)
    for candidate, baseline in config['bin_comparisons']:
        a, b = runs[candidate], runs[baseline]
        for key in ('source', 'atom_id', 'anchor_frame'):
            np.testing.assert_array_equal(a['rows'][key], b['rows'][key])
        comparisons[f'{candidate}_vs_{baseline}_bin_means'] = source_bootstrap(
            a['bin_errors'], b['bin_errors'], a['rows']['source'], 20260911)
    report = dict(variants=summary, comparisons=comparisons,
                  uncertainty='Source bootstrap after seed averaging; previously examined test sources remain exploratory.')
    result_folders(root)
    write_json(root / 'technical/comparison.json', report)
    write_metric_table(report, root, family='forecast', name='model-comparison')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'collect', 'queue', 'all'), required=True)
    parser.add_argument('--variant', help='One configured model; omit to run the configured matrix.')
    parser.add_argument('--seed', type=int, help='One configured seed; omit to use all configured seeds.')
    parser.add_argument('--device', help='Override the configured compute device.')
    parser.add_argument('--resume', action='store_true', help='Restore the exact last optimizer/scaler/RNG checkpoint.')
    parser.add_argument('--epochs-per-invocation', type=int, help='Pause after this many additional complete epochs for a queued continuation.')
    parser.add_argument('--queue-config', type=Path, help='Slurm resources and epoch chunk size for --stage queue.')
    parser.add_argument('--runtime-config', type=Path, help='Execution-only loader and step-logging settings.')
    parser.add_argument('--resume-transition', type=Path, help='Reviewed exact old/new implementation hashes for a resume migration.')
    args = parser.parse_args(argv)
    if args.stage == 'queue':
        if args.queue_config is None:
            parser.error('--stage queue requires --queue-config')
        from .queue import submit_queue
        submit_queue(args.config, args.queue_config)
        return
    config = load_json(args.config)
    runtime = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    device = args.device or config['device']
    if args.stage != 'collect' and device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {device}, but CUDA is unavailable.')
    torch.set_num_threads(config['training']['cpu_threads'])
    if args.stage in ('prepare', 'all'):
        prepare_cache(config['data'], device)
    variants = [v for v in config['variants'] if args.variant is None or v['name'] == args.variant]
    seeds = config['seeds'] if args.seed is None else [args.seed]
    if not variants or any(s not in config['seeds'] for s in seeds):
        raise ValueError('Requested variant/seed is absent from the experiment configuration.')
    if args.stage in ('train', 'evaluate', 'all'):
        for variant in variants:
            for seed in seeds:
                if args.stage == 'evaluate':
                    evaluate_checkpoint(Path(config['output']) / f"{variant['name']}-seed{seed}", device, runtime=runtime)
                else:
                    train(config, variant, seed, device, resume=args.resume,
                          epochs_per_invocation=args.epochs_per_invocation, runtime=runtime,
                          resume_transition=args.resume_transition)
    if args.stage == 'collect':
        collect(config)
