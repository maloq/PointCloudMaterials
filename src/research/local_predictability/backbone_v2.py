"""Fresh, receipt-bound MACE/GATr training with one optimizer and export path."""
import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from src.data.predictive_memory.prepare import file_hash, write_json
from src.data.predictive_memory.targets import BLOCKS
from src.experiment_runner.metric_docs import write_metric_table, numeric_rows
from src.project_runtime.paths import load_json, resolve_path
from .backbone_data import prepare_data
from .metrics import hazard_loss, source_weights, cumulative_risk
from .model_factory import build_model
from .native_data import SourceSampler
from .native_preflight import seed_all, save_checkpoint
from .native_queue import configure_file_limit
from .native_runtime import ObservationPrefetcher, peek_batch

PROTOCOL = 'local_predictability_backbone_v2'
REPO = Path(__file__).resolve().parents[3]


def implementation_identity():
    folders = ('src/models/encoders', 'src/data/predictive_memory', 'src/research/local_predictability')
    files = sorted(p for folder in folders for p in (REPO/folder).rglob('*.py'))
    versions = {name: importlib.metadata.version(name) for name in
        ('torch', 'mace-torch', 'e3nn', 'cuequivariance', 'cuequivariance-torch',
         'cuequivariance-ops-torch-cu13', 'GATr', 'einops', 'xformers', 'opt_einsum', 'numpy')}
    from src.models.encoders.axial_gatr import GATR_REVISION
    distribution = importlib.metadata.distribution('GATr')
    direct = json.loads(distribution.read_text('direct_url.json'))
    if direct['vcs_info']['commit_id'] != GATR_REVISION:
        raise ValueError('Installed GATr differs from the pinned upstream implementation')
    return dict(files={str(p.relative_to(REPO)): file_hash(p) for p in files},
        versions=versions, gatr_revision=GATR_REVISION)


def identity(config, data, kind, objective, variant):
    # A later allocation may extend the operational deadline for an explicit
    # resume. The scientific budget, batching, model and all code stay locked.
    scientific = {k: v for k, v in config.items() if k not in
                  ('output', 'training_deadline_utc', 'hard_deadline_utc')}
    return dict(protocol=PROTOCOL, data=data.identity, config=scientific,
        encoder_kind=kind, objective=objective, variant=variant, implementation=implementation_identity())


def before_deadline(config, reserve=120):
    if time.time()+reserve >= datetime.fromisoformat(config['training_deadline_utc']).timestamp():
        raise TimeoutError('V2 training deadline: checkpoint preserved; extend deadline and explicitly resume')


def per_row_loss(result, data, indices, objective, future_weight=1.):
    if objective == 'onset':
        return hazard_loss(result['logits'], data.events[indices])
    present = (result['present']-data.targets[indices, 0]).square().mean(-1)
    future = (result['future']-data.targets[indices, 1:]).square().mean((-1, -2))
    return present + future_weight*future


def update(model, observed, indices, data, optimizer, microbatch, future_weight=1.):
    """Eight identical sampled examples; execution microbatches only split work."""
    if len(indices) != 8 or microbatch not in (1, 2, 4, 8):
        raise ValueError('V2 uses effective batch eight and microbatch 1, 2, 4 or 8')
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=data.cond.device)
    for start in range(0, 8, microbatch):
        batch = indices[start:start+microbatch]
        result = model(observed[start:start+microbatch], data.cond[batch])
        loss = per_row_loss(result, data, batch, model.objective, future_weight).sum()/8
        loss.backward(); total += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return float(total), float(norm)


def summarize(predictions, data, indices, objective):
    weight = source_weights(data.labels['source_id'][indices])
    if objective == 'onset':
        losses = hazard_loss(predictions['logits'], data.events[indices].cpu()).numpy()
        return dict(score=float(weight@losses), hazard_nll=float(weight@losses))
    target = data.targets[indices].cpu()
    present = (predictions['present']-target[:, 0]).square().numpy()
    future = (predictions['future']-target[:, 1:]).square().numpy()
    p, f = float(weight@present.mean(-1)), float(weight@future.mean((1, 2)))
    return dict(score=p+f, present_mse=p, future_mse=f,
        present_blocks={name: float(weight@present[:, a:b].mean(-1)) for name, (a, b) in BLOCKS.items()},
        future_by_horizon={str(lag): float(weight@future[:, k].mean(-1))
                           for k, lag in enumerate(data.identity['horizons_ps'])},
        future_blocks={name: float(weight@future[:, :, a:b].mean((1, 2))) for name, (a, b) in BLOCKS.items()})


@torch.no_grad()
def evaluate(model, data, indices, microbatch, config):
    was_training = model.training; model.eval()
    parts = {}; started = time.monotonic()
    with ObservationPrefetcher(data.windows, config['prefetch_workers']) as inputs:
        for batch, observed in inputs.batches(indices, model.encoder.variant, microbatch):
            before_deadline(config, reserve=30)
            result = model(observed, data.cond[batch])
            for key, value in result.items():
                parts.setdefault(key, []).append(value.cpu())
        wait = inputs.wait_seconds
    predictions = {key: torch.cat(value) for key, value in parts.items()}
    for key, value in predictions.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Nonfinite {key} while evaluating {model.encoder.variant} on {len(indices)} rows')
    metrics = summarize(predictions, data, indices, model.objective)
    metrics.update(validation_seconds=time.monotonic()-started, input_wait_seconds=wait, windows=len(indices))
    model.train(was_training)
    return metrics, predictions


def restore(path, model, optimizer, sampler, expected, stage):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['identity'] != expected or saved['extra']['stage'] != stage:
        raise ValueError(f'Exact v2 resume identity changed: {path}')
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    sampler.load_state_dict(saved['sampler'])
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    return saved


def export_curve(config, root, name):
    """Readable matched-example/time curves from committed validation points."""
    records = [dict(numeric_rows(json.loads(line))) for line in (root/'validation.jsonl').read_text().splitlines()]
    path = resolve_path(config['output'])/'tables'/f'{name}_curve.csv'
    temporary = path.with_suffix('.building.csv')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    temporary.replace(path)


def truncate_uncommitted_logs(root, step):
    """A hard interruption may leave flushed observations after the checkpoint.

    Keep those records as interruption evidence, but do not mix them into the
    resumed optimization/learning curve as if they were committed updates.
    """
    for name in ('training', 'validation'):
        path = root/f'{name}.jsonl'
        if not path.exists():
            continue
        committed, uncommitted = [], []
        for line in path.read_text().splitlines(keepends=True):
            try:
                keep = json.loads(line)['step'] <= step
            except json.JSONDecodeError:
                keep = False  # Preserve a partially written final line as evidence.
            (committed if keep else uncommitted).append(line)
        if uncommitted:
            with (root/f'{name}-uncommitted.jsonl').open('a') as stream:
                stream.writelines(uncommitted)
            temporary = path.with_suffix('.building.jsonl')
            temporary.write_text(''.join(committed)); temporary.replace(path)


def gate_path(config, kind):
    return resolve_path(config['output'])/'technical'/kind/'gate/receipt.json'


def verify_gate(config, data, kind):
    path = gate_path(config, kind)
    gate = json.loads(path.read_text())
    if gate['state'] != 'passed' or gate['identity'] != identity(config, data, kind, 'physical_means', 'snapshot'):
        raise ValueError(f'Fresh v2 small-fit receipt does not verify this model: {path}')
    return file_hash(path)


def fit(config, data, kind, objective, variant, *, diagnostic=False, stop_step=None,
        resume=False, parent=None):
    before_deadline(config)
    expected = identity(config, data, kind, objective, variant)
    stage = 'gate' if diagnostic else f'{objective}/{variant}'
    root = resolve_path(config['output'])/'technical'/kind/stage
    root.mkdir(parents=True, exist_ok=True)
    if diagnostic:
        if objective != 'physical_means' or variant != 'snapshot' or parent is not None:
            raise ValueError('Information gate is a fresh snapshot present-only fit')
        data, selected, diagnostic_definition = data.diagnostic()
        train = selected; maximum = config['gate_updates']; frequency = config['gate_evaluate_every']
        expected['diagnostic'] = diagnostic_definition
    else:
        expected['gate_receipt_sha256'] = verify_gate(config, data, kind)
        train = data.splits(objective)['train']; selected = data.selection(objective)
        maximum = config['training_updates']; frequency = config['evaluate_every']
    target_step = min(maximum, stop_step) if stop_step is not None else maximum
    latest = root/'latest.pt'
    if latest.exists() and not resume:
        raise FileExistsError(f'Existing v2 weights require explicit --resume: {latest}')
    seed_all()
    model = build_model(kind, objective, variant, config).to(data.cond.device).train()
    if parent is not None:
        saved = torch.load(parent, map_location='cpu', weights_only=False)
        parent_identity = {**expected, 'variant': 'snapshot'}
        if saved['identity'] != parent_identity or saved['extra']['stage'] != f'{objective}/snapshot':
            raise ValueError('Parent must be a snapshot of the SAME architecture, objective, release and budget')
        model.load_state_dict(saved['model'])
    elif objective == 'onset':
        y = data.labels['event_bin'][train]
        weight = source_weights(data.labels['source_id'][train])
        prior_hazard = np.array([np.clip(weight[y == k].sum()/max(weight[y >= k].sum(), 1e-12),
                                        1e-4, 1-1e-4) for k in range(6)])
        with torch.no_grad():
            model.hazard.bias.copy_(torch.as_tensor(np.log(prior_hazard/(1-prior_hazard)), device=data.cond.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.0001)
    sampler = SourceSampler(data.windows.rows, train)
    step = 0; best = float('inf'); elapsed = 0.; fit_wait = 0.
    if latest.exists():
        saved = restore(latest, model, optimizer, sampler, expected, stage)
        step = saved['step']; best = saved['extra']['best']; elapsed = saved['extra']['elapsed_seconds']
        fit_wait = saved['extra']['input_wait_seconds']
        truncate_uncommitted_logs(root, step)
    microbatch = config['microbatch'][kind][variant]
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    def save(path):
        save_checkpoint(path, model, optimizer, sampler, step, expected,
            dict(stage=stage, best=best, elapsed_seconds=elapsed+time.monotonic()-started,
                 input_wait_seconds=fit_wait+inputs.wait_seconds))
    result = None
    with ObservationPrefetcher(data.windows, config['prefetch_workers']) as inputs:
        future = inputs.submit(peek_batch(sampler), variant) if step < target_step else None
        with (root/'training.jsonl').open('a' if resume else 'x') as log:
            try:
                while step < target_step:
                    before_deadline(config)
                    indices = sampler.batch(); observed = inputs.take(future, indices)
                    if step+1 < target_step:
                        future = inputs.submit(peek_batch(sampler), variant)
                    loss, norm = update(model, observed, indices, data, optimizer, microbatch,
                                        future_weight=0. if diagnostic else 1.)
                    step += 1
                    record = dict(step=step, examples_seen=step*8, loss=loss, gradient_norm=norm,
                        row_indices=indices, elapsed_seconds=elapsed+time.monotonic()-started,
                        input_wait_seconds=fit_wait+inputs.wait_seconds,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    log.write(json.dumps(record)+'\n')
                    if step == 1 or step % 25 == 0:
                        log.flush(); print(json.dumps(dict(encoder=kind, stage=stage, **record)), flush=True)
                        write_json(root/'status.json', dict(state='running', **record))
                    if step == 1 or step % frequency == 0 or step == target_step:
                        inputs.drain()
                        result, _ = evaluate(model, data, selected, microbatch, config)
                        score = result['present_mse'] if diagnostic else result['score']
                        if score < best:
                            best = score; save(root/'best.pt')
                        save(latest)
                        metrics = dict(step=step, examples_seen=step*8,
                            elapsed_seconds=elapsed+time.monotonic()-started, **result)
                        with (root/'validation.jsonl').open('a') as validation:
                            validation.write(json.dumps(metrics)+'\n')
                        table_name = f'{kind}_{stage.replace("/", "_")}'
                        write_metric_table(metrics, resolve_path(config['output']),
                            family='local_predictability_backbone_v2', name=f'{table_name}_latest_validation')
                        export_curve(config, root, table_name)
                        if diagnostic and result['present_mse'] <= .1 and max(result['present_blocks'].values()) <= .25:
                            # Bind the receipt to the checkpoint passing BOTH
                            # criteria, even if an earlier lower aggregate MSE
                            # had a failing individual block.
                            best = score; save(root/'best.pt'); save(latest)
                            receipt_identity = dict(expected); receipt_identity.pop('diagnostic')
                            write_json(root/'receipt.json', dict(state='passed', identity=receipt_identity,
                                diagnostic=diagnostic_definition, steps=step, metrics=result,
                                checkpoint_sha256=file_hash(root/'best.pt')))
                            break
            except TimeoutError:
                inputs.drain(); save(latest)
                write_json(root/'status.json', dict(state='deadline_stopped', step=step))
                raise
    state = 'complete' if step >= maximum else 'paused'
    if diagnostic:
        state = 'passed' if (root/'receipt.json').exists() else 'failed'
        if state == 'failed':
            receipt_identity = dict(expected); receipt_identity.pop('diagnostic')
            write_json(root/'receipt.json', dict(state='failed', identity=receipt_identity,
                diagnostic=diagnostic_definition, steps=step, metrics=result,
                next_action='Block larger fits; one focused fitting investigation, no architecture sweep'))
    write_json(root/'status.json', dict(state=state, step=step, best=best))
    del model, optimizer; torch.cuda.empty_cache()
    return state


def export(config, data, kind, objective, variant):
    root = resolve_path(config['output'])/'technical'/kind/objective/variant
    expected = identity(config, data, kind, objective, variant)
    expected['gate_receipt_sha256'] = verify_gate(config, data, kind)
    saved = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    if saved['identity'] != expected:
        raise ValueError('Export checkpoint does not match v2 identity')
    model = build_model(kind, objective, variant, config).to(data.cond.device)
    model.load_state_dict(saved['model'])
    splits = data.splits(objective)
    # Export one split atomically at a time, allowing an explicit resume after a
    # deadline without losing finished exports. Test data never select weights.
    for split in ('selection', 'calibration', 'test'):
        path = root/f'{split}_predictions.npz'
        receipt = root/f'{split}_export.json'
        checkpoint_hash = file_hash(root/'best.pt')
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['checkpoint_sha256'] != checkpoint_hash or previous['sha256'] != file_hash(path):
                raise ValueError(f'Completed export identity changed: {receipt}')
            continue
        indices = splits[split]
        result, predictions = evaluate(model, data, indices, config['microbatch'][kind][variant], config)
        arrays = dict(indices=indices, source=data.labels['source_id'][indices],
            center=data.labels['center_id'][indices], anchor=data.labels['anchor'][indices],
            **{k: v.numpy() for k, v in predictions.items()})
        if objective == 'physical_means':
            arrays['targets'] = data.targets[indices].cpu().numpy()
        else:
            arrays['event_bin'] = data.events[indices].cpu().numpy()
            arrays['probability'] = cumulative_risk(predictions['logits']).numpy()
        temporary = path.with_suffix('.building.npz'); np.savez(temporary, **arrays); temporary.replace(path)
        write_json(receipt, dict(checkpoint_sha256=checkpoint_hash, sha256=file_hash(path),
            rows=len(indices), selected_step=saved['step'], metrics=result))
        write_metric_table(result, resolve_path(config['output']), family='local_predictability_backbone_v2',
            name=f'{kind}_{objective}_{variant}_{split}')
    del model; torch.cuda.empty_cache()
    if objective == 'onset':
        export_onset_assay(config, data, root, kind, variant)


def export_onset_assay(config, data, root, kind, variant):
    """Use the original prospective calibration and window-risk calculations."""
    from .baselines import score_hazard
    parts = []
    for split in ('selection', 'calibration', 'test'):
        with np.load(root/f'{split}_predictions.npz') as values:
            parts.append({**dict(values), 'split': np.full(len(values['indices']), split)})
    joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    arrays = dict(y=joined['event_bin'], source=joined['source'], center=joined['center'],
        anchor=joined['anchor'], split=joined['split'],
        temperature=np.array([data.windows.rows[i]['temperature_K'] for i in joined['indices']]))
    masks = {split: np.flatnonzero(arrays['split'] == split) for split in ('selection', 'calibration', 'test')}
    result = score_hazard(arrays, joined['probability'], joined['logits'], masks,
                          load_json(resolve_path(config['plan'])))
    write_json(root/'onset_assay.json', result)
    destination = resolve_path(config['output'])
    write_metric_table(result, destination, family='local_predictability_backbone_v2',
        name=f'{kind}_onset_{variant}_assay_summary')
    records = result['population']
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (destination/'tables'/f'{kind}_onset_{variant}_assay.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)


def run_screen(config, data, resume=False):
    """Gates, explicit workload profiling, then alternating matched-size fits."""
    from .backbone_profile import profile
    root = resolve_path(config['output'])/'technical'
    eligible = []
    for kind in ('mace', 'axial_gatr'):
        receipt = gate_path(config, kind)
        if receipt.exists():
            if not resume:
                raise FileExistsError('Existing v2 screen requires explicit --resume')
            saved = json.loads(receipt.read_text())
            if saved['identity'] != identity(config, data, kind, 'physical_means', 'snapshot'):
                raise ValueError(f'Changed gate identity: {receipt}')
            state = saved['state']
        else:
            state = fit(config, data, kind, 'physical_means', 'snapshot', diagnostic=True, resume=resume)
        if state == 'passed':
            eligible.append(kind)
    write_json(root/'screen_status.json', dict(state='profiling', eligible=eligible))
    for kind in eligible:
        profile(config, data, kind, resume=resume)
    for step in range(config['evaluate_every'], config['training_updates']+1, config['evaluate_every']):
        for kind in eligible:
            destination = root/kind/'physical_means/snapshot/latest.pt'
            fit(config, data, kind, 'physical_means', 'snapshot', stop_step=step,
                resume=resume or destination.exists())
        write_json(root/'screen_status.json', dict(state='training', eligible=eligible, matched_updates=step))
    for kind in eligible:
        export(config, data, kind, 'physical_means', 'snapshot')
    write_json(root/'screen_status.json', dict(state='complete' if len(eligible) == 2 else 'gate_limited',
        eligible=eligible, analysis='deferred until requested', matched_updates=config['training_updates']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('screen', 'gate', 'profile', 'fit', 'export'), required=True)
    parser.add_argument('--encoder', choices=('mace', 'axial_gatr'))
    parser.add_argument('--objective', choices=('physical_means', 'onset'), default='physical_means')
    parser.add_argument('--variant', choices=('snapshot', 'history12', 'repeat12'), default='snapshot')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); config = load_json(args.config)
    if config['protocol'] != PROTOCOL or config['seed'] != 20260919 or config['precision'] != 'float32':
        raise ValueError('V2 screen requires its declared protocol, single seed and FP32 reference')
    if config['training_updates'] % config['evaluate_every']:
        raise ValueError('Matched screen budget must be a whole number of validation intervals')
    if args.stage != 'screen' and args.encoder is None:
        parser.error('--encoder is required outside the matched screen')
    configure_file_limit(); torch.set_num_threads(config['torch_threads']); seed_all()
    root = resolve_path(config['output'])/'technical'
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before_deadline(config)
        data = prepare_data(config, root)
        write_json(root/'data_identity.json', data.identity)
        if args.stage == 'screen':
            run_screen(config, data, args.resume)
        elif args.stage == 'profile':
            from .backbone_profile import profile
            profile(config, data, args.encoder, resume=args.resume)
        elif args.stage == 'export':
            export(config, data, args.encoder, args.objective, args.variant)
        else:
            state = fit(config, data, args.encoder, args.objective, args.variant,
                diagnostic=args.stage == 'gate', resume=args.resume, parent=args.parent)
            if state == 'failed':
                raise RuntimeError('Small-fit gate failed; larger experiments are blocked')


if __name__ == '__main__':
    main()
