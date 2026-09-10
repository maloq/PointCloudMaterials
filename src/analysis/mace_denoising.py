"""Held-source topology decoding, temporal interventions and paired source errors."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch

from src.data_utils.temporal_campaign import write_json
from src.training_methods.mace_denoising import BLOCKS, Predictor, predict, raw_prediction, transform_target


def score(prediction, target, contexts, temperatures, scaling):
    error = prediction.astype(np.float64)-target.astype(np.float64)
    context_mean = np.empty_like(target)
    for context in np.unique(contexts):
        selected = contexts == context
        context_mean[selected] = target[selected].mean(0)
    blocks = {}
    for d, block in enumerate(BLOCKS):
        mse = np.mean(error[:, block]**2)
        variance = np.mean((target[:, block]-target[:, block].mean(0))**2)
        local_variance = np.mean((target[:, block]-context_mean[:, block])**2)
        blocks[f'H{d}'] = dict(mse=float(mse), r2=float(1-mse/variance),
            within_frame_r2=float(1-mse/local_variance), scaled_mse=float(mse/scaling['block_scale'][d]**2))
    per_row = np.mean(np.stack([np.mean(error[:, block]**2, axis=1)/scaling['block_scale'][d]**2
                               for d, block in enumerate(BLOCKS)]), axis=0)
    return dict(balanced_mse=float(per_row.mean()), raw_mse=float(np.mean(error**2)),
        mean_block_r2=float(np.mean([b['r2'] for b in blocks.values()])),
        mean_within_frame_r2=float(np.mean([b['within_frame_r2'] for b in blocks.values()])),
        blocks=blocks, by_temperature={str(int(t)): float(per_row[temperatures==t].mean()) for t in np.unique(temperatures)}), per_row


def ridge_predictions(features, targets, train, rows, alpha):
    scaler = StandardScaler().fit(features[train])
    ridge = Ridge(alpha=alpha).fit(scaler.transform(features[train]), targets[train])
    return ridge.predict(scaler.transform(features[rows]))


def paired_source_gain(reference, candidate, sources, seed):
    groups = np.unique(sources)
    a = np.array([reference[sources==s].mean() for s in groups])
    b = np.array([candidate[sources==s].mean() for s in groups])
    draw = np.random.default_rng(seed).integers(0, len(groups), size=(4000, len(groups)))
    gains = 1-b[draw].mean(1)/a[draw].mean(1)
    return dict(relative_mse_reduction=float(1-b.mean()/a.mean()),
        source_bootstrap_95_percent_interval=np.quantile(gains, [.025, .975]).tolist(),
        source_count=len(groups), per_source_reduction=(1-b/a).tolist(),
        limitation='Resamples whole held-out source trajectories; few test sources limit precision. Seeds are averaged first.')


def export_encoder(cfg, data, variant, selected_seed, model, directory):
    from src.models.encoders.mace_denoising import PretrainedMACEDenoisingEncoder
    kwargs = dict(pretrained_checkpoint=cfg['pretrained_checkpoint'], frame_offsets_ps=cfg['frame_offsets_ps'],
        fusion=variant['architecture'], performance=cfg['performance'], frame_batch_size=cfg['feature_batch_size'],
        require_frame_offsets=cfg['protocol']=='denoising80_reuse')
    encoder = PretrainedMACEDenoisingEncoder(**kwargs).cuda().eval()
    frozen = torch.load(Path(cfg['feature_cache']) / 'frozen_mace.pt', map_location='cuda', weights_only=False)
    encoder.mace.load_state_dict(frozen['model'], strict=True)
    encoder.fusion.load_state_dict(model.fusion.state_dict(), strict=True)
    for name in ('pooled_mean', 'pooled_std', 'node_mean', 'node_std'):
        getattr(encoder, name).copy_(torch.as_tensor(data.scaling[name], device='cuda'))
    record = data.records[0]
    points = torch.from_numpy(np.load(Path(record['data_directory']) / 'histories.npy')[:2].astype(np.float32)).cuda()
    with torch.no_grad():
        expected = model.encode(*data.get(np.arange(2), variant['architecture']))
        offsets = torch.tensor(record['provenance']['offsets'], device='cuda').repeat(2, 1)
        actual = encoder(points, torch.zeros(2, device='cuda', dtype=torch.long), offsets)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    path = directory / f"{variant['name']}_encoder.pt"
    torch.save(dict(encoder=encoder.state_dict(), encoder_name='PretrainedMACEDenoising', encoder_kwargs=kwargs,
        trunk=model.trunk.state_dict(), heads=model.heads.state_dict(), scaling=data.scaling,
        variant=variant, seed=selected_seed, direct_cached_max_error=float((actual-expected).abs().max())), path)
    return str(path)


def analyze(cfg, data):
    root = Path(cfg['output'])
    directory = root / 'analysis'
    directory.mkdir(exist_ok=True)
    train = data.indices['train']
    test = data.indices['test']
    val = data.indices['val']
    scaling = data.scaling
    target_blocks = transform_target(data.targets_raw, scaling, 'blocks')
    pooled = data.pooled.cpu().numpy()
    baseline_features = dict(anchor_ridge=pooled[:, -2], mean_ridge=pooled[:, :-1].mean(1),
        relaxed_ridge=pooled[:, -1], hot_tda_ridge=data.hot_targets)
    potentials = np.concatenate([np.full(r['count'], r['potential'] if cfg['protocol']=='denoising80_reuse'
                                        else 'Lee2003_MEAM') for r in data.records])

    def measure(prediction, rows):
        metrics, per_row = score(prediction, data.targets_raw[rows], data.contexts[rows], data.temperatures[rows], scaling)
        metrics['by_potential'] = {}
        for potential in np.unique(potentials[rows]):
            selected = potentials[rows] == potential
            group, _ = score(prediction[selected], data.targets_raw[rows][selected], data.contexts[rows][selected],
                             data.temperatures[rows][selected], scaling)
            group.update(examples=int(selected.sum()), source_count=len(np.unique(data.sources[rows][selected])))
            metrics['by_potential'][potential] = group
        return metrics, per_row

    results, errors, prediction_cache = {}, {}, {}
    for name, features in baseline_features.items():
        result = {}
        for split, rows in (('val', val), ('test', test)):
            predicted = raw_prediction(ridge_predictions(features, target_blocks, train, rows, cfg['ridge_alpha']), scaling, 'blocks')
            result[split], per_row = measure(predicted, rows)
            if split == 'test':
                errors[name] = per_row
                prediction_cache[name] = predicted
        results[name] = result
    for name, predicted in (('train_mean', np.repeat(data.targets_raw[train].mean(0)[None], len(test), axis=0)),
                            ('hot_tda_direct', data.hot_targets[test])):
        metrics, per_row = measure(predicted, test)
        results[name] = dict(test=metrics)
        errors[name] = per_row
    for variant in cfg['variants']:
        runs = []
        all_errors = []
        for seed in cfg['seeds']:
            run = root / 'runs' / f"{variant['name']}_seed{seed}"
            checkpoint = torch.load(run / 'best.pt', map_location='cuda', weights_only=False)
            model = Predictor(cfg, variant).cuda().eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            result = dict(seed=seed, best_epoch=checkpoint['epoch'],
                          training=json.loads((run / 'training_summary.json').read_text()))
            for split, rows in (('val', val), ('test', test)):
                prediction, z = predict(model, data, rows, cfg)
                raw = raw_prediction(prediction, scaling, variant['target'])
                result[split], per_row = measure(raw, rows)
                if split == 'test':
                    all_errors.append(per_row)
                    prediction_cache[f"{variant['name']}_seed{seed}"] = raw
                    np.savez_compressed(run / 'test_predictions.npz', predictions=raw, embedding=z,
                        indices=rows, sources=data.sources[rows], contexts=data.contexts[rows])
            if variant['architecture'] in ('transformer', 'residual', 'atom_temporal', 'atom_anchor'):
                result['history_interventions'] = {}
                for intervention in ('repeat_anchor', 'reverse_past'):
                    prediction, _ = predict(model, data, test, cfg, intervention)
                    raw = raw_prediction(prediction, scaling, variant['target'])
                    result['history_interventions'][intervention], _ = measure(raw, test)
            write_json(run / 'metrics.json', result)
            runs.append(result)
            print('DENOISING_ANALYSIS', variant['name'], seed, result['test']['balanced_mse'], flush=True)
        results[variant['name']] = dict(runs=runs, test_balanced_mse_mean=float(np.mean([r['test']['balanced_mse'] for r in runs])),
            test_balanced_mse_std=float(np.std([r['test']['balanced_mse'] for r in runs])),
            test_mean_within_frame_r2=float(np.mean([r['test']['mean_within_frame_r2'] for r in runs])),
            validation_balanced_mse_mean=float(np.mean([r['val']['balanced_mse'] for r in runs])))
        results[variant['name']]['by_potential'] = {p: float(np.mean([
            r['test']['by_potential'][p]['balanced_mse'] for r in runs])) for p in np.unique(potentials[test])}
        errors[variant['name']] = np.mean(all_errors, axis=0)
    comparisons = {}
    for a, b in (('transformer_pca_regularized', 'transformer_pca_tda'),
                 ('transformer_pca_tda', 'transformer_blocks'),
                 ('anchor_continue_blocks', 'residual_blocks'), ('mean_blocks', 'residual_blocks'),
                 ('atom_anchor_blocks', 'atom_temporal_blocks')):
        comparisons[f'{b}_versus_{a}'] = paired_source_gain(errors[a], errors[b], data.sources[test], cfg['seed'])
    exports = {}
    for variant in (v for v in cfg['variants'] if v['architecture'] in ('residual', 'atom_anchor', 'atom_temporal')):
        runs = results[variant['name']]['runs']
        selected = min(runs, key=lambda r: r['val']['balanced_mse'])
        path = root / 'runs' / f"{variant['name']}_seed{selected['seed']}" / 'best.pt'
        checkpoint = torch.load(path, map_location='cuda', weights_only=False)
        model = Predictor(cfg, variant).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        exports[variant['name']] = export_encoder(cfg, data, variant, selected['seed'], model, directory)
    write_json(directory / 'metrics.json', dict(results=results, paired_comparisons=comparisons,
        exports=exports, splits={s: len(rows) for s, rows in data.indices.items()},
        sources={s: len(np.unique(data.sources[rows])) for s, rows in data.indices.items()},
        context=cfg.get('data_context', 'One material, independent melt-source splits, fixed anchor times. Five observed snapshots span 3 ps.'),
        limitations=cfg.get('data_limitations', 'Limited number of independent held-out sources.')))
    np.savez_compressed(directory / 'predictions.npz', targets=data.targets_raw[test], **prediction_cache)
    names = list(results)
    means = [r['test']['balanced_mse'] if 'test' in r else r['test_balanced_mse_mean'] for r in results.values()]
    stds = [r.get('test_balanced_mse_std', 0.) for r in results.values()]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(names, means, xerr=stds, color=['#778da9' if 'runs' not in results[n] else '#2a9d8f' for n in names])
    ax.invert_yaxis()
    ax.set_xlabel('Held-source balanced H0/H1/H2 MSE (lower is better); error bars: training seeds')
    fig.tight_layout()
    fig.savefig(directory / 'comparison.png', dpi=170)
    plt.close(fig)
    lines = ['# Al temporal denoising — September 10, 2026', '',
        'Question: do several thermal observations improve prediction of the relaxed-anchor topology?', '',
        'All backbone weights are frozen. Whole-source splits are declared in the configuration before fitting. '
        'Target and feature scaling use training data only; checkpoints use validation only. '
        'All reported models and seeds were configured before test evaluation.', '',
        '| Predictor | Test balanced MSE | Within-frame mean block R² |', '| --- | ---: | ---: |']
    for name, r in results.items():
        if 'test' in r:
            lines.append(f"| {name} | {r['test']['balanced_mse']:.5f} | {r['test']['mean_within_frame_r2']:.4f} |")
        else:
            lines.append(f"| {name} | {r['test_balanced_mse_mean']:.5f} ± {r['test_balanced_mse_std']:.5f} | {r['test_mean_within_frame_r2']:.4f} |")
    lines += ['', 'Within-frame R² uses variation among neighborhoods of the same source frame as its denominator; '
        'it penalizes prediction errors including frame-mean bias. It tests more than recognizing temperature or phase.', '',
        '## Matched comparisons', '', '| Candidate versus reference | Relative MSE reduction | Source bootstrap 95% interval |',
        '| --- | ---: | ---: |']
    for name, r in comparisons.items():
        low, high = r['source_bootstrap_95_percent_interval']
        lines.append(f"| {name} | {100*r['relative_mse_reduction']:.2f}% | [{100*low:.2f}%, {100*high:.2f}%] |")
    lines += ['', f'Bootstrap resamples whole test trajectories after averaging training seeds. {len(np.unique(data.sources[test]))} held-out trajectories limit confidence. '
        'The continued anchor control has the same anchor-head warm start and additional training budget as residual fusion. '
        'Atom-anchor repeats the current atom features in every time slot, preserving temporal-module capacity.', '',
        '## History interventions', '', '| Model | Real history MSE | Repeated anchor | Reversed past |', '| --- | ---: | ---: | ---: |']
    for name, r in results.items():
        if 'runs' in r and 'history_interventions' in r['runs'][0]:
            repeated = np.mean([v['history_interventions']['repeat_anchor']['balanced_mse'] for v in r['runs']])
            reversed_past = np.mean([v['history_interventions']['reverse_past']['balanced_mse'] for v in r['runs']])
            lines.append(f"| {name} | {r['test_balanced_mse_mean']:.5f} | {repeated:.5f} | {reversed_past:.5f} |")
    lines += ['', '## Potential-specific test errors', '',
              '| Predictor | ' + ' | '.join(np.unique(potentials[test])) + ' |',
              '| --- | ' + ' | '.join(['---:']*len(np.unique(potentials[test]))) + ' |']
    for name, result in results.items():
        values = (result['by_potential'] if 'runs' in result else
                  {p: r['balanced_mse'] for p, r in result['test']['by_potential'].items()})
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[p]:.5f}' for p in np.unique(potentials[test])) + ' |')
    lines += ['', cfg.get('data_context', 'Five snapshots at 0.75 ps cadence.'), '',
        cfg.get('data_limitations', 'Few independent test sources limit precision.'), '',
        'Full per-temperature and H0/H1/H2 scores, training exposure counts, encoder exports and numerical checks '
        'are in [metrics.json](metrics.json). Raw predictions are in predictions.npz; [comparison plot](comparison.png).', '',
        'All compared models share the same data and receive the actual per-history snapshot times. '
        'Relaxed-input results are diagnostic references; inference still receives thermal observations.']
    (directory / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if cfg['protocol'] == 'denoising80_reuse':
        from src.analysis.mace_potential_audit import audit_existing_minimizers
        audit_existing_minimizers(cfg, scaling, directory)
