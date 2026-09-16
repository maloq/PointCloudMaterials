"""Matched ridge and residual nonlinear probes, with no encoder optimization."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from .recovery_data import GROUPS, OBSERVABLES, PhysicalHeads, feature_arrays, group_errors, inputs, publish, status
from .recovery_data import load_json
from src.research.mace_tda_ridge_audit.math import paired_interval


def fit_ridge(x, y, train, val, weights, alphas):
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1  # Same constant-channel definition as ridge_path.
    xx = (x-mean)/scale
    target_mean = y[train].mean(axis=0)
    u, singular, vt = np.linalg.svd(xx[train], full_matrices=False)
    uy = u.T@(y[train]-target_mean)
    val_projection = xx[val]@vt.T
    selected, coefficients = {}, np.empty((x.shape[1], y.shape[1]))
    for name, section in GROUPS.items():
        candidates = {}
        for alpha in alphas:
            estimate = (val_projection*(singular/(singular**2+alpha)))@uy[:, section]+target_mean[section]
            candidates[alpha] = np.mean(np.sum((estimate-y[val, section])**2*weights[section], axis=1))
        alpha = min(candidates, key=candidates.get)
        coefficients[:, section] = (vt.T*(singular/(singular**2+alpha)))@uy[:, section]
        selected[name] = dict(alpha=alpha, validation_mse=float(candidates[alpha]),
                              alpha_at_grid_edge=alpha in [alphas[0], alphas[-1]])
    prediction = xx@coefficients+target_mean
    return prediction, xx, dict(x_mean=mean, x_scale=scale, coefficients=coefficients, y_mean=target_mean), selected


def nonlinear_correction(config, xx, y, baseline, ids, weights, directory, callback):
    """Validate each head separately, including the unchanged ridge as epoch zero."""
    cfg, device = config['readout'], config['device']
    train, val = ids['train'], ids['val']
    tx = torch.as_tensor(xx[train], dtype=torch.float32, device=device)
    vx = torch.as_tensor(xx[val], dtype=torch.float32, device=device)
    residual = torch.as_tensor(y[train]-baseline[train], dtype=torch.float32, device=device)
    val_residual = torch.as_tensor(y[val]-baseline[val], dtype=torch.float32, device=device)
    ww = torch.as_tensor(weights, dtype=torch.float32, device=device)
    best = {k: float(v.mean()) for k, v in group_errors(baseline[val], y[val], weights).items()}
    states, selection = {}, {k: dict(kind='ridge', validation_mse=v) for k, v in best.items()}
    traces = []
    for seed in cfg['seeds']:
        for decay in cfg['weight_decays']:
            torch.manual_seed(seed)
            model = PhysicalHeads(xx.shape[1], cfg['hidden_width']).to(device)
            for head in model.heads.values():
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=decay)
            candidate_best, improved = best.copy(), {k: 0 for k in GROUPS}
            for epoch in range(1, cfg['max_epochs']+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tx)
                loss = ((prediction-residual).square()*ww).sum(dim=1).mean()/len(GROUPS)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonlinear readout loss: seed={seed}, decay={decay}, epoch={epoch}')
                loss.backward()
                optimizer.step()
                if epoch % cfg['validation_interval']:
                    continue
                model.eval()
                with torch.no_grad():
                    difference = model(vx)-val_residual
                    scores = {k: float((difference[:, s].square()*ww[s]).sum(dim=1).mean()) for k, s in GROUPS.items()}
                traces.append(dict(seed=seed, weight_decay=decay, epoch=epoch, validation=scores))
                for name, score in scores.items():
                    if score < candidate_best[name]:
                        candidate_best[name], improved[name] = score, epoch
                    if score < best[name]:
                        best[name] = score
                        states[name] = {k: v.detach().cpu().clone() for k, v in model.heads[name].state_dict().items()}
                        selection[name] = dict(kind='ridge_plus_mlp', seed=seed, weight_decay=decay,
                                               epoch=epoch, validation_mse=score)
                callback(seed=seed, weight_decay=decay, epoch=epoch)
                if all(epoch-improved[k] >= cfg['patience_epochs'] for k in GROUPS):
                    break
    correction = np.zeros_like(baseline)
    model = PhysicalHeads(xx.shape[1], cfg['hidden_width']).to(device).eval()
    with torch.no_grad():
        for name, state in states.items():
            model.heads[name].load_state_dict(state, strict=True)
            for start in range(0, len(xx), 1024):
                correction[start:start+1024, GROUPS[name]] = model.heads[name](torch.as_tensor(
                    xx[start:start+1024], dtype=torch.float32, device=device)).cpu().numpy()
    torch.save(dict(states=states, selection=selection, input_width=xx.shape[1],
                    hidden_width=cfg['hidden_width']), directory/'nonlinear.pt')
    write_json(directory/'nonlinear-history.json', traces)
    return baseline+correction, selection


def score_predictions(name, kind, prediction, features, data, selection, directory):
    pilot, probes, temporal, ids, y, mean, scales, weights, _ = data
    n = len(y)
    out = dict(method=name, readout=kind, embedding_width=features['z'].shape[1],
               selection=selection, levels={}, temporal=[], boundary=[])
    errors = group_errors(prediction[:n], y, weights)
    for split, rows in ids.items():
        out['levels'][split] = {k: float(v[rows].mean()) for k, v in errors.items()}
    for col, observable in enumerate(OBSERVABLES):
        yy = y[ids['test'], 288+col]
        mse = np.mean((prediction[ids['test'], 288+col]-yy)**2)
        out.setdefault('structural_test_reductions', {})[observable] = float(1-mse/np.mean(yy**2))
    pt = prediction[n:n+144*17].reshape(144, 17, 292)
    actual = np.concatenate([(temporal['hot']-mean[:144])/scales[:144],
                            (temporal['observables'][:, :, 2:]-mean[288:])/scales[288:]], axis=-1)
    temporal_weights = np.r_[weights[:144], weights[288:]]
    zz = features['temporal_z'].astype(np.float64)
    train_z = features['z'][ids['train']].astype(np.float64)
    block_variances = [train_z[:, start:start+256].var(axis=0).mean() for start in range(0, train_z.shape[1], 256)]
    if min(block_variances) <= 0:
        raise ValueError(f'Collapsed feature block: {name}')
    block_scale = np.repeat(np.sqrt(block_variances), 256)
    for lag in pilot['temporal_lags_steps']:
        dp = np.concatenate([pt[:, lag:, :144]-pt[:, :-lag, :144], pt[:, lag:, 288:]-pt[:, :-lag, 288:]], axis=-1)
        dy = actual[:, lag:]-actual[:, :-lag]
        err, base = np.mean((dp-dy)**2, axis=(0, 1)), np.mean(dy**2, axis=(0, 1))
        dz = (zz[:, lag:]-zz[:, :-lag])/block_scale
        out['temporal'].append(dict(lag_ps=float(lag*.75), latent_block_balanced_mse=float(np.mean(dz**2)),
            latent_block_mse=[float(np.mean(dz[:, :, start:start+256]**2)) for start in range(0, dz.shape[-1], 256)],
            tda_increment_reduction=float(1-np.sum(err[:144]*temporal_weights[:144])/np.sum(base[:144]*temporal_weights[:144])),
            structural_increment_reductions=dict(zip(OBSERVABLES, (1-err[144:]/base[144:]).tolist(), strict=True))))
    pc = prediction[n+144*17:].reshape(4, 72, 2, 292)
    natural_hot = np.mean(np.sum(np.diff(actual[:, :, :144], axis=1)**2*weights[:144], axis=-1))
    natural_z = np.mean((np.diff(zz, axis=1)/block_scale)**2)
    for j, epsilon in enumerate(features['epsilons']):
        dz = (features['crossing_z'][j, :, 1].astype(np.float64)-features['crossing_z'][j, :, 0])/block_scale
        dp = pc[j, :, 1, :144]-pc[j, :, 0, :144]
        out['boundary'].append(dict(epsilon_A=float(epsilon), latent_fraction_of_075ps=float(np.mean(dz**2)/natural_z),
            decoded_hot_fraction_of_075ps=float(np.mean(np.sum(dp**2*weights[:144], axis=-1))/natural_hot)))
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f'Nonfinite readout prediction: {name}/{kind}')
    np.savez(directory/f'{kind}-errors.npz', **{k: v[ids['test']] for k, v in errors.items()},
             sources=probes['source'][ids['test']])
    np.savez_compressed(directory/f'{kind}-predictions.npz', prediction=prediction.astype(np.float32))
    write_json(directory/f'{kind}-scores.json', out)
    return out


def evaluate_features(config, name, features, data, directory, *, nonlinear=True):
    directory.mkdir(parents=True, exist_ok=True)
    pilot, probes, temporal, ids, y, mean, scales, weights, _ = data
    x = np.concatenate([features['z'], features['temporal_z'].reshape(-1, features['z'].shape[1]),
                        features['crossing_z'].reshape(-1, features['z'].shape[1])]).astype(np.float64)
    baseline, xx, model, selected = fit_ridge(x, y, ids['train'], ids['val'], weights, pilot['ridge_alphas'])
    np.savez(directory/'ridge.npz', **model, target_mean=mean, target_scales=scales, target_weights=weights)
    ridge = score_predictions(name, 'ridge', baseline, features, data, selected, directory)
    outputs = [ridge]
    if nonlinear:
        prediction, selection = nonlinear_correction(config, xx, y, baseline, ids, weights, directory,
            lambda **fields: status(config, 'cached', state='running', method=name, stage='nonlinear', **fields))
        outputs.append(score_predictions(name, 'nonlinear', prediction, features, data, selection, directory))
    return outputs


def run_cached(config):
    torch.set_num_threads(config['cpu_threads'])
    root = Path(config['output'])/'technical/cached'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'status.json').exists():
        raise FileExistsError(f'Preserve previous readout attempt: {root}')
    data = inputs(config)
    started = time.monotonic()
    provenance = dict(config=config, inputs=data[-1], protocol='cached_context_recovery_v1')
    try:
        for state in config['readout']['states']:
            for mode in config['readout']['modes']:
                name = f'{state}-{mode}'
                status(config, 'cached', state='running', method=name, stage='ridge')
                features, checksums = feature_arrays(data[0], state, mode)
                provenance['inputs'].update(checksums)
                outputs = evaluate_features(config, name, features, data, root/name)
                print('RECOVERY_READOUT', name, [(o['readout'], o['levels']['test']) for o in outputs], flush=True)
                summarize_recovery(config)
        write_json(root/'provenance.json', provenance)
        status(config, 'cached', state='complete', elapsed_seconds=time.monotonic()-started)
        summarize_recovery(config)
    except BaseException as error:
        status(config, 'cached', state='failed', error=repr(error), elapsed_seconds=time.monotonic()-started)
        raise


def summarize_recovery(config):
    root = Path(config['output'])
    paths = sorted((root/'technical/cached').glob('*/*-scores.json'))
    paths += sorted((root/'technical').glob('train-*/evaluation/*-scores.json'))
    results = [load_json(p) for p in paths]
    if not results:
        raise ValueError(f'No complete recovery scores: {root}')
    snapshot_metric_docs(root, 'mace_context_recovery')
    rows = []
    for result in results:
        temporal = result['temporal'][0]
        rows.append(dict(method=result['method'], readout=result['readout'], embedding_width=result['embedding_width'],
            hot_test_mse=result['levels']['test']['hot'], relaxed_test_mse=result['levels']['test']['relaxed'],
            q6_test_reduction=result['structural_test_reductions']['q6'],
            q6_increment_reduction_075ps=temporal['structural_increment_reductions']['q6'],
            density_increment_reduction_075ps=temporal['structural_increment_reductions']['nearest_shell_density'],
            distance_increment_reduction_075ps=temporal['structural_increment_reductions']['mean_r12_A'],
            hot_increment_reduction_075ps=temporal['tda_increment_reduction'],
            latent_block_balanced_mse_075ps=temporal['latent_block_balanced_mse'],
            boundary_latent_fraction=result['boundary'][-1]['latent_fraction_of_075ps'],
            boundary_decoded_hot_fraction=result['boundary'][-1]['decoded_hot_fraction_of_075ps']))
    with (root/'tables/comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    comparisons = {}
    lookup = {(r['method'], r['readout']): p for r, p in zip(results, paths, strict=True)}
    for result, path in zip(results, paths, strict=True):
        state = result['method'].split('-')[0]
        ref = (state+'-mean80', result['readout']) if state in ['frozen', 'trained'] else ('dual_ssl', result['readout'])
        if ref not in lookup or lookup[ref] == path:
            continue
        with np.load(path.with_name(result['readout']+'-errors.npz')) as candidate, np.load(
                lookup[ref].with_name(result['readout']+'-errors.npz')) as reference:
            np.testing.assert_array_equal(candidate['sources'], reference['sources'])
            comparisons[result['method']+'/'+result['readout']] = {k: paired_interval(reference[k][None], candidate[k][None],
                candidate['sources'], seed=config['seed'], draws=config['bootstrap_draws']) for k in GROUPS}
    write_json(root/'technical/summary.json', dict(results=results, comparisons=comparisons))
    lines = ['# MACE context information recovery', '',
        'Exploratory source-held-out comparison. Lower TDA MSE is better; higher increment error reduction is better. '
        'Increment readouts use embeddings at both times and are not forecasts. '
        'Frozen fusion shares one backbone; trained fusion combines two separately trained backbones. '
        'Nonlinear means a validation-selected residual MLP added to ridge, with unchanged encoder features.', '',
        '| Method | Readout | Instantaneous TDA MSE | Relaxed TDA MSE | q6 change error reduction |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['readout']} | {r['hot_test_mse']:.6f} | {r['relaxed_test_mse']:.6f} | {100*r['q6_increment_reduction_075ps']:.2f}% |")
    lines += ['', '[Full comparison](tables/comparison.csv) and [metric definitions](tables/METRICS.md).']
    plot_recovery(rows, root)
    lines += ['', '![Cached readout comparison](plots/cached-readouts.png)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    publish(config, ['README.md', 'tables/comparison.csv', 'tables/METRICS.md', 'technical/summary.json', 'technical/metric-contract.json',
                     'plots/cached-readouts.png', 'plots/cached-readouts.pdf'])


def plot_recovery(rows, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['trained-mean80', 'trained-halo_inner', 'trained-halo_center', 'trained-fusion']
    selected = {(r['method'], r['readout']): r for r in rows}
    available = [name for name in names if all((name, kind) in selected for kind in ['ridge', 'nonlinear'])]
    if not available:
        available = [name.replace('trained-', 'frozen-') for name in names
                     if all((name.replace('trained-', 'frozen-'), kind) in selected for kind in ['ridge', 'nonlinear'])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(available))
    for kind, offset, color in [('ridge', -.19, '#007f70'), ('nonlinear', .19, '#bc5a20')]:
        for ax, key, factor in zip(axes, ['hot_test_mse', 'relaxed_test_mse', 'q6_increment_reduction_075ps'], [1, 1, 100], strict=True):
            ax.bar(x+offset, [factor*selected[(name, kind)][key] for name in available], .38, color=color, label=kind)
    labels = [name.split('-', 1)[1].replace('mean80', 'Original').replace('halo_inner', 'Smooth inner')
              .replace('halo_center', 'Center').replace('fusion', 'Inner + center') for name in available]
    for ax, title, ylabel in zip(axes, ['Instantaneous topology', 'Relaxed topology', 'Local bond-order changes'],
        ['Balanced test MSE (lower is better)', 'Balanced test MSE (lower is better)',
         'q6 increment error reduction at 0.75 ps (%)'], strict=True):
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle('Readout recovery on retained embeddings; encoders unchanged')
    fig.savefig(root/'plots/cached-readouts.png', dpi=180)
    fig.savefig(root/'plots/cached-readouts.pdf')
    plt.close(fig)
