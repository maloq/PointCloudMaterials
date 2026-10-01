"""Local controls, frozen readouts and source-paired finite-shot diagnostics."""
import csv
import math
from pathlib import Path
import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.shooting_laws.common import read
from .data import cache, load, moments, output


def ridge(x, target, data, alphas):
    tr = data['roles'] == 'train'
    selection = data['roles'] == 'selection'
    mean, scale = moments(x[tr], data['weights'][tr])
    x = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    w = data['weights'][tr] / data['weights'][tr].sum()
    gram = x[tr].T @ (w[:, None]*x[tr])
    rhs = x[tr].T @ (w[:, None]*target[tr])
    best, chosen, choices = math.inf, None, []
    for alpha in alphas:
        penalty = np.eye(x.shape[1])*alpha
        penalty[0, 0] = 0
        beta = np.linalg.solve(gram+penalty, rhs)
        prediction = x @ beta
        value = float(np.average(((prediction[selection]-target[selection])**2).sum(1), weights=data['weights'][selection]))
        choices.append(dict(alpha=alpha, selection_feature_error=value))
        if value < best:
            best, chosen = value, dict(alpha=alpha, prediction=prediction, beta=beta, mean=mean, scale=scale)
    return chosen, choices


def controls(c):
    data, manifest = load(c)
    root = output(c) / 'analyses/controls-v1'
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    tr = data['roles'] == 'train'
    prior = np.average(data['target'][tr], axis=0, weights=data['weights'][tr])
    predictions = dict(prior=np.broadcast_to(prior, data['target'].shape))
    selectors = []
    for name in ['descriptors', *c['frozen_controls']]:
        fit, choices = ridge(data[name], data['target'], data, c['ridge_alphas'])
        predictions[name] = fit.pop('prediction').astype(np.float32)
        np.savez(tech/f'{name}-ridge.npz', **fit)
        selectors.extend(dict(arm=name, **row) for row in choices)
    np.savez_compressed(tech/'predictions.npz', **predictions, parent=data['parent'], atom_ids=data['atom_ids'])
    write_metric_rows(selectors, root, family='predictive_baseline', name='selection')
    write_json(tech/'complete.json', dict(state='complete', dataset=manifest['identity'],
        predictions_sha256=sha(tech/'predictions.npz'), created_online_runs=0))


def groups(data):
    test = data['roles'] == 'test'
    yield 'all', test
    for index, name in enumerate(('clear_liquid', 'visible_interface', 'crystalline_center')):
        yield name, test & (data['strata'] == index)
    for temperature in (400, 450, 500):
        yield f'T{temperature}', test & (data['temperature'] == temperature)
        yield f'T{temperature}-clear_liquid', test & (data['temperature'] == temperature) & (data['strata'] == 0)


def source_average(values, data, mask):
    sources = np.unique(data['sources'][mask])
    return sources, np.array([np.average(values[ix], axis=0, weights=data['weights'][ix])
        for source in sources for ix in [mask & (data['sources'] == source)]])


def collect(c):
    data, manifest = load(c)
    root = output(c) / 'analyses/comparison-v1'
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    control_root = output(c) / 'analyses/controls-v1/technical'
    complete = read(control_root/'complete.json')
    if complete['dataset'] != manifest['identity'] or sha(control_root/'predictions.npz') != complete['predictions_sha256']:
        raise ValueError('Control prediction identity mismatch')
    controls_bank = np.load(control_root/'predictions.npz')
    predictions = {(name, -1): controls_bank[name] for name in ('prior', 'descriptors', *c['frozen_controls'])}
    states = {name: data[name] for name in ('descriptors', *c['frozen_controls'])}
    receipts = {}
    for seed in c['fit_seeds']:
        run_root = output(c) / 'analyses' / f'joint-seed-{seed}' / 'technical'
        record = read(run_root/'complete.json')
        if record['state'] != 'complete' or sha(run_root/'predictions.npz') != record['predictions_sha256']:
            raise ValueError(f'Incomplete/changed neural fit: {seed}')
        a = np.load(run_root/'predictions.npz')
        if not np.array_equal(a['parent'], data['parent']) or not np.array_equal(a['atom_ids'], data['atom_ids']):
            raise ValueError(f'Neural row identity mismatch: {seed}')
        predictions[('joint_mace128', seed)] = a['prediction']
        states[f'joint-{seed}'] = a['z']
        receipts[seed] = run_root/'wandb.json'
    # Freeze the encoder, then assess a fresh linear readout of the SAME task.
    probe_selectors = []
    for seed in c['fit_seeds']:
        fitted, choices = ridge(states[f'joint-{seed}'], data['target'], data, c['ridge_alphas'])
        predictions[('joint_linear_readout', seed)] = fitted.pop('prediction')
        np.savez(tech/f'joint-{seed}-readout.npz', **fitted)
        probe_selectors.extend(dict(arm=f'joint-{seed}', **row) for row in choices)
    write_metric_rows(probe_selectors, root, family='predictive_baseline', name='readout-selection')
    features = np.asarray(data['features'], dtype=np.float64)
    target = features.mean(1)
    # Unbiased finite-shot covariance trace / B. This is the all-pairs version
    # of disjoint-half error, using all 12 shots, and can yield negative errors.
    variance_of_mean = features.var(axis=1, ddof=1) / features.shape[1]
    fmap = np.load(cache(c)/'feature_map.npz')
    source_rows, physical_rows = [], []
    for (arm, seed), pred in predictions.items():
        squared = (pred-target)**2
        raw_phi = pred / np.sqrt(fmap['metric']) + fmap['center']
        raw_mean = raw_phi[:, :9]*fmap['y_scale']+fmap['y_mean']
        raw_variance = (raw_phi[:, 9:18]-raw_phi[:, :9]**2)*fmap['y_scale']**2
        y = np.asarray(data['y'], dtype=np.float64)
        mean_error = (raw_mean-y.mean(1))**2
        variance_error = (raw_variance-y.var(1, ddof=1))**2
        for group, mask in groups(data):
            values = np.column_stack((squared.sum(1), (squared-variance_of_mean).sum(1),
                variance_of_mean.sum(1), squared[:, :9].sum(1), squared[:, 9:18].sum(1),
                squared[:, 18:].sum(1), (raw_variance < 0).mean(1)))
            sources, aggregate = source_average(values, data, mask)
            for source, v in zip(sources, aggregate, strict=True):
                source_rows.append(dict(arm=arm, seed=seed, population=group, source=source,
                    feature_error=float(v[0]), corrected_feature_error=float(v[1]), shot_noise=float(v[2]),
                    mean_block_error=float(v[3]), second_moment_block_error=float(v[4]), rff_block_error=float(v[5]),
                    negative_variance_fraction=float(v[6])))
            _, physical = source_average(np.concatenate((mean_error, variance_error), 1), data, mask)
            for i, name in enumerate(manifest['target_columns']):
                physical_rows.append(dict(arm=arm, seed=seed, population=group, observable=name,
                    mean_mse=float(physical[:, i].mean()), variance_mse=float(physical[:, 9+i].mean()), sources=len(sources)))
    summary = []
    for population, _ in groups(data):
        prior = {r['source']: r['corrected_feature_error'] for r in source_rows if r['arm'] == 'prior' and r['population'] == population}
        for arm in sorted({key[0] for key in predictions}):
            rows = [r for r in source_rows if r['arm'] == arm and r['population'] == population]
            sources = sorted(prior)
            source_values = np.array([np.mean([r['corrected_feature_error'] for r in rows if r['source'] == s]) for s in sources])
            delta = source_values - np.array([prior[s] for s in sources])
            rng = np.random.default_rng(c['target']['seed'])
            resamples = rng.integers(len(sources), size=(c['bootstrap_draws'], len(sources)))
            lower, upper = np.quantile(delta[resamples].mean(1), [.025, .975])
            seeds = sorted({r['seed'] for r in rows})
            seed_values = [np.mean([r['corrected_feature_error'] for r in rows if r['seed'] == s]) for s in seeds]
            summary.append(dict(arm=arm, population=population, sources=len(sources), seeds=len(seeds),
                feature_error=float(np.mean([r['feature_error'] for r in rows])),
                corrected_feature_error=float(source_values.mean()), delta_vs_prior=float(delta.mean()),
                delta_ci_low=float(lower), delta_ci_high=float(upper),
                seed_sd=float(np.std(seed_values)),
                negative_variance_fraction=float(np.mean([r['negative_variance_fraction'] for r in rows]))))
    write_metric_rows(source_rows, root, family='predictive_baseline', name='source-scores')
    write_metric_rows(summary, root, family='predictive_baseline', name='comparison')
    write_metric_rows(physical_rows, root, family='predictive_baseline', name='physical-moments')
    # Independent questions excluded from the 274-target training objective.
    tr = data['roles'] == 'train'
    mean, scale = moments(data['extra_future'][tr], data['weights'][tr])
    extra = (data['extra_future']-mean)/scale
    probe_rows = []
    for name, z in states.items():
        fit, _ = ridge(z, extra, data, c['ridge_alphas'])
        pred = fit.pop('prediction')
        np.savez(tech/f'{name}-extra-probe.npz', **fit, target_mean=mean, target_scale=scale)
        error = ((pred-extra)**2).mean(1)
        for group, mask in groups(data):
            sources, values = source_average(error, data, mask)
            probe_rows.append(dict(representation=name, population=group, sources=len(sources),
                extra_observable_mse=float(values.mean()), selected_alpha=float(fit['alpha'])))
    write_metric_rows(probe_rows, root, family='predictive_baseline', name='extra-observable-probes')
    # Saved selected predictions and scientific metrics are durable before API updates.
    for seed, receipt in receipts.items():
        row = [r for r in source_rows if r['arm'] == 'joint_mace128' and r['seed'] == seed and r['population'] == 'all']
        update_recorded_summary(receipt, {'baseline/test_corrected_feature_error': float(np.mean([r['corrected_feature_error'] for r in row])),
            'baseline/test_feature_error': float(np.mean([r['feature_error'] for r in row]))},
            evaluation='predictive-baseline-final-v1', expected=dict(mode='online'))
    plot(c, summary)
    write_json(tech/'complete.json', dict(state='complete', dataset=manifest['identity'],
        independent_test_sources=6, prediction_seed_aggregation='mean of individual fit errors; no fitted ensemble'))


def plot(c, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = output(c)/'analyses/comparison-v1'
    (root/'plots').mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for ax, group in zip(axes, ('all', 'clear_liquid'), strict=True):
        rows = [r for r in summary if r['population'] == group]
        y = np.arange(len(rows))
        delta = np.array([r['delta_vs_prior'] for r in rows])
        lower = np.array([r['delta_ci_low'] for r in rows])
        upper = np.array([r['delta_ci_high'] for r in rows])
        ax.hlines(y, lower, upper, color='#507caa')
        ax.scatter(delta, y, color='#163a59')
        ax.set_yticks(y, [r['arm'] for r in rows]); ax.axvline(0, color='gray', lw=1)
        ax.set_title(group.replace('_', ' ')); ax.set_xlabel('Corrected feature error minus prior (lower is better)')
    fig.suptitle('Future statistics: six historical held-out sources\nSource-paired 95% bootstrap; seed errors averaged before bootstrap')
    fig.savefig(root/'plots/comparison.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for seed in c['fit_seeds']:
        path = output(c)/'analyses'/f'joint-seed-{seed}'/'tables/learning.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        ax.plot([int(r['epoch']) for r in rows], [float(r['selection_feature_error']) for r in rows], label=str(seed))
    ax.set(xlabel='Epoch', ylabel='Selection feature error', title='Checkpoint selection uses future-feature Gaussian likelihood')
    ax.legend(title='Training seed'); fig.savefig(root/'plots/learning.png', dpi=170)
    plt.close(fig)
