"""Proper held-out scores, source uncertainty and split-shot law diagnostics."""
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from .common import result, plan, read
from .data import load, STRATA
from .fit import source_weights, moments


def row_scores(c, target, event, prediction, rng):
    n, branches, dim = target.shape
    lw, mu, scale = (prediction[k] for k in ('log_weight', 'mean', 'scale'))
    logp = []
    energy = []
    means = []
    predicted_phi = []
    # Common, training-defined RFF map supplied by collect; no evaluation fit.
    omega, phase = prediction['omega'], prediction['phase']
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        y, m, s, logw = target[start:stop], mu[start:stop], scale[start:stop], lw[start:stop]
        residual = (y[:, :, None] - m[:, None]) / s[:, None]
        normal = -.5 * np.sum(residual ** 2, axis=-1) - np.log(s).sum(-1)[:, None] - .5 * dim * np.log(2 * np.pi)
        logp.append(-logsumexp(normal + logw[:, None], axis=-1).mean(1))
        weight = np.exp(logw)
        means.append((weight[:, :, None] * m).sum(1))
        draws = []
        for _ in range(2):
            u = rng.random((len(y), c['score_samples']))
            component = (u[:, :, None] > np.cumsum(weight, axis=-1)[:, None]).sum(-1).clip(max=len(weight[0]) - 1)
            rows = np.arange(len(y))[:, None]
            draws.append(m[rows, component] + s[rows, component] * rng.normal(size=(len(y), c['score_samples'], dim)))
        first = np.linalg.norm(draws[0][:, :, None] - y[:, None], axis=-1).mean((1, 2))
        second = np.linalg.norm(draws[0] - draws[1], axis=-1).mean(1)
        energy.append(first - .5 * second)
        predicted_phi.append((np.sqrt(2 / len(phase)) * np.cos(draws[0] @ omega + phase)).mean(1))
    probability = prediction['event_probability']
    valid = np.all(event >= 0, axis=1)
    enll = np.full(n, np.nan)
    enll[valid] = -np.log(np.maximum(probability[valid, None, :], 1e-15)).repeat(branches, axis=1)[
        np.arange(valid.sum())[:, None], np.arange(branches)[None], event[valid]].mean(1)
    scores = dict(path_nll=np.concatenate(logp), energy_score=np.concatenate(energy), event_nll=enll,
                  mean_squared_error=np.mean((np.concatenate(means)[:, None] - target) ** 2, axis=(1, 2)))
    risks = []
    labels = []
    for h in range(3):
        ids = [1 + cause * 3 + k for cause in range(4) for k in range(h + 1)]
        risk = probability[:, ids].sum(1)
        truth = np.isin(event, ids).mean(1)
        brier = np.mean((risk[:, None] - np.isin(event, ids)) ** 2, axis=1)
        brier[~valid] = np.nan
        scores[f'brier_{c["horizons_ps"][h]:g}ps'] = brier
        risks.append(risk); labels.append(truth)
    return scores, np.concatenate(predicted_phi), np.stack(risks, 1), np.stack(labels, 1)


def grouped_scores(scores, data, p, arm, seed, weights):
    rows = []
    for population, mask in [('all', np.ones(len(data['parent']), bool))] + [(name, data['strata'] == i) for i, name in enumerate(STRATA)]:
        for source in sorted(p['source_roles']):
            members = [i for i, parent in enumerate(p['parents']) if parent['source'] == source]
            keep = mask & np.isin(data['parent'], members)
            for metric, values in scores.items():
                valid = keep & np.isfinite(values)
                if valid.any():
                    rows.append(dict(arm=arm, seed=seed, population=population, source=source,
                        role=p['source_roles'][source], metric=metric, value=float(np.average(values[valid], weights=weights[valid])),
                        observations=int(valid.sum())))
    return rows


def reliability(c, target, data, roles, sources, weights, omega, phase, root):
    phi = np.sqrt(2 / len(phase)) * np.cos(target @ omega + phase)
    records = []
    rng = np.random.default_rng(c['seed'] + 17)
    for budget in (2, 4, 8, 12):
        # B is total shots: two disjoint estimates have B/2 observations each.
        for repeat in range(16):
            ix = rng.permutation(12)[:budget]
            left, right = phi[:, ix[:budget // 2]].mean(1), phi[:, ix[budget // 2:]].mean(1)
            mismatch = np.sum((left - right) ** 2, axis=1)
            for s in np.unique(sources[roles == 'test']):
                keep = (sources == s) & (roles == 'test')
                records.append(dict(source=s, total_shots=budget, shots_per_half=budget // 2, repeat=repeat,
                    split_law_squared_discrepancy=float(np.average(mismatch[keep], weights=weights[keep]))))
    write_metric_rows(records, root, family='shooting_laws', name='split-shot-reliability')
    # These independent halves also define a non-circular retrieval oracle.
    return phi[:, :6].mean(1), phi[:, 6:].mean(1)


def retrieval(data, p, representations, left, right, root):
    # Fixed same-temperature/current-stratum candidates from DIFFERENT sources.
    parents = data['parent']
    sources = np.array([p['parents'][int(i)]['source'] for i in parents])
    roles = np.array([p['parents'][int(i)]['role'] for i in parents])
    temp = np.array([p['parents'][int(i)]['temperature_K'] for i in parents])
    records = []
    for name, z in representations.items():
        for i in np.flatnonzero(roles == 'test'):
            pool = np.flatnonzero((roles == 'test') & (sources != sources[i]) & (temp == temp[i]) & (data['strata'] == data['strata'][i]))
            if not len(pool):
                continue
            distance = np.sum((z[pool] - z[i]) ** 2, axis=1)
            nearest = pool[np.argsort(distance, kind='stable')[:5]]
            # Choosing oracle neighbors uses ONLY the first six shots. Scoring
            # every representation uses the other six, shared across candidates.
            score = np.mean(np.sum((right[nearest] - right[i]) ** 2, axis=1))
            records.append(dict(representation=name, source=sources[i], parent=int(parents[i]),
                atom_id=int(data['atom_ids'][i]), stratum=STRATA[int(data['strata'][i])],
                future_law_squared_distance=float(score), candidate_count=len(pool),
                inclusion_weight=float(data['weights'][i])))
    write_metric_rows(records, root, family='shooting_laws', name='future-neighbors')


def collect(c):
    data, manifest = load(c)
    p = plan(c)
    sources, weights = source_weights(c, data)
    roles = np.array([p['parents'][int(i)]['role'] for i in data['parent']])
    root = result(c) / 'analyses/comparison-v1'
    root.mkdir(parents=True, exist_ok=True)
    prior_root = result(c) / 'analyses/readouts-v1/prior' / str(c['fit_seeds'][0])
    with np.load(prior_root / 'normalization.npz') as a:
        ym, ys = a['y_mean'], a['y_scale']
    target = (data['future'].reshape(len(sources), 12, -1) - ym) / ys
    rng = np.random.default_rng(c['seed'])
    train = target[roles == 'train'].reshape(-1, target.shape[-1])
    sample = train[rng.choice(len(train), min(1024, len(train)), replace=False)]
    bandwidth = float(np.median(cdist(sample[:512], sample[512:])))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Degenerate training path kernel bandwidth')
    omega = rng.normal(size=(target.shape[-1], 256)) / bandwidth
    phase = rng.uniform(0, 2 * np.pi, 256)
    np.savez(root / 'kernel.npz', omega=omega, phase=phase, bandwidth=bandwidth)
    left, right = reliability(c, target, data, roles, sources, weights, omega, phase, root)
    records, calibration = [], []
    discrimination = []
    desc = data['descriptors']
    dm, ds = moments(desc[roles == 'train'], weights[roles == 'train'])
    # Fixed descriptor geometry is a transparent static reference, not a
    # separately test-selected embedding or future-conditioned caliper.
    representations = dict(static_descriptors=(desc - dm) / ds, split_shot_oracle=left)
    for arm in c['arms']:
        phis = []
        for seed in c['fit_seeds']:
            fitroot = result(c) / 'analyses/readouts-v1' / arm['name'] / str(seed)
            receipt = read(fitroot / 'complete.json')
            if receipt['data_identity'] != manifest['identity'] or sha(fitroot / 'predictions.npz') != receipt['predictions_sha256']:
                raise ValueError('Changed prediction artifact')
            with np.load(fitroot / 'normalization.npz') as a:
                if not np.array_equal(a['y_mean'], ym) or not np.array_equal(a['y_scale'], ys):
                    raise ValueError('Targets differ between model comparisons')
            with np.load(fitroot / 'predictions.npz') as a:
                prediction = dict(a)
            prediction.update(omega=omega, phase=phase)
            scores, phi, risks, truth = row_scores(c, target, data['event'], prediction, np.random.default_rng(c['seed'] + seed))
            records.extend(grouped_scores(scores, data, p, arm['name'], seed, weights))
            phis.append(phi)
            at_risk = np.all(data['event'] >= 0, axis=1) & (roles == 'test')
            event_weights = weights.copy()
            for source in np.unique(sources[at_risk]):
                group = at_risk & (sources == source)
                event_weights[group] /= event_weights[group].sum()
            from sklearn.metrics import average_precision_score, roc_auc_score
            for h in range(3):
                ids = [1 + cause * 3 + k for cause in range(4) for k in range(h + 1)]
                observed = np.isin(data['event'][at_risk], ids).ravel()
                forecast = np.repeat(risks[at_risk, h], 12)
                diagnostic_weights = np.repeat(event_weights[at_risk] / 12, 12)
                if observed.any() and not observed.all():
                    discrimination.append(dict(arm=arm['name'], seed=seed, horizon_ps=c['horizons_ps'][h],
                        ap=float(average_precision_score(observed, forecast, sample_weight=diagnostic_weights)),
                        auroc=float(roc_auc_score(observed, forecast, sample_weight=diagnostic_weights))))
                bins = np.minimum((risks[:, h] * 10).astype(int), 9)
                for b in range(10):
                    keep = at_risk & (bins == b)
                    if keep.any():
                        calibration.append(dict(arm=arm['name'], seed=seed, horizon_ps=c['horizons_ps'][h], bin=b,
                            predicted=float(np.average(risks[keep, h], weights=event_weights[keep])),
                            observed=float(np.average(truth[keep, h], weights=event_weights[keep])),
                            weight=float(event_weights[keep].sum()), observations=int(keep.sum())))
        # No fitted ensemble weights: metric below averages kernel predictions
        # across the three declared seeds solely for descriptive atlas retrieval.
        representations[arm['name']] = np.mean(phis, axis=0)
    write_metric_rows(records, root, family='shooting_laws', name='source-scores')
    write_metric_rows(discrimination, root, family='shooting_laws', name='ranking-diagnostics',
                      columns=('arm', 'seed', 'horizon_ps', 'ap', 'auroc'))
    write_metric_rows(calibration, root, family='shooting_laws', name='calibration',
                      columns=('arm', 'seed', 'horizon_ps', 'bin', 'predicted', 'observed', 'weight', 'observations'))
    summary = []
    keys = sorted({(r['arm'], r['population'], r['role'], r['metric']) for r in records})
    for arm, population, role, metric in keys:
        current = [r for r in records if (r['arm'], r['population'], r['role'], r['metric']) == (arm, population, role, metric)]
        ss = sorted({r['source'] for r in current})
        value = np.array([np.mean([r['value'] for r in current if r['source'] == s]) for s in ss])
        base = np.array([np.mean([r['value'] for r in records if r['arm'] == 'prior' and
                (r['population'], r['role'], r['metric'], r['source']) == (population, role, metric, s)]) for s in ss])
        draws = rng.integers(len(ss), size=(c['bootstrap_draws'], len(ss)))
        delta = value - base
        lo, hi = np.quantile(delta[draws].mean(1), [.025, .975])
        summary.append(dict(arm=arm, population=population, role=role, metric=metric,
            mean=float(value.mean()), delta_from_prior=float(delta.mean()), delta_ci_low=float(lo), delta_ci_high=float(hi), sources=len(ss)))
    write_metric_rows(summary, root, family='shooting_laws', name='comparison')
    retrieval(data, p, representations, left, right, root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root / 'plots'; plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric in zip(axes, ('path_nll', 'energy_score', 'event_nll')):
        rows = [r for r in summary if r['role'] == 'test' and r['population'] == 'clear_liquid' and r['metric'] == metric]
        y = np.arange(len(rows)); d = np.array([r['delta_from_prior'] for r in rows])
        axis.errorbar(d, y, xerr=[d - [r['delta_ci_low'] for r in rows], [r['delta_ci_high'] for r in rows] - d], fmt='o')
        axis.axvline(0, color='gray'); axis.set_yticks(y, [r['arm'] for r in rows]); axis.set_title(metric)
        axis.set_xlabel('Difference from prior; lower is better')
    fig.tight_layout(); fig.savefig(plots / 'clear-liquid-scores.png', dpi=160); fig.savefig(plots / 'clear-liquid-scores.pdf'); plt.close(fig)
    write_json(root / 'technical/complete.json', dict(data_identity=manifest['identity'], fits=len(c['arms']) * len(c['fit_seeds']),
        uncertainty='paired source bootstrap after averaging seeds; conditional on these fitted seeds',
        unseen_sources=int((np.array(list(p['source_roles'].values())) == 'test').sum())))
