"""Independent-shot conditional-law precision on the sealed Al480 release."""
import numpy as np
from scipy.spatial.distance import cdist

from src.data.fixed_cohort.protocol import write_json
from src.research.shooting_laws.common import read, plan
from src.research.shooting_laws.data import load, STRATA
from src.research.shooting_laws.fit import source_weights, moments
from src.project_runtime.paths import resolve_path
from .common import output, table


def run(c):
    base = read(resolve_path(c['shooting_config']))
    data, manifest = load(base)
    parents = plan(base)
    sources, weights = source_weights(base, data)
    roles = np.array([parents['parents'][int(i)]['role'] for i in data['parent']])
    y = data['future'].reshape(len(sources), 12, -1)
    train = roles == 'train'
    mean, scale = moments(y[train].reshape(-1, y.shape[-1]), np.repeat(weights[train] / 12, 12))
    y = (y - mean) / scale
    rng = np.random.default_rng(c['seed'])
    sample = y[train].reshape(-1, y.shape[-1])
    sample = sample[rng.choice(len(sample), min(1024, len(sample)), replace=False)]
    bandwidth = float(np.median(cdist(sample[:512], sample[512:])))
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError('Degenerate training-defined RFF bandwidth')
    omega = rng.normal(size=(y.shape[-1], 256)) / bandwidth
    phase = rng.uniform(0, 2 * np.pi, 256)
    phi = (np.sqrt(2 / 256) * np.cos(y @ omega + phase)).astype(np.float32)
    root = output(c) / 'analyses/shooting-reliability-v1'; root.mkdir(parents=True, exist_ok=True)
    np.savez(root / 'feature-contract.npz', mean=mean, scale=scale, omega=omega, phase=phase, bandwidth=bandwidth)
    # Static descriptor -> feature mean ridge: train/selection only, no AP selector.
    x = data['descriptors']
    xm, xs = moments(x[train], weights[train]); x = (x - xm) / xs
    x = np.column_stack((np.ones(len(x)), x))
    target = phi.mean(1)
    w = weights[train] / weights[train].sum()
    gram = x[train].T @ (w[:, None] * x[train])
    rhs = x[train].T @ (w[:, None] * target[train])
    selection = roles == 'selection'; best = np.inf
    selectors = []
    for alpha in (.001, .01, .1, 1., 10.):
        penalty = np.eye(x.shape[1]) * alpha; penalty[0, 0] = 0
        beta = np.linalg.solve(gram + penalty, rhs)
        pred = x @ beta
        # Fixed-variance Gaussian feature likelihood differs from squared feature
        # error by fixed scale/constant. Do not claim full physical-law likelihood.
        error = np.mean((pred[selection, None] - phi[selection]) ** 2, axis=(1, 2))
        score = float(np.average(error, weights=weights[selection]))
        selectors.append(dict(alpha=alpha, selection_feature_mse=score))
        if score < best:
            best=score; chosen=alpha; fitted=pred
    prior = np.average(target[train], axis=0, weights=weights[train])
    predictions = dict(prior=np.broadcast_to(prior, fitted.shape), descriptor_ridge=fitted)
    np.savez_compressed(root / 'predictions.npz', **predictions, parent=data['parent'], atom_ids=data['atom_ids'])
    rows, retrieval = [], []
    test = roles == 'test'
    for budget in (2, 4, 8, 12):
        for repeat in range(c['shot_partitions']):
            ix = rng.permutation(12)[:budget]
            left = phi[:, ix[:budget // 2]].mean(1); right = phi[:, ix[budget // 2:]].mean(1)
            discrepancy = np.sum((left - right) ** 2, axis=1)
            for pop, mask in [('all', np.ones(len(x), bool))] + [(name, data['strata'] == i) for i, name in enumerate(STRATA)]:
                for source in np.unique(sources[test]):
                    keep = test & mask & (sources == source)
                    if not keep.any(): continue
                    for arm, pred in predictions.items():
                        corrected = np.sum((pred - left) * (pred - right), axis=1)
                        rows.append(dict(total_shots=budget, repeat=repeat, population=pop, source=source, arm=arm,
                            corrected_squared_feature_error=float(np.average(corrected[keep], weights=weights[keep])),
                            split_discrepancy=float(np.average(discrepancy[keep], weights=weights[keep]))))
            # Fixed descriptor neighbors; the empirical oracle chooses on left,
            # scores on right. Query set is declared from present strata only.
            if repeat >= c['retrieval_partitions']: continue
            for source in np.unique(sources[test]):
                indices = np.flatnonzero(test & (sources == source))
                select_rng = np.random.default_rng(c['seed'] + list(sorted(set(sources))).index(source))
                queries = select_rng.choice(indices, min(c['retrieval_queries_per_source'], len(indices)), replace=False)
                for i in queries:
                    candidates = np.flatnonzero(test & (sources != source) & (data['strata'] == data['strata'][i]))
                    if not len(candidates): continue
                    for name, features in [('descriptor_prediction', fitted), ('split_shot_oracle', left)]:
                        order = np.argsort(np.sum((features[candidates] - features[i]) ** 2, axis=1), kind='stable')[:5]
                        neighbors = candidates[order]
                        retrieval.append(dict(total_shots=budget, repeat=repeat, source=source, population=STRATA[int(data['strata'][i])],
                            parent=int(data['parent'][i]), atom_id=int(data['atom_ids'][i]), arm=name,
                            scored_distance=float(np.mean(np.sum((right[neighbors] - right[i]) ** 2, axis=1))),
                            inclusion_weight=float(data['weights'][i]), neighbor_indices=';'.join(map(str, neighbors))))
    table(c, 'shooting-reliability-v1', 'source-errors', rows)
    table(c, 'shooting-reliability-v1', 'cross-fit-retrieval', retrieval)
    table(c, 'shooting-reliability-v1', 'ridge-selection', selectors)
    # Whole-source uncertainty; partitions do not count as new independent parents.
    summary = []
    for budget in (2, 4, 8, 12):
        for pop in ('all', *STRATA):
            group = [r for r in rows if r['total_shots'] == budget and r['population'] == pop]
            ss = sorted({r['source'] for r in group})
            if not ss: continue
            delta = np.array([np.mean([r['corrected_squared_feature_error'] for r in group if r['source'] == s and r['arm'] == 'descriptor_ridge'])
                              - np.mean([r['corrected_squared_feature_error'] for r in group if r['source'] == s and r['arm'] == 'prior']) for s in ss])
            draws = rng.integers(len(ss), size=(2000, len(ss)))
            low, high = np.quantile(delta[draws].mean(1), [.025, .975])
            summary.append(dict(total_shots=budget, population=pop, sources=len(ss), ridge_minus_prior=float(delta.mean()),
                                ci_low=float(low), ci_high=float(high)))
    table(c, 'shooting-reliability-v1', 'source-bootstrap', summary)
    write_json(root / 'technical/complete.json', dict(data_identity=manifest['identity'], selected_ridge=chosen,
        target='fixed training-standardized joint 3/6/12ps physical path RFF',
        interpretation='historical test reused; corrected errors can be negative; oracle finite-shot distance is noisy',
        candidate_rule='different historical test source, same present stratum; no temperature input or matching',
        uncertainty='paired source bootstrap after averaging shot partitions'))
