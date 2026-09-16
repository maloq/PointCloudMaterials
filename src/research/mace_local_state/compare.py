"""Source-separated local information, temporal smoothness and uncertain states."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.experiment_runner.artifacts import write_json
from src.research.mace_context.cluster_diagnosis import project, projector
from .data import load_prepared, progress
from .methods import (AffineMap, adjusted_pair_agreement, fit_states, neighborhood_metrics,
                      pca_map, residualize, ridge_maps, state_membership, temporal_map, uncertainty)
from .physics import GROUP_NAMES, TEACHER_COLUMNS


def blocks(config, features, local):
    head = projector(config)
    arrays = dict(anchor=features['z'], previous=local['previous'], spatial=local['spatial'],
                  temporal=features['temporal_z'], crossing=features['crossing_z'])
    result = {}
    for name, x in arrays.items():
        shape = x.shape[:-1]
        p = project(head, x[..., :256].reshape(-1, 256).astype(np.float32)).reshape(*shape, 128)
        result[name] = dict(inner=x[..., :256].astype(np.float64), dual=x.astype(np.float64), projector=p.astype(np.float64))
    return result


def correlation(a, b, context):
    values = []
    for key in np.unique(context):
        ids = context == key
        x, y = a[ids]-a[ids].mean(0), b[ids]-b[ids].mean(0)
        denominator = np.sqrt(np.square(x).sum()*np.square(y).sum())
        if denominator <= 0:
            raise ValueError('Collapsed local representation during temporal selection')
        values.append(np.sum(x*y)/denominator)
    return float(np.mean(values))


def fit(config, root):
    probes, temporal, features, local = load_prepared(config, root)
    values = blocks(config, features, local)
    train, val = probes['split'] == 'train', probes['split'] == 'val'
    contexts = probes['context']
    y = local['group'][:, 0, TEACHER_COLUMNS]
    ym = y[train].mean(0)
    ys = residualize(y[train], contexts[train]).std(0)
    if np.any(ys <= 0):
        raise ValueError('A physical teacher has no within-context training variation')
    y = (y-ym)/ys
    np.savez(root/'technical/teacher-scale.npz', mean=ym, scale=ys, columns=TEACHER_COLUMNS)
    directory = root/'technical/maps'; directory.mkdir(exist_ok=True)
    metadata, selection = {}, []
    for block in ['inner', 'dual', 'projector']:
        x = values['anchor'][block]
        name = f'{block}_pca16'
        pca_map(x[train], 16).save(directory/f'{name}.npz')
        metadata[name] = dict(block=block, family='pca', dimensions=16)
        if block == 'projector':
            continue
        for dim in config['temporal_dimensions']:
            name = f'{block}_tcca{dim}'
            best_score = -np.inf
            for regularization in config['temporal_regularization']:
                model, fitted = temporal_map(x[train], values['previous'][block][train], contexts[train], dim, regularization)
                score = correlation(model(x[val]), model(values['previous'][block][val]), contexts[val])
                selection.append(dict(model=name, parameter=regularization, criterion='validation_local_correlation', value=score))
                if score > best_score:
                    best_score, best_model, best_fitted, chosen = score, model, fitted, regularization
            best_model.save(directory/f'{name}.npz'); best_fitted.save(directory/f'{name}-covariance.npz')
            metadata[name] = dict(block=block, family='tcca', dimensions=dim, regularization=chosen,
                                  validation_local_correlation=best_score)
        # A supervised Mahalanobis distance: Euclidean distance after this map
        # measures differences in predicted, standardized group observables.
        candidates = ridge_maps(residualize(x[train], contexts[train]),
                                residualize(y[train], contexts[train]), config['ridge_alphas'])
        scores = []
        for alpha, candidate in zip(config['ridge_alphas'], candidates, strict=True):
            candidate.mean = x[train].mean(0); candidate.offset = y[train].mean(0)
            error = residualize(candidate(x[val])-y[val], contexts[val])
            score = float(np.mean(np.square(error)))
            scores.append(score)
            selection.append(dict(model=f'{block}_physical10', parameter=alpha,
                                  criterion='validation_within_context_teacher_mse', value=score))
        chosen = int(np.argmin(scores)); name = f'{block}_physical10'
        candidates[chosen].save(directory/f'{name}.npz')
        metadata[name] = dict(block=block, family='physical', dimensions=len(TEACHER_COLUMNS),
            alpha=config['ridge_alphas'][chosen], validation_within_context_teacher_mse=scores[chosen])
    write_json(root/'technical/maps.json', metadata)
    pd.DataFrame(selection).to_csv(root/'tables/selection.csv', index=False)
    progress(root, 'maps_fitted', models=len(metadata))


def physical_probes(z, target, probes, alphas):
    train, val, test = [probes['split'] == s for s in ['train', 'val', 'test']]
    ym, ys = target[train].mean(0), target[train].std(0)
    # TDA pixels include exact zeros and very small Gaussian tails. Use one
    # RMS training scale per complete 144D block, not inverse pixel variances.
    for section in [slice(16, 160), slice(160, 304)]:
        ys[section] = np.sqrt(np.mean(np.square(ys[section])))
    if np.any(ys <= 0):
        raise ValueError('A held-out physical probe target has zero training variation')
    y = (target-ym)/ys
    candidates = ridge_maps(z[train], y[train], alphas)
    errors = np.stack([np.mean(np.square(m(z[val])-y[val]), axis=0) for m in candidates])
    chosen = np.argmin(errors, axis=0)
    predictions = {k: candidates[k](z) for k in np.unique(chosen)}
    prediction = np.column_stack([predictions[k][:, j] for j, k in enumerate(chosen)])*ys+ym
    rows = []
    for source in np.unique(probes['source'][test]):
        ids = test & (probes['source'] == source)
        for j in range(target.shape[1]):
            mse = np.mean(np.square(prediction[ids, j]-target[ids, j]))
            variance = target[ids, j].var()
            rows.append(dict(source=int(source), target=j, alpha=alphas[chosen[j]],
                test_r2=1-mse/variance if variance > 0 else np.nan, normalized_mse=mse/ys[j]**2))
    return rows


def evaluate(config, root):
    probes, temporal, features, local = load_prepared(config, root)
    values = blocks(config, features, local)
    metadata = json.loads((root/'technical/maps.json').read_text())
    train = probes['split'] == 'train'; test = probes['split'] == 'test'
    teacher = np.load(root/'technical/teacher-scale.npz')
    target = (local['group'][:, 0, TEACHER_COLUMNS]-teacher['mean'])/teacher['scale']
    rows_info, rows_neighbor, rows_smooth, rows_states, rows_agreement, rows_stability = [], [], [], [], [], []
    rng = np.random.default_rng(config['seed'])
    state_dir = root/'technical/states'; state_dir.mkdir(exist_ok=True)
    # Topology is never used to fit the representation or choose its parameters.
    targets = np.column_stack([local['group'][:, 0], probes['hot'], probes['relaxed']])
    target_names = GROUP_NAMES+[f'instant_tda_{i}' for i in range(144)]+[f'relaxed_tda_{i}' for i in range(144)]
    for name, spec in metadata.items():
        model = AffineMap.load(root/f'technical/maps/{name}.npz')
        x = {k: model(v[spec['block']]) for k, v in values.items()}
        z = x['anchor']; z2 = .5*(z+x['previous'])
        for window, zz in [('snapshot', z), ('trailing_075ps', z2)]:
            for row in physical_probes(zz, targets, probes, config['ridge_alphas']):
                row.update(model=name, window=window, observable=target_names[row.pop('target')])
                rows_info.append(row)
            for context, imbalance, recall in neighborhood_metrics(zz[test], target[test], probes['context'][test], config['neighbor_k']):
                source = int(probes['source'][np.flatnonzero(probes['context'] == context)[0]])
                rows_neighbor.append(dict(model=name, window=window, source=source, context=int(context),
                    physical_rank_imbalance=imbalance, physical_neighbor_recall=recall))
        # Train normalization uses within-context variation to prevent phase-only
        # coordinates obtaining an artificially tiny normalized temporal change.
        variance = np.mean(np.sum(residualize(z[train], probes['context'][train])**2, axis=1))
        if variance <= 0:
            raise ValueError(f'Collapsed map {name}')
        for source in np.unique(temporal['source']):
            ids = temporal['source'] == source
            trajectory = x['temporal'][ids]
            for window, trajectory2 in [('snapshot', trajectory), ('trailing_075ps', .5*(trajectory[:, 1:]+trajectory[:, :-1]))]:
                for lag in config['evaluation_lags_steps']:
                    delta = np.mean(np.sum((trajectory2[:, lag:]-trajectory2[:, :-lag])**2, axis=-1))
                    rows_smooth.append(dict(model=name, window=window, source=int(source), lag_ps=lag*.75,
                        temporal_change_over_local_variance=delta/variance))
        natural = np.mean(np.sum(np.diff(x['temporal'], axis=1)**2, axis=-1))
        cross = np.mean(np.sum((x['crossing'][-1, :, 0]-x['crossing'][-1, :, 1])**2, axis=-1))
        # Spatial paired centers come from the actual training-view producer.
        for source in np.unique(probes['source'][test]):
            ids = test & (probes['source'] == source)
            rows_smooth.append(dict(model=name, window='snapshot', source=int(source), lag_ps=0.,
                spatial_change_over_local_variance=np.mean(np.sum((z[ids]-x['spatial'][ids])**2, axis=-1))/variance,
                crossing_1e4_over_075ps=cross/natural))
        for size in config['state_minimum_sizes']:
            key = f'{name}-m{size}'
            state = fit_states(z[train], size, config['state_minimum_samples'])
            joblib.dump(state, state_dir/f'{key}.joblib')
            a, strength, membership = state_membership(state, z[test])
            uncertainty_values = uncertainty(membership)
            np.savez(state_dir/f'{key}-test.npz', rows=np.flatnonzero(test), labels=a, strength=strength,
                membership=membership, **uncertainty_values)
            for source in np.unique(probes['source'][test]):
                ids = probes['source'][test] == source
                rows_states.append(dict(model=name, minimum_size=size, source=int(source),
                    states=len(state.cluster_persistence_), assigned_fraction=float(np.mean(a[ids] >= 0)),
                    mean_strength=float(strength[ids].mean()),
                    mean_unassigned_mass=float(uncertainty_values['unassigned_mass'][ids].mean()),
                    mean_ambiguity=float(uncertainty_values['ambiguity'][ids].mean())))
            b = state_membership(state, x['spatial'][test], soft=False)[0]
            rows_agreement.append(dict(model=name, minimum_size=size, pair='spatial', lag_ps=0.,
                **adjusted_pair_agreement(a, b)))
            for window, zz in [('snapshot', x['temporal']), ('trailing_075ps', .5*(x['temporal'][:, 1:]+x['temporal'][:, :-1]))]:
                labels = state_membership(state, zz.reshape(-1, zz.shape[-1]), soft=False)[0].reshape(zz.shape[:2])
                np.save(state_dir/f'{key}-{window}-temporal-labels.npy', labels)
                for lag in config['evaluation_lags_steps']:
                    rows_agreement.append(dict(model=name, minimum_size=size, pair=window, lag_ps=lag*.75,
                        **adjusted_pair_agreement(labels[:, lag:].ravel(), labels[:, :-lag].ravel())))
            # Source subsampling (not atom bootstrapping) measures catalog stability.
            sources = np.unique(probes['source'][train])
            for draw in range(config['state_source_subsamples']):
                keep_sources = rng.choice(sources, int(np.ceil(.8*len(sources))), replace=False)
                keep = train & np.isin(probes['source'], keep_sources)
                repeated = fit_states(z[keep], size, config['state_minimum_samples'])
                labels = state_membership(repeated, z[test], soft=False)[0]
                both = (a >= 0) & (labels >= 0)
                identifiable = both.sum() > 1 and len(np.unique(a[both])) > 1 and len(np.unique(labels[both])) > 1
                rows_stability.append(dict(model=name, minimum_size=size, draw=draw,
                    assigned_overlap=float(both.mean()), states=len(repeated.cluster_persistence_),
                    assigned_ari=adjusted_rand_score(a[both], labels[both]) if identifiable else np.nan))
        progress(root, 'evaluate', model=name, models=len(metadata))
    tables = dict(information=rows_info, neighborhoods=rows_neighbor, smoothness=rows_smooth,
                  states=rows_states, state_agreement=rows_agreement, state_stability=rows_stability)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root/f'tables/{name}.csv', index=False)
    summarize(root)


def summarize(root):
    info = pd.read_csv(root/'tables/information.csv')
    def family(name):
        if name.startswith('instant_tda_'): return 'instantaneous_TDA'
        if name.startswith('relaxed_tda_'): return 'relaxed_TDA'
        return 'teacher_group' if GROUP_NAMES.index(name) in TEACHER_COLUMNS else 'untrained_group'
    info['target_family'] = info['observable'].map(family)
    # One source, one vote; TDA channels normalized separately by train variance.
    per_source = info.groupby(['model', 'window', 'source', 'target_family'], as_index=False)[['normalized_mse', 'test_r2']].mean()
    per_source.to_csv(root/'tables/information_summary.csv', index=False)
    summary = per_source.groupby(['model', 'window', 'target_family'])['normalized_mse'].mean().unstack('target_family')
    summary.to_csv(root/'tables/comparison.csv')
    text = ['# Frozen local-group representation comparison', '',
        'The encoder checkpoint is unchanged. Maps describe a group within a C2-tapered 5–7 A region with complete message-passing context.',
        'The short local history contains only current and previous observations, spanning 0.75 ps. There are no forecasts or global process labels.', '',
        'See `tables/comparison.csv` for source-averaged information errors, `tables/neighborhoods.csv` for within-context geometry,',
        '`tables/smoothness.csv` for short-lag changes and `tables/states.csv` for density-state coverage and uncertainty.',
        'Lower information error is better; high assignment coverage alone does not establish meaningful states.', '',
        'All definitions and limitations are in `tables/METRICS.md`. Static disordered-region discovery is a separate descriptive stage.', '']
    (root/'README.md').write_text('\n'.join(text))
