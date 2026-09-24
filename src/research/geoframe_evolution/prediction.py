"""The existing 45-root Al future assay, evaluated with GeoFrame's input convention."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from src.project_runtime.paths import resolve_path
from src.research.structural_state.data import Corpus
from src.research.structural_state.dynamics import targets
from src.research.structural_state.evaluation import hazard_probe, regression_score
from .metrics import participation
from .reference import write_json
from src.research.trajectory_stability.audit import corpus_dynamics


def prepare():
    path = Path('configs/structural_state/future_metric_seed20260923.json')
    config = json.loads(path.read_text())
    corpus = Corpus(SimpleNamespace(cache=resolve_path(config['cache'])))
    with np.load(resolve_path(config['cache'])/'relaxed-graphs.npz') as bank:
        positions, offsets = bank['positions'], bank['offsets']
        clouds = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            x = positions[a:b]
            if len(x)<80 or not np.array_equal(x[0], np.zeros(3)):
                raise ValueError('Future-assay patch cannot support GeoFrame center/80-neighbor convention.')
            nearest = np.argsort(np.square(x).sum(1), kind='stable')[:80]
            if np.linalg.norm(x[nearest[-1]]) >= 8.:
                raise ValueError('80-neighbor crop is truncated by the stored 8 Å support.')
            clouds.append(x[nearest]/9.192189)
    return corpus, config, np.stack(clouds)


def temporal_response(z, corpus):
    fit, test = corpus.split['fit'], corpus.split['development']
    order = StandardScaler().fit(corpus.targets['current_order'][fit]).transform(corpus.targets['current_order'])
    by_atom = {}
    for i, r in enumerate(corpus.records):
        by_atom.setdefault((r['root'], r['center_atom_id']), []).append(i)
    pairs = []
    for rows in by_atom.values():
        rows.sort(key=lambda i: corpus.records[i]['frame'])
        pairs.extend(zip(rows[:-1], rows[1:]))
    pairs = np.array(pairs)
    fit_pair = np.isin(pairs, fit).all(1); test_pair = np.isin(pairs, test).all(1)
    change = np.linalg.norm(order[pairs[:, 1]]-order[pairs[:, 0]], axis=1)
    low, high = np.quantile(change[fit_pair], [.25, .75])
    distance = np.linalg.norm(z[pairs[:, 1]]-z[pairs[:, 0]], axis=1)
    scale = participation(z[test])['pair_rms']
    result = dict(lags_ps=sorted({(corpus.records[b]['frame']-corpus.records[a]['frame'])*.75 for a,b in pairs}),
                  low_change_threshold=float(low), high_change_threshold=float(high), domains={})
    for name, mask in [('all', test_pair), ('both_PTM_other', test_pair & (corpus.targets['phase'][pairs] == 0).all(1))]:
        quiet, changed = mask & (change<=low), mask & (change>=high)
        extreme = quiet | changed
        rho = spearmanr(change[mask], distance[mask]).statistic if mask.sum()>2 and np.std(distance[mask])>0 else None
        result['domains'][name] = dict(pairs=int(mask.sum()), quiet=int(quiet.sum()), changed=int(changed.sum()),
            change_distance_spearman=float(rho) if rho is not None and np.isfinite(rho) else None,
            response_auc=float(roc_auc_score(changed[extreme], distance[extreme])) if quiet.any() and changed.any() else None,
            quiet_normalized_distance=float(distance[quiet].mean()/scale) if quiet.any() and scale>0 else None,
            changed_normalized_distance=float(distance[changed].mean()/scale) if changed.any() and scale>0 else None)
    return result


def evaluate(z, corpus, config, folder, name):
    fit, test = corpus.split['fit'], corpus.split['development']
    auxiliary, receipt, baseline = targets(corpus, config)
    source = np.array([r['source'] for r in corpus.records])
    temperature = np.array([r['temperature_K'] for r in corpus.records])
    phase = corpus.targets['phase']
    x = StandardScaler().fit(z[fit]).transform(z)
    prediction = Ridge(alpha=10.).fit(x[fit], auxiliary['future_residual'][fit]).predict(x[test])
    scores = regression_score(auxiliary['future_residual'][test], prediction, source[test], temperature[test], phase[test])
    constant = regression_score(auxiliary['future_residual'][test], np.zeros_like(prediction), source[test], temperature[test], phase[test])
    conditions = np.c_[temperature[:, None] == np.unique(temperature[fit]), corpus.targets['current_order'], corpus.geometry['relaxed']]
    conditions = StandardScaler().fit(conditions[fit]).transform(conditions).astype(np.float32)
    hazard, model, forecasts = hazard_probe(z, conditions, corpus, config, 'linear', 'cuda', None)
    np.savez(folder/f'{name}-future-predictions.npz', residual=prediction, indices=test, **{'hazard_'+k:v for k,v in forecasts.items()})
    torch.save(model, folder/f'{name}-hazard-readout.pt')
    write_json(folder/'future-target-receipt.json', receipt)
    return dict(future_residual_9ps=scores, residual_constant=constant, conditional_hazard=hazard,
                temporal_response=temporal_response(z, corpus),
                embedding_dynamics=corpus_dynamics(z, corpus))
