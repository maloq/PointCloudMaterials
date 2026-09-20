"""Identical source-held-out probes with symmetric controls for both encoders."""
import json
from pathlib import Path
import time

import numpy as np

from src.data.structural_pretraining.prepare import file_hash, save_json
from .comparison_data import load
from .probe import selected_prediction, predict_path


def feature_sets(a, task, spatial=False):
    radial = np.concatenate((a['radial'], a['context']), axis=1).astype(float)
    base = np.concatenate((radial, a['radial_gatr'], a['radial_mace']), axis=1)
    result = dict(radial=radial, radial_control=base)
    for architecture in ('gatr', 'mace'):
        z, r = a[architecture].astype(float), a['radial_'+architecture].astype(float)
        result['plus_'+architecture] = np.concatenate((base, z), axis=1)
        result['delta_'+architecture] = np.concatenate((base, z-r), axis=1)
        result['duplicate_'+architecture] = np.concatenate((base, r), axis=1)
        if not spatial:
            result['old_'+architecture] = np.concatenate((base, a['old_'+architecture]), axis=1)
    result['plus_soap'] = np.concatenate((base, a['soap']), axis=1)
    if not spatial:
        result['plus_tda'] = np.concatenate((base, a['tda']), axis=1)
    if task == 'future':
        current = np.concatenate((base, a['bond'], a['angular']), axis=1)
        result['current_order'] = current
        for architecture in ('gatr', 'mace'):
            z, r = a[architecture].astype(float), a['radial_'+architecture].astype(float)
            result['current_'+architecture] = np.concatenate((current, z), axis=1)
            result['current_delta_'+architecture] = np.concatenate((current, z-r), axis=1)
            result['current_duplicate_'+architecture] = np.concatenate((current, r), axis=1)
    return result


def targets(a, task):
    return np.concatenate((a['bond'], a['angular']), axis=1).astype(float) if task == 'structure' else a['future'].astype(float)


def run(config):
    a, sources, _ = load(config)
    spatial, _, _ = load(config, 'spatial')
    root = Path(config['output'])/'technical'
    for task in ('structure', 'future'):
        y = targets(a, task)
        eligible = np.ones(len(y), bool) if task == 'structure' else a['future_eligible']
        test_sets = feature_sets(spatial, 'structure', spatial=True) if task == 'structure' else {}
        for method, x in feature_sets(a, task).items():
            for family in ('linear', 'nonlinear'):
                for source in sources:
                    sid = source['id']
                    dest = root/'probes'/task/family/method/f'{sid}.npz'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fit = np.flatnonzero(eligible & (a['frame'] % config['probe_stride'] == 0) & (a['source'] != sid))
                    test = np.flatnonzero(eligible & (a['source'] == sid))
                    seed = config['seed']+sid
                    if not dest.with_suffix('.json').exists():
                        started = time.monotonic()
                        p, mean, scale, selected = selected_prediction(x, y, fit, test, a['source'], family, config,
                            seed, task == 'future', pooled_tail=144 if method == 'plus_tda' else 0)
                        if not np.isfinite(p).all():
                            raise FloatingPointError(f'Nonfinite {task}/{family}/{method}/{sid}')
                        np.savez(dest, indices=test, prediction=p, target_mean=mean, target_scale=scale)
                        save_json(dest.with_suffix('.json'), dict(task=task, family=family, method=method, test_source=sid,
                            training_sources=np.unique(a['source'][fit]).tolist(), training_rows=len(fit), test_rows=len(test),
                            input_dimensions=x.shape[1], rff_seed=seed, sha256=file_hash(dest),
                            seconds=time.monotonic()-started, **selected))
                    receipt = json.loads(dest.with_suffix('.json').read_text())
                    if file_hash(dest) != receipt['sha256'] or sid in receipt['training_sources']:
                        raise ValueError('Corrupt or non-held-out completed probe')
                    if method not in test_sets:
                        continue
                    out = root/'spatial-probes'/family/method/f'{sid}.npz'
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if out.with_suffix('.json').exists():
                        continue
                    rows = np.flatnonzero(spatial['source'] == sid)
                    all_pred, mean, scale = predict_path(x[fit], y[fit], test_sets[method][rows], a['source'][fit], family,
                        config['ridge_penalties'], seed, config['rff_dimensions'], False)
                    indices = [config['ridge_penalties'].index(p) for p in receipt['penalty_per_target']]
                    p = np.stack([all_pred[k, :, j] for j, k in enumerate(indices)], axis=1)
                    np.savez(out, indices=rows, prediction=p, target_mean=mean, target_scale=scale)
                    save_json(out.with_suffix('.json'), dict(source=sid, training_sources=receipt['training_sources'],
                        sha256=file_hash(out), trajectory_selection_sha256=file_hash(dest.with_suffix('.json'))))
                print(f'Completed probes: {task}/{family}/{method}', flush=True)
