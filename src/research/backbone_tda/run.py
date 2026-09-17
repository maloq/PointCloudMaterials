"""Source-held-out TDA information retention on frozen physical snapshot states."""
import fcntl
import importlib.metadata
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.analysis.topology_metrics import paired_source_gain
from src.data.predictive_memory.prepare import file_hash, write_json
from src.data_utils.topology_targets import fit_targets, transform_target, raw_prediction
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_table
from src.project_runtime.paths import load_json, resolve_path
from src.research.local_predictability.backbone_data import prepare_data
from src.research.local_predictability.native_queue import configure_file_limit
from src.research.mace_tda_ridge_audit.math import balanced_errors
from .data import (KINDS, check_deadline, progress, prepare_targets, extract_states,
                   current_classes, save_arrays)
from .probes import select_ridge, residual_probe, scores


def run(config_path):
    config = load_json(config_path)
    parent = load_json(resolve_path(config['parent_config']))
    output = resolve_path(config['output'])
    root = output/'technical'; root.mkdir(parents=True, exist_ok=True)
    cache = resolve_path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(config, parent, output, root, cache)
        except Exception as exc:
            write_json(root/'status.json', dict(state='failed', error=repr(exc), traceback=traceback.format_exc()))
            raise


def execute(config, parent, output, root, cache):
    torch.set_num_threads(config['torch_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure_file_limit()
    check_deadline(config)
    contract = check_metric_docs(family='backbone_tda')['backbone_tda']
    progress(root, 'verify_data')
    # Spawn CPU label workers before initializing CUDA.
    data = prepare_data(parent, root, device='cpu')
    identity = dict(protocol='backbone_instantaneous_tda_v1', data=data.identity,
        config={k: v for k, v in config.items() if k != 'deadline_utc'},
        files=contract['files'], versions={name: importlib.metadata.version(name)
            for name in ('numpy', 'scikit-learn', 'torch', 'gudhi')},
        checkpoints={kind: file_hash(resolve_path(parent['output'])/'technical'/kind/
                     'physical_means/snapshot/best.pt') for kind in KINDS})
    manifest = root/'identity.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('TDA identity changed; choose a distinct output/cache for a new experiment')
    write_json(manifest, identity)
    targets = prepare_targets(config, parent, data, root, cache, identity)
    data.windows.device = torch.device('cuda')
    progress(root, 'states', completed_sources=0, sources=150)
    states = extract_states(config, parent, data, root, cache, identity)
    splits = data.splits('physical_means')
    train, selection = splits['train'], splits['selection']
    scaling = fit_targets(targets[train], components=144, floor_fraction=config['block_scale_floor_fraction'])
    if not np.all(scaling['block_scale'] > 0):
        raise ValueError('Training TDA blocks have no usable variance')
    scaled = transform_target(targets, scaling, 'blocks')
    classes = current_classes(parent, data)
    np.savez(root/'target_scaling.npz', **scaling)
    save_arrays(root/'paired_data.npz', identity, targets=targets,
        **{f'{kind}_state': value for kind, value in states.items()},
        source=data.labels['source_id'], center=data.labels['center_id'], anchor=data.labels['anchor'],
        split=data.labels['split'], current_ptm_class=classes, conditions=data.cond.numpy())
    predictions = {'training_mean': np.broadcast_to(scaling['tda_mean'], targets.shape).copy()}
    features = {'conditions': data.cond.numpy(), **states}
    fits = {}
    for kind, x in features.items():
        progress(root, 'readouts', model=kind)
        check_deadline(config)
        ridge, selected = select_ridge(x, targets, train, selection,
                                      config['ridge_alphas'], scaling['block_scale'])
        fits[kind] = dict(ridge=selected)
        predictions[f'{kind}_ridge'] = ridge
        # Retain directly usable weights, independently checked against ridge_path.
        mean, scale = x[train].mean(0, dtype=np.float64), x[train].std(0, dtype=np.float64)
        scale[scale == 0] = 1.
        linear = Ridge(alpha=selected['alpha'], solver='svd').fit((x[train]-mean)/scale, targets[train])
        np.savez(root/f'{kind}_ridge.npz', feature_mean=mean, feature_scale=scale,
                 coefficient=linear.coef_, intercept=linear.intercept_, alpha=selected['alpha'])
        np.testing.assert_allclose(linear.predict((x-mean)/scale), ridge, rtol=1e-5, atol=1e-6)
        if kind in KINDS:
            enhanced, artifact = residual_probe(x, scaled, transform_target(ridge, scaling, 'blocks'),
                                               train, selection, config)
            predictions[f'{kind}_nonlinear'] = raw_prediction(enhanced, scaling, 'blocks')
            torch.save(dict(identity=identity, config=config, scaling=scaling, **artifact),
                       root/f'{kind}_nonlinear.pt')
            fits[kind]['nonlinear'] = {k: artifact[k] for k in ('best_step', 'updates', 'trace')}
    write_json(root/'readout_selection.json', fits)
    save_arrays(root/'predictions.npz', identity, **predictions)
    progress(root, 'evaluation')
    source = data.labels['source_id']
    contexts = np.array([f'{sid}:{frame}' for sid, frame in zip(source, data.labels['anchor'], strict=True)])
    temperatures = np.array([row['temperature_K'] for row in data.windows.rows])
    metrics, comparisons = {}, {}
    for split in ('selection', 'calibration', 'test'):
        rows = splits[split]
        populations = {'all': rows, 'noncrystalline': rows[~np.isin(classes[rows], [1, 2, 3])]}
        populations.update({f'temperature_{t}K': rows[temperatures[rows] == t] for t in np.unique(temperatures)})
        metrics[split] = {}
        for population, indices in populations.items():
            metrics[split][population] = {name: scores(value[indices], targets[indices], source[indices],
                contexts[indices], scaling['block_scale']) for name, value in predictions.items()}
            if split == 'test' and population in ('all', 'noncrystalline') and len(indices):
                comparisons[population] = {}
                for readout in ('ridge', 'nonlinear'):
                    a = balanced_errors(predictions[f'mace_{readout}'][indices], targets[indices], scaling['block_scale'])[0]
                    b = balanced_errors(predictions[f'axial_gatr_{readout}'][indices], targets[indices], scaling['block_scale'])[0]
                    result = paired_source_gain(a, b, source[indices], config['seed'])
                    result['limitation'] = 'One seed; 4000 paired whole-test-source resamples measure source uncertainty only.'
                    comparisons[population][readout] = result
    result = dict(metrics=metrics, gatr_relative_to_mace=comparisons)
    write_json(root/'metrics.json', result)
    write_metric_table(result, output, family='backbone_tda', name='topology')
    write_json(root/'status.json', dict(state='complete', rows=len(targets), test_sources=30,
        identity_sha256=file_hash(root/'identity.json'), predictions_sha256=file_hash(root/'predictions.npz')))
    print('TDA experiment complete; metrics and paired predictions saved.', flush=True)
