"""Exact frozen velocity-checkpoint features for the direct-slowness comparison."""
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from src.research.mace_velocity.inference import load_encoder
from src.research.mace_velocity.inventory import read
from src.research.mace_velocity.train import encode


def prepare(config, root):
    cache = Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    source = load_json(config['source_config'])
    manifest_path = Path(source['cache'])/'manifest.json'
    manifest = read(manifest_path)
    teacher_path = Path(source['output'])/'technical/teacher.npz'
    checkpoint = Path(config['checkpoint'])
    if sha256(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash differs from the declared experiment')
    identity = dict(protocol=config['protocol'], checkpoint_sha256=sha256(checkpoint),
                    source_manifest_sha256=sha256(manifest_path), teacher_sha256=sha256(teacher_path),
                    preparation_sha256=sha256(Path(__file__)))
    stamp = cache/'preparation.json'
    if stamp.exists() and read(stamp) != identity:
        raise ValueError(f'Preparation identity changed: {stamp}; use a new cache')
    if not stamp.exists(): write_json(stamp, identity)
    target = cache/'features.npz'
    completed = cache/'complete.json'
    if completed.exists():
        report = read(completed)
        if report['identity'] != identity or sha256(target) != report['features_sha256']:
            raise ValueError('Prepared feature cache changed')
        return
    torch.set_num_threads(config['cpu_threads'])
    model, inference = load_encoder(checkpoint, config['device'])
    inference = dict(inference, micro_batch_size=config['micro_batch_size'])
    features = []; labels = []; metadata = []; contexts = []; context_ids = {}
    started = time.monotonic()
    for number, record in enumerate(manifest['records']):
        path = Path(source['cache'])/record['file']
        if sha256(path) != record['sha256']: raise ValueError(f'Source cache changed: {path}')
        saved = cache/f"source-{record['source']['id']:04d}.npz"
        sidecar = saved.with_suffix('.json')
        with np.load(path) as a:
            x, v, pointers = a['positions'], a['velocities'], a['pointers']
            y = a['targets']
            if sidecar.exists():
                stamp_record = read(sidecar)
                if stamp_record['source_sha256'] != record['sha256'] or sha256(saved) != stamp_record['sha256']:
                    raise ValueError(f'Prepared source changed: {saved}')
                z = np.load(saved)['embedding']
            else:
                clouds = [(x[i:j], v[i:j]) for i,j in zip(pointers[:-1], pointers[1:], strict=True)]
                z = encode(inference, model, clouds, config['device']).cpu().numpy()
                if z.shape != (2*len(record['pairs']), 304):
                    raise ValueError(f'Unexpected encoder shape at {path}: {z.shape}')
                np.savez(saved, embedding=z)
                write_json(sidecar, dict(source_sha256=record['sha256'], sha256=sha256(saved)))
            features.append(z); labels.append(y)
        for pair in record['pairs']:
            metadata.append(dict(source_id=record['source']['id'], split=record['source']['split'],
                balance_group=record['source']['balance_group'], **pair))
            for step in (pair['timestep'], pair['previous_timestep']):
                key = (record['source']['id'], step)
                if key not in context_ids: context_ids[key] = len(context_ids)
                contexts.append(context_ids[key])
        if number%25 == 0:
            report = dict(state='extracting', sources=number+1, total_sources=len(manifest['records']),
                          elapsed_seconds=time.monotonic()-started)
            write_json(root/'technical/prepare-status.json', report)
            print('SMOOTH PREPARE', report, flush=True)
    z = np.concatenate(features); raw_y = np.concatenate(labels)
    norm = dict(np.load(teacher_path))
    split = np.array([{'train':0, 'val':1, 'test':2}[m['split']] for m in metadata], dtype=np.int8)
    old = (norm['z']-norm['feature_mean'])/norm['feature_scale']
    y = ((raw_y-norm['target_mean'])/norm['target_scale']).astype(np.float32)
    counts = Counter(m['balance_group'] for m in metadata if m['split']=='train')
    weights = np.array([1/counts[m['balance_group']] if m['split']=='train' else 1 for m in metadata])
    weights[split==0] /= weights[split==0].mean()
    previous = np.load(Path(source['output'])/'technical/coordinates_velocity/test-predictions.npz')
    test = np.flatnonzero(split==2)
    np.testing.assert_array_equal(test, previous['pair_ids'])
    rows = np.ravel(np.c_[2*test, 2*test+1])
    replay_error = np.linalg.norm(z[rows]-previous['embedding'])/np.linalg.norm(previous['embedding'])
    if replay_error > 2e-4: raise ValueError(f'Frozen checkpoint replay mismatch: {replay_error}')
    np.testing.assert_allclose(y[rows], previous['target'], rtol=2e-6, atol=2e-6)
    lag = np.array([m['lag_ps'] for m in metadata])
    if not np.allclose(lag[test], config['evaluation_lag_ps'], rtol=0, atol=1e-9):
        raise ValueError('Held-out physical lag differs from the declared evaluation lag')
    np.savez(target, embedding=z, teacher_structure=old.astype(np.float32), target=y,
        raw_target=raw_y, split=split, lag=lag, weights=weights.astype(np.float32),
        context=np.array(contexts), source_id=np.array([m['source_id'] for m in metadata]),
        reference_pair_ids=previous['training_reference_pair_ids'])
    report = dict(state='complete', identity=identity, features_sha256=sha256(target),
        sources=len(manifest['records']), pairs=len(metadata), replay_relative_error=float(replay_error),
        train_lags_ps=np.unique(np.round(lag[split==0],9)).tolist(),
        elapsed_seconds=time.monotonic()-started)
    write_json(completed, report); write_json(root/'technical/prepare-status.json', report)
    print('SMOOTH PREPARED', report, flush=True)


def load(config):
    path = Path(config['cache'])/'features.npz'
    report = read(Path(config['cache'])/'complete.json')
    if sha256(path) != report['features_sha256']: raise ValueError(f'Feature cache changed: {path}')
    return dict(np.load(path))
