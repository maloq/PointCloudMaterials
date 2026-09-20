"""Pinned checkpoint comparison on the original, unchanged assay population."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from src.data.structural_pretraining.prepare import file_hash, save_json, source_arrays, chart, offsets
from src.data.structural_pretraining.support import SUPPORT, local_crop, OUTER_RADIUS, REFERENCE_RADIUS
from src.project_runtime.paths import resolve_path
from .data import radial_control, future_labels


def freeze(config):
    root = Path(config['output']).resolve()
    for name in ('technical', 'tables', 'plots'):
        (root/name).mkdir(parents=True, exist_ok=True)
    path = root/'technical/plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['config'] != config:
            raise ValueError('Pinned comparison configuration changed')
        return plan
    parent_path = Path(config['parent_audit'])/'technical/plan.json'
    parent = json.loads(parent_path.read_text())
    records = {}
    for name, spec in config['checkpoints'].items():
        source = Path(spec['path'])
        if file_hash(source) != spec['sha256']:
            raise ValueError(f'Latest checkpoint changed before capture: {source}')
        dest = root/'technical'/f'{name}-last.pt'
        shutil.copy2(source, dest)
        if file_hash(dest) != spec['sha256']:
            raise ValueError('Checkpoint changed during capture')
        saved = torch.load(dest, map_location='cpu', weights_only=False)
        identity = saved['identity']
        if identity['config']['architecture'] != name or identity['config']['history_frames'] != 1:
            raise ValueError('Requires the declared snapshot architecture')
        if identity['observation_support'] != SUPPORT:
            raise ValueError('Checkpoint support differs from local observation producer')
        producer = Path(spec['producer_code']).resolve()
        for filename, expected in identity['implementation']['files'].items():
            p = Path(filename)
            p = p if p.is_absolute() else producer/p
            if file_hash(p) != expected:
                raise ValueError(f'Frozen training implementation changed: {p}')
        release = resolve_path(identity['config']['release'])/'manifest.json'
        manifest = json.loads(release.read_text())
        fitted = {s['lineage'] for s in manifest['sources'] if s['split'] in ('train', 'selection') and 'lineage' in s}
        test = {s['lineage'] for s in parent['sources'] if s['split'] == 'test'}
        if fitted & test:
            raise ValueError(f'Test ancestry overlaps training/selection: {fitted & test}')
        export = dict(architecture=name, input_frames=1, state_dim=128, scales=manifest['scales'],
            encoder={k.removeprefix('encoder.'): v for k, v in saved['model'].items() if k.startswith('encoder.')},
            identity=identity, step=saved['step'])
        export_path = root/'technical'/f'{name}.pt'
        torch.save(export, export_path)
        records[name] = dict(path=str(export_path), sha256=file_hash(export_path), original=str(source.resolve()),
            original_sha256=spec['sha256'], step=saved['step'], producer_code=str(producer),
            release_manifest_sha256=file_hash(release), scale=manifest['scales']['Al'],
            identity=identity, excluded_test_lineages=sorted(test), fitted_lineages=len(fitted))
    scales = {v['scale'] for v in records.values()}
    if scales != {parent['scale']}:
        raise ValueError(f'Material scale changed: {scales}, parent {parent["scale"]}')
    plan = dict(protocol='paired_conditional_information_local_v1', config=config,
        sources=parent['sources'], checkpoints=records, scale=parent['scale'], cadence_ps=parent['cadence_ps'],
        support=SUPPORT, parent_plan_sha256=file_hash(parent_path), created_at=datetime.now(timezone.utc).isoformat())
    save_json(path, plan)
    return plan


def pack_clouds(clouds, centers, scale, quantiles):
    cropped = []
    radialized = []
    center_rows = []
    local_features = []
    error = 0.
    for cloud, center in zip(clouds, centers, strict=True):
        _, rows = local_crop(cloud, scale)
        p = np.asarray(cloud[rows], dtype=np.float32)
        center_rows.append(int(np.flatnonzero(rows == center).item()))
        replacement, _, q, moments, e = radial_control(p, scale, quantiles)
        if len(replacement) != len(p):
            raise ValueError('Physical and model-coordinate support crops disagree')
        cropped.append(p)
        radialized.append(replacement)
        local_features.append(np.r_[q, moments])
        error = max(error, e)
    ptr = np.r_[0, np.cumsum([len(p) for p in cropped])]
    return dict(positions=np.concatenate(cropped), radial_positions=np.concatenate(radialized),
        offsets=ptr, center_indices=np.array(center_rows)), np.array(local_features), error


def verified_npz(path, receipt_path):
    rec = json.loads(receipt_path.read_text())
    if file_hash(path) != rec['sha256']:
        raise ValueError(f'Changed prior observations: {path}')
    return dict(np.load(path))


def prepare(config):
    plan = freeze(config)
    root = Path(config['output'])/'technical/inputs'
    for kind in ('temporal', 'spatial'):
        (root/kind).mkdir(parents=True, exist_ok=True)
    old = Path(config['previous_conditional'])
    audit = Path(config['spatial_audit'])
    frames = json.loads((audit/'technical/config.json').read_text())['spatial_frames']
    for source in plan['sources']:
        sid = source['id']
        dest = root/'temporal'/f'{sid}.npz'
        if not dest.with_suffix('.json').exists():
            prior = Path(config['parent_audit'])/'technical/sources'/str(sid)
            rec = json.loads((prior/'complete.json').read_text())
            for name, digest in rec['hashes'].items():
                if file_hash(prior/name) != digest:
                    raise ValueError(f'Changed original trajectory cache: {prior/name}')
            a = dict(np.load(prior/'observations.npz'))
            positions = np.load(prior/'positions.npy', mmap_mode='r')
            clouds = [positions[lo:hi] for lo, hi in zip(a['offsets'][:-1], a['offsets'][1:], strict=True)]
            packed, local, error = pack_clouds(clouds, a['center_indices'], plan['scale'], config['reference_quantiles'])
            nf, nc = len(a['frames']), len(a['centers'])
            cov = dict(frame=np.repeat(a['frames'], nc), atom=np.tile(a['centers'], nf),
                source=np.full(nf*nc, sid), temperature=np.full(nf*nc, source['temperature_K']),
                frames=a['frames'], centers=a['centers'], times_ps=a['times_ps'], labels=a['labels'], order=a['order'])
            for name in ('gatr', 'mace'):
                receipt = json.loads((prior/f'{name}-verification.json').read_text())
                if file_hash(prior/f'{name}.npy') != receipt['features_sha256']:
                    raise ValueError('Changed historical embedding')
                cov['old_'+name] = np.load(prior/f'{name}.npy')
            if source['split'] == 'test':
                p = old/'technical/sources'/str(sid)
                b = verified_npz(p/'observations.npz', p/'complete.json')
                for k in ('frames', 'centers', 'times_ps', 'order', 'labels'):
                    np.testing.assert_array_equal(b[k], a[k])
                cov.update({k: b[k] for k in ('radial', 'radii80', 'radial_quantiles', 'context', 'soap', 'tda', 'bond', 'angular')})
                cov['radial'] = np.concatenate((cov['radial'], local), axis=1)
            np.savez(dest, **packed, **cov)
            save_json(dest.with_suffix('.json'), dict(source=sid, split=source['split'], rows=nf*nc,
                sha256=file_hash(dest), parent_sha256=rec['hashes']['observations.npz'], radius_preservation_max_A=error))
            print(f'Prepared temporal {sid}: {nf*nc} observations', flush=True)
        if source['split'] != 'test':
            continue
        missing = [frame for frame in frames if not (root/'spatial'/f'{sid}-{frame:04d}.json').exists()]
        if not missing:
            continue
        raw = source_arrays(dict(source, kind='dynamic'))
        for frame in missing:
            dest = root/'spatial'/f'{sid}-{frame:04d}.npz'
            panel = audit/'technical/spatial'/dest.name
            prior = verified_npz(panel, panel.with_suffix('.json'))
            p = old/'technical/spatial'/dest.name
            cov = verified_npz(p, p.with_suffix('.json'))
            points, tree, box = chart(raw, frame, False)
            rows = np.searchsorted(raw['atom_ids'], prior['atom_ids'])
            np.testing.assert_array_equal(raw['atom_ids'][rows], prior['atom_ids'])
            np.testing.assert_array_equal(points[rows], prior['positions'])
            np.testing.assert_array_equal(cov['atom'], prior['atom_ids'])
            neighbors = tree.query_ball_point(points[rows], OUTER_RADIUS*plan['scale']/REFERENCE_RADIUS, return_sorted=True)
            clouds = [offsets(points, c, np.asarray(ids), box) for c, ids in zip(rows, neighbors, strict=True)]
            centers = [int(np.flatnonzero(np.asarray(ids) == c).item()) for c, ids in zip(rows, neighbors, strict=True)]
            packed, local, error = pack_clouds(clouds, centers, plan['scale'], config['reference_quantiles'])
            cov['old_gatr'] = cov.pop('gatr')
            del cov['radial_gatr']
            cov['radial'] = np.concatenate((cov['radial'], local), axis=1)
            np.savez(dest, **packed, **cov)
            save_json(dest.with_suffix('.json'), dict(source=sid, frame=frame, rows=len(rows),
                sha256=file_hash(dest), prior_conditional_sha256=file_hash(p), radius_preservation_max_A=error))
            print(f'Prepared spatial {sid}/{frame}: {len(rows)} observations', flush=True)


def load(config, kind='temporal', split='test'):
    root = Path(config['output'])/'technical'
    plan = json.loads((root/'plan.json').read_text())
    selected = [s for s in plan['sources'] if s['split'] == split]
    parts = []
    for source in selected:
        paths = [root/'inputs/temporal'/f'{source["id"]}.npz'] if kind == 'temporal' else sorted((root/'inputs/spatial').glob(f'{source["id"]}-*.npz'))
        if kind == 'spatial' and len(paths) != 7:
            raise ValueError('Incomplete spatial source')
        for path in paths:
            a = verified_npz(path, path.with_suffix('.json'))
            for key in ('positions', 'radial_positions', 'offsets', 'center_indices'):
                del a[key]
            for model in config['checkpoints']:
                p = root/'features'/model/kind/path.name
                b = verified_npz(p, p.with_suffix('.json'))
                if len(b['original']) != len(a['source']):
                    raise ValueError('Embedding row count differs')
                a[model] = b['original']
                a['radial_'+model] = b['radial']
            if kind == 'temporal':
                nf, nc = len(a['frames']), len(a['centers'])
                if split == 'test':
                    risk, future, onset = future_labels(a['labels'].reshape(nf, nc), plan['cadence_ps'], config)
                    a.update(future_eligible=risk.ravel(), future=future.reshape(nf*nc, -1), onset_frame=np.tile(onset, nf))
                for key in ('frames', 'centers', 'times_ps'):
                    del a[key]
            parts.append(a)
    return {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}, selected, plan
