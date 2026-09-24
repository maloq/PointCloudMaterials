"""Run existing native snapshot producers on the frozen dense trajectory assay."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.encoder_screen.common import checked, sha, write


def pack_observations(positions, ids, observations, radius=10.):
    """Preserve the stored frame-major row order, center and nearest-80 identity."""
    patches, centers, nearest = [], [], []
    for row, (a, b, center) in enumerate(zip(observations['offsets'][:-1],
            observations['offsets'][1:], observations['center_indices'], strict=True)):
        x, atom = positions[a:b], ids[a:b]
        if not np.array_equal(x[center], np.zeros(3)):
            raise ValueError(f'Dense cache center is not at origin: row {row}')
        distance2 = np.square(x.astype(np.float64)).sum(-1)
        near = np.lexsort((atom, distance2))[:80]
        if len(near) != 80 or distance2[near[-1]] >= radius**2:
            raise ValueError(f'Incomplete nearest-80 context: row {row}')
        np.testing.assert_array_equal(atom[near], observations['nearest_ids'][row])
        keep = distance2 < radius**2
        centers.append(int(np.count_nonzero(keep[:center])))
        patches.append(x[keep]); nearest.append(x[near])
    return dict(positions=np.concatenate(patches), offsets=np.r_[0, np.cumsum([len(x) for x in patches])],
                centers=np.asarray(centers), nearest80=np.stack(nearest))


def prepare(config):
    root = resolve_path(config['inference_output'])
    source_root = resolve_path(config['dense']['root'])
    plan = json.loads((source_root/'technical/plan.json').read_text())
    inputs = root/'technical/inputs'
    plan_hash = sha(source_root/'technical/plan.json')
    manifest_path = inputs/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_plan_sha256'] != plan_hash:
            raise ValueError('Dense observation plan changed')
        for file, expected in manifest['files'].items():
            checked(inputs/file, expected)
        return inputs
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_root=str(source_root), source_plan_sha256=plan_hash,
                    observation='original observed MD, not relaxed', frames=[], files={})
    for i, source in enumerate(plan['sources']):
        folder = source_root/'technical/sources'/str(source['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        for name in ('positions.npy', 'atom_ids.npy', 'observations.npz'):
            checked(folder/name, receipt['hashes'][name])
        with np.load(folder/'observations.npz') as obs:
            np.testing.assert_allclose(obs['times_ps'], obs['frames']*.75, rtol=0, atol=1e-8)
            if source['split'] == 'test':
                np.testing.assert_array_equal(obs['frames'], np.arange(source['frame_count']))
            arrays = pack_observations(np.load(folder/'positions.npy', mmap_mode='r'),
                                      np.load(folder/'atom_ids.npy', mmap_mode='r'), obs)
            np.savez(inputs/f'frame-{i:02d}.npz', **arrays)
            manifest['frames'].append(dict(frame_index=i, material='Al', source=source['id'],
                split=source['split'], count=len(obs['center_indices']),
                source_inputs=receipt['hashes'], observations_sha256=receipt['hashes']['observations.npz']))
        manifest['files'][f'frame-{i:02d}.npz'] = sha(inputs/f'frame-{i:02d}.npz')
        print('prepared dense source', source['id'], flush=True)
    write(manifest_path, manifest)
    return inputs


def run(config, *, smoke=False):
    inputs = prepare(config)
    root = resolve_path(config['inference_output'])
    driver = Path(__file__).resolve().parents[1]/'encoder_screen/native.py'
    for entry in config['dense']['extra_exports']:
        folder = resolve_path(entry['root'])
        original = resolve_path(entry['source_evaluation'])/'complete.json'
        original_receipt = json.loads(original.read_text())
        task = original_receipt['task']
        if task['kind'] not in ('geoframe', 'geometry'):
            raise ValueError(f'Unplanned dense export kind: {task["kind"]}')
        task = dict(task, inputs=str(inputs.resolve()), destination=str((folder/'embeddings').resolve()),
                    materials=['Al'], inference_driver_sha256=sha(driver), source_receipt_sha256=sha(original))
        record = folder/'task.json'
        if record.exists() and json.loads(record.read_text()) != task:
            raise ValueError(f'Native extraction identity changed: {record}')
        write(record, task)
        done = folder/'complete.json'
        if done.exists():
            completed = json.loads(done.read_text())
            if completed['task_sha256'] != sha(record):
                raise ValueError(f'Completed dense task changed: {folder}')
            for name, expected in completed['feature_files'].items():
                checked(folder/'embeddings'/name, expected)
            continue
        command = [sys.executable, '-u', str(driver), '--record', str(record.resolve()), '--static-only']
        if smoke:
            command.append('--smoke')
        with (folder/'inference.log').open('w') as log:
            subprocess.run(command, cwd=task['producer'], stdout=log, stderr=subprocess.STDOUT, check=True)
        extraction = json.loads((folder/'embeddings/extraction.json').read_text())
        if extraction['smoke']:
            print('smoke complete', task['name'], flush=True)
            continue
        feature_files = {p.name: sha(p) for p in (folder/'embeddings').glob('frame-*.npz')}
        if len(feature_files) != len(json.loads((inputs/'manifest.json').read_text())['frames']):
            raise ValueError(f'Incomplete dense source coverage: {folder}')
        write(done, dict(state='complete', task_sha256=sha(record), task=task,
            source_plan_sha256=json.loads((inputs/'manifest.json').read_text())['source_plan_sha256'],
            extraction=extraction, feature_files=feature_files))
        print('completed dense checkpoint', task['name'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), smoke=args.smoke)
