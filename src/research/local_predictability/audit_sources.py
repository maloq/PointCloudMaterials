"""Verify the complete inherited raw cohort, without selecting on outcomes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from src.data.trajectories.shooting import ShootingBinaryTrajectory, _array_sha256
from src.project_runtime.paths import dataset_path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(chunk)
    return value.hexdigest()


def audit_one(source):
    start = time.monotonic()
    root = dataset_path(source['dataset']) / source['relative_trajectory_path']
    result = dict(id=source['id'], lineage=source['lineage'], split=source['split'],
                  dataset=source['dataset'], relative_trajectory_path=source['relative_trajectory_path'])
    try:
        if sha256(root / 'manifest.json') != source['manifest_sha256']:
            raise ValueError(f'Manifest hash mismatch: {root}')
        raw = ShootingBinaryTrajectory.load(root)
        if raw.frame_count != 801 or raw.atom_count != 70304:
            raise ValueError(f'Wrong source shape: {root}')
        if not np.array_equal(raw.timesteps * source['timestep_fs'] / 1000, np.arange(801) * .75):
            raise ValueError(f'Wrong timeline: {root}')
        if raw.manifest['velocity_units'] != 'angstrom_per_ps' or not np.all(raw.atom_types == 1):
            raise ValueError(f'Wrong velocity units or species: {root}')
        hashes = {}
        for name, description in raw.manifest['arrays'].items():
            values = getattr(raw, name)
            observed = _array_sha256(values)
            if observed != description['sha256']:
                raise ValueError(f'Array checksum mismatch: {root / description["file"]}')
            hashes[name] = observed
        result.update(status='passed', manifest_sha256=source['manifest_sha256'], arrays=hashes)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    result['seconds'] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sources = manifest['sources']
    if len(sources) != 150 or len({s['lineage'] for s in sources}) != 150:
        raise ValueError('Require exactly 150 unique source lineages')
    if Counter(s['split'] for s in sources) != dict(train=90, val=30, test=30):
        raise ValueError('Require inherited 90/30/30 folds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(audit_one, sources):
            results.append(result)
            print(json.dumps(dict(id=result['id'], status=result['status'], seconds=result['seconds'],
                                  error=result.get('error'))), flush=True)
    report = dict(protocol=manifest['protocol'], manifest_sha256=sha256(args.manifest),
                  status='passed' if all(s['status'] == 'passed' for s in results) else 'failed',
                  counts=dict(Counter(s['status'] for s in results)), sources=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
