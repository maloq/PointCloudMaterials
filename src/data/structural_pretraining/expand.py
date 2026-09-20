"""Immutable expansion of an existing structural release with complete hot TDA."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from src.project_runtime.paths import resolve_path
from .prepare import digest, file_hash, save_json, prepare_task, finalize


def shooting_tasks(parent, extra, seed, rows_per_task):
    """Balance new draws across training lineages and cover every eligible branch."""
    return dynamic_tasks(parent, {'al_shooting': extra}, seed, rows_per_task)


def dynamic_tasks(parent, counts, seed, rows_per_task):
    """Add distinct source/frame anchors, balanced within each dynamic stratum."""
    tasks = []
    for stratum, extra in sorted(counts.items()):
        tasks.extend(stratum_tasks(parent, stratum, extra, seed, rows_per_task))
    return tasks


def stratum_tasks(parent, stratum, extra, seed, rows_per_task):
    roots = defaultdict(list)
    for index, source in enumerate(parent['sources']):
        if (source['split'] == 'train' and source['stratum'] == stratum
                and source['kind'] == 'dynamic'):
            roots[source['lineage']].append(index)
    if not roots or extra < sum(map(len, roots.values())):
        raise ValueError(f'Expansion must cover every eligible dynamic source in {stratum}')
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(digest(stratum)[:8], 16)]))
    occupied = {(t['source'], t['frame']) for t in parent['tasks']}
    tasks = []
    for r, lineage in enumerate(sorted(roots)):
        indices = sorted(roots[lineage])
        quota = extra // len(roots) + (r < extra % len(roots))
        if quota < len(indices):
            raise ValueError(f'Too few new anchors for lineage {lineage}')
        available = {i: [k for k in range(2, parent['sources'][i]['frame_count']-1)
                         if (i, k) not in occupied] for i in indices}
        capacity = np.array([len(available[i])*rows_per_task for i in indices])
        if np.any(capacity == 0) or capacity.sum() < quota:
            raise ValueError(f'Insufficient unused frames for lineage {lineage}: {capacity.tolist()}, requested {quota}')
        # Equal source shares until a short trajectory exhausts its unused frames.
        # Redistribute the remainder within the same lineage, never into held-out data.
        counts = np.zeros(len(indices), dtype=int)
        while counts.sum() < quota:
            active = np.flatnonzero(counts < capacity)
            remaining = quota-int(counts.sum())
            shares = remaining//len(active)+(np.arange(len(active)) < remaining % len(active))
            counts[active] += np.minimum(shares, capacity[active]-counts[active])
        for index, count in zip(indices, counts.tolist(), strict=True):
            source = parent['sources'][index]
            frames = available[index]
            rng.shuffle(frames)
            if len(frames) < (count + rows_per_task-1)//rows_per_task:
                raise ValueError(f'Not enough unused {stratum} frames: {source["id"]}')
            while count:
                n = min(count, rows_per_task); frame = frames.pop()
                tasks.append(dict(source=index, frame=frame, count=n,
                    seed=int(rng.integers(2**31)), split='train'))
                count -= n
    return tasks


def build_plan(config):
    parent_root = resolve_path(config['parent_release'])
    parent = json.loads((parent_root/'plan.json').read_text())
    manifest = json.loads((parent_root/'manifest.json').read_text())
    if manifest['state'] != 'complete' or manifest['identity'] != parent['identity']:
        raise ValueError('Expansion requires a completed, matching parent release')
    # Targets must retain the exact original geometric descriptor definition.
    for path in ('src/data/predictive_memory/targets.py', 'src/analysis/liquid_structure.py'):
        if file_hash(path) != parent['producer_hashes'][path]:
            raise ValueError(f'Parent target producer differs: {path}')
    tasks = copy.deepcopy(parent['tasks'])
    if 'additional_dynamic_anchors' in config:
        tasks += dynamic_tasks(parent, config['additional_dynamic_anchors'], config['seed'], config['rows_per_task'])
    else:
        tasks += shooting_tasks(parent, config['additional_shooting_anchors'], config['seed'], config['rows_per_task'])
    for i, task in enumerate(tasks):
        task['id'] = f'{i:06d}'
    plan = dict(protocol='structural_neighbors_full_tda_v1', config=config,
        sources=parent['sources'], tasks=tasks, scales=parent['scales'],
        calibration=parent['calibration'], reference_radius=parent['reference_radius'],
        source_evidence=parent['source_evidence'], parent_identity=parent['identity'],
        parent_manifest_sha256=file_hash(parent_root/'manifest.json'),
        parent_plan_sha256=file_hash(parent_root/'plan.json'), parent_shards=len(parent['tasks']),
        producer_hashes={p:file_hash(p) for p in ('src/data/structural_pretraining/expand.py',
            'src/data/structural_pretraining/prepare.py', 'src/data/predictive_memory/targets.py',
            'src/analysis/liquid_structure.py')})
    plan['identity'] = digest(plan)
    return plan


def fill_tda(folder, record):
    """Fill only supervised endpoints, using the producer's physical nearest-80 rule."""
    from src.analysis.liquid_structure import persistence_image
    folder = Path(folder)
    arrays = {name:np.load(folder/f'{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in ('positions', 'atom_ids', 'offsets', 'views', 'tda', 'tda_valid')}
    slots = [2, 4] if record['static'] else [2, 3, 4]
    endpoints = arrays['views'][:, slots].ravel()
    if np.any(endpoints < 0):raise ValueError(f'Missing endpoint in {folder}')
    targets = arrays['tda'].copy(); valid = arrays['tda_valid'].copy()
    if targets.shape != (len(valid), 144):raise ValueError(f'Unexpected TDA schema in {folder}')
    for view in endpoints:
        if valid[view]:continue
        lo, hi = arrays['offsets'][view:view+2]
        x = arrays['positions'][lo:hi]
        squared = np.square(x.astype(np.float64)).sum(-1)
        nearest = np.lexsort((arrays['atom_ids'][lo:hi], squared))[:80]
        if len(nearest) != 80:raise ValueError(f'Incomplete target neighborhood in {folder}, view {view}')
        targets[view] = persistence_image(x[nearest]);valid[view] = True
    if not np.isfinite(targets[endpoints]).all():raise FloatingPointError(f'Nonfinite TDA in {folder}')
    for name, value in [('tda', targets), ('tda_valid', valid)]:
        temporary = folder/f'{name}.building.npy'
        np.save(temporary, value, allow_pickle=False);temporary.replace(folder/f'{name}.npy')
        record['hashes'][name] = file_hash(folder/f'{name}.npy')
    record['labelled_views'] = int(valid.sum())
    record['tda_coverage'] = 'all_supervised_views'
    return record


def expand_task(arguments):
    root, parent_root, source, task, scale, identity, inherited = arguments
    folder = Path(root)/'shards'/task['id']; receipt = folder/'complete.json'
    started = time.monotonic()
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record['identity'] != identity:raise ValueError(f'Changed shard identity: {folder}')
        if record.get('tda_coverage') == 'all_supervised_views':return record
    elif inherited:
        old = Path(parent_root)/'shards'/task['id']
        record = json.loads((old/'complete.json').read_text())
        if record['task'] != task:raise ValueError(f'Inherited task changed: {task["id"]}')
        folder.mkdir(parents=True, exist_ok=True)
        for name in record['hashes']:
            destination = folder/f'{name}.npy'
            if not destination.exists():os.link(old/f'{name}.npy', destination)
        record.update(identity=identity)
    else:
        record = prepare_task((root, source, task, scale, identity))
    record = fill_tda(folder, record)
    record['full_tda_seconds'] = time.monotonic()-started
    save_json(receipt, record)
    return record


def run(config, workers):
    root = resolve_path(config['release']);root.mkdir(parents=True, exist_ok=True)
    with (root/'expansion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan(config)
        path = root/'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Existing expansion has a different identity')
        save_json(path, plan)
        arguments = [(str(root), str(resolve_path(config['parent_release'])), plan['sources'][t['source']],
            t, plan['scales'][plan['sources'][t['source']]['material']], plan['identity'],
            i < plan['parent_shards']) for i,t in enumerate(plan['tasks'])]
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(expand_task, args) for args in arguments]
                for count, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    status = dict(state='preparing', completed_shards=count, total_shards=len(arguments),
                                  last_shard=row['task']['id'])
                    save_json(root/'status.json', status)
                    if count % 20 == 0 or count == len(arguments):print(json.dumps(status), flush=True)
            manifest = finalize(root, plan)
            manifest['tda_coverage'] = 'all_supervised_views'
            save_json(root/'manifest.json', manifest)
            save_json(root/'status.json', dict(state='complete', shards=len(manifest['shards'])))
        except Exception as error:
            save_json(root/'status.json', dict(state='failed', error=repr(error)))
            raise
