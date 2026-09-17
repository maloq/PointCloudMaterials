"""Freeze source identities, ancestry and outcome-independent sampling before work."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from src.project_runtime.paths import dataset_path, load_json, portable_config, resolve_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.simulation.relaxation import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def spaced_frames(timesteps, timestep_fs, start_ps, stop_ps, spacing_ps):
    """Select observed times on the declared physical-time grid, never by outcome.

    Nonuniform first-passage trajectories may end before the next grid point.
    Do not add their event-conditioned final frame to the training grid.
    """
    time = (np.asarray(timesteps, dtype=np.int64) - timesteps[0]) * timestep_fs / 1000.
    if spacing_ps <= 0 or np.any(np.diff(time) <= 0):
        raise ValueError('Expected positive spacing and strictly increasing trajectory times')
    grid = np.arange(start_ps, min(stop_ps, time[-1]) + 1e-8, spacing_ps)
    index = np.searchsorted(time, grid)
    if np.any(index >= len(time)) or not np.allclose(time[index], grid, rtol=0, atol=1e-7):
        raise ValueError(f'Trajectory does not contain the requested {spacing_ps} ps grid')
    return index.tolist()


def coarse_first(frames):
    """Midpoint subdivision: cover the full interval before densifying it."""
    if not frames:
        return []
    result, pending = [frames[0]], [(1, len(frames))]
    while pending:
        lo, hi = pending.pop(0)
        if lo < hi:
            middle = (lo + hi) // 2
            result.append(frames[middle])
            pending.extend(((lo, middle), (middle + 1, hi)))
    return result


def ancestry(source_run_id):
    return re.sub(r'^source_group_\d+/', '', source_run_id)


@lru_cache(None)
def campaign_checksum(root):
    return sha256(root/'manifest.json')


@lru_cache(None)
def potential_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if 'potential' in manifest:
        return manifest['potential']
    # The fixed-24 ps producer records the exact source manifest it extends.
    parent = resolve_path(manifest['source_campaign_root']) / 'manifest.json'
    if sha256(parent) != manifest['source_campaign_manifest_sha256']:
        raise ValueError(f'Changed fixed-24 ps parent manifest: {parent}')
    return json.loads(parent.read_text())['potential']


def prepare(config_path):
    cfg = load_json(config_path)
    output = Path(cfg['output']) / 'technical'
    signature = digest(portable_config(cfg))
    destination = output / 'plan.json'
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan['config_signature'] != signature:
            raise ValueError(f'Configuration differs from immutable plan: {destination}')
        return plan
    expected = cfg['potential_sha256']
    sources, excluded, seen = [], [], {}

    def add(record, root, timestep_fs, sampling, center_ids=None):
        potential = potential_manifest(root)
        if any(potential[k] != v for k, v in expected.items()):
            raise ValueError(f'Generating potential differs from target potential: {root}')
        trajectory = ShootingBinaryTrajectory.load(record['trajectory'])
        if trajectory.atom_count != 70304 or np.any(trajectory.atom_types != 1):
            raise ValueError(f'Expected the 70,304-atom pure Al producer: {trajectory.root}')
        # Only exact whole-trajectory duplicates are removed here. Derived float16
        # continuations retain their own input precision and shared ancestry.
        identity = digest({k: trajectory.manifest['arrays'][k]['sha256']
                           for k in ('positions', 'box_low', 'box_high', 'timesteps', 'atom_ids')})
        if identity in seen:
            excluded.append(dict(dataset=record['dataset'], trajectory=record['trajectory'],
                                 reason='exact_duplicate_positions_timeline', duplicate_of=seen[identity]))
            return
        seen[identity] = record['id']
        if center_ids is None:
            key = digest([cfg['seed'], record['lineage'], record['parent_id']])
            rng = np.random.default_rng(int(key[:16], 16))
            center_ids = sorted(rng.choice(trajectory.atom_ids, cfg['centers_per_frame'], replace=False).tolist())
        frames = spaced_frames(trajectory.timesteps, timestep_fs, **sampling)
        if not frames:
            raise ValueError(f'No selected frames in {trajectory.root}')
        record.update(trajectory=str(trajectory.root), manifest_sha256=sha256(trajectory.root/'manifest.json'),
            campaign_manifest=str(root/'manifest.json'), campaign_manifest_sha256=campaign_checksum(root),
            center_atom_ids=center_ids, frames=coarse_first(frames), timestep_fs=timestep_fs,
            frame_count=trajectory.frame_count, input_dtype=trajectory.positions.dtype.name)
        sources.append(record)

    # This is producer metadata, not a configuration tree: source IDs and
    # lineage strings must not be interpreted as machine path aliases.
    cohort = json.loads(Path(cfg['cohort']).read_text())
    for row in cohort['sources']:
        if row['split'] != 'train':
            continue
        root = dataset_path(row['dataset'])
        path = root / row['relative_trajectory_path']
        if sha256(path/'manifest.json') != row['manifest_sha256']:
            raise ValueError(f'Native cohort trajectory changed: {path}')
        add(dict(id=f"native_{row['id']}", dataset=row['dataset'], family='independent_train',
            trajectory=str(path), lineage=row['lineage'], parent_id=None,
            declared_split='train', temperature_K=row['temperature_K'], smoke=False),
            root, row['timestep_fs'], cfg['native_sampling'], row['pool_atom_ids'])
    for collection in cfg['shooting_collections']:
        root = dataset_path(collection['dataset'])
        manifest = json.loads((root/'manifest.json').read_text())
        for branch in manifest['branches']:
            directory = root/branch['branch_dir']
            outcome_path = directory/'outcome.json'
            if not outcome_path.exists():
                excluded.append(dict(dataset=collection['dataset'], branch=branch['branch_id'], reason='missing_outcome'))
                continue
            outcome = json.loads(outcome_path.read_text())
            if outcome['state'] != 'complete':
                excluded.append(dict(dataset=collection['dataset'], branch=branch['branch_id'],
                                     reason='incomplete', state=outcome['state']))
                continue
            path = Path(outcome['trajectory_artifact']['path'])
            path = resolve_path(path) if path.is_absolute() else directory/path
            add(dict(id=digest([collection['dataset'], branch['branch_id']])[:24],
                dataset=collection['dataset'], family=collection['family'], trajectory=str(path),
                lineage=ancestry(branch['source_run_id']), parent_id=branch['parent_id'],
                declared_split=branch['source_split'], temperature_K=branch['temperature_K'],
                smoke=collection['smoke'], outcome_sha256=sha256(outcome_path)),
                root, collection['timestep_fs'], cfg['shooting_sampling'])
    lineage_splits = defaultdict(set)
    for source in sources:
        lineage_splits[source['lineage']].add(source['declared_split'])
    conflicts = {}
    train_labels = {'train', 'optimization'}
    for source in sources:
        labels = lineage_splits[source['lineage']]
        # Never promote a trajectory with any declared held-out relative to train.
        source['training_eligible'] = labels <= train_labels and not source['smoke']
        source['split'] = 'train' if source['training_eligible'] else 'heldout_or_diagnostic'
        if labels & train_labels and labels - train_labels:
            conflicts[source['lineage']] = sorted(labels)
    # Alternate families at each temporal level so a long independent trajectory
    # cannot starve the short shooting collections during a bounded allocation.
    groups = defaultdict(list)
    for index, source in enumerate(sources):
        groups[source['family']].append(index)
    order = [values[i] for i in range(max(map(len, groups.values())))
             for values in groups.values() if i < len(values)]
    tasks = [dict(source_index=i, frame=sources[i]['frames'][level])
             for level in range(max(len(s['frames']) for s in sources))
             for i in order if level < len(sources[i]['frames'])]
    for index, task in enumerate(tasks):
        task['id'] = f'{index:07d}'
    plan = dict(schema_version=1, config_signature=signature, config=portable_config(cfg),
        cohort_sha256=sha256(cfg['cohort']), sources=sources, tasks=tasks, exclusions=excluded,
        split_conflicts_held_out=conflicts, explicit_exclusions=cfg['excluded_collections'],
        counts=dict(trajectories=len(sources), frames=len(tasks),
            target_windows=sum(len(s['frames'])*len(s['center_atom_ids']) for s in sources),
            training_windows=sum(len(s['frames'])*len(s['center_atom_ids']) for s in sources if s['training_eligible']),
            trajectories_by_family=dict(Counter(s['family'] for s in sources))))
    save(destination, plan)
    save(output/'inventory.json', {k:v for k,v in plan.items() if k not in ('tasks','sources','config')})
    return plan
