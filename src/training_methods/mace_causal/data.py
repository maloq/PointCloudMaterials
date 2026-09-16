"""Verified sequence-cache adapter; fixed targets never enter AtomicHistory."""
from pathlib import Path
import hashlib

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data_utils.causal_history import build_history, physical_windows
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import resolve_path
from src.research.mace_velocity.inventory import read
from src.research.mace_velocity.sequence_data import read_sequence
from .objective import hazard_label


PROTOCOL = 'mace_causal_physical_state_v1'


def preparation_identity():
    paths = [Path(__file__), Path('src/data_utils/causal_history.py'),
             Path('src/training_methods/mace_causal/objective.py'),
             Path('src/research/mace_velocity/data.py'), Path('src/research/mace_local_state/physics.py'),
             Path('src/analysis/liquid_structure.py'), Path('src/models/encoders/mace_context.py')]
    return {str(p.resolve().relative_to(Path.cwd())): sha256(p) for p in paths}


def extended_record(record, trajectory, additional_followup_ps):
    """Extend the retained label segment by an exact physical duration."""
    if additional_followup_ps < 0:
        raise ValueError('Additional follow-up duration must be nonnegative')
    times = np.asarray(trajectory.timesteps, dtype=float)*record['source']['timestep_fs']/1000
    last = record['frames'][-1]
    wanted = times[last]+additional_followup_ps
    found = np.flatnonzero(np.abs(times-wanted) < 1e-8)
    if len(found) != 1:
        raise ValueError(f'Source {record["source"]["id"]} lacks exact follow-up at {wanted:g} ps; '
                         f'last stored time={times[-1]:g} ps')
    return dict(record, frames=[*record['frames'], *range(last+1, int(found[0])+1)])


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_splits(records):
    lineages, identifiers = {}, set()
    for record in records:
        source = record['source']
        if source['id'] in identifiers:
            raise ValueError(f'Duplicate source ID: {source["id"]}')
        identifiers.add(source['id'])
        split = source['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unsupported source split: {split}')
        old = lineages.setdefault(source['lineage'], split)
        if split != old:
            raise ValueError(f'Preparation lineage leaks across splits: {source["lineage"]}')
    if set(lineages.values()) != {'train', 'val', 'test'}:
        raise ValueError('Require whole-source train, validation and test populations')
    if len(lineages) != len(records):
        raise ValueError('Source bootstrap requires distinct independent preparation lineages')


def prepare(config):
    if config['protocol'] != PROTOCOL:
        raise ValueError('Wrong causal-state preparation protocol')
    cache = Path(config['cache'])
    if cache.exists():
        raise FileExistsError(f'Preserve existing cache; select a fresh output: {cache}')
    previous = Path(config['sequence_cache'])
    plan = read(previous/'plan.json')
    records = [r for r in plan['records'] if r['source']['lineage'].startswith('independent_melt')]
    if config.get('source_ids') is not None:
        requested = set(config['source_ids'])
        records = [r for r in records if r['source']['id'] in requested]
        if {r['source']['id'] for r in records} != requested:
            raise ValueError('Requested source IDs are not retained independent preparations')
    validate_splits(records)
    # This adapter deliberately supports the actual verified elemental Al producer.
    if any(r['source']['format'] != 'shooting_binary' for r in records):
        raise ValueError('Causal preparation requires the verified shooting-binary ID/species producer')
    cache.mkdir(parents=True)
    write_json(cache/'plan.json', dict(protocol=PROTOCOL, config=config, records=records,
                                     sequence_plan_sha256=sha256(previous/'plan.json'),
                                     implementation=preparation_identity()))
    units = []
    architecture = config['encoder']
    for record in records:
        source = record['source']
        sid = source['id']
        label_path = previous/f'source-{sid:04d}.npz'
        stamp = read(label_path.with_suffix('.json'))
        if stamp['record'] != record or sha256(label_path) != stamp['sha256']:
            raise ValueError(f'Changed sequence label provenance: {label_path}')
        x, v, box, times, identity = read_sequence(record, config)
        if identity != stamp['source_identity']:
            raise ValueError(f'Observed arrays differ from physical target producer: source {sid}')
        trajectory = ShootingBinaryTrajectory.load(resolve_path(source['path']))
        if not np.array_equal(np.unique(trajectory.atom_types), [1]):
            raise ValueError(f'Expected elemental Al type 1 in source {sid}')
        atom_ids = np.asarray(trajectory.atom_ids)
        species = np.full(len(atom_ids), 13, dtype=np.int64)
        centers = np.searchsorted(atom_ids, record['center_atom_ids'])
        np.testing.assert_array_equal(atom_ids[centers], record['center_atom_ids'])
        with np.load(label_path) as a:
            y = a['raw_target'].copy()
            np.testing.assert_array_equal(a['center_atom_id'], record['center_atom_ids'])
            np.testing.assert_allclose(a['time_ps'], np.tile(times, (len(centers), 1)), rtol=0, atol=1e-8)
        if y.shape != (len(centers), len(times), 169) or not np.isfinite(y).all():
            raise ValueError(f'Invalid fixed physical targets for source {sid}: {y.shape}')
        retained_identity = identity
        effective_record = extended_record(record, trajectory, config['additional_followup_ps'])
        extra = len(effective_record['frames'])-len(times)
        if extra:
            from src.research.mace_velocity.data import labels, local_clouds
            old_frames = len(times)
            xx, vv, bb, tt, identity = read_sequence(effective_record, config)
            for original, expanded in ((x, xx), (v, vv), (box, bb), (times, tt)):
                np.testing.assert_array_equal(original, expanded[:old_frames])
            new_targets = np.stack([np.stack([labels(cloud) for cloud in local_clouds(
                xx[k], vv[k], bb[k], centers, 18.)]) for k in range(old_frames, len(tt))], axis=1)
            y = np.concatenate((y, new_targets), axis=1)
            x, v, box, times = xx, vv, bb, tt
        windows = physical_windows(times, config['history_offsets_ps'], config['future_lags_ps'])
        crystal = None
        if config['events']['enabled']:
            from src.research.smooth_temporal_encoder.prepare import ptm_labels
            crystal = np.empty((len(centers), len(times)), dtype=bool)
            for k in range(len(times)):
                points = np.mod(x[k].astype(float), box[k])
                _, neighbors = cKDTree(points, boxsize=box[k]).query(points[centers], k=80)
                np.testing.assert_array_equal(neighbors[:, 0], centers)
                offsets = points[neighbors[:, 1:]]-points[centers, None]
                offsets -= box[k]*np.round(offsets/box[k])
                crystal[:, k] = np.isin(ptm_labels(offsets/10., config['events']['ptm_rmsd_cutoff']), [1, 2, 3])
        samples = []
        for c, center in enumerate(record['center_atom_ids']):
            for anchor, observed, future in windows:
                graph = build_history(x[observed], v[observed], box[observed], times[observed],
                    atom_ids, species, center, cutoff_A=architecture['cutoff_A'],
                    context_radius_A=max(s[1] for s in architecture['scales_A']),
                    spatial_layers=architecture['num_layers'])
                event = (hazard_label(crystal[c], times, anchor, config['events']['bin_edges_ps'],
                                     config['events']['persistence_frames']) if crystal is not None
                         else dict(event_bin=-1, observed_bins=0, at_risk=False))
                path = np.array([[y[c, anchor+1:f+1, 4].mean(), y[c, anchor+1:f+1, 4].std()]
                                 for f in future], dtype=np.float32)
                samples.append(dict(history=graph, present=y[c, anchor], future=y[c, future], path=path,
                    source_id=sid, lineage=source['lineage'], split=source['split'], center_atom_id=center,
                    anchor_ps=float(times[anchor]), temperature_K=source['temperature_K'], **event))
        destination = cache/f'source-{sid:04d}.pt'
        atomic_save(destination, samples)
        units.append(dict(source_id=sid, path=destination.name, sha256=sha256(destination),
                          samples=len(samples), label_sha256=stamp['sha256'], source_identity=identity,
                          retained_source_identity=retained_identity, frames=effective_record['frames'],
                          physical_target_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
        print(f'CAUSAL PREPARE source={sid} windows={len(samples)} ({len(units)}/{len(records)})', flush=True)
    write_json(cache/'complete.json', dict(protocol=PROTOCOL, plan_sha256=sha256(cache/'plan.json'), units=units))


def load(config):
    cache = Path(config['cache'])
    plan, complete = read(cache/'plan.json'), read(cache/'complete.json')
    if complete['protocol'] != PROTOCOL or complete['plan_sha256'] != sha256(cache/'plan.json'):
        raise ValueError('Changed/incomplete causal cache')
    if plan['implementation'] != preparation_identity():
        raise ValueError('Causal graph/label preparation implementation changed; build a fresh verified cache')
    validate_splits(plan['records'])
    for key in ('history_offsets_ps', 'future_lags_ps', 'source_ids', 'additional_followup_ps'):
        if config[key] != plan['config'][key]:
            raise ValueError(f'Cache and training differ: {key}')
    for key in ('enabled', 'bin_edges_ps', 'persistence_frames', 'ptm_rmsd_cutoff'):
        if config['events'][key] != plan['config']['events'][key]:
            raise ValueError(f'Cache and event labels differ: {key}')
    for key in ('cutoff_A', 'num_layers', 'scales_A'):
        if config['encoder'][key] != plan['config']['encoder'][key]:
            raise ValueError(f'Cache geometry and encoder differ: {key}')
    samples = []
    sources = {r['source']['id']: r['source'] for r in plan['records']}
    if len(complete['units']) != len(sources) or {u['source_id'] for u in complete['units']} != set(sources):
        raise ValueError('Cache units do not match the declared source population')
    for unit in complete['units']:
        path = cache/unit['path']
        if sha256(path) != unit['sha256']:
            raise ValueError(f'Changed causal cache unit: {path}')
        unit_samples = torch.load(path, map_location='cpu', weights_only=False)
        source = sources[unit['source_id']]
        if len(unit_samples) != unit['samples'] or any(s['source_id'] != source['id']
                or s['split'] != source['split'] or s['lineage'] != source['lineage'] for s in unit_samples):
            raise ValueError(f'Cache sample/source metadata mismatch: {path}')
        samples.extend(unit_samples)
    return samples, dict(cache_sha256=sha256(cache/'complete.json'), plan_sha256=complete['plan_sha256'])
