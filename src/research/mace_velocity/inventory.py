"""Explicitly audit recorded repository coordinate/velocity producers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def inventory(config):
    root = Path(config['output'])/'technical'
    binaries = read(root/'binary-metadata.json')
    accepted, excluded, fingerprints = [], [], {}
    # These named attempts were quarantined by their simulation producers.
    invalid = ('invalid_canary_', 'interrupted_attempt_', 'invalid_endpoint_descriptor_preparation')
    for record in sorted(binaries, key=lambda r: (r['manifest']['storage_dtype'] != 'float32', r['path'])):
        path = Path(record['path'])
        if any(part in str(path) for part in invalid):
            excluded.append(dict(path=str(path), reason='producer-quarantined simulation attempt')); continue
        manifest = read(path/'manifest.json')
        if manifest != record['manifest']:
            raise ValueError(f'Manifest changed since discovery: {path}')
        trajectory = ShootingBinaryTrajectory.load(path)
        if manifest['velocity_units'] != 'angstrom_per_ps' or not np.all(trajectory.atom_types == 1):
            raise ValueError(f'Unexpected velocity units or Al atom types: {path}')
        if not record['dataset'].startswith(('al_', 'al_meam_')):
            raise ValueError(f'Element must be established from the simulation producer: {path}')
        fingerprint = tuple(manifest['source']['semantic_float32_sha256'][k] for k in ('positions', 'velocities')) + tuple(manifest['arrays'][k]['sha256'] for k in ('timesteps', 'atom_ids', 'box_low', 'box_high'))
        if fingerprint in fingerprints:
            excluded.append(dict(path=str(path), reason='identical source arrays and timeline', represented_by=fingerprints[fingerprint])); continue
        fingerprints[fingerprint] = str(path)
        metadata = record['metadata']
        independent = manifest.get('provenance', {}).get('campaign_type') == 'independently_melted_boundary_parent_sources'
        if independent:
            if metadata is None:
                campaign = read(path.parents[2]/'manifest.json')
                metadata = next(r for r in campaign['runs'] if r['run_id'] == manifest['provenance']['run_id'])
            split = {'optimization': 'train', 'model_selection': 'val', 'final_validation': 'test'}[metadata['source_split']]
            lineage = 'independent_melt_'+str(metadata['preparation_seed'])
            balance_group = lineage
            timestep_fs = 3.0  # Checked against each independent campaign below.
            producer = read(path.parents[2]/'manifest.json')
            if producer['protocol']['timestep_fs'] != timestep_fs:
                raise ValueError(f'Changed independent-source time units: {path}')
        else:
            # Shared prepared liquid: all descendants stay together in training.
            split, lineage, timestep_fs = 'train', 'legacy_shared_prepared_liquid', 3.0
            if metadata is None:
                ancestors = [p/'metadata.json' for p in path.parents if (p/'metadata.json').is_file()]
                if ancestors:
                    metadata = read(ancestors[0])
                elif manifest.get('provenance', {}).get('purpose') == 'uninterrupted_24ps_continuation_comparator':
                    campaign = path.parents[2]
                    branch = manifest['provenance']['branch_id']
                    metadata = read(campaign/'branches'/branch/'metadata.json')
                else:
                    raise ValueError(f'Unresolved branch provenance: {path}')
            if 'timestep_fs' in metadata and metadata['timestep_fs'] != timestep_fs:
                raise ValueError(f'Unexpected branch time units: {path}')
            source = metadata.get('root_source_lineage_id', metadata.get('source_run_id'))
            if source is None:
                raise ValueError(f'Missing original branch source: {path}')
            balance_group = re.sub(r'^source_group_\d+/', '', source)
        frames = np.arange(1, trajectory.frame_count)
        provenance = manifest.get('provenance', {})
        # The original first-passage record retains the finer prefix cadence.
        # A composed fixed-24ps record contributes its continuation only.
        if provenance.get('campaign_type') == 'fixed_horizon_compatibility_from_nested_first_passage':
            cutoff = int(metadata['source_last_timestep'])
            frames = frames[trajectory.timesteps[frames] > cutoff]
        if not len(frames):
            excluded.append(dict(path=str(path), reason='no new paired frames after removing retained prefix')); continue
        accepted.append(dict(dataset=record['dataset'], path=str(path), format='shooting_binary',
            manifest_sha256=sha256(path/'manifest.json'), split=split, lineage=lineage,
            balance_group=balance_group, timestep_fs=timestep_fs, eligible_frames=frames.tolist(),
            frame_count=trajectory.frame_count, temperature_K=float(metadata['temperature_K']),
            producer_split=metadata.get('source_split'), atom_count=trajectory.atom_count))
    # Legacy NPZ is the verified producer for ten paired text dumps. Its identity
    # and velocity schemas are checked in preparation, with the original producer.
    paired = []
    for group in read(root/'additional-files.json'):
        for name in group['files']:
            if not name.endswith('.npz'):
                continue
            with zipfile.ZipFile(name) as archive:
                if 'velocities_A_per_ps.npy' not in archive.namelist():
                    continue
            path = Path(name)
            campaign = read(path.parents[2]/'manifest.json')
            if campaign['shared_liquid_source']['prepared_liquid_sha256'] != '24181fba2b4b1facb9a32124170683932012b34cdc94fc57de884a59a50cca7c':
                raise ValueError(f'Unrecognized preparation lineage: {path}')
            with np.load(path) as values:
                steps = values['step']
            accepted.append(dict(dataset=group['dataset'], path=str(path), format='legacy_npz',
                split='train', lineage='legacy_shared_prepared_liquid', balance_group=str(path.parent),
                timestep_fs=campaign['protocol']['timestep_fs'], eligible_frames=list(range(1,len(steps))),
                frame_count=len(steps), temperature_K=campaign['protocol']['temperature_K'], atom_count=campaign['atom_count']))
            paired.append(str(path.parent/'velocities.lammpstrj'))
    # One completed older stream has no NPZ. Convert selected paired dump frames
    # using the maintained conversion dispatcher before preparation.
    for record in read(root/'text-headers.json'):
        if record.get('state') != 'velocity_dump':
            continue
        name = record['path']
        if name in paired:
            excluded.append(dict(path=name, reason='represented by verified legacy NPZ')); continue
        if record['fields'] == ['id','type','vx','vy','vz'] and record['dataset'] != 'interrupted_attempts':
            path = Path(name)
            campaign = read(path.parents[2]/'manifest.json')
            paired_positions = path.parent/'trajectory.lammpstrj'
            if not paired_positions.is_file():
                raise FileNotFoundError(f'Velocity dump has no matching coordinates: {name}')
            target = Path(config['cache'])/'converted'/hashlib.sha256(name.encode()).hexdigest()[:12]
            accepted.append(dict(dataset=record['dataset'], path=str(target), format='paired_dump_conversion',
                positions_dump=str(paired_positions), velocities_dump=name, split='train',
                lineage='legacy_shared_prepared_liquid', balance_group=str(path.parent),
                timestep_fs=campaign['protocol']['timestep_fs'], atom_count=record['atoms'],
                temperature_K=campaign['protocol']['temperature_K']))
        else:
            excluded.append(dict(path=name, reason='quarantined/interrupted attempt, restart audit, or binary-represented comparator'))
    for index, record in enumerate(accepted):
        record['id'] = index
    splits = {r['lineage']:set() for r in accepted}
    for r in accepted: splits[r['lineage']].add(r['split'])
    if any(len(s)>1 for s in splits.values()):
        raise AssertionError('Preparation lineage crosses train/validation/test')
    payload = dict(protocol='mace_local_phase_space_v1', records=accepted, excluded=excluded,
        counts=dict(records=len(accepted), by_split=dict(Counter(r['split'] for r in accepted)),
                    independent_preparations=len(splits), by_format=dict(Counter(r['format'] for r in accepted))),
        scope='All discovered usable completed Al coordinate/velocity trajectories; stratified local samples, not every atom/frame.',
        holdout='Original independent-melt train/val/test; entire shared-liquid legacy family in training.',
        discovery_hashes={p.name:sha256(p) for p in (root/'binary-metadata.json',root/'additional-files.json',root/'text-headers.json',root/'discovery-files.json')})
    write_json(root/'inventory.json', payload)
    print('INVENTORY', payload['counts'], flush=True)
    return payload
