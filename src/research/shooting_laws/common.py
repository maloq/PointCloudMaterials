"""Immutable campaign binding and explicit lineage/input contracts."""
import json
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.fixed_cohort.dataset import read_release
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path, resolve_path


def read(path):
    return json.loads(Path(path).read_text())


def canonical_source(value):
    return value.split('/', 1)[1] if value.startswith('source_group_') else value


def folder(c):
    return resolve_path(c['cache'])


def result(c):
    return resolve_path(c['output'])


def exact_frames(raw, timestep_fs, times):
    actual = np.asarray(raw.timesteps, dtype=float) * timestep_fs / 1000
    indices = np.searchsorted(actual, times)
    if np.any(indices >= len(actual)) or not np.allclose(actual[indices], times, rtol=0, atol=1e-8):
        raise ValueError(f'Missing exact physical observations: {raw.root}, wanted={times}, actual={actual}')
    return indices


def load_raw(branch):
    if sha(Path(branch['binary']) / 'manifest.json') != branch['binary_manifest_sha256']:
        raise ValueError(f'Changed binary manifest: {branch["binary"]}')
    return ShootingBinaryTrajectory.load(branch['binary'])


def bind(c):
    parents, shots, inputs, base_protocol = {}, [], [], None
    for campaign in c['campaigns']:
        root = dataset_path(campaign)
        manifest = read(root / 'manifest.json')
        protocol = manifest['protocol']
        if base_protocol is None:
            base_protocol = protocol
        elif protocol != base_protocol:
            raise ValueError(f'Cannot pool different dynamical protocols: {campaign}')
        inputs.append(dict(dataset=campaign, manifest_sha256=sha(root / 'manifest.json')))
        for p in manifest['parents']:
            parent = {k: p[k] for k in ('parent_id', 'source_run_id', 'source_split', 'source_velocity_seed',
                                       'temperature_K', 'source_frame_time_ps', 'data_sha256')}
            source = canonical_source(p['source_run_id'])
            parent['source'] = source
            parent['role'] = ('test' if p['source_split'] == 'validation' else
                              'selection' if p['source_velocity_seed'] == c['selection_source_seed'] else 'train')
            if p['source_split'] not in ('train', 'validation'):
                raise ValueError(f'Unexpected historical role: {p}')
            if p['parent_id'] in parents and parents[p['parent_id']] != parent:
                raise ValueError(f'Parent identity differs between campaigns: {p["parent_id"]}')
            parents[p['parent_id']] = parent
        for b in manifest['branches']:
            path = root / b['branch_dir'] / 'outcome.json'
            o = read(path)
            if o['state'] != 'complete':
                raise ValueError(f'Incomplete shooting future: {path}')
            for key in ('parent_id', 'velocity_seed', 'thermostat_seed', 'source_run_id', 'source_split'):
                if o[key] != b[key]:
                    raise ValueError(f'Outcome/manifest mismatch: {path}/{key}')
            binary = path.parent / o['trajectory_artifact']['path']
            raw = ShootingBinaryTrajectory.load(binary)
            exact_frames(raw, protocol['timestep_fs'], np.arange(51) * .3)
            shots.append(dict(parent_id=b['parent_id'], campaign=campaign, branch_id=b['branch_id'],
                binary=str(binary), binary_manifest_sha256=sha(binary / 'manifest.json'),
                outcome_sha256=sha(path), velocity_seed=b['velocity_seed'], thermostat_seed=b['thermostat_seed']))
    if len(parents) != 40 or len(shots) != 480:
        raise ValueError(f'Frozen 40-parent/480-shot release changed: {len(parents)}/{len(shots)}')
    if len({(b['velocity_seed'], b['thermostat_seed']) for b in shots}) != 480:
        raise ValueError('Duplicate shooting random-seed pairs')
    ordered = [parents[k] for k in sorted(parents)]
    source_roles = {}
    for p in ordered:
        if p['source'] in source_roles and source_roles[p['source']] != p['role']:
            raise ValueError('Source crosses roles')
        source_roles[p['source']] = p['role']
        p['shots'] = [s for s in shots if s['parent_id'] == p['parent_id']]
        if len(p['shots']) != 12:
            raise ValueError(f'Unbalanced shots: {p["parent_id"]}')
    _, fixed = read_release(c['fixed_dataset']['root'])
    if fixed['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Changed Al64 provenance release')
    # These frozen encoders were initialized randomly, then fitted on native
    # Al64 train ancestors only. Verify actual source paths, not model names.
    training_paths = [str(dataset_path(s['dataset']) / s['relative_trajectory_path'])
                      for s in fixed['sources'] if s['role'] == 'train']
    overlap = [s for s in source_roles if any(s in path for path in training_paths)]
    if overlap:
        raise ValueError(f'Frozen encoder pretraining overlaps shooting sources: {overlap}')
    encoders = []
    for spec in c['encoders']:
        p = resolve_path(spec['checkpoint'])
        if sha(p) != spec['checkpoint_sha256']:
            raise ValueError(f'Changed encoder: {p}')
        for name, expected in spec['inference_dependencies'].items():
            if sha(resolve_path(spec['producer']) / name) != expected:
                raise ValueError(f'Changed frozen encoder source: {name}')
        encoders.append(dict(name=spec['name'], checkpoint_sha256=sha(p)))
    binding = dict(config=c, campaigns=inputs, protocol=base_protocol, parents=ordered,
                   encoders=encoders, al64_identity=fixed['identity'], encoder_train_source_paths=training_paths,
                   producer_sha256=sha(Path(__file__)))
    plan = dict(identity=digest(binding), binding=binding, parents=ordered,
                source_roles=source_roles, protocol=base_protocol)
    root = folder(c)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists() and read(root / 'plan.json') != plan:
        raise ValueError('Frozen shooting release changed; use a new output/cache')
    write_json(root / 'plan.json', plan)
    technical = result(c) / 'technical'
    write_json(technical / 'ancestry.json', dict(source_roles=source_roles, encoder_train_source_paths=training_paths,
        overlap=overlap, fixed_dataset=c['fixed_dataset'], interpretation='Historical shooting release; no Al64 resplit or row replacement'))
    write_json(technical / 'prediction-context.json', dict(
        encoder_inputs=dict(coordinates='observed, minimum-image, Al Angstrom', atoms=80, radius_A=8,
                            edge_cutoff_A=5, halo=None, constant_atom_channel=True, history=0,
                            motion=False, conditions=[], relaxation=False),
        predictor_inputs=dict(banks=c['arms'], conditions=[], history=0, label_horizons_ps=c['horizons_ps']),
        frozen_pretraining=dict(views='same-time observed/relaxed Al64 train pairs', initial_weights='random',
                               epi_teacher='fixed random initial encoder; training-only'),
        sampling='Present full-cell PTM strata; all actual encoder atoms lie within the audited 8-A sphere',
        selection='source-held-out predictive NLL; original outer validation is reused historical test',
        external_inputs=[]))
    return plan


def plan(c):
    return read(folder(c) / 'plan.json')
