"""Separate CSLD, nested-Al and Ta future-path assays; no pooled fitting."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path, resolve_path
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import write_metric_rows
from .common import read, plan, folder, result, canonical_source, exact_frames
from .data import patches, descriptors, producer_identity
from .features import encode, bank


def main_config(c):
    path = resolve_path(c['main_config'])
    if sha(path) != c['main_config_sha256']:
        raise ValueError('Main experiment configuration changed after diagnostic binding')
    return read(path)


def raw(item, kind):
    path = Path(item['binary'])
    if sha(path / 'manifest.json') != item['binary_manifest_sha256']:
        raise ValueError('Changed diagnostic trajectory')
    if kind != 'ta':
        return ShootingBinaryTrajectory.load(path)
    # Explicit second repository producer: positions-only elemental trajectories.
    m = read(path / 'manifest.json')
    if m['format'] != 'pointcloudmaterials.temporal_lammps_trajectory' or m['state'] != 'complete':
        raise ValueError(f'Unexpected Ta producer: {path}')
    arrays = {k: np.load(path / m['arrays'][k]['file'], mmap_mode='r')
              for k in ('positions', 'box_low', 'box_high', 'atom_ids', 'timesteps')}
    return SimpleNamespace(root=path, **arrays)


def bind(c):
    base = main_config(c)
    main = plan(base)
    parents, provenance = [], []
    for kind, dataset in c['al_campaigns'].items():
        root = dataset_path(dataset)
        m = read(root / 'manifest.json')
        provenance.append(dict(dataset=dataset, sha256=sha(root / 'manifest.json')))
        for p in m['parents']:
            shots = []
            for b in [b for b in m['branches'] if b['parent_id'] == p['parent_id']]:
                o = read(root / b['branch_dir'] / 'outcome.json')
                if o['state'] != 'complete':
                    raise ValueError('Incomplete diagnostic future')
                binary = root / b['branch_dir'] / o['trajectory_artifact']['path']
                shots.append(dict(binary=str(binary), binary_manifest_sha256=sha(binary / 'manifest.json'),
                    branch_id=b['branch_id'], momentum_index=b.get('momentum_index'), noise_index=b.get('thermostat_index')))
            if len(shots) != 4:
                raise ValueError(f'Expected four diagnostic futures: {kind}/{p["parent_id"]}')
            parents.append(dict(kind=kind, parent_id=p['parent_id'], source=canonical_source(p['source_run_id']),
                recorded_role=p['source_split'], parent_sha256=p['data_sha256'], timestep_fs=3., factor=1., shots=shots))
    for i in range(6):
        root = dataset_path(f'ta-shooting-20260926-parent{i:02d}')
        shots = []
        hashes, parent_ids = set(), set()
        for j in range(4):
            branch = root / 'branches' / f'shot{j:02d}'
            o = read(branch / 'outcome.json')
            if o['state'] != 'complete':
                raise ValueError('Incomplete Ta future')
            hashes.add(o['origin']['source_sha256']); parent_ids.add(o['origin']['parent_id'])
            binary = branch / 'trajectory_binary_float16'
            shots.append(dict(binary=str(binary), binary_manifest_sha256=sha(binary / 'manifest.json'),
                branch_id=f'{i:02d}/{j:02d}', velocity_seed=o['origin']['velocity_seed']))
        if len(hashes) != 1 or len(parent_ids) != 1:
            raise ValueError('Ta sibling parent mismatch')
        parents.append(dict(kind='ta', parent_id=parent_ids.pop(), parent_sha256=hashes.pop(),
            source='archived-Ta-unknown-common-preparation', recorded_role='conditional-transfer-only',
            timestep_fs=2., factor=c['ta_coordinate_factor'], shots=shots))
    for p in parents:
        p['readout_exposure'] = main['source_roles'].get(p['source'], 'not_in_main_readout')
        p['encoder_source_path_match'] = any(p['source'] in path for path in main['binding']['encoder_train_source_paths'])
        for shot in p['shots']:
            trajectory = raw(shot, p['kind'])
            exact_frames(trajectory, p['timestep_fs'], [0., *base['horizons_ps']])
    binding = dict(config=c, main_identity=main['identity'], parents=parents, provenance=provenance,
                   producer_sha256=sha(Path(__file__)), data_producer=producer_identity())
    value = dict(identity=digest(binding), binding=binding, parents=parents)
    path = folder(c) / 'plan.json'
    if path.exists() and read(path) != value:
        raise ValueError('Diagnostic binding changed')
    write_json(path, value)
    return value


def prepare(c, index):
    base = main_config(c)
    binding = read(folder(c) / 'plan.json')
    if binding['binding']['data_producer'] != producer_identity():
        raise ValueError('Changed diagnostic descriptor producer or numerical libraries')
    p = binding['parents'][index]
    root = folder(c) / 'parents' / f'{index:03d}'
    root.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(plan=read(folder(c) / 'plan.json')['identity'], parent=index))
    if (root / 'complete.json').exists():
        r = read(root / 'complete.json')
        if r['identity'] != identity or sha(root / 'data.npz') != r['sha256']:
            raise ValueError('Changed diagnostic targets')
        return
    first = raw(p['shots'][0], p['kind'])
    rng = np.random.default_rng(np.random.SeedSequence([base['seed'], index, 42]))
    if p['kind'] == 'csld':
        main_index = next(i for i, x in enumerate(plan(base)['parents']) if x['parent_id'] == p['parent_id'])
        with np.load(folder(base) / 'parents' / f'{main_index:03d}' / 'parent.npz') as a:
            centers, weights, strata = a['centers'], a['weights'], a['strata']
    else:
        centers = rng.choice(len(first.atom_ids), c['centers_per_parent'], replace=False)
        weights = np.full(len(centers), 1 / len(centers)); strata = np.full(len(centers), -1)
    if np.any(np.diff(first.atom_ids) <= 0):
        raise ValueError('Diagnostic identities must be sorted and unique')
    box0 = first.box_high[0].astype(float) - first.box_low[0].astype(float)
    points0 = np.mod(first.positions[0].astype(float), box0)
    positions = patches(points0, box0, centers) * p['factor']
    observed, names = descriptors(positions)
    columns = [names.index(x) for x in base['path_descriptors']]
    future = np.empty((len(centers), 4, 3, len(columns)), np.float32)
    differences = []
    for j, shot in enumerate(p['shots']):
        trajectory = raw(shot, p['kind'])
        if not np.array_equal(trajectory.atom_ids, first.atom_ids):
            raise ValueError('Sibling atom identities differ')
        current_box = trajectory.box_high[0].astype(float) - trajectory.box_low[0].astype(float)
        points = np.mod(trajectory.positions[0].astype(float), current_box)
        delta = points - points0
        delta -= box0 * np.rint(delta / box0)
        error = float(np.abs(delta).max())
        # Nested continuations explicitly mix historical float16/float32 exports.
        # Preserve the measured discrepancy instead of pretending bitwise identity.
        allowed = c['nested_max_export_error_A'] if p['kind'] == 'nested' else 0.
        if not np.array_equal(current_box, box0) or error > allowed:
            raise ValueError(f'Sibling parent geometry changed: {p["parent_id"]}/{j}: {error} A')
        differences.append(error)
        indices = exact_frames(trajectory, p['timestep_fs'], base['horizons_ps'])
        for h, f in enumerate(indices):
            box = trajectory.box_high[f].astype(float) - trajectory.box_low[f].astype(float)
            points = np.mod(trajectory.positions[f].astype(float), box)
            bank, actual = descriptors(patches(points, box, centers) * p['factor'])
            if actual != names:
                raise ValueError('Changed diagnostic descriptor schema')
            future[:, j, h] = bank[:, columns]
        write_json(root / 'progress.json', dict(shots=j + 1, total=4))
    np.savez_compressed(root / 'data.npz', positions=positions, descriptors=observed, future=future,
                        weights=weights, strata=strata, atom_ids=first.atom_ids[centers])
    write_json(root / 'complete.json', dict(identity=identity, sha256=sha(root / 'data.npz'), columns=names,
        parent_geometry_max_error_A=differences, coordinate_multiplier=p['factor']))


def seal(c):
    p = read(folder(c) / 'plan.json')
    banks = {k: [] for k in ('positions', 'descriptors', 'future', 'weights', 'strata', 'atom_ids', 'parent')}
    receipts = []
    for i, parent in enumerate(p['parents']):
        root = folder(c) / 'parents' / f'{i:03d}'
        r = read(root / 'complete.json')
        if sha(root / 'data.npz') != r['sha256']:
            raise ValueError('Changed diagnostic shard')
        receipts.append(r)
        with np.load(root / 'data.npz') as a:
            for k in banks:
                banks[k].append(np.full(len(a['positions']), i, np.int16) if k == 'parent' else a[k])
    files = {}
    for k, values in banks.items():
        path = folder(c) / f'{k}.npy'
        np.save(path, np.concatenate(values)); files[path.name] = sha(path)
    write_json(folder(c) / 'manifest.json', dict(identity=digest(dict(plan=p['identity'], receipts=receipts)),
        files=files, columns=receipts[0]['columns'], parents=p['parents']))


def evaluate(c):
    import torch
    from .fit import Head
    from .evaluate import row_scores
    base = main_config(c)
    m = read(folder(c) / 'manifest.json')
    data = {}
    for name, expected in m['files'].items():
        if sha(folder(c) / name) != expected:
            raise ValueError('Changed diagnostic sealed array')
        data[name.removesuffix('.npy')] = np.load(folder(c) / name)
    with np.load(result(base) / 'analyses/comparison-v1/kernel.npz') as a:
        omega, phase = a['omega'], a['phase']
    records = []
    for arm in base['arms']:
        x = bank(c, data, m, arm)
        for seed in base['fit_seeds']:
            fitted = result(base) / 'analyses/readouts-v1' / arm['name'] / str(seed)
            with np.load(fitted / 'normalization.npz') as a:
                transformed = ((x - a['x_mean']) / a['x_scale']).astype(np.float32)
                target = (data['future'].reshape(len(x), 4, -1) - a['y_mean']) / a['y_scale']
            saved = torch.load(fitted / 'path/checkpoint.pt', map_location='cpu', weights_only=False)
            head = Head(saved['input_dim'], saved['target_dim'], saved['mixtures'], saved['hidden'])
            head.load_state_dict(saved['model']); head.eval()
            with torch.no_grad():
                parts = [head.parameters_for(torch.from_numpy(transformed[i:i + 256])) for i in range(0, len(x), 256)]
                lw, mu, scale = [torch.cat([v[j] for v in parts]).numpy() for j in range(3)]
            prediction = dict(log_weight=lw, mean=mu, scale=scale, omega=omega, phase=phase,
                              event_probability=np.full((len(x), 13), 1 / 13))
            scores, _, _, _ = row_scores(base, target, np.full((len(x), 4), -1), prediction,
                                         np.random.default_rng(base['seed'] + seed))
            for i, p in enumerate(m['parents']):
                keep = data['parent'] == i
                for metric in ('path_nll', 'energy_score', 'mean_squared_error'):
                    records.append(dict(campaign=p['kind'], parent=p['parent_id'], source=p['source'],
                        readout_exposure=p['readout_exposure'], encoder_source_path_match=p['encoder_source_path_match'],
                        arm=arm['name'], seed=seed, metric=metric,
                        value=float(np.average(scores[metric][keep], weights=data['weights'][keep]))))
    out = result(c) / 'analyses/transfer-v1'
    write_metric_rows(records, out, family='shooting_laws', name='conditional-transfer')
    # Paired thermostat contrast: exact same parent/atom IDs and physical targets.
    contrasts = []
    main_parents = plan(base)['parents']
    for i, p in enumerate(m['parents']):
        if p['kind'] != 'csld':
            continue
        j = next(j for j, item in enumerate(main_parents) if item['parent_id'] == p['parent_id'])
        old = []
        for shot in range(12):
            with np.load(folder(base) / 'parents' / f'{j:03d}' / f'shot-{shot:02d}.npz') as a:
                old.append(a['future'])
        old = np.mean(old, axis=0)
        keep = data['parent'] == i
        delta = data['future'][keep].mean(1) - old
        for h, horizon in enumerate(base['horizons_ps']):
            for d, name in enumerate(base['path_descriptors']):
                contrasts.append(dict(parent=p['parent_id'], source=p['source'], horizon_ps=horizon, descriptor=name,
                    csld_minus_langevin=float(np.average(delta[:, h, d], weights=data['weights'][keep]))))
    write_metric_rows(contrasts, out, family='shooting_laws', name='thermostat-contrast')
    write_json(out / 'technical/complete.json', dict(parents=len(m['parents']), shots=4,
        interpretation='Conditional protocol/transfer diagnostics; no pooled fit, no independent-Ta generalization claim',
        event_labels='No new event classification in this structural diagnostic; nested historical censoring preserved at source'))


def submit(path):
    c = read(path); p = bind(c)
    base = main_config(c)
    main_launch = read(result(base) / 'technical/launch.json')
    tech = result(c) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Diagnostics already submitted')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src', 'docs/metrics'))
    receipt = dict(jobs={}, code=str(bundle.root), plan_identity=p['identity'])
    q = SlurmQueue(tech, bundle, 'src.research.shooting_laws.diagnostics',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
             TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'), tech / 'launch.json', receipt, 'LAW-D')
    with q.submission():
        grouped = SlurmQueue(tech, bundle, 'src.research.shooting_laws.queue', q.environment,
                            q.receipt_path, receipt, 'LAW-D')
        prep = grouped.submit('prepare', ['--array=0-3', '--cpus-per-task=2', '--mem=24G', '--time=08:00:00'],
            'afterok:' + main_launch['jobs']['seal'], command_stage='group',
            arguments=('--worker-stage', 'prepare', '--worker-module', 'diagnostics',
                       '--count', str(len(p['parents'])), '--groups', '4'))
        sealed = q.submit('seal', ['--cpus-per-task=2', '--mem=8G', '--time=00:30:00'], 'afterok:' + prep)
        enc = q.submit('encode', ['--array=0-1%1', '--gpus=1', '--cpus-per-task=2', '--mem=12G', '--time=02:00:00'], 'afterok:' + sealed, partition=base['gpu_partition'])
        q.submit('evaluate', ['--cpus-per-task=4', '--mem=16G', '--time=04:00:00'], 'afterok:' + enc + ':' + main_launch['jobs']['collect'])
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('bind', 'submit', 'prepare', 'seal', 'encode', 'evaluate'))
    parser.add_argument('--config', required=True); parser.add_argument('--index', type=int)
    args = parser.parse_args(); c = read(args.config)
    index = args.index if args.index is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    if args.stage == 'bind':
        print(json.dumps(dict(identity=bind(c)['identity']))); return
    if args.stage == 'submit':
        print(json.dumps(submit(args.config), indent=2)); return
    with recorded_stage(result(c) / 'technical' / f'{args.stage}-{index}.json', job=os.environ.get('SLURM_JOB_ID')):
        if args.stage == 'prepare': prepare(c, index)
        elif args.stage == 'seal': seal(c)
        elif args.stage == 'encode': encode(c, index)
        elif args.stage == 'evaluate': evaluate(c)


if __name__ == '__main__':
    main()
