"""Pinned, complete-ball spatial context for the existing Ni kinetic cohort.

This derivative collection reuses the local244 bank unchanged. Each raw source
is an independently resumable collection job, shared by native trajectories and
the separate birth assay. Normalization is fitted only on native train frames.
"""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.research.continuous_motion.data import (
    Source, read, root, cache, resolve_path, sha, digest, write_json, atomic_save,
)
from src.research.ni_kinetic_latent import data as base_data
from .features import RegionalMeasurements


FAMILY = 'ni_spatial_scales'
transform = base_data.transform
inverse = base_data.inverse


def _base(c):
    path = resolve_path(c['base_run'])/'technical/plan.json'
    if sha(path) != c['base_plan_sha256']:
        raise ValueError('Changed frozen kinetic base plan')
    return read(path)


def _requests(p, job):
    requests = defaultdict(lambda: defaultdict(list))
    for local_row, row_index in enumerate(job['native_rows']):
        row = p['rows'][row_index]
        for t, frame in enumerate(range(row['frame']-3, row['frame']+61)):
            requests[frame][row['center']].append(('native', local_row, t))
    for local_row, row_index in enumerate(job['birth_rows']):
        row = p['birth_unique_rows'][row_index]
        for t, frame in enumerate(range(row['frame']-3, row['frame']+1)):
            requests[frame][row['center']].append(('birth', local_row, t))
    return requests


def prepare(c):
    base = _base(c)
    history_path = resolve_path(base['config']['birth_history_run'])/'technical/plan.json'
    if sha(history_path) != base['birth']['plan_sha256']:
        raise ValueError('Changed birth history producer')
    history = read(history_path)
    birth_path = resolve_path(history['config']['birth_run'])/'technical/plan.json'
    if sha(birth_path) != history['config']['birth_plan_sha256']:
        raise ValueError('Changed full-cell birth inventory')
    birth_plan = read(birth_path)
    if sha(base['birth']['cohort_path']) != base['birth']['cohort_sha256']:
        raise ValueError('Changed birth assay identities')
    cohort = torch.load(base['birth']['cohort_path'], map_location='cpu', weights_only=False)
    unique = [None]*len(cohort['state'])
    for row, index in zip(cohort['rows'], cohort['indices']):
        record = {key: row[key] for key in ('source', 'frame', 'center', 'ancestor', 'role')}
        if unique[index] is not None and unique[index] != record:
            raise ValueError('Birth assay aliases different physical states')
        unique[index] = record
    if any(row is None for row in unique):
        raise ValueError('Birth state bank has unreferenced rows')
    sources = {item['key']: item for item in birth_plan['sources']}
    for item in base['sources']:
        key = item['key']
        if key in sources and sources[key]['source']['input_identity'] != item['source']['input_identity']:
            raise ValueError('Native and birth full-cell identities disagree: '+key)
        sources[key] = item
    model = RegionalMeasurements(c['context']['radii_A'][0], c['context']['shell_fractions'],
                                 c['context']['shell_width_fraction'])
    contract = dict(radii_A=c['context']['radii_A'], columns=model.columns,
        radial_channels=4, max_ell=6, radial_centers=np.linspace(.15, .95, 4).tolist(),
        radial_width=.12, density_scale=12., taper='quintic from0.8R toR',
        shell_fractions=c['context']['shell_fractions'],
        shell_width_fraction=c['context']['shell_width_fraction'], affine_ridge_A2=1e-4,
        support='every observed atom with0<minimum-image radius<R; same central atom ID; support reselected at every frame; no nearest-neighbor cap',
        inputs='center-relative positions Angstrom and velocities Angstrom/ps; no box/time/temperature/phase labels fed to model',
        padding='explicit valid mask; invalid entries outside compact support with zero velocity',
        arithmetic='float64 source minimum images and center subtraction, float32 GPU features',
        normalization='native train only, all64frames; raw mean/std floor1e-4, asinh, mean/std floor1e-4; no clipping',
        implementations={str(path): sha(path) for path in (
            Path(__file__), Path(__file__).with_name('features.py'),
            Path('src/models/encoders/smooth_density.py'),
            Path('src/research/ni_latent_sim/measurements.py'),
            Path('src/research/continuous_motion/data.py'))})
    native_by_source = defaultdict(list); birth_by_source = defaultdict(list)
    for index, row in enumerate(base['rows']):
        native_by_source[row['source']].append(index)
    for index, row in enumerate(unique):
        birth_by_source[row['source']].append(index)
    p = dict(base, config=c, base_run=c['base_run'], base_plan_sha256=c['base_plan_sha256'],
             base_config=base['config'], base_data_identity=base['data_identity'],
             context=contract, context_columns=model.columns, birth_unique_rows=unique,
             birth_inventory_path=str(birth_path), birth_inventory_sha256=sha(birth_path),
             context_sources=[sources[key] for key in sorted(set(native_by_source)|set(birth_by_source))])
    jobs = []
    for item in p['context_sources']:
        key = item['key']
        job = dict(key=key, native_rows=native_by_source[key], birth_rows=birth_by_source[key])
        requests = _requests(p, job)
        times = np.asarray(item['source']['times_ps'])
        frames = sorted(requests)
        if min(frames) < 0 or max(frames) >= len(times):
            raise ValueError('Missing requested context frame: '+key)
        for indices, rows, span in ((job['native_rows'], p['rows'], 61),
                                    (job['birth_rows'], unique, 1)):
            for index in indices:
                frame = rows[index]['frame']; offsets = np.arange(-3, span)
                if not np.allclose(times[frame+offsets]-times[frame], offsets*.1, atol=1e-8, rtol=0):
                    raise ValueError('Context observation cadence differs from0.1ps: '+key)
        job.update(frames=frames, unique_observations=sum(len(requests[f]) for f in frames))
        jobs.append(job)
    p['context_jobs'] = jobs
    p['context_counts'] = dict(native_rows=len(p['rows']), native_frames=64,
        birth_unique_states=len(unique), birth_history_frames=4, birth_assay_rows=len(cohort['rows']),
        sources=len(jobs), source_frames=sum(len(j['frames']) for j in jobs),
        unique_observations=sum(j['unique_observations'] for j in jobs),
        requested_observations=len(p['rows'])*64+len(unique)*4)
    p['data_identity'] = digest([base['data_identity'], contract, p['birth_inventory_sha256'],
                                 unique, jobs, p['base_plan_sha256']])
    p.pop('identity')
    p['identity'] = digest(p)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != p:
        raise ValueError('Changed spatial-scale plan; use a new revision')
    write_json(destination, p)
    return p


def _modules(c, device):
    return [RegionalMeasurements(radius, c['context']['shell_fractions'],
            c['context']['shell_width_fraction']).to(device).eval()
            for radius in c['context']['radii_A']]


def _frame_inputs(reader, frame, centers, radii):
    start = time.perf_counter()
    q, v, box = reader.frame(frame)
    if 2*max(radii) >= np.min(box):
        raise ValueError('Regional ball reaches the periodic half-box limit')
    index = np.searchsorted(reader.ids, centers)
    if not np.array_equal(reader.ids[index], centers):
        raise ValueError('Requested context center is missing')
    loaded = time.perf_counter()
    tree = cKDTree(q, boxsize=box)
    near = tree.query_ball_point(q[index], r=max(radii), workers=1)
    position, velocity = [], []
    for center, neighbors in zip(index, near):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        neighbors = neighbors[neighbors != center]
        relative = q[neighbors]-q[center]
        relative -= box*np.rint(relative/box)
        position.append(relative.astype(np.float32))
        velocity.append((v[neighbors]-v[center]).astype(np.float32))
    return position, velocity, dict(read_s=loaded-start, index_s=time.perf_counter()-loaded)


def _pack(position, velocity, radius, device):
    masks = [np.linalg.norm(q, axis=1) < radius for q in position]
    counts = np.array([mask.sum() for mask in masks])
    if np.any(counts == 0):
        raise ValueError('Empty observed regional support')
    q = np.zeros((len(position), int(counts.max()), 3), dtype=np.float32)
    v = np.zeros_like(q); valid = np.zeros(q.shape[:2], dtype=bool)
    for i, (p, speed, mask, n) in enumerate(zip(position, velocity, masks, counts)):
        q[i, :n] = p[mask]; v[i, :n] = speed[mask]; valid[i, :n] = True
    return tuple(torch.from_numpy(x).to(device) for x in (q, v, valid)), counts


@torch.inference_mode()
def measure_frame(reader, frame, centers, modules, batch_size):
    device = next(modules[0].buffers()).device
    radii = [module.radius_A for module in modules]
    position, velocity, timing = _frame_inputs(reader, frame, centers, radii)
    start = time.perf_counter()
    output = torch.empty((len(centers), len(modules), 87), dtype=torch.float32)
    counts = np.empty((len(centers), len(modules)), dtype=np.int32)
    for first in range(0, len(centers), batch_size):
        last = min(first+batch_size, len(centers))
        for r, module in enumerate(modules):
            inputs, n = _pack(position[first:last], velocity[first:last], module.radius_A, device)
            values = module(*inputs)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite context frame={frame}, radius={module.radius_A}')
            output[first:last, r] = values.cpu()
            counts[first:last, r] = n
    timing['features_s'] = time.perf_counter()-start
    return output, counts, timing


def prepare_source(c, source_key, device='cuda', deadline=float('inf')):
    p = read(root(c)/'technical/plan.json')
    job = next(job for job in p['context_jobs'] if job['key'] == source_key)
    item = next(item for item in p['context_sources'] if item['key'] == source_key)
    folder = cache(c)/'context-sources'/source_key
    marker = folder/'complete.json'; identity = digest([p['data_identity'], job])
    if marker.exists():
        receipt = read(marker)
        if receipt['identity'] != identity or sha(folder/'data.pt') != receipt['sha256']:
            raise ValueError('Changed completed regional source: '+source_key)
        return True
    requests = _requests(p, job)
    payload = dict(identity=identity, frame=-1,
        native=torch.full((len(job['native_rows']), 64, 3, 87), float('nan')),
        birth=torch.full((len(job['birth_rows']), 4, 3, 87), float('nan')),
        native_rows=job['native_rows'], birth_rows=job['birth_rows'],
        completed_frames=0, measured_centers=0, counts_sum=[0, 0, 0],
        timing=dict(read_s=0., index_s=0., features_s=0.))
    partial = folder/'partial.pt'
    if partial.exists():
        payload = torch.load(partial, map_location='cpu', weights_only=False)
        if payload['identity'] != identity:
            raise ValueError('Changed partial regional source: '+source_key)
    def checkpoint(state):
        atomic_save(partial, payload)
        write_json(root(c)/'technical/collection'/source_key/'progress.json', dict(
            identity=identity, source=source_key, state=state, last_frame=payload['frame'],
            completed_frames=payload['completed_frames'], total_frames=len(job['frames']),
            measured_centers=payload['measured_centers'], timing=payload['timing'],
            updated_unix=time.time()))
    reader = Source(item['source']); modules = _modules(c, device)
    for frame in job['frames']:
        if frame <= payload['frame']:
            continue
        if time.time() > deadline-30:
            checkpoint('paused')
            return False
        centers = sorted(requests[frame])
        values, counts, timing = measure_frame(reader, frame, centers, modules, c['context']['batch_size'])
        for i, center in enumerate(centers):
            for dataset, row, t in requests[frame][center]:
                payload[dataset][row, t] = values[i]
        payload['frame'] = frame
        payload['completed_frames'] += 1
        payload['measured_centers'] += len(centers)
        payload['counts_sum'] = (np.array(payload['counts_sum'])+counts.sum(0)).tolist()
        for key in timing:
            payload['timing'][key] += timing[key]
        if payload['completed_frames'] % c['context']['checkpoint_every_frames'] == 0:
            checkpoint('collecting')
    if not torch.isfinite(payload['native']).all() or not torch.isfinite(payload['birth']).all():
        raise FloatingPointError('Incomplete or nonfinite regional source: '+source_key)
    atomic_save(folder/'data.pt', payload)
    write_json(marker, dict(identity=identity, sha256=sha(folder/'data.pt'),
        source=source_key, completed_frames=payload['completed_frames'],
        measured_centers=payload['measured_centers'], timing=payload['timing'],
        mean_atoms=(np.asarray(payload['counts_sum'])/payload['measured_centers']).tolist()))
    write_json(root(c)/'technical/collection'/source_key/'progress.json', dict(
        identity=identity, source=source_key, state='complete', last_frame=payload['frame'],
        completed_frames=payload['completed_frames'], total_frames=len(job['frames']),
        measured_centers=payload['measured_centers'], timing=payload['timing'],
        updated_unix=time.time()))
    partial.unlink(missing_ok=True)
    return True


def assemble(c):
    p = read(root(c)/'technical/plan.json')
    marker = cache(c)/'context.json'; destination = cache(c)/'context.pt'
    if marker.exists():
        receipt = read(marker)
        if receipt['identity'] != p['data_identity'] or sha(destination) != receipt['sha256']:
            raise ValueError('Changed assembled regional context')
        write_json(root(c)/'technical/context-complete.json', receipt)
        return receipt
    native = torch.empty((len(p['rows']), 64, 3, 87), dtype=torch.float32)
    birth = torch.empty((len(p['birth_unique_rows']), 4, 3, 87), dtype=torch.float32)
    source_hashes = {}
    for job in p['context_jobs']:
        folder = cache(c)/'context-sources'/job['key']
        done = read(folder/'complete.json')
        if done['identity'] != digest([p['data_identity'], job]) or sha(folder/'data.pt') != done['sha256']:
            raise ValueError('Changed source context at assembly: '+job['key'])
        piece = torch.load(folder/'data.pt', map_location='cpu', weights_only=False)
        native[job['native_rows']] = piece['native']; birth[job['birth_rows']] = piece['birth']
        source_hashes[job['key']] = done['sha256']
    train = np.flatnonzero([row['role'] == 'train' for row in p['rows']])
    raw = native[train].flatten(0, 1).double()
    raw_mean, std = raw.mean(0), raw.std(0)
    raw_scale = std.clamp_min(1e-4)
    u = torch.asinh((raw-raw_mean)/raw_scale)
    normal = dict(raw_mean=raw_mean.float(), raw_scale=raw_scale.float(),
                  mean=u.mean(0).float(), scale=u.std(0).clamp_min(1e-4).float(), active=std>1e-8)
    del raw, u
    payload = dict(native=transform(native, normal), birth=transform(birth, normal),
                   normalization=normal, radii_A=p['context']['radii_A'], columns=p['context_columns'])
    if not torch.isfinite(payload['native']).all() or not torch.isfinite(payload['birth']).all():
        raise FloatingPointError('Nonfinite normalized regional context')
    atomic_save(destination, payload)
    receipt = dict(identity=p['data_identity'], sha256=sha(destination), sources=source_hashes,
                   native_shape=list(native.shape), birth_shape=list(birth.shape),
                   train_rows=len(train), fitted_observations=len(train)*64,
                   active_per_radius=normal['active'].sum(-1).tolist())
    write_json(marker, receipt)
    write_json(root(c)/'technical/context-complete.json', receipt)
    return receipt


def _context(c, p, device):
    path = cache(c)/'context.pt'; receipt = read(path.with_suffix('.json'))
    if receipt['identity'] != p['data_identity'] or sha(path) != receipt['sha256']:
        raise ValueError('Changed normalized regional context')
    return torch.load(path, map_location=device, weights_only=False)


def _scale_index(arm, radii):
    return {f'r{radius:g}': index for index, radius in enumerate(radii)}[arm['scale']]


def load(c, arm, device='cpu'):
    p = read(root(c)/'technical/plan.json')
    _, bank = base_data.load(p['base_config'], device=device)
    if arm['scale'] == 'local':
        regional = torch.zeros((*bank['x'].shape[:-1], 87), dtype=bank['x'].dtype, device=device)
        normalization = dict(radii_A=p['context']['radii_A'], selected_scale='local',
                             fitted=False, reason='Zero context is defined without regional statistics')
    else:
        context = _context(c, p, device)
        regional = context['native'][..., _scale_index(arm, context['radii_A']), :]
        normalization = dict(context['normalization'], radii_A=context['radii_A'],
                             selected_scale=arm['scale'])
    bank['input'] = torch.cat((bank['x'], regional), -1)
    bank['context_normalization'] = normalization
    bank['context_scale'] = arm['scale']
    return p, bank


def load_births(c, bank, arm):
    p = read(root(c)/'technical/plan.json')
    cohort, target_history = base_data.load_births(p['base_config'], bank)
    if arm['scale'] == 'local':
        regional = torch.zeros((*target_history.shape[:-1], 87), dtype=target_history.dtype)
    else:
        context = _context(c, p, 'cpu')
        regional = context['birth'][..., _scale_index(arm, context['radii_A']), :]
    return cohort, target_history, torch.cat((target_history, regional), -1)


@torch.inference_mode()
def benchmark(c, device='cuda', frames=3):
    """Local-only integration benchmark on complete real training-frame requests."""
    started = time.perf_counter(); p = read(root(c)/'technical/plan.json')
    job = next(job for job in p['context_jobs'] if job['native_rows']
               and p['rows'][job['native_rows'][0]]['role'] == 'train')
    item = next(item for item in p['context_sources'] if item['key'] == job['key'])
    reader = Source(item['source']); modules = _modules(c, device)
    requests = _requests(p, job)
    chosen = job['frames'][:frames]
    records = []
    for repeat in range(2):
        for frame in chosen:
            centers = sorted(requests[frame])
            start = time.perf_counter()
            values, counts, timing = measure_frame(reader, frame, centers, modules, c['context']['batch_size'])
            records.append(dict(repeat=repeat, frame=frame, centers=len(centers),
                elapsed_s=time.perf_counter()-start, timing=timing,
                mean_atoms=counts.mean(0).tolist(), max_atoms=counts.max(0).tolist()))
    positions, velocities, _ = _frame_inputs(reader, chosen[0], sorted(requests[chosen[0]])[:4],
                                              c['context']['radii_A'])
    symmetry = []
    generator = torch.Generator(device=device).manual_seed(c['seed'])
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), device=device, generator=generator))
    for module in modules:
        (q, v, mask), counts = _pack(positions, velocities, module.radius_A, device)
        expected = module(q, v, mask)
        rotated = module(q@rotation, v@rotation, mask)
        padded = module(torch.cat((q, torch.randn((len(q), 7, 3), device=device)), 1),
                        torch.cat((v, torch.randn((len(q), 7, 3), device=device)), 1),
                        torch.cat((mask, torch.zeros((len(q), 7), dtype=torch.bool, device=device)), 1))
        # A uniform boost applied to absolute velocities cancels in the producer.
        boost = np.array([7., -3., 2.])
        q_full, v_full, box = reader.frame(chosen[0])
        centers = np.searchsorted(reader.ids, sorted(requests[chosen[0]])[:4])
        boost_error = max(float(np.max(np.abs(((v_full+boost)-(v_full[center]+boost))
                                                -(v_full-v_full[center])))) for center in centers)
        errors = dict(radius_A=module.radius_A,
            rotation_max_relative=float(((rotated-expected).abs()/(1+expected.abs())).max()),
            padding_max_absolute=float((padded-expected).abs().max()),
            padding_max_relative=float(((padded-expected).abs()/(1+expected.abs())).max()),
            uniform_boost_relative_velocity_error_A_per_ps=boost_error)
        if errors['rotation_max_relative'] > 2e-4 or errors['padding_max_relative'] > 2e-4 or boost_error > 1e-10:
            raise ValueError('Regional measurement symmetry audit failed: '+str(errors))
        symmetry.append(errors)
    warm = [record for record in records if record['repeat'] == 1]
    center_rate = sum(r['centers'] for r in warm)/sum(r['elapsed_s'] for r in warm)
    frame_rate = len(warm)/sum(r['elapsed_s'] for r in warm)
    receipt = dict(data_identity=p['data_identity'], source=job['key'], training_only=True,
        device=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else str(device),
        frames=records, symmetry=symmetry, total_elapsed_s=time.perf_counter()-started,
        centers_per_second=center_rate, frames_per_second=frame_rate,
        extrapolation_single_worker_hours=max(p['context_counts']['unique_observations']/center_rate,
                                               p['context_counts']['source_frames']/frame_rate)/3600,
        extrapolation_limit='Few warm-cache training frames; excludes source startup, checkpoint and assembly; not a queue completion guarantee')
    write_json(root(c)/'technical/context-benchmark.json', receipt)
    return receipt
