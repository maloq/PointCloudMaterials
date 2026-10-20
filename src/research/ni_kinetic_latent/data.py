"""Sealed native Ni measurements and independent physical annotation views."""
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.research.continuous_motion.data import (
    read, root, cache, sha, digest, write_json, atomic_save, resolve_path,
)
from src.postprocessing.io import arrays, geometry


FAMILY = 'ni_kinetic_latent'
GROUPS = ((0, 40), (40, 153), (153, 227), (227, 244))
GROUP_NAMES = ('bond_order', 'tda', 'density', 'velocity')


def transform(raw, bank):
    """Training-fitted affine/asinh/affine transform, without clipping."""
    mean, scale, raw_mean, raw_scale = [bank[k].to(raw) for k in
                                      ('mean', 'scale', 'raw_mean', 'raw_scale')]
    return (torch.asinh((raw-raw_mean)/raw_scale)-mean)/scale


def inverse(x, bank):
    """Exact mathematical inverse; excursions are retained, never clipped."""
    mean, scale, raw_mean, raw_scale = [bank[k].to(x) for k in
                                      ('mean', 'scale', 'raw_mean', 'raw_scale')]
    return torch.sinh(x*scale+mean)*raw_scale+raw_mean


def _source(c):
    source_path = resolve_path(c['source_run'])/'technical/plan.json'
    if sha(source_path) != c['source_plan_sha256']:
        raise ValueError('Changed native measurement plan')
    source = read(source_path)
    path = cache(source['config'])/'data.pt'
    receipt = read(path.with_suffix('.json'))
    if receipt['identity'] != source['identity'] or sha(path) != receipt['sha256']:
        raise ValueError('Changed native measurement bank')
    return source, path, receipt


def _birth_source(c, columns):
    history_path = resolve_path(c['birth_history_run'])/'technical/plan.json'
    if sha(history_path) != c['birth_history_plan_sha256']:
        raise ValueError('Changed birth history plan')
    history = read(history_path)
    directory = cache(history['config'])
    cohort_path = directory/'births.pt'
    measurement_path = directory/'birth-measurements/measurements.pt'
    cohort_hash = read(directory/'complete.json')['files']['births.pt']
    measured = read(measurement_path.with_suffix('.json'))
    if sha(cohort_path) != cohort_hash or sha(measurement_path) != measured['sha256']:
        raise ValueError('Changed frozen birth measurements or rows')
    if measured['columns'] != columns:
        raise ValueError('Native and birth measurement columns disagree')
    cohort = torch.load(cohort_path, map_location='cpu', weights_only=False)
    roles = {role: sorted({r['ancestor'] for r in cohort['rows'] if r['role'] == role})
             for role in ('train', 'test')}
    if set(roles['train']) & set(roles['test']):
        raise ValueError('Birth ancestry leakage')
    return dict(plan_sha256=c['birth_history_plan_sha256'],
                cohort_path=str(cohort_path), cohort_sha256=cohort_hash,
                measurement_path=str(measurement_path), measurement_sha256=measured['sha256'],
                rows=len(cohort['rows']), unique_states=len(cohort['state']),
                roles=dict(Counter(r['role'] for r in cohort['rows'])), ancestors=roles,
                declared_ancestors=history['partition'],
                support='present nearest80 atom IDs held fixed through four past frames; native training reselects nearest80 at each frame')


def prepare(c):
    source, path, receipt = _source(c)
    source_by_key = {s['key']: s for s in source['sources']}
    rows = [dict(r, original_role=r['role'],
                 source_original_role=source_by_key[r['source']]['original_role'],
                 role='train' if r['role'] == 'train' else 'test') for r in source['rows']]
    ancestors = {role: sorted({r['ancestor'] for r in rows if r['role'] == role})
                 for role in ('train', 'test')}
    if set(ancestors['train']) & set(ancestors['test']):
        raise ValueError('Native ancestry leakage')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicated native window identity')
    timelines = []
    for item in source['sources']:
        s = item['source']
        for part in s['parts']:
            saved = arrays(part['path'], part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(saved)*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Actual native cadence changed: '+s['id'])
        times = np.asarray(s['times_ps'])
        for job in source['jobs']:
            if job['source'] == item['key']:
                frame = job['frame']
                if not np.allclose(times[np.arange(frame-3, frame+61)]-times[frame],
                                   np.arange(-3, 61)*.1, atol=1e-8, rtol=0):
                    raise ValueError('Exact native window missing: '+job['key'])
        timelines.append(dict(source=s['id'], input_identity=s['input_identity'],
                              frames=len(times), spacing_ps=.1))
    birth = _birth_source(c, source['columns'])
    if set(ancestors['train']) & set(birth['declared_ancestors']['test']):
        raise ValueError('Native representation training leaks into birth test ancestry')
    p = dict(config=c, source_plan_sha256=c['source_plan_sha256'],
             source_identity=source['identity'], source_data_path=str(path),
             source_data_sha256=receipt['sha256'], rows=rows, columns=source['columns'],
             sources=source['sources'], jobs=source['jobs'], timelines=timelines,
             roles=dict(Counter(r['role'] for r in rows)), ancestors=ancestors, birth=birth,
             offsets_ps=(np.arange(-3, 61)/10).tolist(),
             split='train/test only; original fitting ancestry preserved; former selection is test',
             normalization='raw training mean/std, asinh, transformed training mean/std; floors 1e-4; no clipping',
             normalization_implementation_sha256=sha(Path(__file__)),
             sampling='4800 original +4800 future-selected liquid-liquid +4800 future-selected growing-front; not natural occupancy',
             native_support='same central atom ID; nearest80 membership reselected independently at each native frame',
             birth_input_policy='encoder and reconstructed measurement features use present Y only; cached observed history is an explicit readout control; no neural birth rollout uses the mismatched fixed-membership past')
    p['data_identity'] = digest([p['source_data_sha256'], p['normalization'],
                                 p['normalization_implementation_sha256'], ancestors])
    p['identity'] = digest(p)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != p:
        raise ValueError('Changed kinetic latent plan; use a new revision')
    write_json(destination, p)
    marker = cache(c)/'data.json'
    if marker.exists():
        done = read(marker)
        if done['identity'] != p['data_identity'] or sha(cache(c)/'data.pt') != done['sha256']:
            raise ValueError('Changed normalized native bank')
        return p
    original = torch.load(path, map_location='cpu', weights_only=False)
    raw = original['x']
    train = np.flatnonzero([r['role'] == 'train' for r in rows])
    test = np.flatnonzero([r['role'] == 'test' for r in rows])
    values = raw[train].flatten(0, 1).double()
    raw_mean = values.mean(0)
    raw_std = values.std(0)
    raw_scale = raw_std.clamp_min(1e-4)
    values = torch.asinh((values-raw_mean)/raw_scale)
    mean, scale = values.mean(0), values.std(0).clamp_min(1e-4)
    active = raw_std > 1e-8
    weights = torch.zeros(244)
    for start, stop in GROUPS:
        a = active[start:stop].float()
        weights[start:stop] = a/a.sum().clamp_min(1)/len(GROUPS)
    bank = dict(mean=mean.float(), scale=scale.float(), raw_mean=raw_mean.float(),
                raw_scale=raw_scale.float(), active=active, weights=weights,
                rows=rows, columns=p['columns'], train_rows=train, test_rows=test)
    bank['x'] = transform(raw, bank)
    if not torch.isfinite(bank['x']).all():
        raise FloatingPointError('Nonfinite transformed native measurements')
    atomic_save(cache(c)/'data.pt', bank)
    write_json(marker, dict(identity=p['data_identity'], sha256=sha(cache(c)/'data.pt'),
                            shape=list(bank['x'].shape), train_rows=len(train), test_rows=len(test)))
    return p


def load(c, device='cpu'):
    p = read(root(c)/'technical/plan.json')
    path = cache(c)/'data.pt'
    done = read(path.with_suffix('.json'))
    if done['identity'] != p['data_identity'] or sha(path) != done['sha256']:
        raise ValueError('Changed prepared native measurements')
    bank = torch.load(path, map_location=device, weights_only=False)
    return p, bank


def load_births(c, bank):
    p = read(root(c)/'technical/plan.json'); spec = p['birth']
    for key in ('cohort', 'measurement'):
        if sha(spec[key+'_path']) != spec[key+'_sha256']:
            raise ValueError('Changed birth '+key)
    cohort = torch.load(spec['cohort_path'], map_location='cpu', weights_only=False)
    raw = torch.load(spec['measurement_path'], map_location='cpu', weights_only=False)
    return cohort, transform(raw.reshape(-1, 4, 244), bank)


def annotate(c, deadline=float('inf')):
    """Exact PTM diagnostics; never a feature, filtering rule or training target.

    Per-source checkpoints make this independent CPU stage resumable. Only the
    established Labels producer may fill missing atomic PTM; no phase is inferred
    from minimum-size connected components.
    """
    from src.research.ni_latent_sim.enrichment import Labels
    p = read(root(c)/'technical/plan.json')
    source = read(resolve_path(c['source_run'])/'technical/plan.json')
    enrichment = read(resolve_path(source['config']['source_run'])/'technical/plan.json')
    contract = dict(plan=p['data_identity'], ptm_crystal_types=[1, 2, 3],
                    ptm_rmsd_cutoff=.1, nearest=80, normalized_radius_A=8.,
                    length_factor=source['config']['length_factor'],
                    cluster_zero='not in a tracked component of at least4; not a liquid classification',
                    implementation_sha256=sha(Path(__file__)))
    identity = digest(contract); marker = cache(c)/'labels.json'
    if marker.exists():
        done = read(marker)
        if done['identity'] != identity or sha(cache(c)/'labels.pt') != done['sha256']:
            raise ValueError('Changed physical annotations')
        write_json(root(c)/'technical/labels-complete.json', dict(
            state='complete', identity=identity, sha256=done['sha256']))
        return True
    grouped = defaultdict(list)
    for i, row in enumerate(p['rows']):
        grouped[row['source']].append(i)
    begin = time.monotonic()
    for item in enrichment['sources']:
        key = item['key']; selected = grouped[key]
        folder = cache(c)/'annotations'/key; receipt = folder/'complete.json'
        local_identity = digest([identity, key, selected])
        if receipt.exists():
            done = read(receipt)
            if done['identity'] != local_identity or sha(folder/'labels.pt') != done['sha256']:
                raise ValueError('Changed source annotations: '+key)
            continue
        labels = Labels(enrichment['config'], enrichment, item)
        ids = arrays(item['source']['parts'][0]['path'], item['source']['parts'][0]['manifest_sha256'])['atom_ids']
        requests = defaultdict(list)
        for local_row, i in enumerate(selected):
            row = p['rows'][i]
            for t, frame in enumerate(range(row['frame']-3, row['frame']+61)):
                requests[frame].append((local_row, t, row['center']))
        central = np.empty((len(selected), 64), dtype=np.uint8)
        fraction = np.empty((len(selected), 64), dtype=np.float32)
        component = np.empty((len(selected), 64), dtype=np.int32)
        progress_path = folder/'partial.pt'
        start = -1
        if progress_path.exists():
            previous = torch.load(progress_path, map_location='cpu', weights_only=False)
            if previous['identity'] != local_identity:
                raise ValueError('Changed resumed annotation identity: '+key)
            central, fraction, component = [previous[k] for k in ('ptm', 'fraction', 'component_size')]
            start = previous['frame']; labels.used.update(previous['provenance'])
        def checkpoint(frame):
            atomic_save(progress_path, dict(identity=local_identity, frame=frame, ptm=central,
                fraction=fraction, component_size=component, provenance=labels.used))
        for n, frame in enumerate(sorted(requests)):
            if frame <= start:
                continue
            if time.time() > deadline-120:
                checkpoint(start)
                return False
            request = requests[frame]
            centers = np.array([r[2] for r in request], dtype=np.int64)
            index = np.searchsorted(ids, centers)
            if not np.array_equal(ids[index], centers):
                raise ValueError('Fixed center missing from annotation frame: '+key)
            ptm = labels.get(frame)
            clusters = labels.reader.labels(frame)
            counts = np.bincount(clusters)
            counts[0] = 0
            q, box = geometry(item['source'], frame)
            distance, near = cKDTree(q, boxsize=box).query(q[index], k=80, workers=1)
            support = distance*contract['length_factor'] < contract['normalized_radius_A']
            local_fraction = (np.isin(ptm[near], [1, 2, 3])*support).sum(1)/support.sum(1)
            rr, tt = np.array([(r[0], r[1]) for r in request]).T
            central[rr, tt] = ptm[index]
            fraction[rr, tt] = local_fraction
            component[rr, tt] = counts[clusters[index]]
            start = frame
            if n % 64 == 0:
                checkpoint(frame)
                write_json(root(c)/'technical/annotation-progress.json', dict(source=key,
                    frames_done=n+1, frames_total=len(requests), seconds=time.monotonic()-begin))
        packet = dict(indices=selected, ptm=torch.from_numpy(central),
                      crystal=torch.from_numpy(np.isin(central, [1, 2, 3])),
                      fraction=torch.from_numpy(fraction), component_size=torch.from_numpy(component))
        atomic_save(folder/'labels.pt', packet)
        write_json(receipt, dict(identity=local_identity, sha256=sha(folder/'labels.pt'),
                                rows=len(selected), provenance=labels.used))
        progress_path.unlink(missing_ok=True)
        print('annotated', key, len(selected), 'windows', flush=True)
    result = dict(ptm=torch.empty(len(p['rows']), 64, dtype=torch.uint8),
                  crystal=torch.empty(len(p['rows']), 64, dtype=torch.bool),
                  fraction=torch.empty(len(p['rows']), 64),
                  component_size=torch.empty(len(p['rows']), 64, dtype=torch.int32))
    provenance = {}
    for key in grouped:
        folder = cache(c)/'annotations'/key
        packet = torch.load(folder/'labels.pt', map_location='cpu', weights_only=False)
        for field in result:
            result[field][packet['indices']] = packet[field]
        provenance[key] = sha(folder/'complete.json')
    atomic_save(cache(c)/'labels.pt', result)
    write_json(marker, dict(identity=identity, sha256=sha(cache(c)/'labels.pt'),
                           contract=contract, sources=provenance, shape=[len(p['rows']), 64],
                           elapsed_seconds=time.monotonic()-begin))
    write_json(root(c)/'technical/labels-complete.json', dict(
        state='complete', identity=identity, sha256=sha(cache(c)/'labels.pt')))
    return True
