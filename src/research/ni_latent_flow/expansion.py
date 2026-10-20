"""Outcome-blind native-window expansion from the thirty additional Ni sources.

The original measurements, normalizers, and source roles remain immutable.
New windows retain one central atom across all 64 observations; nearest-80
membership is reselected at each actual native frame, as in the parent cohort.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.postprocessing.extraction import Frames
from src.postprocessing.io import arrays, geometry
from src.research.continuous_motion.data import Source
from src.research.measurement_sim import data as measured
from src.research.measurement_sim.features import SmoothMeasurements
from src.research.ni_kinetic_latent import data as native
from .data import read, root, cache, resolve_path, sha, digest, write_json, atomic_save


def directory(c):
    return resolve_path(c['expansion']['cache'])


def prepare(c):
    spec = c['expansion']
    parent_path = resolve_path(c['base_run'])/'technical/plan.json'
    if sha(parent_path) != c['base_plan_sha256']:
        raise ValueError('Changed original native plan')
    parent = read(parent_path)
    inventory_path = resolve_path(spec['source_plan'])
    if sha(inventory_path) != spec['source_plan_sha256']:
        raise ValueError('Changed expanded source inventory')
    inventory = read(inventory_path)
    old_sources = {s['key'] for s in parent['sources']}
    sources = [s for s in inventory['sources'] if s['key'] not in old_sources]
    rows, jobs, timelines = [], [], []
    for item in sources:
        s = item['source']; reader = Source(s)
        for part in s['parts']:
            steps = arrays(part['path'], part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(steps)*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Not native 0.1 ps: '+s['id'])
        times = np.asarray(s['times_ps'])
        eligible = np.flatnonzero((times >= 20.) & (np.arange(len(times)) >= 3)
                                  & (np.arange(len(times)) + 60 < len(times)))
        rng = np.random.default_rng(int(digest([spec['seed'], s['id'], 'frames'])[:12], 16))
        # Random positions in equal-duration strata avoid future-label selection.
        frame_groups = np.array_split(eligible, spec['anchors_per_source'])
        if any(len(group) == 0 for group in frame_groups):
            raise ValueError('Insufficient complete source history: '+s['id'])
        frames = [int(rng.choice(group)) for group in frame_groups]
        role = 'train' if item['original_role'] == 'train' else 'test'
        for frame in frames:
            offsets = times[np.arange(frame-3, frame+61)]-times[frame]
            if not np.allclose(offsets, np.arange(-3, 61)*.1, atol=1e-8, rtol=0):
                raise ValueError('Incomplete exact native window: '+s['id'])
            centers = rng.choice(reader.ids, spec['centers_per_anchor'], replace=False)
            selected = []
            for center in centers:
                selected.append(len(rows))
                rows.append(dict(id=digest([s['id'], frame, int(center)]), source=s['id'],
                    ancestor=s['ancestry_group'], role=role, original_role=item['original_role'],
                    source_original_role=item['original_role'], frame=frame, center=int(center),
                    regime='original', cohort='additional30',source_cohort='additional30'))
            jobs.append(dict(key=digest([s['id'], frame])[:20], source=s['id'], frame=frame, rows=selected))
        timelines.append(dict(source=s['id'], frames=len(times), cadence_ps=.1,
                              selected_frames=frames, input_identity=s['input_identity']))
    if len(rows) != len(sources)*spec['windows_per_source'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expanded cohort cardinality or identity differs')
    old_measurement = read(resolve_path(parent['config']['source_run'])/'technical/plan.json')
    configuration = dict(old_measurement['config'], cache=str(directory(c)))
    plan = dict(config=configuration, expansion=spec, sources=sources, rows=rows, jobs=jobs,
        columns=parent['columns'], base_config=parent['config'], base_identity=parent['identity'],
        source_plan_sha256=spec['source_plan_sha256'], timelines=timelines,
        roles=dict(Counter(r['role'] for r in rows)),
        sampling='Fifteen equal-duration strata after20ps; seeded random anchor and16 uniform atom IDs per stratum; no future or phase selection; native stopping rules retained',
        annotation='Exact existing PTM1/2/3 and historical nearest80/radius8 normalized local fraction; not a birth filter',
        normalization='Reuse original16-source affine/asinh/affine normalizers unchanged; no transform fitting on added sources or test',
        implementation_sha256=sha(Path(__file__)))
    plan['identity'] = digest(plan)
    path = root(c)/'technical/expansion-plan.json'
    if path.exists() and read(path) != plan:
        raise ValueError('Changed frozen expansion plan; use a new revision')
    write_json(path, plan)
    return plan


def _source_labels(c, p, item):
    """Reuse cached full-cell PTM, preserving the parent's exact annotation rule."""
    folder = directory(c)/'annotations'/item['key']; marker = folder/'complete.json'
    indices = [i for i, r in enumerate(p['rows']) if r['source'] == item['key']]
    identity = digest([p['identity'], item['key'], indices])
    if marker.exists():
        done = read(marker)
        if done['identity'] != identity or sha(folder/'labels.pt') != done['sha256']:
            raise ValueError('Changed expanded labels: '+item['key'])
        return done
    extraction = read(Path(item['search'])/'technical/plans.json')['parent']
    if extraction['config']['ptm_rmsd_cutoff'] != .1:
        raise ValueError('Changed inherited PTM cutoff')
    reader = Frames(extraction, item['source'], 'ptm')
    ids = arrays(item['source']['parts'][0]['path'], item['source']['parts'][0]['manifest_sha256'])['atom_ids']
    requests = defaultdict(list)
    for rr, index in enumerate(indices):
        row = p['rows'][index]
        for tt, frame in enumerate(range(row['frame']-3, row['frame']+61)):
            requests[frame].append((rr, tt, row['center']))
    central = np.empty((len(indices),64), np.uint8)
    fraction = np.empty((len(indices),64), np.float32)
    component = np.empty((len(indices),64), np.int32)
    provenance = {}; begin = time.monotonic()
    for number, (frame, request) in enumerate(sorted(requests.items())):
        rr, tt, centers = np.asarray(request).T
        selected = np.searchsorted(ids, centers)
        if not np.array_equal(ids[selected], centers):
            raise ValueError('Missing fixed center: '+item['key'])
        clusters = reader.labels(frame)
        ptm = reader.raw_ptm(frame)  # Never infer atomic phase from tracked components.
        provenance[str(reader.path)] = reader.checksum
        counts = np.bincount(clusters); counts[0] = 0
        q, box = geometry(item['source'], frame)
        distance, near = cKDTree(q, boxsize=box).query(q[selected], k=80, workers=1)
        support = distance*p['config']['length_factor'] < 8.
        central[rr,tt] = ptm[selected]
        fraction[rr,tt] = (np.isin(ptm[near],[1,2,3])*support).sum(1)/support.sum(1)
        component[rr,tt] = counts[clusters[selected]]
        if number % 64 == 0:
            write_json(folder/'progress.json',dict(frames_done=number+1,frames_total=len(requests),
                seconds=time.monotonic()-begin))
    packet = dict(indices=indices,ptm=torch.from_numpy(central),
        crystal=torch.from_numpy(np.isin(central,[1,2,3])),fraction=torch.from_numpy(fraction),
        component_size=torch.from_numpy(component))
    atomic_save(folder/'labels.pt',packet)
    receipt = dict(identity=identity,sha256=sha(folder/'labels.pt'),rows=len(indices),
                   seconds=time.monotonic()-begin,provenance=provenance)
    write_json(marker,receipt)
    return receipt


def collect(c, workers=12, device='cuda'):
    """Resumable per-group measurement collection and per-source cached labels."""
    p = prepare(c); folder = directory(c); marker = root(c)/'technical/expansion-complete.json'
    if marker.exists():
        done = read(marker)
        if done['identity'] != p['identity'] or any(sha(folder/name) != value for name,value in done['files'].items()):
            raise ValueError('Changed completed expansion')
        return done
    torch.set_num_threads(1)
    begin = time.monotonic()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(measured.geometry_job,p['config'],p,j):j for j in p['jobs']}
        for count,future in enumerate(as_completed(pending),1):
            future.result()
            write_json(root(c)/'technical/expansion-progress.json',dict(stage='geometry',done=count,
                total=len(p['jobs']),seconds=time.monotonic()-begin))
    transform = SmoothMeasurements().to(device).eval()
    raw = torch.empty(len(p['rows']),64,244)
    for count,job in enumerate(p['jobs'],1):
        path = folder/'native'/job['key']/'geometry.pt'
        receipt = read(path.with_suffix('.json'))
        if receipt['identity'] != digest([p['identity'],job]) or sha(path) != receipt['sha256']:
            raise ValueError('Changed prepared geometry')
        packet = torch.load(path,map_location='cpu',weights_only=False)
        with torch.no_grad():
            smooth = torch.cat([transform(v.to(device)).cpu() for v in packet['state'].flatten(0,1).split(512)])
        raw[packet['rows']] = torch.cat((packet['structural'],smooth.reshape(-1,64,91)),-1)
        if count%20 == 0:
            write_json(root(c)/'technical/expansion-progress.json',dict(stage='smooth',done=count,
                total=len(p['jobs']),seconds=time.monotonic()-begin))
    if not torch.isfinite(raw).all():
        raise FloatingPointError('Nonfinite expanded measurements')
    atomic_save(folder/'raw.pt',dict(x=raw,rows=p['rows'],columns=p['columns'],identity=p['identity']))
    _,bank = native.load(p['base_config'])
    x = native.transform(raw,bank)
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite normalized expanded measurements')
    atomic_save(folder/'data.pt',dict(x=x,rows=p['rows'],columns=p['columns'],identity=p['identity']))
    del raw,x,bank,transform
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(_source_labels,c,p,item):item['key'] for item in p['sources']}
        for count,future in enumerate(as_completed(pending),1):
            future.result()
            write_json(root(c)/'technical/expansion-progress.json',dict(stage='labels',done=count,
                total=len(p['sources']),seconds=time.monotonic()-begin))
    base_labels = torch.load(native.cache(p['base_config'])/'labels.pt',map_location='cpu',weights_only=False)
    extra = {k:torch.empty((len(p['rows']),64),dtype=v.dtype)for k,v in base_labels.items()}
    for item in p['sources']:
        packet = torch.load(folder/'annotations'/item['key']/'labels.pt',map_location='cpu',weights_only=False)
        for key in extra: extra[key][packet['indices']] = packet[key]
    atomic_save(folder/'labels.pt',{key:torch.cat((base_labels[key],extra[key]))for key in extra})
    result = dict(identity=p['identity'],state='complete',rows=len(p['rows']),roles=p['roles'],
        files={name:sha(folder/name)for name in ('raw.pt','data.pt','labels.pt')},seconds=time.monotonic()-begin)
    write_json(marker,result)
    return result


def load_expanded(c, arm=None):
    """Common heldout rows; training subsets are explicit per-arm contracts."""
    p = read(root(c)/'technical/plan.json')
    extension = read(root(c)/'technical/expansion-plan.json')
    done = read(root(c)/'technical/expansion-complete.json')
    if done['identity'] != extension['identity'] or sha(directory(c)/'data.pt') != done['files']['data.pt']:
        raise ValueError('Unsealed expanded native measurements')
    _,bank = native.load(p['base_config'])
    old_count = len(bank['rows'])
    added = torch.load(directory(c)/'data.pt',map_location='cpu',weights_only=False)
    bank['x'] = torch.cat((bank['x'],added['x']))
    bank['input'] = bank['x']
    bank['rows'] = [dict(r,cohort='old60',source_cohort='old60')for r in bank['rows']]+added['rows']
    cohort = arm['training_cohort'] if arm is not None else 'old_mixed16'
    allowed = [i for i,r in enumerate(bank['rows']) if r['role']=='train' and
               ((cohort=='old_mixed16' and i<old_count) or
                (cohort=='original16' and i<old_count and r['regime']=='original') or
                (cohort=='original28' and r['regime']=='original'))]
    if cohort not in ('old_mixed16','original16','original28'):
        raise ValueError('Unknown declared training cohort: '+cohort)
    if cohort=='original28':
        groups = defaultdict(list)
        for index in allowed: groups[bank['rows'][index]['ancestor']].append(index)
        allowed = [index for source in sorted(groups) for index in
                   sorted(groups[source],key=lambda i:digest([c['expansion']['seed'],bank['rows'][i]['id'],'training80']))[:80]]
    bank['train_rows'] = np.asarray(allowed,dtype=np.int64)
    bank['test_rows'] = np.asarray([i for i,r in enumerate(bank['rows'])if r['role']=='test'],dtype=np.int64)
    return p,bank


def labels_path(c):
    path = directory(c)/'labels.pt'
    if sha(path) != read(root(c)/'technical/expansion-complete.json')['files']['labels.pt']:
        raise ValueError('Changed expanded labels')
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    print(collect(read(args.config),args.workers,args.device),flush=True)


if __name__ == '__main__':
    main()
