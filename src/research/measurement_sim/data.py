"""Pinned physical observations; no learned encoder or broad spatial dataset."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch
from src.experiment_runner.artifacts import file_hash as sha, write_json, json_digest as digest
from src.project_runtime.paths import resolve_path
from src.research.continuous_motion.data import Source, read, atomic_save
from src.postprocessing.io import arrays
from .features import SmoothMeasurements, structural


def root(c): return resolve_path(c['output'])
def cache(c): return resolve_path(c['cache'])


def prepare(c):
    parent_path = resolve_path(c['source_run'])/'technical/plan.json'
    if sha(parent_path) != c['source_plan_sha256']:
        raise ValueError('Changed parent sequence plan')
    old = read(parent_path)
    initial = read(resolve_path(old['config']['enrichment']['base_output'])/'technical/plan.json')
    selected_path = parent_path.parent/'selected-sequences.json'
    selected = read(selected_path)
    source = {x['key']: x for x in old['sources']}
    rows = []
    # Match the original producer's identity serialization exactly.
    from src.research.ni_latent_sim.data import digest as row_digest
    for job in initial['jobs']:
        s = source[job['source_key']]
        for center in job['centers']:
            rows.append(dict(id=row_digest([job['source_key'], job['frame'], center]),
                source=job['source_key'], ancestor=s['source']['ancestry_group'],
                role=s['role'], frame=job['frame'], center=center, regime='original'))
    rows.extend(selected['rows'])
    if len(rows) != 14400 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Changed fixed-atom cohort identity')
    roles = defaultdict(set)
    for row in rows: roles[row['role']].add(row['ancestor'])
    if any(roles[a] & roles[b] for a in roles for b in roles if a != b):
        raise ValueError('Ancestry leakage')
    for item in old['sources']:
        s = item['source']
        for part in s['parts']:
            steps = arrays(part['path'], part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(steps)*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Changed exact native cadence: '+s['id'])
    groups = defaultdict(list)
    for index, row in enumerate(rows): groups[(row['source'], row['frame'])].append(index)
    jobs = []
    for (key, frame), indices in groups.items():
        times = np.asarray(source[key]['source']['times_ps'])
        if not np.allclose(times[np.arange(frame-3, frame+61)]-times[frame],
                           np.arange(-3, 61)*.1, atol=1e-8, rtol=0):
            raise ValueError('Missing exact 6ps fixed-atom window')
        jobs.append(dict(key=digest([key, frame])[:20], source=key, frame=frame, rows=indices))
    selection = read(resolve_path(c['structural_columns']))
    columns = [v['name'] for v in selection['columns'] if v['family'] != 'cna']
    columns += SmoothMeasurements().columns
    plan = dict(config=c, sources=old['sources'], rows=rows, jobs=jobs,
        columns=columns, parent_identity=old['identity'], parent_plan_sha256=sha(parent_path),
        selected_rows_sha256=sha(selected_path), partition=old['partition'],
        roles=dict(Counter(r['role'] for r in rows)), structural_columns_sha256=sha(resolve_path(c['structural_columns'])))
    plan['identity'] = digest(plan)
    path = root(c)/'technical/plan.json'
    if path.exists() and read(path) != plan: raise ValueError('Changed measurement recipe; new revision required')
    write_json(path, plan)
    write_json(root(c)/'technical/prediction-context.json', dict(
        input='four native0.1ps observations of 227 structural +17 velocity measurements',
        spatial='fixed central atom, nearest80 reselected at native times; no extra phase/neighborhood filtering',
        targets='BO40/TDA113 exact original fixed measurements; smooth density74 and velocity17; no eSEN',
        state='full244 measurements; initial observation exact; transformer hidden projection is not a state bottleneck',
        conditions='no temperature, source age, absolute time, box, species or phase inputs',
        teacher='separate saved fixed-box Ni BAOAB paths/tangents every5fs; native long paths remain NPT',
        sampling='existing phase/future-selected liquid/front enrichment retained; not natural population frequencies'))
    return plan


def geometry_job(c, p, job):
    """CPU producer: cache physical Q,V and exact historical BO/TDA values."""
    folder = cache(c)/'native'/job['key']; done = folder/'geometry.json'
    if done.exists(): return read(done)
    item = next(s for s in p['sources'] if s['key'] == job['source'])
    reader = Source(item['source'])
    centers = [p['rows'][i]['center'] for i in job['rows']]
    states = np.stack([reader.patches(f, centers)[0] for f in range(job['frame']-3, job['frame']+61)], 1).astype(np.float32)
    flat = states.reshape(-1, 80, 6)
    # No row exclusion: lack of geometric support is a preparation error.
    if (np.linalg.norm(flat[:, 1:, :3]-flat[:, :1, :3], axis=-1).max(-1) <= 5.5).any():
        raise ValueError('nearest80 truncates smooth support: '+job['key'])
    begin = time.monotonic()
    values = structural(flat[..., :3], p['columns'][:153], c['length_factor'])
    atomic_save(folder/'geometry.pt', dict(state=torch.from_numpy(states),
        structural=torch.from_numpy(values.reshape(len(centers), 64, 153)), rows=job['rows']))
    receipt = dict(identity=digest([p['identity'], job]), sha256=sha(folder/'geometry.pt'),
                   rows=len(centers), seconds=time.monotonic()-begin)
    write_json(done, receipt); return receipt


def native(c, p, workers):
    jobs = p['jobs']; begin = time.monotonic(); complete = 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(geometry_job, c, p, j): j for j in jobs}
        for task in as_completed(pending):
            result = task.result(); complete += 1
            write_json(root(c)/'technical/native-progress.json', dict(state='preparing',
                groups_done=complete, groups_total=len(jobs), elapsed_seconds=time.monotonic()-begin,
                last=pending[task]['key'], last_seconds=result['seconds']))
            if complete % 10 == 0: print('native geometry', complete, '/', len(jobs), flush=True)
    write_json(root(c)/'technical/native-geometry-complete.json', dict(groups=complete))


def gpu_native(c, p):
    device = SmoothMeasurements().cuda().eval()
    computed = torch.compile(device, fullgraph=True)
    result = torch.empty((len(p['rows']), 64, 244))
    for number, job in enumerate(p['jobs']):
        folder = cache(c)/'native'/job['key']
        record = read(folder/'geometry.json')
        if record['identity'] != digest([p['identity'], job]) or sha(folder/'geometry.pt') != record['sha256']:
            raise ValueError('Changed fixed measurements')
        packet = torch.load(folder/'geometry.pt', weights_only=False, map_location='cpu')
        x = packet['state'].flatten(0, 1)
        with torch.no_grad():
            smooth = torch.cat([computed(v.cuda()).cpu() for v in x.split(256)])
        result[packet['rows']] = torch.cat((packet['structural'], smooth.reshape(-1, 64, 91)), -1)
        if number % 20 == 0: print('smooth native', number, '/', len(p['jobs']), flush=True)
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite direct measurements')
    train = torch.tensor([r['role'] == 'train' for r in p['rows']])
    values = result[train].flatten(0, 1).double()
    mean = values.mean(0).float(); scale = values.std(0).float().clamp_min(1e-4)
    normalized = (result[train]-mean)/scale
    # Per-channel OU initialization, fitted only on training observations.
    rho = (normalized[:, :-1]*normalized[:, 1:]).mean((0, 1))/normalized[:, :-1].square().mean((0, 1)).clamp_min(1e-5)
    rate = -rho.clamp(.01, .995).log()/.1*(scale > 1e-4)
    delta = normalized[:, 1:]-normalized[:, :-1]
    increment_scale = delta.square().mean((0, 1)).sqrt().clamp_min(.05)
    atomic_save(cache(c)/'data.pt', dict(x=result, mean=mean, scale=scale,
                rate=rate, increment_scale=increment_scale, rows=p['rows'], columns=p['columns']))
    write_json(cache(c)/'data.json', dict(identity=p['identity'], sha256=sha(cache(c)/'data.pt')))
    write_json(root(c)/'technical/native-complete.json', dict(state='complete', rows=len(result), roles=p['roles']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'native', 'physical', 'gpu'])
    parser.add_argument('--config', required=True); parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(); c = read(args.config)
    torch.set_num_threads(1)
    if args.stage == 'prepare': print(prepare(c)['identity'])
    else:
        p = read(root(c)/'technical/plan.json')
        if args.stage == 'native': native(c, p, args.workers)
        elif args.stage == 'physical':
            from .physical import collect
            collect(c, p)
        else: gpu_native(c, p)


if __name__ == '__main__': main()
