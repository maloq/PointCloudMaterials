"""Native 0.1-ps local trajectories; fixed eSEN observations and physical ancestry."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from src.research.ni_response.data import Source, read, root, cache, sha, digest, write_json, atomic_save, resolve_path
from src.research.ni_response.model import Motion
from src.research.response_geometry.features import observed
from src.postprocessing.io import arrays

FAMILY = 'ni_latent_sim'


def parent(c):
    path = Path(c['parent']) / 'technical/plan.json'
    if sha(path) != c['parent_plan_sha256']:
        raise ValueError('Changed parent Ni source/sample contract')
    return read(path)


def prepare(c):
    p = parent(c)
    if sha(resolve_path(p['config']['checkpoint'])) != p['config']['checkpoint_sha256']:
        raise ValueError('Changed frozen eSEN checkpoint')
    sources = {item['key']: item for item in p['sources']}
    for item in sources.values():
        s = item['source']
        for part in s['parts']:
            actual = arrays(part['path'], part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(actual)*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Changed actual native cadence: '+s['id'])
    jobs = []
    for j in p['jobs']:
        item = sources[j['source_key']]
        s = item['source']; times = np.asarray(s['times_ps'])
        frames = np.arange(j['frame']-3, j['frame']+61)
        if frames.min() < 0 or frames.max() >= len(times) or not np.allclose(times[frames]-times[j['frame']], np.arange(-3,61)*.1, atol=1e-8, rtol=0):
            raise ValueError('Missing exact history/future window: '+j['key'])
        teacher = item['centers'][j['teacher_center']]
        centers = [teacher] + [a for a in item['centers'] if a != teacher][:c['centers_per_anchor']-1]
        jobs.append(dict(j, centers=centers, frames=frames.tolist(), role=item['role'], ancestor=s['ancestry_group']))
    plan = dict(config=c, parent_identity=p['identity'], jobs=jobs, partition=p['partition'], sources=p['sources'])
    plan['identity'] = digest(plan)
    path = root(c)/'technical/plan.json'
    if path.exists() and read(path) != plan: raise ValueError('Changed latent experiment; use a new revision')
    write_json(path, plan)
    write_json(root(c)/'technical/prediction-context.json', dict(
        encoder='frozen unrelaxed geometry-only eSEN(Q), velocity secant 0.1ps DE(Q)V, 24 radial/angular observables; then learned bottleneck',
        predictor='one code or four codes at exact 0.1ps offsets; autonomous stochastic transition; no temperature, age, time, material, box or species inputs',
        spatial_support='nearest80 centered at the same atom identity; membership is recomputed each saved frame; fixed material length normalization',
        teachers='native NPT sequences; separate 100fs fixed-box BAOAB values/analytic derivatives from pinned prior bank',
        scope='local observed-process surrogate, not a full-cell autonomous MD replacement; NPT cell/thermostat and outside environment are unresolved variables',
        reconstruction='fixed observation-vector decoder, no atomic reconstruction loss or broad spatial replay',
        history='[-0.3,-0.2,-0.1,0]ps; no future input; history1 uses only zero',
        selection='label-free validation NLL on four original training ancestors; evaluation pools original held-out roles'))
    return plan


def observation(model, qv, micro):
    result = []
    with torch.no_grad():
        for state in qv.split(micro):
            state = state.float().cuda()
            result.append(torch.cat((model.base_encode(state, create_graph=False), observed(state[...,:3])), -1).cpu())
    return torch.cat(result)


def collect(c, p, lane, deadline):
    pc = parent(c)['config']; model = Motion(pc).cuda().eval().requires_grad_(False)
    micro = c['encoder_microbatch']; directory = cache(c)/'sequences'; directory.mkdir(parents=True, exist_ok=True)
    sources = {item['key']: item for item in p['sources']}
    for i, job in enumerate(p['jobs'][lane::2]):
        f = directory/(job['key']+'.pt'); receipt = f.with_suffix('.json')
        if receipt.exists():
            saved = read(receipt)
            if saved['identity'] != digest([p['identity'],job]) or sha(f) != saved['sha256']: raise ValueError('Changed sequence packet')
            continue
        if time.time() > deadline-600: raise TimeoutError('Sequence preparation reached deadline')
        reader = Source(sources[job['source_key']]['source'])
        states = np.stack([reader.patches(frame, job['centers'])[0] for frame in job['frames']],1)
        qv = torch.from_numpy(states*pc['length_factor']).float()
        x = observation(model, qv.flatten(0,1), micro).reshape(len(job['centers']),64,280)
        rows = [dict(id=digest([job['source_key'],job['frame'],a]),source=job['source_key'],ancestor=job['ancestor'],role=job['role'],frame=job['frame'],center=a,job=job['key']) for a in job['centers']]
        atomic_save(f,dict(x=x,rows=rows))
        write_json(receipt,dict(identity=digest([p['identity'],job]),sha256=sha(f)))
        write_json(root(c)/'technical'/f'data-{lane}.json',dict(stage='dense_sequences',done=i+1,total=len(p['jobs'][lane::2]),job=job['key']))
    # Encode birth histories using the parent's exact rows and labels.
    births = torch.load(cache(pc)/'births.pt',map_location='cpu',weights_only=False)
    bd = cache(c)/'births'; bd.mkdir(parents=True,exist_ok=True)
    for item in p['sources'][lane::2]:
        ids = [i for i,r in enumerate(births['rows']) if r['source']==item['key']]
        if not ids: continue
        f=bd/(item['key']+'.pt')
        if f.exists(): continue
        reader=Source(item['source']); values={}; groups={}
        for i in ids:
            row=births['rows'][i];groups.setdefault(row['frame'],[]).append(i)
        for frame, indices in groups.items():
            if frame<3: raise ValueError('Exact birth history missing; do not silently drop rows')
            centers=sorted({births['rows'][i]['center'] for i in indices})
            qv=np.stack([reader.patches(f,centers)[0] for f in range(frame-3,frame+1)],1)
            xx=observation(model,torch.from_numpy(qv*pc['length_factor']).float().flatten(0,1),micro).reshape(len(centers),4,280)
            for i in indices: values[i]=xx[centers.index(births['rows'][i]['center'])]
        atomic_save(f,dict(ids=ids,x=torch.stack([values[i] for i in ids])))
        if time.time()>deadline-600: raise TimeoutError('Birth histories reached deadline')
    write_json(root(c)/'technical'/f'data-complete-{lane}.json',dict(identity=p['identity']))


def seal(c,p):
    if (cache(c)/'data.json').exists(): return
    pc=parent(c)['config']; packets=[]
    for j in p['jobs']:
        f=cache(c)/'sequences'/(j['key']+'.pt')
        if sha(f)!=read(f.with_suffix('.json'))['sha256']:raise ValueError('Changed dense packet')
        packets.append(torch.load(f,map_location='cpu',weights_only=False))
    x=torch.cat([v['x'] for v in packets]); rows=[r for v in packets for r in v['rows']]
    train=np.flatnonzero([r['role']=='train' for r in rows]);a=x[train].flatten(0,1)
    mean=a.mean(0);scale=a.std(0).clamp_min(.01)
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite sequence observation')
    data=dict(x=x,rows=rows,mean=mean,scale=scale)
    atomic_save(cache(c)/'data.pt',data)
    birth=torch.load(cache(pc)/'births.pt',map_location='cpu',weights_only=False);bx=torch.empty(len(birth['rows']),4,280);seen=[]
    for f in (cache(c)/'births').glob('*.pt'):
        v=torch.load(f,map_location='cpu',weights_only=False);bx[v['ids']]=v['x'];seen+=v['ids']
    if sorted(seen)!=list(range(len(bx))):raise ValueError('Birth history rows incomplete or duplicated')
    atomic_save(cache(c)/'births.pt',dict(x=bx,rows=birth['rows']))
    old=torch.load(cache(pc)/'targets.pt',map_location='cpu',weights_only=False)
    # prepare() puts the physical teacher first; collect() retains that order.
    histories={job['key']:packet['x'][0,:4] for job,packet in zip(p['jobs'],packets)}
    short=[]
    for r in old['short']:
        base=histories[r['job']].clone()
        if not torch.allclose(base[-1,:256],r['z'][0],rtol=1e-3,atol=.015):raise ValueError('Frozen response/sequence feature mismatch')
        pairs=[]
        for k in range(2):
            values=[]
            for si,sign in enumerate([-1,1]):
                qv=r['state']+sign*pc['student_difference']['response_epsilon']*r['basis'][...,k]
                values.append(torch.cat([r['z_pairs'][k,si,0],observed(qv[...,:3])[0]],-1))
            pairs.append(torch.stack(values))
        short.append(dict(history=base,pairs=torch.stack(pairs),target=r['target'][0],response=r['response'][0],role=r['role'],ancestor=r['ancestor'],job=r['job']))
    atomic_save(cache(c)/'short.pt',short)
    write_json(cache(c)/'data.json',dict(identity=p['identity'],files={n:sha(cache(c)/(n+'.pt')) for n in ['data','births','short']},rows=len(rows),roles=dict(Counter(r['role'] for r in rows)),frames=64,spacing_ps=.1))


def load(c,name):
    receipt=read(cache(c)/'data.json');f=cache(c)/(name+'.pt')
    if sha(f)!=receipt['files'][name]:raise ValueError('Changed sealed data '+name)
    return torch.load(f,map_location='cpu',weights_only=False)
