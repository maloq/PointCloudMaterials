"""Outcome-blind Ni teacher expansion and instantaneous density-rate inputs."""
from collections import Counter
from copy import deepcopy
import time
import numpy as np
import torch
from src.data.splits import train_test_role
from src.postprocessing.io import arrays
from src.research.ni_response import simulator
from .data import Source, root, cache, read, sha, digest, write_json, atomic_save, resolve_path
from .features import SmoothMeasurements, structural

FAMILY = 'measurement_rates'
GROUPS = [(0, 74), (74, 83), (83, 87), (87, 91)]


def density_rates(measure, state):
    """D_x rho[v], per ps, with the instantaneous periodic box held fixed."""
    direction = torch.cat((state[..., 3:], torch.zeros_like(state[..., 3:])), -1)
    return torch.func.jvp(measure.density, (state,), (direction,))[1]


def prepare(c):
    upstream = {}
    for name in ['physical_plan', 'new_sources_plan', 'measurement_plan']:
        path = resolve_path(c[name])
        if sha(path) != c[name+'_sha256']:
            raise ValueError('Changed pinned input: '+name)
        upstream[name] = read(path)
    old = upstream['physical_plan']
    measured = upstream['measurement_plan']
    sources = deepcopy(old['sources'])
    for item in sources:
        item.update(original_role=item['role'], role=train_test_role(item['role']), cohort='original60')
    for s in upstream['new_sources_plan']['parent']['sources']:
        reader = Source(s)
        rng = np.random.default_rng(int(digest([c['seed'], s['id'], 'centers'])[:12], 16))
        sources.append(dict(key=s['id'], source=s, centers=rng.choice(reader.ids, 64, replace=False).tolist(),
                            original_role=s['role'], role=train_test_role(s['role']), cohort='new30'))
    by_source = {s['key']:s for s in sources}
    if len(by_source) != len(sources):
        raise ValueError('Duplicate Ni source IDs')
    roles = {}
    for item in sources:
        s = item['source']; ancestor = s['ancestry_group']
        if ancestor in roles and roles[ancestor] != item['role']:
            raise ValueError('Ancestry crosses train/test: '+ancestor)
        roles[ancestor] = item['role']
        if s['potential'] != ['ni-mendelev2012-eam']:
            raise ValueError('Unexpected Ni generating potential')
        for part in s['parts']:
            steps = arrays(part['path'], part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(steps)*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Not native 0.1-ps observations: '+s['id'])
    jobs = []
    per_source = {s['key']:[] for s in sources}
    for original in old['jobs']:
        item = by_source[original['source_key']]
        job = dict(original, reuse=True, role=item['role'], original_role=item['original_role'], cohort=item['cohort'])
        job['center'] = item['centers'][job['teacher_center']]
        receipt = read(resolve_path(c['reuse_archive'])/job['key']/'complete.json')
        job['physical_sha256'] = receipt['sha256']
        jobs.append(job); per_source[item['key']].append(job['key'])
    base = [j['key'] for j in jobs if j['role'] == 'train']
    fitting = sorted([s for s in sources if s['role']=='train'], key=lambda s:digest([c['seed'],s['key']]))
    # Match the same320 temperature composition; temperature never enters models.
    temperature_totals = Counter()
    for s in fitting:
        if s['cohort']=='original60': temperature_totals[s['source']['temperature_K']] += 20
    diverse_counts = {}
    for temperature,total in temperature_totals.items():
        group = [s for s in fitting if s['source']['temperature_K']==temperature]
        diverse_counts.update({s['key']:total//len(group)+int(i<total%len(group)) for i,s in enumerate(group)})
    for item in sources:
        current = per_source[item['key']]
        count = (20 if item['cohort']=='original60' else diverse_counts[item['key']]) if item['role']=='train' else 5
        s = item['source']; times = np.asarray(s['times_ps'])
        eligible = np.flatnonzero((times >= 20)&(times <= times[-1]-12))
        if len(eligible) < count:
            raise ValueError('Insufficient available observation offsets: '+s['id'])
        used = {j['frame'] for j in jobs if j['source_key']==item['key']}
        candidates = eligible[np.rint(np.linspace(0,len(eligible)-1,count*4)).astype(int)]
        # Seeded permutation of a regular time grid, independent of birth labels.
        candidates = np.random.default_rng(int(digest([c['seed'],item['key'],'frames'])[:12],16)).permutation(candidates)
        for frame in candidates:
            if len(current) == count: break
            frame = int(frame)
            if frame in used: continue
            center_index = int(digest([c['seed'],s['id'],frame,'center'])[:8],16)%len(item['centers'])
            center = item['centers'][center_index]
            key = digest([c['protocol'],s['id'],frame,center])[:20]
            job = dict(key=key, source_key=s['id'], frame=frame, teacher_center=center_index,
                       center=center, reuse=False, role=item['role'], original_role=item['original_role'],
                       cohort=item['cohort'], ancestor=s['ancestry_group'])
            jobs.append(job); current.append(key); used.add(frame)
        if len(current) != count: raise ValueError('Incomplete anchor sampling: '+s['id'])
    banks = dict(base80=base,
        same320=[key for s in sources if s['role']=='train' and s['cohort']=='original60' for key in per_source[s['key']]],
        diverse320=[key for s in fitting for key in per_source[s['key']][:diverse_counts[s['key']]]])
    test = [j['key'] for j in jobs if j['role']=='test']
    plan = dict(config=c, sources=sources, jobs=jobs, banks=banks, test=test, columns=measured['columns'],
                source_roles=dict(Counter(s['role'] for s in sources)),
                sampling='Outcome-blind frames/centers; sampling available history inherits native stopping rules',
                bank_sources={name:len({j['ancestor'] for j in jobs if j['key'] in keys}) for name,keys in banks.items()})
    plan['identity'] = digest(plan)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != plan:
        raise ValueError('Changed density-rate plan; use a new revision')
    write_json(destination,plan)
    # The same old train-only output scaling is used in every data-size arm.
    original = measured['config']; packet = read(cache(original)/'data.json')
    if sha(cache(original)/'data.pt') != packet['sha256']: raise ValueError('Changed original measurements')
    native = torch.load(cache(original)/'data.pt',map_location='cpu',weights_only=False)
    atomic_save(cache(c)/'normalization.pt',dict(mean=native['mean'],scale=native['scale'],columns=plan['columns']))
    write_json(root(c)/'technical/prediction-context.json',dict(
        inputs='Present BO40/TDA113/density74/velocity17; optionally initial D_x density[V]74; no learned encoder',
        outputs='Same 244 measurements at 5,10,...100 fs; no future density-rate target',
        rates='Initial conditioning only; current fixed-box atomic-motion rate in inverse ps; recalculate for each perturbed Q/V input',
        conditions='No temperature, age, absolute time, box or species model inputs',
        physical='Full-cell FastNi EAM/FS BAOAB, original Q/V, fixed box, 1fs integration, f64 states/tangents/noise, f32 forces',
        roles='Only train/test; prior selection/validation/calibration pooled into test; no new inner split',
        checkpoint='Predeclared final update; test never enters fitting, preprocessing, scheduler or selection',
        support='Fixed central atom and initial80 identities throughout each physical continuation'))
    return plan


def measure_job(c, p, job, deadline):
    destination = cache(c)/'physical'/job['key']
    marker = destination/'complete.json'
    identity = digest([p['identity'],job])
    if marker.exists():
        done = read(marker)
        if done['identity'] != identity or sha(destination/'data.pt') != done['sha256']:
            raise ValueError('Changed density-rate packet: '+job['key'])
        return
    begin = time.monotonic()
    if job['reuse']:
        folder = resolve_path(c['reuse_archive'])/job['key']
        if sha(folder/'response.pt') != job['physical_sha256']: raise ValueError('Changed archived teacher')
    else:
        simulator.collect(c,p,job,deadline-300)
        folder = simulator.archive(c,job)
    saved = torch.load(folder/'response.pt',map_location='cpu',weights_only=False)
    measure = SmoothMeasurements().cuda().eval(); epsilon = c['training']['response_epsilon']
    state = saved['state'].float()[None]
    pair_states = [saved['state']+sign*size*saved['basis'][...,k]
                   for size in [epsilon,epsilon/2] for k in range(2) for sign in [-1,1]]
    present = torch.cat((state,torch.stack(pair_states).float()))
    with torch.no_grad(): rates = density_rates(measure,present.cuda()).cpu()
    future = saved['future'].float().flatten(0,1)
    if job['reuse']:
        old_folder = resolve_path(c['reuse_measurements'])/job['key']
        if sha(old_folder/'data.pt') != read(old_folder/'complete.json')['sha256']:
            raise ValueError('Changed reused measurement packet')
        old = torch.load(old_folder/'data.pt',map_location='cpu',weights_only=False)
        values = torch.cat((old['history'][-1:],old['pairs'].flatten(0,1),old['half_pairs'].flatten(0,1),old['target'].flatten(0,1)))
    else:
        states = torch.cat((present,future))
        hard = structural(states[...,:3].numpy(),p['columns'][:153],c['length_factor'])
        with torch.no_grad(): smooth = measure(states.cuda()).cpu()
        values = torch.cat((torch.from_numpy(hard).float(),smooth),-1)
    tangents = saved['tangent'].float().flatten(0,1).cuda()
    responses, initial = [], []
    for k in range(2):
        responses.append(torch.func.jvp(measure,(future.cuda(),),(tangents[...,k],))[1].cpu().reshape(2,20,91))
        initial.append(torch.func.jvp(measure,(state.cuda(),),(saved['basis'][None,...,k].float().cuda(),))[1][0].cpu())
    response_shots = torch.stack(responses,-1)
    record = dict(initial=values[0],pairs=values[1:5].reshape(2,2,244),half_pairs=values[5:9].reshape(2,2,244),
        target=values[9:].reshape(2,20,244),rates=rates[0],pair_rates=rates[1:5].reshape(2,2,74),
        half_pair_rates=rates[5:9].reshape(2,2,74),response=response_shots.mean(0),response_shots=response_shots,
        initial_response=torch.stack(initial,-1),role=job['role'],original_role=job['original_role'],
        ancestor=saved['ancestor'],job=job['key'],center=saved['center'],cohort=job['cohort'],
        temperature_K=next(s['source']['temperature_K'] for s in p['sources'] if s['key']==job['source_key']))
    if not all(torch.isfinite(v).all() for v in record.values() if torch.is_tensor(v)):
        raise FloatingPointError('Nonfinite density-rate observations: '+job['key'])
    atomic_save(destination/'data.pt',record)
    write_json(marker,dict(identity=identity,sha256=sha(destination/'data.pt'),physical_sha256=sha(folder/'response.pt'),
        seconds=time.monotonic()-begin,reused_physics=job['reuse']))


def ready(c, p, bank, *, test=False):
    keys = p['test'] if test else p['banks'][bank]
    return all((cache(c)/'physical'/key/'complete.json').exists() for key in keys)


def load(c, bank, *, test=False, device='cpu'):
    p = read(root(c)/'technical/plan.json'); train_keys = p['banks'][bank]
    keys = list(train_keys)+(p['test'] if test else [])
    records = []
    for key in keys:
        folder = cache(c)/'physical'/key
        if sha(folder/'data.pt') != read(folder/'complete.json')['sha256']:
            raise ValueError('Changed measured density-rate bank: '+key)
        records.append(torch.load(folder/'data.pt',map_location='cpu',weights_only=False))
    norm = torch.load(cache(c)/'normalization.pt',map_location='cpu',weights_only=False)
    mean,scale = norm['mean'].to(device),norm['scale'].to(device)
    tensors = {k:torch.stack([r[k] for r in records]).to(device) for k,v in records[0].items() if torch.is_tensor(v)}
    for key in ['initial','pairs','half_pairs','target']: tensors[key] = (tensors[key]-mean)/scale
    for key in ['response','response_shots','initial_response']: tensors[key] = tensors[key]/scale[153:,None]
    n = len(train_keys)
    rate_mean = tensors['rates'][:n].mean(0); rate_scale = tensors['rates'][:n].std(0).clamp_min(1e-4)
    for key in ['rates','pair_rates','half_pair_rates']: tensors[key] = (tensors[key]-rate_mean)/rate_scale
    paths = torch.cat((tensors['initial'][:n,None,None].expand(-1,2,1,-1),tensors['target'][:n]),2)
    rho = (paths[:,:,:-1]*paths[:,:,1:]).mean((0,1,2))/paths[:,:,:-1].square().mean((0,1,2)).clamp_min(1e-5)
    tensors['decay_rate'] = -rho.clamp(.01,.9995).log()/.005*(scale>1e-4)
    tensors['increment_scale'] = (paths[:,:,1:]-paths[:,:,:-1]).square().mean((0,1,2)).sqrt().clamp_min(.05)
    tensors['normalizers'] = torch.stack([tensors['response'][:n,:,a:b].square().mean((0,2)).clamp_min(1e-5) for a,b in GROUPS])
    norm.update(rate_mean=rate_mean.cpu(),rate_scale=rate_scale.cpu())
    return p,records,tensors,n,norm
