"""Extend observed histories only; preserve every sealed future and ancestral role."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from src.research.continuous_motion.data import Source, read, root, cache, resolve_path, sha, digest, write_json, atomic_save
from src.research.ni_latent_flow import data as parent_data
from src.research.ni_kinetic_latent.data import transform, inverse
from src.research.measurement_sim.features import structural, SmoothMeasurements
from src.research.ni_spatial_scales.data import measure_frame
from src.research.ni_spatial_scales.features import RegionalMeasurements

FAMILY = 'ni_ordering_flow'


def prepare(c):
    target = root(c)/'technical/plan.json'
    if target.exists():
        p = read(target)
        if p['config'] != c:
            raise ValueError('Changed frozen ordering plan')
        return p
    pp = resolve_path(c['parent_run'])/'technical/plan.json'
    if sha(pp) != c['parent_plan_sha256']:
        raise ValueError('Changed parent trajectory plan')
    parent = read(pp)
    _, bank = parent_data.load(parent['config'])
    bank['train_rows'] = np.flatnonzero([r['role']=='train' for r in bank['rows']])
    bank['test_rows'] = np.flatnonzero([r['role']=='test' for r in bank['rows']])
    spatial = read(resolve_path(parent['config']['spatial_run'])/'technical/plan.json')
    birth = torch.load(parent['birth']['cohort_path'], map_location='cpu', weights_only=False)
    sources = {s['key']: s for s in spatial['context_sources']}
    sources.update({s['key']:s for s in parent['sources']})
    rows = bank['rows']; birth_rows = spatial['birth_unique_rows']
    jobs=[]
    for key in sorted(sources):
        ni=[i for i,r in enumerate(rows) if r['source']==key]
        bi=[i for i,r in enumerate(birth_rows) if r['source']==key]
        if not ni and not bi: continue
        times=np.asarray(sources[key]['source']['times_ps'])
        for row in [rows[i] for i in ni]+[birth_rows[i] for i in bi]:
            frame=row['frame']; offsets=np.arange(-12,1)
            if frame<12 or not np.allclose(times[frame+offsets]-times[frame], offsets*.1, atol=1e-8,rtol=0):
                raise ValueError('Missing exact 1.2ps observed history: '+key)
        jobs.append(dict(key=key,native_indices=ni,birth_indices=bi))
    # All existing origins satisfy eligibility: no crystallinity or distance filter.
    atomic_save(cache(c)/'base.pt', bank)
    atomic_save(cache(c)/'births.pt',birth)
    labels=parent_data.labels_path(parent['config'])
    for job in jobs:
        indices=job['native_indices']
        atomic_save(cache(c)/'source-input'/f'{job["key"]}.pt',dict(
            x=bank['x'][indices], rows=[rows[i] for i in indices],
            normalizer={k:bank[k] for k in ('mean','scale','raw_mean','raw_scale')},
            columns=bank['columns']))
    p=dict(config=c,rows=rows,birth_rows=birth_rows,jobs=jobs,sources=sources,
        parent_identity=parent['identity'],parent_plan_sha256=c['parent_plan_sha256'],
        spatial_config=spatial['config'],spatial_jobs=spatial['context_jobs'],
        labels_path=str(labels),labels_sha256=sha(labels), columns=bank['columns'],
        ancestors={role:sorted({r['ancestor'] for r in rows if r['role']==role}) for role in ('train','test')},
        counts=dict(native=len(rows),train=len(bank['train_rows']),test=len(bank['test_rows']),birth_unique=len(birth_rows),birth_rows=len(birth['rows'])),
        support='fixed central atom; nearest80 reselected every observed frame in native AND birth histories; 12A complete-ball observed context',
        observation_offsets_ps=(np.arange(-12,1)*.1).tolist(),
        future_offsets_ps=(np.arange(1,61)*.1).tolist(),
        no_explicit_conditions=True,no_neighborhood_filter=True,
        base_sha256=sha(cache(c)/'base.pt'),birth_sha256=sha(cache(c)/'births.pt'))
    p['data_identity']=digest([parent['data_identity'],p['base_sha256'],p['birth_sha256'],jobs,p['observation_offsets_ps'],p['support']])
    p['identity']=digest(p);write_json(target,p)
    return p


def collect_source(c,key,deadline):
    p=read(root(c)/'technical/plan.json');job=next(j for j in p['jobs'] if j['key']==key)
    folder=cache(c)/'sources'/key;marker=root(c)/'analyses'/('collect-'+key)/'technical/complete.json'
    identity=digest([p['data_identity'],job])
    if marker.exists(): return True
    original=torch.load(cache(c)/'source-input'/f'{key}.pt',map_location='cpu',weights_only=False)
    # Reuse any observation already in the sealed native bank, including overlaps.
    local={}
    for row,values in zip(original['rows'], original['x']):
        for t,frame in enumerate(range(row['frame']-3,row['frame']+61)):
            local[frame,row['center']]=values[t]
    regional={}
    previous_job=next((j for j in p['spatial_jobs'] if j['key']==key),None)
    if previous_job is not None:
        old=torch.load(cache(p['spatial_config'])/'context-sources'/key/'data.pt',map_location='cpu',weights_only=False)
        # Per-source packets store RAW values; radius index1 is12 Angstrom.
        for row,values in zip(original['rows'],old['native']):
            for t,frame in enumerate(range(row['frame']-3,row['frame']+61)):
                regional[frame,row['center']]=values[t,1]
        old_birth=[p['birth_rows'][i] for i in previous_job['birth_rows']]
        for row,values in zip(old_birth,old['birth']):
            for t,frame in enumerate(range(row['frame']-3,row['frame']+1)):
                regional[frame,row['center']]=values[t,1]
    requests=defaultdict(lambda:defaultdict(list))
    for kind,indices,records in [('native',job['native_indices'],p['rows']),('birth',job['birth_indices'],p['birth_rows'])]:
        for rr,index in enumerate(indices):
            row=records[index]
            for t,frame in enumerate(range(row['frame']-12,row['frame']+1)):
                requests[frame][row['center']].append((kind,rr,t))
    payload=dict(identity=identity,next=0,native=torch.full((len(job['native_indices']),13,331),float('nan')),
                 birth=torch.full((len(job['birth_indices']),13,331),float('nan')))
    partial=folder/'partial.pt'
    if partial.exists():
        payload=torch.load(partial,map_location='cpu',weights_only=False)
        if payload['identity']!=identity:raise ValueError('Changed collection identity')
    reader=Source(p['sources'][key]['source']);smooth=SmoothMeasurements().cuda().eval()
    region=RegionalMeasurements(12.).cuda().eval();frames=sorted(requests);begin=time.monotonic()
    def checkpoint():
        atomic_save(partial,payload)
        write_json(marker.parent/'progress.json',dict(done=payload['next'],total=len(frames),seconds=time.monotonic()-begin))
    for number in range(payload['next'],len(frames)):
        if time.time()>deadline-180: checkpoint();return False
        frame=frames[number];centers=list(requests[frame])
        missing=[center for center in centers if (frame,center) not in local]
        if missing:
            state,_=reader.patches(frame,missing)
            state=state.astype(np.float32)
            if (np.linalg.norm(state[:,1:,:3],axis=-1).max(-1)<=5.5).any():
                raise ValueError('Nearest80 truncates smooth support: '+key)
            geometry=structural(state[...,:3],original['columns'][:153],c['length_factor'])
            with torch.no_grad():
                value=torch.cat((torch.from_numpy(geometry),smooth(torch.from_numpy(state).cuda()).cpu()),-1)
            value=transform(value,original['normalizer'])
            for center,v in zip(missing,value):local[frame,center]=v
        missing=[center for center in centers if (frame,center) not in regional]
        if missing:
            values,_,_=measure_frame(reader,frame,missing,[region],64)
            for center,v in zip(missing,values[:,0]):regional[frame,center]=v
        for center in centers:
            value=torch.cat((local[frame,center],regional[frame,center]))
            if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite prepared history')
            for kind,rr,t in requests[frame][center]:payload[kind][rr,t]=value
        payload['next']=number+1
        if number%32==0:checkpoint()
    checkpoint();atomic_save(folder/'data.pt',payload)
    write_json(marker,dict(state='complete',identity=identity,sha256=sha(folder/'data.pt'),seconds=time.monotonic()-begin))
    return True


def assemble(c):
    p=read(root(c)/'technical/plan.json');marker=root(c)/'analyses/assemble/technical/complete.json'
    if marker.exists():return True
    native=torch.empty(len(p['rows']),13,331);birth=torch.empty(len(p['birth_rows']),13,331)
    checks={}
    for job in p['jobs']:
        receipt=read(root(c)/'analyses'/('collect-'+job['key'])/'technical/complete.json')
        path=cache(c)/'sources'/job['key']/'data.pt'
        if sha(path)!=receipt['sha256']:raise ValueError('Changed collected source '+job['key'])
        packet=torch.load(path,map_location='cpu',weights_only=False)
        native[job['native_indices']]=packet['native'];birth[job['birth_indices']]=packet['birth'];checks[job['key']]=receipt['sha256']
    take=torch.tensor([r['role']=='train' for r in p['rows']])
    raw=native[take,:,244:].flatten(0,1).double();raw_mean=raw.mean(0);raw_scale=raw.std(0).clamp_min(1e-4)
    values=torch.asinh((raw-raw_mean)/raw_scale);mean=values.mean(0);scale=values.std(0).clamp_min(1e-4)
    norm={k:v.float() for k,v in dict(raw_mean=raw_mean,raw_scale=raw_scale,mean=mean,scale=scale).items()}
    for x in (native,birth):x[...,244:]=transform(x[...,244:],norm)
    atomic_save(cache(c)/'histories.pt',dict(native=native,birth=birth,context_normalizer=norm,identity=p['data_identity']))
    write_json(marker,dict(state='complete',sha256=sha(cache(c)/'histories.pt'),sources=checks))
    return True


def load(c,arm=None):
    p=read(root(c)/'technical/plan.json')
    bank=torch.load(cache(c)/'base.pt',map_location='cpu',weights_only=False)
    if arm is None or arm['treatment']=='local4':
        bank['history']=torch.cat((bank['x'][:,:4],torch.zeros(len(bank['x']),4,87)),-1)
    else:
        h=torch.load(cache(c)/'histories.pt',map_location='cpu',weights_only=False)['native']
        bank['history']=h[:,-arm['history_frames']:].clone()
        if not arm['regional']:bank['history'][...,244:]=0
    return p,bank


def load_births(c,arm):
    data=torch.load(cache(c)/'births.pt',map_location='cpu',weights_only=False)
    h=torch.load(cache(c)/'histories.pt',map_location='cpu',weights_only=False)['birth'][:,-arm['history_frames']:].clone()
    if not arm['regional']:h[...,244:]=0
    return data,h


def teacher_bank(c,bank):
    tc=read(resolve_path(c['teacher_config']));packet=torch.load(cache(tc)/'teachers.pt',map_location='cpu',weights_only=False)
    raw=packet['value'].float();x=transform(raw,bank)
    # Chain rule through the SAME train-fitted affine/asinh/affine coordinates.
    scale=bank['raw_scale'][153:];offset=(raw[...,153:]-bank['raw_mean'][153:])/scale
    jac=1/(scale*torch.sqrt(1+offset.square())*bank['scale'][153:])
    response=packet['response'].float()*jac[...,None]
    train=torch.tensor([r=='train' for r in packet['roles']])
    power=response[train,1:].square().mean((0,1,3)).clamp_min(1e-6)
    return dict(value=x,response=response,power=power,train=train,keys=packet['keys'])
