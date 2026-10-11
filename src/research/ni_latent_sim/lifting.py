"""Empirical full-cell decoder and independent fixed-box physical continuations.

The decoder selects real training microstates; it is not an atomistic generator.
Matching physical thermostat settings is an audit-level experimental stratum,
never a tensor input to either learned encoder or predictor.
"""
import math
import time
import numpy as np
import torch
from ase import units
from scipy.spatial import cKDTree
from src.simulation.eam_gpu.potential import FastNi
from src.experiment_runner.metric_docs import write_metric_rows
from .data import Source,load,parent,root,cache,resolve_path,sha,digest,read,atomic_save,write_json,observation,FAMILY
from .evaluate import fitted
from src.research.ni_response.model import Motion


def physical(c,p,row,encoder,deadline):
    key=digest([row['source'],row['frame'],row['center'],c['lifting'],'fixed_box_baoab'])[:24]
    archive=resolve_path(c['simulation_archive'])/key;file=archive/'observations.pt'
    if (archive/'complete.json').exists():
        if sha(file)!=read(archive/'complete.json')['sha256']:raise ValueError('Changed physical lifting bank')
        return torch.load(file,map_location='cpu',weights_only=False)
    source=next(s for s in p['sources'] if s['key']==row['source']);reader=Source(source['source']);q,v,box=reader.frame(row['frame'])
    cfg=c['lifting'];pc=parent(c)['config'];mass=58.71;shots=cfg['shots'];n=len(q)
    model=FastNi(resolve_path(pc['potential_file']),n,box=box,dtype=torch.float32,expected_sha256=pc['potential_sha256'])
    x=torch.tensor(q,device='cuda',dtype=torch.float64).reshape(1,-1).repeat(shots,1)
    momentum=torch.tensor(v,device='cuda',dtype=torch.float64).reshape(1,-1).repeat(shots,1)*mass/(1000*units.fs)
    seed=int(key[:12],16);generators=[torch.Generator(device='cuda').manual_seed(seed+i) for i in range(shots)]
    dt=units.fs;decay=math.exp(-1/1000);amplitude=math.sqrt(mass*units.kB*source['source']['temperature_K']*(1-decay**2))
    _,force=model.force(x);states=[];begin=time.monotonic();center=int(np.searchsorted(reader.ids,row['center']))
    for step in range(1,501):
        if time.time()>deadline-60:raise TimeoutError('Physical lifting deadline')
        momentum+=.5*dt*force;x+=.5*dt*momentum/mass
        noise=torch.stack([torch.randn(3*n,device='cuda',dtype=torch.float64,generator=g) for g in generators])
        momentum=decay*momentum+amplitude*noise;x+=.5*dt*momentum/mass;_,force=model.force(x);momentum+=.5*dt*force
        if step in [100,500]:
            a=x.reshape(shots,n,3).cpu().numpy();b=(momentum.reshape(shots,n,3)/mass*(1000*units.fs)).cpu().numpy();patch=[]
            for qs,vs in zip(a,b):
                wrapped=qs%box;_,neighbors=cKDTree(wrapped,boxsize=box).query(wrapped[center],k=80)
                local=qs[neighbors]-qs[center];local-=box*np.rint(local/box);patch.append(np.concatenate((local,vs[neighbors]),-1))
            states.append(np.stack(patch))
    states=torch.from_numpy(np.stack(states,1));features=observation(encoder,states.flatten(0,1).float()*pc['length_factor'],32).reshape(shots,2,280)
    result=dict(x=features,state=states,row=row,box=box,temperature_K=source['source']['temperature_K'],seed=seed)
    work=resolve_path(c['simulation_scratch'])/key;work.mkdir(parents=True,exist_ok=True);atomic_save(work/'observations.pt',result)
    import shutil
    archive.mkdir(parents=True,exist_ok=True);shutil.copy2(work/'observations.pt',file)
    if sha(file)!=sha(work/'observations.pt'):raise IOError('Lifting publication mismatch')
    write_json(archive/'complete.json',dict(sha256=sha(file),seconds=time.monotonic()-begin,state_dtype='float64',force_dtype='float32',protocol='original full32000 Q,V; own fixed box; source thermostat T; 1fs BAOAB,1ps friction; independent two-shot 100/500fs continuations',storage='prequantization cropped response observations, not archived full trajectories'))
    return result


def energy(a,b):
    dim=a.shape[-1]
    return float((2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean())/dim**.5)


def run(c,p,arm,deadline):
    out=root(c)/'analyses'/arm['name']/'lifting';out.mkdir(parents=True,exist_ok=True)
    if (out/'technical/complete.json').exists():return
    data=load(c,'data');model=fitted(c,arm);pc=parent(c)['config'];encoder=Motion(pc).cuda().eval().requires_grad_(False)
    xs=(data['x'][:,:4]-data['mean'])/data['scale'];zs=[]
    with torch.no_grad():
        for block in xs.split(256):zs.append(model.encode(block.cuda()[:,-arm['history']:]).flatten(1).cpu())
    z=torch.cat(zs);rows=data['rows'];source={s['key']:s['source'] for s in p['sources']};train=np.flatnonzero([r['role']=='train' for r in rows]);test=np.flatnonzero([r['role']=='test' for r in rows])
    eligible=[i for i in test if any(source[rows[j]['source']]['temperature_K']==source[rows[i]['source']]['temperature_K'] for j in train)]
    # One outcome-independent anchor per held-out source, evenly sample sources.
    first={}
    for i in eligible:first.setdefault(rows[i]['source'],i)
    queries=[list(first.values())[i] for i in np.linspace(0,len(first)-1,min(c['lifting']['queries'],len(first)),dtype=int)]
    rng=np.random.default_rng(c['seed']);report=[];links=[]
    with torch.no_grad():
        for qi in queries:
            if time.time()>deadline-300:raise TimeoutError('Lifting budget exhausted')
            temp=source[rows[qi]['source']]['temperature_K'];candidates=[i for i in train if source[rows[i]['source']]['temperature_K']==temp]
            distance=(z[candidates]-z[qi]).square().mean(-1).numpy();nearest=np.asarray(candidates)[np.argsort(distance)[:2]];random=rng.choice(candidates,2,replace=False)
            base=physical(c,p,rows[qi],encoder,deadline);record=dict(query=rows[qi],temperature_K=temp,nearest=[rows[i] for i in nearest],random=[rows[i] for i in random]);links.append(record)
            for method,selected in [('nearest_latent',nearest),('random_same_protocol',random)]:
                bank=[physical(c,p,rows[i],encoder,deadline) for i in selected];future=torch.cat([b['x'] for b in bank])
                for k,h in enumerate([.1,.5]):
                    for space,sl in [('structure',slice(0,128)),('descriptors',slice(256,280))]:
                        a=(future[:,k,sl]-data['mean'][sl])/data['scale'][sl];b=(base['x'][:,k,sl]-data['mean'][sl])/data['scale'][sl]
                        report.append(dict(model=arm['name'],query_id=rows[qi]['id'],method=method,space=space,horizon_ps=h,empirical_energy_distance=energy(a,b),ensemble_mean_mse=float((a.mean(0)-b.mean(0)).square().mean()),reference_shots=len(b),decoded_shots=len(a),mean_relative_box_volume_difference=float(np.mean([np.prod(v['box'])/np.prod(base['box'])-1 for v in bank]))))
            write_json(out/'technical/progress.json',dict(queries_complete=len(links),queries_total=len(queries)))
    write_metric_rows(report,out,family=FAMILY,name='physical_lifting',columns=list(report[0]));write_json(out/'technical/decoder-links.json',links)
    write_json(out/'technical/complete.json',dict(state='complete',queries=len(links),decoder='empirical training-state retrieval; not learned atomic generation',comparison='physical fixed-box MD against physical fixed-box MD; not equality with NPT-trained latent dynamics'))
