"""Numerical gates and shot-resumable, checksum-archived phase-space queries."""
import gc
import time
import fcntl
import shutil
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.local_response.gates import error
from .common import FAMILY,root,read,save,sha,write_json,deadline
from .data import prepare,initial,velocity
from .oracle import Oracle


def normalizer(c):
    base=resolve_path(c['geometry_reference'])/'technical'
    if sha(base/'feature-normalization.pt') != read(base/'parents.json')['normalizer_sha256']:
        raise ValueError('Training-only observable normalizer changed')
    if sha(base/'feature-normalization.pt') != read(root(c)/'technical/binding.json')['observable_normalizer_sha256']:
        raise ValueError('Bound observable normalizer changed')
    return torch.load(base/'feature-normalization.pt',weights_only=False)


def publish(c,name,payload):
    scratch=resolve_path(c['simulation_scratch'])/name
    archive=resolve_path(c['simulation_archive'])/name
    save(scratch,payload);archive.parent.mkdir(parents=True,exist_ok=True)
    temporary=archive.with_suffix('.copying.pt');shutil.copy2(scratch,temporary)
    if sha(scratch)!=sha(temporary):raise IOError('Simulation archive checksum mismatch')
    temporary.replace(archive)
    return archive


def gate(c):
    if c.get('teacher_backend')=='cuda_meam':
        binding=read(root(c)/'technical/binding.json')
        write_json(root(c)/'technical/gates/complete.json',dict(state='complete',identity=binding['identity'],
            radius_A=None,boundary='full native periodic cell',horizons_fs=c['horizons_fs'],
            decision='Use completed CUDA MEAM validation; user explicitly requested no further MEAM checks',
            state_dtype=c.get('state_dtype','float64'),force_dtype=c['force_dtype'],
            prior_evidence=binding['fast_meam_evidence'],new_meam_checks_performed=False))
        return
    from .validation import gate as periodic_gate
    periodic_gate(c)


def collect(c,lane):
    states=prepare(c);norm=normalizer(c);radius=read(root(c)/'technical/gates/complete.json')['radius_A']
    identity=read(root(c)/'technical/binding.json')['identity']
    archive=resolve_path(c['simulation_archive']);locks=root(c)/'technical/locks';locks.mkdir(parents=True,exist_ok=True)
    stop_at=deadline(c,reserve_seconds=300)
    for state in states:
        for field in range(c['velocity_fields']):
            name=f'parent{state["index"]:03d}-velocity{field}'
            with (locks/(name+'.lock')).open('a+') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                dest=archive/name
                if (dest/'complete.json').exists():
                    receipt=read(dest/'complete.json')
                    if receipt['identity']!=identity or sha(dest/'query.pt')!=receipt['sha256']:raise ValueError('Changed archived query')
                    continue
                q,v,basis,ids=initial(c,state,field,radius);oracle=Oracle(c,state,ids,norm,periodic=radius is None)
                role=state['role'];nv=c['shots'][role]['values'];nh=c['shots'][role]['responses']
                base=c['noise_seed']+state['index']*100000+field*1000
                seedsets={'shared':list(range(base,base+nv))} if role=='train' else {
                    'values':list(range(base,base+nv)),'response':list(range(base+500,base+500+nh))}
                pieces={};elapsed=0.
                for mode,seeds in seedsets.items():
                    pieces[mode]=[]
                    for start in range(0,len(seeds),c['oracle_batch']):
                        if time.time()>=stop_at:raise TimeoutError('Collection time budget; completed batches archived')
                        path=dest/f'{mode}-{start:03d}.pt'
                        if path.exists():part=torch.load(path,weights_only=False)
                        else:
                            write_json(root(c)/'technical'/f'collection-{lane}.json',dict(state='running',parent=state['index'],
                                velocity=field,mode=mode,shot=start,started_at=time.time()))
                            selected=seeds[start:start+c['oracle_batch']];begin=time.monotonic()
                            value,response=oracle.simulate(q,v,basis if mode!='values' else basis[...,:0],selected)
                            if mode!='response':
                                free,_=oracle.simulate(q,v,basis[...,:0],selected,force_free=True)
                            else:
                                free=torch.empty(0)
                            part=dict(identity=identity,seeds=selected,values=value,responses=response,force_free=free,seconds=time.monotonic()-begin)
                            if c.get('initial_velocity')=='saved_md':
                                part['execution']=dict(physical_temperature_K=oracle.temperature_K,
                                    integrator='BAOAB',timestep_fs=c['timestep_fs'],damping_fs=c['friction_fs'],
                                    state_dtype=str(q.dtype),velocity_dtype=str(v.dtype),tangent_dtype=str(basis.dtype),
                                    values_dtype=str(value.dtype),response_dtype=str(response.dtype),force_arithmetic=c['force_dtype'],
                                    box='fixed full periodic cell',initial_velocity='saved_md',
                                    force_backend='FastAl',device=torch.cuda.get_device_name(),new_meam_validation=False)
                            publish(c,f'{name}/{path.name}',part)
                        if part['identity']!=identity or part['seeds']!=seeds[start:start+c['oracle_batch']]:raise ValueError('Changed saved branch')
                        pieces[mode].append(part);elapsed+=part['seconds']
                valueparts=pieces['shared'if role=='train' else 'values']
                responseparts=pieces['shared'if role=='train' else 'response']
                value=torch.cat([p['values']for p in valueparts]);free=torch.cat([p['force_free']for p in valueparts])
                response=torch.cat([p['responses']for p in responseparts])if responseparts else torch.empty(0,128*len(c['horizons_fs']),4)
                payload=dict(identity=identity,values=value,responses=response,force_free=free,
                    velocity=v[:80].cpu(),position=q[:80].cpu(),basis=basis[:,:80].cpu(),source=state['source'],
                    parent=state['index'],field=field,role=role,value_seeds=[s for p in valueparts for s in p['seeds']],
                    response_seeds=[s for p in responseparts for s in p['seeds']])
                if c.get('initial_velocity')=='saved_md':
                    payload['audit_metadata']=dict(physical_temperature_K=oracle.temperature_K,
                        original_frame=state['frame'],original_time_ps=state['time_ps'],
                        initial_velocity='saved_md',velocity_sha256=state['native_velocity_sha256'])
                path=publish(c,f'{name}/query.pt',payload)
                write_json(dest/'complete.json',dict(state='complete',identity=identity,sha256=sha(path),seconds=elapsed))
                print('collected',name,role,elapsed,flush=True)
                del oracle,q,v,basis;gc.collect();torch.cuda.empty_cache()
    write_json(root(c)/'technical'/f'collection-{lane}.json',dict(state='complete'))


def seal(c):
    identity=read(root(c)/'technical/binding.json')['identity'];records=[]
    for state in prepare(c):
        for field in range(c['velocity_fields']):
            d=resolve_path(c['simulation_archive'])/f'parent{state["index"]:03d}-velocity{field}'
            receipt=read(d/'complete.json')
            if receipt['identity']!=identity or sha(d/'query.pt')!=receipt['sha256']:raise ValueError('Unsealed velocity query')
            r=torch.load(d/'query.pt',weights_only=False);spec=c['shots'][state['role']];features=len(c['horizons_fs'])*128
            if r['values'].shape!=(spec['values'],features) or r['responses'].shape!=(spec['responses'],features,4):
                raise ValueError('Unexpected shot or feature shape')
            if state['role']=='test' and set(r['value_seeds'])&set(r['response_seeds']):raise ValueError('Test streams overlap')
            records.append(r)
    train=[r for r in records if r['role']=='train'];values=torch.stack([r['values']for r in train])
    center=values.mean((0,1));scale=values.flatten(0,1).std(0,unbiased=False).clamp_min(1e-4)
    h=torch.stack([r['responses'].mean(0) for r in train])/scale[None,:,None]
    hscale=torch.stack([h[:,:,j:j+2].square().mean().sqrt().clamp_min(.001) for j in [0,2]])
    path=resolve_path(c['cache'])/'training.pt';save(path,dict(identity=identity,records=records,center=center,scale=scale,response_scale=hscale))
    write_json(root(c)/'technical/data-complete.json',dict(identity=identity,sha256=sha(path),cache=str(path),
        state='complete',geometries=len(prepare(c)),velocity_states=len(records)))
