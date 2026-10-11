"""Shared-trajectory feature ablations with bounded, archived GPU collection."""
import fcntl
import gc
import time
import numpy as np
import torch
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.local_response.oracle import Teacher
from src.research.local_response.gates import error
from src.research.response_performance.collection import acquire_parent
from .common import FAMILY,root,read,sha,save,write_json,legacy
from .data import prepare
from .features import make


def normalizer(c):
    p=root(c)/'technical/feature-normalization.pt'
    if sha(p)!=read(root(c)/'technical/parents.json')['normalizer_sha256']:raise ValueError('Changed feature normalization')
    return torch.load(p,map_location='cpu',weights_only=False)


def gate(c):
    states=prepare(c);out=root(c)/'technical/gates';out.mkdir(parents=True,exist_ok=True)
    identity=read(root(c)/'technical/binding.json')['identity']
    if (out/'complete.json').exists():
        if read(out/'complete.json')['identity']!=identity:raise ValueError('Changed gate')
        return
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    config=legacy(c)['oracle'];teacher=Teacher(config);norm=normalizer(c);rows=[]
    for index in c['gate_parent_indices']:
        state=states[index]
        if state['role']!='train':raise ValueError('Gate source is not training')
        results={}
        for radius in [24,36]:
            path=out/f'parent{index}-radius{radius}.pt'
            if path.exists():results[radius]=torch.load(path,map_location='cpu',weights_only=False);continue
            oracle,q,v=make(config,state,radius,teacher,norm);seeds=[c['branch_seed_base']-100-index]
            result=oracle.query(q,v,seeds);payload=dict(values=result.values.cpu(),responses=result.responses.cpu())
            if index==c['gate_parent_indices'][0] and radius==24:
                errors=[]
                for eps in [.03,.01]:
                    plus=oracle.query(q+eps*v[:,0],q.new_empty(len(q),0),seeds)
                    minus=oracle.query(q-eps*v[:,0],q.new_empty(len(q),0),seeds)
                    errors.append(dict(epsilon=eps,**error((plus.values-minus.values)/(2*eps),result.responses[:,:,0])))
                if min(x['relative']for x in errors)>.02:raise ValueError(f'New target AD/FD mismatch: {errors}')
                write_json(out/'finite-difference.json',errors)
            save(path,payload);results[radius]=payload
            del oracle,q,v,result;gc.collect();torch.cuda.empty_cache()
        for block,name in enumerate(['legacy_delta','rich_delta','rich_joint']):
            s=slice(256*block,256*(block+1));a,b=results[24],results[36]
            ve=error(a['values'][:,s],b['values'][:,s]);he=error(a['responses'][:,s],b['responses'][:,s])
            rows.append(dict(parent=index,target=name,value_rms=ve['rms'],response_relative=he['relative'],
                passed=ve['rms']<=1e-4 and he['relative']<=.05))
    write_metric_rows(rows,root(c)/'analyses/environment-gate-v1',family=FAMILY,name='gate')
    if not all(r['passed']for r in rows):raise ValueError('New observables fail 24/36A convergence; collection stopped')
    write_json(out/'complete.json',dict(state='complete',identity=identity,radius_A=24,reference_A=36,
        precision='Inherited validated float32 force/HVP producer; new observable JVP checked by finite differences',
        qualification='nested open environments, not exact full-cell equivalence'))


def collect(c,lane):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    states=prepare(c);identity=read(root(c)/'technical/binding.json')['identity']
    if read(root(c)/'technical/gates/complete.json')['identity']!=identity:raise ValueError('Gate identity mismatch')
    config=legacy(c)['oracle'];teacher=Teacher(config);norm=normalizer(c)
    scratch=resolve_path(c['simulation_scratch']);archive=resolve_path(c['simulation_archive'])
    locks=root(c)/'technical/locks';locks.mkdir(parents=True,exist_ok=True)
    deadline=allocation_deadline(reserve_seconds=300)
    for state in states:
        i=state['index'];dest=archive/f'parent-{i:04d}'
        with (locks/f'parent-{i}.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'complete.json').exists():
                r=read(dest/'complete.json')
                if r['identity']!=identity or sha(dest/'query.pt')!=r['query_sha256']:raise ValueError('Changed query')
                continue
            if time.time()>=deadline:raise TimeoutError('Allocation ending; archived batches retained')
            write_json(root(c)/'technical'/f'collection-lane{lane}.json',dict(state='running',parent=i,
                source=state['source'],role=state['role'],started_at=time.time()))
            oracle,q,v=make(config,state,24,teacher,norm)
            batch=min(config['maximum_batch'],max(1,config['batch_atom_budget']//(len(q)//3)))
            seed=(legacy(c)['branch_seed_base']+state['original_index']*10000 if state['original_index'] is not None
                  else c['branch_seed_base']+i*10000)
            values=list(range(seed,seed+(8 if state['role']=='train' else 32)))
            responses=values if state['role']=='train' else list(range(seed+1000,seed+1008)) if state['role']=='test' else []
            acquire_parent(oracle,q,v,role=state['role'],value_seeds=values,response_seeds=responses,
                destination=scratch/f'parent-{i:04d}',identity=identity,value_batch=batch,response_batch=batch,
                audit_seeds=responses[:1]if state['role']=='train' else [],agreement_atol=1e-6,agreement_rtol=1e-5,
                archive=dest,deadline=deadline)
            print('Collected',i,'/',len(states),'lane',lane,flush=True)
            del oracle,q,v;gc.collect();torch.cuda.empty_cache()
    write_json(root(c)/'technical'/f'collection-lane{lane}.json',dict(state='complete'))


def seal(c):
    states=prepare(c);identity=read(root(c)/'technical/binding.json')['identity'];records=[];costs=[]
    oldreceipt=read(resolve_path(c['reference'])/'technical/data-complete.json')
    if sha(oldreceipt['cache'])!=oldreceipt['sha256']:raise ValueError('Changed legacy normalization bank')
    old=torch.load(oldreceipt['cache'],map_location='cpu',weights_only=False)
    checks=[]
    for state in states:
        folder=resolve_path(c['simulation_archive'])/f'parent-{state["index"]:04d}';rec=read(folder/'complete.json')
        if rec['identity']!=identity or sha(folder/'query.pt')!=rec['query_sha256']:raise ValueError('Unsealed query')
        q=torch.load(folder/'query.pt',map_location='cpu',weights_only=False)
        count=8 if state['role']=='train' else 32
        if q['values'].shape!=(count,768):raise ValueError('Unexpected feature dimensions')
        # The reused collector stores an unused (0,256,2) tensor for selection.
        # Selection has no responses; explicitly construct its new empty schema.
        if state['role']=='selection':
            if q['responses'].shape!=(0,256,2):raise ValueError('Changed inherited empty-response schema')
            h=torch.empty(0,768,2)
        else:
            h=q['responses']
            if h.shape!=(8,768,2):raise ValueError('Unexpected derivatives')
        if state['original_index'] is not None:
            prior=old['parents'][state['original_index']]
            ve=error(q['values'][:,:256],prior['values'][:count])
            if ve['maximum_absolute']>1e-5:raise ValueError('Legacy common-path values changed')
            checks.append(dict(parent=state['index'],maximum_absolute=ve['maximum_absolute']))
        records.append(dict(index=state['index'],source=state['source'],role=state['role'],
            original_index=state['original_index'],q=state['q'][:80].float(),basis=state['basis'].float(),
            values=q['values'],responses=h))
        costs.append(dict(parent=state['index'],role=state['role'],seconds=rec['total_batch_seconds']))
    train=[r for r in records if r['role']=='train'];values=torch.stack([r['values']for r in train])
    center=values.mean((0,1));scale=values.flatten(0,1).std(0,unbiased=False).clamp_min(1e-4)
    # The legacy target uses exactly its original train-only scaling for comparability.
    center[:256]=old['center'];scale[:256]=old['scale']
    h=torch.stack([r['responses'].mean(0)for r in train])/scale[None,:,None]
    hscale=torch.stack([h[:,s:s+256].square().mean().sqrt().clamp_min(.001)for s in [0,256,512]])
    hscale[0]=old['response_scale']
    path=resolve_path(c['cache'])/'training.pt';save(path,dict(identity=identity,parents=records,center=center,scale=scale,response_scale=hscale))
    write_json(root(c)/'technical/data-complete.json',dict(state='complete',identity=identity,sha256=sha(path),
        cache=str(path),parents=len(records),legacy_execution_audit=checks))
    write_metric_rows(costs,root(c)/'analyses/collection-v1',family=FAMILY,name='costs')
