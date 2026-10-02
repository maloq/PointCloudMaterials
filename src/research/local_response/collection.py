"""GPU lanes share a locked, resumable and durably archived query bank."""
import fcntl
import gc
import time
import torch

from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.response_performance.collection import acquire_parent
from .common import FAMILY,root,read,sha,save,write_json
from .data import prepare
from .oracle import Teacher,make


def collect(c,lane):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    states=prepare(c);identity=read(root(c)/'technical/binding.json')['identity']
    gate=read(root(c)/'technical/gates/complete.json')
    if gate['identity']!=identity or gate['state']!='complete':raise ValueError('Convergence gate missing or changed')
    radius=gate['selected_radius_A'];teacher=Teacher(c['oracle'])
    scratch=resolve_path(c['simulation_scratch']);archive=resolve_path(c['simulation_archive'])
    locks=root(c)/'technical/locks';locks.mkdir(parents=True,exist_ok=True)
    deadline=allocation_deadline(reserve_seconds=240)
    for state in states:
        index=state['index'];dest=archive/f'parent-{index:03d}'
        with (locks/f'parent-{index:03d}.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'complete.json').exists():
                record=read(dest/'complete.json')
                if record['identity']!=identity or sha(dest/'query.pt')!=record['query_sha256']:
                    raise ValueError(f'Changed completed parent {index}')
                continue
            if time.time()>=deadline:raise TimeoutError('Allocation reserve; collected batches archived')
            write_json(root(c)/'technical'/f'collection-lane{lane}.json',dict(state='running',parent=index,
                source=state['source'],role=state['role'],radius_A=radius,started_at=time.time()))
            oracle,q,basis=make(c,state,radius,teacher)
            batch=min(c['oracle']['maximum_batch'],max(1,c['oracle']['batch_atom_budget']//(len(q)//3)))
            seed=c['branch_seed_base']+index*10000
            values=list(range(seed,seed+c['value_shots']))
            responses=(values[:c['response_shots']] if state['role']=='train' else
                list(range(seed+1000,seed+1000+c['response_shots'])) if state['role']=='test' else [])
            try:
                acquire_parent(oracle,q,basis,role=state['role'],value_seeds=values,response_seeds=responses,
                    destination=scratch/f'parent-{index:03d}',identity=identity,value_batch=batch,response_batch=batch,
                    audit_seeds=responses[:1] if state['role']=='train' else [],agreement_atol=1e-6,
                    agreement_rtol=1e-5,archive=dest,deadline=deadline)
            except BaseException as exc:
                write_json(dest/'failure.json',dict(identity=identity,error=repr(exc),lane=lane));raise
            print(dict(stage='collected',lane=lane,parent=index,source=state['source'],atoms=len(q)//3,batch=batch),flush=True)
            del oracle,q,basis;gc.collect();torch.cuda.empty_cache()
    write_json(root(c)/'technical'/f'collection-lane{lane}.json',dict(state='complete',finished_at=time.time()))


def seal(c):
    states=prepare(c);identity=read(root(c)/'technical/binding.json')['identity'];records=[];hashes={};costs=[]
    for state in states:
        out=resolve_path(c['simulation_archive'])/f'parent-{state["index"]:03d}'
        receipt=read(out/'complete.json')
        if receipt['identity']!=identity or sha(out/'query.pt')!=receipt['query_sha256']:
            raise ValueError(f'Unsealed parent {state["index"]}')
        query=torch.load(out/'query.pt',map_location='cpu',weights_only=False)
        if query['values'].shape!=(32,256) or query['responses'].shape!=((0 if state['role']=='selection' else 8),256,2):
            raise ValueError('Unexpected value/response bank dimensions')
        records.append(dict(index=state['index'],source=state['source'],role=state['role'],frame=state['frame'],
            center_atom_id=state['center_atom_id'],q=state['q'][:80].float(),basis=state['basis'].float(),
            values=query['values'],responses=query['responses'],cost_seconds=receipt['total_batch_seconds']))
        hashes[str(state['index'])]=receipt['query_sha256']
        costs.append(dict(parent=state['index'],source=state['source'],role=state['role'],
            seconds=receipt['total_batch_seconds'],executed_trajectories=receipt['executed_trajectories']))
    train=[r for r in records if r['role']=='train'];values=torch.stack([r['values'][:8] for r in train])
    center=values.mean((0,1));scale=values.flatten(0,1).std(0,unbiased=False).clamp_min(c['training']['value_scale_floor'])
    h=torch.stack([r['responses'].mean(0) for r in train])/scale[None,:,None]
    response_scale=h.square().mean().sqrt().clamp_min(c['training']['response_scale_floor'])
    path=resolve_path(c['cache'])/'training.pt'
    save(path,dict(identity=identity,parents=records,center=center,scale=scale,response_scale=response_scale))
    write_json(root(c)/'technical/data-complete.json',dict(state='complete',identity=identity,cache=str(path),
        sha256=sha(path),archived_query_sha256=hashes,parents=len(records)))
    write_metric_rows(costs,root(c)/'analyses/collection-v1',family=FAMILY,name='costs')


def load(c):
    record=read(root(c)/'technical/data-complete.json')
    if record['identity']!=read(root(c)/'technical/binding.json')['identity'] or sha(record['cache'])!=record['sha256']:
        raise ValueError('Training label bank changed')
    return torch.load(record['cache'],map_location='cpu',weights_only=False)
