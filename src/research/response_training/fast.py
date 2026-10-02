"""Default Al256 oracle and durable batched collection for the float32 protocol."""
import shutil
import time

import torch

from src.experiment_runner.execution import allocation_deadline
from src.project_runtime.paths import resolve_path
from src.research.response_atlas.atomistic import configuration, basis_for, physical_screen
from src.research.response_performance.oracle import GPUPotential, BatchedFeatures, BatchedOracle
from src.research.response_performance.collection import acquire_parent
from .common import simulation_profile, oracle_config, metric_family


def make(c):
    profile = simulation_profile(c)
    if profile is None or profile['protocol'] != 'al256_response_float32_cueq_v1':
        raise ValueError('The fast Al256 simulator requires its explicit versioned profile')
    o = oracle_config(c)
    if (c['oracle_recipe_sha256'] != profile['reference_recipe_sha256'] or
            o['horizons_steps'] != [20,100] or c['directions'] != 2):
        raise ValueError('The default fast profile is validated only for the fixed Al256 20/100fs two-direction case')
    if (profile['dtype'],profile['backend'],profile['graph'],profile['reuse_response_values']) != ('float32','cueq','gpu',True):
        raise ValueError('Changed numerical settings require a separately validated profile')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    atoms=configuration(o,0)
    potential=GPUPotential(o,atoms,backend=profile['backend'],dtype=torch.float32,skin=profile['skin_A'])
    features=BatchedFeatures(atoms.cell.lengths(),o,torch.float32)
    return BatchedOracle(potential,features,o), profile


def acquire(c, oracle, profile, state, identity, work, dest, deadline):
    """One parent; exact original basis generation, then explicit float32 cast."""
    from . import data
    o=oracle_config(c)
    index,role=state['index'],state['role']
    q64=state['q'].flatten().double().cuda()
    basis=basis_for(q64,c['directions'],o['seed']+100+index).float()
    q=q64.float()
    screen_path=dest/'screen.json'
    if screen_path.exists():
        screen=data.read(screen_path)
        if screen['identity'] != identity:raise ValueError('Changed float32 parent screen')
    else:
        torch.cuda.synchronize();start=time.monotonic()
        screen=dict(identity=identity,**physical_screen(oracle.potential,q))
        torch.cuda.synchronize();screen['seconds']=time.monotonic()-start
        data.write_json(screen_path,screen)
    start=c['branch_seed_base']+1000*index
    value_seeds=list(range(start,start+c['value_shots']))
    response_start=start+(100 if role=='test' else 0)
    response_seeds=list(range(response_start,response_start+c['response_shots'])) if role!='selection' else []
    audit=response_seeds[:profile['training_audit_shots']] if role=='train' else []
    result=acquire_parent(oracle,q,basis,role=role,value_seeds=value_seeds,response_seeds=response_seeds,
        destination=work/'batches',archive=dest/'batches',identity=identity,
        value_batch=profile['value_batch'],response_batch=profile['response_batch'],audit_seeds=audit,
        agreement_atol=profile['audit_atol'],agreement_rtol=profile['audit_rtol'],deadline=deadline)
    batches=result['costs']
    cost=dict(parent=index,role=role,collection_seconds=sum(b['batch_seconds'] for b in batches),
        screen_seconds=screen['seconds'],
        ordinary_seconds=sum(b['batch_seconds'] for b in batches if b['kind']=='value'),
        response_seconds=sum(b['batch_seconds'] for b in batches if b['kind']=='response'),
        audit_seconds=sum(b['batch_seconds'] for b in batches if b['kind']=='audit'),
        executed_trajectories=sum(len(b['seeds']) for b in batches),
        reused_value_rows=len(result['reused_value_seeds']),
        logical_force_calls=sum(b['logical_force_calls'] for b in batches),
        logical_hvp_calls=sum(b['logical_hvp_calls'] for b in batches))
    payload=dict(identity=identity,**state,basis=basis.cpu().reshape(256,3,2),
        values=result['values'],responses=result['responses'],value_seeds=value_seeds,
        response_seeds=response_seeds,cost=cost,execution_audit=result['audit'],
        simulation_profile=profile,batch_costs=batches)
    payload['q']=q.cpu().reshape(256,3)
    payload['box']=oracle.potential.box.cpu()
    data.save_pt(work/'query.pt',payload)
    shutil.copy2(work/'query.pt',dest/'query.pt.publishing')
    (dest/'query.pt.publishing').replace(dest/'query.pt')
    data.write_json(dest/'complete.json',dict(identity=identity,sha256=data.sha(dest/'query.pt'),cost=cost))
    return payload


def collect(c):
    # Access through the data module so the existing locked multi-GPU adapter
    # retains control of parent assignment, publication and global sealing.
    from . import data
    torch.set_num_threads(2)
    identity=data.bind(c)['identity']
    states=data.prepare(c)
    oracle,profile=make(c)
    archive,scratch=resolve_path(c['simulation_archive']),resolve_path(c['simulation_scratch'])
    deadline=allocation_deadline(reserve_seconds=240)
    costs=[]
    for state in states:
        index=state['index'];dest=archive/f'parent-{index:03d}';work=scratch/f'parent-{index:03d}'
        dest.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():
            receipt=data.read(dest/'complete.json')
            if receipt['identity']!=identity or data.sha(dest/'query.pt')!=receipt['sha256']:
                raise ValueError(f'Changed query: {dest}')
            costs.append(receipt['cost']);continue
        try:
            record=acquire(c,oracle,profile,state,identity,work,dest,deadline)
            costs.append(record['cost'])
            data.write_json(data.root(c)/'technical/collection-progress.json',dict(
                state='running',parent=index,role=state['role'],completed_parents=len(costs),
                total_parents=len(states),numerical_profile=profile['protocol']))
            print(f'Collected parent {index:02d}: float32, batched, {record["cost"]["executed_trajectories"]} trajectories',flush=True)
        except BaseException as error:
            data.write_json(dest/'failure.json',dict(identity=identity,error=repr(error)))
            raise
        finally:
            shutil.copytree(work,dest,dirs_exist_ok=True)
    data.write_metric_rows(costs,data.root(c)/'analyses/collection-v2',family=metric_family(c),name='oracle-cost')
    data.seal(c)


def oracle_gate(c):
    """Fresh full-horizon batch compared to the original float64 AD oracle."""
    from src.research.response_atlas.atomistic import MACEPotential, PathFeatures, oracle_for
    oracle,p=make(c);o=oracle_config(c);atoms=configuration(o,0)
    q64=torch.tensor(atoms.positions.flatten(),dtype=torch.float64,device='cuda')
    b64=basis_for(q64,2,20261203);q,b=q64.float(),b64.float()
    seeds=list(range(69000000,69000000+p['response_batch']))
    torch.cuda.synchronize();start=time.monotonic()
    response=oracle.query(q,b,seeds);torch.cuda.synchronize();ad_seconds=(time.monotonic()-start)/len(seeds)
    start=time.monotonic()
    value=oracle.query(q,q.new_empty(len(q),0),seeds);torch.cuda.synchronize();value_seconds=(time.monotonic()-start)/len(seeds)
    audit=float((response.values-value.values).abs().max())
    if not torch.allclose(response.values,value.values,atol=p['audit_atol'],rtol=p['audit_rtol']):
        raise ValueError(f'Float32 value/AD execution mismatch: {audit}')
    reference_potential=MACEPotential(o,atoms,'cuda')
    reference=oracle_for(o,reference_potential,PathFeatures(atoms.cell.lengths(),o['horizons_steps'],o,'cuda'),q64,o['horizons_steps'])
    original=reference.query(q64,b64,seeds[:1])
    errors={}
    for key in ('values','responses'):
        a=getattr(response,key)[:1].double();b=getattr(original,key)
        delta=a-b
        relative=float(delta.norm()/b.norm().clamp_min(1e-12));absolute=float(delta.abs().max())
        errors[key]=dict(relative=relative,maximum_absolute=absolute)
        if relative>p['path_relative_tolerance'] or absolute>p['path_absolute_tolerance']:
            raise ValueError(f'Float32 full100fs {key} agreement failed: {errors[key]}')
    return dict(physical_100fs_float64_agreement=errors,execution_audit_max_absolute=audit,
        value_branch_seconds=value_seconds,response_branch_seconds=ad_seconds,
        simulation_profile=p,numerical_precision='float32',
        approximate_collection_hours=((32*25+24*32)*value_seconds+48*8*ad_seconds)/3600,
        timing_scope='amortized batch throughput; one-shot audits and setup can increase total time')
