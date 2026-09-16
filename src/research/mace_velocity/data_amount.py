"""Matched-update, end-to-end native MACE learning curves on an allocated H100."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import check_metric_docs
from src.experiment_runner.registry import sha256, write_json
from .data_amount_data import batch, load
from .data_amount_eval import export, extract, summarize
from .data_amount_model import calibrate, initialize, normalized_targets, objective, replay
from .train import encode


def optimizer_for(config,model,heads,directions):
    return torch.optim.AdamW([
        dict(params=[p for p in model.mace.parameters() if p.requires_grad],lr=config['backbone_learning_rate']),
        dict(params=list(model.activity.parameters())+list(model.flow.parameters())+list(heads.parameters()),
             lr=config['head_learning_rate']),
        dict(params=directions.parameters(),lr=config['head_learning_rate'])],weight_decay=.0001)


def clip(model,heads,directions):
    torch.nn.utils.clip_grad_norm_(model.mace.parameters(),1.,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(list(model.activity.parameters())+list(model.flow.parameters())+
                                   list(heads.parameters()),5.,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(directions.parameters(),5.,error_if_nonfinite=True)


def verification(config, model, heads, directions, sources, core, norm, device):
    clouds,y,times=batch(sources,core[:config['batch_sources']],np.zeros(config['batch_sources'],int))
    targets=normalized_targets(y,norm,device); tt=torch.as_tensor(times,dtype=torch.float64,device=device)
    optimizer=optimizer_for(config,model,heads,directions)
    parameters=list(model.named_parameters())+[(f'head.{n}',p) for n,p in heads.named_parameters()]+[
        (f'direction.{n}',p) for n,p in directions.named_parameters()]
    optimizer.zero_grad(set_to_none=True)
    z=encode(config,model,clouds,device,gradients=True)
    value,_=objective(config,heads,directions,z,targets,tt,1.);value.backward()
    expected={n:p.grad.detach().cpu().clone() for n,p in parameters if p.grad is not None}
    del z,value
    optimizer.zero_grad(set_to_none=True)
    replay(config,model,heads,directions,clouds,targets,tt,1.,device)
    numerator=denominator=0.
    for n,p in parameters:
        if n in expected:
            if p.grad is None: raise AssertionError(f'Missing replay gradient: {n}')
            a=p.grad.detach().cpu().double(); b=expected[n].double()
            numerator+=float((a-b).square().sum());denominator+=float(b.square().sum())
    gradient_error=numerator/denominator
    if gradient_error>1e-6: raise AssertionError(f'Direct/replayed gradient mismatch: {gradient_error}')
    trainable=[(n,p) for n,p in model.mace.named_parameters() if p.grad is not None and p.grad.square().sum()>0]
    if not trainable: raise AssertionError('Actual MACE backbone receives no gradient')
    interactions=[(n,p) for n,p in trainable if '.interactions.' in n]
    if not interactions: raise AssertionError('MACE message-passing interaction weights receive no gradient')
    name,parameter=interactions[0];before=parameter.detach().clone()
    clip(model,heads,directions);optimizer.step()
    change=float((parameter.detach()-before).abs().max())
    if change==0: raise AssertionError(f'MACE parameter did not update: {name}')
    durations=[]
    for step in range(6):
        torch.cuda.synchronize(); start=time.monotonic(); optimizer.zero_grad(set_to_none=True)
        replay(config,model,heads,directions,clouds,targets,tt,1.,device)
        clip(model,heads,directions);optimizer.step();torch.cuda.synchronize()
        durations.append(time.monotonic()-start)
    # Test actual trained-model invariants, preserving the center atom at index 0.
    x,v=clouds[0];rng=np.random.default_rng(config['core_seed'])
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)));order=np.r_[0,rng.permutation(np.arange(1,len(x)))].astype(int)
    variants=[(x,v),(x,v),((x@rotation).astype(np.float32),(v@rotation).astype(np.float32)),
              (x[order],v[order]),(x,(v+np.array([3.,-2.,7.])).astype(np.float32)),(x,-v)]
    zz=encode(config,model,variants,device).cpu().numpy()
    errors={k:float(np.linalg.norm(zz[i]-zz[0])/max(np.linalg.norm(zz[0]),1e-12))
            for k,i in [('repeat',1),('rotation',2),('permutation',3),('velocity_boost',4)]}
    if max(errors.values())>2e-4: raise AssertionError(f'Encoder invariance failed: {errors}')
    np.testing.assert_allclose(zz[5,:288],zz[0,:288],atol=2e-5,rtol=2e-4)
    np.testing.assert_allclose(zz[5,288:],-zz[0,288:],atol=2e-5,rtol=2e-4)
    return dict(state='complete',replay_gradient_relative_squared_error=gradient_error,
        updated_backbone_parameter=name,backbone_parameter_max_change=change,
        symmetry_relative_errors=errors,seconds_per_update=durations,
        median_seconds_per_update=float(np.median(durations[2:])),batch_clouds=len(clouds),
        trainable_mace_parameters=sum(p.numel() for p in model.mace.parameters() if p.requires_grad),
        device=torch.cuda.get_device_name(device))


def atomic_checkpoint(path,payload):
    temporary=path.with_suffix('.building.pt');torch.save(payload,temporary);temporary.replace(path)


def train_fit(config,sources,selection,norm,fit,root,device,identity,deadline,steps):
    name=f"n{fit['count']:03d}-seed{fit['seed']}"; folder=root/'technical'/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'last.pt').exists(): raise FileExistsError(f'Preserve completed/partial fit: {folder}')
    model,heads,directions=initialize(config,norm,fit['seed'],device)
    optimizer=optimizer_for(config,model,heads,directions)
    rng=np.random.default_rng(fit['seed']); history=[];best=float('inf');best_step=0
    started=time.monotonic();seen=set();observed=set();last_step=0
    def save(filename,step):
        atomic_checkpoint(folder/filename,dict(protocol=config['protocol'],config=config,fit=fit,step=step,
            encoder_state=model.state_dict(),head_state=heads.state_dict(),direction_state=directions.state_dict(),
            optimizer_state=optimizer.state_dict(),normalization=norm,identity=identity,
            numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device),history=history,best_validation=best,best_step=best_step,
            representation='Native pooled MACE 256 structure + 32 activity + 16 flow; no fitted embedding map'))
    for step in range(steps+1):
        if time.monotonic()>deadline-120:
            save('last.pt',last_step)
            raise TimeoutError(f'Wall budget reached in {name}; retained exact state at update {last_step}')
        if step:
            ids=rng.choice(fit['source_ids'],config['batch_sources'],replace=False)
            starts=rng.integers(0,7,size=len(ids))
            seen.update(int(i) for i in ids)
            observed.update((int(sid),c,t) for sid,start in zip(ids,starts,strict=True)
                            for c in range(4) for t in range(int(start),int(start)+3))
            clouds,y,times=batch(sources,ids,starts)
            targets=normalized_targets(y,norm,device)
            tt=torch.as_tensor(times,dtype=torch.float64,device=device)
            ramp=float(np.clip((step-config['warmup_steps'])/config['ramp_steps'],0,1))
            model.train(); heads.train();directions.train(); optimizer.zero_grad(set_to_none=True)
            metrics=replay(config,model,heads,directions,clouds,targets,tt,ramp,device)
            clip(model,heads,directions);optimizer.step();last_step=step
            if step%50==0:
                status=dict(state='training',fit=name,step=step,total_steps=steps,
                            elapsed_seconds=time.monotonic()-started,training=metrics)
                write_json(root/'technical/status.json',status);print('NATIVE ENCODER',status,flush=True)
        if step%config['validation_every']==0 or step==steps:
            validation=extract(config,model,heads,directions,sources,selection['validation_source_ids'],norm,device)
            score=validation['loss']['selection_score']
            history.append(dict(step=step,validation=validation['loss'],elapsed_seconds=time.monotonic()-started))
            if score<best:
                best=score;best_step=step;save('best.pt',step)
            save('last.pt',step); write_json(folder/'history.json',history)
            print('NATIVE VALIDATION',name,step,validation['loss'],flush=True)
            del validation
    if steps>=100 and len(seen)!=fit['count']:
        raise AssertionError(f'Training never sampled all selected sources in {name}: {len(seen)}')
    # The primary learning curve uses exactly the same final update in every fit.
    # Preserve validation-selected checkpoints separately for later deployment.
    reference=extract(config,model,heads,directions,sources,selection['core_source_ids'],norm,device)
    rows=[];source_rows=[]
    for split,ids in [('train',fit['source_ids']),('validation',selection['validation_source_ids']),
                      ('development_test',selection['development_test_source_ids'])]:
        result=extract(config,model,heads,directions,sources,ids,norm,device)
        rr,ss=summarize(config,result,reference,fit,split)
        for row in rr:
            row.update(completed_steps=steps,evaluated_step=steps,validation_selected_step=best_step,
                       unique_training_sources_seen=len(seen),
                       unique_training_observations_seen=len(observed),
                       cloud_presentations=steps*config['batch_sources']*4*3,
                       equivalent_passes=steps*config['batch_sources']*4*3/(36*fit['count']))
        rows.extend(rr);source_rows.extend(ss)
        np.savez(folder/f'{split}-features.npz',**{k:v for k,v in result.items() if k!='loss'})
    write_json(folder/'status.json',dict(state='complete',steps=steps,selected_step=best_step,
        elapsed_seconds=time.monotonic()-started,unique_sources_seen=len(seen),unique_observations_seen=len(observed),
        best_checkpoint_sha256=sha256(folder/'best.pt')))
    del model,heads,directions,optimizer,reference
    torch.cuda.empty_cache()
    return rows,source_rows


def run(config,args,smoke=False):
    torch.set_num_threads(config['cpu_threads']);torch.cuda.set_device(args.device)
    if socket.gethostname()!=config['runtime']['node'] or os.environ.get('SLURM_JOB_ID')!=str(config['runtime']['gpu_job_id']):
        raise RuntimeError('This recipe must run inside the specified allocated H100 job/node')
    if 'H100' not in torch.cuda.get_device_name(args.device): raise RuntimeError('Expected the allocated H100')
    contracts=check_metric_docs()
    root=Path(config['output']); root=root/'technical/smoke' if smoke else root
    (root/'technical').mkdir(parents=True,exist_ok=True)
    if (root/'technical/run.json').exists(): raise FileExistsError(f'Preserve existing study: {root}')
    start=time.monotonic(); deadline=start+config['runtime']['maximum_seconds']
    plan,sources=load(config);selection=plan['selection']
    identity=dict(implementation=contracts['mace_data_amount']['files'],
        config=config,cloud_cache_sha256=sha256(Path(config['cache'])/'complete.json'),
        foundation_sha256=config['foundation_sha256'])
    for relative,digest in identity['implementation'].items():
        path=Path(relative)
        if sha256(path)!=digest: raise ValueError(f'Implementation changed while snapshotting: {path}')
        target=root/'technical/source-snapshot'/path
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    (root/'technical/source-snapshot/resolved-config.json').write_text(json.dumps(config,indent=2)+'\n')
    write_json(root/'technical/run.json',dict(identity=identity,smoke=smoke,pid=os.getpid(),node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),
        started_at=datetime.now(timezone.utc).isoformat(),selection=selection))
    write_json(root/'technical/status.json',dict(state='calibrating',smoke=smoke))
    model,heads,directions=initialize(config,dict(feature_mean=np.zeros(256),feature_scale=np.ones(256)),config['seeds'][0],args.device)
    norm=calibrate(config,model,sources,selection['core_source_ids'],args.device)
    del model,heads,directions
    model,heads,directions=initialize(config,norm,config['seeds'][0],args.device)
    np.savez(root/'technical/normalization.npz',**norm)
    report=verification(config,model,heads,directions,sources,selection['core_source_ids'],norm,args.device)
    write_json(root/'technical/verification.json',report);print('NATIVE VERIFIED',report,flush=True)
    del model,heads,directions;torch.cuda.empty_cache()
    fits=selection['fits']
    if smoke: fits=[fits[0],fits[len(config['training_sources'])-1]]
    steps=4 if smoke else config['steps']
    estimate=len(fits)*steps*report['median_seconds_per_update']
    if not smoke and estimate>config['runtime']['maximum_seconds']-900:
        raise RuntimeError(f'Actual update benchmark predicts {estimate:.1f}s before evaluation; reduce common step budget explicitly')
    rows=[];source_rows=[]
    for fit in fits:
        rr,ss=train_fit(config,sources,selection,norm,fit,root,args.device,identity,deadline,steps)
        rows.extend(rr);source_rows.extend(ss);export(config,rows,source_rows,root)
        write_json(root/'technical/partial-results.json',dict(completed_fits=len(rows)//6,total_fits=len(fits),rows=rows))
    write_json(root/'technical/status.json',dict(state='complete',smoke=smoke,fits=len(fits),steps_per_fit=steps,
        elapsed_seconds=time.monotonic()-start,results='tables/quality.csv'))
    print('NATIVE STUDY COMPLETE',root,flush=True)
