"""Large statistical batches, cosine warmup, online tracking and exact continuation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback
import numpy as np
import torch
from torch import nn
from src.data.structural_pretraining.batches import Release,collate,move
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.models.encoders.structural import StructuralModel,ATOMIC_NUMBERS,normalized_readout,ARCHITECTURE_REVISION
from src.project_runtime.paths import resolve_path
from src.training_methods.structural_pretraining.objective import Objective,PHYSICAL_BLOCKS,TDA_BLOCKS,block_errors
from src.training_methods.structural_pretraining.train import implementation,selection_rows,prepare_batch,target_batch
from src.experiment_runner.metric_docs import write_metric_table
from .data import CausalRelease
from .health import spread,training_means,check_learning,check_regression
from .compilation import compile_encoder,compilation_counters
from .normalization import calibrate_heads
from .initialization import initialize_structural,require_complete_tda,continue_mixed_objective
from .tracking import Dashboard
from . import mixed as mixed_training
from src.models.encoders.mixed_gatr import MixedSnapshotGATr,MIXED_ARCHITECTURE_REVISION
from src.models.encoders.mixed_mace import MixedSnapshotMACE,MACE_BOND_REVISION,training_encode
from src.data.structural_pretraining.bond_order import bond_order_errors,bond_order_magnitudes


def configure():
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def learning_rate(step,total,peak,warmup_fraction=.1,minimum_ratio=.01):
    """One-based update index; warmup reaches peak, final update reaches minimum."""
    if not 1<=step<=total or total<2:raise ValueError('Learning-rate step outside training schedule')
    warmup=max(1,min(total-1,math.ceil(total*warmup_fraction)))
    if step<=warmup:return peak*step/warmup
    phase=(step-warmup)/(total-warmup)
    return peak*(minimum_ratio+(1-minimum_ratio)*.5*(1+math.cos(math.pi*phase)))


def update_budget(config,train_rows):
    batches=math.ceil(config['epochs']*train_rows/config['batch_size'])
    if config['phase']=='causal':
        # Three native batches then one broad structural replay batch.
        return batches+math.ceil(batches/3)
    return batches


def make_optimizer(model,peak,encoder_lr_multiplier):
    if not 0<encoder_lr_multiplier<=1:
        raise ValueError('Encoder LR multiplier must be in (0,1]')
    encoder=list(model.encoder.parameters());owned={id(p) for p in encoder}
    heads=[p for p in model.parameters() if id(p) not in owned]
    return torch.optim.AdamW([
        dict(params=encoder,lr=peak*encoder_lr_multiplier,lr_scale=encoder_lr_multiplier),
        dict(params=heads,lr=peak,lr_scale=1.)],weight_decay=1e-4)


class CausalModel(StructuralModel):
    def __init__(self,architecture,backend='cueq',history=False):
        super().__init__(architecture,backend,history=history)
        self.future=normalized_readout(128,256,3*229)
        nn.init.zeros_(self.future[-1].weight);nn.init.zeros_(self.future[-1].bias)


class CausalObjective(Objective):
    def __init__(self,manifest,method,physical_correlation_weight=0.):
        super().__init__(manifest['normalization'],method,physical_correlation_weight)
        self.register_buffer('future_mean',torch.tensor(manifest['forecast_normalization']['mean'],dtype=torch.float32))
        self.register_buffer('future_std',torch.tensor(manifest['forecast_normalization']['std'],dtype=torch.float32))

    def future_errors(self,prediction,raw):
        target=(raw-self.future_mean)/self.future_std
        shape=target.shape[:2]
        p=block_errors(prediction[...,:85].reshape(-1,85),target[...,:85].reshape(-1,85),PHYSICAL_BLOCKS).reshape(*shape,4)
        h=block_errors(prediction[...,85:].reshape(-1,144),target[...,85:].reshape(-1,144),TDA_BLOCKS).reshape(*shape,3)
        return p,h

    def forward(self,model,z,targets,temporal,delta):
        loss,terms=super().forward(model,z,targets,temporal,delta)
        if 'future_physical' in targets:
            raw=torch.cat((targets['future_physical'],targets['future_tda']),-1)
            pred=model.future(z[:len(z)//2]).reshape(-1,3,229)
            p,h=self.future_errors(pred,raw);future=p.mean()+.25*h.mean()
            loss=loss+future;terms.update(loss=loss,future=future,future_physical=p.mean(),future_tda=h.mean())
        return loss,terms


def cached_update(model,objective,batches,optimizer,temporal,delta,extra=None,precision='float32'):
    device=next(model.parameters()).device;optimizer.zero_grad(set_to_none=True)
    resident=[move(b,device) for b in batches];states=[];rngs=[]
    with torch.no_grad():
        for batch in resident:
            rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
                states.append(training_encode(model,batch).float())
    z=torch.cat(states).detach().requires_grad_(True);target=target_batch(resident,device)
    if isinstance(model,MixedSnapshotMACE):target['bond_order']=torch.cat([b['bond_order'] for b in resident])
    if extra is not None:target.update({k:torch.as_tensor(v,device=device) for k,v in extra.items()})
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
        loss,terms=objective(model,z,target,temporal,torch.as_tensor(delta,dtype=z.dtype,device=device))
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite shared loss: {terms}')
    loss.backward();derivative=z.grad.detach();offset=0
    for batch,(cpu,cuda) in zip(resident,rngs,strict=True):
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            torch.set_rng_state(cpu)
            if cuda is not None:torch.cuda.set_rng_state(cuda,device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
                current=training_encode(model,batch).float()
            current.backward(derivative[offset:offset+len(current)]);offset+=len(current)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    return {k:float(v.detach()) for k,v in terms.items()}|dict(gradient_norm=float(norm))


def selected_rows(release,seed):
    by_source={}
    for i in release.selection:by_source.setdefault(release.rows[i][2]['source'],[]).append(i)
    result=[]
    for source,rows in sorted(by_source.items(),key=lambda v:str(v[0])):
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(digest(str(source))[:8],16)]))
        result.extend(sorted(rng.choice(rows,min(64,len(rows)),replace=False).tolist()))
    return result


@torch.no_grad()
def evaluate(model,objective,release,config,means=None):
    model.eval();p=[];h=[];states=[];fp=[];fh=[];sources=[];bp=[];bh=[]
    bond=[];bond_zero=[];bond_magnitude=[];bond_prediction=[];bond_target=[]
    mace_bond=isinstance(model,MixedSnapshotMACE)
    predictions={name:[] for name in ('q','physical','tda')}
    ids=selected_rows(release,config['seed']);device=next(model.parameters()).device
    micro=min(config['microbatch_size'],64)
    for start in range(0,len(ids),micro):
        indices=ids[start:start+micro]
        samples=[release.observation(i,'anchor',config['history_frames']==3,config['architecture']=='mace') for i in indices]
        batch=move(collate(samples,config['architecture'],bond_order=mace_bond),device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
            encoded=training_encode(model,batch).float();z=encoded[:,:128]
            if config.get('batch_mode')=='mixed_triplets':
                domains=torch.as_tensor(mixed_training.group_ids(release,indices),device=device)
                heads={k:v.float() for k,v in model.heads(z,domains).items()}
            else:heads={k:v.float() for k,v in model.heads(z).items()}
        if mace_bond:
            predicted=model.bond_order(encoded[:,128:]);target=batch['bond_order']
            bond.append(bond_order_errors(predicted,target).cpu().numpy())
            bond_zero.append(bond_order_errors(torch.zeros_like(target),target).cpu().numpy())
            bond_magnitude.append((bond_order_magnitudes(predicted)-bond_order_magnitudes(target)).square().cpu().numpy())
            bond_prediction.append(predicted.cpu().numpy());bond_target.append(target.cpu().numpy())
        pe,he=objective.physical_errors(heads,batch)
        for name in predictions:predictions[name].append(heads[name].cpu().numpy())
        if means is not None:
            keys=[(release.rows[i][2]['material'],release.rows[i][2]['potential'],release.rows[i][2]['static']) for i in indices]
            baseline={name:torch.as_tensor(np.stack([means[k][name] for k in keys]),device=device,dtype=z.dtype)
                for name in ('physical','tda')}
            baseline={name:(values-getattr(objective,f'{name}_mean'))/getattr(objective,f'{name}_std')
                for name,values in baseline.items()}
            a,b=objective.physical_errors(baseline,batch);bp.append(a.cpu().numpy());bh.append(b.cpu().numpy())
        if not batch['tda_valid'].all():raise ValueError('Selection requires complete instantaneous TDA')
        p.append(pe.cpu().numpy());h.append(he.cpu().numpy());states.append(z.cpu().numpy())
        if config['phase']=='causal':
            f=release.futures(indices);raw=torch.as_tensor(np.concatenate((f['future_physical'],f['future_tda']),-1),device=device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                prediction=model.future(z).float().reshape(-1,3,229)
            a,b=objective.future_errors(prediction,raw);fp.append(a.cpu().numpy());fh.append(b.cpu().numpy())
        sources.extend(release.rows[i][2]['source'] for i in indices)
    sources=np.array(sources)
    def balanced(parts):
        values=np.concatenate(parts);return np.stack([values[sources==s].mean(0) for s in np.unique(sources)]).mean(0)
    pv,hv=balanced(p),balanced(h)
    metrics=dict(physical=float(pv.mean()),instantaneous_tda=float(hv.mean()),physical_blocks=dict(zip(PHYSICAL_BLOCKS,map(float,pv))),
        tda_blocks=dict(zip(TDA_BLOCKS,map(float,hv))),selection_sources=len(np.unique(sources)),selection_rows=len(ids),
        state_std_mean=spread(np.concatenate(states)),projector_std_mean=spread(np.concatenate(predictions['q'])),
        physical_prediction_std_mean=spread(np.concatenate(predictions['physical'])),
        tda_prediction_std_mean=spread(np.concatenate(predictions['tda'])))
    metrics['score']=metrics['physical']+.25*metrics['instantaneous_tda']
    metrics['present_score']=metrics['score']
    if bp:
        a,b=balanced(bp),balanced(bh)
        metrics['training_mean_score']=float(a.mean()+.25*b.mean())
        metrics['gain_over_training_mean']=metrics['training_mean_score']-metrics['present_score']
    if fp:
        a,b=balanced(fp),balanced(fh);metrics['future']={str(lag):dict(physical=float(a[k].mean()),tda=float(b[k].mean())) for k,lag in enumerate([.75,3.,9.])}
        metrics['score']+=float(a.mean()+.25*b.mean())
    predictions=dict(indices=np.array(ids),state=np.concatenate(states),physical_errors=np.concatenate(p),tda_errors=np.concatenate(h),source=sources)
    if bond:
        metrics.update(bond_order=float(balanced(bond).mean()),
            bond_order_blocks=dict(zip(('q4','q6'),map(float,balanced(bond)))),
            bond_order_zero_baseline=float(balanced(bond_zero).mean()),
            bond_order_magnitude_mse=dict(zip(('Q4','Q6'),map(float,balanced(bond_magnitude)))))
        predictions.update(bond_order_prediction=np.concatenate(bond_prediction),bond_order_target=np.concatenate(bond_target),
            bond_order_errors=np.concatenate(bond))
    model.train()
    return metrics,predictions


def atomic_checkpoint(path,model,objective,optimizer,step,best,identity,total):
    path=Path(path);temp=path.with_suffix('.building.pt')
    torch.save(dict(model=model.state_dict(),objective=objective.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,identity=identity,
        scheduler=dict(next_update=step+1,total_updates=total,**identity['config']['schedule']),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp);temp.replace(path)


def identity_for(config,release,replay,parent,continuation=None):
    code=implementation();code['files'].update({p:file_hash(p) for p in [__file__,'src/training_methods/shared_pretraining/data.py',
        'src/training_methods/shared_pretraining/health.py','src/training_methods/shared_pretraining/compilation.py',
        'src/training_methods/shared_pretraining/normalization.py',
        'src/training_methods/shared_pretraining/initialization.py',
        'src/training_methods/shared_pretraining/tracking.py',
        'src/training_methods/shared_pretraining/mixed.py','src/models/encoders/mixed_gatr.py',
        'src/models/encoders/mixed_mace.py']})
    mixed=config.get('batch_mode')=='mixed_triplets'
    mace_bond=mixed and config['architecture']=='mace'
    return dict(protocol='shared_pretraining_mixed_mace_bond_v9' if mace_bond else ('shared_pretraining_mixed_v8' if mixed else 'shared_pretraining_v6'),
        architecture_revision=MACE_BOND_REVISION if mace_bond else (MIXED_ARCHITECTURE_REVISION if mixed else ARCHITECTURE_REVISION),
        data=release.manifest['identity'],replay=None if replay is None else replay.manifest['identity'],
        implementation=code,parent_checkpoint_sha256=None if parent is None else file_hash(parent),
        continuation_checkpoint_sha256=None if continuation is None else file_hash(continuation),
        config={k:v for k,v in config.items() if k not in ('output','microbatch_size','wandb')})


def run(config,deadline):
    configure();root=resolve_path(config['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    with (technical/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute(config,deadline,root,technical)


def execute(config,deadline,root,technical):
    stop=False;tracking=None;step=0
    def signal_stop(signum,frame):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGUSR1,signal_stop)
    def status(state,**extra):
        value=dict(state=state,step=step,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),allocation=os.environ.get('SLURM_JOB_ID'),**extra)
        save_json(technical/'status.json',value);print(json.dumps(value),flush=True)
    try:
        if config['precision'] not in ('float32','bf16'):raise ValueError('Precision must be float32 or bf16')
        if config['precision']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('Selected GPU does not support BF16')
        torch.cuda.set_per_process_memory_fraction(config['memory_limit_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(config['seed']);np.random.seed(config['seed'])
        causal=config['phase']=='causal'
        mixed=config.get('batch_mode')=='mixed_triplets'
        if mixed and (causal or config['architecture'] not in ('gatr','mace') or config['history_frames']!=1 or 'parent' in config):
            raise ValueError('Mixed triplet run requires structural snapshot MACE/GATr initialized from scratch')
        if 'continue_from' in config and (not mixed or config['architecture']!='gatr'):
            raise ValueError('Objective continuation is defined only for mixed GATr')
        release=(CausalRelease(resolve_path(config['release'])) if causal else
                 Release(resolve_path(config['release']),materials=config['materials'],dynamic_only=mixed))
        if config.get('require_full_tda',False):require_complete_tda(release)
        replay=Release(resolve_path(config['replay_release']),materials=config['materials']) if causal else None
        means=training_means(replay if causal else release)
        parent=resolve_path(config['parent']) if 'parent' in config else None
        continuation=resolve_path(config['continue_from']) if 'continue_from' in config else None
        if causal:
            parent_status=json.loads((parent.parent/'status.json').read_text())
            if parent_status['state']!='complete':raise ValueError('Causal initialization requires a completed structural fit')
        identity=identity_for(config,release,replay,parent,continuation)
        if (technical/'identity.json').exists() and json.loads((technical/'identity.json').read_text())!=identity:
            raise ValueError('Changed scientific identity; use a new run')
        save_json(technical/'identity.json',identity)
        if (technical/'status.json').exists() and json.loads((technical/'status.json').read_text())['state']=='complete':return True
        if mixed:
            if config['architecture']=='mace':
                model=MixedSnapshotMACE(release.group_keys).cuda()
                objective=mixed_training.MACEBondObjective(release.manifest['normalization'],release.group_keys,
                    config['physical_correlation_weight'],config['backtracking_weight'],config['bond_order_weight']).cuda()
            else:
                model=MixedSnapshotGATr(release.group_keys).cuda()
                objective=mixed_training.MixedObjective(release.manifest['normalization'],release.group_keys,
                    config['physical_correlation_weight'],config['backtracking_weight']).cuda()
            save_json(technical/'sampling.json',dict(group_keys=[list(k) for k in release.group_keys],
                group_counts=mixed_training.quotas(release.group_weights,config['batch_size'],config['minimum_group_size']).tolist(),
                train_rows=[len(release.groups[k]) for k in release.group_keys],dynamic_only=True,
                curvature_frame_slots=[1,2,3],backtracking_updates='temporal_only',
                snapshot_views=dict(temporal=3,spatial=2),supervised_views=['anchor','spatial_or_future']))
        else:
            model=(CausalModel if causal else StructuralModel)(config['architecture'],history=config['history_frames']>1).cuda()
            objective=(CausalObjective(release.manifest,config['method'],config['physical_correlation_weight']) if causal else
                Objective(release.manifest['normalization'],config['method'],config['physical_correlation_weight'])).cuda()
        if causal:
            saved=torch.load(parent,map_location='cpu',weights_only=False)
            missing,unexpected=model.load_state_dict(saved['model'],strict=False)
            if unexpected or set(missing)!={f'future.{k}' for k in model.future.state_dict()}:raise ValueError('Parent encoder/head schema differs')
            if saved['identity']['data']!=replay.manifest['identity']:raise ValueError('Causal replay differs from structural parent data')
        elif parent is not None:
            saved=torch.load(parent,map_location='cpu',weights_only=False)
            parent_release=resolve_path(saved['identity']['config']['release'])
            parent_manifest=json.loads((parent_release/'manifest.json').read_text())
            if parent_manifest['scales']!=release.manifest['scales']:
                raise ValueError('Structural continuation changed fixed material coordinate scales')
            transfer=initialize_structural(model,objective,saved,config)
            save_json(technical/'initialization.json',dict(parent_sha256=file_hash(parent),**transfer))
        optimizer=make_optimizer(model,config['schedule']['peak'],config['encoder_lr_multiplier'])
        train_rows=sum(map(len,release.groups.values()));total=update_budget(config,train_rows);best=float('inf')
        resuming=(technical/'last.pt').exists()
        if resuming:
            saved=torch.load(technical/'last.pt',map_location='cuda:0',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Exact resume identity mismatch')
            model.load_state_dict(saved['model']);objective.load_state_dict(saved['objective']);optimizer.load_state_dict(saved['optimizer'])
            step=saved['step'];best=saved['best'];torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
            if saved['scheduler']['next_update']!=step+1 or saved['scheduler']['total_updates']!=total:raise ValueError('Scheduler/checkpoint step mismatch')
        elif continuation is not None:
            saved=torch.load(continuation,map_location='cpu',weights_only=False)
            transfer=continue_mixed_objective(model,objective,optimizer,saved,identity,total)
            step=saved['step']
            save_json(technical/'initialization.json',dict(parent_sha256=file_hash(continuation),**transfer))
        if config['compile_encoder']:
            sample=release.observation(release.groups[release.group_keys[0]][0],
                'anchor',config['history_frames']>1,config['architecture']=='mace')
            compile_encoder(model.encoder,move(collate([sample],config['architecture']),'cuda'),config['precision'])
        import wandb
        settings=config['wandb']
        tracking=wandb.init(entity=settings['entity'],project=settings['project'],id=settings['id'],name=settings['name'],group=settings['group'],
            resume='allow',mode='online',dir=str(technical),config=identity['config'],save_code=False,settings=wandb.Settings(init_timeout=60))
        if tracking.offline:raise RuntimeError('Online W&B logging is required')
        dashboard=Dashboard(tracking)
        save_json(technical/'wandb_run.json',dict(id=tracking.id,url=tracking.url,entity=tracking.entity,project=tracking.project))
        def prepare(k):
            if mixed:return mixed_training.prepare(release,k,config)
            is_replay=causal and k%4==3
            chosen=replay if is_replay else release
            cfg=config if not causal or is_replay else dict(config,method='lejepa')
            indices,temporal,delta,group=selection_rows(chosen,k,cfg)
            packed,*_=prepare_batch(chosen,indices,temporal,delta,config)
            return packed,temporal,delta,indices,group,is_replay,None if not causal or is_replay else release.futures(indices)
        best_present=float('inf')
        present_scores=[]
        if (technical/'validation.jsonl').exists():
            present_scores=[json.loads(line)['present_score'] for line in (technical/'validation.jsonl').read_text().splitlines()]
            best_present=min(present_scores)
        def save_selection():
            nonlocal best,best_present
            calibration=(mixed_training.calibrate(model,release,config) if mixed else
                         calibrate_heads(model,replay if causal else release,config))
            save_json(technical/'head_calibration.json',dict(step=step,**calibration))
            metrics,predictions=evaluate(model,objective,release,config,means)
            best_present=min(best_present,metrics['present_score'])
            present_scores.append(metrics['present_score'])
            with (technical/'validation.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**metrics))+'\n')
            write_metric_table(dict(step=step,**metrics),root,family='shared_pretraining',name='selection')
            native_updates=step-step//4 if causal else step
            dashboard.validation(step,native_updates*config['batch_size']/train_rows,metrics)
            if metrics['score']<best:
                best=metrics['score'];atomic_checkpoint(technical/'best.pt',model,objective,optimizer,step,best,identity,total)
                np.savez(technical/'selection_predictions.npz',**predictions)
                torch.save(dict(architecture=config['architecture'],input_frames=config['history_frames'],atomic_numbers=ATOMIC_NUMBERS,
                    scales=release.manifest['scales'],state_dim=128,encoder=model.encoder.state_dict(),identity=identity,step=step),technical/'encoder.pt')
            check_learning(metrics,metrics['training_mean_score'],best_present,step,config['health'])
            check_regression(present_scores,metrics['training_mean_score'],step,config['health'])
        status('running',total_updates=total,train_rows=train_rows,device=torch.cuda.get_device_name(),microbatch=config['microbatch_size'],wandb=tracking.url)
        started=time.monotonic();last_validation=-1
        if not resuming and config.get('evaluate_initial',False):
            save_selection();last_validation=step
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(prepare,step)
            while step<total and not stop and time.time()<deadline-180:
                wait=time.monotonic();batches,temporal,delta,indices,group,is_replay,extra=future.result();wait=time.monotonic()-wait
                if step+1<total:future=pool.submit(prepare,step+1)
                rate=learning_rate(step+1,total,**config['schedule'])
                for param in optimizer.param_groups:param['lr']=rate*param['lr_scale']
                torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
                terms=cached_update(model,objective,batches,optimizer,temporal,delta,extra,config['precision']);step+=1
                native_batches=step-step//4 if causal else step
                row=dict(step=step,epoch_equivalent=native_batches*config['batch_size']/train_rows,group=list(group),temporal=temporal,replay=is_replay,
                    lr=rate,encoder_lr=rate*config['encoder_lr_multiplier'],seconds=time.monotonic()-tick,input_wait_seconds=wait,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,indices=indices,**terms)
                with (technical/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                dashboard.record(row)
                if step%config['log_every']==0 or step==1:
                    dashboard.flush()
                if step%config['checkpoint_every']==0 or step==1:
                    atomic_checkpoint(technical/'last.pt',model,objective,optimizer,step,best,identity,total)
                    if config['compile_encoder']:save_json(technical/'compilation.json',compilation_counters())
                    status('running',total_updates=total,**{k:v for k,v in row.items() if k not in ('step','indices','group')})
                if step%config['validate_every']==0 or step==total:
                    dashboard.flush()
                    save_selection();last_validation=step
        dashboard.flush()
        atomic_checkpoint(technical/'last.pt',model,objective,optimizer,step,best,identity,total)
        # Export a validated candidate even if an allocation ends before the first interval.
        if step>0 and last_validation!=step and time.time()<deadline-90:save_selection()
        atomic_checkpoint(technical/'last.pt',model,objective,optimizer,step,best,identity,total)
        complete=step==total
        status('complete' if complete else 'checkpointed',total_updates=total,session_seconds=time.monotonic()-started,best_selection_score=best if np.isfinite(best) else None)
        tracking.summary['training_complete']=complete;tracking.finish();return complete
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc())
        if tracking is not None:tracking.finish(exit_code=1)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--deadline-utc',required=True)
    args=p.parse_args();run(json.loads(Path(args.config).read_text()),datetime.fromisoformat(args.deadline_utc).timestamp())
