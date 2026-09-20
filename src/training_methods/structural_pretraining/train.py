"""Deadline-aware detached training with exact global-batch gradient caching."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.data.structural_pretraining.batches import Release,collate,move
from src.models.encoders.structural import StructuralModel,ATOMIC_NUMBERS,ARCHITECTURE_REVISION
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_table
from .objective import Objective,PHYSICAL_BLOCKS,TDA_BLOCKS


def implementation():
    paths=[p for folder in ('src/data/structural_pretraining','src/training_methods/structural_pretraining')
           for p in Path(folder).glob('*.py')]
    paths += [Path(p) for p in ('src/models/encoders/structural.py','src/models/encoders/mace_causal.py',
        'src/models/encoders/axial_gatr.py','src/models/encoders/mace_backend.py','src/models/encoders/structural_precision.py',
        'src/models/encoders/compensated_bf16.py','src/models/encoders/equivariant_bond.py','src/training_methods/shared_pretraining/normalization.py')]
    return dict(files={str(p):file_hash(p) for p in paths},
        versions={name:importlib.metadata.version(name) for name in ('torch','triton','mace-torch','cuequivariance',
            'cuequivariance-torch','cuequivariance-ops-torch-cu13','GATr','lejepa')},
        dependencies={name:json.loads(importlib.metadata.distribution(name).read_text('direct_url.json')) for name in ('GATr','lejepa')})


def selection_rows(release,step,config):
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],step]))
    group=release.group_keys[int(rng.choice(len(release.group_keys),p=release.group_weights))]
    rows=release.groups[group]
    if len(rows)<config['batch_size']:raise ValueError(f'Group too small for full statistical batch: {group}')
    indices=rng.choice(rows,config['batch_size'],replace=False).tolist()
    temporal=not group[2] and (config['method']=='lejepa' or bool(rng.integers(2)))
    delta=[0. if group[2] else float(release.arrays[release.rows[i][0]]['times'][3]) for i in indices]
    return indices,temporal,delta,group


def prepare_batch(release,indices,temporal,delta,config):
    architecture=config['architecture']; history=config['history_frames']>1; mace=architecture=='mace'
    partner='future' if temporal else 'spatial'
    observations=[release.observation(i,'anchor',history,mace) for i in indices]
    observations += [release.observation(i,partner,False,mace) for i in indices]
    micro=config['microbatch_size']; batches=[]
    # Keep the two endpoints separate: causal inputs can have T=3, targets T=1.
    for base in (0,len(indices)):
        for start in range(base,base+len(indices),micro):
            batch=collate(observations[start:min(start+micro,base+len(indices))],architecture)
            batches.append({k:v.pin_memory() for k,v in batch.items()})
    return batches,temporal,delta,indices


def target_batch(batches,device):
    return {k:torch.cat([b[k] for b in batches]).to(device) for k in ('physical','tda','tda_valid')}


def cached_update(model,objective,batches,optimizer,temporal,delta):
    """Differentiate heads once over all anchors, then recompute encoder chunks.

    Full-batch VICReg/SIGReg sees the concatenated states. This is not an average
    of microbatch regularizers. CPU/CUDA RNG states make stochastic replay exact.
    """
    device=next(model.parameters()).device;optimizer.zero_grad(set_to_none=True)
    resident=[move(b,device) for b in batches]
    states=[];rngs=[]
    with torch.no_grad():
        for batch in resident:
            rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
            states.append(model.encoder(batch))
    z=torch.cat(states).detach().requires_grad_(True)
    target=target_batch(resident,device)
    dt=torch.as_tensor(delta,dtype=z.dtype,device=device)
    if dt.shape!=(len(z)//2,):raise ValueError('Expected one actual time delta per pair')
    loss,terms=objective(model,z,target,temporal,dt)
    if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite full-batch loss: {terms}')
    loss.backward();derivative=z.grad.detach();offset=0
    for batch,(cpu_rng,cuda_rng),original in zip(resident,rngs,states,strict=True):
        devices=[device.index] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
            current=model.encoder(batch)
            current.backward(derivative[offset:offset+len(current)])
            offset+=len(current)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return {k:float(v.detach()) for k,v in terms.items()}|dict(gradient_norm=float(norm))


@torch.no_grad()
def evaluate(model,objective,release,config):
    from src.training_methods.shared_pretraining.normalization import calibrate_heads
    calibrate_heads(model,release,dict(config,precision='float32'))
    model.eval();device=next(model.parameters()).device;p=[];h=[];z=[];sources=[]
    indices=release.selection
    for start in range(0,len(indices),config['microbatch_size']):
        ids=indices[start:start+config['microbatch_size']]
        observations=[release.observation(i,'anchor',config['history_frames']>1,config['architecture']=='mace') for i in ids]
        batch=move(collate(observations,config['architecture']),device)
        state=model.encoder(batch);heads=model.heads(state);pe,he=objective.physical_errors(heads,batch)
        if not bool(batch['tda_valid'].all()):raise ValueError('Selection TDA targets are incomplete')
        p.append(pe.cpu());h.append(he.cpu());z.append(state.cpu())
        sources.extend(release.rows[i][2]['source'] for i in ids)
    p=torch.cat(p).numpy();h=torch.cat(h).numpy();states=torch.cat(z).numpy();sources=np.array(sources)
    def aggregate(a):return np.stack([a[sources==s].mean(0) for s in np.unique(sources)]).mean(0)
    pv,hv=aggregate(p),aggregate(h)
    metrics=dict(score=float(pv.mean()+.25*hv.mean()),physical=float(pv.mean()),instantaneous_tda=float(hv.mean()),
        physical_blocks=dict(zip(PHYSICAL_BLOCKS,map(float,pv))),tda_blocks=dict(zip(TDA_BLOCKS,map(float,hv))),
        selection_sources=len(np.unique(sources)),selection_windows=len(indices),state_std_mean=float(states.std(0).mean()))
    model.train();return metrics,dict(indices=np.array(indices),state=states,physical_errors=p,tda_errors=h,source=sources)


def checkpoint(path,model,objective,optimizer,step,identity,best):
    path=Path(path);temp=path.with_suffix('.building.pt')
    torch.save(dict(model=model.state_dict(),objective=objective.state_dict(),optimizer=optimizer.state_dict(),
        step=step,identity=identity,best=best,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
    temp.replace(path)


def run(config,resume=False):
    root=resolve_path(config['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    lock=(technical/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    torch.set_num_threads(config['torch_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    release=Release(resolve_path(config['release']),materials=config['materials']);model=StructuralModel(config['architecture'],history=config['history_frames']>1).to(device)
    objective=Objective(release.manifest['normalization'],config['method']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    identity=dict(protocol='structural_neighbors_local_v10',architecture_revision=ARCHITECTURE_REVISION,
        data=release.manifest['identity'],implementation=implementation(),
        config={k:v for k,v in config.items() if k not in ('output','deadline_utc')})
    if (technical/'identity.json').exists() and json.loads((technical/'identity.json').read_text())!=identity:
        raise ValueError('Existing output belongs to a different scientific identity')
    save_json(technical/'identity.json',identity)
    step=0;best=float('inf')
    if resume:
        saved=torch.load(technical/'last.pt',map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('Exact-resume model/data/code identity differs')
        model.load_state_dict(saved['model']);objective.load_state_dict(saved['objective']);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];best=saved['best'];torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    elif (technical/'last.pt').exists():raise ValueError('Existing run requires explicit --resume')
    end=datetime.fromisoformat(config['deadline_utc']).timestamp();started=time.monotonic();last_validation=-1
    def prepare(k):
        ids,temporal,delta,group=selection_rows(release,k,config)
        return prepare_batch(release,ids,temporal,delta,config),group
    def status(state,**kwargs):
        value=dict(state=state,step=step,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            allocation=os.environ.get('SLURM_JOB_ID'),**kwargs)
        save_json(technical/'status.json',value);print(json.dumps(value),flush=True)
    status('running',device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(prepare,step)
        while step<config['updates'] and time.time()<end-120:
            wait_start=time.monotonic();(batches,temporal,delta,indices),group=future.result();wait=time.monotonic()-wait_start
            future=pool.submit(prepare,step+1)
            tick=time.monotonic();terms=cached_update(model,objective,batches,optimizer,temporal,delta);step+=1
            record=dict(step=step,group=list(group),temporal=temporal,delta_ps=delta,indices=indices,
                seconds=time.monotonic()-tick,input_wait_seconds=wait,**terms)
            with (technical/'updates.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if step%config['checkpoint_every']==0 or step==1:
                checkpoint(technical/'last.pt',model,objective,optimizer,step,identity,best)
                status('running',**{k:v for k,v in record.items() if k not in ('step','indices','delta_ps')},
                       delta_ps_range=[min(delta),max(delta)])
            if step%config['validate_every']==0 or step==config['updates']:
                metrics,predictions=evaluate(model,objective,release,config);last_validation=step
                with (technical/'validation.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**metrics))+'\n')
                write_metric_table(dict(step=step,**metrics),root,family='structural_pretraining',name='selection')
                if metrics['score']<best:
                    best=metrics['score'];checkpoint(technical/'best.pt',model,objective,optimizer,step,identity,best)
                    np.savez(technical/'selection_predictions.npz',**predictions)
                    torch.save(dict(architecture=config['architecture'],input_frames=config['history_frames'],
                        atomic_numbers=ATOMIC_NUMBERS,scales=release.manifest['scales'],state_dim=128,
                        encoder=model.encoder.state_dict(),identity=identity,step=step),technical/'encoder.pt')
    checkpoint(technical/'last.pt',model,objective,optimizer,step,identity,best)
    status('complete' if step==config['updates'] else 'checkpointed_deadline',elapsed_seconds=time.monotonic()-started,
           last_validation_step=last_validation,best_selection_score=best if np.isfinite(best) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    if 'materials' not in config or 'head_calibration_rows' not in config:
        raise ValueError('Current structural heads require explicit materials and head_calibration_rows; '
            'use the active shared_pretraining/al_stable recipes. Historical exact resumes require their frozen source.')
    if config['batch_size']%config['microbatch_size'] or config['batch_size']<2:
        raise ValueError('Full batch must be divisible by positive microbatch')
    if config['architecture']=='mace' and config['history_frames']!=1:raise ValueError('Requested MACE run is snapshot-only')
    if config['method']=='lejepa' and (config['architecture']!='gatr' or config['history_frames']!=3):
        raise ValueError('Requested JEPA run is GATr with three causal input frames')
    run(config,args.resume)


if __name__=='__main__':main()
