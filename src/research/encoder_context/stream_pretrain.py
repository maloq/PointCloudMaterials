"""Geometry-only structural initialization from bounded, material-normalized shards."""
import json
import fcntl
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.data.structural_pretraining.native_dataset import NativeStructuralDataset
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.mace_epi.objective import Objective
from src.research.structural_state.common import sha,write_json,save_checkpoint
from src.research.supervised_onset.model import CapacityEncoder
from src.research.supervised_onset.tracking import tracked_run
from .geometry import graph,physical_targets


def run(study,method,device,deadline):
    c=study.config;settings=c['pretraining'];root=study.root/'pretraining'/method;technical=root/'technical'
    technical.mkdir(parents=True,exist_ok=True);done=technical/'complete.json'
    checkpoint=technical/f'epoch-{settings["epochs"]:03d}.pt'
    milestones=settings.get('checkpoint_epochs',[settings['epochs']])
    if any(e<0 or e>settings['epochs'] for e in milestones):raise ValueError('Checkpoint outside epoch budget')
    treatment=settings.get('objective',{}).get('treatment','epi-variance' if method=='epi_variance' else 'vicreg')
    is_epi=treatment=='epi-variance' and method!='physical'
    if done.exists():
        result=json.loads(done.read_text())
        if result['identity']!=study.identity or sha(checkpoint)!=result['sha256']:
            raise ValueError('Completed streaming pretraining changed')
        return result
    normalization=c['structural_dataset']['normalization']
    train=NativeStructuralDataset(c['structural_dataset']['root'],'train',paired=method!='physical',normalization=normalization)
    selection=NativeStructuralDataset(c['structural_dataset']['root'],'selection',paired=method!='physical',normalization=normalization)
    if train.identity!=c['structural_dataset']['identity']:
        raise ValueError('Structural release changed')
    if c['batch_size']!=c['microbatch']:
        raise ValueError('Paired covariance objectives require the declared full batch; use equal batch and microbatch')
    torch.manual_seed(c['seed']);chunk=c['microbatch'];n=len(train)
    encoder_config=dict(c['encoder'],d0=2.8,n_ref=80.)
    encoder=CapacityEncoder(**encoder_config).to(device)
    decoder=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,32)).to(device)
    objective=None if method=='physical' else Objective(treatment,
        settings.get('objective',{}).get('epi_weight',.1),
        alignment_weight=settings.get('objective',{}).get('alignment_weight',25/51)).to(device)
    def load(dataset,ids):
        values=dataset.batch(ids)
        return {key:torch.as_tensor(value,device=device) for key,value in values.items()}
    def encode(model,values,domain='hot'):
        return model(graph(values[domain],model))
    initial=np.random.default_rng(c['seed']+11).choice(n,min(8192,n),replace=False)
    pooled=[];targets=[]
    with torch.no_grad():
        for begin in range(0,len(initial),chunk):
            values=load(train,initial[begin:begin+chunk])
            pooled.append(encoder.pooled_graph(graph(values['hot'],encoder)))
            if method=='physical':targets.append(physical_targets(values['hot']))
        pooled=torch.cat(pooled)
        encoder.pooled_mean.copy_(pooled.mean(0));encoder.pooled_scale.copy_(pooled.std(0,unbiased=False).clamp_min(1e-5))
        if targets:
            targets=torch.cat(targets);mean=targets.mean(0);scale=targets.std(0,unbiased=False).clamp_min(1e-5)
        else:mean=scale=None
    if 'shared_initialization' in c:
        shared=Path(c['shared_initialization']);shared.parent.mkdir(parents=True,exist_ok=True)
        with shared.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            contract=dict(seed=c['seed'],encoder_config=encoder_config,normalization_rows=initial.tolist(),
                observation_identity=train.observation_identity)
            if shared.exists():
                initial_state=torch.load(shared,map_location=device,weights_only=False)
                if initial_state['contract']!=contract:raise ValueError('Matched initialization contract changed')
                encoder.load_state_dict(initial_state['encoder'],strict=True)
            else:save_checkpoint(shared,dict(contract=contract,encoder=encoder.state_dict()))
    # The Epi reference is fixed at initialization; compute it per batch so the
    # release and reservoir never need to reside together in GPU memory.
    reference=None;projection=None
    if is_epi:
        # Rebuild cuEq runtime graphs; deepcopy does not preserve their device
        # buffers or fused input bindings. Keep the scientific RNG stream fixed.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            reference=CapacityEncoder(**encoder_config).to(device)
        reference.load_state_dict(encoder.state_dict())
        reference.eval().requires_grad_(False)
        projection=torch.randn(128,64,device=device,generator=torch.Generator(device=device).manual_seed(c['seed']+37))/math.sqrt(128)
        from src.training_methods.neighborhood_jepa.regularization.objective import epiplexity
        scores=[]
        with torch.no_grad():
            for ids in np.array_split(initial[:min(4*chunk,len(initial))],4):
                values=load(train,ids)
                for domain in ('hot','cold'):
                    z=encode(reference,values,domain);scores.append(epiplexity(z,z@projection))
            objective.epi_initial_scale.copy_(torch.stack(scores).mean())
            if not torch.isfinite(objective.epi_initial_scale) or objective.epi_initial_scale<=1e-6:
                raise FloatingPointError('Degenerate fixed-reference epiplexity scale')
    sample=load(train,np.arange(min(n,chunk)))
    if c['runtime']['compile']:
        compile_spatial_encoder(encoder,graph(sample['hot'],encoder))
        if reference is not None:compile_spatial_encoder(reference,graph(sample['hot'],reference))
    parameters=list(encoder.parameters())+(list(decoder.parameters()) if method=='physical' else [])
    optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=1e-4)
    last=technical/'last.pt';start=0;steps=math.ceil(n/chunk);total=steps*settings['epochs']
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if (saved['identity']!=study.identity or saved['structural_identity']!=train.identity
                or saved['observation_identity']!=train.observation_identity):
            raise ValueError('Streaming resume belongs to another study/release')
        encoder.load_state_dict(saved['encoder']);decoder.load_state_dict(saved['decoder'])
        if objective is not None:objective.load_state_dict(saved['objective'])
        if reference is not None:reference.load_state_dict(saved['reference'])
        if is_epi and 'reference_projection' in saved:
            if not torch.equal(projection,saved['reference_projection']):raise ValueError('Reference projection changed')
        mean,scale=saved['target_mean'],saved['target_scale']
        optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    def save(step,path):
        save_checkpoint(path,dict(identity=study.identity,structural_identity=train.identity,
            observation_identity=train.observation_identity,coordinate_normalization=normalization,
            release_identity=c['fixed_dataset']['identity'],method=method,step=step,epoch=step/steps,
            encoder=encoder.state_dict(),decoder=decoder.state_dict(),encoder_config=encoder_config,
            objective=None if objective is None else objective.state_dict(),
            reference=None if reference is None else reference.state_dict(),
            reference_projection=projection,
            target_mean=mean,target_scale=scale,optimizer=optimizer.state_dict(),config=c,
            normalization_rows=initial.tolist(),normalization='8192 fixed train-only rows; material-cutoff normalized geometry in Al units',
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            selection=f'fixed epoch {settings["epochs"]}; no crystallization labels',
            sample_order=dict(seed=c['seed'],algorithm='NativeStructuralDataset.epoch; 16-shard blocks',
                rows=n,batch_size=chunk)))
        if path.name.startswith('epoch-'):
            write_json(path.with_suffix('.json'),dict(identity=study.identity,sha256=sha(path),
                method=method,epoch=step/steps,release_identity=c['fixed_dataset']['identity']))
    def loss(values):
        z=encode(encoder,values)
        if method=='physical':
            with torch.no_grad():target=(physical_targets(values['hot'])-mean)/scale
            error=(decoder(z)-target).square()
            value=(error[:,:24].mean()+error[:,24:26].mean()+error[:,26:].mean())/3
            return value,{'physical_mse':value.detach()}
        cold=encode(encoder,values,'cold');target={'index':np.arange(len(z))}
        if is_epi:
            with torch.no_grad():
                target['reservoir']=torch.stack([encode(reference,values,d)@projection for d in ('hot','cold')],1)
        return objective(None,torch.stack((z,cold),1).reshape(-1,128),target)
    tracked=SimpleNamespace(root=root,technical=technical,identity=study.identity,config=dict(c,
        branch='self_supervised',wandb=dict(c['wandb'],display_name=f'Structural | {method} | seed {c["seed"]} | batch {chunk}')))
    if start==0 and 0 in milestones and not (technical/'epoch-000.pt').exists():save(0,technical/'epoch-000.pt')
    completed=start
    try:
        with tracked_run(tracked,f'pretrain-{method}') as tracking:
            tracking.summary.update({'data/train_windows':n,'data/validation_windows':len(selection),
                'data/structural_identity':train.identity,'data/materials':sorted({s['material'] for s in train.shards}),
                'data/observation_identity':train.observation_identity,'model/inputs':'geometry only; constant atom channel',
                'data/coordinate_normalization':'fixed training material cutoff; Al reference',
                'training/epochs_requested':settings['epochs'],'checkpoint/selection_rule':f'fixed epoch {settings["epochs"]}; label-free',
                'model/encoder_parameters':sum(p.numel() for p in encoder.parameters()),
                'data/sampling':'each row once per epoch; shuffled 16-shard blocks; partial batch retained'})
            for epoch in range(start//steps,settings['epochs']):
                skip=start%steps if epoch==start//steps else 0
                for batch_index,ids in train.epoch(chunk,epoch,c['seed'],skip):
                    step=epoch*steps+batch_index
                    if time.time()>deadline-300:
                        save(step,last);raise TimeoutError('Streaming pretraining checkpointed before allocation deadline')
                    values=load(train,ids);encoder.train();decoder.train();optimizer.zero_grad(set_to_none=True)
                    factor=.05+.95*.5*(1+math.cos(math.pi*step/total))
                    for group in optimizer.param_groups:group['lr']=settings['learning_rate']*min((step+1)/128,1)*factor
                    value,terms=loss(values)
                    if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite {method} at update {step}')
                    value.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
                    completed=step+1
                    if completed%32==0:
                        record={'optimizer_update':completed,'train/epoch':completed/steps,'train/objective':float(value.detach()),
                            'train/gradient_norm':float(norm),'train/learning_rate':optimizer.param_groups[0]['lr']}
                        record.update({f'train/{k}':float(v.detach()) for k,v in terms.items()})
                        tracking.log(record)
                        with (technical/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    if completed%256==0:save(completed,last)
                encoder.eval();decoder.eval();score=0.
                with torch.no_grad():
                    for begin in range(0,len(selection),chunk):
                        ids=np.arange(begin,min(begin+chunk,len(selection)));value,_=loss(load(selection,ids))
                        score+=len(ids)*float(value)
                score/=len(selection)
                if not math.isfinite(score):raise FloatingPointError('Nonfinite structural validation')
                tracking.log({'optimizer_update':completed,'validation/objective':score})
                save(completed,last)
                if epoch+1 in milestones:save(completed,technical/f'epoch-{epoch+1:03d}.pt')
                write_json(technical/'state.json',dict(state='training',completed_epochs=epoch+1,steps=completed))
                print(json.dumps(dict(method=method,epoch=epoch+1,validation=score)),flush=True)
            if completed!=total:raise ValueError('Incomplete streaming epoch budget')
            save(completed,checkpoint)
            result=dict(state='complete',identity=study.identity,structural_identity=train.identity,
                epochs=settings['epochs'],steps=completed,sha256=sha(checkpoint),method=method,train_rows=n)
            tracking.summary.update({'training/completed_epochs':settings['epochs'],'checkpoint/fixed_epoch':settings['epochs']})
            write_json(done,result)
            return result
    finally:
        train.close();selection.close()
