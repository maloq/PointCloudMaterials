"""Joint physical/event supervision with exact full-cohort ranking gradients."""
import copy
import json
import math
import time
import numpy as np
import torch

from src.research.structural_state.data import Corpus
from src.research.structural_state.model import block_error, calibrate_heads
from src.research.local_predictability.metrics import hazard_loss, cumulative_risk, weighted_scores
from src.research.trajectory_stability.spectrum import source_weights
from .common import write_json, save_checkpoint, sha, remaining
from .data import targets
from .model import Model, GraphBank
from .metrics import smooth_ap, horizon_index


def setup(study, name, device):
    c=study.config; a=study.arm(name); corpus=Corpus(study)
    values,scalers,conditions,risk,sources=targets(corpus)
    torch.manual_seed(c['seed'])
    ec=dict(c['encoder'],d0=corpus.manifest['d0'],n_ref=corpus.manifest['n_ref'])
    model=study.make_model(ec,a,conditions.shape[1]).to(device)
    bank=model.make_bank(study.graph_arrays(a),device)
    noisy=[]
    if a['noise']:
        manifest=json.loads((study.augmented/'manifest.json').read_text())
        for i in range(len(c['noise']['training_rms_fractions'])):
            p=study.augmented/f'view-{i}.npz'
            if sha(p)!=manifest['files'][p.name]:raise ValueError(f'Changed noisy input: {p}')
            with np.load(p) as arrays:noisy.append(model.make_bank(dict(arrays),device))
    return model,bank,noisy,corpus,{k:torch.as_tensor(v,device=device) for k,v in values.items()},scalers,torch.as_tensor(conditions,device=device),risk,sources,ec


@torch.no_grad()
def encode(model,bank,ids,chunk,pooled=False):
    model.eval()
    return torch.cat([(model.encoder.pooled_graph if pooled else model)(bank.batch(ids[s:s+chunk]))
                      for s in range(0,len(ids),chunk)])


def physical_loss(model,z,target,arm):
    loss=block_error(model.heads[arm['input']](z),target[arm['input']]).mean()
    loss=loss+.25*(model.heads['current'](z)-target['current']).square().mean()
    loss=loss+.25*(model.heads['future'](z)-target['future']).square().mean()
    if arm['teacher']:
        loss=loss+.5*block_error(model.heads['relaxed'](z),target['relaxed']).mean()
    return loss


def ranking_backward(model,bank,ids,conditions,labels,weights,chunk,temperature,coefficient,*,horizon_ps):
    """Gradient caching: full risk-set AP, replayed through microbatched MACE.

    No optimizer update between caching and replay; encoder has no dropout or
    batch normalization. Both hazard-head and encoder gradients are exact for
    this full-population loss (up to floating point execution order).
    """
    z=encode(model,bank,ids,chunk).detach().requires_grad_(True)
    probability=cumulative_risk(model.logits(z,conditions[ids]))[:,horizon_index(horizon_ps)]
    loss=coefficient*(1-smooth_ap(probability,labels,weights,temperature))
    loss.backward()
    gradients=z.grad.detach()
    for start in range(0,len(ids),chunk):
        (model(bank.batch(ids[start:start+chunk]))*gradients[start:start+chunk]).sum().backward()
    return float(loss.detach())


def step(model,bank,noisy,corpus,target,conditions,risk,sources,arm,config,rng,number,reference_variance):
    tc=config['training']; chunk=tc['microbatch']; batch=tc['batch_size']
    fit=corpus.split['fit']; ri=risk['fit']
    ids=rng.choice(fit,batch,p=source_weights(sources[fit]))
    events=rng.choice(ri,batch,p=source_weights(sources[ri]))
    event_bin=torch.as_tensor(corpus.targets['event_bin'],device=bank.device)
    total=0.
    for start in range(0,batch,chunk):
        ix=ids[start:start+chunk]; z=model(bank.batch(ix)); t={k:v[ix] for k,v in target.items()}
        loss=physical_loss(model,z,t,arm)
        if noisy:
            zn=model(noisy[number % len(noisy)].batch(ix))
            loss=loss+.5*physical_loss(model,zn,t,arm)+.1*(zn-z.detach()).square().mean()/reference_variance
        (loss*len(ix)/batch).backward();total+=float(loss.detach())*len(ix)/batch
        ix=events[start:start+chunk]; z=model(bank.batch(ix))
        if not arm['event_encoder']:z=z.detach()
        event=hazard_loss(model.logits(z,conditions[ix]),event_bin[ix]).mean()
        (event*len(ix)/batch).backward();total+=float(event.detach())*len(ix)/batch
    rank=0.
    if arm['ap'] and (number+1)%tc['ranking_every']==0:
        primary=config['primary_horizon_ps']
        rank=ranking_backward(model,bank,ri,conditions,event_bin[ri]<=horizon_index(primary),
            torch.as_tensor(source_weights(sources[ri]),device=bank.device,dtype=torch.float32),
            chunk,tc['ap_temperature'],tc['ap_weight'],horizon_ps=primary)
    return dict(loss=total,ranking_loss=rank)


@torch.no_grad()
def validate(model,bank,corpus,target,conditions,risk,sources,arm,chunk,*,horizon_ps):
    ids=corpus.split['tune']; z=encode(model,bank,ids,chunk)
    w=torch.as_tensor(source_weights(sources[ids]),device=z.device,dtype=z.dtype)
    present=float(w@block_error(model.heads[arm['input']](z),target[arm['input']][ids]))
    current=float(w@(model.heads['current'](z)-target['current'][ids]).square().mean(1))
    future=float(w@(model.heads['future'](z)-target['future'][ids]).square().mean(1))
    ids=risk['tune']; z=encode(model,bank,ids,chunk); logits=model.logits(z,conditions[ids])
    bins=corpus.targets['event_bin'][ids]
    k=horizon_index(horizon_ps)
    metrics=weighted_scores(bins<=k,cumulative_risk(logits)[:,k].cpu().numpy(),sources[ids])
    if metrics['average_precision'] is None:
        raise ValueError(f'No tuning onset events at {horizon_ps:g} ps; AP checkpoint selection is undefined')
    return dict(primary_horizon_ps=horizon_ps,present_mse=present,current_mse=current,future_mse=future,**metrics)


def train(study,name,device='cuda',deadline=None):
    c=study.config;arm=study.arm(name);tc=c['training'];folder=study.technical/'fits'/name
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        if json.loads((folder/'complete.json').read_text())['identity']!=study.identity:raise ValueError('Completed fit identity changed')
        return True
    model,bank,noisy,corpus,target,scalers,conditions,risk,sources,ec=setup(study,name,device)
    fit=corpus.split['fit'];rng=np.random.default_rng(c['seed']+1)
    head_parameters=list(model.heads.parameters())+list(model.hazard.parameters())
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=tc['encoder_lr']),
        dict(params=head_parameters,lr=tc['head_lr'])],weight_decay=tc['weight_decay'])
    number=0;best=-1.
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        if saved['identity']!=study.identity:raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        number,best,initial,reference_variance=saved['step'],saved['best'],saved['initial'],saved['reference_variance']
    else:
        with torch.no_grad():
            p=encode(model,bank,fit,tc['microbatch'],pooled=True)
            model.encoder.pooled_mean.copy_(p.mean(0));model.encoder.pooled_scale.copy_(p.std(0,correction=0).clamp_min(1e-5))
            z=encode(model,bank,fit,tc['microbatch'])
            reference_variance=float(z.var(0,correction=0).mean())
            calibrate_heads(model,z,{k:v[fit] for k,v in target.items()},1.,10.)
            ix=risk['fit'];w=source_weights(sources[ix]);bins=corpus.targets['event_bin'][ix]
            prior=np.array([w[bins==k].sum()/w[bins>=k].sum() for k in range(5)]).clip(1e-4,1-1e-4)
            model.hazard[-1].weight.zero_();model.hazard[-1].bias.copy_(torch.as_tensor(np.log(prior/(1-prior))))
        initial=validate(model,bank,corpus,target,conditions,risk,sources,arm,tc['microbatch'],horizon_ps=c['primary_horizon_ps'])
    if reference_variance<=0:raise ValueError('Collapsed initial export')
    def state():
        return dict(identity=study.identity,arm=name,step=number,best=best,primary_horizon_ps=c['primary_horizon_ps'],model=model.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state() if str(device).startswith('cuda') else None,
            initial=initial,reference_variance=reference_variance,encoder_config=ec,target_scalers=scalers)
    if number==0:
        best=initial['average_precision'];save_checkpoint(folder/'initial.pt',state());save_checkpoint(folder/'best.pt',state())
    write_json(folder/'environment.json',dict(torch=torch.__version__,backend=c['encoder']['backend'],
        gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu',
        parameters=sum(p.numel() for p in model.parameters()), precision='float32, TF32 disabled',
        primary_horizon_ps=c['primary_horizon_ps'],
        fit_events=int((corpus.targets['event_bin'][risk['fit']]<=horizon_index(c['primary_horizon_ps'])).sum())))
    started=time.monotonic()
    try:
        while number<tc['updates']:
            remaining(deadline,reserve=900)  # reserve export/evaluation time
            model.train();optimizer.zero_grad(set_to_none=True)
            factor=min(1.,(number+1)/tc['warmup'])*(.05+.95*.5*(1+math.cos(math.pi*number/tc['updates'])))
            for group,lr in zip(optimizer.param_groups,[tc['encoder_lr'],tc['head_lr']],strict=True):group['lr']=lr*factor
            record=step(model,bank,noisy,corpus,target,conditions,risk,sources,arm,c,rng,number,reference_variance)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if not math.isfinite(record['loss']):raise FloatingPointError(f'Nonfinite objective: {name} step {number}')
            optimizer.step();model.bound_heads();number+=1
            if number%tc['log_every']==0:
                record.update(step=number,seconds=time.monotonic()-started,gradient_norm=float(norm))
                with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(name,json.dumps(record),flush=True)
            if number%tc['evaluate_every']==0 or number==tc['updates']:
                val=validate(model,bank,corpus,target,conditions,risk,sources,arm,tc['microbatch'],horizon_ps=c['primary_horizon_ps'])
                retained=all(val[k]<=initial[k]*c['retention_ratio'] for k in ('present_mse','current_mse','future_mse'))
                val.update(step=number,retained=retained)
                with (folder/'validation.jsonl').open('a') as f:f.write(json.dumps(val)+'\n')
                if retained and val['average_precision']>best:
                    best=val['average_precision'];save_checkpoint(folder/'best.pt',state())
            if number%tc['save_every']==0:save_checkpoint(folder/'last.pt',state())
    except TimeoutError:
        save_checkpoint(folder/'last.pt',state())
        write_json(folder/'status.json',dict(state='checkpointed',step=number,identity=study.identity))
        return False
    save_checkpoint(folder/'last.pt',state())
    write_json(folder/'complete.json',dict(state='complete',step=number,identity=study.identity,
        checkpoint_sha256=sha(folder/'last.pt'),selected_sha256=sha(folder/'best.pt')))
    return True
