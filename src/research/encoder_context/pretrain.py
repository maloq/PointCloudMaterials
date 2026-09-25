"""Label-free structural initialization; fixed twelve-epoch exports, no onset selection."""
import json
import math
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from src.data.fixed_cohort.dataset import read_release
from src.research.structural_state.common import sha,write_json,save_checkpoint
from src.research.supervised_onset.model import CapacityEncoder
from src.research.supervised_onset.tracking import tracked_run
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.mace_epi.objective import Objective
from .epochs import batches
from .geometry import graph,physical_targets


def observations(config,method,device):
    root,plan=read_release(config['fixed_dataset']['root'])
    if plan['identity']!=config['fixed_dataset']['identity']:raise ValueError('Fixed release changed')
    paired=method!='physical'
    manifest=json.loads((root/('benchmark' if paired else 'structural')/'manifest.json').read_text())
    result={}
    for role in ('train','selection'):
        views={'hot':[],'cold':[]} if paired else {'hot':[]}
        sources=[]
        for source in manifest['sources']:
            if source['role']!=role:continue
            folder=root/('benchmark' if paired else 'structural')/'sources'/str(source['id'])
            for domain,values in views.items():
                name=domain if paired else 'positions'
                path=folder/f'{name}.npy'
                if sha(path)!=source['files'][path.name]:raise ValueError(f'Changed pretraining input: {path}')
                x=np.load(path).reshape(-1,80,3)
                values.append(x)
            sources.extend([source['id']]*len(x))
        result[role]={name:torch.as_tensor(np.concatenate(values),device=device) for name,values in views.items()}
        result[role]['source']=np.asarray(sources,dtype=np.int32)
    return result


@torch.no_grad()
def encode(encoder,x,chunk):
    return torch.cat([encoder(graph(x[s:s+chunk],encoder)) for s in range(0,len(x),chunk)])


def run(study,method,device,deadline):
    c=study.config;settings=c['pretraining'];root=study.root/'pretraining'/method;technical=root/'technical'
    technical.mkdir(parents=True,exist_ok=True)
    done=technical/'complete.json'
    if done.exists():
        record=json.loads(done.read_text())
        if record['identity']!=study.identity or sha(technical/'epoch-012.pt')!=record['sha256']:
            raise ValueError('Completed pretraining changed')
        return record
    data=observations(c,method,device);torch.manual_seed(c['seed'])
    encoder=CapacityEncoder(**c['encoder'],d0=2.8,n_ref=80.).to(device)
    decoder=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,32)).to(device)
    objective=None if method=='physical' else Objective('epi-variance' if method=='epi_variance' else 'vicreg',.1).to(device)
    chunk=c['microbatch'];train=data['train'];selection=data['selection'];n=len(train['hot'])
    initial=np.random.default_rng(c['seed']+11).choice(n,min(8192,n),replace=False)
    last=technical/'last.pt';start=0
    with torch.no_grad():
        pooled=torch.cat([encoder.pooled_graph(graph(train['hot'][ids],encoder))
            for ids in np.array_split(initial,math.ceil(len(initial)/chunk))])
        encoder.pooled_mean.copy_(pooled.mean(0));encoder.pooled_scale.copy_(pooled.std(0,unbiased=False).clamp_min(1e-5))
    if c['runtime']['compile']:compile_spatial_encoder(encoder,graph(train['hot'][:chunk],encoder))
    # Precompute fixed target features from geometry or a random encoder, never labels.
    with torch.no_grad():
        if method=='physical':
            for values in data.values():
                x=values['hot'];values['target']=torch.cat([physical_targets(x[s:s+chunk]) for s in range(0,len(x),chunk)])
            mean=train['target'].mean(0);scale=train['target'].std(0,unbiased=False).clamp_min(1e-5)
            for values in data.values():values['target']=(values['target']-mean)/scale
        elif method=='epi_variance':
            generator=torch.Generator(device=device).manual_seed(c['seed']+37)
            projection=torch.randn(128,64,device=device,generator=generator)/math.sqrt(128)
            for values in data.values():
                values['reservoir']=torch.stack([encode(encoder,values[d],chunk)@projection for d in ('hot','cold')],1)
            from src.training_methods.neighborhood_jepa.regularization.objective import epiplexity
            score=[]
            for ids in np.array_split(initial[:min(4*chunk,len(initial))],4):
                for view,domain in enumerate(('hot','cold')):
                    z=encoder(graph(train[domain][ids],encoder))
                    score.append(epiplexity(z,train['reservoir'][ids,view]))
            objective.epi_initial_scale.copy_(torch.stack(score).mean())
            if not torch.isfinite(objective.epi_initial_scale) or objective.epi_initial_scale<=1e-6:
                raise FloatingPointError('Degenerate random-reservoir scale')
    parameters=list(encoder.parameters())+(list(decoder.parameters()) if method=='physical' else [])
    optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=1e-4)
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=study.identity or saved['method']!=method:raise ValueError('Pretraining resume changed')
        encoder.load_state_dict(saved['encoder']);decoder.load_state_dict(saved['decoder'])
        if objective is not None:objective.load_state_dict(saved['objective'])
        optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    steps=math.ceil(n/c['batch_size']);total=steps*settings['epochs']
    tracked=SimpleNamespace(root=root,technical=technical,identity=study.identity,config=dict(c,
        branch='self_supervised',wandb=dict(c['wandb'],display_name=f'Al64 | {method} | Structural pretraining')))
    def loss(values,ids):
        z=encoder(graph(values['hot'][ids],encoder))
        if method=='physical':
            error=(decoder(z)-values['target'][ids]).square()
            # Give radial, count and angular blocks equal total influence.
            value=(error[:,:24].mean()+error[:,24:26].mean()+error[:,26:].mean())/3
            return value,{'physical_mse':value.detach()}
        cold=encoder(graph(values['cold'][ids],encoder))
        target={'index':ids}
        if method=='epi_variance':target['reservoir']=values['reservoir'][ids]
        return objective(None,torch.stack((z,cold),1).reshape(-1,128),target)
    def save(step,path):
        save_checkpoint(path,dict(identity=study.identity,release_identity=c['fixed_dataset']['identity'],
            method=method,step=step,epoch=step/steps,encoder=encoder.state_dict(),decoder=decoder.state_dict(),
            objective=None if objective is None else objective.state_dict(),optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),config=c,
            encoder_config=dict(c['encoder'],d0=2.8,n_ref=80.),
            normalization_rows=initial.tolist(),selection='fixed epoch 12; no crystallization labels'))
    completed=start
    with tracked_run(tracked,f'pretrain-{method}') as tracking:
        tracking.summary.update({'data/train_windows':n,'data/validation_windows':len(selection['hot']),
            'training/epochs_requested':settings['epochs'],'checkpoint/selection_rule':'fixed epoch 12; label-free',
            'model/encoder_parameters':sum(p.numel() for p in encoder.parameters())})
        for step,ids in batches(np.arange(n),c['batch_size'],settings['epochs'],c['seed'],start):
            if time.time()>deadline-300:
                save(step,last);raise TimeoutError('Pretraining checkpointed before complete epoch budget')
            encoder.train();decoder.train();optimizer.zero_grad(set_to_none=True)
            factor=.05+.95*.5*(1+math.cos(math.pi*step/total))
            for group in optimizer.param_groups:group['lr']=settings['learning_rate']*min((step+1)/128,1)*factor
            value,terms=loss(train,ids)
            if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite {method} loss at {step}')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step()
            completed=step+1
            if completed%32==0:
                record={'optimizer_update':completed,'train/epoch':completed/steps,'train/objective':float(value.detach()),
                    'train/gradient_norm':float(norm),'train/learning_rate':optimizer.param_groups[0]['lr']}
                record.update({f'train/{key}':float(v.detach()) for key,v in terms.items()})
                tracking.log(record)
                with (technical/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if completed%256==0:save(completed,last)
            if completed%steps==0:
                encoder.eval();decoder.eval();scores=[]
                with torch.no_grad():
                    for begin in range(0,len(selection['hot']),c['batch_size']):
                        ix=np.arange(begin,min(begin+c['batch_size'],len(selection['hot'])))
                        value,_=loss(selection,ix);scores.append((len(ix),float(value)))
                validation=sum(count*value for count,value in scores)/sum(count for count,_ in scores)
                if not math.isfinite(validation):raise FloatingPointError('Nonfinite structural validation')
                tracking.log({'optimizer_update':completed,'validation/objective':validation})
                save(completed,last)
                write_json(technical/'state.json',dict(state='training',completed_epochs=completed//steps,steps=completed))
                print(json.dumps(dict(method=method,epoch=completed//steps,validation=validation)),flush=True)
        if completed!=total:raise ValueError('Incomplete pretraining epochs')
        save(completed,technical/'epoch-012.pt')
        result=dict(state='complete',identity=study.identity,epochs=settings['epochs'],steps=completed,
            sha256=sha(technical/'epoch-012.pt'),method=method,train_rows=n)
        tracking.summary.update({'training/completed_epochs':settings['epochs'],'checkpoint/fixed_epoch':12})
        write_json(done,result)
    return result
