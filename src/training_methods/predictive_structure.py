"""Task-supervised atomic representations: geometry retention and shooting forecasts."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from src.models.encoders.atomic_graph import ReferenceMACEEncoder,SchNetEncoder,DensityMLPEncoder
from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder

ROOT=Path(__file__).resolve().parents[2]
NEURAL=('MACE','SchNet','DensityMLP','GeoFrame')
CURRENT_SLICES=(slice(0,8),slice(8,24),slice(24,88))
FUTURE_SLICES=(slice(0,48),slice(48,66),slice(66,72))
FAMILIES=('topology','order','mobility')


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class BenchmarkData:
    def __init__(self,cfg,out):
        base=ROOT/cfg['benchmark'];self.meta=dict(np.load(base/'metadata.npz'))
        self.clouds=torch.tensor(np.load(base/'clouds.npy'),device='cuda')
        self.edges=torch.tensor(np.load(base/'edges.npy'),device='cuda')
        self.counts=torch.tensor(np.load(base/'edge_counts.npy'),device='cuda')
        self.material=torch.tensor(self.meta['material'],device='cuda')
        self.temperature=torch.tensor(self.meta['temperature'][:,None]==np.array([400.,450.,500.])[None],device='cuda',dtype=torch.float32)
        self.indices={s:torch.tensor(np.flatnonzero(self.meta['split']==s),device='cuda') for s in ('train','val','test')}
        self.by_material={s:[ids[self.material[ids]==m] for m in range(3)] for s,ids in self.indices.items()}
        saved=dict(np.load(base/'evaluation/split_and_targets.npz'))
        n=len(self.clouds);self.future_mask=torch.zeros(n,device='cuda',dtype=torch.bool)
        self.future_mask[saved['rows']]=True
        self.future=torch.zeros(n,72,device='cuda')
        target=np.concatenate([saved[f].reshape(len(saved['rows']),-1) for f in FAMILIES],axis=1)
        self.future[saved['rows']]=torch.tensor(target,device='cuda',dtype=torch.float32)
        self.future_indices={s:ids[self.future_mask[ids]] for s,ids in self.indices.items()}
        raw=np.concatenate((np.load(base/'order.npy'),np.load(base/'embeddings/TDA_16.npy'),np.load(base/'embeddings/SOAP.npy')[:,:64]),axis=1).astype(np.float64)
        means=[];stds=[];current=np.empty_like(raw)
        for m in range(3):
            train=(self.meta['split']=='train')&(self.meta['material']==m)
            mean=raw[train].mean(0);std=raw[train].std(0)
            std=np.maximum(std,1e-3*np.median(std))
            current[self.meta['material']==m]=(raw[self.meta['material']==m]-mean)/std
            means.append(mean);stds.append(std)
        self.current=torch.tensor(current,dtype=torch.float32,device='cuda')
        np.savez(out/'current_target_scaling.npz',mean=np.stack(means),std=np.stack(stds))
        self.fixed={}
        for name in cfg['models']:
            if name in NEURAL:continue
            values=np.load(base/'order.npy') if name=='CoarseBOO' else np.load(base/'embeddings'/f'{name}.npy')
            self.fixed[name]=torch.tensor(values,device='cuda')
        write_json(out/'data_protocol.json',dict(train_by_material={m:len(ids) for m,ids in zip(('Al','Mg','Ta'),self.by_material['train'])},
            future_counts={s:len(ids) for s,ids in self.future_indices.items()},
            supervision='Current BOO(8), TDA(16), SOAP(64), plus repeated-future mean topology(48), order(18), mobility(6). No PTM labels.',
            source_splits={s:np.unique(saved['source'][saved['split']==s]).tolist() for s in ('train','val','test')},
            test_policy='Previously examined test sources retained; no test loss used for fitting, early stopping or learning-rate selection'))


class Predictor(nn.Module):
    def __init__(self,name,cfg,data):
        super().__init__();self.name=name
        base=ROOT/cfg['benchmark']
        if name=='MACE':self.encoder=ReferenceMACEEncoder(accelerated=True)
        elif name=='SchNet':self.encoder=SchNetEncoder()
        elif name=='DensityMLP':
            self.encoder=DensityMLPEncoder();scaling=torch.load(base/'density_scaling.pt',weights_only=True)
            self.encoder.mean.copy_(scaling['mean']);self.encoder.std.copy_(scaling['std'])
        elif name=='GeoFrame':
            saved=torch.load(ROOT/cfg['geoframe_config_checkpoint'],map_location='cpu',weights_only=False)
            settings=OmegaConf.create(saved['hyper_parameters'])
            kwargs=OmegaConf.to_container(settings.encoder.kwargs,resolve=True)
            self.encoder=GeoFrameTransformerV2Encoder(**kwargs)
            self.register_buffer('radii',torch.tensor(cfg['geoframe_radii_A']))
        else:self.encoder=None
        dim=128 if name in NEURAL else data.fixed[name].shape[1]
        self.normalization=nn.BatchNorm1d(dim,eps=1e-8,momentum=.05)
        width=cfg['head_width']
        # Same decoder architecture for learned and fixed representations.
        self.current_head=nn.Sequential(nn.Linear(dim+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,88))
        self.future_head=nn.Sequential(nn.Linear(dim+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,72))

    def features(self,ids,data,clouds=None):
        if self.name not in NEURAL:raw=data.fixed[self.name][ids]
        else:
            x=data.clouds[ids].clone() if clouds is None else clouds
            if self.name=='GeoFrame':
                order=x.square().sum(-1).argsort(dim=1)
                x=x.gather(1,order[:,:,None].expand(-1,-1,3))
                raw=self.encoder.forward_features(x[:,:80]/self.radii[data.material[ids],None,None])
            else:raw=self.encoder(x,data.material[ids],data.edges[ids],data.counts[ids])
        return self.normalization(raw)

    def forward(self,ids,data,clouds=None):
        z=self.features(ids,data,clouds)
        types=torch.nn.functional.one_hot(data.material[ids],3).to(z.dtype)
        return z,self.current_head(torch.cat((z,types),1)),self.future_head(torch.cat((z,data.temperature[ids]),1))


def current_loss(prediction,target,material):
    # Material balancing prevents the Al-heavy dataset from deciding this loss.
    return torch.stack([torch.stack([(prediction[material==m,s]-target[material==m,s]).square().mean() for s in CURRENT_SLICES]).mean() for m in range(3)]).mean()


def future_errors(prediction,target):
    return torch.stack([(prediction[:,s]-target[:,s]).square().mean(1) for s in FUTURE_SLICES],dim=1)


@torch.no_grad()
def validation(model,data,batch_size):
    model.eval();ids=data.indices['val'];current=[];future=[]
    for part in ids.split(batch_size):
        _,c,f=model(part,data);current.append(c);future.append(f)
    c=torch.cat(current);f=torch.cat(future)
    a=current_loss(c,data.current[ids],data.material[ids])
    keep=data.future_mask[ids];b=future_errors(f[keep],data.future[ids[keep]]).mean(0)
    return float(a),b.cpu().numpy()


def check_model(model,data):
    ids=data.future_indices['train'][:8]
    model.eval()
    with torch.no_grad():
        z,c,f=model(ids,data)
        assert torch.isfinite(z).all() and torch.isfinite(c).all() and torch.isfinite(f).all()
        if model.name in ('MACE','SchNet','DensityMLP','GeoFrame'):
            axis=torch.tensor([.3,.4,.5],device='cuda');angle=axis.norm();axis=axis/angle
            k=torch.zeros(3,3,device='cuda');k[0,1]=-axis[2];k[0,2]=axis[1];k[1,0]=axis[2];k[1,2]=-axis[0];k[2,0]=-axis[1];k[2,1]=axis[0]
            rotation=torch.eye(3,device='cuda')+torch.sin(angle)*k+(1-torch.cos(angle))*(k@k)
            zz,_,_=model(ids,data,data.clouds[ids]@rotation)
            torch.testing.assert_close(z,zz,rtol=2e-4,atol=1e-5)


def train_trial(name,seed,rate,cfg,data,directory,*,continue_to_plateau=False):
    if not continue_to_plateau:directory.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);model=Predictor(name,cfg,data).cuda();check_model(model,data)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=rate,weight_decay=cfg['weight_decay'],fused=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=cfg['lr_patience'],threshold=1e-4,min_lr=cfg['minimum_lr'])
    rng=torch.Generator(device='cuda').manual_seed(seed+1000)
    noise_rng=torch.Generator(device='cuda').manual_seed(seed+2000)
    future_ids=data.future_indices['train'];bs=cfg['batch_size'];fb=cfg['future_batch_size']
    steps=math.ceil(len(future_ids)/fb)
    sizes=[(bs-fb)//3]*3;sizes[-1]+=bs-fb-sum(sizes)
    best,best_epoch=float('inf'),-1;anchor=float('inf');last_improvement=-1;started=time.monotonic();records=[]
    start_epoch=0;previous_seconds=0.;limit=cfg['max_epochs']
    if continue_to_plateau:
        previous=json.loads((directory/'status.json').read_text());previous_seconds=previous['seconds']
        saved=torch.load(directory/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        records=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
        start_epoch=records[-1]['epoch'];limit=max(limit,start_epoch+300)
        best=previous['selection_score'];best_epoch=previous['best_epoch']-1
        for r in records[cfg['geometry_warmup_epochs']+4:]:
            if r['selection_score']<anchor-1e-4:anchor=r['selection_score'];last_improvement=r['epoch']-1
        # Checkpoints store optimizer and scheduler state. Continuations use an
        # explicitly seeded independent sample stream, recorded below.
        rng.manual_seed(seed+1000+1000003*start_epoch);noise_rng.manual_seed(seed+2000+1000003*start_epoch)
        write_json(directory/'continuation.json',dict(from_epoch=start_epoch,maximum_epoch=limit,
            reason='Accumulated validation improvements must reset early-stopping patience',
            sample_seed=seed+1000+1000003*start_epoch,noise_seed=seed+2000+1000003*start_epoch))
        del saved
    for epoch in range(start_epoch,limit):
        tick=time.monotonic();model.train()
        current_total=future_total=0.
        ramp=0. if epoch<cfg['geometry_warmup_epochs'] else min(1.,(epoch-cfg['geometry_warmup_epochs']+1)/5)
        for _ in range(steps):
            selected=[future_ids[torch.randint(len(future_ids),(fb,),device='cuda',generator=rng)]]
            for pool,n in zip(data.by_material['train'],sizes):selected.append(pool[torch.randint(len(pool),(n,),device='cuda',generator=rng)])
            ids=torch.cat(selected);x=None
            if name in NEURAL:
                noise=(torch.randn((len(ids),193,3),device='cuda',generator=noise_rng)*cfg['jitter_std_A']).clamp(-cfg['jitter_clip_A'],cfg['jitter_clip_A']);noise[:,0]=0
                x=data.clouds[ids]+noise
            z,c,f=model(ids,data,x)
            structure=current_loss(c,data.current[ids],data.material[ids])
            dynamic=future_errors(f[:fb],data.future[ids[:fb]]).mean()
            loss=(1. if ramp==0 else cfg['current_weight'])*structure+ramp*dynamic
            if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{seed}/lr={rate}/epoch={epoch}: nonfinite task loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();current_total+=float(structure.detach());future_total+=float(dynamic.detach())
        cv,fv=validation(model,data,bs)
        score=float(fv.mean()+cfg['validation_current_weight']*cv)
        record=dict(epoch=epoch+1,train_current=current_total/steps,train_future=future_total/steps,
            validation_current=cv,validation_future=fv.tolist(),selection_score=score,learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=float(gradient),seconds=time.monotonic()-tick)
        records.append(record)
        with (directory/'epochs.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
        if ramp==1.:
            if score<anchor-1e-4:anchor=score;last_improvement=epoch
            if score<best:
                best,best_epoch=score,epoch
                torch.save(dict(state_dict=model.state_dict(),name=name,seed=seed,initial_learning_rate=rate,config=cfg,**record),directory/'best.pt')
            scheduler.step(score)
        payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),name=name,seed=seed,initial_learning_rate=rate,config=cfg,**record)
        if epoch%10==0:torch.save(payload,directory/'last.pt')
        write_json(directory/'status.json',dict(state='running',best_epoch=best_epoch+1,**record))
        if epoch%10==0:print(name,seed,rate,record,flush=True)
        if epoch+1>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['early_stopping_patience']:break
    torch.save(payload,directory/'last.pt')
    converged=epoch-last_improvement>=cfg['early_stopping_patience']
    summary=dict(name=name,seed=seed,learning_rate=rate,best_epoch=best_epoch+1,epochs=epoch+1,selection_score=best,
        converged=converged,seconds=previous_seconds+time.monotonic()-started,parameters=sum(p.numel() for p in parameters),
        checkpoint=str(directory/'best.pt'),last_checkpoint=str(directory/'last.pt'))
    write_json(directory/'status.json',dict(state='complete',**summary));print('COMPLETE',summary,flush=True)
    del model,optimizer,scheduler,parameters,payload;torch.cuda.empty_cache()
    return summary


def complete_plateau_audit(cfg,out,data,selected):
    """Finish accumulated-improvement plateaus before permitting test evaluation."""
    trials=json.loads((out/'trials.json').read_text());audit=[]
    for i,t in enumerate(trials):
        directory=Path(t['checkpoint']).parent
        history=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
        anchor=float('inf');last=0
        for r in history[cfg['geometry_warmup_epochs']+4:]:
            if r['selection_score']<anchor-1e-4:anchor=r['selection_score'];last=r['epoch']
        age=history[-1]['epoch']-last
        audit.append(dict(model=t['name'],seed=t['seed'],learning_rate=t['learning_rate'],initial_plateau_age=age,continued=age<cfg['early_stopping_patience']))
        if age<cfg['early_stopping_patience']:
            trials[i]=train_trial(t['name'],t['seed'],t['learning_rate'],cfg,data,directory,continue_to_plateau=True)
            if not trials[i]['converged']:raise RuntimeError(f'Continuation did not reach a validation plateau: {directory}')
            write_json(out/'trials.json',trials)
    selected.clear()
    for name in cfg['models']:
        best=min((t for t in trials if t['name']==name and t['seed']==123),key=lambda t:t['selection_score'])
        selected.append(best)
        for seed in cfg['seeds'][1:]:
            matches=[t for t in trials if t['name']==name and t['seed']==seed and t['learning_rate']==best['learning_rate']]
            if matches:selected.append(matches[0])
            else:
                directory=out/'trials'/f'{name}_seed{seed}_lr{best["learning_rate"]:g}'
                t=train_trial(name,seed,best['learning_rate'],cfg,data,directory)
                trials.append(t);selected.append(t);write_json(out/'trials.json',trials)
    write_json(out/'selected_runs.json',selected);write_json(out/'convergence_audit.json',audit)


def run(cfg,out):
    data=BenchmarkData(cfg,out);trials=[];selected=[]
    # All fitting and selection finish before any test predictions are evaluated.
    for name in cfg['models']:
        candidates=[]
        for rate in cfg['learning_rates']:
            directory=out/'trials'/f'{name}_seed123_lr{rate:g}'
            result=train_trial(name,123,rate,cfg,data,directory)
            candidates.append(result);trials.append(result);write_json(out/'trials.json',trials)
        best=min(candidates,key=lambda item:item['selection_score'])
        selected.append(best);write_json(out/'selected_runs.json',selected)
        for seed in cfg['seeds'][1:]:
            directory=out/'trials'/f'{name}_seed{seed}_lr{best["learning_rate"]:g}'
            result=train_trial(name,seed,best['learning_rate'],cfg,data,directory)
            trials.append(result);selected.append(result)
            write_json(out/'trials.json',trials);write_json(out/'selected_runs.json',selected)
    from src.analysis.predictive_structure import evaluate
    evaluate(cfg,out,data,selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output'];out.mkdir(exist_ok=True)
    if (out/'status.json').exists():raise FileExistsError(f'Run already initialized: {out}. Use a new output directory.')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_json(out/'config.json',cfg)
    shutil.copyfile(__file__,out/'training_source_at_launch.py')
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat());write_json(out/'status.json',status)
    try:
        run(cfg,out);status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(out/'status.json',status)
