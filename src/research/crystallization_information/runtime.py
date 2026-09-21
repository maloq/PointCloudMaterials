"""Matched source-balanced hazard add-backs and frozen-feature physical decoders."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from src.data.structural_pretraining.prepare import digest,save_json
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.metrics import (source_weights,hazard_loss,cumulative_risk,weighted_scores,threshold_at_fpr)
from src.experiment_runner.metric_docs import write_metric_table
from .data import HORIZONS,BLOCKS,GROUPS,variants,permutation_control


def standardize(values,train,sources):
    w=source_weights(sources[train]);x=values[train]
    mean=np.einsum('n,nd->d',w,x).astype(np.float32)
    scale=np.sqrt(np.einsum('n,nd->d',w,(x-mean)**2)).clip(.001).astype(np.float32)
    return ((values-mean)/scale).astype(np.float32),mean,scale


def model_for(dim,out,kind,width):
    if kind=='linear':return nn.Linear(dim,out)
    if kind=='mlp':return nn.Sequential(nn.Linear(dim,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,out))
    if kind=='strong':return nn.Sequential(nn.Linear(dim,2*width),nn.LayerNorm(2*width),nn.SiLU(),nn.Linear(2*width,2*width),nn.SiLU(),nn.Linear(2*width,out))
    raise ValueError(kind)


def tasks(config):
    result=[]
    for name in config['encoders']:
        result.extend(dict(encoder=name,variant=v,readout=k,task='hazard') for v in variants() if v.startswith('z') for k in ('linear','mlp'))
        result.extend(dict(encoder=name,variant=v,readout='strong',task='hazard') for v in ('z','z+all'))
        result.extend(dict(encoder=name,variant='decode',readout=k,task='decoder') for k in ('linear','strong'))
    for v in ('conditions','geometry','current','all'):
        result.extend(dict(encoder='baseline',variant=v,readout=k,task='hazard') for k in ('linear','mlp'))
    result.append(dict(encoder='baseline',variant='decode',readout='strong',task='decoder'))
    # Run the most informative joined comparisons first, then isolate blocks.
    priority={'z':0,'geometry':0,'current':0,'conditions':0,'z+all':1,'all':1,'decode':2}
    return sorted(result,key=lambda t:priority.get(t['variant'],3))


def score_hazard(pop,ids,logits,cal_ids,cal_logits):
    source=pop['source'][ids];event=pop['event'][ids];weight=source_weights(source)
    risk=cumulative_risk(torch.from_numpy(logits)).numpy().astype(np.float64)
    calibration=cumulative_risk(torch.from_numpy(cal_logits)).numpy().astype(np.float64)
    nll=hazard_loss(torch.from_numpy(logits),torch.from_numpy(event)).numpy()
    classification={};timing={};per_source={}
    for sid in np.unique(source):per_source[str(sid)]={'temperature_K':float(pop['temperature'][ids[source==sid]][0]),'event_nll':float(nll[source==sid].mean())}
    for k,h in enumerate(HORIZONS):
        y=event<=k;threshold=threshold_at_fpr(pop['event'][cal_ids]<=k,calibration[:,k],pop['source'][cal_ids],.05)
        row=weighted_scores(y,risk[:,k],source,threshold)
        row['auroc']=float(roc_auc_score(y,risk[:,k],sample_weight=weight)) if y.any() and not y.all() else None
        classification[str(h)]=row
        prob=np.diff(np.c_[np.zeros(len(ids)),risk[:,:k+1]],axis=1)
        midpoint=(np.r_[0,HORIZONS[:k]]+HORIZONS[:k+1])/2
        predicted=(prob@midpoint)/np.maximum(risk[:,k],1e-12)
        hit=y&(risk[:,k]>=threshold);errors=predicted[hit]-pop['delay'][ids[hit]]
        timing[str(h)]=dict(event_windows=int(y.sum()),missed_windows=int((y&~hit).sum()),
            detected_timing_mae_ps=float(np.abs(errors).mean()) if len(errors) else None,
            timed_within_3ps_recall=float((np.abs(errors)<=3).sum()/y.sum()) if y.any() else None)
        for sid in np.unique(source):
            mask=source==sid;s=weighted_scores(y[mask],risk[mask,k],source[mask])
            per_source[str(sid)][str(h)]={key:s[key] for key in ('log_loss','brier')}
    return dict(event_nll=float(weight@nll),classification=classification,timing=timing,per_source=per_source)


def decoder_score(y,pred,source):
    w=source_weights(source);mse=np.einsum('n,nd->d',w,(y-pred)**2)
    center=np.einsum('n,nd->d',w,y);var=np.einsum('n,nd->d',w,(y-center)**2)
    r2=np.divide(mse,var,out=np.full_like(mse,np.nan),where=var>1e-12)
    groups={k:dict(standardized_mse=float(mse[ix].mean()),r2=float(1-mse[ix].sum()/var[ix].sum()) if var[ix].sum()>1e-12 else None)
            for k,ix in BLOCKS.items() if k!='history'}
    return dict(groups=groups,component_mse={str(i):float(v) for i,v in enumerate(mse)},
                component_r2={str(i):float(1-v) if np.isfinite(v) else None for i,v in enumerate(r2)})


def fit(config,task,pop,x,target,root):
    folder=root/'technical/fits'/task['encoder']/task['variant']/task['readout'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return
    ids={r:np.flatnonzero(pop['role']==r) for r in ('train','selection','calibration','test')}
    if any(not len(v) for v in ids.values()):raise ValueError('Empty source split')
    torch.manual_seed(config['seed']);device=config['device'];train=ids['train'];w=source_weights(pop['source'][train]);cdf=np.cumsum(w);cdf[-1]=1
    hazard=task['task']=='hazard';out=len(HORIZONS) if hazard else 148
    model=model_for(x.shape[1],out,task['readout'],config['width']).to(device)
    x=torch.as_tensor(x,device=device);y=torch.as_tensor(target,device=device)
    if hazard:
        final=model if task['readout']=='linear' else model[-1]
        event=pop['event'][train]
        frequency=np.array([(w@(event==k)+1e-6)/(w@(event>=k)+2e-6) for k in range(out)]).clip(1e-5,1-1e-5)
        with torch.no_grad():final.bias.copy_(torch.as_tensor(np.log(frequency/(1-frequency)),dtype=torch.float32,device=device));final.weight.mul_(.01)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    identity=digest(dict(config=config,task=task));step=0;best=float('inf');last=folder/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('Information diagnostic resume identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best']
    def save(path):
        tmp=path.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,identity=identity),tmp);tmp.replace(path)
    @torch.no_grad()
    def predict(indices):return torch.cat([model(x[j]) for j in np.array_split(indices,max(1,math.ceil(len(indices)/4096)))]).cpu().numpy()
    def loss(prediction,actual):
        if hazard:return hazard_loss(prediction,actual)
        error=(prediction-actual).square()
        return torch.stack([error[:,ix].mean(-1) for k,ix in BLOCKS.items() if k!='history'],-1).mean(-1)
    sw=source_weights(pop['source'][ids['selection']]);start=time.time()
    while step<config['updates']:
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],step]));chosen=train[np.searchsorted(cdf,rng.random(config['batch_size']))]
        factor=min(1.,(step+1)/64)*(.05+.95*.5*(1+math.cos(math.pi*step/config['updates'])))
        optimizer.param_groups[0]['lr']=config['lr']*factor;optimizer.zero_grad(set_to_none=True)
        value=loss(model(x[chosen]),y[chosen]).mean()
        if not torch.isfinite(value):raise FloatingPointError(f'Diagnostic loss diverged: {task}, step={step}')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1
        if step%config['evaluate_every']==0 or step==config['updates']:
            score=float(sw@loss(torch.as_tensor(predict(ids['selection']),device=device),y[ids['selection']]).cpu().numpy())
            if score<best:best=score;save(folder/'best.pt')
            save(last);save_json(folder/'status.json',dict(state='running',step=step,best=best))
    model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=False)['model']);test=predict(ids['test'])
    if hazard:
        calibration=predict(ids['calibration']);metrics=score_hazard(pop,ids['test'],test,ids['calibration'],calibration)
        np.savez_compressed(folder/'predictions.npz',test=test,calibration=calibration,test_indices=ids['test'],calibration_indices=ids['calibration'])
    else:
        metrics=decoder_score(target[ids['test']],test,pop['source'][ids['test']]);np.savez_compressed(folder/'predictions.npz',test=test,test_indices=ids['test'])
    metrics.update(task=task,best_selection_loss=best,updates=step,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start)
    save_json(folder/'metrics.json',metrics)
    write_metric_table(metrics,root,family='crystallization_information',name='--'.join(task.values()))
    save_json(folder/'status.json',dict(state='complete',step=step));print(json.dumps(dict(state='complete',task=task,selection=best)),flush=True)


def run(config,lane,lanes):
    torch.set_num_threads(config['threads']);root=resolve_path(config['output']);data=root/'technical/data'
    pop=dict(np.load(data/'population.npz'));observed=np.load(data/'observed.npy');train=np.flatnonzero(pop['role']=='train')
    standardized,mean,scale=standardize(observed,train,pop['source'])
    shuffled=permutation_control(standardized,pop['role'],pop['temperature'],config['seed']+19)
    encoders={'baseline':np.zeros((len(observed),128),np.float32)};normalizers={'observed_mean':mean,'observed_scale':scale}
    for name in config['encoders']:
        encoders[name],m,s=standardize(np.load(data/f'{name}.npy'),train,pop['source']);normalizers[name+'_mean']=m;normalizers[name+'_scale']=s
    np.savez(root/'technical'/f'normalizers-{lane}.npz',**normalizers)
    for task in tasks(config)[lane::lanes]:
        x=np.zeros((len(observed),128+427+7),np.float32);x[:,-7:]=pop['condition'];x[:,:128]=encoders[task['encoder']]
        if task['task']=='hazard':
            names=variants()[task['variant']];columns=np.concatenate([BLOCKS[k] for k in names]) if names else np.array([],int)
            values=shuffled if task['variant']=='z+shuffled' else standardized
            x[:,128+columns]=values[:,columns];target=pop['event']
        else:target=standardized[:,:148]
        save_json(root/'technical'/f'lane-{lane}.json',dict(state='running',task=task))
        fit(config,task,pop,x,target,root)
        del x
    save_json(root/'technical'/f'lane-{lane}.json',dict(state='complete'))
