"""Frozen structural/causal probes and physical/TDA metrics on whole held-out sources."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_table
from src.training_methods.structural_pretraining.objective import PHYSICAL_BLOCKS,TDA_BLOCKS
from .data import CausalRelease
from .runtime import configure,StructuralModel,CausalModel,learning_rate

BLOCKS={**PHYSICAL_BLOCKS,**{k:(a+85,b+85) for k,(a,b) in TDA_BLOCKS.items()}}
LAGS=[0.,.75,3.,9.]


def errors(prediction,target):
    square=(prediction-target)**2
    return np.stack([square[...,a:b].mean(-1) for a,b in BLOCKS.values()],-1)


def source_mean(values,source):
    return np.stack([values[source==sid].mean(0) for sid in np.unique(source)]).mean(0)


def task_score(prediction,target,source):
    value=source_mean(errors(prediction,target),source)
    return float(value[...,:4].mean()+.25*value[...,4:].mean())


def fit_ridge(features,target,train,selection,sources,alphas):
    mean=features[train].mean(0,dtype=np.float64);scale=np.maximum(features[train].std(0,dtype=np.float64),1e-6)
    x=(features-mean)/scale;y=target.reshape(len(target),-1)
    best=float('inf');selected=None
    for alpha in alphas:
        model=Ridge(alpha=alpha,solver='svd').fit(x[train],y[train]);prediction=model.predict(x[selection]).reshape(-1,4,229)
        score=task_score(prediction,target[selection],sources[selection])
        if score<best:best=score;selected=copy.deepcopy(model)
    prediction=selected.predict(x).reshape(target.shape).astype(np.float32)
    return prediction,dict(mean=mean,scale=scale,coefficient=selected.coef_,intercept=selected.intercept_,alpha=selected.alpha,selection_score=best)


def fit_nonlinear(features,target,ridge,train,selection,config,deadline):
    torch.manual_seed(config['seed']);mean=features[train].mean(0,dtype=np.float64);scale=np.maximum(features[train].std(0,dtype=np.float64),1e-6)
    x=torch.tensor((features-mean)/scale,device='cuda',dtype=torch.float32);y=torch.tensor(target,device='cuda');base=torch.tensor(ridge,device='cuda')
    net=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,4*229)).cuda()
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.02,weight_decay=1e-4)
    ti=torch.tensor(train,device='cuda');si=torch.tensor(selection,device='cuda');generator=torch.Generator(device='cuda').manual_seed(config['seed'])
    def loss(pred,target):
        v=torch.stack([(pred[...,a:b]-target[...,a:b]).square().mean(-1) for a,b in BLOCKS.values()],-1)
        return v[...,:4].mean()+.25*v[...,4:].mean()
    best=float(loss(base[si],y[si]));best_state=copy.deepcopy(net.state_dict());best_step=0;trace=[]
    for step in range(1,config['probe_updates']+1):
        if time.time()>deadline-60:raise TimeoutError('Analysis deadline before nonlinear probe completed; cached extraction is preserved')
        ids=ti[torch.randint(len(ti),(config['probe_batch_size'],),generator=generator,device='cuda')]
        for group in optimizer.param_groups:group['lr']=learning_rate(step,config['probe_updates'],config['probe_peak_lr'])
        optimizer.zero_grad(set_to_none=True);value=loss(base[ids]+net(x[ids]).reshape(-1,4,229),y[ids]);value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if step%50==0 or step==config['probe_updates']:
            with torch.no_grad():score=float(loss(base[si]+net(x[si]).reshape(-1,4,229),y[si]))
            trace.append(dict(step=step,selection_score=score))
            if score<best:best=score;best_step=step;best_state=copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    with torch.no_grad():
        prediction=np.concatenate([(base[start:start+1024]+net(x[start:start+1024]).reshape(-1,4,229)).cpu().numpy() for start in range(0,len(x),1024)])
    return prediction,dict(model={k:v.cpu() for k,v in best_state.items()},mean=mean,scale=scale,best_step=best_step,trace=trace)


@torch.no_grad()
def extract(checkpoint,phase,release,config,root,deadline):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['identity']['config'];model=(CausalModel if phase=='causal' else StructuralModel)(cfg['architecture'],history=cfg['history_frames']>1).cuda()
    model.load_state_dict(saved['model']);model.eval()
    ids_by_source={}
    for i,(_,_,r) in enumerate(release.rows):ids_by_source.setdefault(r['source'],[]).append(i)
    states=np.empty((len(release.rows),128),np.float32);direct=np.full((len(release.rows),4,229),np.nan,np.float32)
    objective=saved['objective'];mean=np.r_[objective['physical_mean'].numpy(),objective['tda_mean'].numpy()];std=np.r_[objective['physical_std'].numpy(),objective['tda_std'].numpy()]
    checksum=file_hash(checkpoint);folder=root/'states'/phase;folder.mkdir(parents=True,exist_ok=True)
    for sid,indices in sorted(ids_by_source.items()):
        path=folder/f'source-{sid}.npz'
        if path.exists():
            with np.load(path) as a:
                if str(a['checkpoint_sha256'])!=checksum:raise ValueError('Changed checkpoint in extraction cache')
                np.testing.assert_array_equal(a['indices'],indices);states[indices]=a['state'];direct[indices]=a['direct']
            continue
        if time.time()>deadline-120:raise TimeoutError('Extraction deadline; completed source shards retained')
        zs=[];pred=[]
        for start in range(0,len(indices),config['microbatch_size']):
            ii=indices[start:start+config['microbatch_size']]
            samples=[release.observation(i,'anchor',cfg['history_frames']==3,cfg['architecture']=='mace') for i in ii]
            batch=move(collate(samples,cfg['architecture']),'cuda:0');z=model.encoder(batch);heads=model.heads(z)
            out=np.full((len(ii),4,229),np.nan,np.float32)
            out[:,0]=np.concatenate((heads['physical'].cpu().numpy(),heads['tda'].cpu().numpy()),-1)*std+mean
            if phase=='causal':out[:,1:]=model.future(z).reshape(-1,3,229).cpu().numpy()*objective['future_std'].numpy()+objective['future_mean'].numpy()
            zs.append(z.cpu().numpy());pred.append(out)
        states[indices]=np.concatenate(zs);direct[indices]=np.concatenate(pred)
        temp=path.with_suffix('.building.npz');np.savez(temp,indices=indices,state=states[indices],direct=direct[indices],checkpoint_sha256=checksum);temp.replace(path)
    del model;torch.cuda.empty_cache();return states,direct


def bootstrap_gain(base,candidate,sources,seed):
    s=np.unique(sources);b=np.stack([base[sources==sid].mean(0) for sid in s]);c=np.stack([candidate[sources==sid].mean(0) for sid in s])
    rng=np.random.default_rng(seed);draw=rng.integers(len(s),size=(4000,len(s)))
    delta=(b-c).mean(0);samples=(b[draw]-c[draw]).mean(1);lo,hi=np.quantile(samples,[.025,.975],axis=0)
    return dict(mean_error_reduction=delta.tolist(),ci95_low=lo.tolist(),ci95_high=hi.tolist(),sources=len(s),resamples=4000)


def summarize(predictions,target,data,seed):
    result={};all_errors={name:errors(p,target) for name,p in predictions.items()}
    test=data['split']=='test';populations={'all':test,'noncrystalline':test&~np.isin(data['current_ptm'],[1,2,3])}
    populations.update({f'temperature_{t}K':test&(data['temperature_K']==t) for t in np.unique(data['temperature_K'])})
    for population,mask in populations.items():
        if not mask.any():continue
        metrics={}
        for name,value in all_errors.items():
            e=source_mean(value[mask],data['source'][mask]);metrics[name]={str(lag):dict(physical=float(e[k,:4].mean()),tda=float(e[k,4:].mean()),blocks=dict(zip(BLOCKS,map(float,e[k])))) for k,lag in enumerate(LAGS)}
        gains={name:bootstrap_gain(all_errors['persistence'][mask,1:],value[mask,1:],data['source'][mask],seed) for name,value in all_errors.items() if name in ['ridge','nonlinear']}
        result[population]=dict(rows=int(mask.sum()),sources=len(np.unique(data['source'][mask])),metrics=metrics,gain_over_persistence=gains)
    return result


def run(config,deadline):
    configure();root=resolve_path(config['output']);technical=root/'technical';technical.mkdir(parents=True,exist_ok=True)
    release=CausalRelease(resolve_path(config['release']));data=release.targets()
    if len(np.unique(data['row_id']))!=len(data['row_id']):raise ValueError('Duplicate evaluation row identities')
    train=np.flatnonzero(data['split']=='train');selection=np.flatnonzero(data['split']=='selection')
    normalizer=release.manifest['forecast_normalization'];mean=np.array(normalizer['mean'],np.float32);std=np.array(normalizer['std'],np.float32)
    target=(data['target']-mean)/std
    identity=dict(data=release.manifest['identity'],config=config,checkpoints={k:file_hash(resolve_path(v)) for k,v in config['checkpoints'].items()},
        implementation={p:file_hash(p) for p in [__file__,'src/training_methods/shared_pretraining/runtime.py','src/training_methods/shared_pretraining/data.py']})
    if (technical/'identity.json').exists() and json.loads((technical/'identity.json').read_text())!=identity:raise ValueError('Analysis identity changed')
    save_json(technical/'identity.json',identity)
    if (technical/'status.json').exists() and json.loads((technical/'status.json').read_text())['state']=='complete':return True
    condition,condition_fit=fit_ridge(data['temperature_K'][:,None].astype(np.float32),target,train,selection,data['source'],config['ridge_alphas'])
    np.savez(technical/'condition_ridge.npz',**condition_fit)
    results={}
    for phase,filename in config['checkpoints'].items():
        checkpoint=resolve_path(filename)
        if json.loads((checkpoint.parent/'status.json').read_text())['state']!='complete':raise ValueError('Analysis requires completed parent fits')
        save_json(technical/'status.json',dict(state='running',phase=phase))
        features,direct=extract(checkpoint,phase,release,config,technical,deadline)
        ridge,fit=fit_ridge(features,target,train,selection,data['source'],config['ridge_alphas']);np.savez(technical/f'{phase}-ridge.npz',**fit)
        nonlinear,artifact=fit_nonlinear(features,target,ridge,train,selection,config,deadline);torch.save(artifact,technical/f'{phase}-nonlinear.pt')
        predictions=dict(ridge=ridge,nonlinear=nonlinear,conditions=condition,training_mean=np.broadcast_to(target[train].mean(0),target.shape),
            persistence=np.broadcast_to(target[:,:1],target.shape))
        results[phase]=summarize(predictions,target,data,config['seed'])
        direct_scaled=(direct-mean)/std
        # A structural checkpoint has only current trained heads; future quality
        # is measured by the matched frozen probes, never invented direct heads.
        direct_lags=range(4) if phase=='causal' else range(1)
        test=data['split']=='test';de=source_mean(errors(direct_scaled[test],target[test]),data['source'][test])
        results[phase]['trained_heads']={str(LAGS[k]):dict(physical=float(de[k,:4].mean()),tda=float(de[k,4:].mean())) for k in direct_lags}
        np.savez_compressed(technical/f'{phase}-predictions.npz',**predictions,direct=direct_scaled,target=target,state=features,
            **{k:v for k,v in data.items() if k!='target'})
    save_json(technical/'metrics.json',results);write_metric_table(results,root,family='shared_pretraining',name='frozen_backbone')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'plots';folder.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(10,4))
    for phase in results:
        for j,metric in enumerate(['physical','tda']):
            axes[j].plot(LAGS[1:],[results[phase]['all']['metrics']['nonlinear'][str(l)][metric] for l in LAGS[1:]],marker='o',label=phase)
            axes[j].set(xlabel='Forecast horizon (ps)',ylabel=f'Standardized {metric} MSE');axes[j].legend()
    fig.tight_layout();fig.savefig(folder/'frozen_prediction.png',dpi=180);plt.close(fig)
    import wandb
    settings=config['wandb'];run=wandb.init(entity=settings['entity'],project=settings['project'],id=settings['id'],name=settings['name'],group=settings['group'],mode='online',resume='allow',dir=str(technical),config=config,save_code=False)
    if run.offline:raise RuntimeError('Analysis W&B logging must be online')
    for phase in results:
        for name in ['ridge','nonlinear']:
            for lag,values in results[phase]['all']['metrics'][name].items():
                for key in ['physical','tda']:run.summary[f'{phase}/{name}/{lag}ps/{key}']=values[key]
    save_json(technical/'wandb_run.json',dict(id=run.id,url=run.url));run.finish()
    save_json(technical/'status.json',dict(state='complete',test_sources=len(np.unique(data['source'][data['split']=='test'])),rows=len(data['row_id'])))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--deadline-utc',required=True)
    args=p.parse_args();run(json.loads(Path(args.config).read_text()),datetime.fromisoformat(args.deadline_utc).timestamp())
