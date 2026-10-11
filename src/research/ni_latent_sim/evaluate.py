"""Free 6-ps rollouts, predictive scores and original Ni birth readouts."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from .artifacts import write_rows as write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws,bootstrap_scores
from src.research.response_dense_birth.pooled import fit
from src.research.equivariant_context.cache import RetainedCache
from .data import load,root,cache,parent,read,sha,digest,write_json,atomic_save,resolve_path,FAMILY
from .model import make_model,nll


def fitted(c,arm):
    folder=root(c)/'analyses'/arm['name']/'technical'
    complete=folder/'complete.json';checkpoint_hash=sha(folder/'best.pt')
    if complete.exists() and checkpoint_hash!=read(complete)['checkpoint_sha256']:raise ValueError('Changed selected checkpoint')
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    plan=read(root(c)/'technical/plan.json')
    if saved['identity']!=digest([plan['identity'],arm]):raise ValueError('Selected checkpoint belongs to a different fit')
    if not complete.exists():
        progress=read(folder/'progress.json')
        write_json(folder/'evaluation-training-status.json',dict(state='partial_training',
            completed_updates=progress['update'],requested_updates=c['training']['updates'],
            selected_update=saved['update'],checkpoint_sha256=checkpoint_hash,
            selection='best available original validation-NLL checkpoint; not a completed fit'))
    model=make_model(arm).cuda().eval();model.load_state_dict(saved['model']);return model


def feature_pool(c):
    if 'feature_cache' in c:return RetainedCache(resolve_path(c['feature_cache']),6)
    folder=resolve_path(c['enrichment']['base_cache']) if 'enrichment' in c else cache(c)
    return RetainedCache(folder/'features',6)


def aggregate(values, ancestors, seed):
    ancestors = np.asarray(ancestors)
    keys = np.unique(ancestors)
    means = np.array([values[ancestors == key].mean() for key in keys])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(keys), size=(1000, len(keys)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(means.mean()), float(lo), float(hi)


def rollout_rows(values, ancestors, offsets, seed, **context):
    rows = []
    sources = len(set(ancestors))
    for (space, metric), scores in values.items():
        for horizon, column in zip(offsets, scores.T):
            value, lo, hi = aggregate(column, ancestors, seed)
            rows.append(dict(context, space=space, metric=metric, horizon_ps=horizon*.1,
                             value=value, lo=lo, hi=hi, sources=sources, rows=len(ancestors)))
    return rows


def run(c,p,arm):
    out=root(c)/'analyses'/arm['name'];tech=out/'technical'
    if (tech/'evaluation-complete.json').exists():return
    model=fitted(c,arm);data=load(c,'data');ids=np.flatnonzero([r['role']=='test' for r in data['rows']]);x=(data['x']-data['mean'])/data['scale'];offsets=[1,5,10,30,60]
    values={};ancestors=[data['rows'][i]['ancestor'] for i in ids]
    with torch.no_grad():
        for block in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
            start=x[block,:4].cuda();truth=x[block][:,[3+h for h in offsets]].cuda()
            g=torch.Generator(device='cuda').manual_seed(c['seed']+int(block[0]));mu,lv=model.rollout(start,60,c['evaluation']['particles'],g)
            mu=mu[:,:,[h-1 for h in offsets]];lv=lv[:,:,[h-1 for h in offsets]]
            samples=mu+torch.exp(.5*lv)*torch.randn(mu.shape,device='cuda',generator=g)
            for space,sl in [('structure',slice(0,128)),('motion',slice(128,256)),('descriptors',slice(256,280))]:
                m=mu[...,sl];v=lv[...,sl];y=truth[...,sl];s=samples[...,sl];dim=s.shape[-1]
                es=(s-y[:,None]).norm(dim=-1).mean(1)/dim**.5-.5*(s[:,:len(s[0])//2]-s[:,len(s[0])//2:]).norm(dim=-1).mean(1)/dim**.5
                metrics=dict(nll=nll(m,v,y),energy_score=es,mse=(m.mean(1)-y).square().mean(-1),coverage90=((y>=s.quantile(.05,dim=1))&(y<=s.quantile(.95,dim=1))).float().mean(-1),persistence_mse=(start[:,-1,None,sl]-y).square().mean(-1),variance_ratio=s.var(1).mean(-1)/(y-start[:,-1,None,sl]).square().mean(-1).clamp_min(1e-4))
                if 'stability' in c or 'dense_response' in c:
                    particle_error=(m-y[:,None]).square().mean(-1)
                    metrics.update(training_mean_mse=y.square().mean(-1),
                        particle_failure_rate=((particle_error>100)|~torch.isfinite(particle_error)).float().mean(1),
                        input_failure_rate=((metrics['mse']>100)|~torch.isfinite(metrics['mse'])).float(),
                        nonfinite_particle_rate=(~torch.isfinite(m).all(-1)).float().mean(1),
                        predictive_variance=s.var(1).mean(-1))
                for key,a in metrics.items():values.setdefault((space,key),[]).append(a.cpu().numpy())
    values={key:np.concatenate(parts) for key,parts in values.items()}
    report=rollout_rows(values,ancestors,offsets,c['seed'],model=arm['name'])
    write_metric_rows(report,out,family=FAMILY,name='free_rollout',columns=list(report[0]))
    by_regime=[]
    if 'enrichment' in c or c['evaluation'].get('by_regime',False):
        regimes=np.array([data['rows'][i]['regime'] for i in ids])
        for regime in ['original','liquid_liquid','growing_front']:
            mask=regimes==regime;parents=np.asarray(ancestors)[mask]
            if not mask.any():raise ValueError('Missing held-out motion regime: '+regime)
            by_regime.extend(rollout_rows({key:a[mask] for key,a in values.items()},parents,
                                         offsets,c['seed'],model=arm['name'],regime=regime))
        write_metric_rows(by_regime,out,family=FAMILY,name='free_rollout_by_regime',columns=list(by_regime[0]))
    atomic_save(tech/'rollout-scores.pt',dict(ids=ids,ancestors=ancestors,offsets_ps=[h*.1 for h in offsets],metrics={f'{s}/{m}':a for (s,m),a in values.items()}))
    birth_codes(c,p,arm,model=model,data=data)
    from .train import short_loss
    bank=load(c,'short');held=np.flatnonzero([r['role']=='test' for r in bank]);short=[]
    with torch.no_grad():
        for block in np.array_split(held,max(1,int(np.ceil(len(held)/16)))):
            value,response=short_loss(model,bank,block,data['mean'].cuda(),data['scale'].cuda(),dict(arm,response=True),c['seed']+int(block[0]))
            short.append(dict(model=arm['name'],first_group=int(block[0]),groups=len(block),value_half_mse=float(value),relative_response=float(response)))
    write_metric_rows(short,out,family=FAMILY,name='heldout_physical_response',columns=list(short[0]))
    write_json(tech/'evaluation-complete.json',dict(state='complete',rows=len(report)))
    update_recorded_summary(tech/'wandb.json',{f"rollout/{r['space']}/{r['horizon_ps']:g}ps/{r['metric']}":r['value'] for r in report},evaluation='free-rollout-v1',expected=dict(mode='online'))
    if by_regime:
        update_recorded_summary(tech/'wandb.json',{f"regime/{r['regime']}/{r['space']}/{r['horizon_ps']:g}ps/{r['metric']}":r['value'] for r in by_regime},evaluation='free-rollout-regimes-v1',expected=dict(mode='online'))


def birth_codes(c,p,arm,*,model=None,data=None):
    """Generate frozen-probe inputs independently of rollout scoring."""
    tech=root(c)/'analyses'/arm['name']/'technical'
    if model is None:model=fitted(c,arm)
    if data is None:data=load(c,'data')
    # Export compact inputs for downstream diagnostic fits, preserving every row.
    birth=load(c,'births');z=[]
    with torch.no_grad():
        for block in birth['x'].split(256):
            xx=((block-data['mean'])/data['scale']).cuda();z.append(model.encode(xx[:,-arm['history']:]).flatten(1).cpu())
    metadata=dict(plan=p['identity'],arm=arm,checkpoint=sha(tech/'best.pt'))
    pool=feature_pool(c)
    with pool.lease(digest(metadata),deadline=time.time()+3600,metadata=metadata) as folder:
        atomic_save(folder/'birth-codes.pt',dict(x=torch.cat(z),rows=birth['rows']))
        write_json(folder/'complete.json',dict(sha256=sha(folder/'birth-codes.pt')))
    write_json(tech/'birth-code-cache.json',dict(path=str(folder),metadata=metadata))


def births(c,p,arm):
    out=root(c)/'analyses'/arm['name'];tech=out/'technical'
    if (tech/'birth-evaluation-complete.json').exists():return
    pointer_path=tech/'birth-code-cache.json'
    if not pointer_path.exists() or not (Path(read(pointer_path)['path'])/'complete.json').exists():
        birth_codes(c,p,arm)
    pointer=read(pointer_path);pool=feature_pool(c)
    with pool.lease(digest(pointer['metadata']),deadline=time.time()+3600,metadata=pointer['metadata'],shared=True) as folder:
        if sha(folder/'birth-codes.pt')!=read(folder/'complete.json')['sha256']:raise ValueError('Changed compact feature cache')
        b=torch.load(folder/'birth-codes.pt',map_location='cpu',weights_only=False)
    pc=parent(c)['config'];report=[]
    for task in ['nucleus32','nucleus64']:
        for lead in [1.,3.,6.]:
            indices=[i for i,r in enumerate(b['rows']) if r['task']==task and r['lead']==lead];records=[b['rows'][i] for i in indices];counts=Counter(r['event'] for r in records)
            rows={k:np.asarray([r[k] for r in records]) for k in ['id','ancestor','role','event','pair','label']};rows['weight']=np.asarray([1/counts[r['event']] for r in records]);xx=b['x'][indices].numpy()
            ec=dict(c,seed=arm['seed'],output=str(root(c)/'analyses'/task));pp=dict(identity=digest([p['identity'],task,lead]),partition=p['partition'])
            fit(ec,dict(config=pc['evaluation']),rows,xx,dict(name=arm['name']),'compact_history',lead,'catboost',plan=pp)
            f=root(ec)/'analyses/fits-v1'/arm['name']/'compact_history'/f'lead-{lead:g}ps/catboost/technical/predictions.npz'
            ids=np.flatnonzero(np.isin(rows['ancestor'],p['partition']['test']));pr=np.load(f)['probability'][ids];y=rows['label'][ids];w=rows['weight'][ids]
            ancestors,slot,draws=source_draws(rows['ancestor'][ids],1000,c['seed']);bs,_=bootstrap_scores(y,pr,w,slot,draws,None);v=scores(y,pr,w)
            for metric in ['ap','auroc','nll','brier']:
                lo,hi=np.quantile(bs[metric],[.025,.975]);report.append(dict(model=arm['name'],task=task,lead_ps=lead,metric=metric,value=v[metric],lo=lo,hi=hi,rows=len(ids),ancestors=len(ancestors),prevalence=v['prevalence'],code_dimension=arm['dimension'],history=arm['history'],total_input_dimension=arm['dimension']*arm['history']))
    write_metric_rows(report,out,family=FAMILY,name='nucleus_prediction',columns=list(report[0]))
    write_json(tech/'birth-evaluation-complete.json',dict(state='complete'))
    update_recorded_summary(tech/'wandb.json',{f"birth/{r['task']}/{r['lead_ps']:g}ps/{r['metric']}":r['value'] for r in report},evaluation='birth-v1',expected=dict(mode='online'))
