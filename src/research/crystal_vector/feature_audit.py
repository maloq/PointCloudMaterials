"""Train-fitted feature concentration, proper-score probes and frozen-head ablations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.spatial_approach.evaluate import csv_rows
from .overfit import population_weights


def load(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['run'])
    out=resolve_path(c['output']);snapshot_metric_docs(out,'crystal_feature_dominance')
    technical=root/'analyses/localization-v1/technical'
    receipt=json.loads((technical/'predictions.json').read_text())
    if sha(technical/'predictions.npz')!=receipt['sha256'] or sha(root/'technical/best.pt')!=receipt['checkpoint_sha256']:
        raise ValueError('Frozen model or inference changed')
    with np.load(technical/'predictions.npz') as a:values={k:a[k] for k in a.files}
    with np.load(technical/'rows.npz') as a:meta={k:a[k] for k in a.files}
    write_json(out/'technical/input.json',dict(config=c,checkpoint_sha256=receipt['checkpoint_sha256'],
        predictions_sha256=receipt['sha256'],rows_sha256=sha(technical/'rows.npz'),
        run=str(root),training=False))
    return c,root,out,values,meta


def pca(x,w):
    mean=w@x;center=x-mean;cov=(center*w[:,None]).T@center
    eig,basis=np.linalg.eigh(cov);order=np.argsort(eig)[::-1]
    return mean,eig[order].clip(0),basis[:,order]


def fit_logistic(x,y,w,ridge):
    mean=w@x;std=np.sqrt(w@(x-mean)**2).clip(1e-6)
    a=np.c_[(x-mean)/std,np.ones(len(x))];d=x.shape[1]
    start=np.zeros(d+1);rate=float(w@y);start[-1]=np.log(rate/(1-rate))
    def loss(coef):
        score=a@coef;residual=(expit(score)-y)*w
        penalty=np.r_[ridge*coef[:-1],0.]
        return float(w@(np.logaddexp(0,score)-y*score)+.5*ridge*(coef[:-1]@coef[:-1])),a.T@residual+penalty
    fitted=minimize(loss,start,jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-7,ftol=1e-11))
    if not fitted.success:raise ValueError(f'Probe optimizer failed: {fitted.message}')
    return mean,std,fitted.x,dict(iterations=int(fitted.nit),objective=float(fitted.fun))


def score_rows(meta,probability,radius,extra):
    y=(meta['distance']<=radius).astype(float);p=np.clip(np.asarray(probability,dtype=np.float64),1e-9,1-1e-9)
    nll=-(y*np.log(p)+(1-y)*np.log1p(-p));brier=(p-y)**2;rows=[]
    if not np.isfinite(nll).all():raise ValueError('Nonfinite proximity log score')
    for role in ('train','selection','test'):
        base=population_weights(meta,role)
        for visibility,mask in [('all',np.ones(len(y),bool)),('invisible',~meta['visible_context']),('visible',meta['visible_context'])]:
            w=base*mask;w/=w.sum()
            rows.append(dict(**extra,role=role,visibility=visibility,radius_A=radius,rows=int((w>0).sum()),
                prevalence=float(w@y),mean_probability=float(w@p),log_loss=float(w@nll),brier=float(w@brier)))
    return rows,nll,brier


def source_interval(meta,loss,reference,seed,draws):
    base=population_weights(meta,'test');ids=base>0;sources=np.unique(meta['source'][ids]);rng=np.random.default_rng(seed)
    result=[]
    for visibility,mask in [('all',np.ones(len(base),bool)),('invisible',~meta['visible_context'])]:
        w=base*mask
        mass=np.asarray([w[meta['source']==s].sum() for s in sources])
        total=np.asarray([w[meta['source']==s]@(loss-reference)[meta['source']==s] for s in sources])
        samples=rng.integers(0,len(sources),(draws,len(sources)))
        delta=total[samples].sum(1)/mass[samples].sum(1)
        result.append(dict(visibility=visibility,test_delta=float(total.sum()/mass.sum()),
            low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),sources=len(sources)))
    return result


def probes(config_path):
    c,root,out,values,meta=load(config_path);weights=population_weights(meta,'train');train=weights>0;w=weights[train]
    table=[];spectra=[];channels=[];associations=[];intervals=[];receipts=[];saved={}
    # Named label-side correlations diagnose cues; none is an encoder/probe covariate.
    cues={'capped_distance':np.minimum(meta['distance'],64),'visible_interface':meta['visible_context'].astype(float),
        'inside_crystal':meta['inside_crystal'].astype(float),'interface_exists':meta['interface_exists'].astype(float),
        'weighted_count_5A':values['physical'][:,24],'weighted_count_8A':values['physical'][:,25]}
    for i,l in enumerate((2,4,6)):
        for j,r in enumerate((5,8)):cues[f'bond_power_l{l}_{r}A']=values['physical'][:,26+2*i+j]
    for field in ('local_z','context_z'):
        x=values[field].astype(float);mean,eig,basis=pca(x[train],w);pc=(x-mean)@basis
        saved[field+'_pca_mean']=mean;saved[field+'_pca_basis']=basis;saved[field+'_eigenvalues']=eig
        order=np.argsort(np.diag((x[train]-mean).T@((x[train]-mean)*w[:,None])))[::-1]
        for role in ('train','selection','test'):
            wr=population_weights(meta,role);center=x-wr@x;std=np.sqrt(wr@center**2);variance=wr@(pc-wr@pc)**2
            for k in c['ranks']:
                spectra.append(dict(field=field,role=role,top_k=k,training_PC_variance_share=float(variance[:k].sum()/variance.sum()),
                    largest_native_channel_variance_share=float(std.max()**2/(std@std))))
            for channel in range(128):channels.append(dict(field=field,role=role,channel=channel,std=float(std[channel]),
                train_variance_rank=int(np.flatnonzero(order==channel)[0]+1)))
            for k in range(8):
                a=pc[:,k]-wr@pc[:,k];sa=np.sqrt(wr@(a*a))
                for name,cue in cues.items():
                    b=cue-wr@cue;sb=np.sqrt(wr@(b*b))
                    associations.append(dict(field=field,role=role,pc=k+1,cue=name,
                        correlation=float(wr@(a*b)/(sa*sb)) if sa*sb>0 else None))
                total=float(wr@(a*a));between=0.
                for sid in np.unique(meta['source'][wr>0]):
                    ws=wr*(meta['source']==sid);mass=ws.sum();between+=float(ws@a)**2/mass
                associations.append(dict(field=field,role=role,pc=k+1,cue='source_between_variance_fraction',
                    correlation=between/total if total>0 else None))
        designs={'full':x,'all_PC':pc,'magnitude_only':np.linalg.norm(x,axis=1)[:,None]}
        for k in c['ranks']:
            designs[f'top_PC_{k}']=pc[:,:k];designs[f'drop_PC_{k}']=pc[:,k:]
        for radius in c['probe_radii_A']:
            y=(meta['distance']<=radius).astype(float);prevalence=float(w@y[train]);losses={}
            for name,design in designs.items():
                mu,scale,coef,fit=fit_logistic(design[train],y[train],w,c['ridge'])
                probability=expit((design-mu)/scale@coef[:-1]+coef[-1])
                scores,nll,brier=score_rows(meta,probability,radius,dict(field=field,design=name,features=design.shape[1]))
                table.extend(scores);losses[name]=nll
                key=f'{field}_{radius}_{name}';saved[key+'_probability']=probability.astype(np.float32)
                saved[key+'_coef']=coef;saved[key+'_mean']=mu;saved[key+'_scale']=scale
                receipts.append(dict(field=field,radius_A=radius,design=name,**fit))
            scores,nll,_=score_rows(meta,np.full(len(y),prevalence),radius,dict(field=field,design='train_prevalence',features=0));table.extend(scores)
            for name,loss in losses.items():
                reference='all_PC' if 'PC' in name else 'full'
                intervals.extend(dict(field=field,radius_A=radius,design=name,reference=reference,**row)
                    for row in source_interval(meta,loss,losses[reference],c['seed'],c['bootstrap_draws']))
            print(json.dumps(dict(stage='feature-probes',field=field,radius_A=radius)),flush=True)
    for radius in c['probe_radii_A']:
        y=(meta['distance']<=radius).astype(float);x=values['physical'].astype(float)
        mu,scale,coef,fit=fit_logistic(x[train],y[train],w,c['ridge'])
        p=expit((x-mu)/scale@coef[:-1]+coef[-1]);table.extend(score_rows(meta,p,radius,dict(field='physical',design='32_smooth_descriptors',features=32))[0])
        original=values['cdf'][:,3 if radius==20 else 4]
        table.extend(score_rows(meta,original,radius,dict(field='trained_predictor',design='original',features=0))[0])
    for name,rows in [('probe-scores',table),('feature-concentration',spectra),('channels',channels),('cue-associations',associations),('probe-source-intervals',intervals)]:
        csv_rows(out/'tables'/f'{name}.csv',rows)
    np.savez(out/'technical/probes.npz',**saved)
    write_json(out/'technical/probes-complete.json',dict(optimizer=receipts,rows=int(train.sum()),
        training_only_transforms=True,tables={p.name:sha(p) for p in (out/'tables').glob('*.csv')}))


def interventions(config_path,device):
    import torch
    from .data import ResidentContexts
    from .model import JointCrystalVector,objective
    from .train import compile_model
    from src.research.spatial_distance.model import cdf
    c,root,out,values,meta=load(config_path);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    checkpoint=torch.load(root/'technical/best.pt',map_location=device,weights_only=False);config=checkpoint['config']
    model=JointCrystalVector(checkpoint['encoder_config'],config).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    rng=np.random.default_rng(c['seed']);ids=[]
    for role in ('train','selection','test'):
        for sid in np.unique(meta['source'][meta['role']==role]):
            for kind in (0,1):
                pool=np.flatnonzero((meta['source']==sid)&(meta['kind']==kind))
                ids.extend(rng.choice(pool,min(len(pool),c['intervention_rows_per_source_population']),replace=False))
    ids=np.sort(ids);selected={k:v[ids] for k,v in meta.items()};cache=out/'technical/intervention-inputs.npz'
    if cache.exists():
        with np.load(cache) as a:
            if not np.array_equal(a['ids'],ids):raise ValueError('Different intervention sample')
            z=a['z'];v=a['v'];actual=a['actual']
    else:
        data=ResidentContexts(config,device);compile_model(model,data,config);zs=[];vs=[]
        with torch.no_grad():
            for begin in range(0,len(ids),c['intervention_batch']):
                b=data.batch(ids[begin:begin+c['intervention_batch']])
                with torch.autocast('cuda',dtype=torch.bfloat16):zz,vv=model.encode(b['positions'])
                zs.append(zz[b['inverse']].cpu().numpy());vs.append(vv[b['inverse']].cpu().numpy())
                if begin%(8*c['intervention_batch'])==0:print(json.dumps(dict(stage='intervention-encode',rows=begin,total=len(ids))),flush=True)
        z=np.concatenate(zs);v=np.concatenate(vs);actual=data.meta['actual'][ids]
        np.savez(cache,ids=ids,z=z,v=v,actual=actual);del data;torch.cuda.empty_cache()
    weights=population_weights(selected,'train');train=weights>0
    mean,eig,basis=pca(z[train].reshape(-1,128).astype(float),np.repeat(weights[train]/25,25))
    std=np.sqrt(np.repeat(weights[train]/25,25)@(z[train].reshape(-1,128)-mean)**2).clip(1e-6)
    zt=torch.tensor(z,device=device);vt=torch.tensor(v,device=device);at=torch.tensor(actual,device=device)
    mu=torch.tensor(mean,dtype=zt.dtype,device=device);vectors=torch.tensor(basis,dtype=zt.dtype,device=device)
    scale=torch.tensor(std,dtype=zt.dtype,device=device);plans=['original','scalar_mean','scalar_clip_3sd','vectors_zero']
    plans += [f'{mode}_{k}' for mode in ('scalar_top_PC','scalar_drop_PC') for k in (1,2,4,8)]
    predictions={};scores=[];intervals=[];native_encode=model.encode
    try:
        with torch.no_grad():
            for treatment in plans:
                parts=[];losses=[]
                for begin in range(0,len(ids),c['intervention_batch']):
                    stop=min(begin+c['intervention_batch'],len(ids));zz=zt[begin:stop];vv=vt[begin:stop]
                    if treatment=='scalar_mean':zz=mu.expand_as(zz)
                    elif treatment=='scalar_clip_3sd':zz=mu+(zz-mu).clamp(-3*scale,3*scale)
                    elif treatment=='vectors_zero':vv=torch.zeros_like(vv)
                    elif treatment.startswith(('scalar_top_PC','scalar_drop_PC')):
                        k=int(treatment.rsplit('_',1)[1]);projection=(zz-mu)@vectors[:,:k]@vectors[:,:k].T
                        zz=mu+projection if treatment.startswith('scalar_top') else zz-projection
                    model.encode=lambda positions:(zz.flatten(0,1),vv.flatten(0,1))
                    b=dict(positions=None,inverse=torch.arange((stop-begin)*25,device=device).reshape(-1,25),actual=at[begin:stop],
                        distance=torch.tensor(selected['distance'][begin:stop],device=device),
                        direction=torch.tensor(selected['direction'][begin:stop],device=device),
                        valid=torch.tensor(selected['direction_valid'][begin:stop],device=device))
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model(b)
                    terms=objective(result,b,config['loss'],True)
                    losses.append(torch.stack([terms[k] for k in ('objective','distance_nll','direction_nll')],-1).cpu().numpy())
                    parts.append(cdf(result['parts'],b['distance'].new_tensor(c['probe_radii_A'])).cpu().numpy())
                pred=np.concatenate(parts);loss=np.concatenate(losses);predictions[treatment+'_probability']=pred;predictions[treatment+'_loss']=loss
                for j,radius in enumerate(c['probe_radii_A']):
                    rows,_,_=score_rows(selected,pred[:,j],radius,dict(treatment=treatment))
                    for row in rows:
                        w=population_weights(selected,row['role'])
                        if row['visibility']!='all':w*=selected['visible_context'] if row['visibility']=='visible' else ~selected['visible_context']
                        w/=w.sum();row.update(predictive_objective=float(w@loss[:,0]),distance_nll=float(w@loss[:,1]))
                    scores.extend(rows)
                print(json.dumps(dict(stage='frozen-head',treatment=treatment)),flush=True)
    finally:model.encode=native_encode
    original=predictions['original_probability'];delta=np.abs(original-values['cdf'][ids][:,[3,4]])
    # Rebatched BF16 inference has small arithmetic drift. Record it and refuse a
    # discrepancy large enough to undermine these interventions.
    fidelity=dict(mean_abs_probability_error=float(delta.mean()),max_abs_probability_error=float(delta.max()),
        rows=len(ids),checkpoint_sha256=sha(root/'technical/best.pt'))
    if delta.max()>.02:raise ValueError(f'Frozen-head baseline failed fidelity check: {fidelity}')
    for treatment in plans:
        intervals.extend(dict(treatment=treatment,score='predictive_objective',reference='original',**r)
            for r in source_interval(selected,predictions[treatment+'_loss'][:,0],predictions['original_loss'][:,0],c['seed'],c['bootstrap_draws']))
    csv_rows(out/'tables/frozen-head-scores.csv',scores);csv_rows(out/'tables/frozen-head-source-intervals.csv',intervals)
    np.savez(out/'technical/intervention-predictions.npz',ids=ids,**predictions)
    np.savez(out/'technical/intervention-pca.npz',mean=mean,basis=basis,eigenvalues=eig,std=std)
    write_json(out/'technical/interventions-complete.json',fidelity)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['probes','interventions'])
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='probes':probes(a.config)
    else:interventions(a.config,a.device)
