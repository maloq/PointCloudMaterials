"""Mean, affine ridge and linear-logit controls for frozen descriptor fits."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows,check_metric_docs
from .data import config
from .descriptor_data import load
from .descriptor_fit import quantities,targets,metrics
from .comparisons import paired_scores,prediction_rows


NAMES=('training_mean','linear_ridge','linear_distribution')


def protocol(c):
    path=resolve_path(c['parent_config'])
    if sha(path)!=c['parent_config_sha256']:raise ValueError('Frozen boosting protocol changed')
    return config(path)


def table(root,name,rows):
    return write_metric_rows(rows, root, family='liquid_descriptor_baselines', name=name)


def point_scores(pred,target,w):
    w=w/w.sum();error=pred-np.minimum(target,64)
    return dict(distance_rmse_A=float(np.sqrt(w@(error*error))),distance_mae_A=float(w@np.abs(error)))


def fit(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    pc=protocol(c);x,rows,columns,manifest=load(pc);root=resolve_path(c['output'])
    tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    binding=dict(config=c,dataset=manifest['identity'],implementation=sha(Path(__file__)))
    if (tech/'identity.json').exists() and config(tech/'identity.json')!=binding:raise ValueError('Baseline identity changed')
    write_json(tech/'identity.json',binding)
    if (tech/'complete.json').exists():return
    train=np.flatnonzero(rows['role']=='train');val=np.flatnonzero(rows['role']=='selection')
    weight=rows['weights'];w=weight[train]/weight[train].sum();y=np.minimum(rows['target'],64).astype(float)
    target_mean=float(w@y[train]);variance=float(w@(y[train]-target_mean)**2)
    if variance<=0:raise ValueError('Constant distance target')
    dimension=x.shape[1];mu=np.zeros(dimension);second=np.zeros(dimension)
    for begin in range(0,len(train),2048):
        ids=train[begin:begin+2048];v=np.asarray(x[ids],float);ws=w[begin:begin+len(ids)]
        mu+=ws@v;second+=ws@(v*v)
    scale=np.sqrt(np.maximum(second-mu*mu,0)).clip(1e-4)
    np.savez(tech/'standardization.npz',mean=mu,scale=scale,train_sources=np.unique(rows['source'][train]))
    predictions={'training_mean':dict(mean_A=np.full(len(y),target_mean))}
    selections={'training_mean':dict(value_A=target_mean,selector='weighted training mean of min(distance,64)')}
    write_json(tech/'state.json',dict(state='fitting_ridge',train_rows=len(train),features=dimension))
    # A single weighted sufficient-statistic pass serves every ridge strength.
    gram=np.zeros((dimension,dimension));cross=np.zeros(dimension)
    for begin in range(0,len(train),2048):
        ids=train[begin:begin+2048];v=(np.asarray(x[ids],float)-mu)/scale;ws=w[begin:begin+len(ids)]
        vw=v*np.sqrt(ws[:,None]);gram+=vw.T@vw;cross+=v.T@(ws*(y[ids]-target_mean))
    eigen,basis=eigh(gram,check_finite=True);eigen=np.maximum(eigen,0);projected=basis.T@cross
    candidates=[];best=None
    for alpha in c['ridge_alphas']:
        coef=basis@(projected/(eigen+alpha));p=np.zeros(len(val))
        for begin in range(0,len(val),2048):
            ids=val[begin:begin+2048];p[begin:begin+len(ids)]=(np.asarray(x[ids],float)-mu)/scale@coef+target_mean
        mse=float(weight[val]@(p-y[val])**2)
        # Fixed train-only variance makes this a declared Gaussian likelihood
        # selector for the capped point target; it is not a censored density score.
        nll=.5*math.log(2*math.pi*variance)+mse/(2*variance)
        candidates.append(dict(alpha=alpha,validation_capped_gaussian_nll=nll,validation_rmse_A=math.sqrt(mse)))
        if best is None or nll<best[0]:best=(nll,coef,alpha)
    _,coef,alpha=best;p=np.empty(len(y))
    for begin in range(0,len(y),2048):p[begin:begin+2048]=(np.asarray(x[begin:begin+2048],float)-mu)/scale@coef+target_mean
    predictions['linear_ridge']=dict(mean_A=p)
    np.savez(tech/'ridge.npz',coefficient=coef,intercept=target_mean,alpha=alpha,normalization_mean=mu,normalization_scale=scale)
    selections['linear_ridge']=dict(alpha=alpha,selector='minimum validation Gaussian NLL on capped distance, fixed training-mean residual variance',
        fixed_variance_A2=variance,candidates=candidates,output='unclipped affine distance prediction')
    del gram,basis
    write_json(tech/'state.json',dict(state='fitting_linear_distribution',train_rows=len(train),features=dimension))
    # L2 multinomial regression has affine logits and no hidden layers. The
    # positive-definite regularizer also keeps the highly redundant basis stable.
    xt=(np.asarray(x[train],float)-mu)/scale;xv=(np.asarray(x[val],float)-mu)/scale
    labels=targets(rows['target'],pc);k=len(pc['distance_edges_A']);best_nll=float('inf');chosen=None;curves=[]
    for strength in c['logistic_C']:
        model=LogisticRegression(C=strength,l1_ratio=0,solver='lbfgs',max_iter=c['max_iter'],tol=c['tolerance'],random_state=c['seed'])
        started=time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(xt,labels[train],sample_weight=w)
        if not np.array_equal(model.classes_,np.unique(labels[train])):raise ValueError('Linear probability fit lost an observed training class')
        probabilities=np.zeros((len(val),k));probabilities[:,model.classes_]=model.predict_proba(xv)
        score=quantities(probabilities,rows['target'][val],pc)
        nll=float(weight[val]@score['nll'])
        curves.append(dict(C=strength,validation_distance_nll=nll,iterations=int(model.n_iter_.max()),seconds=time.time()-started))
        print(json.dumps(curves[-1]),flush=True)
        if nll<best_nll:
            best_nll=nll;fullcoef=np.zeros((k,dimension));fullbias=np.full(k,-np.inf)
            if len(model.classes_)==2:
                fullbias[model.classes_[0]]=0;fullcoef[model.classes_[1]]=model.coef_[0];fullbias[model.classes_[1]]=model.intercept_[0]
            else:
                fullcoef[model.classes_]=model.coef_;fullbias[model.classes_]=model.intercept_
            chosen=(fullcoef,fullbias,strength)
    coef,bias,strength=chosen;prob=np.empty((len(y),k))
    for begin in range(0,len(y),2048):
        z=(np.asarray(x[begin:begin+2048],float)-mu)/scale
        prob[begin:begin+2048]=softmax(z@coef.T+bias,axis=1)
    predictions['linear_distribution']=quantities(prob,rows['target'],pc)
    np.savez(tech/'linear-distribution.npz',coefficient=coef,intercept=bias,C=strength,normalization_mean=mu,normalization_scale=scale)
    selections['linear_distribution']=dict(C=strength,selector='minimum validation censored histogram distance NLL',candidates=curves,weight_sum=float(w.sum()),
        absent_training_bins=np.setdiff1d(np.arange(k),np.unique(labels[train])).tolist(),absent_bin_policy='zero predicted mass; shared 1e-12 scoring floor')
    receipts={};score_rows=[]
    for name,value in predictions.items():
        if not all(np.isfinite(v).all() for v in value.values()):raise FloatingPointError(f'Nonfinite {name}')
        model_root=root/name;analysis=model_root/'analyses/predictability-v1';(analysis/'technical').mkdir(parents=True,exist_ok=True)
        np.savez(analysis/'technical/predictions.npz',ids=rows['ids'],**value)
        write_json(model_root/'technical/prediction-context.json',dict(encoder=None,model=name,
            descriptor_features=0 if name=='training_mean' else dimension,patches=0 if name=='training_mean' else 25,
            max_support_A=0 if name=='training_mean' else 32,geometry_only=True,conditions=[],history=False,motion=False,
            teacher=None,relaxation=pc.get('observation',{}).get('relaxed',False),
            observation=pc.get('observation',{'domain':'original_MD'}),target_protocol=pc.get('target_protocol','original MD crystal distance'),
            targets_as_inputs=False,preprocessing_fit='training only',tracking='local descriptor control'))
        model_scores=[]
        for role in ('train','selection','calibration','test'):
            ids=np.flatnonzero(rows['role']==role)
            r=dict(model=name,role=role,rows=len(ids),sources=len(np.unique(rows['source'][ids])),distance_nll=None,
                brier20A=None,brier32A=None,brier48A=None,**point_scores(value['mean_A'][ids],rows['target'][ids],weight[ids]))
            if 'nll' in value:r.update(metrics({k:v[ids] for k,v in value.items()},rows['target'][ids],weight[ids]))
            model_scores.append(r);score_rows.append(r)
        table(analysis,'scores',model_scores)
        receipts[name]=dict(selection=selections[name],predictions_sha256=sha(analysis/'technical/predictions.npz'),test_used_for_selection=False)
        write_json(model_root/'technical/complete.json',receipts[name])
    table(root/'analyses/baselines-v1','scores',score_rows)
    write_json(tech/'complete.json',dict(identity=digest(binding),models=receipts,rows=manifest['rows']))
    write_json(tech/'state.json',dict(state='complete'))


def compare(c):
    pc=protocol(c);_,rows,_,_=load(pc);parent=resolve_path(pc['output']);root=resolve_path(c['output'])
    analysis=parent/'analyses/comparison-with-baselines-v2';pred={};receipts={}
    for name,base in [(a['name'],parent) for a in pc['arms']]+[(n,root) for n in NAMES]:
        p=base/name/'analyses/predictability-v1/technical/predictions.npz';receipt=config(base/name/'technical/complete.json')
        if sha(p)!=receipt['predictions_sha256']:raise ValueError(f'Changed predictions {name}')
        with np.load(p) as a:pred[name]={k:a[k] for k in a.files if k!='probability'}
        if not np.array_equal(prediction_rows(pred[name]['ids'],rows['ids']),np.arange(len(rows['ids']))):raise ValueError(f'Unmatched rows {name}')
        receipts[name]=dict(receipt=str(base/name/'technical/complete.json'),prediction_sha256=receipt['predictions_sha256'])
    scores=[];paired=[];weight=rows['weights'];truth=np.minimum(rows['target'],64)
    npoint=len(pred)-1;nprob=sum('nll' in v for v in pred.values())-1
    for name,value in pred.items():
        for role in ('selection','calibration','test'):
            ids=np.flatnonzero(rows['role']==role);w=weight[ids];w=w/w.sum()
            row=dict(model=name,role=role,kind='probabilistic' if 'nll' in value else 'point-only',distance_nll=None,brier20A=None,brier32A=None,brier48A=None,
                **point_scores(value['mean_A'][ids],truth[ids],w))
            if 'nll' in value:row.update(metrics({k:v[ids] for k,v in value.items() if k!='ids'},truth[ids],w))
            scores.append(row)
            if name=='training_mean':continue
            mse=(value['mean_A'][ids]-truth[ids])**2;refmse=(pred['training_mean']['mean_A'][ids]-truth[ids])**2
            gain=pred['prior']['nll'][ids]-value['nll'][ids] if 'nll' in value and name!='prior' else None
            result=dict(model=name,role=role,rows=len(ids),rmse_reference='training_mean',
                nll_reference='prior',nll_gain=None,nll_gain_ci95_low=None,nll_gain_ci95_high=None,nll_gain_familywise_low=None,nll_gain_familywise_high=None)
            result.update(paired_scores(rows['source'][ids],w,mse,refmse,gain=gain,seed=c['seed'],
                draws=c['bootstrap_draws'],rmse_comparisons=npoint,nll_comparisons=nprob))
            paired.append(result)
    selection=[s for s in scores if s['role']=='selection' and s['distance_nll'] is not None]
    best=min(selection,key=lambda s:s['distance_nll'])['model']
    table(analysis,'scores',scores);table(analysis,'paired-comparisons',paired)
    write_json(analysis/'technical/complete.json',dict(best_by_validation_distance_nll=best,point_models_excluded_from_density_selection=True,
        prediction_receipts=receipts,one_seed=True,test_used_for_selection=False,rmse_comparisons=npoint,nll_comparisons=nprob))
    lines=['# Boosting comparison with mean and linear controls','',f'Best probabilistic model by validation distance NLL: **{best}**.','',
        '| Model | Test RMSE (Å) | Test MAE (Å) | Test distance NLL |','|---|---:|---:|---:|']
    for r in scores:
        if r['role']=='test':
            nll='—' if r['distance_nll'] is None else f"{r['distance_nll']:.5f}"
            lines.append(f"| {r['model']} | {r['distance_rmse_A']:.5f} | {r['distance_mae_A']:.5f} | {nll} |")
    lines+=['','Mean and affine ridge are point predictors: no censored-density NLL or Brier scores are assigned to them.',
        'The linear distribution model has affine logits, no hidden layers, and the same distance bins as boosting.',
        'All models use matched held-out rows and weights. Transforms, means and coefficients use train only; regularization selection uses validation only.',
        'Paired uncertainty resamples sources, not atom centers. One training seed; no training-seed uncertainty.']
    (analysis/'README.md').write_text('\n'.join(lines)+'\n')


def submit(path):
    c=config(path);pc=protocol(c);root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():raise ValueError('Already submitted; use frozen commands')
    check_metric_docs(family='liquid_descriptor_baselines');repo=Path(__file__).resolve().parents[3];code=tech/'code'
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');(code/'configs/liquid_predictability').mkdir(parents=True)
    shutil.copy2(path,code/'configs/liquid_predictability'/Path(path).name);write_json(code/'config.json',c)
    launch=config(resolve_path(pc['output'])/'technical/launch.json');receipt=dict(code=str(code),config_sha256=sha(Path(path)),jobs={})
    def job(stage,dependency=None):
        command=[sys.executable,'-u','-m','src.research.liquid_predictability.descriptor_baselines',stage,'--config',str(code/'config.json')]
        script=tech/f'{stage}.sbatch'
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=LD-baseline-{stage}',
            '#SBATCH --partition=CPU','#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --cpus-per-task={c["threads"]}',
            '#SBATCH --mem=64G','#SBATCH --time=02:00:00',f'#SBATCH --output={tech}/{stage}-%j.log',
            'set -euo pipefail','cd '+shlex.quote(str(code)),
            'exec env '+shlex.join([f'PCM_PROJECT_ROOT={repo}','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1'])+' '+shlex.join(command),'']))
        args=['sbatch','--parsable']+(['--dependency='+dependency] if dependency else [])+[str(script)]
        ident=subprocess.check_output(args,text=True).strip().split(';')[0];receipt['jobs'][stage]=ident;write_json(tech/'launch.json',receipt);return ident
    fitting=job('fit')
    dependency='afterok:'+fitting
    if not (resolve_path(pc['output'])/'analyses/comparison-v1/technical/complete.json').exists():
        dependency+=':'+launch['jobs']['compare']
    job('compare',dependency)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['submit','fit','compare']);p.add_argument('--config',required=True)
    a=p.parse_args();c=config(a.config)
    if a.stage=='submit':print(json.dumps(submit(a.config),indent=2));return
    root=resolve_path(c['output']);tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    try:
        with threadpool_limits(limits=c['threads']):(fit if a.stage=='fit' else compare)(c)
    except BaseException:
        import traceback
        write_json(tech/f'{a.stage}-failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
