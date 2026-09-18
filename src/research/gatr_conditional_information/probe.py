"""Nested source-held-out linear and nonlinear ridge probes on the A100."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.data.structural_pretraining.prepare import save_json,file_hash
from .data import load,require_hardware

BOND_NAMES=('q4','q6','w4','w6','qbar6','q6_coherence')
TARGET_NAMES=(*BOND_NAMES,*(f'angular_L{i}' for i in range(1,17)))


def source_weights(source):
    _,inverse,count=np.unique(source,return_inverse=True,return_counts=True)
    w=1/count[inverse].astype(float)
    return w/w.sum()


def feature_sets(a,task,methods=None):
    radial=np.concatenate((a['radial'],a['context']),axis=1).astype(float)
    base=np.concatenate((radial,a['radial_gatr']),axis=1)
    delta=a['gatr'].astype(float)-a['radial_gatr'].astype(float)
    if methods is None:
        methods=['radial','radial_control','plus_radial_duplicate','plus_gatr','plus_angular_delta','plus_mace','plus_soap','plus_tda']
        if task=='future':methods+=['current_order','current_order_radial_duplicate','current_order_plus_gatr']
    result={}
    for method in methods:
        if method=='radial':result[method]=radial
        elif method=='radial_control':result[method]=base
        elif method=='plus_radial_duplicate':result[method]=np.concatenate((base,a['radial_gatr']),axis=1)
        elif method=='plus_angular_delta':result[method]=np.concatenate((base,delta),axis=1)
        elif method in ('plus_gatr','plus_mace','plus_soap','plus_tda'):
            result[method]=np.concatenate((base,a[method.removeprefix('plus_')]),axis=1)
        elif method in ('current_order','current_order_plus_gatr','current_order_radial_duplicate'):
            current=np.concatenate((base,a['bond'],a['angular']),axis=1)
            if method=='current_order_plus_gatr':result[method]=np.concatenate((current,delta),axis=1)
            elif method=='current_order_radial_duplicate':result[method]=np.concatenate((current,a['radial_gatr']),axis=1)
            else:result[method]=current
        else:raise ValueError(f'Unknown feature set: {method}')
    return result


@torch.no_grad()
def predict_path(xtrain,ytrain,xtest,sources,family,penalties,seed,rff_dimensions,binary,pooled_tail=0):
    """All candidate penalties in one weighted float64 eigensolve.

    Every preprocessing moment, including nonlinear-map centering and target
    scale, uses this fit's training rows only. No row from its validation/test
    sources enters the fit. RFF frequencies span three fixed length scales.
    """
    device='cuda'
    x=torch.as_tensor(xtrain,dtype=torch.float64,device=device)
    xt=torch.as_tensor(xtest,dtype=torch.float64,device=device)
    y=torch.as_tensor(ytrain,dtype=torch.float64,device=device)
    w=torch.as_tensor(source_weights(sources),dtype=torch.float64,device=device)
    mean=w@x;variance=w@((x-mean).square())
    keep=variance>1e-24
    if not keep.any():raise ValueError('No varying probe inputs')
    normalization_variance=variance.clone()
    if pooled_tail:
        normalization_variance[-pooled_tail:]=variance[-pooled_tail:].mean()
    scale=normalization_variance[keep].sqrt()
    x=(x[:,keep]-mean[keep])/scale;xt=(xt[:,keep]-mean[keep])/scale
    x/=np.sqrt(int(keep.sum()));xt/=np.sqrt(int(keep.sum()))
    if family=='nonlinear':
        rng=np.random.default_rng(seed)
        length=np.resize(np.array([.5,1.,2.]),rff_dimensions)
        omega=torch.tensor(rng.normal(size=(x.shape[1],rff_dimensions))/length,device=device,dtype=torch.float64)
        phase=torch.tensor(rng.uniform(0,2*np.pi,rff_dimensions),device=device,dtype=torch.float64)
        factor=np.sqrt(2/rff_dimensions)
        x=torch.cat((x,factor*torch.cos(x@omega+phase)),1)
        xt=torch.cat((xt,factor*torch.cos(xt@omega+phase)),1)
    elif family!='linear':raise ValueError(f'Unknown probe {family}')
    fm=w@x;x=x-fm;xt=xt-fm
    ym=w@y
    ys=torch.ones_like(ym) if binary else (w@((y-ym).square())).sqrt()
    if (ys<=1e-12).any():raise ValueError('Constant regression target in a training fold')
    yc=(y-ym)/ys
    gram=x.T@(w[:,None]*x)
    rhs=x.T@(w[:,None]*yc)
    eigen,vec=torch.linalg.eigh(gram)
    if eigen.min() < -1e-10:raise FloatingPointError('Weighted ridge Gram is not positive semidefinite')
    projection=vec.T@rhs
    predictions=[]
    for penalty in penalties:
        coef=vec@(projection/(eigen[:,None]+penalty))
        value=(xt@coef)*ys+ym
        if binary:value=value.clamp(0,1)
        predictions.append(value.cpu().numpy())
    return np.stack(predictions),ym.cpu().numpy(),ys.cpu().numpy()


def selected_prediction(x,y,fit,test,sources,family,config,fold_seed,binary,pooled_tail=0):
    penalties=config['ridge_penalties'];errors=np.zeros((len(penalties),y.shape[1]))
    folds=GroupKFold(n_splits=config['inner_folds'])
    for inner,(tr,va) in enumerate(folds.split(fit,groups=sources[fit])):
        tr,va=fit[tr],fit[va]
        if set(sources[tr])&set(sources[va]):raise ValueError('Inner source leakage')
        predicted,_,scale=predict_path(x[tr],y[tr],x[va],sources[tr],family,penalties,
            fold_seed,config['rff_dimensions'],binary,pooled_tail)
        # Scale is fitted only on the corresponding inner training sources.
        loss=((predicted-y[va][None])/scale[None,None])**2
        errors+=np.einsum('n,pnt->pt',source_weights(sources[va]),loss)/config['inner_folds']
    best=np.argmin(errors,axis=0)
    predictions,mean,scale=predict_path(x[fit],y[fit],x[test],sources[fit],family,penalties,
        fold_seed,config['rff_dimensions'],binary,pooled_tail)
    selected=np.stack([predictions[k,:,j] for j,k in enumerate(best)],axis=1)
    return selected,mean,scale,dict(penalty_per_target=[penalties[i] for i in best],inner_mse=errors.tolist())


def run(config):
    require_hardware(config)
    a,sources,parent=load(config);root=Path(config['output'])/'technical/probes';root.mkdir(exist_ok=True)
    stride=a['frame']%config['probe_stride']==0
    for task in ('structure','future'):
        y=np.concatenate((a['bond'],a['angular']),axis=1).astype(float) if task=='structure' else a['future'].astype(float)
        eligible=np.ones(len(y),bool) if task=='structure' else a['future_eligible']
        sets=feature_sets(a,task)
        for family in ('linear','nonlinear'):
            for method,x in sets.items():
                directory=root/task/family/method;directory.mkdir(parents=True,exist_ok=True)
                for source in sources:
                    sid=source['id'];path=directory/f'{sid}.npz'
                    if path.with_suffix('.json').exists():continue
                    started=time.monotonic()
                    fit=np.flatnonzero(eligible&stride&(a['source']!=sid))
                    test=np.flatnonzero(eligible&(a['source']==sid))
                    if not len(test):raise ValueError(f'No eligible {task} rows for source {sid}')
                    if set(a['source'][fit])&set(a['source'][test]):raise ValueError('Outer source leakage')
                    seed=config['seed']+sid
                    pooled_tail=144 if method=='plus_tda' else 0
                    p,mean,scale,selection=selected_prediction(x,y,fit,test,a['source'],family,config,seed,task=='future',pooled_tail)
                    if not np.isfinite(p).all():raise FloatingPointError('Nonfinite out-of-source predictions')
                    np.savez(path,indices=test,prediction=p,target_mean=mean,target_scale=scale)
                    save_json(path.with_suffix('.json'),dict(task=task,family=family,method=method,test_source=sid,
                        training_sources=np.unique(a['source'][fit]).tolist(),training_rows=len(fit),test_rows=len(test),
                        input_dimensions=x.shape[1],rff_seed=seed,pooled_tail_dimensions=pooled_tail,
                        seconds=time.monotonic()-started,sha256=file_hash(path),**selection))
                    print(f'{task}/{family}/{method} source {sid}: fit={len(fit)} test={len(test)} {time.monotonic()-started:.1f}s',flush=True)
