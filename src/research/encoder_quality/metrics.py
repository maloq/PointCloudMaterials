"""Source-paired proper scores and native-distance future-neighbor controls."""
import numpy as np
import torch
from scipy.stats import spearmanr
from src.research.local_predictability.metrics import source_weights, weighted_scores
from src.research.supervised_onset.evaluate import calibrate_risks


def score(corpus, risks):
    if risks.shape != (len(corpus.pop['event']),5) or not np.isfinite(risks).all():
        raise ValueError('Missing or nonfinite fixed-cohort predictions')
    if (np.diff(risks,axis=1)<-1e-7).any() or (risks<0).any() or (risks>1).any():
        raise ValueError('Invalid cumulative event probabilities')
    calibrated, calibration = calibrate_risks(corpus, risks)
    result = dict(calibration=calibration, scores={}, event_nll={})
    for name, prediction in [('raw', risks), ('calibrated', calibrated)]:
        probabilities=np.diff(np.c_[np.zeros(len(prediction)),prediction.astype(float),np.ones(len(prediction))],axis=1)
        losses=-np.log(probabilities[np.arange(len(prediction)),corpus.pop['event']].clip(1e-12))
        result['event_nll'][name]={role:float(source_weights(corpus.pop['source'][ids])@losses[ids])
            for role,ids in corpus.split.items() if role!='train'}
    for horizon,col in ((3,1),(6,2),(12,4)):
        result['scores'][str(horizon)]={}
        for role in ('selection','calibration','test'):
            ids=corpus.split[role];y=corpus.pop['event'][ids]<=col;s=corpus.pop['source'][ids]
            raw=weighted_scores(y,risks[ids,col],s)
            cal=weighted_scores(y,calibrated[ids,col],s)
            result['scores'][str(horizon)][role]=dict(raw=raw,calibrated=cal,
                windows=len(ids),positive_windows=int(y.sum()),sources=len(np.unique(s)))
    return result,calibrated


def paired_scores(corpus, candidate, baseline, draws, seed):
    ids=corpus.split['test'];source=corpus.pop['source'][ids];roots=np.unique(source)
    rng=np.random.default_rng(seed);indices=rng.integers(len(roots),size=(draws,len(roots)))
    result={}
    for horizon,col in ((3,1),(6,2),(12,4)):
        y=(corpus.pop['event'][ids]<=col).astype(float)
        errors=[]
        for risk in (candidate,baseline):
            p=risk[ids,col].astype(float).clip(1e-12,1-1e-12)
            errors.append({'brier':(p-y)**2,'log_loss':-y*np.log(p)-(1-y)*np.log1p(-p)})
        result[str(horizon)]={}
        for metric in ('brier','log_loss'):
            delta=errors[0][metric]-errors[1][metric]
            per=np.array([delta[source==s].mean() for s in roots])
            result[str(horizon)][metric]=dict(delta=float(per.mean()),
                ci95=np.quantile(per[indices].mean(1),[.025,.975]).tolist(),
                sources=len(roots),draws=draws,per_source=dict(zip(map(str,roots),map(float,per))))
    return result


@torch.no_grad()
def neighbor_forecast(corpus, features, neighbors, prior_strength, device):
    """Euclidean neighbors from fitting sources only; source-weighted votes."""
    fit=corpus.split['train'];x=np.asarray(features,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite neighbor features')
    reference=torch.as_tensor(x[fit],device=device)
    weights=source_weights(corpus.pop['source'][fit]);weights=weights/weights.mean()
    event=corpus.pop['event'][fit]
    outcomes=np.stack([event<=i for i in range(5)],axis=1).astype(float)
    prior=source_weights(corpus.pop['source'][fit])@outcomes
    # Fitting predictions are unused and filled with their fitting prior.
    risk=np.tile(prior,(len(x),1));ids=np.concatenate([corpus.split[r] for r in ('selection','calibration','test')])
    rnorm=reference.square().sum(1)
    for first in range(0,len(ids),256):
        take=ids[first:first+256];query=torch.as_tensor(x[take],device=device)
        distances=(query.square().sum(1)[:,None]+rnorm[None]-2*query@reference.T).clamp_min_(0)
        ni=distances.topk(neighbors,largest=False,sorted=True).indices.cpu().numpy()
        w=weights[ni]
        risk[take]=((w[:,:,None]*outcomes[ni]).sum(1)+prior_strength*prior)/(w.sum(1)[:,None]+prior_strength)
    return risk.astype(np.float32)


def movement_relation(z, physical, source, pairs, physical_scale, train_trace):
    a,b=pairs.T
    dz=np.linalg.norm(z[b]-z[a],axis=1)/np.sqrt(2*train_trace)
    dy=np.sqrt(np.mean(((physical[b]-physical[a])/physical_scale)**2,axis=1))
    result={}
    for s in np.unique(source[a]):
        keep=source[a]==s
        corr=spearmanr(dz[keep],dy[keep]).statistic
        result[str(s)]=dict(pairs=int(keep.sum()),spearman=None if not np.isfinite(corr) else float(corr),
            jump_rms=float(np.sqrt(np.mean(dz[keep]**2))),physical_change_rms=float(np.sqrt(np.mean(dy[keep]**2))))
    valid=[v['spearman'] for v in result.values() if v['spearman'] is not None]
    return dict(per_source=result,mean_source_spearman=float(np.mean(valid)) if valid else None,
        lag_ps=.75,coordinate_displacement_matched=False,
        interpretation='Association of exact-lag latent change with descriptor change; not causal sensitivity or atomic displacement')
