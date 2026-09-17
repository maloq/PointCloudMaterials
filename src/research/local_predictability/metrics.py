"""Source-weighted prospective hazard metrics; no event-enriched precision."""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score


def hazard_loss(logits, event_bin):
    """First event in bin 0..K-1, or right survival through K bins (event_bin=K)."""
    bins=torch.arange(logits.shape[-1],device=logits.device)[None]
    survived=bins<event_bin[:,None];event=bins==event_bin[:,None]
    return (F.softplus(logits)*survived+F.softplus(-logits)*event).sum(-1)


def cumulative_risk(logits):
    return -torch.expm1(F.logsigmoid(-logits).cumsum(-1))


def source_weights(source_ids):
    _,inverse,counts=np.unique(source_ids,return_inverse=True,return_counts=True)
    return 1/(len(counts)*counts[inverse])


def weighted_scores(actual, score, source_ids, threshold=None):
    actual=np.asarray(actual,dtype=bool);score=np.asarray(score,dtype=np.float64)
    if len(actual)==0:return dict(n=0,log_loss=None,brier=None,average_precision=None)
    weight=source_weights(source_ids);p=np.clip(score,1e-7,1-1e-7)
    losses=-(actual*np.log(p)+(1-actual)*np.log1p(-p))
    result=dict(n=len(actual),sources=len(np.unique(source_ids)),prevalence=float(weight@actual),
        log_loss=float(weight@losses),brier=float(weight@(score-actual)**2),
        average_precision=float(average_precision_score(actual,score,sample_weight=weight)) if actual.any() else None)
    if threshold is not None:
        positive=score>=threshold;tp=weight@(actual&positive);fp=weight@(~actual&positive)
        result.update(threshold=float(threshold),recall=float(tp/(weight@actual)) if actual.any() else None,
            false_positive_rate=float(fp/(weight@~actual)) if not actual.all() else None,
            precision=float(tp/(tp+fp)) if tp+fp else None)
    return result


def threshold_at_fpr(actual,score,source_ids,maximum=.05):
    negative=~np.asarray(actual,dtype=bool)
    if not negative.any():raise ValueError('Calibration has no negative windows')
    weights=source_weights(source_ids)[negative];scores=score[negative]
    order=np.argsort(-scores,kind='stable');scores=scores[order];weights=weights[order]
    total=weights.sum();cumulative=np.cumsum(weights)
    # Full ties must be included in FPR for the >= decision rule.
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    valid=ends[cumulative[ends]<=maximum*total]
    return float(scores[valid[-1]]) if len(valid) else float(np.nextafter(scores[0],np.inf))


def stratified_bootstrap(per_source,temperatures,draws=1000,seed=20260919):
    values=np.asarray(per_source,dtype=float);temperatures=np.asarray(temperatures)
    rng=np.random.default_rng(seed);selections=[]
    for temperature in np.unique(temperatures):
        group=np.flatnonzero(temperatures==temperature)
        selections.append(rng.choice(group,size=(draws,len(group)),replace=True))
    sampled=values[np.concatenate(selections,axis=1)].mean(1)
    return np.quantile(sampled,[.025,.975],axis=0)


def alarm_episodes(anchor_frames,risks,threshold,onset_frame,horizon_frames,refractory_frames=12):
    """Consecutive positives collapse; each new episode must respect refractory time."""
    previous_positive=False;last=-10**9;alarms=[]
    for anchor,risk in zip(anchor_frames,risks):
        if anchor>=onset_frame:break
        positive=risk>=threshold
        if positive and not previous_positive and anchor-last>=refractory_frames:
            alarms.append(int(anchor));last=int(anchor)
        previous_positive=bool(positive)
    correct=[a for a in alarms if 0<onset_frame-a<=horizon_frames]
    return dict(alarms=alarms,false_alarms=len(alarms)-len(correct),
        detected=bool(correct),lead_ps=(onset_frame-min(correct))*.75 if correct else None)
