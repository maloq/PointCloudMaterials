"""Matched onset scores plus dense timing, proper path scores and physical errors."""
import numpy as np
import torch
from src.research.crystallization_transfer.metrics import evaluate
from src.research.local_predictability.metrics import source_weights,threshold_at_fpr

BLOCKS={'embedding':(0,128),'physical':(128,256),'bond_order':(256,264),'crystallinity':(264,265)}


def coarse_logits(cdf,lags):
    p=np.asarray(cdf,dtype=np.float64)[:,np.asarray(lags)-1]
    if not np.isfinite(p).all() or (np.diff(p,axis=1)<-1e-6).any() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid predicted event CDF')
    before=np.c_[np.zeros(len(p)),p[:,:-1]]
    h=np.clip((p-before)/np.maximum(1-before,1e-12),1e-7,1-1e-7)
    return np.log(h)-np.log1p(-h)


def dense_brier(cdf,event):
    return ((cdf-(np.arange(128)[None]>=np.asarray(event)[:,None]))**2).mean(-1)


def path_scores(paths,target):
    """Per-window mean-path MSE and empirical marginal CRPS; no best-of-sample score."""
    mean=paths.mean(1);ordered=paths.sort(dim=1).values;n=paths.shape[1]
    ranks=torch.arange(1,n+1,device=paths.device,dtype=paths.dtype)
    crps=(paths-target[:,None]).abs().mean(1)-(ordered*(2*ranks-n-1)[None,:,None,None]).sum(1)/(n*n)
    error=(mean-target).square()
    return torch.stack([value[...,a:b].mean(-1) for value in (error,crps) for a,b in BLOCKS.values()],-1)


def summarize(corpus,test,calibration,prediction,calibration_prediction):
    cdf=prediction['cdf'];cal=calibration_prediction['cdf'];lags=corpus.plan['lags']
    result=evaluate(corpus,test,coarse_logits(cdf,lags),calibration,coarse_logits(cal,lags))
    source=corpus.source_ids[test];weights=source_weights(source);fine_event=prediction['event']
    result['dense_integrated_brier']=float(weights@dense_brier(cdf,fine_event))
    # Restricted mean T at 96 ps includes all windows, including survivors and missed alarms.
    restricted=.75*(1-np.c_[np.zeros(len(cdf)),cdf[:,:-1]]).sum(-1)
    actual_restricted=.75*np.minimum(fine_event+1,128)
    result['restricted_mean_time_mae_ps']=float(weights@np.abs(restricted-actual_restricted))
    result['fine_timing']={}
    for k,lag in enumerate(lags):
        threshold=threshold_at_fpr(corpus.events[calibration]<=k,cal[:,lag-1],corpus.source_ids[calibration],.05)
        actual=fine_event<lag;hit=actual&(cdf[:,lag-1]>=threshold)
        mass=np.diff(np.c_[np.zeros(len(cdf)),cdf[:,:lag]],axis=1)
        estimate=(mass@(.75*np.arange(1,lag+1)))/np.maximum(cdf[:,lag-1],1e-12)
        error=estimate-(fine_event+1)*.75
        result['fine_timing'][str(lag*.75)]=dict(detected_window_timing_mae_ps=float(abs(error[hit]).mean()) if hit.any() else None,
            detected_window_timing_bias_ps=float(error[hit].mean()) if hit.any() else None,
            missed_windows=int((actual&~hit).sum()),positive_windows=int(actual.sum()),
            timed_within_3ps_recall=float((hit&(abs(error)<=3)).sum()/actual.sum()) if actual.any() else None,
            all_positive_window_conditional_timing_mae_ps=float(abs(error[actual]).mean()) if actual.any() else None)
    result['path']={}
    scores=prediction['path_scores']
    for j,name in enumerate([f'{metric}_{block}' for metric in ('standardized_mse','standardized_crps') for block in BLOCKS]):
        result['path'][name]={'all_times':float(weights@scores[:,:,j].mean(-1)),
            **{f'{h}ps':float(weights@scores[:,h//3-1,j]) for h in (3,9,24,48,96)}}
    persistence=prediction['persistence_scores']
    result['physical_persistence_standardized_mse']=float(weights@persistence[:,:,1].mean(-1))
    result['embedding_persistence_standardized_mse']=float(weights@persistence[:,:,0].mean(-1))
    return result
