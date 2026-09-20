"""Prospective classification, timing, calibration and sampled-center spatial metrics."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.research.local_predictability.metrics import hazard_loss,cumulative_risk,source_weights,weighted_scores,threshold_at_fpr,stratified_bootstrap,alarm_episodes


def evaluate(corpus,indices,logits,calibration_indices,calibration_logits):
    ids=np.asarray(indices);ci=np.asarray(calibration_indices)
    source=corpus.source_ids[ids];event=corpus.events[ids];ce=corpus.events[ci]
    # Float64 decisions retain nextafter thresholds above a tied float32 score.
    risk=cumulative_risk(torch.tensor(logits)).numpy().astype(np.float64);cr=cumulative_risk(torch.tensor(calibration_logits)).numpy().astype(np.float64)
    horizons=np.array(corpus.plan['lags'])*.75;weights=source_weights(source)
    rows=[];per_source=[];spatial=[];timing=[]
    meta=np.array([corpus.rows[i][:3] for i in ids]);anchors=np.array(corpus.plan['anchors'])[meta[:,1]]
    centers=np.array([corpus.arrays[s]['atom_ids'][c] for s,_,c in meta])
    onset=np.array([corpus.arrays[s]['onset'][c] for s,_,c in meta]);delay=(onset-anchors)*.75
    for k,h in enumerate(horizons):
        actual=event<=k;threshold=threshold_at_fpr(ce<=k,cr[:,k],corpus.source_ids[ci],.05)
        score=weighted_scores(actual,risk[:,k],source,threshold);pred=risk[:,k]>=threshold
        score['auroc']=float(roc_auc_score(actual,risk[:,k],sample_weight=weights)) if actual.any() and not actual.all() else None
        tp=weights@(actual&pred);fp=weights@(~actual&pred);fn=weights@(actual&~pred);tn=weights@(~actual&~pred)
        score.update(f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None,balanced_accuracy=float(.5*(tp/(tp+fn)+tn/(tn+fp))) if (tp+fn)*(tn+fp)>0 else None)
        ece=0.
        for lo in np.linspace(0,1,11)[:-1]:
            m=(risk[:,k]>=lo)&(risk[:,k]<(lo+.1 if lo<.9 else 1.00001))
            if m.any():ece+=abs(weights[m]@(risk[m,k]-actual[m]))
        score.update(horizon_ps=float(h),ece_10bin=float(ece));rows.append(score)
        sr=[]
        for sid in np.unique(source):
            m=source==sid;s=weighted_scores(actual[m],risk[m,k],source[m],threshold)
            temp=next(x['temperature_K'] for x in corpus.plan['sources'] if x['id']==sid)
            sr.append(dict(source_id=int(sid),temperature_K=temp,horizon_ps=float(h),**s))
        bounds=stratified_bootstrap([[s['log_loss'],s['brier']] for s in sr],[s['temperature_K'] for s in sr],draws=500)
        score.update(log_loss_ci95=bounds[:,0].tolist(),brier_ci95=bounds[:,1].tolist());per_source.extend(sr)
        fractions=[];ious=[];pairs=[]
        for sid,anchor_index in np.unique(meta[:,:2],axis=0):
            m=(meta[:,0]==sid)&(meta[:,1]==anchor_index)
            fractions.append(abs(risk[m,k].mean()-actual[m].mean()))
            union=(pred[m]|actual[m]).sum()
            if union:ious.append((pred[m]&actual[m]).sum()/union)
            pos=corpus.arrays[sid]['centers'][corpus.plan['anchors'][anchor_index]//4,meta[m,2]]
            box=corpus.arrays[sid]['boxes'][corpus.plan['anchors'][anchor_index]//4]
            delta=pos[:,None]-pos[None];delta-=box*np.round(delta/box);dist=np.linalg.norm(delta,axis=-1)
            a,b=np.where(np.triu((dist>0)&(dist<=25),1))
            if len(a):pairs.extend(((risk[m,k][a]-risk[m,k][b])-(actual[m][a].astype(float)-actual[m][b])).tolist())
        spatial.append(dict(horizon_ps=float(h),sampled_center_fraction_mae=float(np.mean(fractions)),sampled_center_jaccard=float(np.mean(ious)) if ious else None,nearby_pair_difference_rmse=float(np.sqrt(np.mean(np.square(pairs)))) if pairs else None,nearby_pairs=len(pairs)))
        alarms=[];event_centers=0;detected_centers=0
        for sid,center in np.unique(np.c_[source,centers],axis=0):
            ix=np.flatnonzero((source==sid)&(centers==center));ix=ix[np.argsort(anchors[ix])]
            a=alarm_episodes(anchors[ix],risk[ix,k],threshold,int(onset[ix[0]]),int(corpus.plan['lags'][k]));alarms.append(a)
            event_centers+=int(actual[ix].any());detected_centers+=int(a['detected'])
        hit=actual&pred
        # Conditional mean event time within the forecast horizon; misses reported separately.
        prob=np.diff(np.c_[np.zeros(len(ids)),risk[:,:k+1]],axis=1)
        mid=(np.r_[0,horizons[:k]]+horizons[:k+1])/2
        predicted_time=(prob@mid)/np.maximum(risk[:,k],1e-12);error=predicted_time[hit]-delay[hit]
        exposure=len(ids)*corpus.plan['config']['origin_stride_frames']*.75/1000
        timing.append(dict(horizon_ps=float(h),true_events=int(actual.sum()),missed_windows=int((actual&~pred).sum()),
            detected_window_timing_mae_ps=float(np.abs(error).mean()) if len(error) else None,
            detected_window_timing_bias_ps=float(error.mean()) if len(error) else None,
            timed_within_3ps_recall=float(((np.abs(predicted_time-delay)<=3)&hit).sum()/actual.sum()) if actual.any() else None,
            alarm_episodes=sum(len(a['alarms']) for a in alarms),false_alarm_episodes=sum(a['false_alarms'] for a in alarms),
            observable_event_centers=event_centers,detected_event_centers=detected_centers,missed_event_centers=event_centers-detected_centers,
            event_recall=detected_centers/event_centers if event_centers else None,
            false_alarms_per_sampled_center_ns=sum(a['false_alarms'] for a in alarms)/exposure,
            mean_detected_lead_ps=float(np.mean([a['lead_ps'] for a in alarms if a['detected']])) if any(a['detected'] for a in alarms) else None))
    nll=hazard_loss(torch.tensor(logits),torch.tensor(event)).numpy()
    return dict(test_event_nll=float(weights@nll),classification=rows,timing=timing,spatial=spatial,per_source=per_source,
        bootstrap='500 temperature-stratified whole-source draws, conditional on one training seed',
        timing_grid_ps=corpus.plan['config']['origin_stride_frames']*.75,
        spatial_population='Only observed at-risk sampled centers; not full-cell phase maps or front-speed measurements')
