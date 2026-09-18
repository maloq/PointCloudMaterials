"""Source-held-out gains, matched contrasts, figures and research findings."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

from src.data.structural_pretraining.prepare import file_hash,save_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.trajectory_stability.metrics import stratified_draws
from .data import load,require_hardware
from .probe import feature_sets,TARGET_NAMES
from .metrics import matched_pairs,pair_mask,improvement

STRUCTURE_GROUPS={**{name:[i] for i,name in enumerate(TARGET_NAMES[:6])},
    'bond_order':list(range(6)),'angular_arrangement':list(range(6,22))}


def load_predictions(root,a,sources,task,family,method):
    dimensions=22 if task=='structure' else 3
    predictions=np.full((len(a['source']),dimensions),np.nan)
    means=np.full_like(predictions,np.nan);scales=np.full_like(predictions,np.nan)
    seen=np.zeros(len(predictions),bool)
    for source in sources:
        sid=source['id'];path=root/'technical/probes'/task/family/method/f'{sid}.npz'
        receipt=json.loads(path.with_suffix('.json').read_text())
        if file_hash(path)!=receipt['sha256']:raise ValueError(f'Changed predictions: {path}')
        if sid in receipt['training_sources']:raise ValueError('Probe was trained on its test source')
        p=np.load(path);ix=p['indices']
        if seen[ix].any() or not np.all(a['source'][ix]==sid):raise ValueError('Overlapping or incorrect held-out predictions')
        predictions[ix]=p['prediction'];means[ix]=p['target_mean'];scales[ix]=p['target_scale'];seen[ix]=True
    expected=np.ones(len(seen),bool) if task=='structure' else a['future_eligible']
    np.testing.assert_array_equal(seen,expected)
    return dict(prediction=predictions,mean=means,scale=scales)


def source_scores(a,sources,task,family,method,p,config):
    actual=np.concatenate((a['bond'],a['angular']),axis=1) if task=='structure' else a['future']
    groups=STRUCTURE_GROUPS if task=='structure' else {f'crystallize_{h}ps':[i] for i,h in enumerate(config['future_horizons_ps'])}
    rows=[]
    for source in sources:
        take=(a['source']==source['id'])
        if task=='future':take&=a['future_eligible']
        for target,columns in groups.items():
            truth=actual[take][:,columns];prediction=p['prediction'][take][:,columns]
            residual=(prediction-truth)/p['scale'][take][:,columns]
            constant=(p['mean'][take][:,columns]-truth)/p['scale'][take][:,columns]
            mse=float(np.mean(residual**2));null=float(np.mean(constant**2))
            record=dict(task=task,family=family,method=method,source=source['id'],temperature_K=source['temperature_K'],
                target=target,n=int(take.sum()),loss=mse,constant_loss=null,r2_vs_training_mean=1-mse/null)
            if task=='future':
                y=truth.ravel().astype(bool);score=prediction.ravel()
                record.update(positive=int(y.sum()),prevalence=float(y.mean()),
                    auroc=float(roc_auc_score(y,score)) if y.any() and not y.all() else None,
                    average_precision=float(average_precision_score(y,score)) if y.any() else None)
            rows.append(record)
    return rows


def pair_scores(a,sources,pairs,task,family,method,p,config):
    actual=np.concatenate((a['bond'],a['angular']),axis=1) if task=='structure' else a['future']
    groups=STRUCTURE_GROUPS if task=='structure' else {f'crystallize_{h}ps':[i] for i,h in enumerate(config['future_horizons_ps'])}
    rows=[]
    for caliper in config['match_calipers_A']:
        valid=pair_mask(pairs,caliper,config)
        if task=='future':valid&=a['future_eligible'][pairs['left']]&a['future_eligible'][pairs['right']]
        for source in sources:
            take=valid&(a['source'][pairs['left']]==source['id'])
            left,right=pairs['left'][take],pairs['right'][take]
            if not len(left):continue
            for target,columns in groups.items():
                delta=actual[left][:,columns].astype(float)-actual[right][:,columns].astype(float)
                predicted=p['prediction'][left][:,columns]-p['prediction'][right][:,columns]
                residual=(delta-predicted)/p['scale'][left][:,columns]
                record=dict(task=task,family=family,method=method,source=source['id'],temperature_K=source['temperature_K'],
                    caliper_A=caliper,target=target,n=len(left),loss=float(np.mean(residual**2)),
                    true_contrast_rms=float(np.sqrt(np.mean((delta/p['scale'][left][:,columns])**2))))
                if task=='future':
                    d=delta.ravel();pred=predicted.ravel();discordant=d!=0
                    concordance=(np.sign(d[discordant])*pred[discordant]>0).astype(float)+.5*(pred[discordant]==0)
                    record.update(discordant=int(discordant.sum()),concordance=float(concordance.mean()) if len(concordance) else None)
                rows.append(record)
    return rows


def comparisons(rows,keys,config):
    frame=pd.DataFrame(rows);summaries=[]
    for values,part in frame.groupby(keys,sort=False):
        meta=dict(zip(keys,values if isinstance(values,tuple) else (values,),strict=True))
        for baseline in ('radial','radial_control','plus_radial_duplicate','current_order','current_order_radial_duplicate'):
            if baseline not in part.method.values:continue
            for method in part.method.unique():
                if method==baseline:continue
                if baseline=='current_order' and method not in ('current_order_plus_gatr','current_order_radial_duplicate'):continue
                if baseline=='current_order_radial_duplicate' and method!='current_order_plus_gatr':continue
                if baseline=='plus_radial_duplicate' and method not in ('plus_gatr','plus_angular_delta'):continue
                if baseline=='radial' and method!='radial_control':continue
                if baseline=='radial_control' and method in ('radial','current_order_plus_gatr','current_order_radial_duplicate'):continue
                left=part[part.method==baseline].set_index('source')
                right=part[part.method==method].set_index('source')
                common=left.index.intersection(right.index)
                left,right=left.loc[common],right.loc[common]
                draws=stratified_draws(left.temperature_K.to_numpy(),config['bootstrap_draws'],config['seed'])
                result=improvement(left.loss.to_numpy(),right.loss.to_numpy(),draws)
                if (left.temperature_K.value_counts()==1).all():result.update(low=None,high=None)
                summaries.append(dict(meta,baseline=baseline,method=method,sources=len(common),n=int(right.n.sum()),**result))
    return summaries


def report(config):
    require_hardware(config)
    root=Path(config['output']);(root/'tables').mkdir(exist_ok=True);(root/'plots').mkdir(exist_ok=True)
    a,sources,parent=load(config);pairs=matched_pairs(a,config)
    np.savez(root/'technical/matched-pairs.npz',**pairs)
    source_rows=[];pair_rows=[];all_predictions={}
    for task in ('structure','future'):
        for family in ('linear','nonlinear'):
            for method in feature_sets(a,task):
                p=load_predictions(root,a,sources,task,family,method)
                all_predictions[task,family,method]=p
                source_rows.extend(source_scores(a,sources,task,family,method,p,config))
                pair_rows.extend(pair_scores(a,sources,pairs,task,family,method,p,config))
    balance=[]
    for source in sources:
        belongs=a['source'][pairs['left']]==source['id']
        for caliper in config['match_calipers_A']:
            take=belongs&pair_mask(pairs,caliper,config)
            left,right=pairs['left'][take],pairs['right'][take]
            eligible=a['future_eligible'][left]&a['future_eligible'][right]
            balance.append(dict(source=source['id'],temperature_K=source['temperature_K'],caliper_A=caliper,
                matched_pairs=int(take.sum()),candidate_pairs=int(belongs.sum()),matched_fraction=float(take.sum()/belongs.sum()),
                mean_inner_radial_rms_A=float(pairs['radial_rms_A'][take].mean()) if take.any() else None,
                mean_full_radial_rms_A=float(pairs['full_radial_rms_A'][take].mean()) if take.any() else None,
                mean_relative_density_gap=float(pairs['density_relative'][take].mean()) if take.any() else None,
                future_eligible_pairs=int(eligible.sum()),
                q6_difference_rms=float(np.sqrt(np.mean((a['bond'][left,1]-a['bond'][right,1])**2))) if take.any() else None))
    tables=dict(source_scores=source_rows,matched_scores=pair_rows,matching_balance=balance,
        conditional_gains=comparisons(source_rows,['task','family','target'],config),
        matched_gains=comparisons(pair_rows,['task','family','caliper_A','target'],config))
    data_summary=[]
    for source in sources:
        mask=a['source']==source['id'];first=mask&(a['frame']==0);risk=mask&a['future_eligible']
        data_summary.append(dict(source=source['id'],temperature_K=source['temperature_K'],observations=int(mask.sum()),
            tracked_atoms=int(first.sum()),sustained_crystallizing_atoms=int(np.sum(a['onset_frame'][first]<801)),
            eligible_future_rows=int(risk.sum()),positive_24ps=int(a['future'][risk,0].sum()),
            positive_48ps=int(a['future'][risk,1].sum()),positive_96ps=int(a['future'][risk,2].sum())))
    tables['cohort']=data_summary
    snapshot_metric_docs(root,'gatr_conditional_information')
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(root/'tables'/f'{name}.csv',index=False)
    from .plots import plots,findings
    plots(root,a,pairs,all_predictions,tables,config)
    findings(root,tables,config)
    save_json(root/'technical/report-summary.json',dict(observations=len(a['source']),sources=len(sources),
        eligible_future_rows=int(a['future_eligible'].sum()),radial_control_dimensions=a['radial'].shape[1],
        raw_embedding_dimensions=a['gatr'].shape[1],tests_log='tests.log'))
    print(f'Completed conditional report: {root.resolve()}',flush=True)
