"""Known-oracle recovery, source uncertainty, paired labels and feature fidelity."""
import csv
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.fixed_cohort.protocol import write_json,sha
from .data import config
from .descriptor_data import load
from .descriptor_fit import metrics
from .control_train import table
from .comparisons import source_resamples,prediction_rows


def intervals(rows,chosen,reference,predicted,seed,draws):
    w=rows['weights'][chosen];truth=np.minimum(rows['target'][chosen],64)
    totals,bootstrap=source_resamples(rows['source'][chosen],w,(reference['mean_A'][chosen]-truth)**2,
        (predicted['mean_A'][chosen]-truth)**2,reference['nll'][chosen]-predicted['nll'][chosen],seed=seed,draws=draws)
    mass,r,p,gain=totals.T
    rg=1-np.sqrt(p[bootstrap].sum(1)/r[bootstrap].sum(1));ng=gain[bootstrap].sum(1)/mass[bootstrap].sum(1)
    return dict(rmse_gain=float(1-np.sqrt(p.sum()/r.sum())),rmse_gain_low=float(np.quantile(rg,.025)),rmse_gain_high=float(np.quantile(rg,.975)),
                nll_gain=float(gain.sum()/mass.sum()),nll_gain_low=float(np.quantile(ng,.025)),nll_gain_high=float(np.quantile(ng,.975)))


def report(c):
    root=resolve_path(c['output']);analysis=root/'analyses/controls-v1';records=[]
    for arm in c['signals']:
        name=arm['name'];dc=config(root/'technical/recipes'/f'{name}.json');_,rows,_,manifest=load(dc)
        with np.load(root/'descriptors'/name/'prior/analyses/predictability-v1/technical/predictions.npz') as a:ref={k:a[k] for k in a.files}
        if not np.array_equal(prediction_rows(ref['ids'],rows['ids']),np.arange(len(rows['ids']))):raise ValueError('Control reference row alignment failed')
        oracle=np.load(resolve_path(c['cache'])/name/'oracle.npy');edge=np.asarray(dc['distance_edges_A']);width=np.r_[np.diff(edge),1.]
        mid=np.r_[(edge[:-1]+edge[1:])/2,edge[-1]];second=np.r_[(edge[:-1]**2+edge[:-1]*edge[1:]+edge[1:]**2)/3,edge[-1]**2]
        variants=[('catboost',root/'descriptors'/name/'all_catboost_shallow/analyses/predictability-v1/technical/predictions.npz'),
                  ('mace',root/'mace'/name/'analyses/prediction-v1/technical/distance-predictions.npz')]
        for method,path in variants:
            with np.load(path) as a:pred={k:a[k] for k in a.files}
            if not np.array_equal(prediction_rows(pred['ids'],rows['ids']),np.arange(len(rows['ids']))):raise ValueError('Control report row alignment failed')
            for role in ('selection','calibration','test'):
                ix=np.flatnonzero(rows['role']==role);w=rows['weights'][ix];w/=w.sum()
                pp=np.maximum(pred['probability'][ix].astype(float),1e-12);pp/=pp.sum(1,keepdims=True)
                p=oracle[ix].astype(float);p/=p.sum(1,keepdims=True);mu=p@mid;s=p@second
                predmean=pp@mid;refmean=ref['mean_A'][ix]
                exactrisk=float(w@(s-2*predmean*mu+predmean**2));referencerisk=float(w@(s-2*refmean*mu+refmean**2))
                record=dict(signal=name,method=method,role=role,rows=len(ix),sources=len(np.unique(rows['source'][ix])),
                    declared_training_oracle_gain=arm['oracle_rmse_gain'],
                    oracle_gain_population=manifest['theoretical'][role]['oracle_rmse_gain'],
                    oracle_information_nats=manifest['theoretical'][role]['oracle_information_nats'],
                    expected_rmse_gain=1-np.sqrt(exactrisk/referencerisk),
                    expected_distance_nll=float(np.sum(w[:,None]*p*(-np.log(pp)+np.log(width)))),
                    **metrics({k:pred[k][ix] for k in ('nll','mean_A','cdf')},rows['target'][ix],w),
                    **intervals(rows,ix,ref,pred,c['seed'],c['bootstrap_draws']))
                records.append(record)
    if records:table(analysis,'synthetic-recovery',records)
    paired=[]
    for name in c['paired_protocols']:
        folder=root/'descriptors'/name/'analyses/comparison-with-baselines-v2'
        with (folder/'tables/scores.csv').open() as f:
            for row in csv.DictReader(f):
                if row['role']=='test':paired.append(dict(protocol=name,**row))
    table(analysis,'paired-boosting',paired)
    fidelity=[]
    for arm in c['mace_arms']:
        if arm['task']!='features':continue
        with (root/'mace'/arm['name']/'analyses/prediction-v1/tables/scores.csv').open() as f:
            for row in csv.DictReader(f):fidelity.append(dict(experiment=arm['name'],**row))
    table(analysis,'feature-prediction',fidelity)
    # Paired input-domain uncertainty at fixed label definition; use the fixed
    # depth-4 all-feature model, not the smallest observed test error.
    comparisons=[]
    for suffix in ('','_newlabels'):
        a='matched_raw'+suffix;b='relaxed'+suffix;dc=config(root/'technical/recipes'/f'{a}.json');_,rows,_,_=load(dc)
        predictions=[]
        for name in (a,b):
            path=root/'descriptors'/name/'all_catboost_shallow/analyses/predictability-v1/technical/predictions.npz'
            with np.load(path) as f:predictions.append({k:f[k] for k in f.files})
        for prediction in predictions:
            if not np.array_equal(prediction_rows(prediction['ids'],rows['ids']),np.arange(len(rows['ids']))):raise ValueError('Raw/relaxed prediction IDs differ')
        chosen=np.flatnonzero(rows['role']=='test')
        comparisons.append(dict(labels='relaxed' if suffix else 'original',reference=a,model=b,
                           **intervals(rows,chosen,*predictions,c['seed'],c['bootstrap_draws'])))
    table(analysis,'paired-input-comparison',comparisons)
    text=['# Liquid information sensitivity controls','',
          'One training seed and one synthetic-label realization. Source intervals do not estimate repeated-training power.',
          'Synthetic targets are diagnostics, not physical crystal distances. Oracle gains include within-bin distance noise.',
          'Relaxed labels are instantaneous PTM clusters of at least 64 atoms, not temporally established MD lineages.',
          'Raw/relaxed comparisons condition on the exact archived and jointly crystal-free cohort. Full-cell relaxation has external context.','',
          '| Signal | Method | Test RMSE gain | 95% source interval | Oracle gain |','|---|---|---:|---:|---:|']
    for r in records:
        if r['role']=='test':text.append(f'| {r["signal"]} | {r["method"]} | {100*r["rmse_gain"]:.3f}% | [{100*r["rmse_gain_low"]:.3f}, {100*r["rmse_gain_high"]:.3f}]% | {100*r["oracle_gain_population"]:.3f}% |')
    text+=['','Detailed tables: '+('[synthetic recovery](tables/synthetic-recovery.csv), ' if records else '')+'[paired boosting](tables/paired-boosting.csv),',
           '[feature fidelity](tables/feature-prediction.csv), [paired input effects](tables/paired-input-comparison.csv).']
    (analysis/'README.md').write_text('\n'.join(text)+'\n')
    write_json(analysis/'technical/complete.json',dict(config=c,files={str(p.relative_to(analysis)):sha(p) for p in (analysis/'tables').glob('*.csv')}))
