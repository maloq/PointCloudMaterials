"""Paired source-held-out appearance transfer and combined-pool evaluation."""
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .appearance import FAMILY, specs, inputs, reference
from .data import read
from .fit import scores
from .temporal_analysis import metric_bundle, interval


def predict_original(c,spec,x,rows,old_count):
    prepared=read(resolve_path(c['output'])/'technical/prepared.json')
    ref=next(r for r in prepared['references'] if r['spec']==spec)
    tech=Path(ref['path'])
    if any(sha(tech/f)!=h for f,h in ref['files'].items()):raise ValueError('Original-pool predictor changed')
    if spec['model']=='linear':
        model=joblib.load(tech/'model.joblib')
        p=model['model'].predict_proba((x-model['mean'])/model['scale'])[:,1]
    else:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier();model.load_model(str(tech/'model.cbm'))
        p=model.predict_proba(x,thread_count=c['analysis_threads'])[:,1]
    with np.load(tech/'predictions.npz') as saved:
        if not np.array_equal(saved['id'],rows['id'][:old_count]):raise ValueError('Original rows/order changed')
        if not np.allclose(p[:old_count],saved['probability'],rtol=1e-6,atol=1e-7):
            raise ValueError('Original predictor replay differs from retained predictions')
    return p,read(tech/'selection.json')['threshold_training_oof_fpr05']


def collect(c,progress):
    root=resolve_path(c['output'])/'analyses/comparison-v1'
    (root/'technical').mkdir(parents=True,exist_ok=True)
    records=[];comparisons=[];reliability=[];cv=[];evidence=[]
    with threadpool_limits(limits=c['analysis_threads']):
        for spec in specs(c):
            progress.update(treatment=spec)
            _,rows,x,_,_,old_count=inputs(c,spec)
            old,old_threshold=predict_original(c,spec,x,rows,old_count)
            tech=resolve_path(c['output'])/'analyses/fits-v1'/spec['domain']/spec['arm']/spec['observation']['name']/'technical'
            done=read(tech/'complete.json')
            if sha(tech/'predictions.npz')!=done['predictions_sha256']:raise ValueError('Combined predictions changed')
            with np.load(tech/'predictions.npz') as saved:
                if not np.array_equal(saved['id'],rows['id']):raise ValueError('Combined cohort identity changed')
                combined=saved['probability'].copy()
            combined_threshold=read(tech/'selection.json')['threshold_training_oof_fpr05']
            for pool,folder in [('original',reference(c,spec)/'technical'),('combined',tech)]:
                value=read(folder/'complete.json')
                for result in value['scores']:
                    if result['role']=='training_selection_cv':cv.append(dict(domain=spec['domain'],arm=spec['arm'],
                        observation=spec['observation']['name'],training_pool=pool,**{k:v for k,v in result.items() if k not in ['arm','observation']}))
            filename=f'{spec["domain"]}-{spec["arm"]}-{spec["observation"]["name"]}.npz'
            np.savez_compressed(root/'technical'/filename,original_pool_probability=old,combined_pool_probability=combined,
                                **{k:rows[k] for k in ['id','source','event','pair','role','label','weight','stratum']})
            evidence.append(dict(file=filename,sha256=sha(root/'technical'/filename),original_predictor=str(reference(c,spec)),
                                 combined_identity=done['identity']))
            masks={name:rows['stratum']==name for name in ['established','failed_strong','failed_other']}
            masks.update(failed_all=rows['stratum']!='established',combined=np.ones(len(old),bool))
            for scope,role_mask in [('train_sources',rows['role']=='train'),('merged_test',rows['role']!='train')]:
                for stratum,mask in masks.items():
                    ids=np.flatnonzero(role_mask & mask)
                    if not len(ids):continue  # Exact zero coverage remains explicit in support.csv.
                    pos=rows['label'][ids]==1;neg=~pos;w=rows['weight'][ids]
                    values={}
                    for pool,p,threshold in [('original',old,old_threshold),('combined',combined,combined_threshold)]:
                        if scope=='merged_test':
                            value,boot=metric_bundle(rows,ids,p[ids],c);values[pool]=(value,boot)
                        else:value=scores(rows['label'][ids],p[ids],w)
                        records.append(dict(domain=spec['domain'],arm=spec['arm'],observation=spec['observation']['name'],
                            training_pool=pool,scope=scope,stratum=stratum,rows=len(ids),sources=len(set(rows['source'][ids])),
                            events=len(set(zip(rows['source'][ids][pos],rows['event'][ids][pos]))),
                            positive_mean_probability=float(np.average(p[ids][pos],weights=w[pos])),
                            recall_at_training_oof_fpr05=float(np.average(p[ids][pos]>threshold,weights=w[pos])),
                            observed_fpr=float(np.average(p[ids][neg]>threshold,weights=w[neg])),threshold=threshold,**value))
                        if scope=='merged_test':
                            for k in range(10):
                                take=np.minimum((p[ids]*10).astype(int),9)==k
                                if take.any():reliability.append(dict(domain=spec['domain'],arm=spec['arm'],
                                    observation=spec['observation']['name'],training_pool=pool,stratum=stratum,bin=k,
                                    rows=int(take.sum()),predicted_mean=float(np.average(p[ids][take],weights=w[take])),
                                    positive_fraction=float(np.average(rows['label'][ids][take],weights=w[take]))))
                    if scope=='merged_test':
                        for metric in ['nll','brier','ap','auroc','matched_auc']:
                            a,aa=values['combined'];b,bb=values['original'];lo,hi=interval(aa[metric]-bb[metric])
                            comparisons.append(dict(domain=spec['domain'],arm=spec['arm'],observation=spec['observation']['name'],
                                stratum=stratum,metric=metric,difference_combined_minus_original=a[metric]-b[metric],lo=lo,hi=hi))
    for name,table in [('train-and-test',records),('paired-comparisons',comparisons),('calibration',reliability),('training-selection-cv',cv)]:
        columns=sorted({k for r in table for k in r})
        write_metric_rows(table,root,family=FAMILY,name=name,columns=columns)
    write_json(root/'technical/provenance.json',dict(config=c,predictions=evidence))
    render(root,records)
    support=read(resolve_path(c['output'])/'technical/eligibility.json')
    eligible=sum(r['histories']>0 for r in support);strong=sum(r['histories']>0 and r['strong'] for r in support)
    lines=['# Predicting established and failed crystal appearances','',
        f'Input eligibility: {eligible}/184 candidate episodes retained, including {strong}/28 strong candidates. The remaining candidates are not replaced or relabeled.',
        'The 28 are a subset of 184; failed_other means the disjoint remaining 156 before eligibility screening.',
        'All failed candidates are positive. Original models learned establishment versus liquid; applying them here measures transfer to appearance. Combined models learn either kind of appearance versus liquid.',
        'All reported test scores use original held-out sources (former selection, calibration and test merged). No calibration mapping is fitted. Training-source scores are descriptive and not held-out generalization.',
        'Inputs: original or separately full-cell-relaxed geometry; 442 descriptors; no temperature, time, velocity, species or future geometry. Current ends 0.75 ps before local appearance; history has eight observations over 5.25 ps.',
        'Selection uses training-only five-fold NLL. AP is diagnostic. Bootstrap intervals describe source uncertainty, not training-seed uncertainty. Previously inspected held-out sources make this an exploratory study.',
        'Retrospective case/control prevalence is 20%; probabilities are not calibrated natural nucleation risks.', '',
        '| Input | Readout | History | Train pool | Test population | Events | NLL | AP |',
        '|---|---|---|---|---|---:|---:|---:|']
    for r in records:
        requested=(r['training_pool']=='original' and r['stratum'] in ['failed_strong','failed_other','failed_all']) or (r['training_pool']=='combined' and r['stratum'] in ['failed_strong','combined'])
        if r['scope']=='merged_test' and requested:
            lines.append(f'| {r["domain"]} | {r["arm"]} | {r["observation"]} | {r["training_pool"]} | {r["stratum"]} | {r["events"]} | {r["nll"]:.4f} | {r["ap"]:.4f} |')
    lines += ['', '[All train/test scores and intervals](tables/train-and-test.csv) · [Paired changes](tables/paired-comparisons.csv) · [Coverage](../coverage-v1/tables/support.csv) · [Eligibility exclusions](../coverage-v1/tables/candidate-eligibility.csv)', '']
    (root/'README.md').write_text('\n'.join(lines))
    write_json(root/'technical/complete.json',dict(combined_fits=8,original_predictors=8,eligible_events=eligible,eligible_strong_events=strong))
    (resolve_path(c['output'])/'README.md').write_text('# Appearance transfer\n\n[Completed comparison](analyses/comparison-v1/README.md)\n')


def render(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (root/'plots').mkdir(parents=True,exist_ok=True)
    for metric,label in [('nll','Binary negative log likelihood (lower is better)'),('ap','Average precision (higher is better)')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for i,domain in enumerate(['original','relaxed']):
            for j,stratum in enumerate(['failed_strong','combined']):
                ax=axes[i,j]
                names=[(a,o) for a in ['rich_linear','rich_gbdt'] for o in ['current','history8']]
                for pool,offset,color in [('original',-.16,'#4774a6'),('combined',.16,'#bf723a')]:
                    subset=[next((r for r in records if r['scope']=='merged_test' and r['domain']==domain and
                        r['stratum']==stratum and r['training_pool']==pool and r['arm']==a and r['observation']==o),None) for a,o in names]
                    for k,r in enumerate(subset):
                        if r is None:continue
                        ax.bar(k+offset,r[metric],width=.3,color=color,label=pool+' training' if k==0 else None)
                        lo,hi=r[metric+'_lo'],r[metric+'_hi']
                        if lo is not None:ax.vlines(k+offset,lo,hi,color='black',linewidth=1)
                ax.set_xticks(range(len(names)),[a.replace('rich_','')+'\n'+o for a,o in names])
                ax.set_title(f'{domain} coordinates · {stratum} held-out');ax.set_ylabel(label)
                ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        fig.savefig(root/'plots'/f'{metric}-comparison.png',dpi=160);plt.close(fig)
