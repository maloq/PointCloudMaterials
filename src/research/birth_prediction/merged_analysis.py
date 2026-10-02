"""Merged evaluation, paired physical-information contrasts and reliance plots."""
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import load, read
from .fit import scores
from .leakage import feature_groups, loss
from .merged import base, fit_root, treatments
from .merged_fit import FAMILY
from .temporal_analysis import interval, matched, metric_bundle, weights_boot


def collect(c,progress):
    root=resolve_path(c['output'])/'analyses/merged-comparison-v1'
    summary, errors, contrasts, provenance, reliability = [], [], [], [], []
    measured={}; specs=treatments(c)
    with threadpool_limits(limits=c['analysis_threads']):
        for domain in c['domains']:
            _,rows,_=load(base(c,domain))
            train=np.flatnonzero(rows['role']=='train'); test=np.flatnonzero(rows['role']!='train')
            for spec in specs:
                progress.update(domain=domain['name'],arm=spec['arm'],observation=spec['observation']['name'])
                path=fit_root(c,domain,spec); done=read(path/'technical/complete.json')
                source=path/'technical/predictions.npz'
                if sha(source)!=done['predictions_sha256']:raise ValueError(f'Changed predictions: {source}')
                with np.load(source) as a:
                    if not np.array_equal(a['id'],rows['id']):raise ValueError('Readouts used different rows')
                    p=a['probability'].copy()
                provenance.append(dict(path=str(source),sha256=done['predictions_sha256'],identity=done['identity']))
                for r in done['scores']:errors.append(dict(domain=domain['name'],**r))
                for label,ids in [('merged_test',test)]+[(f'former_{r}',np.flatnonzero(rows['role']==r)) for r in ['selection','calibration','test']]:
                    value,boot=metric_bundle(rows,ids,p[ids],c)
                    key=(domain['name'],spec['arm'],spec['observation']['name'],label)
                    measured[key]=(value,boot)
                    summary.append(dict(domain=domain['name'],arm=spec['arm'],observation=spec['observation']['name'],
                        scope=label,rows=len(ids),sources=len(np.unique(rows['source'][ids])),
                        latest_appearance_lead_ps=(8-spec['observation']['frames'][-1])*.75,**value))
                for k in range(10):
                    ids=test[np.minimum((p[test]*10).astype(int),9)==k]
                    if len(ids):reliability.append(dict(domain=domain['name'],arm=spec['arm'],observation=spec['observation']['name'],
                        bin=k,rows=len(ids),predicted_mean=float(np.average(p[ids],weights=rows['weight'][ids])),
                        event_fraction=float(np.average(rows['label'][ids],weights=rows['weight'][ids]))))
            prior=np.full(len(test),np.average(rows['label'][train],weights=rows['weight'][train]))
            value,boot=metric_bundle(rows,test,prior,c)
            measured[(domain['name'],'prior','prior','merged_test')]=(value,boot)
            summary.append(dict(domain=domain['name'],arm='prior',observation='prior',scope='merged_test',
                rows=len(test),sources=len(np.unique(rows['source'][test])),latest_appearance_lead_ps=None,**value))
        def compare(left,right,label):
            a,aa=measured[left];b,bb=measured[right]
            for metric in ['nll','brier','ap','auroc','matched_auc']:
                lo,hi=interval(aa[metric]-bb[metric])
                contrasts.append(dict(contrast=label,domain=left[0],arm=left[1],observation=left[2],
                    reference_domain=right[0],reference_arm=right[1],reference_observation=right[2],
                    metric=metric,difference=a[metric]-b[metric],difference_lo=lo,difference_hi=hi))
        for domain in c['domains']:
            for s in specs:
                key=(domain['name'],s['arm'],s['observation']['name'],'merged_test')
                compare(key,(domain['name'],'prior','prior','merged_test'),'versus_prior')
                if s['group']=='feature_families' and s['arm'] not in ['rich_linear','rich_gbdt']:
                    reference='rich_linear' if s['model']=='linear' else 'rich_gbdt'
                    compare(key,(domain['name'],reference,'current','merged_test'),'family_vs_full')
            for arm in c['temporal_arms']:
                for left,right in c['temporal_contrasts']:
                    a=(domain['name'],arm,left,'merged_test');b=(domain['name'],arm,right,'merged_test')
                    if a in measured and b in measured:compare(a,b,'temporal')
        for s in specs:
            compare(('relaxed',s['arm'],s['observation']['name'],'merged_test'),
                    ('original',s['arm'],s['observation']['name'],'merged_test'),'relaxed_minus_original')
    for name,records in [('heldout-scores',summary),('train-and-test',errors),('paired-comparisons',contrasts),('reliability',reliability)]:
        write_metric_rows(records,root,family=FAMILY,name=name)
    write_json(root/'technical/provenance.json',dict(config=c,predictions=provenance))
    render(root,summary,errors)
    write_json(root/'technical/complete.json',dict(fits=len(specs)*2,config=c))


def explain(c,progress):
    """Current-snapshot full descriptor reliance on the expanded held-out rows."""
    from catboost import CatBoostClassifier
    from .merged import resident
    import json
    root=resolve_path(c['output'])/'analyses/merged-permutation-v1'
    (root/'technical').mkdir(parents=True,exist_ok=True)
    results=[]
    with threadpool_limits(limits=c['analysis_threads']):
        for domain in c['domains']:
            b,rows,bank,columns=resident(json.dumps(c,sort_keys=True),domain['name'],'descriptors')
            ids=np.flatnonzero(rows['role']!='train');xx=bank[rows['indices'][ids,7]]
            groups={k:v for k,v in feature_groups(columns).items() if k in c['permutation_groups']}
            _,_,members=matched(rows,ids)
            rng=np.random.default_rng(c['seed']);donors=[]
            for _ in range(c['permutation_repeats']):
                donor=np.arange(len(ids))
                for group in members:donor[group]=rng.permutation(group)
                donors.append(donor)
            w,ww,_,_=weights_boot(rows,ids,c)
            for arm in ['rich_linear','rich_gbdt']:
                spec=next(s for s in treatments(c) if s['arm']==arm and s['observation']['name']=='current')
                tech=fit_root(c,domain,spec)/'technical'
                if spec['model']=='linear':
                    model=joblib.load(tech/'model.joblib')
                    def predict(x):return model['model'].predict_proba((x-model['mean'])/model['scale'])[:,1]
                else:
                    model=CatBoostClassifier();model.load_model(str(tech/'model.cbm'))
                    def predict(x):return model.predict_proba(x,thread_count=c['analysis_threads'])[:,1]
                p=predict(xx)
                with np.load(tech/'predictions.npz') as saved:
                    if not np.allclose(saved['probability'][ids],p,rtol=1e-6,atol=1e-7):raise ValueError('Merged full-model replay differs')
                baseline=loss(rows['label'][ids],p)
                detail=dict(id=rows['id'][ids],donors=donors)
                for name,cols in groups.items():
                    progress.update(permutation_domain=domain['name'],arm=arm,group=name)
                    delta=[]
                    for donor in donors:
                        x=xx.copy();x[:,cols]=xx[donor[:,None],cols]
                        delta.append(loss(rows['label'][ids],predict(x))-baseline)
                    detail[name]=np.asarray(delta);values=detail[name].mean(0)
                    lo,hi=interval(ww@values)
                    results.append(dict(domain=domain['name'],arm=arm,group=name,nll_increase=float(w@values),
                        difference_lo=lo,difference_hi=hi,repeats=c['permutation_repeats']))
                np.savez_compressed(root/f'technical/{domain["name"]}-{arm}.npz',**detail)
    write_metric_rows(results,root,family=FAMILY,name='family-permutation')
    write_json(root/'technical/complete.json',dict(config=c,rows=len(results)))


def render(root,summary,errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=root/'plots';plots.mkdir(parents=True,exist_ok=True)
    lines=['# Birth prediction with one merged evaluation set','',
        'Training: 965 histories. Merged evaluation: 510 histories, 102 cases, 36 births in 28 sources.',
        'Settings are chosen inside the original training sources, then refitted on all training rows.',
        'Raw probabilities only; no fitted calibration map. Source-bootstrap intervals condition on fitted models.',
        'Previously inspected sources remain exploratory. Historical rich-MACE checkpoint selection used structural validation.',
        'Training-selection CV is used for tuning and is not an unbiased nested-CV performance estimate.','',
        '| Input | Readout | Observation | Train NLL | Merged test NLL | Merged test AP |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in summary:
        if r['scope']!='merged_test':continue
        fit=next((e for e in errors if e['domain']==r['domain'] and e['arm']==r['arm'] and e['observation']==r['observation'] and e['role']=='train'),None)
        train=f'{fit["nll"]:.5f}' if fit else '0.50040'
        lines.append(f'| {r["domain"]} | {r["arm"]} | {r["observation"]} | {train} | {r["nll"]:.5f} | {r["ap"]:.3f} |')
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    currents=[r for r in summary if r['scope']=='merged_test' and r['observation']=='current']
    for metric in ['nll','ap']:
        fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
        for ax,domain in zip(axes,['original','relaxed']):
            data=[r for r in currents if r['domain']==domain]
            yy=np.arange(len(data))
            ax.hlines(yy,[r[metric+'_lo'] for r in data],[r[metric+'_hi'] for r in data],color='gray')
            ax.plot([r[metric] for r in data],yy,'o')
            ax.set(yticks=yy,yticklabels=[r['arm'] for r in data],title=domain,xlabel=metric)
            ax.axvline(.5004024235 if metric=='nll' else .2,color='gray',linestyle='--')
        fig.suptitle('Merged evaluation · current snapshot · 95% source-bootstrap intervals')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(plots/f'current-{metric}.{ext}',dpi=180)
        plt.close(fig)
    (root.parents[1]/'RESULTS.md').write_text('# Merged birth-prediction evaluation\n\n[Full report](analyses/merged-comparison-v1/README.md)\n')
