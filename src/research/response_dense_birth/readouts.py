"""Frozen diagnostic readouts and paired ancestry uncertainty; NLL selection only."""
from collections import defaultdict
import warnings
import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.birth_prediction.fit import scores, calibrate
from src.research.birth_prediction.analysis import measured, bootstrap_scores, source_draws
from src.research.response_birth import matched
from .data import FAMILY, ROLES, output, read, dataset

METRICS=['nll','brier','auroc','ap','within_pair_auc','within_pair_top1','recall_at_locked_fpr05','observed_fpr']


def destination(c,model,space,lead,kind):
    return output(c)/'analyses/fits-v1'/model['name']/space/f'lead-{lead:g}ps'/kind


def fit(c,rows,x,model,space,lead,kind):
    out=destination(c,model,space,lead,kind);tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    spec=dict(model=model,space=space,lead_ps=lead,readout=kind,cohort=read(output(c)/'technical/plan.json')['identity'])
    identity=digest(spec)
    if (tech/'complete.json').exists():
        d=read(tech/'complete.json')
        if d['identity']!=identity or sha(tech/'predictions.npz')!=d['predictions_sha256']:raise ValueError('Changed readout')
        return
    split={r:np.flatnonzero(rows['role']==r) for r in ROLES};tr,va,ca=[split[r] for r in ROLES[:3]]
    y,w=rows['label'],rows['weight'];recipe=c['readout'];history=[]
    if kind=='prior':
        prior=float(np.average(y[tr],weights=w[tr]));probability=np.full(len(y),prior);selected=dict(prior=prior)
    elif kind=='linear':
        ww=w[tr]/w[tr].sum();mean=ww@x[tr];scale=np.sqrt(ww@(x[tr]-mean)**2).clip(1e-4);xx=(x-mean)/scale;best=np.inf
        with threadpool_limits(c['fit_threads']),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            for strength in recipe['linear_C']:
                m=LogisticRegression(C=strength,solver='lbfgs',max_iter=3000,tol=1e-5,random_state=c['seed'])
                m.fit(xx[tr],y[tr],sample_weight=w[tr]/w[tr].mean());nll=scores(y[va],m.predict_proba(xx[va])[:,1],w[va])['nll']
                history.append(dict(C=strength,selection_nll=nll))
                if nll<best:best=nll;fitted=m;selected=dict(C=strength,selection_nll=nll)
            probability=fitted.predict_proba(xx)[:,1]
        joblib.dump(dict(model=fitted,mean=mean,scale=scale),tech/'model.joblib')
    elif kind=='catboost':
        from catboost import CatBoostClassifier, Pool
        m=CatBoostClassifier(**recipe['catboost'],loss_function='Logloss',eval_metric='Logloss',random_seed=c['seed'],thread_count=c['fit_threads'],train_dir=str(tech/'catboost'))
        def pool(ix):return Pool(x[ix],y[ix],weight=w[ix]/w[ix].mean())
        m.fit(pool(tr),eval_set=pool(va),early_stopping_rounds=recipe['patience'],use_best_model=True,verbose=False,
            save_snapshot=True,snapshot_file=str(tech/'snapshot.cbs'),snapshot_interval=60)
        probability=m.predict_proba(x)[:,1];m.save_model(str(tech/'model.cbm'))
        selected=dict(iterations=m.get_best_iteration()+1,selection_nll=scores(y[va],probability[va],w[va])['nll']);history=m.get_evals_result()
    else:raise ValueError(kind)
    theta=calibrate(y[ca],probability[ca],w[ca]);calibrated=expit(theta[0]*logit(np.clip(probability,1e-6,1-1e-6))+theta[1])
    negatives=ca[y[ca]==0];order=negatives[np.argsort(calibrated[negatives])];cum=np.cumsum(w[order])/w[order].sum()
    threshold=float(calibrated[order[min(np.searchsorted(cum,.95),len(order)-1)]])
    np.savez_compressed(tech/'predictions.npz',probability=probability,calibrated=calibrated,**{k:rows[k] for k in ['id','source','ancestor','event','pair','label','weight','role']})
    write_json(tech/'selection.json',dict(selected=selected,history=history,calibration_parameters=theta.tolist(),threshold_fpr05=threshold))
    write_json(tech/'prediction-context.json',dict(encoder_frozen=True,model=model,space=space,lead_ps=lead,
        observed_frames=1,conditions=[],relaxation=False,velocity=model.get('velocities',False),
        predictor_columns=0 if x is None else x.shape[1],selector='selection-source binary NLL',calibration='separate calibration sources'))
    metrics=[]
    for role,ix in split.items():
        for score,p in [('raw',probability),('calibrated',calibrated)]:metrics.append(dict(role=role,score=score,**scores(y[ix],p[ix],w[ix])))
    write_metric_rows(metrics,out,family=FAMILY,name='scores')
    write_json(tech/'complete.json',dict(identity=identity,spec=spec,predictions_sha256=sha(tech/'predictions.npz')))
    print('fit',model['name'],space,lead,kind,'complete',flush=True)


def analyze(c):
    all_rows=dataset(c);mask=all_rows['role']=='test';rows={k:v[mask] for k,v in all_rows.items()};y,w=rows['label'],rows['weight']
    ancestors,slot,counts=source_draws(rows['ancestor'],c['bootstrap_draws'],c['seed']);bw=counts[:,slot]*w[None];den=bw.sum(1)
    individual=[];groups=defaultdict(list);sources=[]
    for f in sorted((output(c)/'analyses/fits-v1').glob('*/*/*/*/technical/complete.json')):
        done=read(f);spec=done['spec'];tech=f.parent
        if sha(tech/'predictions.npz')!=done['predictions_sha256']:raise ValueError('Changed predictions')
        z=np.load(tech/'predictions.npz');np.testing.assert_array_equal(z['id'],all_rows['id'])
        selection=read(tech/'selection.json');sources.append(dict(path=str(tech/'predictions.npz'),sha256=done['predictions_sha256']))
        for kind in ['raw','calibrated']:
            p=z['probability' if kind=='raw' else kind][mask];alarm=z['calibrated'][mask]>selection['threshold_fpr05']
            vals=measured(y,p,w,alarm if kind=='calibrated' else None);boot,_=bootstrap_scores(y,p,w,slot,counts,alarm if kind=='calibrated' else None)
            auc,top=matched(rows,p)
            for name,value in [('within_pair_auc',auc),('within_pair_top1',top)]:vals[name]=float(np.average(value,weights=w));boot[name]=bw@value/den
            model=spec['model'];key=(model['arm'],spec['space'],spec['lead_ps'],spec['readout'],kind)
            groups[key].append((vals,boot))
            for metric in METRICS:
                if vals[metric] is None:continue
                lo,hi=np.quantile(boot[metric],[.025,.975]);individual.append(dict(model=model['name'],arm=model['arm'],space=spec['space'],lead_ps=spec['lead_ps'],readout=spec['readout'],score=kind,metric=metric,value=vals[metric],lo=float(lo),hi=float(hi)))
    summary=[];means={}
    for key,records in groups.items():
        for metric in METRICS:
            if records[0][0][metric] is None:continue
            values=np.array([r[0][metric] for r in records]);boot=np.mean([r[1][metric] for r in records],0);lo,hi=np.quantile(boot,[.025,.975]);means[key+(metric,)]=(float(values.mean()),boot)
            summary.append(dict(arm=key[0],space=key[1],lead_ps=key[2],readout=key[3],score=key[4],metric=metric,value=float(values.mean()),lo=float(lo),hi=float(hi),seed_sd=float(values.std(ddof=1)) if len(values)>1 else None,fitted_seeds=len(values),test_ancestors=len(ancestors),test_events=len(set(rows['event'])),test_rows=len(y)))
    contrasts=[]
    for space in c['spaces']:
        for lead in c['leads_ps']:
            for readout in c['readouts']:
                for left,right in [('q_response','q_values'),('qv_response','qv_values'),('qv_response','q_response'),('qv_values','q_values')]:
                    for kind in ['raw','calibrated']:
                        for metric in METRICS:
                            a=(left,space,lead,readout,kind,metric);b=(right,)+a[1:]
                            if a not in means or b not in means:continue
                            v=means[a][0]-means[b][0];bs=means[a][1]-means[b][1];lo,hi=np.quantile(bs,[.025,.975])
                            contrasts.append(dict(left=left,right=right,space=space,lead_ps=lead,readout=readout,score=kind,metric=metric,difference=v,lo=float(lo),hi=float(hi)))
    out=output(c)/'analyses/comparison-v1'
    for name,values in [('models',individual),('summary',summary),('paired-comparisons',contrasts)]:write_metric_rows(values,out,family=FAMILY,name=name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['nll','auroc','within_pair_top1']):
        for arm,space in [(a,'embedding') for a in ['q_values','q_response','qv_values','qv_response']]+[('rich','control')]:
            values=[r for r in summary if r['arm']==arm and r['space']==space and r['readout']=='catboost' and r['score']=='calibrated' and r['metric']==metric]
            values.sort(key=lambda r:r['lead_ps']);ax.plot([r['lead_ps'] for r in values],[r['value'] for r in values],marker='o',label=arm)
        ax.set(xlabel='Lead before 8-atom emergence (ps)',ylabel=metric)
    axes[0].legend(fontsize=8);fig.suptitle(f'0.10 ps labelled births: {len(set(rows["event"]))} test events, {len(ancestors)} ancestors')
    fig.savefig(plots/'birth-transfer.png',dpi=180);fig.savefig(plots/'birth-transfer.svg');plt.close(fig)
    lines=['# Dense 0.10 ps response-to-birth transfer','',
        f'{len(set(rows["event"]))} held-out events, {len(ancestors)} melt ancestors, {len(y)} matched observations.',
        'Frozen October6 response encoders; exact0.10ps v3 PTM labels; native/finer input cadences retained in coverage.',
        'Snapshots precede first8-atom emergence; cases later sustain64 atoms for1.5ps. Matched1:4 liquid controls.',
        'Subcritical crystal precursors are allowed. These retrospective probabilities are not population nucleation risk.','',
        '| Lead ps | Encoder/control | Space | CatBoost calibrated NLL | Within-set top1 |',
        '|---:|---|---|---:|---:|']
    for key in sorted(groups):
        arm,space,lead,kind,score=key
        if kind!='catboost' or score!='calibrated':continue
        lines.append(f'| {lead:g} | {arm} | {space} | {means[key+("nll",)][0]:.4f} | {means[key+("within_pair_top1",)][0]:.4f} |')
    lines+=['','Mean per-seed scores, not an ensemble. Paired95% intervals resample whole melt ancestors.',
        '[All scores](tables/summary.csv) · [Paired contrasts](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'technical/complete.json',dict(fits=len(sources),prediction_sources=sources,test_events=len(set(rows['event'])),test_ancestors=ancestors.tolist()))
    from src.experiment_runner.wandb_tracking import update_recorded_summary
    for model in read(output(c)/'technical/plan.json')['models']:
        values={f'{c.get("metric_namespace","dense_birth")}/{r["space"]}/{r["lead_ps"]}ps/{r["readout"]}/{r["score"]}/{r["metric"]}':r['value'] for r in individual if r['model']==model['name']}
        update_recorded_summary(resolve_path(c['campaign'])/'analyses'/model['name']/'technical/wandb.json',values,evaluation=c.get('tracking_evaluation','native010-lineage-birth-v1'),expected=dict(entity='teshbek',project='PointCloudMaterials'))
    write_json(output(c)/'technical/complete.json',dict(state='complete',fits=len(sources),test_events=len(set(rows['event'])),test_ancestors=len(ancestors),new_wandb_runs=0))
