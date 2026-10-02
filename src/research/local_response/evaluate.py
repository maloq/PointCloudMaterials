"""Paired held-out-source errors, finite-shot corrections and scientific figures."""
import math
import numpy as np
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .common import FAMILY,root,read,sha,write_json
from .collection import load
from .train import location


def evaluate(c):
    data=load(c);records=data['parents'];center=data['center'].numpy();scale=data['scale'].numpy()
    rs=float(data['response_scale']);n=len(records);ids=[i for i,r in enumerate(records) if r['role']=='test']
    streams=[('prior',-1,np.zeros((n,256)),np.zeros((n,256,2)))];costs=[]
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            tech=location(c,arm,seed)/'technical';done=read(tech/'complete.json')
            if sha(tech/'predictions.npz')!=done['prediction_sha256'] or sha(tech/'best.pt')!=done['checkpoint_sha256']:
                raise ValueError('Saved checkpoint or predictions changed')
            a=np.load(tech/'predictions.npz')
            if not np.array_equal(a['parent'],np.arange(n)) or not np.array_equal(a['source'],[r['source'] for r in records]):
                raise ValueError('Model-specific evaluation rows are forbidden')
            streams.append((arm,seed,a['prediction'],a['response']))
            costs.append({k:done[k] for k in ('arm','seed','selected_epoch','training_seconds','optimizer_updates','time_control_budget_reached')})
    rows=[];scopes=[('20fs',0,128),('100fs_joint',128,256),('full',0,256)]
    for arm,seed,pred,response in streams:
        for i in ids:
            r=records[i];y=(r['values'].numpy()-center)/scale;h=r['responses'].numpy()/scale[None,:,None]/rs
            for scope,lo,hi in scopes:
                ve=float(np.mean((pred[i,lo:hi]-y.mean(0)[lo:hi])**2))
                he=float(np.mean((response[i,lo:hi]/rs-h.mean(0)[lo:hi])**2))
                rows.append(dict(arm=arm,seed=seed,parent=i,source=r['source'],scope=scope,value_nll=.5*(ve+math.log(2*math.pi)),
                    value_mse=ve,value_mse_corrected=ve-float(y[:,lo:hi].var(0,ddof=1).mean()/len(y)),
                    response_mse=he,response_mse_corrected=he-float(h[:,lo:hi].var(0,ddof=1).mean()/len(h))))
    rng=np.random.default_rng(c['seed']);draws=rng.integers(0,len(ids),size=(c['bootstrap_draws'],len(ids)))
    summary=[];contrasts=[]
    def vector(arm,scope,metric):
        return np.array([np.mean([r[metric] for r in rows if r['arm']==arm and r['scope']==scope and r['parent']==i]) for i in ids])
    for scope,_,_ in scopes:
        for metric in ('value_mse_corrected','response_mse_corrected'):
            for arm in ['prior',*c['arms']]:
                v=vector(arm,scope,metric);lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
                summary.append(dict(arm=arm,scope=scope,metric=metric,mean=float(v.mean()),ci_low=float(lo),ci_high=float(hi),sources=len(ids)))
            for other in [a for a in c['arms'] if a!='responses8']:
                d=vector('responses8',scope,metric)-vector(other,scope,metric)
                lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                contrasts.append(dict(left='responses8',right=other,scope=scope,metric=metric,delta=float(d.mean()),ci_low=float(lo),ci_high=float(hi)))
    out=root(c)/'analyses/comparison-v1'
    for name,values in [('source-errors',rows),('summary',summary),('paired-contrasts',contrasts),('costs',costs)]:
        write_metric_rows(values,out,family=FAMILY,name=name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    labels={'values8':'Values /8','responses8':'Values + responses /8','values32':'Values /32','values8_time':'Values /8, matched training time'}
    for ax,metric,title in zip(axes,('value_mse_corrected','response_mse_corrected'),('Local future-feature error','Local response error'),strict=True):
        for j,arm in enumerate(c['arms']):
            r=next(r for r in summary if r['arm']==arm and r['scope']=='full' and r['metric']==metric)
            ax.plot([r['ci_low'],r['ci_high']],[j,j],color='#29758a',lw=2);ax.scatter(r['mean'],j,color='#173d52')
        ax.set_yticks(range(len(c['arms'])),[labels[a] for a in c['arms']]);ax.set_title(title);ax.set_xlabel('Noise-corrected MSE; lower is better')
    fig.suptitle('Local80 inputs · moving MLIP environments ·20/100fs\n30 held-out source lineages; source-bootstrap95% intervals')
    for ext in ('png','pdf'):fig.savefig(out/'plots'/f'local-response-comparison.{ext}',dpi=180)
    plt.close(fig)
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            selected=[r for r in rows if r['arm']==arm and r['seed']==seed and r['scope']=='full']
            update_recorded_summary(location(c,arm,seed)/'technical/wandb.json',
                {f'test/{m}':float(np.mean([r[m] for r in selected])) for m in ('value_nll','value_mse_corrected','response_mse_corrected')},
                evaluation='local-response-v1',expected=c['wandb'])
    write_json(out/'technical/complete.json',dict(state='complete',identity=data['identity'],fits=len(costs),test_sources=len(ids),
        scope='new local MLIP response assay, not original MEAM dynamics or downstream all64 crystallization benchmark'))
