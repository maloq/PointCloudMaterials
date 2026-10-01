"""Fresh-parent value/response errors; paired uncertainty and explicit costs."""
import math
import numpy as np

from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .common import FAMILY,root,read,sha,write_json
from .data import load
from .train import location


def collect(c):
    data=load(c);records=data['parents'];center=data['center'].numpy();scale=data['scale'].numpy()
    response_scale=float(data['response_scale']);out=root(c)/'analyses/comparison-v1'
    ids=[i for i,r in enumerate(records) if r['role']=='test']
    scopes=[('20fs',0,128),('100fs_joint_prefix',128,256),('full',0,256)]
    streams=[('prior',-1,np.zeros((56,256)),np.zeros((56,256,2)))];costs=[]
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            tech=location(c,arm,seed)/'technical';done=read(tech/'complete.json')
            if sha(tech/'predictions.npz')!=done['prediction_sha256']:raise ValueError('Changed final predictions')
            with np.load(tech/'predictions.npz') as a:
                if not np.array_equal(a['parent'],np.arange(56)):raise ValueError('Wrong evaluated parents')
                streams.append((arm,seed,a['normalized_prediction'],a['normalized_response']))
            costs.append({k:done[k] for k in ('arm','seed','selected_epoch','acquisition_seconds','training_seconds',
                'acquisition_plus_training_seconds','final_evaluation_seconds')})
    rows=[]
    for arm,seed,pred,response in streams:
        for i in ids:
            r=records[i];y=(r['values'].numpy()-center)/scale
            h=r['responses'].numpy()/scale[None,:,None]/response_scale
            for scope,lo,hi in scopes:
                value_mse=float(np.mean((pred[i,lo:hi]-y.mean(0)[lo:hi])**2))
                response_mse=float(np.mean((response[i,lo:hi]/response_scale-h.mean(0)[lo:hi])**2))
                vn=float(y[:,lo:hi].var(0,ddof=1).mean()/len(y))
                hn=float(h[:,lo:hi].var(0,ddof=1).mean()/len(h))
                rows.append(dict(arm=arm,seed=seed,parent=i,sigma_index=r['sigma_index'],scope=scope,
                    value_mse=value_mse,value_mse_corrected=value_mse-vn,
                    value_nll=.5*(value_mse+math.log(2*math.pi)),response_mse=response_mse,
                    response_mse_corrected=response_mse-hn,
                    predicted_response_squared=float(np.mean((response[i,lo:hi]/response_scale)**2)),
                    oracle_response_squared_corrected=float(np.mean(h.mean(0)[lo:hi]**2))-hn))
    metrics=['value_mse_corrected','response_mse_corrected'];summary=[];contrasts=[]
    rng=np.random.default_rng(20261001)
    # Equal four-parent strata are retained in each configuration-bootstrap draw.
    draws=np.concatenate([rng.choice([i for i in ids if i%4==s],(c['bootstrap_draws'],4)) for s in range(4)],axis=1)
    def vector(arm,scope,metric):
        return {i:np.mean([r[metric] for r in rows if r['arm']==arm and r['scope']==scope and r['parent']==i]) for i in ids}
    for scope,_,_ in scopes:
        for metric in metrics:
            for arm in ['prior',*c['arms']]:
                v=vector(arm,scope,metric);prior=vector('prior',scope,metric)
                boot=np.array([[v[i]-prior[i] for i in draw] for draw in draws]).mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                summary.append(dict(arm=arm,scope=scope,metric=metric,mean=float(np.mean(list(v.values()))),
                    delta_vs_prior=float(np.mean([v[i]-prior[i] for i in ids])),delta_ci_low=float(lo),delta_ci_high=float(hi),
                    parents=len(ids),seeds=1 if arm=='prior' else len(c['fit_seeds'])))
            for right in ('values8','values32'):
                a,b=vector('responses8',scope,metric),vector(right,scope,metric)
                boot=np.array([[a[i]-b[i] for i in draw] for draw in draws]).mean(1)
                lo,hi=np.quantile(boot,[.025,.975])
                contrasts.append(dict(left='responses8',right=right,scope=scope,metric=metric,
                    delta=float(np.mean([a[i]-b[i] for i in ids])),ci_low=float(lo),ci_high=float(hi)))
    for name,values in [('parent-errors',rows),('summary',summary),('paired-contrasts',contrasts),('costs',costs)]:
        write_metric_rows(values,out,family=FAMILY,name=name)
    for seed in c['fit_seeds']:
        for arm in c['arms']:
            selected=[r for r in rows if r['arm']==arm and r['seed']==seed and r['scope']=='full']
            fields={f'test/{metric}':float(np.mean([r[metric] for r in selected])) for metric in metrics}
            update_recorded_summary(location(c,arm,seed)/'technical/wandb.json',fields,
                evaluation='response-training-v1',expected=c['wandb'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    labels={'values8':'Values / 8 shots','responses8':'Values + responses / 8 shots','values32':'Values / 32 shots'}
    for ax,metric,title in zip(axes,metrics,['Future-feature error','Simulator-response error'],strict=True):
        selected=[r for r in summary if r['scope']=='full' and r['metric']==metric and r['arm']!='prior']
        for j,r in enumerate(selected):
            ax.plot([r['delta_ci_low'],r['delta_ci_high']],[j,j],color='#29758a',lw=2)
            ax.scatter(r['delta_vs_prior'],j,c='#173d52',s=45)
        ax.set_yticks(range(len(selected)),[labels[r['arm']] for r in selected]);ax.axvline(0,color='gray',lw=1)
        ax.set_title(title);ax.set_xlabel('Error minus constant prior; lower is better')
    fig.suptitle('Full-cell response supervision | 16 held-out synthetic configurations\nThree training seeds; configuration-bootstrap 95% intervals')
    for ext in ('png','pdf'):fig.savefig(out/'plots'/f'response-comparison.{ext}',dpi=170)
    plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete',fits=len(costs),heldout_configurations=len(ids),
        identity=data['identity'],scope='shared-prototype 20/100fs mechanism; no independent-liquid or cost-matched superiority claim'))
