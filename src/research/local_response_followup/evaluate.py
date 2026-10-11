"""All-source evaluation of the declared validation selector and fixed endpoint."""
import math
import numpy as np
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .common import FAMILY, root, read, sha, write_json, load
from .train import location


def evaluate(c):
    data=load(c);records=data['parents'];center=data['center'].numpy();scale=data['scale'].numpy()
    rs=float(data['response_scale']);ids=[i for i,r in enumerate(records) if r['role']=='test']
    arms=[a['name'] for a in c['arms']];rows=[];costs=[]
    for seed in c['fit_seeds']:
        for arm in arms:
            tech=location(c,arm,seed)/'technical';done=read(tech/'complete.json')
            if any(sha(tech/name)!=value for name,value in done['outputs'].items()):
                raise ValueError(f'Changed completed outputs: {tech}')
            costs.append({k:done[k] for k in ('arm','seed','selected_epoch','training_seconds','optimizer_updates')})
            for endpoint in ['selected','final']:
                a=np.load(tech/f'predictions-{endpoint}.npz')
                if not np.array_equal(a['parent'],[r['index'] for r in records]) or not np.array_equal(a['source'],[r['source'] for r in records]):
                    raise ValueError('Model-specific evaluation rows are forbidden')
                for i in ids:
                    r=records[i];y=(r['values'].numpy()-center)/scale
                    h=r['responses'].numpy()/scale[None,:,None]/rs
                    for scope,lo,hi in [('20fs',0,128),('100fs_joint',128,256),('full',0,256)]:
                        ve=float(np.mean((a['prediction'][i,lo:hi]-y.mean(0)[lo:hi])**2))
                        he=float(np.mean((a['response'][i,lo:hi]/rs-h.mean(0)[lo:hi])**2))
                        rows.append(dict(arm=arm,seed=seed,endpoint=endpoint,parent=i,source=r['source'],scope=scope,
                            value_nll=.5*(ve+math.log(2*math.pi)),value_mse=ve,response_mse=he,
                            value_mse_corrected=ve-float(y[:,lo:hi].var(0,ddof=1).mean()/len(y)),
                            response_mse_corrected=he-float(h[:,lo:hi].var(0,ddof=1).mean()/len(h))))
    rng=np.random.default_rng(c['seed']);draws=rng.integers(0,len(ids),size=(c['bootstrap_draws'],len(ids)))
    summary=[];contrasts=[]
    for endpoint in ['selected','final']:
        for scope in ['20fs','100fs_joint','full']:
            for metric in ['value_mse_corrected','response_mse_corrected']:
                vectors={arm:np.array([np.mean([r[metric] for r in rows if r['arm']==arm and r['endpoint']==endpoint and r['scope']==scope and r['parent']==i]) for i in ids]) for arm in arms}
                for arm,v in vectors.items():
                    lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
                    summary.append(dict(arm=arm,endpoint=endpoint,scope=scope,metric=metric,mean=float(v.mean()),
                        ci_low=float(lo),ci_high=float(hi),sources=len(ids),fit_seeds=len(c['fit_seeds'])))
                    if arm==arms[0]:continue
                    d=v-vectors[arms[0]];lo,hi=np.quantile(d[draws].mean(1),[.025,.975])
                    contrasts.append(dict(left=arm,right=arms[0],endpoint=endpoint,scope=scope,metric=metric,
                        delta=float(d.mean()),ci_low=float(lo),ci_high=float(hi)))
    out=root(c)/'analyses/comparison-v1'
    for name,values in [('source-errors',rows),('summary',summary),('paired-contrasts',contrasts),('costs',costs)]:
        write_metric_rows(values,out,family=FAMILY,name=name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    labels=['Value only','Response weight0.1','Response weight0.3','Response weight1','Warm-up + weight0.3']
    for ax,metric,title in zip(axes,['value_mse_corrected','response_mse_corrected'],['Future-feature error','Response error'],strict=True):
        for j,arm in enumerate(arms):
            for endpoint,offset,color in [('selected',-.1,'#245b80'),('final',.1,'#d28442')]:
                r=next(r for r in summary if r['arm']==arm and r['endpoint']==endpoint and r['scope']=='full' and r['metric']==metric)
                ax.plot([r['ci_low'],r['ci_high']],[j+offset,j+offset],color=color)
                ax.scatter(r['mean'],j+offset,color=color,label=endpoint if j==0 else None)
        ax.set_yticks(range(len(arms)),labels);ax.set_title(title);ax.set_xlabel('Noise-corrected MSE; lower is better');ax.legend()
    fig.suptitle('Local80 ·400 updates each ·3 initialization seeds\n30 reused held-out sources; paired source-bootstrap95% intervals')
    for ext in ['png','pdf']:fig.savefig(out/'plots'/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True,constrained_layout=True)
    for ax,seed in zip(axes,c['fit_seeds'],strict=True):
        for arm,label in zip(arms,labels,strict=True):
            import csv
            with (location(c,arm,seed)/'tables/learning.csv').open() as stream:history=list(csv.DictReader(stream))
            ax.plot([int(r['epoch']) for r in history],[float(r['selection_nll']) for r in history],label=label)
        ax.set(title=f'Seed{seed}',xlabel='Optimizer updates')
    axes[0].set_ylabel('Validation feature Gaussian NLL');axes[-1].legend(fontsize=8)
    for ext in ['png','pdf']:fig.savefig(out/'plots'/f'learning-curves.{ext}',dpi=180)
    plt.close(fig)
    for seed in c['fit_seeds']:
        for arm in arms:
            metrics={}
            for endpoint in ['selected','final']:
                values=[r for r in rows if r['arm']==arm and r['seed']==seed and r['endpoint']==endpoint and r['scope']=='full']
                metrics.update({f'test/{endpoint}/{m}':float(np.mean([r[m] for r in values])) for m in ['value_nll','value_mse_corrected','response_mse_corrected']})
            update_recorded_summary(location(c,arm,seed)/'technical/wandb.json',metrics,
                evaluation='local-response-followup-v1',expected=c['wandb'])
    write_json(out/'technical/complete.json',dict(state='complete',identity=read(root(c)/'technical/binding.json')['identity'],
        fits=len(costs),test_sources=len(ids),primary_endpoint='selected',final_endpoint='declared diagnostic, not a competing test selector',
        uncertainty='Source bootstrap conditional on the3 trained seeds; no seed-uncertainty coverage',
        scope='Short-time MACE response assay on reused cohort; not downstream crystallization'))
