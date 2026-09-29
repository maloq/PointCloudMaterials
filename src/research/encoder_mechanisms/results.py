"""Summarize completed matched runs without fitting models or selecting endpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha,write_json
from src.research.encoder_quality.metrics import paired_scores


def scores(record):
    result={}
    for scale in ('raw','calibrated'):
        result[f'{scale}_event_nll']=record['event_nll'][scale]['test']
        for horizon in ('3','6'):
            for metric in ('log_loss','brier'):
                result[f'{scale}_{metric}_{horizon}ps']=record['scores'][horizon]['test'][scale][metric]
    return result


def endpoint(folder,name,provenance):
    p=folder/'technical/evaluations'/name
    receipt=json.loads((p/'complete.json').read_text())
    path=p/'metrics.json'
    if sha(path)!=receipt['metrics_sha256']:raise ValueError(f'Changed completed metrics: {path}')
    d=json.loads(path.read_text());values=[v for k,v in d['static'].items() if '_Al_' in k]
    if len(values)!=3:raise ValueError('Require all three fixed Al static frames')
    row=dict(liquid_neighbor_nmse=float(np.mean([v['supplement']['liquid_order']['embedding_neighbor_nmse'] for v in values])),
        nonbulk_accuracy=float(np.mean([v['supplement']['nonbulk_context']['balanced_accuracy'] for v in values])),
        spatial_auc=float(np.mean([v['nonbulk_spatial']['auc'] for v in values])),
        **scores(d['predictive']['readouts']['z-mlp']))
    provenance[str(path)]=sha(path)
    predictions=folder/f'technical/models/{name}/technical/evaluation/z/mlp/predictions.npz'
    return row,predictions


def mean_rows(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}


def contrast(candidates,baselines,draws,seed,provenance):
    """Pair the same sources and seeds; bootstrap sources conditional on fitted seeds."""
    per_seed={};reference=None
    for index,(candidate,baseline) in enumerate(zip(candidates,baselines)):
        with np.load(candidate) as a, np.load(baseline) as b:
            pop={k:np.array(a[k]) for k in ('sample_id','source','event','role')}
            for key in pop:
                np.testing.assert_array_equal(pop[key],b[key])
                if reference is not None:np.testing.assert_array_equal(pop[key],reference[key])
            reference=pop
            corpus=SimpleNamespace(pop=pop,split={'test':np.flatnonzero(pop['role']=='test')})
            record={}
            for scale,key in [('raw','risks'),('calibrated','calibrated')]:
                record[scale]=paired_scores(corpus,a[key],b[key],draws,seed)
            per_seed[str(index)]=record
        provenance[str(candidate)]=sha(candidate);provenance[str(baseline)]=sha(baseline)
    aggregate={};rng=np.random.default_rng(seed)
    sources=sorted(per_seed['0']['raw']['6']['log_loss']['per_source'])
    boot=rng.integers(len(sources),size=(draws,len(sources)))
    for scale in ('raw','calibrated'):
        aggregate[scale]={}
        for horizon in ('3','6'):
            aggregate[scale][horizon]={}
            for metric in ('log_loss','brier'):
                matrix=np.array([[per_seed[str(i)][scale][horizon][metric]['per_source'][s]
                    for s in sources] for i in range(len(candidates))])
                source_mean=matrix.mean(0)
                aggregate[scale][horizon][metric]=dict(delta=float(source_mean.mean()),
                    ci95=np.quantile(source_mean[boot].mean(1),[.025,.975]).tolist(),
                    seed_deltas=matrix.mean(1).tolist(),sources=len(sources),seeds=len(candidates),draws=draws)
    return dict(mean_over_fitted_seeds=aggregate,per_seed=per_seed,
        interpretation='Average loss contrasts, not prediction ensembling; source bootstrap conditions on these three fitted seeds')


def plots(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=output/'plots';out.mkdir(parents=True,exist_ok=True)
    colors={'R0':'#d55e00','R1':'#0072b2','R2':'#009e73'}
    labels={'R0':'Epi without alignment','R1':'Epi with alignment','R2':'VICReg'}
    panels=[('liquid_neighbor_nmse','Liquid-neighbor error ↓'),('spatial_auc','Nonbulk spatial AUC ↑'),
        ('calibrated_log_loss_6ps','Calibrated 6-ps log loss ↓')]
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(metric,title) in zip(axes,panels):
        for arm in colors:
            epochs=[int(x) for x in result['alignment'][arm]]
            records=list(result['alignment'][arm].values())
            for seed in result['seeds']:
                ax.plot(epochs,[r['per_seed'][str(seed)][metric] for r in records],color=colors[arm],alpha=.25,lw=1)
            ax.plot(epochs,[r['mean'][metric] for r in records],'-o',color=colors[arm],label=labels[arm],lw=2,ms=4)
        ax.set(title=title,xlabel='Training epoch',xticks=epochs);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed training trajectories: three seeds, means bold; individual seeds thin\n'
        'Structure: exploratory relaxed Al frames. Prediction: held-out observed Al, fixed frozen readout.',fontsize=10)
    for extension in ('png','pdf'):fig.savefig(out/f'training-trajectories.{extension}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    panels=[('joint','calibrated_log_loss_6ps','Joint head: calibrated 6-ps loss ↓'),
        ('fresh','calibrated_log_loss_6ps','Fresh readout: calibrated 6-ps loss ↓'),
        ('fresh','spatial_auc','Nonbulk spatial AUC ↑')]
    modes=['frozen','finetune','scratch']
    for ax,(head,metric,title) in zip(axes,panels):
        for index,seed in enumerate(result['seeds']):
            ax.plot(range(3),[result['adaptation']['best'][mode]['per_seed'][str(seed)][head][metric]
                for mode in modes],'-o',alpha=.6,label=str(seed),ms=4)
        ax.set(title=title,xticks=range(3),xticklabels=['Frozen Epi','Fine-tuned Epi','Scratch']);ax.grid(alpha=.2)
    axes[0].legend(title='Training seed',fontsize=8)
    fig.suptitle('Matched adaptation: selection-source hazard NLL, minimum 12 epochs; all fits ran 24 epochs\n'
        'The fresh readout is refitted identically on each exported encoder; test outcomes never select checkpoints.',fontsize=10)
    for extension in ('png','pdf'):fig.savefig(out/f'adaptation.{extension}',dpi=170)
    plt.close(fig)


def run(config,output):
    c=json.loads(Path(config).read_text());root=resolve_path(c['output']);output=resolve_path(output)
    if output.exists():raise FileExistsError(output)
    roots=[root,*reversed([resolve_path(p) for p in c['recovery_results']])]
    provenance={str(Path(config).resolve()):sha(config)};seeds=c['seeds'];paths={}
    result=dict(seeds=seeds,alignment={},adaptation={},contrasts={})
    for arm in c['arms']:
        name=arm['name'];result['alignment'][name]={}
        for epoch in c['pretraining']['checkpoint_epochs']:
            values={}
            for seed in seeds:
                relative=Path(f'analyses/alignment/{seed}/{name}/epoch-{epoch:03d}')
                model=f'{name}-epoch{epoch:03d}'
                folder=next(r/relative for r in roots if (r/relative/f'technical/evaluations/{model}/complete.json').exists())
                values[str(seed)],paths[(name,epoch,seed)]=endpoint(folder,model,provenance)
            result['alignment'][name][str(epoch)]=dict(per_seed=values,mean=mean_rows(list(values.values())))
    for checkpoint in ('best','epoch-024'):
        result['adaptation'][checkpoint]={}
        for mode in ('frozen','finetune','scratch'):
            values={}
            for seed in seeds:
                folder=root/f'analyses/adaptation/{seed}/{mode}/{checkpoint}'
                fresh,path=endpoint(folder,f'{mode}-{checkpoint}',provenance)
                paths[(checkpoint,mode,'fresh',seed)]=path
                path=folder/'analyses/joint-head/technical/metrics.json';provenance[str(path)]=sha(path)
                joint=scores(json.loads(path.read_text()))
                paths[(checkpoint,mode,'joint',seed)]=folder/'analyses/joint-head/predictions.npz'
                values[str(seed)]=dict(fresh=fresh,joint=joint)
            result['adaptation'][checkpoint][mode]=dict(per_seed=values,
                mean={h:mean_rows([r[h] for r in values.values()]) for h in ('fresh','joint')})
    for a,b in [('R1','R0'),('R1','R2')]:
        result['contrasts'][f'epoch24-{a}-minus-{b}']=contrast([paths[a,24,s] for s in seeds],
            [paths[b,24,s] for s in seeds],1000,c['evaluation_seed'],provenance)
    for checkpoint in ('best','epoch-024'):
        for mode in ('finetune','scratch'):
            for head in ('fresh','joint'):
                result['contrasts'][f'{checkpoint}-{mode}-minus-frozen-{head}']=contrast(
                    [paths[checkpoint,mode,head,s] for s in seeds],[paths[checkpoint,'frozen',head,s] for s in seeds],
                    1000,c['evaluation_seed'],provenance)
    write_json(output/'technical/results.json',result)
    write_json(output/'technical/provenance.json',dict(inputs=provenance,implementation={str(Path(__file__)):sha(__file__)},
        aggregation='Equal fitted-seed means; Al static summaries first average the three fixed frames',
        selection='Fixed SSL endpoints; adaptation validation hazard NLL after epoch12 plus fixed epoch24',
        limits='Source bootstrap conditional on fitted encoders/heads; no training-seed population confidence interval or multiplicity correction'))
    write_metric_table(result,output,family='encoder_mechanisms',name='completed-results')
    plots(result,output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(run(args.config,args.output))
