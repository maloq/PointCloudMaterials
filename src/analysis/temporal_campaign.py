"""Frozen-feature structural and dynamical assays after temporal model selection."""
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from src.data_utils.temporal_campaign import ROOT,TemporalPairs,write_json
from src.training_methods.temporal_campaign import Learner,Representation,batch_size,to_gpu
from src.analysis.predictive_structure import linear_forecast,neighbor_errors,cluster_interval,md_table,FAMILIES


def geometry_probes(features,base):
    meta=dict(np.load(base/'metadata.npz'));saved=np.load(base/'evaluation/split_and_targets.npz');rows=saved['rows']
    x=features[rows].astype(np.float64);split=meta['split'][rows];train=split=='train';val=split=='val';test=split=='test'
    mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-3);x=(x-mean)/std
    result={}
    for family,path,dim in [('tda',base/'embeddings/TDA_16.npy',16),('soap',base/'embeddings/SOAP.npy',64),('order',base/'order.npy',8)]:
        y=np.load(path)[rows,:dim].astype(np.float64);mean=y[train].mean(0);std=y[train].std(0);std=np.maximum(std,.001*np.median(std));y=(y-mean)/std
        best=(float('inf'),None,None)
        for alpha in (.1,10.,1000.,100000.):
            estimator=Ridge(alpha=alpha,solver='cholesky').fit(x[train],y[train]);error=float(np.square(estimator.predict(x[val])-y[val]).mean())
            if error<best[0]:best=(error,alpha,estimator)
        error=float(np.square(best[2].predict(x[test])-y[test]).mean());baseline=float(np.square(y[test]).mean())
        result[family]=dict(skill_pct=100*(1-error/baseline),test_mse=error,validation_mse=best[0],alpha=best[1])
    return result


@torch.no_grad()
def controls(model,teacher,data,cfg,hyp):
    rng=np.random.default_rng(2026090661);whole=data.batch('val',192,rng,points=model.representation.points)
    x,_,material,_,_=whole
    q,_=np.linalg.qr(np.array([[.2,.3,.7],[.5,-.2,.1],[.4,.8,.3]]));q[:,0]*=np.linalg.det(q)
    cases={'original':x,'rotation':(x@q).astype(np.float32)}
    for sigma in (.0001,.02):
        noise=rng.normal(size=x.shape).astype(np.float32)*sigma;noise[:,0]=0;cases[f'noise_{sigma:g}A']=x+noise
    zs={}
    for case,value in cases.items():
        zs[case]=np.concatenate([model.representation(torch.tensor(value[s:s+batch_size(hyp,cfg)],device='cuda'),torch.tensor(material[s:s+batch_size(hyp,cfg)],device='cuda')).cpu().numpy() for s in range(0,len(x),batch_size(hyp,cfg))])
    report=[]
    for m,label in enumerate(('Al','Mg','Ta')):
        mask=material==m;scale=np.sqrt(zs['original'][mask].var(0).sum())
        if scale<=0:raise FloatingPointError(f'Collapsed {hyp["name"]}/{label} representation in robustness assay')
        for case in cases:
            if case=='original':continue
            change=np.linalg.norm(zs[case][mask]-zs['original'][mask],axis=1)/scale
            report.append(dict(material=label,case=case,median=float(np.median(change)),p95=float(np.quantile(change,.95))))
    if hyp['model']=='MACE' and max(r['p95'] for r in report if r['case']=='rotation')>.001:
        raise RuntimeError(f'MACE failed final rotation control: {report}')
    # Predict genuinely held-out future embeddings, with a frozen EMA target.
    rng=np.random.default_rng(2026090662);errors=[];persistence=[];shuffled=[];target_features=[]
    n=3072;bs=batch_size(hyp,cfg)
    for start in range(0,n,128):
        whole=data.batch('test',min(128,n-start),rng,points=model.representation.points)
        for s in range(0,len(whole[0]),bs):
            x,y,m,c,_=to_gpu([v[s:s+bs] for v in whole]);z=model.representation(x,m);future=teacher(y,m);current=teacher(x,m)
            forecast=current if hyp['mode'] in ('static','temporal') else z+model.forecast(torch.cat((z,c),1))*c[:,1:2]
            errors.extend((forecast-future).square().mean(1).cpu().tolist());persistence.extend((current-future).square().mean(1).cpu().tolist())
            permutation=torch.arange(len(c),device='cuda')
            for values in c[:,:2].unique(dim=0):
                ids=((c[:,:2]==values).all(1)).nonzero()[:,0];permutation[ids]=ids.roll(1)
            shuffled.extend((forecast-future[permutation]).square().mean(1).cpu().tolist());target_features.append(future.cpu().numpy())
    e=np.mean(errors);p=np.mean(persistence);s=np.mean(shuffled);features=np.concatenate(target_features)
    eig=np.maximum(np.linalg.eigvalsh(np.cov(features.astype(np.float64),rowvar=False)),0);prob=eig/eig.sum();prob=prob[prob>0]
    latent=dict(forecast_rule='persistence' if hyp['mode'] in ('static','temporal') else 'trained_time_conditioned_predictor',
        skill_over_persistence_pct=float(100*(1-e/p)),skill_over_shuffled_pairs_pct=float(100*(1-e/s)),
        mse=float(e),persistence_mse=float(p),teacher_effective_rank=float(np.exp(-(prob*np.log(prob)).sum())),
        note='Frozen EMA target; Al test only, 0.3/1.2/6/12 ps. Low target rank can make latent forecasting deceptively easy.')
    return report,latent


@torch.no_grad()
def evaluate(cfg,out,selected,deadline):
    base=ROOT/cfg['benchmark'];cache=Path(cfg['cache']);data=TemporalPairs(cfg);meta=dict(np.load(base/'metadata.npz'))
    clouds=np.load(cache/'benchmark_clouds.npy',mmap_mode='r');saved=np.load(base/'evaluation/split_and_targets.npz')
    sources=saved['source'][saved['split']=='test'];groups=np.unique(sources);rng=np.random.default_rng(2026090663)
    draws=rng.integers(0,len(groups),(1000,len(groups)));baseline=dict(np.load(base/'evaluation/CoarseBOO_temperature_test_errors.npz'))
    for name in ('evaluation','embeddings','checkpoints'):(out/name).mkdir(exist_ok=False)
    errors={};retrieval={};individual=[];perturbations=[];latent_rows=[]
    for trial in selected:
        if time.time()>deadline-30:raise TimeoutError('Analysis reached campaign deadline; completed artifacts are retained')
        hyp=trial['hypothesis'];key=f"{trial['stage']}_{trial['name']}_seed{trial['seed']}"
        checkpoint=torch.load(trial['checkpoint'],map_location='cuda',weights_only=False)
        model=Learner(hyp,cfg).cuda().eval();model.load_state_dict(checkpoint['model'])
        teacher=Representation(hyp,cfg).cuda().eval();teacher.load_state_dict(checkpoint['teacher'])
        values=[];bs=batch_size(hyp,cfg)
        for start in range(0,len(clouds),bs):
            x=torch.tensor(np.array(clouds[start:start+bs,:model.representation.points]),device='cuda',dtype=torch.float32)
            material=torch.zeros(len(x),device='cuda',dtype=torch.long)
            values.append(model.representation(x,material).cpu().numpy())
        z=np.concatenate(values)
        if not np.isfinite(z).all():raise FloatingPointError(f'Nonfinite benchmark features: {key}')
        np.save(out/'embeddings'/f'{key}.npy',z)
        errors[key],choices=linear_forecast(z,SimpleNamespace(meta=meta),base);retrieval[key]=neighbor_errors(z,base)
        np.savez(out/'evaluation'/f'{key}_errors.npz',**errors[key],neighbors=retrieval[key]);write_json(out/'evaluation'/f'{key}_readout.json',choices)
        write_json(out/'evaluation'/f'{key}_geometry.json',geometry_probes(z,base))
        robustness,latent=controls(model,teacher,data,cfg,hyp)
        write_json(out/'evaluation'/f'{key}_controls.json',dict(perturbations=robustness,latent=latent))
        perturbations.extend(dict(name=trial['name'],seed=trial['seed'],**r) for r in robustness)
        latent_rows.append(dict(name=trial['name'],seed=trial['seed'],**latent))
        individual.append(dict(name=trial['name'],stage=trial['stage'],seed=trial['seed'],**{f:float(errors[key][f].mean()) for f in FAMILIES}))
        torch.save(dict(model=model.state_dict(),teacher=teacher.state_dict(),hypothesis=hyp,config=cfg,training=trial),out/'checkpoints'/f'{key}.pt')
        write_json(out/'analysis_status.json',dict(state='running',completed=len(individual),total=len(selected),last=key))
        print('EVALUATED',key,latent,flush=True)
        del model,teacher,checkpoint;torch.cuda.empty_cache()
    rows=[];uncertainty={}
    for hyp in cfg['hypotheses']:
        trials=[t for t in selected if t['name']==hyp['name']];keys=[f"{t['stage']}_{t['name']}_seed{t['seed']}" for t in trials]
        row=dict(name=hyp['name'],model=hyp['model'],stage=trials[0]['stage'],seeds=len(trials));uncertainty[hyp['name']]={}
        for f in FAMILIES:
            e=np.mean([errors[k][f] for k in keys],axis=0);score,ci=cluster_interval(e,baseline[f],sources,draws)
            row[f+'_skill_pct']=score;uncertainty[hyp['name']][f]=dict(skill_pct=score,source_ci95=ci)
        score,ci=cluster_interval(np.mean([retrieval[k] for k in keys],axis=0),baseline['neighbors'],sources,draws)
        row['neighbor_skill_pct']=score;uncertainty[hyp['name']]['neighbors']=dict(skill_pct=score,source_ci95=ci);rows.append(row)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False);pd.DataFrame(individual).to_csv(out/'evaluation/individual.csv',index=False)
    pd.DataFrame(perturbations).to_csv(out/'evaluation/perturbations.csv',index=False);pd.DataFrame(latent_rows).to_csv(out/'evaluation/latent_forecasting.csv',index=False)
    write_json(out/'uncertainty.json',uncertainty)
    table=md_table(['Hypothesis','Stage','Seeds','Future TDA','Future order','Future mobility','Neighbor future agreement'],[
        [r['name'],r['stage'],r['seeds']]+[f"{r[f+'_skill_pct']:+.2f}%" for f in (*FAMILIES,'neighbor')] for r in rows])
    report=f'''# Twelve-hour temporal hypothesis campaign

No TDA, SOAP, bond-order or PTM targets trained these encoders. All screening,
learning-rate selection and checkpoint selection used material-balanced physical
motion validation. TDA/SOAP/order appear only in the frozen-feature assays after
selection. Their ridge probes are fitted on source-training labels, tuned on
validation labels, and tested on the original six Al test sources.

{table}

Scores reduce MSE relative to the original coarse-order + temperature ridge
baseline. Positive is better. Both that baseline's inputs and each new embedding
enter the matched probe. `screen` means one seed after two 15-minute rate trials;
`confirm` means three fresh seeds at the longer confirmation budget. These are
different compute tiers, not uniformly converged models. All stop reasons and
example counts are retained in training_results.json. Source intervals are in
uncertainty.json; six source groups remain a small independent test population.

The independent primary test is initially noncoherent Al at 12/24/48 ps.
Pretraining includes Al/Mg/Ta with equal material sampling. Ta has only one
trajectory; its validation uses disjoint IDs and later frames. Mg/Ta source
coordinates are stored as float16, limiting fine-scale temporal precision.
Increasing sampled atoms and frames does not increase the number of independent
source simulations. The previously examined Al test split is reused here.

The latent forecast assay in evaluation/latent_forecasting.csv predicts held-out
Al embeddings at 0.3/1.2/6/12 ps against a frozen EMA teacher. Persistence and
condition-matched shuffled-future controls accompany target effective rank;
small latent error alone is not evidence for a useful noncollapsed embedding.
Static and temporal-invariance controls use persistence as their forecast rule.

Additional artifacts: evaluation/perturbations.csv, per-model current TDA/SOAP/
order probe diagnostics, embeddings/, and compact selected checkpoints/.
Large caches and resumable optimizer checkpoints reside on IDS as declared in
config.json. Reports, selected weights, metrics and provenance remain in the repo.
'''
    (out/'RESULTS.md').write_text(report)
    write_json(out/'analysis_status.json',dict(state='complete',completed=len(selected),total=len(selected)))
    print(table,flush=True)


def review(cfg,out):
    """Review the fixed campaign and equal-budget screening artifacts; no fitting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs=json.loads((out/'training_results.json').read_text())
    selected=json.loads((out/'selected_runs.json').read_text())
    screening=json.loads((out/'screen_selected.json').read_text())
    audit=[]
    fig,axes=plt.subplots(4,3,figsize=(15,13),sharex=True)
    for ax,hyp in zip(axes.flat,cfg['hypotheses']):
        for run in (r for r in runs if r['name']==hyp['name']):
            history=pd.read_json(run['history'],lines=True)
            best=history.loc[history.selection_score.idxmin()]
            audit.append(dict(name=run['name'],stage=run['stage'],seed=run['seed'],initial_lr=run['learning_rate'],
                steps=run['steps'],examples_seen=run['examples_seen'],seconds=run['seconds'],
                examples_per_second=run['examples_seen']/run['seconds'],best_validation=float(best.selection_score),
                best_at_minutes=float(best.seconds)/60,final_validation=float(history.iloc[-1].selection_score),
                validation_rank_at_best=float(best.effective_rank),
                **{f'validation_{m}':float(v) for m,v in zip(('Al','Mg','Ta'),best.motion_by_material)}))
            label=f"{run['stage']} s{run['seed']} lr{run['learning_rate']:g}"
            ax.plot(history.seconds/60,history.selection_score,label=label,alpha=.85,lw=1)
        ax.set_title(hyp['name']);ax.set_ylim(.35,1.05);ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.supxlabel('Training wall time (minutes)');fig.supylabel('Balanced validation motion MSE; lower is better')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    pd.DataFrame(audit).to_csv(out/'training_audit.csv',index=False)

    base=ROOT/cfg['benchmark'];saved=np.load(base/'evaluation/split_and_targets.npz')
    sources=saved['source'][saved['split']=='test'];groups=np.unique(sources)
    draws=np.random.default_rng(2026090663).integers(0,len(groups),(1000,len(groups)))
    baseline=dict(np.load(base/'evaluation/CoarseBOO_temperature_test_errors.npz'))

    def errors(directory,trials,name,family):
        return np.mean([np.load(directory/'evaluation'/f"{r['stage']}_{name}_seed{r['seed']}_errors.npz")[family]
            for r in trials if r['name']==name],axis=0)

    # These are paired source intervals, conditional on the fixed selected seeds.
    comparisons=[]
    pairs=[('density_temporal','density_static'),('density_predictive','density_static'),
        ('density_motion','density_predictive'),('density_smooth','density_predictive'),
        ('density_predictive','density_small_data'),('mace_predictive','mace_static'),
        ('mace_wide_halo','mace_predictive'),('mace_capacity','mace_predictive')]
    for candidate,reference in pairs:
        for family in (*FAMILIES,'neighbors'):
            a=errors(out/'screen_analysis',screening,candidate,family)
            b=errors(out/'screen_analysis',screening,reference,family)
            gain,ci=cluster_interval(a,b,sources,draws)
            comparisons.append(dict(candidate=candidate,reference=reference,family=family,
                relative_mse_gain_pct=gain,ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(comparisons).to_csv(out/'equal_budget_pairs.csv',index=False)

    previous=ROOT/'output/predictive_encoder_training_20260905'
    if json.loads((previous/'config.json').read_text())['benchmark']!=cfg['benchmark']:
        raise ValueError('Prior-training comparison requires the identical benchmark and test rows')
    historical=[]
    for name,model in [('density_predictive','DensityMLP'),('mace_predictive','MACE'),('schnet_predictive','SchNet')]:
        for family in (*FAMILIES,'neighbors'):
            a=errors(out,selected,name,family)
            if family=='neighbors':
                b=np.mean([np.load(previous/'evaluation'/f'{model}_seed{s}_neighbor_errors.npy') for s in (123,456,789)],axis=0)
            else:
                b=np.mean([np.load(previous/'evaluation'/f'{model}_seed{s}_linear_errors.npz')[family] for s in (123,456,789)],axis=0)
            gain,ci=cluster_interval(a,b,sources,draws)
            historical.append(dict(name=name,family=family,relative_mse_gain_pct=gain,ci_low=ci[0],ci_high=ci[1],
                old_baseline_skill_pct=100*(1-b.mean()/baseline[family].mean()),
                new_baseline_skill_pct=100*(1-a.mean()/baseline[family].mean())))
    pd.DataFrame(historical).to_csv(out/'previous_training_pairs.csv',index=False)

    summary=[]
    latent=pd.read_csv(out/'evaluation/latent_forecasting.csv')
    robustness=pd.read_csv(out/'evaluation/perturbations.csv')
    for hyp in cfg['hypotheses']:
        name=hyp['name'];trials=[r for r in selected if r['name']==name];l=latent[latent.name==name]
        row=dict(name=name,stage=trials[0]['stage'],seeds=len(trials),latent_skill_pct=l.skill_over_persistence_pct.mean(),
            shuffled_skill_pct=l.skill_over_shuffled_pairs_pct.mean(),teacher_rank=l.teacher_effective_rank.mean())
        for case in ('rotation','noise_0.0001A','noise_0.02A'):
            row[case+'_worst_p95']=robustness[(robustness.name==name)&(robustness.case==case)].p95.max()
        for family in ('tda','soap','order'):
            row['current_'+family+'_skill_pct']=np.mean([json.loads((out/'evaluation'/f"{r['stage']}_{name}_seed{r['seed']}_geometry.json").read_text())[family]['skill_pct'] for r in trials])
        for family in FAMILIES:
            scores=[100*(1-np.load(out/'evaluation'/f"{r['stage']}_{name}_seed{r['seed']}_errors.npz")[family].mean()/baseline[family].mean()) for r in trials]
            row[family+'_seed_min']=min(scores);row[family+'_seed_max']=max(scores)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'representation_summary.csv',index=False)
    comparison=pd.read_csv(out/'comparison.csv');uncertainty=json.loads((out/'uncertainty.json').read_text())
    fig,axes=plt.subplots(1,4,figsize=(15,6),sharey=True)
    for ax,family in zip(axes,(*FAMILIES,'neighbors')):
        for i,row in enumerate(comparison.itertuples()):
            metric=uncertainty[row.name][family];score=metric['skill_pct'];lo,hi=metric['source_ci95']
            ax.errorbar(score,i,xerr=[[score-lo],[hi-score]],fmt='o' if row.stage=='confirm' else 's',
                color='#2077b4' if row.stage=='confirm' else '#777777',capsize=3)
        ax.axvline(0,color='black',lw=.7);ax.set_title(family);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('MSE reduction (%)')
    axes[0].set_yticks(range(len(comparison)),comparison.name);axes[0].invert_yaxis()
    fig.suptitle('Held-out Al future structure: six-source 95% bootstrap intervals\nBlue circles: 3-seed confirmation; gray squares: 1-seed screening')
    fig.tight_layout();fig.savefig(out/'comparison_intervals.png',dpi=160);plt.close(fig)
    write_json(out/'review_status.json',dict(state='complete',trials=len(runs),training_hours=sum(r['seconds'] for r in runs)/3600,
        examples_seen=sum(r['examples_seen'] for r in runs),note='Examples are repeated draws, not unique environments. Source intervals do not include seed or model-selection uncertainty.'))
