"""Evaluate task-trained predictors after all validation selection is complete."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.training_methods.predictive_structure import Predictor,NEURAL,FAMILIES,FUTURE_SLICES,CURRENT_SLICES,ROOT,write_json,future_errors


def cluster_interval(error,baseline,sources,draws):
    groups=np.unique(sources)
    a=np.array([error[sources==s].sum() for s in groups]);b=np.array([baseline[sources==s].sum() for s in groups])
    scores=100*(1-a[draws].sum(1)/b[draws].sum(1))
    return float(100*(1-error.mean()/baseline.mean())),np.quantile(scores,[.025,.975]).tolist()


def linear_forecast(features,data,base):
    ids=np.load(base/'evaluation/split_and_targets.npz')['rows'];meta=data.meta
    coarse=np.load(base/'order.npy')[ids]
    temperature=meta['temperature'][ids,None]==np.array([400,450,500])[None]
    x=np.column_stack((coarse,temperature,features[ids])).astype(np.float64)
    split=meta['split'][ids];train,val,test=(split==s for s in ('train','val','test'))
    mean=x[train].mean(0);std=x[train].std(0);std=np.maximum(std,max(np.median(std)*.001,1e-10));x=(x-mean)/std
    saved=np.load(base/'evaluation/split_and_targets.npz');errors={};choices={}
    for family in FAMILIES:
        y=saved[family];predictions=[]
        for h in range(3):
            best=(float('inf'),None,None)
            for alpha in (.1,10.,1000.,100000.):
                estimator=Ridge(alpha=alpha,solver='cholesky').fit(x[train],y[train,h])
                mse=float(np.square(estimator.predict(x[val])-y[val,h]).mean())
                if mse<best[0]:best=(mse,alpha,estimator)
            predictions.append(best[2].predict(x[test]));choices[f'{family}_{h}']=dict(alpha=best[1],validation_mse=best[0])
        errors[family]=np.square(np.stack(predictions,1)-y[test]).mean(axis=(1,2))
    return errors,choices


def effective_rank(z):
    z=z.astype(np.float64)-z.mean(0)
    eig=np.maximum(np.linalg.eigvalsh(z.T@z),0);p=eig/eig.sum();p=p[p>0]
    return float(np.exp(-(p*np.log(p)).sum()))


def neighbor_errors(features,base):
    saved=np.load(base/'evaluation/split_and_targets.npz')
    z=features[saved['rows']].astype(np.float64);train=saved['split']=='train';test=saved['split']=='test'
    std=z[train].std(0);std=np.maximum(std,max(np.median(std)*.001,1e-10));z=(z-z[train].mean(0))/std
    candidates=saved['candidates']
    distances=np.square(z[candidates]-z[test,None]).mean(-1)
    chosen=np.take_along_axis(candidates,np.argsort(distances,axis=1)[:,:10],axis=1)
    return np.mean([np.square(saved[f][chosen]-saved[f][test,None]).mean(axis=(1,2,3)) for f in FAMILIES],axis=0)


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


@torch.no_grad()
def evaluate(cfg,out,data,selected,*,audit_training=True):
    # Audit training before test access. Analysis-only replay keeps the already
    # audited checkpoints and selection manifest unchanged.
    if audit_training:
        from src.training_methods import predictive_structure as training
        with torch.enable_grad():
            training.complete_plateau_audit(cfg,out,data,selected)
    base=ROOT/cfg['benchmark'];directory=out/'evaluation';directory.mkdir(exist_ok=False)
    (out/'embeddings').mkdir(exist_ok=False)
    test=data.future_indices['test'];test_np=test.cpu().numpy();sources=data.meta['source'][test_np]
    saved=np.load(base/'evaluation/split_and_targets.npz')
    np.testing.assert_array_equal(test_np,saved['rows'][saved['split']=='test'])
    all_ids=torch.arange(len(data.clouds),device='cuda')
    predictions={};linear_errors={};neighbors={};fidelity=[];robustness=[];individual=[]
    controls=dict(np.load(base/'robustness/inputs.npz'));control_ids=torch.tensor(controls['indices'],device='cuda')
    for trial in selected:
        name,seed=trial['name'],trial['seed'];key=f'{name}_seed{seed}'
        model=Predictor(name,cfg,data).cuda().eval()
        model.load_state_dict(torch.load(trial['checkpoint'],map_location='cuda',weights_only=False)['state_dict'])
        zs=[];cs=[];fs=[]
        for ids in all_ids.split(cfg['batch_size']):
            z,c,f=model(ids,data);zs.append(z.cpu().numpy());cs.append(c.cpu().numpy());fs.append(f.cpu().numpy())
        z=np.concatenate(zs);c=np.concatenate(cs);f=np.concatenate(fs)
        if not (np.isfinite(z).all() and np.isfinite(c).all() and np.isfinite(f).all()):raise FloatingPointError(f'Nonfinite final predictions for {key}')
        np.save(out/'embeddings'/f'{key}.npy',z)
        predictions[key]=np.stack([np.square(f[test_np,s]-data.future[test,s].cpu().numpy()).mean(1) for s in FUTURE_SLICES],axis=1)
        np.savez(directory/f'{key}_predictions.npz',rows=test_np,future=f[test_np],errors=predictions[key])
        if name=='CoarseBOO':
            baseline_saved=np.load(base/'evaluation/CoarseBOO_temperature_test_errors.npz')
            linear_errors[key]={family:baseline_saved[family] for family in FAMILIES}
            choice={'protocol':'Original coarse-only baseline; do not duplicate the same eight inputs and change ridge regularization'}
            neighbors[key]=baseline_saved['neighbors']
        else:
            # Fixed descriptors have not learned a representation. Their head's
            # affine BatchNorm must not alter the original probe's numerical
            # variance floor, especially for near-null TDA PCA coordinates.
            probe_features=z if name in NEURAL else data.fixed[name].cpu().numpy()
            linear_errors[key],choice=linear_forecast(probe_features,data,base)
            neighbors[key]=neighbor_errors(probe_features,base)
        np.save(directory/f'{key}_neighbor_errors.npy',neighbors[key])
        np.savez(directory/f'{key}_linear_errors.npz',**linear_errors[key]);write_json(directory/f'{key}_linear_selection.json',choice)
        individual.append(dict(model=name,seed=seed,effective_rank=effective_rank(z[test_np]),**{family:float(predictions[key][:,i].mean()) for i,family in enumerate(FAMILIES)}))
        for split in ('train','val','test'):
            for m,label in enumerate(('Al','Mg','Ta')):
                mask=(data.meta['split']==split)&(data.meta['material']==m)
                truth=data.current[torch.tensor(mask,device='cuda')].cpu().numpy()
                fidelity.append(dict(model=name,seed=seed,split=split,material=label,**{family:float(np.square(c[mask,s]-truth[:,s]).mean()) for family,s in zip(('order','tda','soap'),CURRENT_SLICES)}))
        if name in NEURAL:
            control_features={}
            for case in ('original','rotation','noise_0.02A','noise_0.0001A'):
                value=model.features(control_ids,data,torch.tensor(controls[case],device='cuda').clone()).cpu().numpy()
                control_features[case]=value
            for m,label in enumerate(('Al','Mg','Ta')):
                scale=np.sqrt(z[(data.meta['split']=='train')&(data.meta['material']==m)].var(0).sum())
                mask=controls['material']==m
                for case in ('rotation','noise_0.02A','noise_0.0001A'):
                    delta=np.linalg.norm(control_features[case][mask]-control_features['original'][mask],axis=1)/scale
                    robustness.append(dict(model=name,seed=seed,material=label,case=case,median=float(np.median(delta)),p95=float(np.quantile(delta,.95))))
            if name=='MACE':
                assert max(r['p95'] for r in robustness if r['model']==name and r['seed']==seed and r['case']=='rotation')<.001,'Trained MACE failed rotation control'
        del model;torch.cuda.empty_cache()
        print('EVALUATED',key,individual[-1],flush=True)
    rng=np.random.default_rng(20260908);groups=np.unique(sources)
    draws=rng.integers(0,len(groups),size=(cfg['bootstrap_replicates'],len(groups)))
    baseline=np.mean([predictions[f'CoarseBOO_seed{s}'] for s in cfg['seeds']],axis=0)
    baseline_linear=dict(np.load(base/'evaluation/CoarseBOO_temperature_test_errors.npz'))
    rows=[];linear_rows=[];neighbor_rows=[];uncertainty={};linear_uncertainty={}
    for name in cfg['models']:
        errors=np.stack([predictions[f'{name}_seed{s}'] for s in cfg['seeds']])
        row=dict(model=name);lr=dict(model=name);uncertainty[name]={};linear_uncertainty[name]={}
        for i,family in enumerate(FAMILIES):
            score,ci=cluster_interval(errors[:,:,i].mean(0),baseline[:,i],sources,draws)
            seed_scores=100*(1-errors[:,:,i].mean(1)/baseline[:,i].mean())
            row[family+'_skill_pct']=score;row[family+'_seed_sd_pct']=float(seed_scores.std(ddof=1))
            uncertainty[name][family]=dict(skill_pct=score,source_ci95_pct=ci,seed_sd_pct=float(seed_scores.std(ddof=1)),mse=float(errors[:,:,i].mean()))
            le=np.stack([linear_errors[f'{name}_seed{s}'][family] for s in cfg['seeds']])
            ls,lci=cluster_interval(le.mean(0),baseline_linear[family],sources,draws)
            lr[family+'_skill_pct']=ls
            linear_uncertainty[name][family]=dict(skill_pct=ls,source_ci95_pct=lci)
        nscore,nci=cluster_interval(np.mean([neighbors[f'{name}_seed{s}'] for s in cfg['seeds']],axis=0),baseline_linear['neighbors'],sources,draws)
        neighbor_rows.append(dict(model=name,skill_pct=nscore,ci_low=nci[0],ci_high=nci[1]))
        rows.append(row);linear_rows.append(lr)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(linear_rows).to_csv(out/'linear_probe_comparison.csv',index=False)
    pd.DataFrame(neighbor_rows).to_csv(out/'neighbor_comparison.csv',index=False)
    pairwise=[]
    for a in cfg['models']:
        for b in cfg['models']:
            if a==b:continue
            for i,family in enumerate(FAMILIES):
                ae=np.mean([predictions[f'{a}_seed{s}'][:,i] for s in cfg['seeds']],axis=0)
                be=np.mean([predictions[f'{b}_seed{s}'][:,i] for s in cfg['seeds']],axis=0)
                score,ci=cluster_interval(ae,be,sources,draws)
                pairwise.append(dict(model=a,reference=b,family=family,skill_pct=score,ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(pairwise).to_csv(out/'pairwise_forecast_comparison.csv',index=False)
    previous=[]
    for name in NEURAL:
        old_keys=['GeoFrame_pretrained'] if name=='GeoFrame' else [f'{name}_seed{s}' for s in cfg['seeds']]
        old=[dict(np.load(base/'evaluation'/f'{key}_test_errors.npz')) for key in old_keys]
        for family in FAMILIES:
            new_error=np.mean([linear_errors[f'{name}_seed{s}'][family] for s in cfg['seeds']],axis=0)
            old_error=np.mean([r[family] for r in old],axis=0)
            score,ci=cluster_interval(new_error,old_error,sources,draws)
            previous.append(dict(model=name,reference='earlier GeoFrame checkpoint' if name=='GeoFrame' else 'jitter VICReg',family=family,
                                 skill_pct=score,ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(previous).to_csv(out/'improvement_over_previous_training.csv',index=False)
    pd.DataFrame(individual).to_csv(directory/'individual_results.csv',index=False)
    pd.DataFrame(fidelity).to_csv(directory/'current_target_fit.csv',index=False)
    audit=out/'validation_audit';audit.mkdir(exist_ok=True)
    mace_geometry={split:{r['material']:{family:r[family] for family in ('order','tda','soap')}
        for r in fidelity if r['model']=='MACE' and r['seed']==123 and r['split']==split} for split in ('train','val')}
    write_json(audit/'geometry_by_material.json',mace_geometry)
    tda_energy=data.current[:,8:24].square().mean(1).cpu().numpy();tails=[]
    for split in ('train','val'):
        ids=np.flatnonzero((data.meta['split']==split)&(data.meta['material']==0));largest=ids[np.argsort(tda_energy[ids])[-5:][::-1]]
        tails.append(dict(split=split,n=len(ids),mean=float(tda_energy[ids].mean()),quantiles=np.quantile(tda_energy[ids],[.5,.9,.99,1]).tolist(),
            top_five=[dict(row=int(i),source=int(data.meta['source'][i]),kind=str(data.meta['kind'][i]),mse=float(tda_energy[i]),
                radius_64_A=float(data.clouds[i].norm(dim=1).sort().values[64])) for i in largest]))
    write_json(audit/'tda_target_tail.json',tails)
    target_variance=[]
    for split in ('train','val','test'):
        for m,label in enumerate(('Al','Mg','Ta')):
            ids=data.by_material[split][m]
            target_variance.append(dict(split=split,material=label,**{family:float(data.current[ids,s].square().mean()) for family,s in zip(('order','tda','soap'),CURRENT_SLICES)}))
    pd.DataFrame(target_variance).to_csv(directory/'current_training_mean_baseline.csv',index=False)
    pd.DataFrame(robustness).to_csv(directory/'perturbations.csv',index=False)
    write_json(out/'uncertainty.json',uncertainty);write_json(out/'linear_uncertainty.json',linear_uncertainty)
    report(cfg,out,selected,rows,linear_rows,uncertainty)


def report(cfg,out,selected,rows,linear_rows,uncertainty):
    main_table=md_table(['Representation','Future topology','Future bond order','Future mobility'],[
        [r['model']]+[f"{r[f+'_skill_pct']:+.2f}% ± {r[f+'_seed_sd_pct']:.2f}" for f in FAMILIES] for r in rows])
    linear_table=md_table(['Representation','Topology, linear probe','Order, linear probe','Mobility, linear probe'],[
        [r['model']]+[f"{r[f+'_skill_pct']:+.2f}%" for f in FAMILIES] for r in linear_rows])
    neighbors=pd.read_csv(out/'neighbor_comparison.csv')
    neighbor_table=md_table(['Representation','Matched-neighbor future agreement','95% source CI'],[
        [r.model,f'{r.skill_pct:+.2f}%',f'[{r.ci_low:+.2f}, {r.ci_high:+.2f}]'] for r in neighbors.itertuples()])
    trials=json.loads((out/'trials.json').read_text())
    selection_rows=[]
    for t in selected:
        history=[json.loads(line) for line in (Path(t['checkpoint']).parent/'epochs.jsonl').read_text().splitlines()]
        chosen=history[t['best_epoch']-1]
        selection_rows.append(dict(model=t['name'],seed=t['seed'],learning_rate=t['learning_rate'],best_epoch=t['best_epoch'],
            validation_current=chosen['validation_current'],selection_score=chosen['selection_score'],
            **{f'validation_{family}':value for family,value in zip(FAMILIES,chosen['validation_future'])}))
    pd.DataFrame(selection_rows).to_csv(out/'validation_selection.csv',index=False)
    training_table=md_table(['Model','Selected LR','Selected epochs, seeds 123/456/789','Stopped on plateau'],[
        [name,next(t['learning_rate'] for t in selected if t['name']==name),', '.join(str(t['best_epoch']) for t in selected if t['name']==name),
         ', '.join(str(t['converged']) for t in selected if t['name']==name)] for name in cfg['models']])
    budget_table=md_table(['Model','Trainable parameters, including heads','All trials, epochs','All trials, minutes'],[
        [name,f"{next(t['parameters'] for t in selected if t['name']==name):,}",
         sum(t['epochs'] for t in trials if t['name']==name),
         f"{sum(t['seconds'] for t in trials if t['name']==name)/60:.1f}"] for name in cfg['models']])
    ci_table=md_table(['Model','Topology 95% source CI','Order 95% source CI','Mobility 95% source CI'],[
        [name]+[f"[{uncertainty[name][f]['source_ci95_pct'][0]:+.2f}, {uncertainty[name][f]['source_ci95_pct'][1]:+.2f}]" for f in FAMILIES] for name in cfg['models']])
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for ax,name in zip(axes.ravel(),cfg['models']):
        for t in (r for r in trials if r['name']==name and r['seed']==123):
            history=[json.loads(line) for line in (Path(t['checkpoint']).parent/'epochs.jsonl').read_text().splitlines()]
            ax.plot([r['epoch'] for r in history],[r['selection_score'] for r in history],label=f"LR {t['learning_rate']:g}")
        ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Validation selection loss');ax.set_yscale('log');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out/'validation_curves.png',dpi=160);fig.savefig(out/'validation_curves.pdf');plt.close(fig)
    findings=f'''# Task-supervised training and comparison

All model selection finished before test prediction evaluation. This follow-up
reuses the previously examined source split; it is not a newly blinded benchmark.
The independent future test remains 3,054 initially noncoherent Al centers from
six sources. Current-structure training includes Al/Mg/Ta.

## End-to-end forecasting

Each learned encoder was trained from scratch with current-geometry supervision
and shooting-outcome prediction. Fixed descriptors received trained nonlinear
heads. Scores are percentage MSE reductions relative to the **trained nonlinear
coarse-bond-order + temperature baseline**. The three target families have equal
weight in training and are reported separately. Seeds are averaged at the error
level; ± is seed SD, not source uncertainty.
These nonlinear forecast heads receive the representation and temperature;
they do not additionally receive the eight coarse-order descriptors. The matched
linear probes below explicitly append those descriptors to test added predictive
information. Negative direct scores therefore do not by themselves establish
that a representation contains no information beyond coarse order.

{main_table}

{ci_table}

These scores use a stronger nonlinear baseline than the preceding VICReg table;
they must not be numerically compared with that table without matching readouts.

## Matched linear-probe comparison

Frozen features from the selected task-trained models are evaluated with the
original coarse-order-and-temperature-augmented ridge protocol. These scores use
the original linear coarse baseline, so they can be compared with the preceding
representation benchmark. The new encoders have supervised future-target
training; this is explicitly a different training regime.
Fixed descriptors use their original coordinates for these matched probes;
their prediction heads' learned affine normalization is excluded because it can
interact with the probe's variance floor on near-null TDA coordinates.

{linear_table}

The following retrieval assay checks whether the representation itself groups
environments with similar futures. It selects ten neighbors from the same 64
temperature/coarse-order-matched training candidates used in the original
benchmark. Positive values improve future agreement over coarse-order neighbors.

{neighbor_table}

## Training and convergence

{training_table}

{budget_table}

![Validation tuning curves](validation_curves.png)

Each model receives three learning-rate trials (3e-4, 1e-3, 3e-3) on seed 123,
then seeds 456 and 789 using the selected rate. Seed 123's selected trial is reused.
AdamW weight decay is 1e-4. Geometry warm-up lasts 15 epochs, followed by a five-epoch
forecast-weight ramp and joint training. Current losses balance materials and
the three measurement families (BOO, compact alpha persistence and SOAP).
Forecasting equally weights future topology, order and nonaffine/displacement
mobility. Selection uses validation forecast MSE plus 0.1 times current-structure
loss. Plateau learning-rate reductions, a minimum 80 epochs and 40-epoch stopping
patience replace the earlier fixed short schedule. The initial cap is 300 epochs.
A final audit accumulates small improvements against the last meaningful
validation improvement. Any prematurely stopped trial is continued with its
optimizer and scheduler state before test evaluation; learning-rate selection
is then repeated. Continuation sample streams are recorded explicitly.
See [convergence audit](convergence_audit.json). A True entry reports satisfaction
of this stopping criterion, not a guarantee of a globally optimal model.

MACE is the corrected reference two-interaction model, ell=2, correlation=3,
64 channels, learned radial functions, full two-hop 4 Å support and fused ir_mul
layout. SchNet-style retains its continuous-filter architecture; the density MLP
retains its smooth invariant input. GeoFrame is freshly initialized from its
architecture settings; no old weights substitute for matched training. An
explicit latent BatchNorm and matched nonlinear decoder heads provide usable
feature scaling and direct task gradients. All encoders export 128 dimensions;
fixed TDA-16 and coarse-order inputs retain their native dimensions.
Neural inputs receive 0.01 Å coordinate jitter (clipped at 0.03 Å per component);
fixed descriptors use clean coordinates. This is matched task supervision and
validation tuning, not an exhaustive architecture or compute-budget search.
The architectures also retain different neighborhoods: MACE/SchNet have two
4 Å message-passing layers, SOAP has a 6.5 Å cutoff, density features have an
8 Å cutoff, and GeoFrame uses its nearest 80 points.

## Interpretation limits and retained diagnostics

Current geometry targets are complementary measurements, not phase labels.
Their reconstruction is a training-fit diagnostic, particularly for SOAP/TDA
inputs that already contain these measurements. Future test outcomes remain
source-held-out. Mg/Ta lack independent shooting ensembles here, and the training
split contains fewer Mg/Ta inputs. Geometry losses are material balanced; future
supervision is Al only. No PTM/HCP label or energy/force training claim is used.
The test has finite-shot noise (eight futures) and only six independent test
sources. The geometry-teacher choice can favor representations similar to the
teachers; independent mobility and future assays are therefore retained.
The current Al TDA targets also have a heavy tail: a single static validation
environment contributes about 87% of their squared distance from the training
mean. This is a descriptor diagnostic, not evidence for a rare physical phase.
See the [training/validation tail audit](validation_audit/tda_target_tail.json)
and [per-material geometry audit](validation_audit/geometry_by_material.json).
All samples were retained, and this limitation of the geometric selection term
must be considered when interpreting the comparison.

Artifacts: [comparison.csv](comparison.csv), [linear probes](linear_probe_comparison.csv),
[source uncertainty](uncertainty.json), [selected runs](selected_runs.json),
[all trials](trials.json), [current target fit](evaluation/current_target_fit.csv),
[paired model comparisons](pairwise_forecast_comparison.csv),
[matched-probe improvement over earlier training](improvement_over_previous_training.csv),
[perturbation controls](evaluation/perturbations.csv), and
[experiment configuration](config.json). All output is physically in the repository.
'''
    (out/'RESULTS.md').write_text(findings)
    files=[ROOT/'src/training_methods/predictive_structure.py',ROOT/'src/analysis/predictive_structure.py',ROOT/'src/models/encoders/atomic_graph.py',ROOT/'experiments/predictive_encoder_training_20260905/config.json',out/'training_source_at_launch.py']
    inputs=[ROOT/cfg['benchmark']/p for p in ('metadata.npz','evaluation/split_and_targets.npz','density_scaling.pt','order.npy','embeddings/TDA_16.npy','embeddings/SOAP.npy')]
    write_json(out/'provenance.json',dict(files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        environment=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ('torch','mace-torch','e3nn','cuequivariance','scikit-learn','numpy')},
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
        selected_checkpoints={r['checkpoint']:hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest() for r in selected}))
    print(main_table,flush=True)
