"""Frozen spatial/motion prediction of first observed local crystal appearance."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, allocation_deadline, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from .appearance_data import FAMILY, root, cache, prepare, collect, load
from .birth_audit import read, kinematics, groups

MODULE='src.research.structural_response.appearance'


def descriptor_packet(args):
    c,start,stop=args
    from src.research.liquid_predictability.descriptors import patch_descriptors
    x=np.load(cache(c)/'states.npy',mmap_mode='r');values=[];names=None
    for state in x[start:stop]:
        v,n=patch_descriptors(state[:,:3]);k,kn=kinematics(state)
        values.append(np.r_[v,k]);names=n+kn
    return start,np.asarray(values,np.float32),names


def physical(c):
    x,_,done=load(c);path=cache(c)/'physical.npy';receipt=cache(c)/'physical.json'
    if receipt.exists():
        saved=read(receipt)
        if saved['cohort']!=done['identity'] or sha(path)!=saved['sha256']:
            raise ValueError('Changed appearance physical features')
        return
    features=np.empty((len(x),451),np.float32);names=None
    tasks=[(c,i,min(i+128,len(x))) for i in range(0,len(x),128)]
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(descriptor_packet,t) for t in tasks]):
            start,v,n=future.result()
            if names is not None and names!=n:raise ValueError('Descriptor schema changed')
            names=n;features[start:start+len(v)]=v
    np.save(path,features);write_json(cache(c)/'columns.json',names)
    write_json(receipt,dict(cohort=done['identity'],sha256=sha(path),columns_sha256=sha(cache(c)/'columns.json')))


def models(c):
    parent=resolve_path(c['parent']);recipe=read(parent/'technical/code/config.json')
    result=[]
    for method in recipe['methods']:
        spatial=resolve_path(recipe['spatial_initialization']['output'])/'analyses'/('spatial-'+method)/'technical/best.pt'
        result.append(dict(name=method+'-spatial_start',method=method,arm=None,checkpoint=str(spatial),
            checkpoint_sha256=recipe['spatial_initialization']['checkpoints'][method]))
        for arm in recipe['response_arms']:
            name=method+'-'+arm['name'];folder=parent/'analyses'/name/'technical'
            result.append(dict(name=name,method=method,arm=arm,checkpoint=str(folder/'best.pt'),
                checkpoint_sha256=read(folder/'complete.json')['checkpoint_sha256']))
    return recipe,result


def encode(c):
    import torch
    from .fast import FastMotion
    from src.research.equivariant_context.cache import RetainedCache
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    x,_,done=load(c);recipe,specs=models(c)
    # The model itself must be the original frozen numerical implementation.
    for name in ('src/research/structural_response/fast.py','src/research/structural_response/model.py',
                 'src/research/continuous_motion/model.py','src/models/encoders/spatial_mace.py'):
        if sha(Path(__file__).resolve().parents[3]/name)!=sha(resolve_path(c['parent'])/'technical/code'/name):
            raise ValueError('Changed frozen encoder dependency: '+name)
    pool=RetainedCache(resolve_path(recipe['feature_cache']),6)
    for spec in specs:
        dest=root(c)/'technical/neural'/spec['name'];dest.mkdir(parents=True,exist_ok=True)
        signature=dict(cohort=done['identity'],model=spec,producer=sha(Path(__file__)))
        if (dest/'complete.json').exists():
            if read(dest/'complete.json')['identity']!=digest(signature):raise ValueError('Changed neural appearance features')
            continue
        if sha(Path(spec['checkpoint']))!=spec['checkpoint_sha256']:raise ValueError('Selected encoder checkpoint changed')
        saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        model=FastMotion(recipe).cuda()
        if spec['arm'] is None:
            model.geometry.load_state_dict(saved['geometry'])
        else:
            model.set_arm(spec['arm']);model.load_state_dict(saved['model'])
        model.eval()
        with pool.lease(digest(signature),deadline=allocation_deadline(reserve_seconds=120),metadata=signature) as bank:
            path=bank/'features.npy'
            if not path.exists():
                z=np.empty((len(x),256),np.float32)
                with torch.no_grad():
                    for start in range(0,len(x),c['microbatch']):
                        state=torch.from_numpy(np.array(x[start:start+c['microbatch']],copy=True)).cuda()
                        z[start:start+len(state)]=model.encode(state,create_graph=False).cpu().numpy()
                        if start%1024==0:
                            write_json(dest/'progress.json',dict(completed=start,total=len(x),updated_unix=time.time()))
                if not np.isfinite(z).all():raise FloatingPointError('Nonfinite appearance encoding')
                np.save(path,z)
            # Fit while the disposable code cache is leased; retain predictions.
            evaluate_features(c,spec['name'],np.load(path,mmap_mode='r'),
                ['structural','structural_motion','motion_only'],spec)
        write_json(dest/'complete.json',dict(identity=digest(signature),model=spec))
        print('appearance model complete',spec['name'],flush=True)
        del model;torch.cuda.empty_cache()


def evaluate_features(c,name,x,spaces,metadata):
    from src.research.response_dense_birth.pooled import fit
    from src.research.birth_prediction.fit import scores
    from src.research.birth_prediction.analysis import source_draws,bootstrap_scores
    _,rows,done=load(c);plan=read(root(c)/'technical/readout-plan.json')
    held=rows['role']!='train';weights=rows['weight'][held]
    _,slot,draws=source_draws(rows['ancestor'][held],c['bootstrap_draws'],c['seed'])
    for target,key,horizons in [('any','first_step',c['horizons_ps']),
                               ('persistent','persistent_first_step',c['persistence_horizons_ps'])]:
        for horizon in horizons:
            labelled=dict(rows,label=((rows[key]>0)&(rows[key]<=round(horizon/.1))).astype(np.int64))
            support={part:np.unique(labelled['label'][np.isin(rows['ancestor'],anc)]).tolist()
                     for part,anc in plan['partition'].items()}
            if any(v!=[0,1] for v in support.values()):
                write_json(root(c)/'technical/unsupported'/f'{target}-{horizon:g}.json',support)
                continue
            y=labelled['label'][held]
            for space in spaces:
                values=x[:,:128] if space=='structural' else x[:,128:] if space=='motion_only' else x
                kinds=['prior'] if name=='prior' else c['readouts']
                for kind in kinds:
                    out=root(c)/'analyses'/target
                    fit_c=dict(c,output=str(out));bound=dict(plan,identity=digest([plan['identity'],target,horizon,metadata]))
                    model=dict(metadata,name=name,cohort=done['identity'])
                    fit(fit_c,dict(config=fit_c),labelled,values,model,space,horizon,kind,plan=bound)
                    folder=out/'analyses/fits-v1'/name/space/f'lead-{horizon:g}ps'/kind/'technical'
                    metric_path=folder/'appearance-metrics.json'
                    if metric_path.exists():continue
                    with np.load(folder/'predictions.npz') as saved:
                        if not np.array_equal(saved['id'],rows['id']):raise ValueError('Prediction row order mismatch')
                        probability=saved['probability'][held]
                    point=scores(y,probability,weights)
                    boot,valid=bootstrap_scores(y,probability,weights,slot,draws,None)
                    metrics=[dict(model=name,space=space,target=target,horizon_ps=horizon,readout=kind,
                        metric=k,value=point[k],lower=float(np.quantile(boot[k],.025)),upper=float(np.quantile(boot[k],.975)),
                        test_rows=int(held.sum()),test_positives=int(y.sum()),test_ancestors=len(set(rows['ancestor'][held])),
                        prevalence=float(np.average(y,weights=weights)),bootstrap_valid=valid)
                        for k in ('ap','auroc','nll','brier')]
                    # History is audit-only. Report this subgroup without refitting or selection.
                    clean=rows['history_clear'][held].astype(bool)
                    if set(y[clean])=={0,1}:
                        for k,v in scores(y[clean],probability[clean],weights[clean]).items():
                            if k in ('ap','auroc','nll','brier'):
                                metrics.append(dict(model=name,space=space,target=target+'_past05clear',horizon_ps=horizon,
                                    readout=kind,metric=k,value=v,lower=None,upper=None,test_rows=int(clean.sum()),
                                    test_positives=int(y[clean].sum()),test_ancestors=len(set(rows['ancestor'][held][clean])),
                                    prevalence=float(np.average(y[clean],weights=weights[clean])),bootstrap_valid=0))
                    calibration=[]
                    bins=np.minimum((probability*10).astype(int),9)
                    for b in range(10):
                        mask=bins==b
                        if mask.any():calibration.append(dict(bin=b,count=int(mask.sum()),weight=float(weights[mask].sum()),
                            predicted=float(np.average(probability[mask],weights=weights[mask])),observed=float(np.average(y[mask],weights=weights[mask]))))
                    np.savez_compressed(folder/'bootstrap.npz',**boot)
                    write_json(metric_path,dict(metrics=metrics,calibration=calibration))
                    print('appearance scores',name,space,target,horizon,round(point['ap'],4),flush=True)


def controls(c):
    physical(c)
    columns=read(cache(c)/'columns.json');family=groups(columns)
    x=np.load(cache(c)/'physical.npy',mmap_mode='r')
    evaluate_features(c,'prior',np.zeros((len(x),1),np.float32),['constant'],{})
    for name in c['physical_controls']:
        evaluate_features(c,'physical-'+name,x[:,family[name]],['physical'],
            dict(schema_sha256=sha(cache(c)/'columns.json')))


def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[];calibration=[]
    for path in sorted((root(c)/'analyses').rglob('appearance-metrics.json')):
        value=read(path);metrics.extend(value['metrics'])
        context={k:value['metrics'][0][k] for k in ('model','space','target','horizon_ps','readout')}
        calibration.extend(dict(context,**r) for r in value['calibration'])
    if not metrics:raise ValueError('No completed first-appearance readouts')
    write_metric_rows(metrics,root(c),family=FAMILY,name='scores')
    write_metric_rows(calibration,root(c),family=FAMILY,name='calibration')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    selected=[r for r in metrics if r['target']=='any' and r['metric']=='ap' and
        (r['space']=='structural_motion' or r['model'].startswith('physical-') or r['model']=='prior')]
    for name in sorted({r['model'] for r in selected}):
        data=sorted((r for r in selected if r['model']==name),key=lambda r:r['horizon_ps'])
        axes[0].plot([r['horizon_ps'] for r in data],[r['value'] for r in data],'.-',label=name)
    axes[0].set(xlabel='Future observation horizon (ps)',ylabel='AP',title='First local PTM appearance')
    # The null is prevalence of the new risk set, not the former forced 20%.
    for target in ('any','persistent'):
        with (root(c)/'tables/cohort-coverage.csv').open() as f:
            import csv
            data=[r for r in csv.DictReader(f) if r['role']=='test' and r['target']==target]
        axes[1].plot([float(r['horizon_ps']) for r in data],[float(r['ancestor_weighted_prevalence']) for r in data],'.-',label=target)
    axes[1].set(xlabel='Future observation horizon (ps)',ylabel='Event frequency',title='Observed held-out event frequency')
    for ax in axes:ax.grid(alpha=.2)
    axes[1].legend();axes[0].legend(fontsize=6,loc='best')
    fig.tight_layout();(root(c)/'plots').mkdir(exist_ok=True)
    fig.savefig(root(c)/'plots/first-appearance.png',dpi=180);fig.savefig(root(c)/'plots/first-appearance.pdf');plt.close(fig)
    (root(c)/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>First crystalline appearance</title>'
        '<h1>First observed crystalline appearance</h1><p>Uniformly sampled sites, fixed observation times, '
        'no present crystalline atom in the consumed 80-atom patch. Future PTM uses original full-cell periodic geometry. '
        'Atom identities are fixed at the present. Transient appearances count. Persistent means the same atom is '
        'crystalline in three consecutive saved frames. No eventual nucleus-establishment condition.</p>'
        '<p>Original training ancestors fit NLL-selected frozen readouts. Selection, calibration and test sources '
        'are pooled only for evaluation. Curves use equal total weight per independent ancestor; natural row counts '
        'and event rates are also exported. A future horizon is not a lead before a preselected birth. '
        'These AP values are not directly comparable to the previous 20%-positive founder/control assay.</p>'
        '<img src="plots/first-appearance.png" style="max-width:100%"><p><a href="tables/scores.csv">Scores</a> · '
        '<a href="tables/cohort-coverage.csv">Coverage and prevalence</a> · '
        '<a href="tables/calibration.csv">Calibration</a> · <a href="tables/METRICS.md">Definitions</a></p>')


def run(c):
    with recorded_stage(root(c)/'technical/coordinator.json',stage='collect') as progress:
        collect(c);progress.update(stage='physical_controls');controls(c)
        progress.update(stage='frozen_encoders');encode(c)
        progress.update(stage='publication');report(c)
        write_json(root(c)/'technical/complete.json',dict(state='complete',completed_unix=time.time()))


def launch(c):
    prepare(c);tech=root(c)/'technical'
    if (tech/'launch.json').exists():raise FileExistsError('Existing appearance launch; resume explicitly')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics'))
    command=[sys.executable,'-u','-m',MODULE,'run','--config',str(bundle.config_path)]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        CUEQUIVARIANCE_OPS_PARALLEL_THREADS='4')
    receipt=dict(owner='current response/nucleus-recovery chat; first-crystal target requested 2026-10-07',
        host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'),command=command,
        started_unix=time.time(),deadline_unix=allocation_deadline(reserve_seconds=120),plan=read(tech/'plan.json')['identity'])
    with (tech/'coordinator.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(pid=child.pid,process_group=child.pid);write_json(tech/'launch.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('prepare','collect','controls','encode','report','run','launch'))
    p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config)
    check_metric_docs(family=FAMILY);globals()[a.stage](c)


if __name__=='__main__':main()
