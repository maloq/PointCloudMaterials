"""Nucleus preappearance transfer of frozen Q/QV value/response encoders."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.research.birth_prediction.data import load, read, ROLES
from src.research.birth_prediction.fit import scores, calibrate
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores, measured
from src.research.birth_prediction.temporal_inputs import packet, context
from src.research.equivariant_context.cache import RetainedCache

FAMILY = 'response_birth'
METRICS = ['nll','brier','auroc','ap','within_pair_auc','within_pair_top1',
           'recall_at_locked_fpr05','observed_fpr']


def paths(c):
    return resolve_path(c['campaign']), resolve_path(c['output']), resolve_path(c['cache'])


def setup(c):
    recipe = read(resolve_path(c['birth_recipe']))
    positions, rows, manifest = load(recipe)
    if manifest['identity'] != c['birth_identity']:
        raise ValueError('Changed birth cohort')
    return recipe, positions, rows, manifest


def prepare(c):
    import torch
    from src.postprocessing.protocols.extract import raw_source
    from src.postprocessing.io import frame_geometry
    check_metric_docs(family=FAMILY)
    campaign, out, cache = paths(c)
    recipe, positions, rows, manifest = setup(c)
    cache.mkdir(parents=True,exist_ok=True)
    if read(campaign/'technical/binding.json')['identity'] != c['campaign_identity']:
        raise ValueError('Changed response campaign')
    cfg = read(campaign/'technical/code/config.json')
    if cfg['fixed_identity'] != c['fixed_identity'] or recipe['fixed_dataset']['identity'] != c['fixed_identity']:
        raise ValueError('Different Al64 ancestry release')
    plan = read(resolve_path(recipe['cache'])/'plan.json')
    models = []
    for seed in cfg['fit_seeds']:
        for arm in cfg['arms']:
            name = arm['name']+'-'+str(seed)
            folder = campaign/'analyses'/name/'technical'
            done = read(folder/'complete.json')
            if sha(folder/'best.pt') != done['checkpoint_sha256']:
                raise ValueError('Changed selected model: '+name)
            models.append(dict(name=name, arm=arm['name'], seed=seed, velocities=arm['velocities'],
                checkpoint=str(folder/'best.pt'), sha256=done['checkpoint_sha256']))
    data_receipt = read(campaign/'technical/data-complete.json')
    if sha(Path(data_receipt['cache'])) != data_receipt['sha256']:
        raise ValueError('Changed response training/query bank')
    data = torch.load(data_receipt['cache'], map_location='cpu', weights_only=False)
    for record in data['records']:
        source_role = next(s['role'] for s in plan['sources'] if s['id']==record['source'])
        if source_role != record['role']:
            raise ValueError('Encoder training or selection crosses frozen birth roles')
    test_records = [r for r in data['records'] if r['role']=='test'][:8]
    np.savez_compressed(cache/'replay.npz',state=np.stack([
        torch.cat((r['position'],r['velocity']),-1).numpy() for r in test_records]))
    del data
    # Link each released patch to its actual source, frame, and tracked atom.
    keys = {}
    for i in range(len(rows['id'])):
        for slot, patch_id in enumerate(rows['indices'][i]):
            key = (int(rows['source'][i]),int(rows['start_frame'][i])+slot,int(rows['atom'][i]))
            if int(patch_id) in keys and keys[int(patch_id)] != key:
                raise ValueError('A frozen patch maps to conflicting physical observations')
            keys[int(patch_id)] = key
    if set(keys) != set(range(len(positions))):
        raise ValueError('Missing velocity correspondence for a frozen birth patch')
    velocity_path = cache/'velocities.npy'
    velocity_receipt = cache/'velocities.json'
    if velocity_receipt.exists():
        old = read(velocity_receipt)
        if old['birth_identity'] != c['birth_identity'] or sha(velocity_path) != old['sha256']:
            raise ValueError('Changed actual birth-velocity bank')
        provenance = old['sources']
    else:
        velocity = np.lib.format.open_memmap(velocity_path,mode='w+',dtype=np.float32,shape=positions.shape)
        provenance = []
        grouped = defaultdict(lambda:defaultdict(list))
        for patch_id,(source,frame,atom) in keys.items():
            grouped[source][frame].append((patch_id,atom))
        for source, frames in grouped.items():
            item = next(s for s in plan['sources'] if s['id']==source)
            raw = raw_source(item)
            if raw.manifest['velocity_units'] != 'angstrom_per_ps':
                raise ValueError('Unexpected raw velocity units: '+str(raw.root))
            for frame, observations in sorted(frames.items()):
                ids = np.array([x[0] for x in observations])
                atom_ids = np.array([x[1] for x in observations])
                atoms = np.searchsorted(raw.atom_ids,atom_ids)
                if not np.array_equal(raw.atom_ids[atoms],atom_ids):
                    raise ValueError('Missing actual tracked atom ID')
                q, box = frame_geometry(raw,frame)
                _, neighbors = cKDTree(q,boxsize=box).query(q[atoms],k=80,workers=1)
                delta = q[neighbors]-q[atoms,None]
                delta -= box*np.rint(delta/box)
                if not np.array_equal(neighbors[:,0],atoms) or not np.array_equal(delta.astype(np.float32),positions[ids]):
                    raise ValueError(f'Raw birth patch reconstruction differs: {source}/{frame}')
                # Exact simultaneous MD velocities of the same 80 atom identities.
                velocity[ids] = np.asarray(raw.velocities[frame,neighbors],np.float32)/np.float32(1000.)
            provenance.append(dict(source=source,role=item['role'],root=str(raw.root),
                manifest_sha256=item['manifest_sha256'],raw_velocity=raw.manifest['arrays']['velocities'],
                frames=sorted(frames),positions_reconstructed_exactly=True))
            print('actual velocities',source,len(frames),'frames',flush=True)
        velocity.flush()
        if not np.isfinite(velocity).all():
            raise FloatingPointError('Nonfinite actual initial velocities')
        del velocity
        write_json(velocity_receipt,dict(birth_identity=c['birth_identity'],sha256=sha(velocity_path),
            sources=provenance,conversion='raw angstrom/ps divided by1000 -> float32 angstrom/fs'))
    baseline_records = []
    for observation in c['observations']:
        for name in ['rich_gbdt','rich_linear','mace_rich_gbdt','mace_rich_linear','mace_vicreg_gbdt','mace_vicreg_linear','prior']:
            folder = resolve_path(c['baseline'])/'analyses/fits-v1/fixed-test'/name/('prior' if name=='prior' else observation['name'])/'technical'
            done = read(folder/'complete.json')
            if sha(folder/'predictions.npz') != done['predictions_sha256']:
                raise ValueError('Changed retained baseline prediction')
            saved = np.load(folder/'predictions.npz')
            if not np.array_equal(saved['id'],rows['id']) or not np.array_equal(saved['role'],rows['role']):
                raise ValueError('Baseline evaluated different rows or roles')
            for field in ['label','source','event','pair','weight']:
                if not np.array_equal(saved[field],rows[field]):
                    raise ValueError('Baseline target/weight mismatch: '+field)
            baseline_records.append(dict(name=name,observation=observation['name'],path=str(folder),
                predictions_sha256=done['predictions_sha256'],selection_sha256=sha(folder/'selection.json')))
    dependency_names = ['src/research/response_velocity/model.py','src/research/local_response/model.py',
        'src/research/supervised_onset/model.py','src/models/encoders/spatial_mace.py',
        'src/models/encoders/mace_backend.py','config.json']
    record = dict(config=c,birth_manifest=manifest,birth_recipe=recipe,models=models,
        baselines=baseline_records,velocity_receipt=read(velocity_receipt),
        pool_release_sha256=sha(resolve_path(c['pool_release'])),
        dependencies={f:sha(campaign/'technical/code'/f) for f in dependency_names},
        positions=str(resolve_path(recipe['cache'])/'positions.npy'),
        scope='original unrelaxed isolated establishments; fixed Al64 roles; derived event-centered track',
        deferred=dict(dense010='sealed135 rows but only train/selection ancestors; no fixed-test births',
            shooting='different label cadence and appearance/fate definition; not pooled',
            relaxed='excluded by requested unrelaxed input domain',
            other_Al='full-cadence search running; no new sealed matched birth cohort used'))
    record['identity'] = digest(record)
    target = out/'technical/binding.json'
    if target.exists() and read(target) != record:
        raise ValueError('Changed birth evaluation; use a new revision')
    write_json(target,record)
    write_json(out/'technical/prediction-context.json',dict(
        encoder=dict(positions='exact original centered nearest80; support8A; edges5A; no halo',
            velocities='QV: actual same-frame nearest80 MD velocities in A/fs, center-relative inside frozen model; Q ignores V',
            precision='historical float16 MD archives converted to float32 inference; no precision recovery',
            history=1,conditions=[],relaxation=False,checkpoint_selection='original response-validation feature likelihood',
            teacher='historical 100/500fs FastAl MEAM response targets; no crystal labels in encoder training'),
        predictor=dict(spaces=c['spaces'],readouts=c['readouts'],observations=c['observations'],conditions=[],
            velocity='only through QV frozen representation',selection='selection-source binary NLL',
            calibration='separate calibration-source monotone affine logit calibration'),
        cohort=record['scope'],tracking='local frozen diagnostic probes; no new scientific training runs'))
    print('Prepared twelve encoders, 96 frozen readouts, and retained structural controls',flush=True)


def fit_one(c, recipe, rows, x, model, space, observation, readout, out):
    tech = out/'technical'
    tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():
        done=read(tech/'complete.json')
        if sha(tech/'predictions.npz')!=done['predictions_sha256']:
            raise ValueError('Changed completed birth readout')
        return
    split={r:np.flatnonzero(rows['role']==r) for r in ROLES}
    y,w=rows['label'],rows['weight']; tr,va,ca=split['train'],split['selection'],split['calibration']
    history=[]
    if readout=='linear':
        weight=w[tr]/w[tr].sum(); mean=weight@x[tr]; scale=np.sqrt(weight@(x[tr]-mean)**2).clip(1e-4)
        xx=(x-mean)/scale; best=np.inf
        with threadpool_limits(c['fit_threads']),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            for strength in recipe['linear_C']:
                candidate=LogisticRegression(C=strength,solver='lbfgs',max_iter=3000,tol=1e-5,random_state=c['seed'])
                candidate.fit(xx[tr],y[tr],sample_weight=w[tr]/w[tr].mean())
                nll=scores(y[va],candidate.predict_proba(xx[va])[:,1],w[va])['nll']
                history.append(dict(C=strength,selection_nll=nll))
                if nll<best: best, fitted, selected=nll,candidate,dict(C=strength,nll=nll)
            probability=fitted.predict_proba(xx)[:,1]
        joblib.dump(dict(model=fitted,mean=mean,scale=scale),tech/'model.joblib')
    elif readout=='catboost':
        from catboost import CatBoostClassifier, Pool
        fitted=CatBoostClassifier(**recipe['catboost'],loss_function='Logloss',eval_metric='Logloss',
            random_seed=c['seed'],thread_count=c['fit_threads'],train_dir=str(tech/'catboost'))
        def pool(ids): return Pool(x[ids],y[ids],weight=w[ids]/w[ids].mean())
        fitted.fit(pool(tr),eval_set=pool(va),early_stopping_rounds=recipe['patience'],use_best_model=True,
            verbose=False,save_snapshot=True,snapshot_file=str(tech/'snapshot.cbs'),snapshot_interval=60)
        probability=fitted.predict_proba(x)[:,1]
        fitted.save_model(str(tech/'model.cbm'))
        selected=dict(iterations=fitted.get_best_iteration()+1,nll=scores(y[va],probability[va],w[va])['nll'])
        history=fitted.get_evals_result()
    else: raise ValueError(readout)
    theta=calibrate(y[ca],probability[ca],w[ca])
    calibrated=expit(theta[0]*logit(np.clip(probability,1e-6,1-1e-6))+theta[1])
    neg=ca[y[ca]==0]; order=neg[np.argsort(calibrated[neg])]
    cumulative=np.cumsum(w[order])/w[order].sum()
    threshold=float(calibrated[order[min(np.searchsorted(cumulative,.95),len(order)-1)]])
    np.savez_compressed(tech/'predictions.npz',probability=probability,calibrated=calibrated,
        **{k:rows[k] for k in ['id','source','event','pair','label','weight','role']})
    write_json(tech/'selection.json',dict(selected=selected,history=history,calibration_parameters=theta.tolist(),
        threshold_fpr05=threshold,selector='selection-source binary NLL'))
    input_record=context(observation,.75)
    input_record.update(model=model,space=space,readout=readout,actual_velocity=model['velocities'],
        motion='actual instantaneous velocities through QV encoder; chronological frozen features for history8',
        encoder_frozen=True,conditions=[],relaxation=False,columns=x.shape[1])
    write_json(tech/'prediction-context.json',input_record)
    result=[]
    for role,ids in split.items():
        for kind,p in [('raw',probability),('calibrated',calibrated)]:
            result.append(dict(role=role,score=kind,**scores(y[ids],p[ids],w[ids])))
    write_metric_rows(result,out,family=FAMILY,name='scores')
    write_json(tech/'complete.json',dict(predictions_sha256=sha(tech/'predictions.npz'),
        selection=selected,model=model['name'],space=space,observation=observation,readout=readout))
    print('fit',model['name'],space,observation['name'],readout,'complete',flush=True)


def lane(c,index):
    campaign,out,cache=paths(c)
    binding=read(out/'technical/binding.json')
    recipe,_,rows,_=setup(c)
    pool=RetainedCache(resolve_path(c['feature_cache']),6)
    with recorded_stage(out/'technical'/f'lane-{index}.json',lane=index) as progress:
        for model in binding['models'][index::c['lanes']]:
            metadata=dict(study=binding['identity'],model=model,inference=sha(Path(__file__).with_name('response_birth_infer.py')))
            key=digest(metadata)
            with pool.lease(key,deadline=time.time()+8*3600,metadata=metadata) as bank:
                request=dict(model=model,producer=str(campaign/'technical/code'),dependencies=binding['dependencies'],
                    positions=binding['positions'],velocities=str(cache/'velocities.npy'),destination=str(bank),
                    audit=str(cache/'replay.npz'),audit_reference=str(campaign/'analyses'/model['name']/'technical/predictions.npz'),
                    batch_size=c['inference_batch_size'])
                request_path=out/'technical'/('inference-'+model['name']+'.json')
                write_json(request_path,request)
                if not (bank/'complete.json').exists():
                    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('response_birth_infer.py')),str(request_path)],check=True)
                receipt=read(bank/'complete.json')
                if any(sha(bank/f)!=h for f,h in receipt['files'].items()):
                    raise ValueError('Changed frozen birth representations')
                write_json(out/'technical'/('features-'+model['name']+'.json'),dict(key=key,path=str(bank),**receipt))
                for space in c['spaces']:
                    features=np.load(bank/(space+'.npy'))
                    for observation in c['observations']:
                        x=packet(features,rows,np.arange(features.shape[1]),observation,c['seed'])
                        for readout in c['readouts']:
                            progress.update(model=model['name'],space=space,observation=observation['name'],readout=readout)
                            destination=out/'analyses/fits-v1'/model['name']/space/observation['name']/readout
                            fit_one(c,recipe,rows,x,model,space,observation,readout,destination)


def matched(rows, probability):
    auc=np.empty(len(probability));top=np.empty(len(probability))
    for pair in np.unique(rows['pair']):
        ids=np.flatnonzero(rows['pair']==pair);pos=ids[rows['label'][ids]==1];neg=ids[rows['label'][ids]==0]
        if len(pos)!=1 or len(neg)!=4: raise ValueError('Broken1:4 matched set')
        auc[ids]=np.mean((probability[pos[0]]>probability[neg])+.5*(probability[pos[0]]==probability[neg]))
        maximum=probability[ids].max()
        top[ids]=float(probability[pos[0]]==maximum)/np.count_nonzero(probability[ids]==maximum)
    return auc,top


def analyze(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _,out,_=paths(c);binding=read(out/'technical/binding.json')
    _,_,all_rows,_=setup(c)
    test=all_rows['role']=='test';rows={k:v[test] for k,v in all_rows.items()}
    y,w=rows['label'],rows['weight'];sources,slot,counts=source_draws(rows['source'],c['bootstrap_draws'],c['seed'])
    bw=counts[:,slot]*w[None];den=bw.sum(1)
    specifications=[]
    for model in binding['models']:
        for space in c['spaces']:
            for obs in c['observations']:
                for readout in c['readouts']:
                    specifications.append(dict(name=model['name'],arm=model['arm'],seed=model['seed'],space=space,
                        observation=obs['name'],readout=readout,path=out/'analyses/fits-v1'/model['name']/space/obs['name']/readout/'technical'))
    for item in binding['baselines']:
        name=item['name'];readout='prior' if name=='prior' else ('linear' if name.endswith('linear') else 'catboost')
        arm=name.rsplit('_',1)[0] if name!='prior' else 'prior'
        specifications.append(dict(name=name,arm=arm,seed=-1,space='control',observation=item['observation'],readout=readout,path=Path(item['path'])))
    individual=[];details={};grouped=defaultdict(list);prediction_sources=[]
    for spec in specifications:
        folder=spec['path'];done=read(folder/'complete.json');selection=read(folder/'selection.json')
        if sha(folder/'predictions.npz')!=done['predictions_sha256']: raise ValueError('Changed saved prediction')
        data=np.load(folder/'predictions.npz')
        if not np.array_equal(data['id'],all_rows['id']): raise ValueError('Different scored rows')
        prediction_sources.append(dict(path=str(folder/'predictions.npz'),sha256=done['predictions_sha256']))
        for kind in ['raw','calibrated']:
            p=data['probability' if kind=='raw' else kind][test]
            alarm=data['calibrated'][test]>selection['threshold_fpr05']
            value=measured(y,p,w,alarm if kind=='calibrated' else None)
            boot,_=bootstrap_scores(y,p,w,slot,counts,alarm if kind=='calibrated' else None)
            pair_auc,top=matched(rows,p)
            for name,values in [('within_pair_auc',pair_auc),('within_pair_top1',top)]:
                value[name]=float(np.average(values,weights=w));boot[name]=bw@values/den
            info={k:v for k,v in spec.items() if k!='path'}; info['score']=kind
            for metric in METRICS:
                lo,hi=np.quantile(boot[metric],[.025,.975]) if metric in boot else (None,None)
                individual.append(dict(**info,metric=metric,value=value[metric],lo=lo,hi=hi))
            key=(spec['arm'],spec['space'],spec['observation'],spec['readout'],kind)
            grouped[key].append((value,boot,spec['seed']))
    summary=[];means={}
    for key,entries in grouped.items():
        for metric in METRICS:
            if entries[0][0][metric] is None: continue
            values=np.array([x[0][metric] for x in entries]);boot=np.mean([x[1][metric] for x in entries],axis=0)
            lo,hi=np.quantile(boot,[.025,.975]);point=float(values.mean())
            means[key+(metric,)]=(point,boot)
            summary.append(dict(arm=key[0],space=key[1],observation=key[2],readout=key[3],score=key[4],metric=metric,
                value=point,lo=float(lo),hi=float(hi),seed_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                fitted_seeds=len(values),independent_test_sources=len(sources),test_events=15,test_rows=len(y)))
    contrasts=[]
    for space in c['spaces']:
        for obs in c['observations']:
            for readout in c['readouts']:
                for left,right in [('q_response','q_values'),('qv_response','qv_values'),('qv_values','q_values'),('qv_response','q_response')]:
                    for kind in ['raw','calibrated']:
                        for metric in METRICS:
                            a=(left,space,obs['name'],readout,kind,metric);b=(right,)+a[1:]
                            if a not in means: continue
                            point=means[a][0]-means[b][0];boot=means[a][1]-means[b][1];lo,hi=np.quantile(boot,[.025,.975])
                            contrasts.append(dict(left=left,right=right,space=space,observation=obs['name'],readout=readout,
                                score=kind,metric=metric,difference=point,lo=float(lo),hi=float(hi)))
                for left in ['q_values','q_response','qv_values','qv_response']:
                    for right in ['rich','mace_rich','mace_vicreg']:
                        for kind in ['raw','calibrated']:
                            for metric in METRICS:
                                a=(left,space,obs['name'],readout,kind,metric)
                                b=(right,'control',obs['name'],readout,kind,metric)
                                if a not in means: continue
                                point=means[a][0]-means[b][0];boot=means[a][1]-means[b][1];lo,hi=np.quantile(boot,[.025,.975])
                                contrasts.append(dict(left=left,right=right,space=space,observation=obs['name'],readout=readout,
                                    score=kind,metric=metric,difference=point,lo=float(lo),hi=float(hi)))
    for name,values in [('models',individual),('summary',summary),('paired-comparisons',contrasts)]:
        write_metric_rows(values,out,family=FAMILY,name=name)
    plots=out/'plots';plots.mkdir(exist_ok=True)
    for obs in c['observations']:
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for row,space in enumerate(c['spaces']):
            for col,metric in enumerate(['nll','within_pair_top1']):
                ax=axes[row,col]
                candidates=[(a,space) for a in ['q_values','q_response','qv_values','qv_response']]+[(a,'control') for a in ['rich','mace_rich','mace_vicreg']]
                for i,(arm,sp) in enumerate(candidates):
                    v,boot=means[(arm,sp,obs['name'],'catboost','calibrated',metric)]
                    lo,hi=np.quantile(boot,[.025,.975]);ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o')
                ref=.5004024235381879 if metric=='nll' else .2
                ax.axvline(ref,color='grey',ls='--',lw=1)
                ax.set(yticks=range(len(candidates)),yticklabels=[a for a,_ in candidates],
                    xlabel='Binary NLL (lower better)' if metric=='nll' else 'Case ranked first among five (higher better)',title=space)
                ax.invert_yaxis()
        fig.suptitle(f'Nucleus birth before local crystal appearance: {obs["name"]}\nNonlinear readouts;95% source bootstrap;15 test events /11 sources')
        fig.savefig(plots/(obs['name']+'.png'),dpi=180);fig.savefig(plots/(obs['name']+'.svg'));plt.close(fig)
    lines=['# Frozen response encoders: nucleus birth prediction','',
        'Original unrelaxed birth assay; exact frozen source roles. Inputs end0.75ps before first local crystal appearance.',
        'The entire8A input sphere is crystal-free at all8 observed times. Positive appearances lead to isolated establishment.',
        'Matched1:4 case/control sampling; probabilities are not natural nucleation risk. Previously inspected test sources.',
        'Actual simultaneous MD velocities are consumed by QV; no new MD or encoder training.','',
        '| Observation | Space | Encoder/control | Readout | NLL | AUROC | AP | Within-set top1 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for key in grouped:
        arm,space,obs,readout,kind=key
        if kind!='calibrated':continue
        vals=[means[key+(metric,)][0] for metric in ['nll','auroc','ap','within_pair_top1']]
        lines.append(f'| {obs} | {space} | {arm} | {readout} | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','Three pretrained response seeds are summarized as mean per-model scores, not an ensemble. Structural controls retain their one original fitted seed.',
        'Paired intervals resample whole test sources using common draws, conditional on the fitted models. Raw/calibrated scores and seed variation are saved.',
        'Other Al pools remain separate: shooting differs in protocol; dense0.1ps has no eligible fixed-test ancestors in its sealed release; relaxed views are excluded.',
        '[Summary](tables/summary.csv) · [Paired differences](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'technical/complete.json',dict(state='complete',fits=96,retained_baseline_records=14,
        birth_identity=c['birth_identity'],test_sources=sources.tolist(),prediction_sources=prediction_sources))


def tracking(c):
    import csv
    from src.experiment_runner.wandb_tracking import update_recorded_summary
    campaign,out,_=paths(c)
    with (out/'tables/models.csv').open() as stream: rows=list(csv.DictReader(stream))
    for model in read(out/'technical/binding.json')['models']:
        fields={f'nucleus_birth/{r["space"]}/{r["observation"]}/{r["readout"]}/{r["score"]}/{r["metric"]}':float(r['value'])
            for r in rows if r['name']==model['name'] and r['value']!=''}
        update_recorded_summary(campaign/'analyses'/model['name']/'technical/wandb.json',fields,
            evaluation='nucleus-birth-v1',expected=dict(entity='teshbek',project='PointCloudMaterials'))
        print('updated training summary',model['name'],flush=True)
    write_json(out/'technical/tracking-complete.json',dict(updated_training_runs=12,new_runs=0))


def coordinate(c):
    _,out,_=paths(c)
    with recorded_stage(out/'technical/coordinator.json'):
        children=[]
        for index in range(c['lanes']):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),OMP_NUM_THREADS=str(c['fit_threads']),OPENBLAS_NUM_THREADS='1')
            log=(out/'technical'/f'lane-{index}.log').open('a')
            child=subprocess.Popen([sys.executable,'-u','-m','src.research.response_birth','lane','--config',str(out/'technical/code/config.json'),'--lane',str(index)],
                cwd=out/'technical/code',env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);log.close()
            write_json(out/'technical'/f'process-{index}.json',dict(pid=child.pid,gpu=index,allocation=os.environ.get('SLURM_JOB_ID')))
        codes=[p.wait() for p in children]
        if any(codes):raise RuntimeError('Birth evaluation lane failures: '+str(codes))
        analyze(c)
        tracking(c)


def launch(c):
    _,out,_=paths(c)
    if (out/'technical/launch.json').exists():raise ValueError('Already launched; inspect owned process receipts before resuming')
    prepare(c)
    bundle=ExecutionBundle.freeze(REPO,out/'technical/code',c,directories=('src','docs/metrics','configs'),
        files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    log=(out/'technical/coordinator.log').open('a')
    process=subprocess.Popen([sys.executable,'-u','-m','src.research.response_birth','coordinate','--config',str(bundle.config_path)],
        cwd=bundle.root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'))
    log.close()
    receipt=dict(pid=process.pid,allocation=os.environ.get('SLURM_JOB_ID'),gpus=[0,1],code=str(bundle.root),
        command='response_birth coordinate',started_at=time.time(),ownership='launched in current nuclei-birth request')
    write_json(out/'technical/launch.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','launch','coordinate','lane','analyze','tracking'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int)
    args=parser.parse_args();c=read(args.config)
    if args.stage=='lane':lane(c,args.lane)
    else:globals()[args.stage](c)


if __name__=='__main__':main()
