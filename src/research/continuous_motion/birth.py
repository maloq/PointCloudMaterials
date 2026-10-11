"""Frozen continuous-motion transfer to established nucleus birth at exact leads."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, allocation_deadline, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.research.response_dense_birth.data import read, output, cache, source, seal, dataset
from src.research.response_dense_birth.pooled import fit, METRICS
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
from src.research.response_birth import matched
from src.research.equivariant_context.cache import RetainedCache

FAMILY = 'continuous_motion_birth'
OWNER = 'current response/nucleus-recovery chat; continuous-motion birth evaluation'
MODULE = 'src.research.continuous_motion.birth'


def prepare(c):
    parent = resolve_path(c['source_plan'])
    upstream = read(parent)
    if upstream['identity'] != c['source_identity']:raise ValueError('Changed source inventory')
    split = read(resolve_path(c['fixed_release'])/'splits.json')
    if split['identity'] != c['fixed_identity']:raise ValueError('Changed fixed Al64 split')
    roles = {s['lineage']:s['role'] for s in split['sources']}
    sources = [{k:v for k,v in item.items() if k!='reuse'} for item in upstream['sources']]
    for item in sources:
        s = item['source']
        if roles[s['ancestry_group']] != s['role']:raise ValueError('Source ancestry role changed')
        times = np.asarray(s['times_ps'])[s['views']['010']]
        np.testing.assert_allclose(np.diff(times),.1,atol=1e-8,rtol=0)
        for part in s['parts']:
            if sha(Path(part['path'])/'manifest.json') != part['manifest_sha256']:
                raise ValueError('Changed source manifest: '+s['id'])
        done = read(Path(item['label_folder'])/'complete.json')
        if done['identity'] != item['label_identity']:raise ValueError('Changed lineage labels')
    campaign = resolve_path(c['campaign']); producer = campaign/'technical/code'
    cfg = read(producer/'config.json'); training = read(campaign/'technical/plan.json')
    if training['identity'] != c['training_plan_identity'] or cfg['fixed_identity'] != c['fixed_identity']:
        raise ValueError('Changed continuous-motion training data')
    for item in training['sources']:
        if roles[item['source']['ancestry_group']] != item['original_role']:
            raise ValueError('Encoder ancestor contamination')
        if item['role'] in ['train','selection'] and item['original_role']!='train':
            raise ValueError('Held-out ancestor used in encoder fitting/selection')
    models = [dict(name=f'{arm}-{seed}',arm=arm,seed=seed,velocities=True)
              for seed in cfg['fit_seeds'] for arm in cfg['arms']]
    names = ['config.json','src/research/continuous_motion/model.py','src/research/local_response/model.py',
             'src/research/supervised_onset/model.py','src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py']
    p = dict(config=c,sources=sources,releases=upstream['releases'],excluded=upstream['excluded'],models=models,
             inventory_sha256=sha(parent),training_plan_sha256=sha(campaign/'technical/plan.json'),
             dependencies={name:sha(producer/name) for name in names},
             implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    p['identity'] = digest(p)
    path = output(c)/'technical/plan.json'
    if path.exists() and read(path)!=p:raise ValueError('Changed cohort plan; use a new revision')
    write_json(path,p);cache(c).mkdir(parents=True,exist_ok=True)
    write_json(output(c)/'technical/prediction-context.json',dict(
        encoder=dict(positions='unrelaxed nearest80, smooth8A support,5A edges,no halo',
            velocity='native same-frame same-atom A/ps; birth cache A/fs explicitly multiplied by1000',
            history=1,conditions=[],species='one constant channel',relaxation=False,
            training_only_teacher='100fs fixed-box BAOAB FastAl float32 force / float64 integration; saved NPT futures to12ps'),
        predictor=dict(spaces={'structural':128,'structural_motion':256,'future_law':48},
            future_law='24 means +24 log total variances at the declared lead;8 latent SDE particles',
            conditions=[],lead_as_network_input=False,selection='NLL within original training ancestors',
            calibration='none; original selection/calibration/test pooled only for evaluation'),
        labels='1/3/6ps before root8-atom emergence that later sustains64 atoms for1.5ps;1:4 matched liquid controls',
        eligibility='No8+ connected component within8A at any of the three inputs; exact native0.1ps grid, no interpolation',
        scope='Pinned92 saved-velocity views; unknown-potential historical views retained as transfer data; noV/million and other cadences excluded',
        limitations='Event-enriched retrospective assay, not population nucleation probability; prior related evaluations have been inspected'))
    return p


def readout_plan(c,rows,p):
    ancestors = sorted(set(rows['ancestor'][rows['role']=='train']),
                       key=lambda a:digest([c['seed'],str(a),'inner-selection']))
    n = c['inner_selection_ancestors']
    if len(ancestors)<=n:raise ValueError('Insufficient training ancestors for declared inner split')
    held = sorted(set(rows['ancestor'][rows['role']!='train']))
    if set(ancestors)&set(held):raise ValueError('Ancestor crosses held-out roles')
    partition = dict(inner_selection=ancestors[:n],inner_fit=ancestors[n:],final_train=ancestors,test=held)
    counts=[]
    for name,aa in partition.items():
        ix=np.isin(rows['ancestor'],aa)
        if set(rows['label'][ix])!={0,1}:raise ValueError('Both classes required in '+name)
        counts.append(dict(partition=name,ancestors=len(aa),events=len(set(rows['event'][ix])),rows=int(ix.sum())))
    if counts[-1]['events']<c['minimum_test_events']:raise ValueError('Too few pooled test events; labels remain unchanged')
    result=dict(cohort=p['identity'],partition=partition,counts=counts,
                selector='NLL on inner training ancestors; refit all training; AP diagnostic only')
    result['identity']=digest(result);path=output(c)/'technical/readout-plan.json'
    if path.exists():
        if read(path)!=result:raise ValueError('Changed diagnostic split')
        return result
    write_json(path,result)
    write_metric_rows(counts,output(c)/'analyses/coverage-v1',family=FAMILY,name='partitions')
    return result


def bind_model(c,spec):
    tech=resolve_path(c['campaign'])/'analyses'/spec['name']/'technical'
    done=read(tech/'complete.json')
    checkpoint=tech/'best.pt'
    if sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('Changed selected checkpoint')
    bound=dict(spec,checkpoint=str(checkpoint),sha256=done['checkpoint_sha256'],identity=done['identity'])
    path=output(c)/'technical'/('model-'+spec['name']+'.json')
    if path.exists() and read(path)!=bound:raise ValueError('Selected checkpoint binding changed')
    write_json(path,bound)
    metric_root=Path(done['metrics_root']) if 'metrics_root' in done else tech.parent
    return bound,metric_root/'technical/future-predictions.npz'


def evaluate_model(c,p,rows,partition,spec,deadline,progress):
    model,audit=bind_model(c,spec)
    inference=Path(__file__).with_name('birth_infer.py')
    metadata=dict(plan=p['identity'],model=model,inference=sha(inference))
    pool=RetainedCache(resolve_path(c['feature_cache']),6)
    with pool.lease(digest(metadata),deadline=deadline,metadata=metadata) as bank:
        request=dict(producer=str(resolve_path(c['campaign'])/'technical/code'),dependencies=p['dependencies'],model=model,
            positions=str(cache(c)/'positions.npy'),velocities=str(cache(c)/'velocities.npy'),stored_velocity_units='angstrom_per_fs',
            destination=str(bank),batch_size=c['inference_batch_size'],leads_ps=c['leads_ps'],seed=c['seed']+111,
            audit_dataset=str(resolve_path(c['training_cache'])/'dataset.pt'),audit_predictions=str(audit))
        req=output(c)/'technical'/('inference-'+model['name']+'.json');write_json(req,request)
        if not (bank/'complete.json').exists():
            progress.update(stage='inference',model=model['name'])
            subprocess.run([sys.executable,'-u',str(inference),str(req)],check=True,timeout=max(1,deadline-time.time()))
        done=read(bank/'complete.json')
        if any(sha(bank/f)!=h for f,h in done['files'].items()):raise ValueError('Changed feature cache')
        write_json(output(c)/'technical'/('features-'+model['name']+'.json'),dict(path=str(bank),**done))
        for space in c['spaces']:
            features=np.load(bank/(space+'.npy'))
            for j,lead in enumerate(c['leads_ps']):
                x=features[rows['indices'][:,j]]
                if space=='future_law':x=x[:,j]
                for kind in c['readouts']:
                    progress.update(stage='readouts',model=model['name'],space=space,lead_ps=lead,readout=kind)
                    fit(c,p,rows,x,model,space,lead,kind,plan=partition)
    write_json(output(c)/'technical'/('evaluated-'+model['name']+'.json'),dict(checkpoint=model,finished_unix=time.time()))


def analyze(c,p,rows):
    mask=rows['role']!='train';held={k:v[mask] for k,v in rows.items()};y,w=held['label'],held['weight']
    ancestors,slot,counts=source_draws(held['ancestor'],c['bootstrap_draws'],c['seed'])
    bw=counts[:,slot]*w[None];groups=defaultdict(list);individual=[];receipts=[]
    for f in sorted((output(c)/'analyses/fits-v1').glob('*/*/*/*/technical/complete.json')):
        done=read(f);spec=done['spec'];path=f.parent/'predictions.npz'
        if sha(path)!=done['predictions_sha256']:raise ValueError('Changed readout predictions')
        with np.load(path) as z:
            np.testing.assert_array_equal(z['id'],rows['id']);prob=z['probability'][mask]
        vals=scores(y,prob,w);boot,_=bootstrap_scores(y,prob,w,slot,counts,None)
        for name,values in zip(['within_pair_auc','within_pair_top1'],matched(held,prob)):
            vals[name]=float(np.average(values,weights=w));boot[name]=bw@values/bw.sum(1)
        model=spec['model'];key=(model['arm'],spec['space'],spec['lead_ps'],spec['readout'])
        groups[key].append((vals,boot));receipts.append(dict(path=str(path),sha256=done['predictions_sha256']))
        for metric in METRICS:
            lo,hi=np.quantile(boot[metric],[.025,.975])
            individual.append(dict(model=model['name'],arm=model['arm'],space=spec['space'],lead_ps=spec['lead_ps'],
                readout=spec['readout'],metric=metric,value=vals[metric],lo=float(lo),hi=float(hi)))
    expected=len(p['models'])*len(c['spaces'])*len(c['leads_ps'])*len(c['readouts'])+len(c['leads_ps'])*(len(c['readouts'])+1)
    if len(receipts)!=expected:raise ValueError(f'Incomplete readouts {len(receipts)}/{expected}')
    summary=[];means={}
    for key,records in groups.items():
        for metric in METRICS:
            values=np.array([r[0][metric] for r in records]);bs=np.mean([r[1][metric] for r in records],0)
            lo,hi=np.quantile(bs,[.025,.975]);means[key+(metric,)]=(float(values.mean()),bs)
            summary.append(dict(arm=key[0],space=key[1],lead_ps=key[2],readout=key[3],metric=metric,
                value=float(values.mean()),lo=float(lo),hi=float(hi),seed_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                fitted_seeds=len(values),test_events=len(set(held['event'])),test_ancestors=len(ancestors),test_rows=len(y)))
    contrasts=[]
    for space in c['spaces']:
        for lead in c['leads_ps']:
            for kind in c['readouts']:
                for left,right in [('continuous','discrete'),('derivative','continuous'),('derivative','discrete')]:
                    for metric in METRICS:
                        a,b=[means[(arm,space,lead,kind,metric)] for arm in (left,right)]
                        lo,hi=np.quantile(a[1]-b[1],[.025,.975])
                        contrasts.append(dict(left=left,right=right,space=space,lead_ps=lead,readout=kind,
                            metric=metric,difference=a[0]-b[0],lo=float(lo),hi=float(hi)))
    out=output(c)/'analyses/comparison-v1'
    for name,values in [('models',individual),('summary',summary),('paired-comparisons',contrasts)]:
        write_metric_rows(values,out,family=FAMILY,name=name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric in zip(axes,['ap','nll','brier']):
        for arm in ['discrete','continuous','derivative','rich']:
            rr=sorted([r for r in summary if r['arm']==arm and r['space']==('control' if arm=='rich' else 'structural_motion') and r['readout']=='catboost' and r['metric']==metric],key=lambda r:r['lead_ps'])
            ax.plot([r['lead_ps'] for r in rr],[r['value'] for r in rr],'o-',label=arm)
        ax.set(xlabel='Lead before nucleus emergence (ps)',ylabel=metric)
    axes[0].legend(fontsize=8);fig.suptitle(f'{len(set(held["event"]))} pooled-test births, {len(ancestors)} ancestors')
    (out/'plots').mkdir(exist_ok=True);fig.savefig(out/'plots/birth-scores.png',dpi=180);plt.close(fig)
    lines=['# Continuous-motion nucleus-birth transfer','',f'{len(set(held["event"]))} pooled-test births from {len(ancestors)} ancestors.',
        'Exact1/3/6ps pre-emergence observations; frozen encoders; NLL-selected training-only readouts. AP is diagnostic.',
        'Mean per-seed scores, not an ensemble. Event-enriched1:4 assay; chance AP0.20. Bootstrap resamples whole melt ancestors.',
        '', '| Arm | Space | Lead ps | CatBoost AP | NLL |','|---|---|---:|---:|---:|']
    for key in sorted(groups):
        if key[3]=='catboost':lines.append(f'| {key[0]} | {key[1]} | {key[2]:g} | {means[key+("ap",)][0]:.4f} | {means[key+("nll",)][0]:.4f} |')
    lines+=['','[Scores](tables/summary.csv) · [Paired contrasts](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    from src.experiment_runner.wandb_tracking import update_recorded_summary
    for model in p['models']:
        values={f'birth_136_v1/{r["space"]}/{r["lead_ps"]}ps/{r["readout"]}/{r["metric"]}':r['value'] for r in individual if r['model']==model['name']}
        update_recorded_summary(resolve_path(c['campaign'])/'analyses'/model['name']/'technical/wandb.json',values,
            evaluation='continuous-motion-birth-136-v1',expected=dict(entity='teshbek',project='PointCloudMaterials'))
    write_json(output(c)/'technical/complete.json',dict(state='complete',fits=len(receipts),predictions=receipts,
        test_events=len(set(held['event'])),test_ancestors=len(ancestors),new_wandb_runs=0,finished_unix=time.time()))


def run(c):
    p=read(output(c)/'technical/plan.json');deadline=read(output(c)/'technical/launch.json')['deadline_unix']
    with recorded_stage(output(c)/'technical/coordinator.json',stage='preparation',owner=OWNER) as progress:
        if not (cache(c)/'complete.json').exists():
            with ProcessPoolExecutor(max_workers=c['prepare_workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
                futures=[executor.submit(source,c,p,item) for item in p['sources']]
                for i,future in enumerate(as_completed(futures)):
                    done=future.result();progress.update(completed_sources=i+1,total_sources=len(futures),last_source=done['source'])
                    print('prepared',done['source'],done['accepted_events'],flush=True)
            rows=seal(c,p,metric_family=FAMILY)
        else:rows=dataset(c)
        partition=readout_plan(c,rows,p)
        pending=list(p['models'])
        while pending:
            if time.time()>deadline-300:raise TimeoutError('Evaluation deadline; completed readouts retained')
            ready=[m for m in pending if (resolve_path(c['campaign'])/'analyses'/m['name']/'technical/complete.json').exists()]
            if not ready:
                progress.update(stage='waiting_for_final_encoder',models=[m['name'] for m in pending]);time.sleep(20);continue
            for model in ready:
                evaluate_model(c,p,rows,partition,model,deadline,progress);pending.remove(model)
        descriptors=np.load(cache(c)/'descriptors.npy')
        for j,lead in enumerate(c['leads_ps']):
            for kind in c['readouts']:
                fit(c,p,rows,descriptors[rows['indices'][:,j]],dict(name='rich',arm='rich',seed=c['seed'],velocities=False),'control',lead,kind,plan=partition)
            fit(c,p,rows,None,dict(name='prior',arm='prior',seed=c['seed'],velocities=False),'control',lead,'prior',plan=partition)
        progress.update(stage='analysis');analyze(c,p,rows)


def launch(c):
    prepare(c);tech=output(c)/'technical'
    if (tech/'launch.json').exists():raise FileExistsError('Already launched')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics'))
    started=time.time();deadline=min(started+c['wall_seconds'],allocation_deadline(reserve_seconds=300))
    if deadline-started<1800:raise TimeoutError('Not enough allocation time for evaluation')
    command=[sys.executable,'-u','-m',MODULE,'run','--config',str(bundle.config_path)]
    receipt=dict(owner=OWNER,allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,gpu=c['gpu'],
        started_unix=started,deadline_unix=deadline,deadline_Paris=datetime.fromtimestamp(deadline,ZoneInfo('Europe/Paris')).isoformat(),
        code=str(bundle.root),command=command,automatic_requeue=False)
    write_json(tech/'launch.json',receipt)
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),CUDA_VISIBLE_DEVICES=str(c['gpu']),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1')
    with (tech/'coordinator.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(pid=child.pid,process_group=child.pid);write_json(tech/'launch.json',receipt)
    print(receipt,flush=True)


def resume(c):
    """New execution bundle for an audit-only fix; original deadline and fits."""
    import shutil
    tech=output(c)/'technical';old=read(tech/'launch.json')
    if old['owner']!=OWNER:raise ValueError('Evaluation is not owned by this chat')
    previous=read(tech/'coordinator.json')
    if previous['state']!='failed':raise ValueError('Resume requires recorded failed evaluation')
    if c!=read(Path(old['code'])/'config.json'):raise ValueError('Resume changed scientific configuration')
    if read(tech/'plan.json')['config']!=c:raise ValueError('Cohort recipe changed')
    process=Path('/proc')/str(old['pid'])/'cmdline'
    if process.exists() and MODULE.encode() in process.read_bytes():raise RuntimeError('Recorded evaluation still running')
    deadline=min(old['deadline_unix'],allocation_deadline(reserve_seconds=300))
    if deadline<time.time()+900:raise TimeoutError('Original evaluation deadline nearly reached')
    attempt=tech/'resume-v2';attempt.mkdir(exist_ok=False)
    for name in ('coordinator.json','coordinator.log','launch.json'):
        shutil.copy2(tech/name,attempt/('previous-'+name))
    bundle=ExecutionBundle.freeze(REPO,attempt/'code',c,directories=('src','docs/metrics'))
    command=[sys.executable,'-u','-m',MODULE,'run','--config',str(bundle.config_path)]
    receipt=dict(owner=OWNER,allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,gpu=c['gpu'],
        started_unix=time.time(),deadline_unix=deadline,code=str(bundle.root),command=command,
        reason='Audit-only portable Philox prefix replay; scientific inputs, features, readouts and original deadline unchanged',
        cohort_identity=read(tech/'plan.json')['identity'],previous_launch_sha256=sha(tech/'launch.json'))
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),CUDA_VISIBLE_DEVICES=str(c['gpu']),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1')
    with (attempt/'coordinator.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(pid=child.pid,process_group=child.pid);write_json(attempt/'launch.json',receipt)
    print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','launch','resume','run','analyze'])
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(args.config)
    check_metric_docs(family=FAMILY)
    def stop(signum,frame):raise TimeoutError(f'Birth evaluation deadline/signal {signum}; results preserved')
    signal.signal(signal.SIGTERM,stop)
    if args.stage=='run':
        signal.signal(signal.SIGALRM,stop)
        signal.setitimer(signal.ITIMER_REAL,max(.1,read(output(c)/'technical/launch.json')['deadline_unix']-time.time()))
    if args.stage=='analyze':analyze(c,read(output(c)/'technical/plan.json'),dataset(c))
    else:globals()[args.stage](c)


if __name__=='__main__':main()
