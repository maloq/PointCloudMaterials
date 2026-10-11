"""Detached native-cadence birth evaluation; reuse complete labels as they arrive."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import REPO, resolve_path
from src.experiment_runner.execution import ExecutionBundle, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.equivariant_context.cache import RetainedCache
from .data import FAMILY, prepare, read, output, cache, source, seal, dataset
from .readouts import fit, analyze


def lane(c,index):
    p=read(output(c)/'technical/plan.json');rows=dataset(c);pool=RetainedCache(resolve_path(c['feature_cache']),6)
    with recorded_stage(output(c)/'technical'/f'lane-{index}.json',lane=index) as progress:
        for model in p['models'][index::c['lanes']]:
            metadata=dict(plan=p['identity'],model=model,inference=sha(Path(__file__).parents[1]/'response_birth_infer.py'))
            key=digest(metadata)
            with pool.lease(key,deadline=read(output(c)/'technical/launch.json')['deadline_unix'],metadata=metadata) as bank:
                request=dict(model=model,producer=str(resolve_path(c['campaign'])/'technical/code'),dependencies=p['dependencies'],
                    positions=str(cache(c)/'positions.npy'),velocities=str(cache(c)/'velocities.npy'),destination=str(bank),
                    audit=str(cache(c)/'replay.npz'),audit_reference=str(resolve_path(c['campaign'])/'analyses'/model['name']/'technical/predictions.npz'),batch_size=c['inference_batch_size'])
                path=output(c)/'technical'/('inference-'+model['name']+'.json');write_json(path,request)
                if not (bank/'complete.json').exists():
                    subprocess.run([sys.executable,'-u',str(Path(__file__).parents[1]/'response_birth_infer.py'),str(path)],check=True)
                done=read(bank/'complete.json')
                if any(sha(bank/n)!=h for n,h in done['files'].items()):raise ValueError('Changed inferred features')
                write_json(output(c)/'technical'/('features-'+model['name']+'.json'),dict(path=str(bank),key=key,**done))
                for space in c['spaces']:
                    features=np.load(bank/(space+'.npy'))
                    for j,lead in enumerate(c['leads_ps']):
                        x=features[rows['indices'][:,j]]
                        for kind in c['readouts']:
                            progress.update(model=model['name'],space=space,lead_ps=lead,readout=kind)
                            fit(c,rows,x,model,space,lead,kind)
        if index==0:
            descriptors=np.load(cache(c)/'descriptors.npy');model=dict(name='rich',arm='rich',seed=c['seed'],velocities=False)
            for j,lead in enumerate(c['leads_ps']):
                for kind in c['readouts']:fit(c,rows,descriptors[rows['indices'][:,j]],model,'control',lead,kind)
                prior=dict(name='prior',arm='prior',seed=c['seed'],velocities=False)
                fit(c,rows,None,prior,'control',lead,'prior')


def coordinate(c):
    p=read(output(c)/'technical/plan.json');receipt=read(output(c)/'technical/launch.json');deadline=receipt['deadline_unix']
    children=[]
    with recorded_stage(output(c)/'technical/coordinator.json',stage='preparation',completed_sources=0,total_sources=len(p['sources'])) as progress:
        try:
            pending={s['key']:s for s in p['sources'] if not (cache(c)/'sources'/s['key']/'complete.json').exists()}
            with ProcessPoolExecutor(max_workers=c['prepare_workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
                running={}
                while pending or running:
                    if time.time()>deadline-1800:raise TimeoutError('Preparation reached campaign reserve; resumable sources retained')
                    for key,item in list(pending.items()):
                        if len(running)>=c['prepare_workers']:break
                        if (Path(item['label_folder'])/'complete.json').exists():
                            future=executor.submit(source,c,p,item);running[future]=key;del pending[key]
                    if running:
                        done,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                        for future in done:
                            result=future.result();del running[future];print('prepared',result['source'],result['accepted_events'],result['rows'],flush=True)
                    else:time.sleep(30)
                    waiting=sum(not (Path(s['label_folder'])/'complete.json').exists() for s in pending.values())
                    progress.update(completed_sources=len(p['sources'])-len(pending)-len(running),running_sources=len(running),waiting_for_labels=waiting,queued_preparation=len(pending)-waiting)
            while any(not (Path(r['root'])/'technical/complete.json').exists() for r in p['releases']):
                if time.time()>deadline-1800:raise TimeoutError('Waiting for final lineage seal reached reserve')
                progress.update(stage='waiting_for_lineage_collection');time.sleep(30)
            rows=seal(c,p);progress.update(stage='inference_and_readouts',rows=len(rows['id']))
            for index in range(c['lanes']):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),OMP_NUM_THREADS=str(c['fit_threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log=(output(c)/'technical'/f'lane-{index}.log').open('a')
                command=[sys.executable,'-u','-m','src.research.response_dense_birth.workflow','lane','--config',str(Path('config.json').resolve()),'--lane',str(index)]
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);children.append((child,log))
                write_json(output(c)/'technical'/f'lane-{index}-launch.json',dict(pid=child.pid,gpu=index,command=command,owner='current response/nucleus-recovery chat',allocation=os.environ.get('SLURM_JOB_ID')))
            while any(child.poll() is None for child,_ in children):
                if any(child.poll() not in (None,0) for child,_ in children):raise RuntimeError('Readout lane failed; inspect lane logs')
                time.sleep(15)
            if any(child.returncode!=0 for child,_ in children):raise RuntimeError('Readout lane failed')
            expected=len(p['models'])*len(c['spaces'])*len(c['leads_ps'])*len(c['readouts'])+len(c['leads_ps'])*(len(c['readouts'])+1)
            actual=len(list((output(c)/'analyses/fits-v1').glob('*/*/*/*/technical/complete.json')))
            if actual!=expected:raise ValueError(f'Incomplete readouts {actual}/{expected}')
            progress.update(stage='analysis',fits=actual);analyze(c)
        finally:
            for child,log in children:
                if child.poll() is None:child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                log.close()


def launch(c):
    import datetime
    from zoneinfo import ZoneInfo
    check_metric_docs(family=FAMILY);p=prepare(c);tech=output(c)/'technical'
    path=tech/'launch.json'
    if path.exists():raise FileExistsError(path)
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics'))
    started=time.time();deadline=started+c['wall_seconds']
    if c.get('deadline_reference'):
        deadline=min(deadline,read(resolve_path(c['deadline_reference']))['deadline_unix'])
    if deadline<=started+1800:raise TimeoutError('Insufficient time remaining for this scope revision')
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OVITO_THREAD_COUNT='1')
    command=[sys.executable,'-u','-m','src.research.response_dense_birth.workflow','coordinate','--config',str(bundle.config_path)]
    # Publish the deadline before the child can read it; then add exact ownership.
    receipt=dict(owner='current response/nucleus-recovery chat',allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,
        started_unix=started,deadline_unix=deadline,deadline_Paris=datetime.datetime.fromtimestamp(deadline,ZoneInfo('Europe/Paris')).isoformat(),
        code=str(bundle.root),command=command,automatic_requeue=False)
    write_json(path,receipt)
    with (tech/'coordinator.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(pid=child.pid,process_group=child.pid);write_json(path,receipt);print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','launch','coordinate','lane','analyze'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int,default=0);args=parser.parse_args();c=read(args.config)
    check_metric_docs(family=FAMILY)
    def stop(signum,frame):raise TimeoutError(f'Campaign deadline/signal {signum}; artifacts retained')
    signal.signal(signal.SIGTERM,stop)
    if args.stage in ['coordinate','lane']:
        signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,max(.1,read(output(c)/'technical/launch.json')['deadline_unix']-time.time()))
    if args.stage=='lane':lane(c,args.lane)
    else:globals()[args.stage](c)

if __name__=='__main__':main()
