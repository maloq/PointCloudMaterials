"""Frozen-code detached two-H100 queue with exact resumes and validation-only promotions."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.training_methods.shared_pretraining.queue import snapshot,deadline_for_job
from src.experiment_runner.metric_docs import check_metric_docs
from .variants import screens,promotions


def tasks(config,root):
    base=screens(config);path=root/'promotions.json'
    with (root/'promotion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return base+json.loads(path.read_text())['tasks']
        status={}
        for spec in base:
            p=root/'runs'/spec['name']/'status.json'
            if not p.exists():return base
            record=json.loads(p.read_text())
            if record['state']=='failed':raise RuntimeError(f'Failed screen: {spec["name"]}')
            if record['state']!='complete':return base
            status[spec['name']]=record
        valid=[s for s in base if status[s['name']]['learned']]
        if not any(s['prediction']=='none' for s in valid) or sum(s['prediction']!='none' for s in valid)<2:
            raise RuntimeError('Insufficient learned controls/predictive settings for promotion; inspect fixed-target validation')
        longer=promotions(valid,status,config['long_epochs'])
        save_json(path,dict(criterion='Training-mean baseline must be beaten; lowest selection physical + 0.25 instantaneous TDA; no latent or test ranking',tasks=longer))
        save_json(root/'queue.json',base+longer);return base+longer


def worker(config_path,lane):
    from .runtime import run
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output'])/'technical';deadline=deadline_for_job()
    def state(value,**extra):save_json(root/f'lane-{lane}.json',dict(state=value,pid=os.getpid(),updated_at=time.time(),**extra))
    try:
        while time.time()<deadline-600:
            specs=tasks(config,root);unfinished=False
            for spec in specs:
                folder=root/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:unfinished=True;continue
                    path=folder/'status.json'
                    if path.exists():
                        prior=json.loads(path.read_text())
                        if prior['state']=='complete':continue
                        if prior['state']=='failed':raise RuntimeError(f'Previous failure: {spec["name"]}')
                    unfinished=True;state('training',variant=spec['name'])
                    try:done=run(config,spec,deadline)
                    except Exception as error:
                        save_json(path,dict(state='failed',error=repr(error),traceback=traceback.format_exc()));raise
                    if not done:state('checkpointed',variant=spec['name']);return
            if not unfinished:
                if len(tasks(config,root))>len(specs):continue
                state('complete');return
            time.sleep(10)
        state('checkpointed')
    except Exception as error:
        state('failed',error=repr(error),traceback=traceback.format_exc());raise


def coordinate(config_path,lane_offset=0):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output'])/'technical'
    # Slurm gives this step both allocated GPUs; split its visible list, never host-wide IDs.
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices)!=2:raise ValueError(f'Expected two allocated GPUs, got {devices}')
    processes=[]
    for idx,device in enumerate(devices):
        lane=idx+lane_offset
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        log=(root/f'lane-{lane}.log').open('a')
        process=subprocess.Popen([sys.executable,'-u','-m',__package__+'.queue','worker','--config',config_path,'--lane',str(lane)],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(process);log.close()
    codes=[p.wait() for p in processes]
    if any(codes):raise RuntimeError(f'Neighborhood workers exited: {codes}')


def submit(config_path):
    config=json.loads(Path(config_path).read_text());root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    if (root/'launch.json').exists():raise FileExistsError('Already launched; continue using its frozen code')
    check_metric_docs(family='neighborhood_jepa')
    manifest=resolve_path(config['cache'])/'manifest.json'
    if json.loads(manifest.read_text())['state']!='complete':raise ValueError('Dataset incomplete')
    save_json(root/'queue.json',screens(config));save_json(root/'data-identity.json',dict(path=str(manifest),sha256=file_hash(manifest)))
    code=snapshot(root);config_frozen=code/config_path
    env=dict(os.environ,PCM_PROJECT_ROOT=str(code),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    cpus=str(config.get('cpus_per_task',20))
    command=['srun','--jobid='+config['allocation'],'--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task='+cpus,'--gres=gpu:2',sys.executable,'-u','-m',__package__+'.queue','coordinate','--config',str(config_frozen)]
    with (root/'coordinator.log').open('a') as log:
        p=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save_json(root/'launch.json',dict(pid=p.pid,allocation=config['allocation'],command=command,code=str(code),submitted_at=time.time()))
    print(json.dumps(dict(pid=p.pid,allocation=config['allocation'],code=str(code))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['submit','coordinate','worker']);parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int);parser.add_argument('--lane-offset',type=int,default=0);args=parser.parse_args()
    if args.command=='submit':submit(args.config)
    elif args.command=='coordinate':coordinate(args.config,args.lane_offset)
    else:worker(args.config,args.lane)
