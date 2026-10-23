"""Detached overnight workers; each preparation, fit and evaluation resumes independently."""
import argparse,fcntl,math,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.ni_phase_sim.campaign import environment
from src.experiment_runner.stages import run_stage
from .data import prepare,read,root,sha,write_json,FAMILY
MODULE='src.research.ni_ordering_flow.queue'
REPO=Path(__file__).resolve().parents[3]


def launch(c):
    tech=root(c)/'technical'
    if (tech/'submission.json').exists():raise FileExistsError('Queue already launched')
    p=prepare(c);check_metric_docs(family=FAMILY)
    bundle=ExecutionBundle.freeze(REPO,tech/'source',c,directories=['src','configs','docs/metrics'],files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    receipt=dict(started=time.time(),deadline=time.time()+c['runtime']['hours']*3600,jobs={},workers=[],source=str(bundle.root),origin_allocation=os.environ.get('SLURM_JOB_ID'),independent_of_chat=True)
    write_json(tech/'submission.json',receipt)
    for lane,device in enumerate(c['runtime']['local_devices']):
        log=tech/f'worker-{lane}.log';command=[sys.executable,'-u','-m',MODULE,'worker','--config',str(bundle.config_path),'--lane',str(lane)]
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=bundle.root,env=dict(os.environ,**environment(bundle),CUDA_VISIBLE_DEVICES=str(device)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['workers'].append(dict(lane=lane,pid=process.pid,device=device,log=str(log)))
        write_json(tech/'submission.json',receipt)
    queue=SlurmQueue(tech,bundle,MODULE,environment(bundle),tech/'submission.json',receipt,'ni-order')
    with queue.submission():
        for lane in range(len(c['runtime']['local_devices']),len(c['runtime']['local_devices'])+c['runtime']['remote_workers']):
            queue.submit(f'worker-{lane}',['--gres=gpu:1','--cpus-per-task=6','--mem=48G',f'--time={c["runtime"]["hours"]}:00:00'],partition=c['runtime']['partitions'],command_stage='worker',arguments=['--lane',str(lane)])
    print(receipt,flush=True)


def tasks(c,p):
    assembled=root(c)/'analyses/assemble/technical/complete.json'
    # Baseline fits do not wait for feature collection or any chat review.
    for arm in c['arms']:
        if arm['treatment']=='local4':yield arm,'train','complete.json',True
    for job in p['jobs']:yield dict(name='collect-'+job['key']),'collect','complete.json',True
    ready=all((root(c)/'analyses'/('collect-'+j['key'])/'technical/complete.json').exists() for j in p['jobs'])
    yield dict(name='assemble'),'assemble','complete.json',ready
    for arm in c['arms']:
        if arm['treatment']!='local4':yield arm,'train','complete.json',assembled.exists()
    for arm in c['arms']:
        trained=(root(c)/'analyses'/arm['name']/'technical/complete.json').exists()
        if arm['teacher']!='none':yield arm,'response','response-complete.json',trained
        yield arm,'evaluate','evaluation-complete.json',trained
        yield arm,'rare','rare-complete.json',trained
        yield arm,'births','births-complete.json',trained and assembled.exists()


def worker(c,bundle,lane):
    tech=root(c)/'technical';global_deadline=read(tech/'submission.json')['deadline']
    deadline=min(global_deadline,allocation_deadline(reserve_seconds=180));p=read(tech/'plan.json')
    write_json(tech/f'worker-start-{lane}.json',dict(started=time.time(),allocation=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),deadline=deadline))
    while time.time()<deadline-600:
        progress=False;pending=False;paused=False
        for arm,name,marker,ready in tasks(c,p):
            folder=root(c)/'analyses'/arm['name']/'technical'
            if (folder/marker).exists() or (folder/(name+'-failed.json')).exists():continue
            pending=True
            if not ready:continue
            result = run_stage(
                root(c), bundle, lane, deadline, arm['name'], name, marker,
                module=MODULE, hard_deadline=True, max_attempts=c['runtime']['max_attempts'])
            if result=='paused':paused=True;break
            if result is not None:progress=True;break
        if paused:break
        if progress:continue
        if not pending:break
        # Dependency failure must be visible, and must not spin forever.
        busy=False
        for path in (root(c)/'analyses').glob('*/technical/worker.lock'):
            with path.open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:busy=True;break
        if not busy:break
        time.sleep(15)
    pending=[dict(name=a['name'],stage=s,ready=r) for a,s,m,r in tasks(c,p) if not (root(c)/'analyses'/a['name']/'technical'/m).exists()]
    write_json(tech/f'worker-exit-{lane}-{time.time_ns()}.json',dict(pending=pending,finished=time.time()))
    if pending and deadline<global_deadline-600 and time.time()>=deadline-600:
        receipt=dict(jobs={},deadline=global_deadline);queue=SlurmQueue(tech,bundle,MODULE,environment(bundle),tech/f'continuation-{lane}.json',receipt,'ni-order')
        with queue.submission():queue.submit(f'continue-{lane}',['--gres=gpu:1','--cpus-per-task=6','--mem=48G',f'--time={math.ceil((global_deadline-time.time())/60)}'],partition=c['runtime']['partitions'],command_stage='worker',arguments=['--lane',str(lane)])
    with (tech/'comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            from .assessment import compare
            compare(c)
        except Exception:write_json(tech/f'comparison-failed-{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        write_json(tech/'queue-status.json',dict(state='complete' if not pending else 'partial',pending=pending,updated=time.time()))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','launch','worker','collect','assemble','train','evaluate','rare','response','births','compare'])
    parser.add_argument('--config',required=True);parser.add_argument('--name');parser.add_argument('--lane',type=int,default=0);parser.add_argument('--deadline',type=float,default=float('inf'))
    args=parser.parse_args();c=read(args.config);torch.set_num_threads(1)
    if args.stage=='prepare':prepare(c);return
    if args.stage=='launch':launch(c);return
    if args.stage=='worker':worker(c,ExecutionBundle(Path(args.config).resolve().parent),args.lane);return
    if args.stage=='collect':
        from .data import collect_source
        done=collect_source(c,args.name.removeprefix('collect-'),args.deadline)
    elif args.stage=='assemble':
        from .data import assemble
        done=assemble(c)
    elif args.stage=='compare':
        from .assessment import compare
        compare(c);return
    else:
        arm=next(a for a in c['arms'] if a['name']==args.name)
        if args.stage=='train':
            from .learning import fit
            done=fit(c,arm,args.deadline)
        else:
            from . import assessment
            done=assessment.evaluate(c,arm,args.deadline,rare=args.stage=='rare') if args.stage in ('evaluate','rare') else getattr(assessment,args.stage)(c,arm,args.deadline)
    if not done:sys.exit(75)

if __name__=='__main__':main()
