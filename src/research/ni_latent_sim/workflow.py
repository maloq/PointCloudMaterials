"""Two local GPU lanes, frozen provenance and an absolute 8.5-hour budget."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from src.experiment_runner.execution import ExecutionBundle,allocation_deadline
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO
from .data import read,root,cache,prepare,collect,seal,write_json,resolve_path,FAMILY


def barrier(c,name,deadline):
    out=root(c)/'technical'
    def completed(lane):
        path=out/f'{name}-{lane}.json'
        if not path.exists():return False
        if name=='dense-complete':
            launch=read(out/'launch.json')
            return read(path).get('execution_started')==launch.get('resumed_at',launch['started'])
        return True
    while not all(completed(i) for i in [0,1]):
        for lane in [0,1]:
            if (out/f'failed-worker-{lane}.json').exists():raise RuntimeError('Other lane failed; see failure receipt')
            if (out/f'budget-stop-{lane}.json').exists():raise TimeoutError('Other lane reached its budget')
        if time.time()>deadline-600:raise TimeoutError('Budget reached at '+name)
        time.sleep(10)


def worker(c,lane):
    if 'dense_response' in c:
        from .resilience import worker as resilient_worker
        return resilient_worker(c,lane)
    import torch
    from . import train,evaluate,lifting
    torch.set_num_threads(2);out=root(c)/'technical';p=read(out/'plan.json');deadline=read(out/'launch.json')['deadline_unix']
    with (out/f'lane-{lane}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if 'after_output' in c['runtime']:
            previous=resolve_path(c['runtime']['after_output'])/'technical'
            write_json(out/f'worker-{lane}.json',dict(stage='waiting_for_previous_gpu_lane',after=str(previous),lane=lane))
            while not (previous/f'worker-complete-{lane}.json').exists():
                if any((previous/f'{name}-{lane}.json').exists() for name in ['failed-worker','budget-stop']):
                    raise RuntimeError('Preceding GPU lane did not complete; see '+str(previous))
                if time.time()>deadline-600:raise TimeoutError('Budget reached waiting for previous GPU lane')
                time.sleep(10)
        if 'dense_response' in c:
            from . import dense_response
            dense_response.collect(c,p,lane,deadline)
            if 'reuse_output' not in c['dense_response']:barrier(c,'dense-complete',deadline)
        elif 'enrichment' in c:
            from . import enrichment
            enrichment.scan(c,p,lane,deadline);barrier(c,'scan-complete',deadline)
            with (out/'selection.lock').open('a+') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX);enrichment.select(c,p)
            enrichment.encode(c,p,lane,deadline);barrier(c,'data-complete',deadline)
            with (out/'seal.lock').open('a+') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX);enrichment.seal(c,p)
        elif 'reuse_data' not in c:
            collect(c,p,lane,deadline);barrier(c,'data-complete',deadline)
            with (out/'seal.lock').open('a+') as shared:
                fcntl.flock(shared,fcntl.LOCK_EX);seal(c,p)
        for arm in c['arms'][lane::2]:
            write_json(out/f'worker-{lane}.json',dict(stage='training',arm=arm['name'],deadline_unix=deadline))
            if not train.run(c,p,arm,deadline):raise TimeoutError('Training checkpointed at budget')
            evaluate.run(c,p,arm);evaluate.births(c,p,arm)
            if 'dense_response' in c:dense_response.evaluate(c,p,arm)
        # Two declared primary models, one per lane; never selected by AP.
        if c['lifting'].get('enabled',True):
            arm=next(a for a in c['arms'] if a['dimension']==16 and a['history']==4 and a['seed']==c['seed'] and a['response']==bool(lane))
            write_json(out/f'worker-{lane}.json',dict(stage='physical_lifting',arm=arm['name']))
            lifting.run(c,p,arm,deadline)
        write_json(out/f'worker-complete-{lane}.json',dict(state='complete',time=time.time()))
        with (out/'report.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if all((out/f'worker-complete-{i}.json').exists() for i in [0,1]):
                if 'dense_response' in c:from .dense_response import report as run
                elif 'stability' in c:from .stability import report as run
                else:from .report import run
                run(c)


def submit(c):
    check_metric_docs(family=FAMILY);p=prepare_run(c);out=root(c)/'technical'
    if (out/'launch.json').exists():raise ValueError('Already launched; use receipt')
    deadline=min(time.time()+c['runtime']['hours']*3600,allocation_deadline(reserve_seconds=300))
    bundle=ExecutionBundle.freeze(REPO,out/'code',c,directories=['src','configs','docs/metrics'],files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    env=dict(os.environ,PYTHONPATH=str(bundle.root),PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',UCX_TLS='self,sm,tcp',CUDA_HOME='/usr/local/cuda',PATH=str(Path(sys.executable).parent)+':/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
    receipt=dict(identity=p['identity'],allocation=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,bundle=str(bundle.root),started=time.time(),deadline_unix=deadline,pids=[],gpus=[0,1],owner='current chat: '+c['protocol'],budget_hours=c['runtime']['hours'])
    write_json(out/'launch.json',receipt)
    for lane in [0,1]:
        with (out/f'worker-{lane}.log').open('a') as log:
            stage='supervise' if 'dense_response' in c else 'worker'
            command=['timeout','--signal=TERM','--kill-after=60',str(max(1,int(deadline-time.time())))+'s',sys.executable,'-u','-m','src.research.ni_latent_sim.workflow',stage,'--config',str(bundle.config_path),'--lane',str(lane)]
            proc=subprocess.Popen(command,cwd=bundle.root,env=dict(env,CUDA_VISIBLE_DEVICES=str(lane)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['pids'].append(proc.pid);write_json(out/'launch.json',receipt)
    return receipt


def prepare_run(c):
    if 'dense_response' in c:
        from .dense_response import prepare as dense
        return dense(c)
    if 'enrichment' in c:
        from .enrichment import prepare as expanded
        return expanded(c)
    if 'reuse_data' in c:
        from .stability import prepare as reuse
        return reuse(c)
    return prepare(c)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','submit','worker','supervise']);ap.add_argument('--config',required=True);ap.add_argument('--lane',type=int,default=0);ap.add_argument('--adopt-pid',type=int);a=ap.parse_args();c=read(a.config)
    try:
        if a.stage=='prepare':print(prepare_run(c)['identity'])
        elif a.stage=='submit':print(submit(c))
        elif a.stage=='supervise':
            from .resilience import supervise
            supervise(c,a.lane,a.adopt_pid)
        else:worker(c,a.lane)
    except TimeoutError as error:
        write_json(root(c)/'technical'/f'budget-stop-{a.lane}.json',dict(state='budget_reached',error=str(error),time=time.time()))
    except Exception:
        write_json(root(c)/'technical'/f'failed-{a.stage}-{a.lane}.json',dict(error=traceback.format_exc(),time=time.time()));raise


if __name__=='__main__':main()
