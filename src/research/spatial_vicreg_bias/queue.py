"""Freeze the scientific producer and submit preparation, matched fits and assays."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from src.research.structural_state.common import write_json,sha,digest
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from .data import load,freeze,source,seal


def submit(config):
    c=load(config);root=Path(c['output']);technical=root/'technical/queue'
    launch=technical/'launch.json'
    if launch.exists():raise FileExistsError(f'Queue already submitted: {launch}')
    technical.mkdir(parents=True,exist_ok=True)
    code=technical/'code';repo=Path(__file__).resolve().parents[3]
    # Resolve storage paths at submission, while binding the architecture to the frozen copy.
    c['encoder_recipe']=str(code/'configs/spatial_vicreg_bias/geoframe_plain.yaml')
    bundle=ExecutionBundle.freeze(repo,code,c,directories=('src','configs','docs/metrics'),
        files=((repo/'machine.local.yaml','machine.local.yaml'),))
    cfg=bundle.config_path
    _,plan=freeze(cfg)
    ptm=Path(c['ptm_audit'])/'technical/extraction-contract.json'
    if sha(ptm)!=c['ptm_contract_sha256']:raise ValueError('PTM extraction definition changed')
    contract=json.loads(ptm.read_text())
    if contract['release_identity']!=c['fixed_identity'] or contract['ptm']!={'rmsd_cutoff':.1,'crystal_types':[1,2,3],'chunk_frames':32}:
        raise ValueError('Wrong physical reference protocol')
    for item in plan['sources']:
        record=json.loads((Path(c['ptm_audit'])/f'technical/sources/{item["id"]}/ptm-complete.json').read_text())
        if record['identity']!=digest(dict(contract=contract,source=item)) or record['center_mismatches']:
            raise ValueError(f'Physical reference identity/center mismatch for source {item["id"]}')
    receipt=dict(state='submitting',config=str(cfg),code=str(code),data_identity=plan['identity'],jobs={},stages={})
    write_json(launch,receipt)
    env=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        OVITO_THREAD_COUNT='1',QT_QPA_PLATFORM='offscreen',TORCHINDUCTOR_COMPILE_THREADS='1')
    queue=SlurmQueue(technical,bundle,'src.research.spatial_vicreg_bias.queue',env,launch,receipt,'SVB')
    def job(name,stage,*,index=0,gpu=False,dependency=None,hours=23):
        options=['--cpus-per-task='+('2' if gpu else str(c['execution']['preparation_lanes'])),
            '--mem=20G',f'--time={hours:02d}:00:00']
        if gpu:options+=['--gres=gpu:1','--exclude=node58','--requeue','--open-mode=append']
        receipt['stages'][name]=dict(stage=stage,index=index,dependency=dependency)
        return queue.submit(name,options,dependency,partition='RTX6000PRO' if gpu else 'CPU',
            command_stage='worker',arguments=('--stage',stage,'--index',str(index)))
    with queue.submission():
        prepared=job('prepare','prepare',hours=6)
        for lane in range(c['execution']['gpu_lanes']):
            job(f'lane{lane}','lane',index=lane,gpu=True,dependency='afterok:'+prepared)
    receipt['state']='submitted';write_json(launch,receipt);print(json.dumps(receipt,indent=2))


def worker(args):
    c=load(args.config);stage=args.stage
    try:
        if stage=='prepare':
            _,plan=freeze(args.config)
            with ProcessPoolExecutor(max_workers=c['execution']['preparation_lanes']) as pool:
                futures=[pool.submit(source,args.config,item['id']) for item in plan['sources']]
                for future in as_completed(futures):future.result()
            seal(args.config)
        elif stage=='seal':seal(args.config)
        elif stage=='smoke':
            from .train import smoke
            smoke(args.config)
        elif stage=='train':
            from .train import fit
            done=fit(args.config,args.index)
            if not done:raise TimeoutError('Training requires continuation; use the lane worker')
        elif stage=='evaluate':
            from .train import settings
            from .evaluate import run
            study=settings(args.config,args.index)[-1]
            if not (study.root/'technical/complete.json').exists():raise ValueError('Training is incomplete')
            run(args.config,args.index)
        elif stage=='nulls':
            from .evaluate import nulls
            nulls(args.config)
        elif stage=='report':
            from .report import run
            run(args.config)
        elif stage=='lane':
            from .train import fit,smoke
            from .evaluate import run,nulls
            from .report import run as report
            from src.training_methods.shared_pretraining.queue import deadline_for_job
            deadline=deadline_for_job()
            # The first scheduled lane performs the final-code smoke. Requiring
            # lane 0 specifically could occupy every GPU while lane 0 is pending.
            gate=Path(c['output'])/'technical/production-smoke.json'
            with gate.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not gate.exists():
                    smoke(args.config)
                    write_json(gate,dict(state='complete',train_sha256=sha(Path(__file__).with_name('train.py'))))
                if json.loads(gate.read_text())['train_sha256']!=sha(Path(__file__).with_name('train.py')):
                    raise ValueError('Production preflight does not match this producer')
            completed=True
            for index in range(args.index,9,c['execution']['gpu_lanes']):
                if time.time()>deadline or not fit(args.config,index) or not run(args.config,index,deadline=deadline):
                    completed=False;break
            if completed and args.index==2:completed=nulls(args.config,deadline=deadline)
            with (Path(c['output'])/'technical/report.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);report(args.config)
            if not completed:
                write_json(Path(c['output'])/'technical'/f'lane{args.index}-continuation.json',
                    dict(job=os.environ['SLURM_JOB_ID'],state='requeue',time=time.time()))
                subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
            else:
                write_json(Path(c['output'])/'technical'/f'lane{args.index}-complete.json',dict(state='complete'))
    except BaseException as error:
        write_json(Path(c['output'])/'technical/failures'/f'{os.environ["SLURM_JOB_ID"]}-{stage}-{args.index}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),stage=stage,index=args.index))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['submit','worker'])
    p.add_argument('--config',required=True);p.add_argument('--stage');p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.action=='submit':submit(a.config)
    else:worker(a)
