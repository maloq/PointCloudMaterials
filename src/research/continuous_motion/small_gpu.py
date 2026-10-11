"""Separate A40/A100/L40S requests for the existing scientific fit queue.

The canonical fit identity and data remain the existing campaign's. This is an
explicit execution implementation with a separately frozen source/metric hash,
global256 noncollapse gradients and sample-matched latent noise. Model/optimizer
checkpoints can therefore move between256 and128x2 execution without changing
their scientific objective or sampler stream.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.experiment_runner.metric_docs import check_metric_docs
from .data import FAMILY, read, root, resolve_path, digest, sha, write_json
from .workflow import OWNER, environment, worker

MODULE='src.research.continuous_motion.small_gpu'


def launch(profile):
    campaign=resolve_path(profile['campaign']);canonical=read(campaign/'technical/launch.json')
    c=read(Path(canonical['code'])/'config.json')
    if root(c)!=campaign or c['training']['batch_size']!=256 or c['training']['microbatch']!=256:
        raise ValueError('Small-GPU profile requires the canonical256-batch campaign')
    if profile['microbatch']!=128 or profile['partitions']!=['A40','A100','L40S']:
        raise ValueError('This execution profile requires128x2 and separate A40/A100/L40S requests')
    check_metric_docs(family=FAMILY)
    validation=read(campaign/'technical/development-micro128.json')
    if not validation['passed']:
        raise ValueError('Manual accumulation equivalence verification is missing')
    tech=campaign/'technical'/profile['name'];tech.mkdir(parents=True,exist_ok=True)
    if (tech/'launch.json').exists():
        raise FileExistsError('Small-GPU requests already submitted')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'))
    write_json(bundle.root/'profile.json',profile)
    files={str(p.relative_to(bundle.root)):sha(p) for p in (bundle.root/'src').rglob('*')
           if p.is_file() and p.suffix in ('.py','.cu','.h','.cpp')}
    files['docs/metrics/continuous_motion.md']=sha(bundle.root/'docs/metrics/continuous_motion.md')
    implementation=dict(identity=digest(files),files=files,canonical_code=canonical['code'],
        canonical_identity=read(campaign/'technical/code-identity.json')['identity'],
        verification_sha256=sha(campaign/'technical/development-micro128.json'),
        microbatch=128,effective_batch=256,accumulation_steps=2,
        scientific_change=False,scope='explicit memory execution variant; complete global covariance and identical noise samples')
    write_json(tech/'implementation.json',implementation)
    budget=read(campaign/'technical/budget.json');end=budget['deadline_unix']
    minutes=math.floor((end-time.time())/60)-10
    if minutes<120:
        raise TimeoutError('Insufficient remaining campaign allocation window')
    env=environment(c,repo)
    env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    receipt=dict(owner=OWNER,jobs={},code=str(bundle.root),campaign=str(campaign),
                 profile=profile,implementation=implementation['identity'],budget=budget)
    queue=SlurmQueue(tech,bundle,MODULE,env,tech/'launch.json',receipt,'CM128')
    with queue.submission():
        for partition in profile['partitions']:
            queue.submit(partition.lower(),['--gpus=1','--cpus-per-task=4','--mem=32G',
                f'--time={minutes}','--time-min=120','--no-requeue',
                '--deadline='+datetime.fromtimestamp(end,ZoneInfo('Europe/Paris')).strftime('%Y-%m-%dT%H:%M:%S')],
                partition=partition,command_stage='worker',arguments=('--lane',partition.lower()))
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['launch','worker']);parser.add_argument('--config',required=True)
    parser.add_argument('--lane');args=parser.parse_args()
    if args.stage=='launch':
        launch(read(args.config));return
    c=read(args.config);profile=read(Path(args.config).parent/'profile.json')
    tech=root(c)/'technical'/profile['name'];implementation=read(tech/'implementation.json')
    lane=f"small-{os.environ['SLURM_JOB_ID']}-{args.lane}"
    write_json(root(c)/'technical'/f'launch-{lane}.json',dict(owner=OWNER,host=os.uname().nodename,
        allocation=os.environ['SLURM_JOB_ID'],pid=os.getpid(),command=sys.argv,
        visible=os.environ['CUDA_VISIBLE_DEVICES'],profile_identity=implementation['identity'],
        microbatch=128,effective_batch=256,started_unix=time.time()))
    try:
        worker(c,lane,runtime_microbatch=128,execution_profile=dict(
            name=profile['name'],implementation_identity=implementation['identity'],
            implementation_receipt=str(tech/'implementation.json')))
    except TimeoutError as error:
        print(str(error),flush=True);sys.exit(124)


if __name__=='__main__':
    main()
