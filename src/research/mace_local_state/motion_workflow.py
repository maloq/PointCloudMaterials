"""Run the consecutive-motion protocol within an explicitly selected allocation."""
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from src.experiment_runner.tracking import tracked_run
from .motion_data import AllocationEnding, check_deadline, plan, prepare, assemble
from .motion_train import fit
from .motion_evaluate import evaluate


def workers(config, config_path, root, stage):
    children = [];streams = []
    try:
        for lane in range(len(config['devices'])):
            stream = (root/f'technical/{stage}-lane{lane}.log').open('a',buffering=1)
            command = [sys.executable,'-u','-m','src.research.mace_local_state.run',
                '--config',str(config_path),'--stage','motion-'+stage,'--lane',str(lane)]
            env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            children.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True))
            streams.append(stream)
        write_json(root/f'technical/{stage}-workers.json',dict(pids=[p.pid for p in children],stage=stage))
        while any(p.poll() is None for p in children):
            for process in children:
                code = process.poll()
                if code not in (None,0):
                    if code==75: raise AllocationEnding(f'{stage} paused at saved unit before deadline')
                    raise RuntimeError(f'{stage} worker PID {process.pid} exited with {code}; inspect lane log')
            time.sleep(2)
        for process in children:
            if process.returncode != 0:
                if process.returncode==75: raise AllocationEnding(f'{stage} paused before deadline')
                raise RuntimeError(f'{stage} worker exited with {process.returncode}')
    finally:
        for process in children:
            if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
        for process in children: process.wait()
        for stream in streams: stream.close()


def run(args, config, root):
    if config['protocol']!='mace_local_motion_v1': raise ValueError('Motion stages require their separate scientific protocol')
    runtime = config['runtime']
    if socket.gethostname()!=runtime['node'] or os.environ.get('SLURM_JOB_ID')!=str(runtime['job_id']):
        raise RuntimeError('Node/allocation differs from runtime recipe; update runtime explicitly before starting')
    check_deadline(config)
    if args.stage in ('motion-prepare','motion-fit'):
        if args.lane is None or args.lane not in range(len(config['devices'])): raise ValueError('Worker needs a valid --lane')
        try:
            {'motion-prepare':prepare,'motion-fit':fit}[args.stage](config,root,args.lane)
        except AllocationEnding as error:
            write_json(root/f'technical/{args.stage}-lane{args.lane}-paused.json',dict(state='paused',reason=str(error)))
            raise SystemExit(75)
        return
    with tracked_run(root/f'technical/execution-{args.stage}',kind='research',configs=[Path(args.config)],
        command=[sys.executable,*sys.argv],question='Can shared local motion directions and weak bending constraints preserve instantaneous local physics?'):
        snapshot_metric_docs(root,'mace_local_motion')
        started = time.monotonic()
        try:
            if args.stage=='motion-all':
                plan(config,root)
                workers(config,args.config,root,'prepare')
                assemble(config,root)
                workers(config,args.config,root,'fit')
                evaluate(config,root)
            elif args.stage=='motion-evaluate': evaluate(config,root)
            else: raise ValueError(args.stage)
            write_json(root/'technical/status.json',dict(state='complete',elapsed_seconds=time.monotonic()-started))
        except AllocationEnding as error:
            write_json(root/'technical/status.json',dict(state='paused',reason=str(error),resume_stage='motion-all'))
            print('MOTION PAUSED',str(error),flush=True)
        except BaseException as error:
            write_json(root/'technical/status.json',dict(state='failed',error=repr(error)))
            raise
