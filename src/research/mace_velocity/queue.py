"""Detached local preparation/teacher/training sequence within an allocation."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from src.experiment_runner.registry import write_json
from src.project_runtime.paths import load_json
from .inventory import read


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',default='configs/analysis/mace_velocity.json')
    parser.add_argument('--finish-paused',action='store_true',
        help='Evaluate retained best checkpoints if the running queue stops at its allocation deadline')
    args=parser.parse_args();config=load_json(args.config);root=Path(config['output'])/'technical'
    if args.finish_paused:
        # A separately detached finalizer can accompany an already running queue.
        # It never trains twice or evaluates while the trainers are still active.
        while True:
            state=read(root/'queue-status.json')['state']
            if state=='complete':return
            if state=='failed':raise RuntimeError('Training queue failed; inspect the retained failure before evaluation')
            if state=='paused':break
            if time.time()>datetime.fromisoformat(config['deadline_utc']).timestamp()+300:
                raise TimeoutError('Queue did not finish within the retained evaluation margin')
            time.sleep(10)
        processes=[]
        for gpu,variant in enumerate(config['variants']):
            if read(root/variant/'status.json')['state']=='complete':continue
            with (root/f'{variant}-deadline-evaluation.log').open('x') as stream:
                process=subprocess.Popen([sys.executable,'-m','src.research.mace_velocity','evaluate',
                    '--config',args.config,'--variant',variant,'--device',f'cuda:{gpu}'],stdout=stream,stderr=subprocess.STDOUT)
            processes.append((variant,process))
        results={variant:process.wait() for variant,process in processes}
        if any(code!=0 for code in results.values()):
            write_json(root/'post-training-evaluation-status.json',dict(state='failed',exit_codes=results))
            raise RuntimeError(f'Deadline checkpoint evaluation failed: {results}')
        states={v:read(root/v/'status.json') for v in config['variants']}
        write_json(root/'post-training-evaluation-status.json',dict(state='complete',variants=states))
        write_json(root/'queue-status.json',dict(state='complete_with_partial_training',variants=states,
            reason='Allocation deadline; best completed-epoch checkpoints evaluated, exact resume states retained.'))
        return
    write_json(root/'queue-status.json',dict(state='waiting_for_preparation',pid=os.getpid()))
    launch=read(root/'prepare-launch.json')
    try:
        while not (Path(config['cache'])/'manifest.json').exists():
            if (root/'prepare-coordinates_velocity-failure.json').exists():
                raise RuntimeError('Preparation failed; see its retained failure report')
            os.kill(launch['pid'],0)
            if time.time()>datetime.fromisoformat(config['deadline_utc']).timestamp()-1800:
                raise TimeoutError('Insufficient allocation time remains after preparation')
            time.sleep(10)
        command=[sys.executable,'-m','src.research.mace_velocity']
        write_json(root/'queue-status.json',dict(state='teacher_extraction',pid=os.getpid()))
        with (root/'teacher.log').open('x') as stream:
            subprocess.run(command+['teacher','--config',args.config,'--device','cuda:0'],stdout=stream,stderr=subprocess.STDOUT,check=True)
        processes=[]
        for gpu,variant in enumerate(config['variants']):
            with (root/f'{variant}.log').open('x') as stream:
                process=subprocess.Popen(command+['train','--config',args.config,'--variant',variant,'--device',f'cuda:{gpu}'],stdout=stream,stderr=subprocess.STDOUT)
            processes.append((variant,process))
        write_json(root/'queue-status.json',dict(state='training',pid=os.getpid(),
            processes={v:p.pid for v,p in processes}))
        results={variant:process.wait() for variant,process in processes}
        if any(code!=0 for code in results.values()):raise RuntimeError(f'Training process failed: {results}')
        states={v:read(root/v/'status.json')['state'] for v,p in processes}
        write_json(root/'queue-status.json',dict(state='complete' if all(s=='complete' for s in states.values()) else 'paused',variants=states))
    except BaseException as error:
        write_json(root/'queue-status.json',dict(state='failed',error=repr(error)));raise


if __name__=='__main__':main()
