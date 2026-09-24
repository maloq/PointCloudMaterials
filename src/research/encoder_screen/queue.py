"""Locked durable task queue; independent failures are recorded, never hidden."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import load_config,write


def worker(config_path,lane):
    c=load_config(config_path); root=Path(c['output']);tech=root/'technical';status=tech/f'lane-{lane}.json'
    from src.training_methods.shared_pretraining.queue import deadline_for_job
    deadline=deadline_for_job() if 'SLURM_JOB_ID' in os.environ else float('inf')
    for t in c['tasks']:
        name=t['name'];folder=tech/'evaluations'/name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'task.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (folder/'complete.json').exists():
                complete=json.loads((folder/'complete.json').read_text())
                if complete['checkpoint_sha256'] != t['checkpoint_sha256'] or complete['task'] != t:
                    raise ValueError(f'Completed task differs from pinned queue: {name}')
                continue
            if (folder/'failed.json').exists():continue
            if time.time()>deadline-900:
                write(status,dict(state='allocation_ending',lane=lane));return
            write(status,dict(state='running',lane=lane,task=name,pid=os.getpid(),started=time.time()))
            with (folder/'evaluation.log').open('w') as log:
                process=subprocess.run([sys.executable,'-u','-m','src.research.encoder_screen.run','--config',str(config_path),'--name',name],stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                write(folder/'failed.json',dict(state='failed',returncode=process.returncode,task=name,log=str(folder/'evaluation.log')))
            # A report lock keeps two lanes from replacing shared result tables.
            with (tech/'report.lock').open('a') as reporting:
                fcntl.flock(reporting,fcntl.LOCK_EX)
                subprocess.run([sys.executable,'-m','src.research.encoder_screen.report','--config',str(config_path)],check=True)
    write(status,dict(state='lane_finished',lane=lane,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--lane',required=True)
    a=p.parse_args();worker(Path(a.config).resolve(),a.lane)
