"""Gate legacy claims while unstarted tasks move to a new immutable code release.

The old workers and their execute children continue unchanged. This CPU-only gate
holds their original task locks; replacement workers use release-specific locks.
Each replacement waits for its GPU's original fit to finish and release its lock.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import time
from src.data.structural_pretraining.prepare import save_json, file_hash
from src.project_runtime.paths import resolve_path
from .queue import directory

TERMINAL = ('complete', 'failed', 'blocked')


def check_gate(path):
    heartbeat=json.loads(path.with_suffix('.heartbeat.json').read_text())
    if heartbeat['state']!='holding' or time.time()-heartbeat['time']>90:
        raise RuntimeError(f'Legacy queue gate is not live: {path}: {heartbeat}')


def start_worker(path,root,lane,deadline):
    release=json.loads(path.read_text())
    if Path.cwd().resolve()!=Path(release['code']).resolve():
        raise ValueError('Replacement worker must execute from its immutable release')
    if file_hash(root/'tasks.json')!=release['tasks_sha256']:
        raise ValueError('Campaign task definitions changed during release')
    check_gate(path)
    previous=release['legacy_lanes'][str(lane)]['task']
    folder=root/'runs'/previous
    # The original parent owns this lock until its execute subprocess has exited.
    while time.time()<deadline-300:
        check_gate(path)
        status=folder/'status.json'
        if status.exists() and json.loads(status.read_text())['state'] in TERMINAL:
            with (folder/'worker.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:return release
        time.sleep(5)
    return None


def gate(config_path,code,path):
    config=json.loads(Path(config_path).read_text())
    root=resolve_path(config['output']).resolve()/'technical'
    path=Path(path).resolve();code=Path(code).resolve()
    if path.exists():raise FileExistsError(path)
    if not (code/'src/training_methods/neighborhood_jepa/regularization/release.py').is_file():
        raise ValueError('Freeze the updated source before gating the queue')
    all_tasks=json.loads((root/'tasks.json').read_text())
    with ExitStack() as stack:
        selected=[]
        for task in all_tasks:
            dest=directory(root,task);dest.mkdir(parents=True,exist_ok=True)
            lock=(dest/'worker.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close();continue
            # Never change the producer of any task that has begun execution.
            if any(p.name!='worker.lock' for p in dest.iterdir()):
                lock.close();continue
            stack.enter_context(lock)
            selected.append(task)
        lanes={p.stem.split('-')[1]:json.loads(p.read_text()) for p in root.glob('lane-*.json')}
        if not selected:raise ValueError('No unstarted tasks available for the release')
        if any(v['state']!='running' for v in lanes.values()):
            raise ValueError(f'Expected active legacy fit lanes, found {lanes}')
        receipt=dict(id='performance-20260920',code=str(code),config=str(Path(config_path).resolve()),
            tasks=[t['type']+'/'+t['name'] for t in selected],tasks_sha256=file_hash(root/'tasks.json'),
            legacy_lanes=lanes,pid=os.getpid(),host=os.uname().nodename,created=time.time())
        heartbeat=path.with_suffix('.heartbeat.json')
        save_json(heartbeat,dict(state='holding',time=time.time()))
        save_json(path,receipt)
        print(json.dumps(receipt),flush=True)
        while True:
            remaining=[]
            for task in selected:
                status=directory(root,task)/'status.json'
                if not status.exists() or json.loads(status.read_text())['state'] not in TERMINAL:
                    remaining.append(task['type']+'/'+task['name'])
            if not remaining:break
            save_json(heartbeat,dict(state='holding',time=time.time(),remaining=remaining))
            time.sleep(10)
        save_json(heartbeat,dict(state='complete',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--code',required=True);parser.add_argument('--receipt',required=True)
    args=parser.parse_args();gate(args.config,args.code,args.receipt)
