"""Copy verified extraction chunks into the selective detector layout.

This performs no PTM, trajectory reading, resampling or event detection. The
source cache stays untouched, including caches read by frozen running workers.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path, path_snapshot
from .extraction import source_root
from .io import read


@path_snapshot()
def repack(plan_path, output, cache):
    plan_path=resolve_path(plan_path);original=read(plan_path)
    if digest({k:v for k,v in original.items() if k!='identity'})!=original['identity']:
        raise ValueError(f'Changed source extraction plan: {plan_path}')
    root=resolve_path(output);destination=resolve_path(cache)
    if root==resolve_path(original['config']['output']) or destination==resolve_path(original['config']['cache']):
        raise ValueError('Repacking requires a distinct output and cache; original readers must remain valid')
    plan=dict(original,config=dict(original['config'],output=str(output),cache=str(cache)),
              cache_preparation=dict(source_plan=str(plan_path),source_plan_sha256=sha(plan_path),
                                     source_identity=original['identity'],layout='detector_arrays_v1',
                                     producer_sha256=sha(Path(__file__))))
    plan.pop('identity');plan['identity']=digest(plan)
    target_plan=root/'technical/plan.json'
    if (root/'technical/launch.json').exists():raise ValueError('Destination search already launched')
    if target_plan.exists() and read(target_plan)!=plan:raise ValueError('Different preparation exists at destination')
    write_json(target_plan,plan)
    sources={s['id']:s for s in plan['sources']};started=time.monotonic();copied=0
    for task in plan['tasks']:
        source=sources[task['source']];old=source_root(original,source);new=source_root(plan,source)
        new.mkdir(parents=True,exist_ok=True)
        name=f'clusters-{task["start"]:06d}-{task["stop"]:06d}'
        target=new/(name+'.npz');receipt=new/(name+'.json');identity=digest(dict(plan=plan['identity'],task=task))
        with (new/(name+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if receipt.exists():
                done=read(receipt)
                if done['identity']!=identity or sha(target)!=done['sha256']:
                    raise ValueError(f'Changed repacked chunk: {target}')
                continue
            before=read(old/(name+'.json'));path=old/(name+'.npz')
            if before['identity']!=digest(dict(plan=original['identity'],task=task)):
                raise ValueError(f'Original chunk belongs to a different extraction plan: {path}')
            if sha(path)!=before['sha256']:raise ValueError(f'Changed original chunk: {path}')
            with np.load(path,allow_pickle=False) as arrays:
                if before.get('layout')=='detector_arrays_v1':
                    packet={key:arrays[key] for key in arrays.files}
                elif 'layout' not in before:
                    clusters=arrays['clusters']
                    packet={key:arrays[key] for key in arrays.files if key!='clusters'}
                    packet.update({f'clusters_{detector}':clusters[:,i]
                                   for i,detector in enumerate(original['config']['detectors'])})
                else:raise ValueError(f'Unknown extraction layout: {path}')
            temporary=target.with_suffix('.building.npz');np.savez_compressed(temporary,**packet);temporary.replace(target)
            write_json(receipt,dict(before,identity=identity,sha256=sha(target),layout='detector_arrays_v1',
                                   repacked_from=dict(path=str(path),sha256=before['sha256'])))
            copied+=1
        write_json(root/'technical/repack-progress.json',dict(copied=copied,total=len(plan['tasks'])))
    result=dict(identity=plan['identity'],state='complete',chunks=len(plan['tasks']),copied=copied,
                seconds=time.monotonic()-started,source_plan_sha256=sha(plan_path))
    write_json(root/'technical/repack-complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('repack',))
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache',required=True)
    args=parser.parse_args()
    print(repack(args.plan,args.output,args.cache),flush=True)


if __name__=='__main__':main()
