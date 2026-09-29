"""Full-cohort fixed-cell FIRE production using the established relaxation producer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.project_runtime.paths import resolve_path,dataset_path,machine
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions,publish,verify_archive,lock
from src.simulation.relaxation import relax_frame
from src.data.conversion.relaxation import convert
from .data import config,population
from .descriptor_data import load,parent


def freeze(c):
    root=resolve_path(c['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        plan=config(root/'plan.json')
        if plan['config']!=c:raise ValueError('Full relaxation configuration changed')
        return plan
    pc=config(resolve_path(c['descriptor_parent']));_,rows,_,manifest=load(pc)
    _,_,dataset,meta,_=population(parent(pc));ids=rows['ids']
    needed=sorted(set(zip(meta['source'][ids].tolist(),meta['frame'][ids].tolist())))
    cells={};releases=[]
    for name in c['reuse_plans']:
        p=resolve_path(name);release=config(p);releases.append(dict(path=str(p),sha256=sha(p)))
        for folder in sorted((resolve_path(release['config']['cache'])/'cells').iterdir()):
            f=folder/'complete.json'
            if f.exists():
                r=config(f)
                if r['identity']!=release['identity']:raise ValueError(f'Changed source release {f}')
                key=(r['task']['source'],r['task']['frame']);cells.setdefault(key,dict(path=str(f),sha256=sha(f),archive=r['archive']))
    tasks=[dict(id=f'{s}-{f}',source=s,frame=f) for s,f in needed if (s,f) not in cells]
    # Mix sources across workers so slowly relaxing conditions are not isolated
    # in one tail lane. Identity and roles are unchanged.
    np.random.default_rng(c['seed']).shuffle(tasks)
    potential=[str(resolve_path(p)) for p in c['potential_files']]
    for p,h in zip(potential,c['potential_sha256']):
        if sha(Path(p))!=h:raise ValueError(f'Wrong relaxation potential: {p}')
    binary=Path(sys.executable).parent/'lmp'
    if not binary.is_file():raise FileNotFoundError(binary)
    plan=dict(config=c,sources=dataset['sources'],tasks=tasks,required_cells=len(needed),
              reused=[dict(source=s,frame=f,**cells[(s,f)]) for s,f in needed if (s,f) in cells],
              releases=releases,descriptor_identity=manifest['identity'],potential_files=potential,
              lammps=str(binary),lammps_sha256=sha(binary))
    plan['identity']=digest(plan);write_json(root/'plan.json',plan)
    # Expose the new collection through the same documented cell-receipt schema
    # consumed by control_data.freeze, without inventing another archive format.
    return plan


def cell(plan,task):
    c=plan['config'];cache=resolve_path(c['cache'])/'cells'/task['id'];cache.mkdir(parents=True,exist_ok=True)
    receipt=cache/'complete.json'
    if receipt.exists():
        done=config(receipt)
        if done['identity']!=plan['identity']:raise ValueError('Changed full-relaxation identity')
        verify_archive(Path(done['archive']));return
    source=next(s for s in plan['sources'] if s['id']==task['source'])
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if sha(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Relaxation ancestor changed')
    if sha(Path(plan['lammps']))!=plan['lammps_sha256']:raise ValueError('LAMMPS binary changed')
    frame=task['frame'];archive=resolve_path(c['archive'])/'cells'/task['id'];work=resolve_path(c['scratch'])/'cells'/task['id']
    settings=dict(c['relaxation']);execution=machine()['execution'];os.environ.update(execution['mpi_environment'])
    settings.update(lammps_command=[p.format(ranks=c['ranks']) for p in execution['mpi_launcher']]+[plan['lammps']],
       potential_files=plan['potential_files'],pair_commands=['pair_style meam',f'pair_coeff * * {plan["potential_files"][0]} Al {plan["potential_files"][1]} Al'])
    for p,h in zip(plan['potential_files'],c['potential_sha256']):
        if sha(Path(p))!=h:raise ValueError('Potential changed after submission')
    if not archive.exists():
        if not (work/'metadata.json').exists():
            absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count);absolute.positions=AbsolutePositions(raw)
            try:relax_frame(absolute,frame,work,settings)
            except BaseException:
                failure=resolve_path(c['archive'])/'failures'/task['id']
                if work.exists() and not failure.exists():publish(work,failure)
                raise
        meta=config(work/'metadata.json')
        if meta['source_manifest_sha256']!=source['manifest_sha256'] or meta['source_frame']!=frame or meta['fmax_eV_per_A']>.01:
            raise ValueError('Recovered relaxed cell has incorrect provenance/convergence')
        # Global coordinate conversion uses the repository converter; no local
        # full-precision cloud claim is made for this archival replay protocol.
        if not (work/'conversion.json').exists():convert(work,delete_source=True,local_cloud_dtype='none')
        publish(work,archive)
    verify_archive(archive);meta=config(archive/'metadata.json')
    write_json(receipt,dict(identity=plan['identity'],task=task,archive=str(archive),relaxation=meta,
        metadata_sha256=sha(archive/'metadata.json'),conversion_sha256=sha(archive/'conversion.json'),
        local_clouds_saved=False,coordinate_protocol='verified global float16; later centered float32 extraction'))


def worker(c,index):
    plan=freeze(c);root=resolve_path(c['output'])/'technical';errors=[]
    tasks=plan['tasks'][index::c['workers']]
    for task in tasks:
        state=root/'workers'/f'{index}.json';write_json(state,dict(state='running',task=task,job=os.environ.get('SLURM_JOB_ID')))
        with lock(resolve_path(c['cache'])/'cells'/task['id']/'worker.lock') as acquired:
            if not acquired:raise RuntimeError(f'Concurrent producer for {task["id"]}')
            try:
                cell(plan,task)
                print(json.dumps(dict(state='complete',task=task)),flush=True)
            except Exception:
                record=dict(task=task,traceback=traceback.format_exc());write_json(root/'failures'/f'{task["id"]}.json',record);errors.append(task['id'])
                print(json.dumps(record),flush=True)
    write_json(root/'workers'/f'{index}.json',dict(state='failed' if errors else 'complete',errors=errors,tasks=len(tasks)))
    if errors:raise RuntimeError(f'Relaxation failures preserved: {errors}; full-cohort fitting remains gated')


def worker_group(c,path,index,group_size):
    """Pack independent MPI workers into one allocation with disjoint CPU sets."""
    cpus=sorted(os.sched_getaffinity(0));needed=group_size*c['ranks']
    if len(cpus)<needed:raise RuntimeError(f'Worker group needs {needed} CPUs; affinity exposes {len(cpus)}')
    root=resolve_path(c['output'])/'technical/workers';root.mkdir(parents=True,exist_ok=True)
    processes=[]
    for slot in range(group_size):
        worker_index=index*group_size+slot
        if worker_index>=c['workers']:break
        mask=cpus[slot*c['ranks']:(slot+1)*c['ranks']]
        command=['taskset','-c',','.join(map(str,mask)),sys.executable,'-u','-m',
            'src.research.liquid_predictability.control_relaxation','worker','--config',str(path),'--index',str(worker_index)]
        with (root/f'{worker_index}-{os.environ["SLURM_JOB_ID"]}.log').open('a') as log:
            processes.append((worker_index,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
    codes={i:p.wait() for i,p in processes}
    write_json(root/f'group-{index}.json',dict(returncodes=codes,cpus=cpus[:needed],group_size=group_size))
    if any(codes.values()):raise RuntimeError(f'Relaxation worker failure(s): {codes}; see per-worker logs')


def submit(path,group_size=4):
    c=config(path);plan=freeze(c);root=resolve_path(c['output'])/'technical';launch=root/'launch.json'
    if launch.exists():raise ValueError('Full relaxation already submitted')
    repo=Path(__file__).resolve().parents[3];code=root/'code'
    if code.exists():
        # A rejected first sbatch has no live jobs. Preserve that frozen attempt
        # before creating a new scheduling snapshot; scientific plan is unchanged.
        code.rename(root/f'code-unsubmitted-{time.time_ns()}')
    shutil.copytree(repo/'src',code/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    shutil.copytree(repo/'docs/metrics',code/'docs/metrics');shutil.copytree(repo/'configs/liquid_predictability',code/'configs/liquid_predictability')
    write_json(code/'relaxation.json',c)
    groups=(c['workers']+group_size-1)//group_size
    record=dict(code=str(code),required_cells=plan['required_cells'],reused_cells=len(plan['reused']),new_cells=len(plan['tasks']),
        worker_groups=groups,workers_per_group=group_size,mpi_workers=c['workers'],jobs={})
    def job(stage,opts,after=None):
        script=root/f'{stage}.sbatch';cmd=[sys.executable,'-u','-m','src.research.liquid_predictability.control_relaxation',stage,'--config',str(code/'relaxation.json')]
        if stage=='worker-group':cmd+=['--group-size',str(group_size)]
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=LC-full-{stage}','#SBATCH --partition=CPU',
            '#SBATCH --nodes=1','#SBATCH --ntasks=1',f'#SBATCH --output={root}/{stage}-%A_%a.log',*['#SBATCH '+s for s in opts],
            'set -euo pipefail','ulimit -n 4096','cd '+shlex.quote(str(code)),
            'exec env '+shlex.join([f'PCM_PROJECT_ROOT={repo}','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1'])+' '+shlex.join(cmd),'']))
        args=['sbatch','--parsable']+(['--dependency=afterok:'+after] if after else [])+[str(script)]
        jid=subprocess.check_output(args,text=True).strip().split(';')[0];record['jobs'][stage]=jid;write_json(launch,record);return jid
    try:
        workers=job('worker-group',[f'--array=0-{groups-1}%{groups}',f'--cpus-per-task={group_size*c["ranks"]}',
            f'--mem={16*group_size}G','--time=2-00:00:00'])
        job('study',['--cpus-per-task=4','--mem=48G','--time=02:00:00'],workers)
    except BaseException:
        record['submission_error']=traceback.format_exc();write_json(root/f'submission-failure-{time.time_ns()}.json',record);raise
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['submit','worker','worker-group','study']);p.add_argument('--config',required=True);p.add_argument('--index',type=int);p.add_argument('--group-size',type=int,default=4)
    a=p.parse_args();c=config(a.config)
    if a.stage=='submit':print(json.dumps(submit(a.config,a.group_size),indent=2))
    elif a.stage=='worker':worker(c,a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif a.stage=='worker-group':worker_group(c,a.config,a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']),a.group_size)
    else:
        plan=freeze(c)
        for task in plan['tasks']:
            f=resolve_path(c['cache'])/'cells'/task['id']/'complete.json'
            if not f.exists() or config(f)['identity']!=plan['identity']:raise ValueError(f'Full coverage incomplete: {task}')
        from .control_queue import submit as study_submit
        print(json.dumps(study_submit(str(Path(__file__).resolve().parents[3]/c['study_config'])),indent=2))


if __name__=='__main__':main()
