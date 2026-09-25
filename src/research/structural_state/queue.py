"""Frozen Slurm or detached in-allocation queues for structural-state fits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import torch

from src.training_methods.shared_pretraining.queue import snapshot, deadline_for_job
from .common import Study, write_json, sha


def script(study, code, name=None, dependencies=None):
    cfg = study.config['slurm']
    relative = study.config_path.relative_to(Path.cwd().resolve())
    command = [sys.executable, '-u', '-m', 'src.research.structural_state.queue',
               'collect' if name is None else 'worker', '--config', str(code / relative)]
    if name is not None:
        command += ['--arm', name]
    directives = [f'#SBATCH --job-name=state-{name or "collect"}',
        f'#SBATCH --partition={cfg["partitions"] if name else "CPU"}',
        f'#SBATCH --cpus-per-task={cfg["cpus"]}',
        f'#SBATCH --mem={cfg["memory_GiB"] if name else 8}G',
        f'#SBATCH --time={cfg["hours"] if name else 1:02d}:00:00',
        f'#SBATCH --output={study.technical}/{name or "collect"}-%j.log',
        f'#SBATCH --chdir={code}']
    if name:
        directives.extend(('#SBATCH --gres=gpu:1', f'#SBATCH --exclude={cfg["exclude"]}'))
    if dependencies:
        directives.append('#SBATCH --dependency=afterany:' + ':'.join(dependencies))
    environment = ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                   'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
                   'PYTORCH_ALLOC_CONF=expandable_segments:True', f'PCM_PROJECT_ROOT={code}']
    return '\n'.join(['#!/bin/bash', *directives, 'set -euo pipefail',
                      'exec env ' + ' '.join(shlex.quote(v) for v in environment) + ' ' + shlex.join(command), ''])


def submit(study):
    study.bind()
    root = study.technical
    receipt = json.loads((root / 'preflight.json').read_text())
    if not receipt['passed'] or receipt['identity'] != study.identity:
        raise ValueError('Submission requires passing tests and production-size preflight for this exact implementation')
    path = root / 'launch.json'
    if path.exists():
        raise FileExistsError(f'Study already submitted; inspect {path}')
    code = snapshot(root)
    launch = dict(state='submitting', identity=study.identity, code=str(code), jobs=[], submitted_at=time.time())
    write_json(path, launch)
    for arm in study.config['arms']:
        name = arm['name']
        batch = root / f'{name}.sbatch'
        batch.write_text(script(study, code, name))
        job = subprocess.check_output(['sbatch', '--parsable', str(batch)], text=True).strip().split(';')[0]
        launch['jobs'].append(dict(name=name, job=job, script=str(batch)))
        write_json(path, launch)
    batch = root / 'collect.sbatch'
    batch.write_text(script(study, code, dependencies=[j['job'] for j in launch['jobs']]))
    launch['collector_job'] = subprocess.check_output(['sbatch', '--parsable', str(batch)], text=True).strip().split(';')[0]
    launch['state'] = 'submitted'
    write_json(path, launch)
    print(json.dumps(launch, indent=2), flush=True)


def worker(study, name, device):
    from .runtime import train
    from .evaluation import run as evaluate
    study.bind()
    deadline = deadline_for_job()
    status = study.technical / f'{name}-status.json'
    with (study.technical / f'{name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = 'training'
        try:
            write_json(status, dict(state='running', stage=stage, identity=study.identity, job=os.environ['SLURM_JOB_ID']))
            if not train(study, name, device, deadline):
                raise TimeoutError('Training checkpoint saved at allocation deadline')
            torch.cuda.empty_cache()
            stage = 'evaluation'
            write_json(status, dict(state='running', stage=stage, identity=study.identity, job=os.environ['SLURM_JOB_ID']))
            evaluate(study, name, device, deadline)
            write_json(status, dict(state='complete', identity=study.identity, job=os.environ['SLURM_JOB_ID']))
        except TimeoutError as error:
            write_json(status, dict(state='checkpointed', stage=stage, reason=str(error), identity=study.identity))
            return 75
        except Exception as error:
            write_json(status, dict(state='failed', stage=stage, error=repr(error), traceback=traceback.format_exc(), identity=study.identity))
            raise
    return 0


def launch_local(study):
    """Use only GPUs inside the current allocation; no new Slurm submission."""
    study.bind()
    receipt = json.loads((study.technical / 'preflight.json').read_text())
    if not receipt['passed'] or receipt['identity'] != study.identity:
        raise ValueError('Local launch requires preflight for this exact implementation')
    job = os.environ['SLURM_JOB_ID']
    deadline = deadline_for_job()
    if deadline - time.time() < 3600:
        raise ValueError('Less than one hour remains in the allocation')
    count = torch.cuda.device_count()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(str(i) for i in range(count))).split(',')
    if len(devices) != count:
        raise ValueError('Visible GPU identifiers do not match the allocation device count')
    groups = study.config['local_queue']
    if len(groups) > count:
        raise ValueError('Queue requests more GPUs than are visible in this allocation')
    names = [n for group in groups for n in group]
    if sorted(names) != sorted(a['name'] for a in study.config['arms']):
        raise ValueError('Local queues must cover every scientific arm exactly once')
    path = study.technical / 'launch.json'
    if path.exists():
        raise FileExistsError(f'Already launched; inspect {path}')
    code = snapshot(study.technical)
    relative = study.config_path.relative_to(Path.cwd().resolve())
    launch = dict(state='launching', identity=study.identity, code=str(code),
                  job=job, host=os.uname().nodename, deadline=deadline, workers=[])
    write_json(path, launch)
    for i, arms in enumerate(groups):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[i], PCM_PROJECT_ROOT=str(code),
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   PYTORCH_ALLOC_CONF='expandable_segments:True')
        command = [sys.executable, '-u', '-m', 'src.research.structural_state.queue', 'serial',
                   '--config', str(code / relative), '--arms', *arms]
        log = study.technical / f'local-gpu{i}.log'
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launch['workers'].append(dict(arms=arms, gpu=devices[i], pid=process.pid, log=str(log), command=command))
        write_json(path, launch)
    launch['state'] = 'running'
    write_json(path, launch)
    print(json.dumps(launch, indent=2), flush=True)


def serial(study, names, device):
    from .report import run as collect
    for name in names:
        study.arm(name)
        code = worker(study, name, device)
        # Two local GPUs may finish simultaneously. Serialize table snapshots.
        with (study.technical / 'collect.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            collect(study)
        if study.config['protocol']=='fixed_geometry_future_relation_v3':
            from .factorial_report import run as collect_factorial
            collect_factorial(study.config['campaign'])
        if code:
            return code
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'submit', 'worker', 'collect', 'launch-local', 'serial'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm')
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    study = Study(args.config)
    if study.config['protocol'] == 'fixed_geometry_parameter_search_v4':
        raise ValueError('v4 uses src.research.encoder_parameter_search.queue and its fixed native snapshot assay')
    if args.action == 'prepare':
        from .data import prepare
        print(json.dumps(prepare(study), indent=2), flush=True)
    elif args.action == 'preflight':
        from .preflight import run
        run(study, args.device)
    elif args.action == 'submit':
        submit(study)
    elif args.action == 'launch-local':
        launch_local(study)
    elif args.action == 'serial':
        if not args.arms:
            parser.error('serial requires --arms')
        sys.exit(serial(study, args.arms, args.device))
    elif args.action == 'worker':
        if args.arm is None:
            parser.error('worker requires --arm')
        sys.exit(worker(study, args.arm, args.device))
    else:
        from .report import run
        study.bind()
        print(run(study), flush=True)


if __name__ == '__main__':
    main()
