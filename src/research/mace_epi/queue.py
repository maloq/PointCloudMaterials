"""Frozen Slurm training and the existing native physical/prediction assays."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from src.project_runtime.paths import resolve_path
from src.research.encoder_screen.common import load_config, sha, write
from src.training_methods.shared_pretraining.queue import snapshot, deadline_for_job
from src.experiment_runner.slurm import submit_sbatch
from .cached_queue import cached_copies

# Each task needs its own mutable copy, but resolving the full immutable screen
# catalogue is required only once per process, not once per checkpoint.
load_config = cached_copies(load_config)


def read(path):
    c = json.loads(Path(path).read_text())
    if c['protocol'] != 'mace_paired_epi_v1':
        raise ValueError('Wrong paired-MACE protocol')
    for key in ('output', 'screen_config'):
        c[key] = str(resolve_path(c[key]).resolve())
    return c


def task_for(c, item, epoch, *, require=True):
    screen = load_config(c['screen_config'])
    template = next(t for t in screen['tasks'] if t['name']=='jepa-epi-direct-order')
    task = dict(template)
    task.pop('encoder_sha256')
    code = Path(__file__).resolve().parents[3]
    path = Path(c['output'])/'technical/fits'/item['name']/f'epoch-{epoch:03d}.pt'
    files = {str(code/'src'/p.split('/src/', 1)[1]): sha(code/'src'/p.split('/src/', 1)[1])
             for p in template['producer_files']}
    task.update(name=f'{item["name"]}-epoch{epoch:03d}', checkpoint=str(path),
        checkpoint_sha256=sha(path) if require or path.exists() else '',
        producer=str(code), producer_files=files, step=epoch*64, precision=c['precision'])
    return task


def screen_config(c):
    screen = load_config(c['screen_config'])
    screen.update(output=c['output'], excluded=[])
    screen['tasks'] = [task_for(c, item, epoch, require=False)
                       for item in c['fits'] for epoch in c['milestones']]
    return screen


def report(c):
    from src.research.encoder_screen.report import report as native_report
    tech = Path(c['output'])/'technical'
    with (tech/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        native_report(screen_config(c))


def evaluate(c, item, epoch):
    from src.research.encoder_screen.run import run
    from src.research.encoder_parameter_search.assess import supplement
    from src.research.encoder_screen.dense import evaluate as dense
    task = task_for(c, item, epoch)
    screen = screen_config(c)
    folder = Path(c['output'])/'technical/evaluations'/task['name']
    if (folder/'complete.json').exists():
        if json.loads((folder/'complete.json').read_text())['task'] != task:
            raise ValueError('Completed native evaluation identity changed')
    else:
        run(screen, task)
    out = Path(c['output'])/'technical/supplements'/task['name']
    if not (out/'technical/metrics.json').exists():
        supplement(folder, screen['reference'], out)
    if epoch in (12, c['epochs']) and not (folder/'dense8-figures.json').exists():
        dense(screen, task)
    report(c)


def child(command, log):
    with Path(log).open('a') as stream:
        result = subprocess.run([sys.executable, '-u', *command], stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def worker(c, config_path, lane):
    tech = Path(c['output'])/'technical'
    deadline = deadline_for_job()
    gate = json.loads((tech/'preflight.json').read_text())
    if not gate['passed'] or gate['config_sha256'] != sha(config_path):
        raise ValueError('Require successful matching GPU preflight')
    for item in c['fits']:
        folder = tech/'tasks'/item['name']
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'task.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (folder/'complete.json').exists() or (folder/'failed.json').exists():
                continue
            stage = 'training'
            try:
                if time.time() > deadline-1200:
                    return
                write(tech/f'lane-{lane}.json', dict(state='running', task=item['name'], stage=stage))
                rc = child(['-m', 'src.research.mace_epi.queue', 'fit', '--config', str(config_path),
                            '--name', item['name']], folder/'training.log')
                if rc == 75:
                    return
                if rc:
                    raise RuntimeError(f'Fit exited {rc}; see {folder}/training.log')
                for epoch in c['milestones']:
                    if time.time() > deadline-900:
                        return
                    stage = f'evaluate-{epoch}'
                    write(tech/f'lane-{lane}.json', dict(state='running', task=item['name'], stage=stage))
                    rc = child(['-m', 'src.research.mace_epi.queue', 'evaluate', '--config', str(config_path),
                                '--name', item['name'], '--epoch', str(epoch)], folder/f'evaluate-{epoch}.log')
                    if rc:
                        raise RuntimeError(f'Evaluation exited {rc}; see {folder}/evaluate-{epoch}.log')
                write(folder/'complete.json', dict(state='complete', item=item))
            except Exception as exc:
                write(folder/'failed.json', dict(state='failed', stage=stage, error=repr(exc),
                                                traceback=traceback.format_exc()))
    write(tech/f'lane-{lane}.json', dict(state='finished'))
    report(c)


def submit(path):
    c = read(path)
    root = Path(c['output'])
    tech = root/'technical'
    tech.mkdir(parents=True, exist_ok=True)
    if (tech/'launch.json').exists():
        raise FileExistsError('Already submitted; use the frozen worker command to resume')
    screen = load_config(c['screen_config'])
    for part in ('inputs', 'dense8-inputs'):
        shutil.copytree(Path(screen['output'])/'technical'/part, tech/part)
    code = snapshot(tech)
    frozen = code/Path(path).resolve().relative_to(Path.cwd())
    env = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
           'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1', 'PYTORCH_ALLOC_CONF=expandable_segments:True',
           'PYTHONUNBUFFERED=1', f'PCM_PROJECT_ROOT={code}']
    launch = dict(code=str(code), config=str(frozen), jobs=[])

    def job(stage, hours, lane=0, dependency=None):
        command = [sys.executable, '-u', '-m', 'src.research.mace_epi.queue', stage,
                   '--config', str(frozen), '--lane', str(lane)]
        lines = ['#!/bin/bash', f'#SBATCH --job-name=mace-epi-{stage}-{lane}',
                 f'#SBATCH --partition={c["slurm"]["partitions"]}', '#SBATCH --gres=gpu:1',
                 '#SBATCH --cpus-per-task=6', '#SBATCH --mem=64G',
                 f'#SBATCH --time={hours}:00:00', f'#SBATCH --output={tech}/slurm-%j.log',
                 f'#SBATCH --chdir={code}', '#SBATCH --signal=B:USR1@300',
                 *([f'#SBATCH --dependency={dependency}'] if dependency else []),
                 'set -euo pipefail',
                 'export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pcm-epi-${SLURM_JOB_ID}-inductor"',
                 'export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pcm-epi-${SLURM_JOB_ID}-triton"',
                 'exec env '+' '.join(shlex.quote(v) for v in env)+' '+shlex.join(command), '']
        dest = tech/f'{stage}-{lane}-{len(launch["jobs"])}.sbatch'
        jid = submit_sbatch('\n'.join(lines), dest)
        launch['jobs'].append(dict(stage=stage, lane=lane, job_id=jid, dependency=dependency, script=str(dest)))
        write(tech/'launch.json', launch)
        return jid

    gate = job('preflight', 1)
    for lane in range(c['slurm']['workers']):
        first = job('worker', c['slurm']['hours'], lane, f'afterok:{gate}')
        job('worker', c['slurm']['hours'], lane, f'afterok:{first}')
    # Pending checkpoints are reported without pretending they already exist.
    report(c)
    print(json.dumps(launch, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['submit', 'preflight', 'worker', 'fit', 'evaluate', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--name')
    p.add_argument('--lane', type=int, default=0)
    p.add_argument('--epoch', type=int)
    a = p.parse_args()
    c = read(a.config)
    if a.stage == 'submit':
        submit(a.config)
    elif a.stage == 'preflight':
        from .preflight import run
        run(c, a.config)
    elif a.stage == 'worker':
        worker(c, a.config, a.lane)
    elif a.stage == 'report':
        report(c)
    else:
        item = next(i for i in c['fits'] if i['name']==a.name)
        if a.stage == 'evaluate':
            evaluate(c, item, a.epoch)
        else:
            from .train import train
            if not train(c, item, deadline_for_job()):
                raise SystemExit(75)


if __name__ == '__main__':
    main()
