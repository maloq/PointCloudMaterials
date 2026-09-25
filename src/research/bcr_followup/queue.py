"""Prioritized, resumable eight-hour frozen-BCR audit on one GPU."""
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

from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.queue import snapshot, deadline_for_job
from .common import Study, extract, write_json, remaining, file_hash
from . import readouts, interventions, decoders, relaxed, report


def run(config, device, stage='all', deadline=None):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    study = Study(config)
    stages = [('features', extract), ('probes', readouts.run), ('interventions', interventions.run),
              ('relaxed', relaxed.run), ('decoders', decoders.run)]
    if stage != 'all': stages = [(name, fn) for name, fn in stages if name == stage]
    if stage == 'report': return report.run(study)
    queue_status = study.technical/'queue-status.json'; started = time.time()
    with (study.technical/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []; current = None
        try:
            for current, function in stages:
                remaining(deadline)
                write_json(queue_status, dict(state='running', stage=current, completed=completed, started=started, deadline=deadline))
                function(study, device, deadline)
                completed.append(current)
                report.run(study)
            write_json(queue_status, dict(state='complete', completed=completed, started=started, finished=time.time()))
        except TimeoutError as exc:
            report.run(study)
            write_json(queue_status, dict(state='checkpointed', stage=current, completed=completed, reason=str(exc), deadline=deadline))
        except Exception as exc:
            write_json(queue_status, dict(state='failed', stage=current, completed=completed,
                       error=repr(exc), traceback=traceback.format_exc()))
            raise


def submit(config_path):
    study = Study(config_path)
    root = study.technical; launch = root/'launch.json'
    if launch.exists(): raise FileExistsError(f'Already submitted: {launch}')
    preflight = json.loads((root/'preflight.json').read_text())
    if not preflight['passed'] or preflight['identity'] != study.identity:
        raise ValueError('Submission requires passing preflight bound to current code/data/configuration')
    relaxed.freeze_selection(study)
    code = snapshot(root)
    # Snapshot() preserves data/output routing; the scientific root selection is
    # frozen before submission, never expanded by later producer completions.
    relative = Path(config_path).resolve().relative_to(Path.cwd().resolve())
    command = [sys.executable, '-u', '-m', 'src.research.bcr_followup.queue', 'run',
               '--config', str(code/relative), '--device', 'cuda']
    script = root/'audit.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=bcr-audit-8h
#SBATCH --partition={study.config['partition']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time={study.config['wall_hours']:02d}:00:00
#SBATCH --output={root}/audit-%j.log
set -euo pipefail
cd {shlex.quote(str(code))}
export PCM_PROJECT_ROOT={shlex.quote(str(code))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec {shlex.join(command)}
''')
    job = subprocess.check_output(['sbatch', '--parsable', str(script)], text=True).strip()
    write_json(launch, dict(job=job, hours=study.config['wall_hours'], gpus=1, frozen_code=str(code),
                            config_sha256=file_hash(config_path), identity=study.identity, submitted_at=time.time()))
    print(json.dumps(dict(job=job, output=str(study.root), hours=study.config['wall_hours'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'submit', 'freeze', 'preflight', 'report'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=['all', 'features', 'probes', 'interventions', 'relaxed', 'decoders'], default='all')
    args = parser.parse_args()
    if args.action == 'submit': submit(args.config)
    elif args.action == 'freeze': relaxed.freeze_selection(Study(args.config))
    elif args.action == 'preflight':
        from .preflight import run as preflight
        preflight(args.config, args.device)
    else:
        deadline = deadline_for_job() if 'SLURM_JOB_ID' in os.environ else time.time()+8*3600-300
        run(args.config, args.device, 'report' if args.action == 'report' else args.stage, deadline)


if __name__ == '__main__': main()
