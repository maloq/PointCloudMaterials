"""Check and submit independent capacity studies through the existing trainer."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import torch

from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.queue import snapshot
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .common import Study, write_json
from .queue import configure, preflight


def submit(plan, root):
    receipt = root/'technical/submissions.json'
    if receipt.exists():
        raise FileExistsError(f'Campaign already submitted: {receipt}')
    studies = [Study(item['config']) for item in plan['runs']]
    for study in studies:
        study.bind()
        check = json.loads((study.technical/'preflight.json').read_text())
        if not check['passed'] or check['identity'] != study.identity:
            raise ValueError(f'Preflight does not match source/config/data: {study.path}')
    code = snapshot(root/'technical')
    jobs = []
    for item, study in zip(plan['runs'], studies, strict=True):
        (study.technical/'code').symlink_to(code, target_is_directory=True)
        config = code/study.path.relative_to(Path.cwd().resolve())
        command = [sys.executable, '-u', '-m', 'src.research.supervised_onset.queue', 'worker', '--config', str(config)]
        script = study.technical/'worker.sbatch'
        script.write_text('\n'.join([
            '#!/bin/bash', f'#SBATCH --job-name=OnsetNLL-{item["name"]}',
            f'#SBATCH --partition={plan["partitions"]}', '#SBATCH --gres=gpu:1',
            '#SBATCH --nodes=1', '#SBATCH --ntasks=1', f'#SBATCH --cpus-per-task={plan["cpus"]}',
            f'#SBATCH --mem={plan["memory"]}', f'#SBATCH --time={plan["walltime"]}',
            f'#SBATCH --output={study.technical}/slurm-%j.log',
            'set -euo pipefail', f'cd {shlex.quote(str(code))}',
            f'export PCM_PROJECT_ROOT={shlex.quote(str(code))}',
            'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            'export PYTORCH_ALLOC_CONF=expandable_segments:True',
            # An explicit Slurm step guarantees the same allocation semantics
            # as the existing detached worker (including SLURM_STEP_ID).
            'exec srun --exact --nodes=1 --ntasks=1 --gres=gpu:1 '+shlex.join(command), '']))
        job = subprocess.check_output(['sbatch', '--parsable', str(script)], text=True).strip().split(';')[0]
        record = dict(name=item['name'], job=job, identity=study.identity, output=str(study.root),
                      code=str(code), config=str(config), parameters=study.config['parameter_budget'])
        write_json(study.technical/'launch.json', record)
        jobs.append(record)
        # Retain every accepted job even if a later submission is rejected.
        write_json(receipt, dict(state='submitting', jobs=jobs, plan=plan))
        print(json.dumps(record), flush=True)
    write_json(receipt, dict(state='submitted', jobs=jobs, plan=plan))


def collect(plan, root):
    rows, coverage = [], []
    for item in plan['runs']:
        study = Study(item['config'])
        path = study.root/'tables/comparison.csv'
        state = study.technical/'queue-state.json'
        coverage.append(dict(name=item['name'], state=json.loads(state.read_text()) if state.exists() else None))
        if not path.exists():
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                rows.append(dict(capacity=item['name'], **study.config['parameter_budget'],
                                 channels=study.config['encoder']['channels'], **row))
    if rows:
        snapshot_metric_docs(root, 'supervised_onset')
        with (root/'tables/comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root/'technical/coverage.json', dict(runs=coverage, result_rows=len(rows)))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('check', 'submit', 'collect'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    root = resolve_path(plan['output'])
    for folder in ('technical', 'tables'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    configure()
    if args.action == 'check':
        for item in plan['runs']:
            # Each worker owns one capacity; preflight checks several capacities
            # serially here without accumulating unrelated graph specializations.
            torch._dynamo.reset()
            print(json.dumps(dict(stage='preflight', capacity=item['name'])), flush=True)
            preflight(Study(item['config']), 'cuda')
    elif args.action == 'submit':
        submit(plan, root)
    else:
        print(json.dumps(dict(result_rows=collect(plan, root))))


if __name__ == '__main__':
    main()
