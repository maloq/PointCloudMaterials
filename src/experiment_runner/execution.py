"""Frozen execution bundles, explicit Slurm submissions and stage receipts.

Scientific dependency graphs, resource choices and continuation policies belong
to the calling workflow. These methods own only execution and provenance I/O.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from .artifacts import write_json


def allocation_deadline(*, reserve_seconds, job=None):
    """Use Slurm's epoch timestamp, independent of the worker's local timezone."""
    job = job or os.environ.get('SLURM_JOB_ID')
    if not job:
        return math.inf
    result = subprocess.run(['scontrol', 'show', 'job', str(job), '--json'],
                            check=True, text=True, capture_output=True)
    value = json.loads(result.stdout)['jobs'][0]['end_time']
    end = value['number'] if isinstance(value, dict) else value
    if (type(end) not in (int, float) or not math.isfinite(end)
            or end < time.time()):
        raise ValueError(f'Invalid Slurm end time for job {job}: {value!r}')
    return end - reserve_seconds


@dataclass(frozen=True)
class ExecutionBundle:
    root: Path

    @classmethod
    def freeze(cls, repository, destination, config, *, directories, files=()):
        """Create a new source copy; an existing destination is never replaced."""
        repository = Path(repository)
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f'Frozen execution bundle exists: {destination}')
        for directory in directories:
            shutil.copytree(
                repository / directory,
                destination / directory,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'),
            )
        for source, relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_json(destination / 'config.json', config)
        return cls(destination)

    @property
    def config_path(self):
        return self.root / 'config.json'


@dataclass
class SlurmQueue:
    technical: Path
    bundle: ExecutionBundle
    module: str
    environment: dict
    receipt_path: Path
    receipt: dict
    job_prefix: str

    def command(self, stage, *arguments):
        return [sys.executable, '-u', '-m', self.module, stage,
                '--config', str(self.bundle.config_path), *arguments]

    def render(self, stage, options, *, partition='CPU', command_stage=None, arguments=()):
        command = self.command(command_stage or stage, *arguments)
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_prefix}-{stage}',
            f'#SBATCH --partition={partition}',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            f'#SBATCH --output={self.technical}/{stage}-%A_%a.log',
            *['#SBATCH ' + option for option in options],
            'set -euo pipefail',
            'ulimit -n 4096',
            'cd ' + shlex.quote(str(self.bundle.root)),
            'exec env ' + shlex.join(
                [f'{key}={value}' for key, value in self.environment.items()]
            ) + ' ' + shlex.join(command),
            '',
        ]
        return '\n'.join(lines)

    def submit(self, stage, options, dependency=None, partition='CPU', *,
               command_stage=None, arguments=()):
        script = self.technical / f'{stage}.sbatch'
        script.write_text(self.render(stage, options, partition=partition,
                                     command_stage=command_stage, arguments=arguments))
        sbatch_args = ['sbatch', '--parsable']
        if dependency:
            sbatch_args.append('--dependency=' + dependency)
        result = subprocess.check_output(sbatch_args + [str(script)], text=True)
        job = result.strip().split(';', 1)[0]
        if not job.isdigit():
            raise ValueError(f'Invalid sbatch job ID for {script}: {result!r}')
        self.receipt['jobs'][stage] = job
        write_json(self.receipt_path, self.receipt)
        return job

    @contextmanager
    def submission(self):
        """Keep submission failures together with jobs already accepted by Slurm."""
        try:
            yield self
        except BaseException:
            self.receipt['submission_error'] = traceback.format_exc()
            write_json(self.receipt_path, self.receipt)
            raise


@dataclass
class StageRecord:
    path: Path
    fields: dict

    def update(self, **fields):
        self.fields.update(fields)
        write_json(self.path, self.fields)


@contextmanager
def recorded_stage(path, **context):
    """Persist stage progress and failure context, and propagate all exceptions."""
    stage = StageRecord(Path(path), dict(context, started_at=time.time()))
    stage.update(state='running')
    try:
        yield stage
    except BaseException as error:
        stage.update(state='failed', finished_at=time.time(), error=repr(error),
                     traceback=traceback.format_exc())
        raise
    else:
        stage.update(state='complete', finished_at=time.time())
