"""Submit frozen-source preparation and explicit epoch continuations to Slurm."""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from src.experiment_runner.registry import sha256, write_json


def submit_queue(config_path, queue_path):
    config_path, queue_path = Path(config_path).resolve(), Path(queue_path).resolve()
    config = json.loads(config_path.read_text())
    plan = json.loads(queue_path.read_text())
    repo = Path(__file__).resolve().parents[3]
    root = Path(plan['output']).resolve()
    # One immutable submission directory prevents an accidental second campaign.
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / 'source'
    shutil.copytree(repo / 'src', frozen / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(root / 'source_manifest.json', {str(p.relative_to(frozen)): sha256(p)
        for p in sorted(frozen.rglob('*')) if p.is_file()})
    bootstrap = (f"import runpy,sys;sys.path.insert(0,{str(frozen)!r});"
                 "runpy.run_module('src.training_methods.embedding_forecast',run_name='__main__')")
    frozen_sources = root / 'sources.json'
    shutil.copyfile(Path(config['data']['sources_config']).resolve(), frozen_sources)
    config['data']['sources_config'] = str(frozen_sources)
    frozen_config = root / 'config.json'
    write_json(frozen_config, config)
    shutil.copyfile(queue_path, root / 'queue_config.json')
    base = [sys.executable, '-c', bootstrap, '--config', str(frozen_config)]
    preparation_job = plan.get('preparation_job_id')
    jobs = ([] if preparation_job is not None else
            [dict(name='prepare', arguments=['--stage', 'prepare'], resources=plan['gpu'], parents=[])])
    chunk = plan['epochs_per_invocation']
    if chunk < 1:
        raise ValueError('Queue epochs_per_invocation must be positive.')
    tails = []
    for variant in config['variants']:
        for seed in config['seeds']:
            previous = 'prepare'
            for start in range(0, config['training']['epochs'], chunk):
                arguments = ['--stage', 'train', '--variant', variant['name'], '--seed', str(seed),
                             '--epochs-per-invocation', str(chunk)]
                if start:
                    arguments.append('--resume')
                name = f"{variant['name']}-s{seed}-e{start:03d}"
                jobs.append(dict(name=name, arguments=arguments, resources=plan['gpu'], parents=[previous]))
                previous = name
            tails.append(previous)
    jobs.append(dict(name='collect', arguments=['--stage', 'collect'], resources=plan['cpu'], parents=tails))
    # Finish all concrete job scripts/specifications before the first submission.
    for job in jobs:
        directory = root / job['name']
        directory.mkdir()
        spec = dict(kind='training' if '--stage' in job['arguments'] and 'train' in job['arguments'] else 'analysis',
            question=plan['question'], cwd=str(repo), output=str(directory), completion='process_exit',
            configs=[str(frozen_config), str(root / 'queue_config.json'), str(frozen_sources),
                     str(root / 'source_manifest.json')], command=base + job['arguments'], dependencies=[])
        write_json(directory / 'run_spec.json', spec)
        script = directory / 'job.sbatch'
        command = [sys.executable, str(repo / 'scripts/experiment_registry.py'), 'run',
                   '--spec', str(directory / 'run_spec.json')]
        script.write_text('#!/bin/bash\nset -euo pipefail\n' +
                          'cd ' + shlex.quote(str(repo)) + '\nexec ' + shlex.join(command) + '\n')
        job['script'] = str(script)
        job['log'] = str(directory / 'slurm.log')
    write_json(root / 'submission_plan.json', dict(config=str(config_path), plan=plan, jobs=jobs))
    external = {} if preparation_job is None else {'prepare': preparation_job}
    receipt = dict(state='submitting', jobs=[], external_dependencies=external,
                   source_manifest_sha256=sha256(root / 'source_manifest.json'))
    write_json(root / 'submission.json', receipt)
    accepted = dict(external)
    for job in jobs:
        resources = job['resources']
        command = ['sbatch', '--parsable', '--job-name', f"ef-{job['name']}",
                   '--partition', resources['partition'], '--cpus-per-task', str(resources['cpus']),
                   '--mem', resources['memory'], '--time', resources['time'],
                   '--output', job['log'], '--error', job['log'],
                   '--export', 'ALL,CONDA_DEFAULT_ENV=pointnet', '--kill-on-invalid-dep=yes']
        if resources['gpus']:
            command += ['--gres', f"gpu:{resources['gpus']}"]
        parents = [accepted[name] for name in job['parents']]
        if parents:
            command += ['--dependency', 'afterok:' + ':'.join(parents)]
        command.append(job['script'])
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            receipt.update(state='submission_failed', failed_command=command, error=result.stderr)
            write_json(root / 'submission.json', receipt)
            raise RuntimeError(f"Slurm submission failed; previously accepted jobs remain recorded in {root / 'submission.json'}: {result.stderr}")
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Unexpected sbatch --parsable result: {result.stdout!r}; inspect Slurm before resubmitting.')
        receipt['jobs'].append(dict(name=job['name'], job_id=job_id, afterok=parents,
                                    command=command, log=job['log']))
        write_json(root / 'submission.json', receipt)
        print(f"Queued {job['name']}: Slurm {job_id}, afterok={parents}", flush=True)
        accepted[job['name']] = job_id
    receipt['state'] = 'submitted'
    write_json(root / 'submission.json', receipt)
    return receipt
