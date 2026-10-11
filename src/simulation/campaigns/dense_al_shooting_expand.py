"""Add V100 lanes to a running frozen shooting queue without duplicate MD.

The original worker has an in-memory assignment. Its existing status-file guard
ends that lane at a reserved future source, after all retained sources publish.
The new lane assigned that boundary source waits for the old job to end before
removing the reservation. All scientific execution uses the original frozen code.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from src.project_runtime.paths import load_json
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.birth_sources import now, sha256
from src.simulation.campaigns.dense_al import manifest_at, run_source, sbatch
from src.simulation.campaigns.dense_al_shooting import check_runtime, collect, environment


def read(path):
    return json.loads(Path(path).read_text())


def queued(job):
    # Completed jobs can disappear from Slurm before this queue finishes; querying
    # the user queue avoids treating an expired job ID as a scheduler failure.
    rows = subprocess.check_output(['squeue', '--array', '-h', '--user', str(os.getuid()),
                                    '-o', '%i %T'], text=True).splitlines()
    return '\n'.join(row.strip() for row in rows
        if row.split()[0] == str(job) or row.split()[0].startswith(str(job)+'_'))


def verify(expansion):
    plan = read(expansion/'plan.json')
    if sha256(expansion/'plan.json') != (expansion/'plan.sha256').read_text().strip():
        raise ValueError('Expansion assignments changed')
    if Path(__file__).resolve() != expansion/'driver.py' or sha256(__file__) != plan['driver_sha256']:
        raise ValueError('Use the frozen expansion driver')
    parent = Path(plan['parent_launch'])
    for name, expected in plan['parent_hashes'].items():
        if sha256(parent/name) != expected:
            raise ValueError(f'Parent campaign changed: {name}')
    return plan, parent, manifest_at(parent)


def submit(c):
    parent = Path(c['parent_launch']); original = read(parent/'submission.json')
    campaign = manifest_at(parent); order, = campaign['assignments']
    expansion = parent/'technical'/c['expansion_id']
    if expansion.exists():
        raise FileExistsError(f'Preserve existing expansion: {expansion}')
    # These job IDs are bound to this campaign's launch receipt, not Unix ownership.
    old_job = str(original['worker_job']); old_collector = str(original['collect_job'])
    if ' RUNNING' not in queued(old_job) or queued(old_collector) != old_collector+' PENDING':
        raise ValueError('Expected original worker running and its collector pending')
    retained = order[:c['retain_original_entries']]
    moved = order[c['retain_original_entries']:]
    if not retained or len(moved) < c['additional_workers']:
        raise ValueError('Invalid queue partition')
    gate = read(parent/'technical/audit/validation.json')
    if gate['state'] != 'passed' or gate['code_inventory_sha256'] != sha256(parent/'code_sha256.json'):
        raise ValueError('Missing matching numerical validation')
    for i in moved:
        record = campaign['runs'][i]; directory = Path(campaign['root'])/record['run_dir']
        if (directory/'status.json').exists():
            raise ValueError(f'Cannot transfer a started source: {directory}')
        for name, expected in record['input_sha256'].items():
            if sha256(directory/name) != expected:
                raise ValueError(f'Prepared input changed: {directory/name}')
    assignments = [moved[i::c['additional_workers']] for i in range(c['additional_workers'])]
    # The boundary marker remains until the old worker has exited. Handle that
    # branch last in its new lane; this ordering depends only on queue identity.
    assignments[0] = assignments[0][1:] + assignments[0][:1]
    expansion.mkdir(parents=True)
    shutil.copy2(__file__, expansion/'driver.py')
    plan = dict(parent_launch=str(parent), original_worker_job=old_job,
        original_collector_job=old_collector, retained_indices=retained,
        assignments=assignments, boundary_index=moved[0], config=c,
        driver_sha256=sha256(expansion/'driver.py'), created_at=now(),
        parent_hashes={name:sha256(parent/name) for name in
            ('manifest.json', 'code_sha256.json', 'technical/audit/validation.json')},
        original_exit='Expected pre-source FileExistsError at reserved boundary; no active MD interrupted')
    write_json(expansion/'plan.json', plan)
    (expansion/'plan.sha256').write_text(sha256(expansion/'plan.json')+'\n')
    boundary = Path(campaign['root'])/campaign['runs'][moved[0]]['run_dir']/'status.json'
    reservation = dict(state='expansion_reserved', expansion=str(expansion), reserved_at=now())
    with boundary.open('x') as stream:
        json.dump(reservation, stream, indent=2); stream.write('\n')
    env, profile = environment(parent)
    command = [profile['python'], '-u', str(expansion/'driver.py')]
    base = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
        f'#SBATCH --chdir={parent}/code', '#SBATCH --no-requeue']
    body = ['set -euo pipefail', 'export '+shlex.join([k+'='+v for k,v in env.items()])]
    def script(action, options):
        return '\n'.join(base+options+body+['exec '+shlex.join(command+
            [action, '--expansion', str(expansion)]), ''])
    subprocess.run(['scontrol', 'hold', old_collector], check=True)
    receipt = dict(state='reserved', original_worker_job=old_job,
        original_collector_job=old_collector, expansion=str(expansion), created_at=now())
    write_json(expansion/'submission.json', receipt)
    job = sbatch(script('worker', ['#SBATCH --partition=V100', '#SBATCH --gres=gpu:1',
        '#SBATCH --cpus-per-task=4', '#SBATCH --mem=24G', '#SBATCH --hold',
        '#SBATCH --job-name=al-dense-V100-extra', '#SBATCH --time='+campaign['shooting_config']['walltime'],
        f'#SBATCH --array=0-{c["additional_workers"]-1}%{c["additional_workers"]}',
        f'#SBATCH --output={expansion}/worker-%A_%a.log']), expansion/'workers.sbatch')
    receipt.update(state='workers_held', worker_job=job)
    write_json(expansion/'submission.json', receipt)
    collector = sbatch(script('reconcile', ['#SBATCH --partition=CPU', '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=8G', '#SBATCH --time=04:00:00', '#SBATCH --job-name=al-shot-collect',
        f'#SBATCH --dependency=afterany:{old_job}:{job}',
        f'#SBATCH --output={expansion}/collect-%j.log']), expansion/'collect.sbatch')
    receipt.update(state='submitted', collect_job=collector, additional_workers=c['additional_workers'],
        additional_branches=len(moved), retained_original_entries=len(retained), submitted_at=now())
    write_json(expansion/'submission.json', receipt)
    # Replace only this chat's old pending collector; it must not inspect live new workers.
    if queued(old_collector) != old_collector+' PENDING':
        raise ValueError('Original collector changed during submission')
    subprocess.run(['scancel', old_collector], check=True)
    subprocess.run(['scontrol', 'release', job], check=True)
    return receipt


def worker(expansion):
    plan, parent, campaign = verify(expansion)
    check_runtime(parent)
    gate = read(parent/'technical/audit/validation.json')
    lane = int(os.environ['SLURM_ARRAY_TASK_ID']); start = time.monotonic()
    config = campaign['shooting_config']; completed = []; skipped = []
    def remaining():
        return config['walltime_seconds'] - (time.monotonic()-start) - config['cleanup_reserve_seconds']
    def progress(state, **extra):
        write_json(expansion/f'lane-{lane}.json', dict(state=state, job_id=os.environ['SLURM_JOB_ID'],
            completed=completed, not_started=skipped, elapsed_seconds=time.monotonic()-start, **extra))
    for i in plan['assignments'][lane]:
        record = deepcopy(campaign['runs'][i])
        estimate = 2*gate['seconds_per_step']*(record['stopping']['maximum_measurement_steps']+
            campaign['config']['equilibration_steps']) + 1800
        if i == plan['boundary_index']:
            progress('waiting_for_original_boundary', current=record['run_id'])
            while remaining() >= estimate and queued(plan['original_worker_job']):
                time.sleep(30)
        if remaining() < estimate:
            skipped.append(record['run_id']); continue
        if i == plan['boundary_index']:
            status = Path(campaign['root'])/record['run_dir']/'status.json'
            value = read(status)
            if value['state'] != 'expansion_reserved' or value['expansion'] != str(expansion):
                raise ValueError('Boundary reservation changed')
            status.unlink()  # Original job has ended; it can no longer start this branch.
        record['execution_timeout_seconds'] = max(1, int(remaining()-900))
        progress('running', current=record['run_id'])
        run_source(campaign, record)
        completed.append(record['run_id']); progress('running')
    progress('finished')


def reconcile(expansion):
    plan, parent, campaign = verify(expansion)
    receipt = read(expansion/'submission.json')
    for job in (plan['original_worker_job'], receipt['worker_job']):
        if queued(job):
            raise ValueError(f'Cannot collect while workers remain queued/running: {job}')
    boundary = Path(campaign['root'])/campaign['runs'][plan['boundary_index']]['run_dir']/'status.json'
    if boundary.exists() and read(boundary)['state'] == 'expansion_reserved':
        if read(boundary)['expansion'] != str(expansion):
            raise ValueError('Foreign reservation')
        boundary.unlink()  # Budget-limited, genuinely unstarted branch.
    result = collect(parent)
    write_json(expansion/'completion.json', dict(result, reconciled_at=now()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'worker', 'reconcile'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--expansion', type=Path)
    args = parser.parse_args()
    if args.action == 'submit':
        result = submit(load_json(args.config))
    else:
        result = globals()[args.action](args.expansion.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
