"""Replace reserved, unstarted V100 shoots with continuous birth-focused runs."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from src.project_runtime.paths import REPO, load_json, storage_path
from src.project_runtime.transfer import write_json
from .birth_sources import now, register_entry, sha256
from .dense_al import manifest_at, run_source, sbatch
from .dense_al_gpu import execute_gpu
from .dense_al_shooting import check_runtime, collect as collect_sources, environment
from .dense_al_shooting_expand import queued
from .dense_al_birth_stop import POLICY, PROTOCOL, source_input


def read(path):
    return json.loads(Path(path).read_text())


def prepare(c):
    if (c['protocol'], c['record_from_initialization'], c['sampling_ps'], c['monitor_ps'],
        c['stopping_cluster_atoms'], c['followup_ps']) != (PROTOCOL, True, .1, 3., 128, 12.):
        raise ValueError('Recipe differs from the declared birth-focused protocol')
    launch = storage_path('archive')/'simulation-launches'/c['campaign_id']
    handoff = read(launch/'handoff.json')
    if handoff['state'] != 'reserved' or (launch/'manifest.json').exists():
        raise ValueError('Requires the explicit unstarted-source reservation receipt')
    parent = Path(handoff['parent_launch']); old = manifest_at(parent)
    if sha256(parent/'manifest.json') != handoff['parent_manifest_sha256']:
        raise ValueError('Original campaign changed after handoff reservation')
    evidence = read(parent/'technical/birth-focus-audit/evidence.json')
    initial = {r['source_id']: r for r in evidence['parents']}
    root = storage_path('simulation_runs')/c['campaign_id']; root.mkdir()
    config = deepcopy(old['config']); records = []; old_to_new = {}
    for reservation in handoff['reservations']:
        i = reservation['index']; previous = old['runs'][i]
        origin = Path(old['root'])/previous['run_dir']
        actual = read(origin/'status.json')
        expected = reservation.get('status', reservation.get('existing_reservation'))
        if actual != expected:
            raise ValueError(f'Reservation changed; do not duplicate a started source: {origin}')
        for name, digest in previous['input_sha256'].items():
            if sha256(origin/name) != digest:
                raise ValueError(f'Original prepared input changed: {origin/name}')
        qc = initial[previous['source_id']]
        if qc['largest_cluster_atoms'] >= 8 or qc['sha256'] != previous['input_sha256']['prepared_liquid.lammps.data']:
            raise ValueError(f'Initial liquid QC failed for source {previous["source_id"]}')
        identifier = f'{c["campaign_id"]}-source{previous["source_id"]:04d}-shot{previous["shot_index"]:02d}'
        r = deepcopy(previous); r.pop('input_sha256')
        maximum = previous['stopping']['maximum_measurement_steps']+old['config']['equilibration_steps']
        r.update(run_id=identifier, run_dir='runs/'+identifier, protocol=PROTOCOL,
            collection_id=c['campaign_id'], replaces_unstarted_run=previous['run_id'],
            stopping=dict(POLICY, maximum_measurement_steps=maximum,
                maximum_measurement_ps=maximum*.002,
                interpretation='Online cluster-growth proxy only; full-cadence lineage search determines birth origins and fates'))
        directory = root/r['run_dir']
        (directory/'potential').mkdir(parents=True)
        (directory/'technical/stop-monitor').mkdir(parents=True)
        for name in ('prepared_liquid.lammps.data', 'melt_final.restart.bin', 'parent_source.in.lammps',
                     'potential/Lee2003_Al.library.meam', 'potential/Lee2003_Al.meam'):
            shutil.copy2(origin/name, directory/name)
        meta = read(origin/'input_metadata.json'); meta.update(r)
        meta.update(equilibration_steps=0, initial_settling_steps_recorded=7500,
            observation_start_step=0, observation_origin='Velocity initialization; first 15 ps retained',
            measurement_steps=maximum, measurement_ps=maximum*.002,
            initial_melt_qc=qc,
            interpretation='Continuous velocity reshoot from prepared liquid; early growth proxy stops collection, not a verified birth label')
        meta['execution'].update(validation=str(parent/'technical/audit/validation.json'),
            control_path_audit=str(launch/'technical'),
            initial_state='Full-precision prepared-liquid coordinates, fresh Maxwell velocities and NPT bath; record before any integration')
        write_json(directory/'input_metadata.json', meta)
        write_json(directory/'technical/launch_config.json', config)
        (directory/'source.in.lammps').write_text(source_input(config, r, directory))
        r['input_sha256'] = {str(p.relative_to(directory)): sha256(p)
            for p in directory.rglob('*') if p.is_file()}
        old_to_new[i] = len(records); records.append(r)
    lanes = []
    for lane in handoff['lanes']:
        deadline = datetime.fromisoformat(lane['old_end_time']).replace(tzinfo=ZoneInfo('Europe/Paris'))
        lanes.append(dict(predecessor=lane['predecessor'], deadline_utc=deadline.astimezone(timezone.utc).isoformat(),
            assignments=[old_to_new[i] for i in lane['transferred_indices']]))
    shooting = dict(old['shooting_config'], campaign_id=c['campaign_id'], workers=len(lanes))
    plan = dict(protocol=PROTOCOL, sampling_protocol=old['sampling_protocol'], created_at=now(),
        root=str(root), launch_root=str(launch), config=config, shooting_config=shooting,
        birth_config=c, parent_launch=str(parent), runs=records, lanes=lanes,
        initial_melt_qc=evidence, handoff_sha256=sha256(launch/'handoff.json'))
    write_json(launch/'manifest.json', plan)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    (root/'manifest.json').symlink_to(launch/'manifest.json')
    register_entry(c['campaign_id'], dict(root='simulation_runs', path=c['campaign_id'], kind='simulation',
        dependencies=[old['shooting_config']['campaign_id']], metadata=dict(materials=['Al'],
            potential_ids=['al-lee2003-meam'], role='raw_dynamics', classification='research',
            source_origin='shooting', sampling_protocol=old['sampling_protocol'],
            title='Birth-focused continuous Al V100 velocity reshoots',
            description=f'{len(records)} transferred unstarted branches; Q/V from initialization every 0.10 ps, early cluster follow-up stop.',
            ancestry='Existing prepared melts and original roles/seeds retained; no additional independent ancestors.',
            evidence=[str(launch/'manifest.json'), str(launch/'submission.json'), str(launch/'status.json')])))
    write_json(launch/'status.json', dict(state='prepared', planned=len(records), completed=0, birth_labels='not_searched'))
    return str(launch)


def freeze(launch):
    plan = manifest_at(launch); parent = Path(plan['parent_launch'])
    old_inventory = read(parent/'code_sha256.json')
    for name, digest in old_inventory.items():
        if sha256(parent/'code'/name) != digest:
            raise ValueError(f'Original validated code changed: {name}')
    code = launch/'code'
    shutil.copytree(parent/'code', code, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    overlays = ['src/simulation/campaigns/dense_al.py',
        'src/simulation/campaigns/dense_al_birth_stop.py',
        'src/simulation/campaigns/dense_al_birth_queue.py',
        'src/simulation/campaigns/dense_al_shooting_expand.py',
        'src/data/conversion/dense_al_birth.py']
    for name in overlays:
        shutil.copy2(REPO/name, code/name)
    # The shared working CLI also has unrelated new commands. Extend only the
    # actual frozen dispatcher, so this snapshot has no missing dependencies.
    dispatcher = code/'src/data/conversion/cli.py'
    text = dispatcher.read_text()
    needle = '    "dense-al-half": "src.data.conversion.dense_al_half",\n'
    if text.count(needle) != 1:
        raise ValueError('Unexpected original converter dispatcher')
    dispatcher.write_text(text.replace(needle, needle+'    "dense-al-birth": "src.data.conversion.dense_al_birth",\n'))
    inventory = {str(p.relative_to(code)): sha256(p) for p in code.rglob('*') if p.is_file() and not p.is_symlink()}
    changed = sorted(n for n in old_inventory if inventory[n] != old_inventory[n])
    if changed != ['src/data/conversion/cli.py', 'src/simulation/campaigns/dense_al.py']:
        raise ValueError(f'Unexpected change to validated execution code: {changed}')
    write_json(launch/'code_sha256.json', inventory)
    (launch/'technical').mkdir(exist_ok=True)
    gate = read(parent/'technical/audit/validation.json')
    if gate['state'] != 'passed' or gate['code_inventory_sha256'] != sha256(parent/'code_sha256.json'):
        raise ValueError('Original numerical gate is not valid')
    write_json(launch/'technical/numerical-validation-reuse.json', dict(state='passed',
        source=str(parent/'technical/audit/validation.json'), source_sha256=sha256(parent/'technical/audit/validation.json'),
        seconds_per_step=gate['seconds_per_step'], unchanged_original_files=len(old_inventory)-len(changed),
        changed_original_files=changed, added_files=sorted(set(inventory)-set(old_inventory)),
        scope='Unchanged force, virial and native-NPT adapter. New observation/stopping/conversion path requires per-lane execution audit.',
        code_inventory_sha256=sha256(launch/'code_sha256.json')))


def check_entrypoint(launch, env, python):
    """Import the actual frozen worker and converter before reserving GPUs."""
    program = '''import json,sys
from pathlib import Path
from src.simulation.campaigns import dense_al_birth_queue, dense_al_birth_stop, dense_al_shooting_expand
from src.data.conversion import dense_al_birth
from src.simulation.campaigns.dense_al import verify_code
verify_code(Path(sys.argv[1]))
print(json.dumps(dict(python=sys.executable, modules={m.__name__:m.__file__ for m in
    [dense_al_birth_queue,dense_al_birth_stop,dense_al_shooting_expand,dense_al_birth]})))
'''
    result = subprocess.run([python, '-c', program, str(launch)], cwd=launch/'code',
        env=dict(os.environ, **env), text=True, capture_output=True)
    receipt = dict(state='passed' if result.returncode == 0 else 'failed',
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        code_inventory_sha256=sha256(launch/'code_sha256.json'), checked_at=now())
    write_json(launch/'technical/entrypoint-check.json', receipt)
    if result.returncode:
        raise RuntimeError(f'Frozen Al worker import failed: {result.stderr}')
    return receipt


def submit(launch):
    plan = manifest_at(launch); handoff = read(launch/'handoff.json')
    if (launch/'submission.json').exists():
        raise FileExistsError('Preserve submitted birth-focused queue')
    freeze(launch)
    parent = Path(plan['parent_launch'])
    # Keep the already compiled sm70 cache; inherited environment otherwise
    # points to the new frozen code. No force recompilation for a policy change.
    env, profile = environment(parent); env['PYTHONPATH'] = str(launch/'code')
    env['PYTHONNOUSERSITE'] = '1'
    check_entrypoint(launch, env, profile['python'])
    base = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --no-requeue',
        f'#SBATCH --chdir={launch}/code']
    body = ['set -euo pipefail', 'export '+shlex.join([k+'='+v for k, v in env.items()])]
    command = [profile['python'], '-u', '-m', 'src.simulation.campaigns.dense_al_birth_queue']
    def script(action, options, extra=()):
        return '\n'.join(base+options+body+['exec '+shlex.join(command+[action, '--launch', str(launch), *extra]), ''])
    receipt = dict(state='submitting', workers=[], previous_collector=handoff['previous_collector'],
        created_at=now(), budget='Original per-lane absolute deadlines, no extension')
    write_json(launch/'submission.json', receipt)
    for i, lane in enumerate(plan['lanes']):
        remaining = math.floor((datetime.fromisoformat(lane['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
        if remaining <= 3600:
            raise ValueError(f'Insufficient remaining budget for lane {i}')
        wall = f'{remaining//3600:02d}:{remaining//60%60:02d}:{remaining%60:02d}'
        job = sbatch(script('worker', ['#SBATCH --partition=V100', '#SBATCH --gres=gpu:1',
            '#SBATCH --cpus-per-task=4', '#SBATCH --mem=24G', '#SBATCH --hold',
            '#SBATCH --job-name=al-birth-V100', '#SBATCH --time='+wall,
            '#SBATCH --dependency=afterany:'+lane['predecessor'],
            f'#SBATCH --output={launch}/technical/worker-{i}-%j.log'], ['--lane', str(i)]),
            launch/f'worker-{i}.sbatch')
        receipt['workers'].append(dict(lane=i, job=job, predecessor=lane['predecessor'],
            deadline_utc=lane['deadline_utc'], sources=len(lane['assignments'])))
        write_json(launch/'submission.json', receipt)
    collector = sbatch(script('collect', ['#SBATCH --partition=CPU', '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=8G', '#SBATCH --time=04:00:00', '#SBATCH --job-name=al-birth-collect',
        '#SBATCH --dependency=afterany:'+':'.join(str(r['job']) for r in receipt['workers']),
        f'#SBATCH --output={launch}/technical/collect-%j.log']), launch/'collect.sbatch')
    receipt.update(state='submitted', collect_job=collector, submitted_at=now())
    write_json(launch/'submission.json', receipt)
    if queued(handoff['previous_collector']) != handoff['previous_collector']+' PENDING':
        raise ValueError('Previous collector changed; replacements remain held')
    subprocess.run(['scancel', handoff['previous_collector']], check=True)
    for worker in receipt['workers']:
        subprocess.run(['scontrol', 'release', str(worker['job'])], check=True)
    return receipt


def repair(launch):
    """Replace the failed import-only launch, preserving its code and receipts."""
    plan = manifest_at(launch); previous = read(launch/'submission.json')
    jobs = [str(r['job']) for r in previous['workers']] + [str(previous['collect_job'])]
    jobs += [lane['predecessor'] for lane in plan['lanes']]
    for job in jobs:
        if queued(job):
            raise ValueError(f'Import repair requires all previous jobs to have ended: {job}')
    error = "ModuleNotFoundError: No module named 'src.simulation.campaigns.dense_al_shooting_expand'"
    logs = [launch/f'technical/worker-{r["lane"]}-{r["job"]}.log' for r in previous['workers']]
    if any(error not in p.read_text() for p in logs):
        raise ValueError('This repair applies only to the recorded missing-module startup failure')
    for r in plan['runs']:
        if (Path(plan['root'])/r['run_dir']/'status.json').exists():
            raise ValueError(f'Refusing to duplicate a started Al trajectory: {r["run_id"]}')
    inventory = read(launch/'code_sha256.json')
    for name, expected in inventory.items():
        if sha256(launch/'code'/name) != expected:
            raise ValueError(f'Failed launch code changed: {name}')
    added = 'src/simulation/campaigns/dense_al_shooting_expand.py'
    if (launch/'code'/added).exists():
        raise ValueError('Expected the documented missing dependency')
    gate = read(launch/'technical/numerical-validation-reuse.json')
    if gate['state'] != 'passed' or gate['code_inventory_sha256'] != sha256(launch/'code_sha256.json'):
        raise ValueError('Failed launch has no matching reusable numerical audit')
    if sha256(gate['source']) != gate['source_sha256']:
        raise ValueError('Original numerical audit changed')
    repaired = launch/'technical/dependency-repair-01'
    repaired.mkdir(); (repaired/'technical').mkdir()
    shutil.copytree(launch/'code', repaired/'code', symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(REPO/added, repaired/'code'/added)
    new_inventory = dict(inventory, **{added:sha256(repaired/'code'/added)})
    write_json(repaired/'code_sha256.json', new_inventory)
    recovered = deepcopy(plan); recovered['launch_root'] = str(repaired)
    write_json(repaired/'manifest.json', recovered)
    (repaired/'manifest.sha256').write_text(sha256(repaired/'manifest.json')+'\n')
    shutil.copy2(launch/'handoff.json', repaired/'handoff.json')
    write_json(repaired/'technical/numerical-validation-reuse.json', dict(gate,
        previous_reuse_receipt=str(launch/'technical/numerical-validation-reuse.json'),
        previous_reuse_sha256=sha256(launch/'technical/numerical-validation-reuse.json'),
        repair_added_files=[added], code_inventory_sha256=sha256(repaired/'code_sha256.json')))
    write_json(repaired/'repair.json', dict(state='prepared', original_launch=str(launch),
        authorization='Current chat: user requested fixing V100 shooting after the missing-module failure',
        original_manifest_sha256=sha256(launch/'manifest.json'),
        original_submission_sha256=sha256(launch/'submission.json'),
        original_code_inventory_sha256=sha256(launch/'code_sha256.json'),
        failure_logs={str(p):sha256(p) for p in logs}, added_files={added:new_inventory[added]},
        unchanged_existing_files=len(inventory), changed_existing_files=[],
        protocol='Identical sources, seeds, inputs, force/integrator, observation/stopping and absolute deadlines',
        created_at=now()))
    env, profile = environment(Path(plan['parent_launch']))
    env.update(PYTHONPATH=str(repaired/'code'), PYTHONNOUSERSITE='1')
    check_entrypoint(repaired, env, profile['python'])
    base = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --no-requeue',
        f'#SBATCH --chdir={repaired}/code', '#SBATCH --hold']
    body = ['set -euo pipefail', 'export '+shlex.join([k+'='+v for k,v in env.items()])]
    command = [profile['python'], '-u', '-m', 'src.simulation.campaigns.dense_al_birth_queue']
    def script(action, options, extra=()):
        return '\n'.join(base+options+body+['exec '+shlex.join(command+
            [action, '--launch', str(repaired), *extra]), ''])
    receipt = dict(state='submitting', workers=[], replaces_submission=str(launch/'submission.json'),
        created_at=now(), budget='Original per-lane absolute deadlines, no extension')
    write_json(repaired/'submission.json', receipt)
    created = []
    try:
        for i, lane in enumerate(plan['lanes']):
            remaining = math.floor((datetime.fromisoformat(lane['deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
            if remaining <= 3600:
                raise ValueError(f'Insufficient original budget for lane {i}')
            wall = f'{remaining//3600:02d}:{remaining//60%60:02d}:{remaining%60:02d}'
            job = sbatch(script('worker', ['#SBATCH --partition=V100', '#SBATCH --gres=gpu:1',
                '#SBATCH --cpus-per-task=4', '#SBATCH --mem=24G', '#SBATCH --job-name=al-birth-V100',
                '#SBATCH --time='+wall, f'#SBATCH --output={repaired}/technical/worker-{i}-%j.log'],
                ['--lane',str(i)]), repaired/f'worker-{i}.sbatch')
            created.append(job)
            receipt['workers'].append(dict(lane=i, job=job, predecessor=lane['predecessor'],
                deadline_utc=lane['deadline_utc'], sources=len(lane['assignments'])))
            write_json(repaired/'submission.json', receipt)
        collector = sbatch(script('collect', ['#SBATCH --partition=CPU', '#SBATCH --cpus-per-task=1',
            '#SBATCH --mem=8G', '#SBATCH --time=04:00:00', '#SBATCH --job-name=al-birth-collect',
            '#SBATCH --dependency=afterany:'+':'.join(created),
            f'#SBATCH --output={repaired}/technical/collect-%j.log']), repaired/'collect.sbatch')
        created.append(collector); receipt.update(state='submitted', collect_job=collector, submitted_at=now())
        write_json(repaired/'submission.json', receipt)
    except BaseException:
        # Only jobs created held by this repair invocation are canceled.
        for job in created:
            subprocess.run(['scancel',job],check=True)
        receipt['state']='submission_failed'; write_json(repaired/'submission.json',receipt)
        raise
    write_json(launch/'repair.json',dict(state='submitted', launch=str(repaired),
        submission_sha256=sha256(repaired/'submission.json'), created_at=now()))
    for job in created:
        subprocess.run(['scontrol','release',job],check=True)
    return dict(launch=str(repaired), **receipt)


def execution_audit(launch, lane):
    plan = manifest_at(launch); r = deepcopy(plan['runs'][plan['lanes'][lane]['assignments'][0]])
    original = Path(plan['root'])/r['run_dir']
    directory = launch/'technical'/f'execution-audit-{lane}'
    (directory/'potential').mkdir(parents=True)
    (directory/'technical/stop-monitor').mkdir(parents=True)
    for name in ('prepared_liquid.lammps.data', 'potential/Lee2003_Al.library.meam', 'potential/Lee2003_Al.meam'):
        shutil.copy2(original/name, directory/name)
    meta = read(original/'input_metadata.json')
    r['run_id'] += '-execution-audit'; r['stopping']['maximum_measurement_steps'] = 3000
    r['stopping']['maximum_measurement_ps'] = 6.
    meta.update(run_id=r['run_id'], stopping=r['stopping'], measurement_steps=3000, measurement_ps=6.,
                scientific_role='execution_audit_excluded_from_datasets')
    write_json(directory/'input_metadata.json', meta)
    (directory/'source.in.lammps').write_text(source_input(plan['config'], r, directory))
    seconds = execute_gpu(directory, timeout_seconds=900)
    endpoint = read(directory/'technical/stop-monitor/outcome.json')
    if endpoint['final_step'] != 3000 or endpoint['state'] != 'dynamics_complete':
        raise ValueError('Execution audit did not reach its declared endpoint')
    if [v['step'] for v in read(directory/'technical/stop-monitor/history.json')] != [0, 1500, 3000]:
        raise ValueError('Execution audit did not exercise continuous monitoring')
    meta.update(state='dynamics_complete', source_sha256=sha256(directory/'trajectory.lammpstrj'))
    write_json(directory/'metadata.json', meta)
    converter = launch/'code/scripts/convert_trajectory.py'
    with (directory/'conversion.log').open('x') as log:
        subprocess.run([sys.executable, str(converter), 'dense-al-birth', str(directory), '--delete-source'],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    report = read(directory/'paired_conversion.json')
    if report['frame_count'] != 61 or not (directory/'final.restart.bin').stat().st_size:
        raise ValueError('Execution audit timeline/restart is invalid')
    result = dict(state='passed', frame_count=61, first_step=0, last_step=3000,
        dynamics_seconds=seconds, completed_at=now(), quantization=report['quantization'],
        code_inventory_sha256=sha256(launch/'code_sha256.json'))
    write_json(directory/'audit.json', result)
    return result


def worker(launch, lane):
    check_runtime(launch)
    plan = manifest_at(launch); assignment = plan['lanes'][lane]
    gate = read(launch/'technical/numerical-validation-reuse.json')
    if gate['state'] != 'passed' or gate['code_inventory_sha256'] != sha256(launch/'code_sha256.json'):
        raise ValueError('Numerical validation no longer matches frozen code')
    deadline = datetime.fromisoformat(assignment['deadline_utc'])
    def remaining():
        return (deadline-datetime.now(timezone.utc)).total_seconds()-1800
    complete = []; skipped = []
    def progress(state, **values):
        write_json(launch/f'technical/lane-{lane}.json', dict(state=state, completed=complete,
            not_started=skipped, job_id=os.environ['SLURM_JOB_ID'], deadline_utc=assignment['deadline_utc'],
            remaining_seconds=remaining(), updated_at=now(), **values))
    if queued(assignment['predecessor']):
        raise ValueError('Predecessor still queued/running; refusing simultaneous workers')
    if remaining() < 3600:
        skipped.extend(plan['runs'][i]['run_id'] for i in assignment['assignments'])
        progress('budget_exhausted'); return
    progress('execution_audit'); execution_audit(launch, lane)
    for i in assignment['assignments']:
        r = deepcopy(plan['runs'][i])
        estimate = 2*gate['seconds_per_step']*r['stopping']['maximum_measurement_steps']+1800
        if remaining() < estimate:
            skipped.append(r['run_id']); continue
        r['execution_timeout_seconds'] = max(1, int(remaining()-900))
        progress('running', current=r['run_id'])
        run_source(plan, r)
        complete.append(r['run_id']); progress('running')
    progress('finished')


def collect(launch):
    plan = manifest_at(launch); receipt = read(launch/'submission.json'); handoff = read(launch/'handoff.json')
    for job in [r['job'] for r in receipt['workers']]+[l['predecessor'] for l in plan['lanes']]:
        if queued(job):
            raise ValueError(f'Cannot collect while simulation worker is live: {job}')
    result = collect_sources(launch)
    # An earlier queue-expansion marker remains valid until both generations
    # have ended. Only now turn it into the same explicit transfer record.
    for reserved in handoff['reservations']:
        if 'existing_reservation' in reserved:
            path = Path(reserved['path'])
            if read(path) != reserved['existing_reservation']:
                raise ValueError('Original expansion boundary changed')
            write_json(path, dict(state='birth_focus_transferred', new_launch=str(launch),
                original_run_id=reserved['run_id'], reserved_at=handoff['created_at']))
    old = collect_sources(Path(plan['parent_launch']))
    result['preserved_original_campaign'] = dict(counts=old['counts'], launch=plan['parent_launch'])
    result['birth_labels'] = 'not_searched'
    write_json(launch/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'submit', 'repair', 'worker', 'collect'])
    p.add_argument('--config', type=Path); p.add_argument('--launch', type=Path)
    p.add_argument('--lane', type=int)
    a = p.parse_args()
    if a.action == 'prepare': result = prepare(load_json(a.config))
    elif a.action == 'worker': result = worker(a.launch.resolve(), a.lane)
    else: result = globals()[a.action](a.launch.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
