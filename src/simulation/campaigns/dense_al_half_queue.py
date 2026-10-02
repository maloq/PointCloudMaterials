"""Apply halfway/peer stopping only to unstarted records in the active Al queue."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from src.project_runtime.paths import REPO, catalog, storage_path
from src.project_runtime.transfer import write_json
from .birth_sources import now, sha256, register_entry, freeze_code
from .dense_al import manifest_at, sbatch
from .dense_al_half_stop import PROTOCOLS, source_input


def prepare_half_queue(config):
    previous = Path(config['source_launch'])
    original = manifest_at(previous)
    prior_status = json.loads((previous / 'status.json').read_text())
    wave = prior_status['wave']
    old_wave = previous / f'wave-{wave:03d}'
    submission = json.loads((old_wave / 'submission.json').read_text())
    controller, workers = submission['controller_job'], submission['worker_job']
    # The add-on queue retains the active worker script under this explicit
    # producer path; its own controller is the only newly submitted script.
    retained_workers = old_wave / 'previous-workers.sbatch'
    retained_controller = old_wave / 'controller.sbatch'
    retained_workers.read_text()
    retained_controller.read_text()
    if prior_status['state'] != 'waiting_for_active_wave' or prior_status['controller_job'] != controller:
        raise ValueError('Require the explicitly recorded pending successor of the current Al wave')
    pending = subprocess.check_output(['squeue', '-h', '-j', controller, '-o', '%i %t %j'], text=True).strip()
    if pending != f'{controller} PD al-dense-next':
        raise ValueError(f'Current successor is not our pending controller: {pending}')
    validation = json.loads(Path(config['validation_receipt']).read_text())
    if validation['state'] != 'passed' or set(validation['cases']) != {'half_010ps', 'cap_010ps', 'half_001ps', 'cap_001ps'}:
        raise ValueError('All native stop/cap and both-cadence conversion checks must pass before activation')
    for filename, digest in validation['implementation_sha256'].items():
        if sha256(REPO / filename) != digest:
            raise ValueError(f'Validated implementation changed: {filename}')
    expected_caps = {'400': 600, '450': 291, '500': 411, '510': 411, '520': 561}
    if config['maximum_measurement_ps_by_temperature'] != expected_caps:
        raise ValueError('Peer caps differ from the reviewed half-crystal source-cohort audit')
    launch = storage_path('archive') / 'simulation-launches' / config['launch_id']
    root = Path(original['root'])
    collection = root / config['collection_id']
    if launch.exists() or collection.exists():
        raise FileExistsError('Preserve any existing halfway execution preparation')
    if config['collection_id'] in catalog():
        raise ValueError('Preserve and reconcile the existing halfway collection registration before preparation')
    environment = json.loads((previous / 'environment.json').read_text())
    if sha256(Path(sys.prefix) / 'bin/lmp') != environment['lammps_sha256']:
        raise ValueError('Use the same native LAMMPS executable as the existing campaign')
    old_assignments = json.loads((old_wave / 'assignments.json').read_text())
    plan = deepcopy(original)
    protected, changed, reserved = [], [], []
    for i, record in enumerate(plan['runs']):
        directory = root / record['run_dir']
        path = directory / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else None
        if status is not None and status['state'] in {'complete', 'running'}:
            protected.append(i)
            continue
        if status is not None and not (status['state'] == 'queue_handoff' and
                status['successor_launch'] == str(launch)):
            raise ValueError(f'Unexpected source state; preserve and inspect: {directory}, {status}')
        if (directory / 'source.stdout.log').exists() or 'recovery' in record:
            raise ValueError(f'Unstarted-source conversion refuses evolved dynamics: {directory}')
        for name, digest in record['input_sha256'].items():
            if sha256(directory / name) != digest:
                raise ValueError(f'Original prepared input changed: {directory / name}')
        changed.append(i)
        if status is not None:
            reserved.append(i)
    if len(plan['runs']) != 152 or len(changed) != config['expected_unstarted_trajectories']:
        raise ValueError(f'Availability changed: {len(changed)} unstarted, {len(protected)} protected; review a new capture')
    launch.mkdir()
    collection.mkdir()
    write_json(launch / 'status.json', dict(state='preparing', created_at=now()))
    shutil.copy2(config['validation_receipt'], launch / 'validation.json')
    for i in changed:
        old = original['runs'][i]
        old_directory = root / old['run_dir']
        record = plan['runs'][i]
        interval = record.get('sample_interval_steps', 50)
        cap_ps = config['maximum_measurement_ps_by_temperature'][str(int(record['temperature_K']))]
        cap_steps = cap_ps * 500
        record.pop('input_sha256')
        record.update(previous_prepared_run_dir=old['run_dir'],
            run_dir=f'{config["collection_id"]}/runs/{record["run_id"]}',
            protocol=PROTOCOLS[interval], sample_interval_steps=interval,
            failure_id=record['run_id'] + '-half-stop-failed',
            stopping=dict(crystal_fraction=.5, monitor_interval_steps=7500,
                confirmation='Two consecutive complete 15-ps monitoring observations at or above 50%',
                prediction_tail_steps=3000, maximum_measurement_steps=cap_steps,
                maximum_measurement_ps=cap_ps, peer_fraction=.9,
                peer90_observed_in_reference=record['temperature_K'] != 400,
                peer_reference=str(Path(config['peer_reference']) / 'tables/peer_cutoffs.csv'),
                peer_reference_sha256=sha256(Path(config['peer_reference']) / 'tables/peer_cutoffs.csv'),
                maximum_selection='Reviewed full historical same-temperature cohort; 400 K quorum unobserved, retain 600 ps',
                half_event_censoring='Cap without confirmation is right-censored, never a permanent-liquid label',
                input_dtype='Established float32 consumer of native full-precision monitoring snapshot'))
        directory = root / record['run_dir']
        (directory / 'potential').mkdir(parents=True)
        (directory / 'technical/stop-monitor').mkdir(parents=True)
        for name in ('prepared_liquid.lammps.data', 'melt_final.restart.bin', 'parent_source.in.lammps',
                     'potential/Lee2003_Al.library.meam', 'potential/Lee2003_Al.meam'):
            shutil.copy2(old_directory / name, directory / name)
            if sha256(directory / name) != old['input_sha256'][name]:
                raise ValueError(f'Halfway preparation copy differs: {directory / name}')
        (directory / 'source.in.lammps').write_text(source_input(plan['config'], record, directory))
        metadata = json.loads((old_directory / 'input_metadata.json').read_text())
        metadata.pop('expected_frame_count', None)
        metadata.update(record)
        metadata.update(state='prepared', protocol=record['protocol'],
            sample_interval_steps=interval, sampling_ps=interval * .002,
            measurement_steps=cap_steps, measurement_ps=cap_ps,
            maximum_frame_count=cap_steps // interval + 1,
            duration_policy='Variable endpoint: confirmed halfway plus 6 ps tail, or declared peer cap',
            stopping_rule=record['stopping'], positions=True, velocities=True,
            previous_preparation_manifest_sha256=sha256(previous / 'manifest.json'))
        if interval == 5:
            metadata['sampling_policy'] = 'Save every five 2-fs steps through the recorded variable endpoint'
        write_json(directory / 'input_metadata.json', metadata)
        write_json(directory / 'technical/launch_config.json', dict(plan['config'], stopping=record['stopping']))
        record['input_sha256'] = {str(p.relative_to(directory)): sha256(p)
                                  for p in directory.rglob('*') if p.is_file()}
    plan.update(protocol='al_dense_half_stop_queue_v1', created_at=now(), launch_root=str(launch),
        stopping_version_of=dict(launch=str(previous), manifest_sha256=sha256(previous / 'manifest.json')),
        protection_policy='All already-running/complete source records and native inputs are identical to the prior queue',
        protected_indices=protected, half_stop_indices=changed,
        half_stop_policy=config, peer_reference_sha256=sha256(Path(config['peer_reference']) / 'tables/peer_cutoffs.csv'))
    plan['previous_additional_observation_protocols'] = deepcopy(original['additional_observation_protocols'])
    plan['additional_observation_protocols'] = {
        PROTOCOLS[5]: dict(timestep_ps=.002, sampling_ps=.01, sample_interval_steps=5,
            source_ids=[plan['runs'][i]['source_id'] for i in changed if plan['runs'][i]['sample_interval_steps'] == 5],
            fixed_duration=False, maximum_ps_by_temperature=expected_caps,
            storage_dtype='float16', positions=True, velocities=True,
            protocol='Confirmed half-crystal plus 6 ps tail, or explicit peer-time cap')}
    plan['config']['stopping_policy'] = 'Per-record versioned halfway/peer stopping for unstarted sources; protected sources retain original fixed-duration inputs'
    for i in protected:
        if plan['runs'][i] != original['runs'][i]:
            raise ValueError('Protected running/completed scientific record changed')
    write_json(launch / 'manifest.json', plan)
    (launch / 'manifest.sha256').write_text(sha256(launch / 'manifest.json') + '\n')
    selected_collection = deepcopy(plan)
    selected_collection.update(root=str(collection), runs=[deepcopy(plan['runs'][i]) for i in changed],
        roles=dict(Counter(plan['runs'][i]['split'] for i in changed)),
        trajectory_roles=dict(Counter(plan['runs'][i]['split'] for i in changed)),
        independent_ancestor_count=len({plan['runs'][i]['root_lineage'] for i in changed}),
        main_source_count=sum(i < 150 for i in changed), supplemental_source_count=sum(i >= 150 for i in changed))
    for record in selected_collection['runs']:
        record['run_dir'] = str(Path(record['run_dir']).relative_to(config['collection_id']))
    write_json(launch / 'half_stop_collection.json', selected_collection)
    (collection / 'manifest.json').symlink_to(launch / 'half_stop_collection.json')
    register_entry(config['collection_id'], dict(root='simulation_runs',
        path=str(collection.relative_to(storage_path('simulation_runs'))), kind='simulation',
        dependencies=sorted({plan['runs'][i]['parent_dataset'] for i in changed} | {'potentials', original['config']['campaign_id']}),
        metadata=dict(materials=['Al'], potential_ids=['al-lee2003-meam'], role='raw_dynamics', classification='research',
            title='Unstarted main Al descendants with confirmed-halfway stopping and peer caps',
            description='128 variable-duration 70304-atom trajectories, 126 main and two dense additions; 2 fs integration, exact 0.1/0.01 ps observations and velocities.',
            ancestry='126 retained independent melt ancestors, including two extra observations; no new independent lineages.',
            protocol='al_dense_half_stop_queue_v1',
            evidence=[str(launch / 'manifest.json'), str(launch / 'validation.json'), str(launch / 'status.json')])))
    wave_new = launch / f'wave-{wave:03d}'
    wave_new.mkdir()
    filtered = [[i for i in lane if i in protected] for lane in old_assignments]
    write_json(wave_new / 'assignments.json', filtered)
    write_json(wave_new / 'original_assignments.json', old_assignments)
    shutil.copy2(retained_workers, wave_new / 'previous-workers.sbatch')
    shutil.copy2(retained_controller, wave_new / 'previous-controller.sbatch')
    code = freeze_code(launch)
    env = ['PCM_PROJECT_ROOT=' + str(REPO), 'PYTHONPATH=' + str(code), 'OMP_NUM_THREADS=1',
           'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OVITO_THREAD_COUNT=1', 'QT_QPA_PLATFORM=offscreen']
    command = [sys.executable, '-u', '-m', 'src.simulation.campaigns.dense_al', 'collect',
               '--launch', str(launch), '--wave', str(wave)]
    script = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=al-dense-next', '#SBATCH --partition=CPU',
        '#SBATCH --qos=normal', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=4G', '#SBATCH --time=12:00:00', f'#SBATCH --dependency=afterany:{workers}',
        f'#SBATCH --chdir={code}', f'#SBATCH --output={wave_new}/controller-%j.log', 'set -euo pipefail',
        'exec env ' + shlex.join(env + command), ''])
    # Recheck current status immediately before committing a changed queue. Markers
    # exclude reserved unstarted second sources from old in-memory worker loops.
    handoffs = []
    for i in changed:
        old_directory = root / original['runs'][i]['run_dir']
        path = old_directory / 'status.json'
        current = json.loads(path.read_text()) if path.exists() else None
        if current is not None and current['state'] != 'queue_handoff':
            raise ValueError(f'Source started while preparing; preserve it and revise the capture: {old_directory}')
        payload = dict(state='queue_handoff', created_at=now(), run_id=original['runs'][i]['run_id'],
            reason='User-authorized unstarted-source transfer; no dynamics run here',
            successor_launch=str(launch), successor_directory=str(root / plan['runs'][i]['run_dir']))
        if current is None:
            with path.open('x') as handle:
                json.dump(payload, handle, indent=2)
                handle.write('\n')
        else:
            write_json(path, payload)
        handoffs.append(dict(source_id=original['runs'][i]['source_id'], run_id=original['runs'][i]['run_id'],
                             previous_directory=str(old_directory), new_directory=payload['successor_directory']))
    write_json(launch / 'handoffs.json', handoffs)
    pending = subprocess.check_output(['squeue', '-h', '-j', controller, '-o', '%i %t %j'], text=True).strip()
    if pending != f'{controller} PD al-dense-next':
        raise ValueError(f'Controller changed before activation: {pending}')
    subprocess.run(['scancel', controller], check=True)
    try:
        replacement = sbatch(script, wave_new / 'controller.sbatch')
    except BaseException:
        # The old queue must not resume handed-off inputs under its frozen recipe.
        write_json(launch / 'status.json', dict(state='activation_failed', active_worker_job=workers,
            canceled_controller=controller, frozen_launch=str(launch), handoffs=str(launch / 'handoffs.json'),
            reason='Replacement controller submission failed; all native running dynamics remain unchanged'))
        raise
    receipt = dict(state='waiting_for_active_wave', wave=wave, worker_job=workers, controller_job=replacement,
        replaced_controller_job=controller, launch=str(launch), total=152,
        protected_trajectories=len(protected), changed_unstarted_trajectories=len(changed),
        changed_main_sources=sum(i < 150 for i in changed), changed_dense_additions=sum(i >= 150 for i in changed),
        protected_source_ids=[plan['runs'][i]['source_id'] for i in protected],
        maximum_measurement_ps_by_temperature=expected_caps,
        active_worker_completion='Old workers retain current dynamics; two reserved next sources intentionally hand off before dynamics, and are excluded from wave collection',
        submitted_at=now())
    write_json(wave_new / 'submission.json', receipt)
    write_json(launch / 'status.json', receipt)
    write_json(previous / 'successor.json', receipt)
    write_json(root / 'continuation.json', receipt)
    return receipt
