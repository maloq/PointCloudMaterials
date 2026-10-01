"""Version the active Al execution queue to include two 0.01-ps descendants."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from src.project_runtime.paths import REPO, storage_path
from src.project_runtime.transfer import write_json
from .birth_sources import now, sha256, register_entry, freeze_code
from .dense_al import manifest_at, source_input, sbatch

PROTOCOL = 'al_dense_replay_001ps_v1'


def add_dense_sources(config):
    previous = Path(config['source_launch'])
    original = manifest_at(previous)
    environment=json.loads((previous/'environment.json').read_text())
    if sha256(Path(sys.prefix)/'bin/lmp') != environment['lammps_sha256']:
        raise ValueError('Use the original pointnet-torch214 LAMMPS executable for the cadence comparison')
    status = json.loads((previous/'status.json').read_text())
    if status['state'] != 'submitted':
        raise ValueError(f'An active submitted wave is required: {status}')
    wave = status['wave']; wave_old = previous/f'wave-{wave:03d}'
    submission = json.loads((wave_old/'submission.json').read_text())
    assigned={i for lane in json.loads((wave_old/'assignments.json').read_text()) for i in lane}
    controller = submission['controller_job']; worker = submission['worker_job']
    observed = subprocess.check_output(['squeue','-h','-j',controller,'-o','%i %t %j'],text=True).strip()
    if observed != f'{controller} PD al-dense-next':
        raise ValueError(f'Only this campaign\'s pending next-wave controller may be replaced: {observed}')
    if len(config['source_ids']) != 2 or len(set(config['source_ids'])) != 2 or config['sample_interval_steps'] != 5:
        raise ValueError('This add-on requires exactly two distinct source IDs and exact 0.01 ps sampling')
    identifier = config['launch_id']; collection = config['collection_id']
    for value in (identifier,collection):
        if Path(value).name != value or value in {'.','..'}:
            raise ValueError('Launch and collection IDs must be single directory names')
    launch = storage_path('archive')/'simulation-launches'/identifier
    root = Path(original['root']); extra = root/collection
    if launch.exists() or extra.exists():
        raise FileExistsError('Preserve existing add-on execution and prepared sources')
    if min(shutil.disk_usage(root).free,shutil.disk_usage(storage_path('archive')).free) < config['required_free_bytes']:
        raise OSError('Insufficient room for dense dumps and verified final arrays')
    plan = deepcopy(original)
    plan.update(created_at=now(),launch_root=str(launch),
                augmentation_of=dict(launch=str(previous),manifest_sha256=sha256(previous/'manifest.json')),
                main_source_count=len(original['runs']),supplemental_source_count=2,
                independent_ancestor_count=len({r['root_lineage'] for r in original['runs']}))
    dense_config = deepcopy(original['config'])
    dense_config.update(protocol=PROTOCOL,campaign_id=collection,sample_interval_steps=5)
    launch.mkdir(); extra.mkdir()
    write_json(launch/'status.json',dict(state='preparing',created_at=now()))
    additions=[]
    for source_id in config['source_ids']:
        parent, = [r for r in original['runs'] if r['source_id']==source_id]
        if original['runs'].index(parent) in assigned:
            raise ValueError(f'Source is already assigned to the active wave: {source_id}')
        original_directory = root/parent['run_dir']
        if (original_directory/'status.json').exists() or 'recovery' in parent:
            raise ValueError(f'Choose a declared unstarted source, not an active or recovered run: {source_id}')
        for name,digest in parent['input_sha256'].items():
            if sha256(original_directory/name) != digest:
                raise ValueError(f'Matched main source input changed: {original_directory/name}')
        record=deepcopy(parent)
        record.pop('input_sha256'); record.pop('failure_id')
        run_id=f'{collection}-source{source_id:04d}'
        record.update(run_id=run_id,run_dir=f'{collection}/runs/{run_id}',protocol=PROTOCOL,
                      sample_interval_steps=5,queue_priority=0,matched_010ps_run_id=parent['run_id'],
                      failure_id=run_id+'-failed')
        directory=root/record['run_dir']; (directory/'potential').mkdir(parents=True)
        for name in ('prepared_liquid.lammps.data','melt_final.restart.bin','parent_source.in.lammps',
                     'potential/Lee2003_Al.library.meam','potential/Lee2003_Al.meam'):
            shutil.copy2(original_directory/name,directory/name)
            if sha256(directory/name) != parent['input_sha256'][name]:
                raise ValueError(f'Prepared dense input copy differs: {directory/name}')
        (directory/'source.in.lammps').write_text(source_input(dense_config,record))
        metadata=json.loads((original_directory/'input_metadata.json').read_text())
        metadata.update(record)
        metadata.update(protocol=PROTOCOL,state='prepared',sample_interval_steps=5,sampling_ps=.01,
                        expected_frame_count=60001,extra_descendant=True,
                        sampling_policy='Save every five 2-fs integration steps; full 600 ps measurement')
        write_json(directory/'input_metadata.json',metadata)
        write_json(directory/'technical/launch_config.json',dense_config)
        record['input_sha256']={str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*') if p.is_file()}
        additions.append(record)
    plan['runs'].extend(additions)
    plan['role_count_scope']='roles records the 150 original independent ancestors; trajectory_roles includes the two additional descendants'
    plan['trajectory_roles']=dict(Counter(r['split'] for r in plan['runs']))
    plan['additional_observation_protocols']={PROTOCOL:dict(timestep_ps=.002,sample_interval_steps=5,
        sampling_ps=.01,measurement_ps=600,expected_frames=60001,positions=True,velocities=True,
        storage_dtype='float16',box_dtype='float32',source_ids=config['source_ids'],
        selection_rule=config['selection_rule'])}
    write_json(launch/'manifest.json',plan)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    additional=deepcopy(plan)
    additional.update(protocol=PROTOCOL,config=dense_config,root=str(extra),runs=deepcopy(additions),
                      roles=dict(Counter(r['split'] for r in additions)),independent_ancestor_count=2,
                      trajectory_roles=dict(Counter(r['split'] for r in additions)),
                      main_source_count=0,role_count_scope='Two retained train ancestors in this add-on collection')
    for record in additional['runs']:
        record['run_dir']=str(Path(record['run_dir']).relative_to(collection))
    write_json(launch/'additional_sources.json',additional)
    (extra/'manifest.json').symlink_to(launch/'additional_sources.json')
    register_entry(collection,dict(root='simulation_runs',path=str(extra.relative_to(storage_path('simulation_runs'))),
        kind='simulation',dependencies=sorted({r['parent_dataset'] for r in additions}|{'potentials',original['config']['campaign_id']}),
        metadata=dict(materials=['Al'],potential_ids=['al-lee2003-meam'],role='raw_dynamics',classification='research',
            title='Two main-Al descendants saved every 0.01 ps with velocities',
            description='70304 atoms per run, 600 ps measurement, 2 fs integration; source 886 at 400 K and source 1004 at 520 K.',
            sampling_ps=.01,protocol=PROTOCOL,
            ancestry='Two existing main-Al melt ancestors and their frozen train roles; no additional independent melt lineages.',
            evidence=[str(launch/'additional_sources.json'),str(launch/'status.json')])))
    wave_new=launch/f'wave-{wave:03d}'; wave_new.mkdir()
    for name in ('assignments.json','submission.json','workers.sbatch','controller.sbatch'):
        shutil.copy2(wave_old/name,wave_new/('previous-'+name if name.endswith('.sbatch') else name))
    code=freeze_code(launch)
    env=['PCM_PROJECT_ROOT='+str(REPO),'PYTHONPATH='+str(code),'OMP_NUM_THREADS=1',
         'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','QT_QPA_PLATFORM=offscreen']
    command=[sys.executable,'-u','-m','src.simulation.campaigns.dense_al','collect',
             '--launch',str(launch),'--wave',str(wave)]
    script='\n'.join(['#!/bin/bash','#SBATCH --job-name=al-dense-next','#SBATCH --partition=CPU',
        '#SBATCH --qos=normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=4G','#SBATCH --time=12:00:00',f'#SBATCH --dependency=afterany:{worker}',
        f'#SBATCH --chdir={code}',f'#SBATCH --output={wave_new}/controller-%j.log','set -euo pipefail',
        'exec env '+shlex.join(env+command),''])
    # Prepare and freeze everything before replacing our own pending controller.
    observed = subprocess.check_output(['squeue','-h','-j',controller,'-o','%i %t %j'],text=True).strip()
    if observed != f'{controller} PD al-dense-next':
        raise ValueError(f'Controller changed while preparing add-on: {observed}')
    write_json(launch/'controller_replacement.json',dict(state='prepared',previous_controller=controller,
        active_worker_array=worker,prepared_at=now(),previous_launch=str(previous)))
    subprocess.run(['scancel',controller],check=True)
    try:
        replacement=sbatch(script,wave_new/'controller.sbatch')
    except BaseException:
        restored=sbatch((wave_new/'previous-controller.sbatch').read_text(),wave_new/'restored-controller.sbatch')
        write_json(launch/'controller_replacement.json',dict(state='rolled_back',previous_controller=controller,
                   restored_controller=restored,restored_at=now()))
        raise
    receipt=dict(state='waiting_for_active_wave',wave=wave,worker_job=worker,controller_job=replacement,
        replaced_controller_job=controller,main_source_count=len(original['runs']),supplemental_source_count=2,
        total=len(plan['runs']),source_ids=config['source_ids'],launch=str(launch),collection=collection,
        submitted_at=now(),canonical_vector_bytes=2*60001*70304*3*2*2,
        new_source_schedule='Prioritize the two additional sources in the next wave; one source per lane for that wave')
    write_json(wave_new/'submission.json',receipt)
    write_json(launch/'controller_replacement.json',receipt)
    write_json(launch/'status.json',receipt)
    write_json(previous/'successor.json',receipt)
    write_json(root/'continuation.json',receipt)
    return receipt
