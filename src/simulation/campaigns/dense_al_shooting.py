"""Bounded additional velocity reshoots of dense-Al liquid preparations."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from src.project_runtime.paths import REPO, load_json, machine, storage_path
from src.project_runtime.transfer import write_json
from .birth_sources import freeze_code, now, register_entry, sha256
from .dense_al import manifest_at, publish_record, run_source, sbatch, verify_code
from .dense_al_gpu import BACKEND
from .dense_al_half_stop import source_input

SAMPLING = 'prepared_liquid_velocity_reshoot_v1'


def key(seed, *values):
    return hashlib.sha256(json.dumps([seed, *values]).encode()).hexdigest()


def prepare(c):
    launch = storage_path('archive')/'simulation-launches'/c['campaign_id']
    root = storage_path('simulation_runs')/c['campaign_id']
    if (launch/'manifest.json').exists() or root.exists():
        raise FileExistsError(f'Preserve prepared shooting campaign: {launch}')
    original = manifest_at(c['source_launch'])
    stopping = manifest_at(c['stopping_launch'])
    policies = {r['temperature_K']: r['stopping'] for r in stopping['runs'] if 'stopping' in r}
    config = dict(original['config'], mpi_ranks=1)
    config.pop('required_free_bytes')
    parents = []
    for temperature in c['temperatures_K']:
        used = set()
        for role in c['parent_roles_by_temperature'][str(temperature)]:
            choices = [r for r in original['runs'] if r['temperature_K'] == temperature
                       and r['split'] == role and r['source_id'] not in used]
            r = min(choices, key=lambda r: key(c['selection_seed'], r['source_id']))
            parents.append(r); used.add(r['source_id'])
    if len({r['root_lineage'] for r in parents}) != len(parents):
        raise ValueError('Parent set contains repeated melt ancestry')
    root.mkdir(parents=True); launch.mkdir(parents=True, exist_ok=True)
    records = []
    seeds = {r['velocity_seed'] for r in original['runs']}
    for shot in range(c['replicates_per_parent']):
        for parent_record in parents:
            r = deepcopy(parent_record)
            parent = Path(r['parent_directory'])
            # Select only by source identity and original role, never outcomes.
            for name in ('prepared_liquid.lammps.data', 'melt_final.restart.bin'):
                if sha256(parent/name) != r['parent_input_sha256'][name]:
                    raise ValueError(f'Prepared parent changed: {parent/name}')
            seed = 1 + int(key(c['selection_seed'], r['source_id'], shot, 'velocity')[:12], 16) % 899999999
            if seed in seeds:
                raise ValueError('Velocity seed collision; change declared selection seed')
            seeds.add(seed)
            identifier = f'{c["campaign_id"]}-source{r["source_id"]:04d}-shot{shot:02d}'
            r.pop('input_sha256')
            r.update(run_id=identifier, run_dir='runs/'+identifier,
                     original_velocity_seed=r['velocity_seed'], velocity_seed=seed,
                     protocol='al_dense_half_stop_010ps_v1', sample_interval_steps=50,
                     stopping=deepcopy(policies[r['temperature_K']]),
                     execution_backend=BACKEND, shot_index=shot, collection_id=c['campaign_id'],
                     sampling_protocol=SAMPLING, source_origin='shooting')
            directory = root/r['run_dir']; (directory/'potential').mkdir(parents=True)
            (directory/'technical/stop-monitor').mkdir(parents=True)
            for name in ('prepared_liquid.lammps.data', 'melt_final.restart.bin'):
                shutil.copy2(parent/name, directory/name)
            shutil.copy2(parent/'source.in.lammps', directory/'parent_source.in.lammps')
            for item in config['potential_files']:
                if sha256(item['path']) != item['sha256']:
                    raise ValueError(f'Potential changed: {item["path"]}')
                shutil.copy2(item['path'], directory/'potential'/Path(item['path']).name)
            execution = dict(engine='FastAl analytic GPU MEAM with native LAMMPS isotropic NPT',
                force_dtype='float32', integration_dtype='float64', energy_reduction_dtype='float64',
                gpu_count=1, mpi_ranks=1, runtime_profile=c['runtime_profile'],
                validation=str(launch/'technical/audit/validation.json'),
                thermostat_ps=.3, barostat_ps=3., pressure_bar=0.,
                stress_scope='Normal virials; shear not computed',
                initial_state='Full-precision prepared-liquid coordinates; new Maxwell velocities; fresh NPT bath',
                ancestry='Two velocity reshoots per retained melt; descendants are not independent ancestors')
            metadata = dict(r, state='prepared', material='Al', atom_count=70304,
                timestep_ps=.002, sampling_ps=.1, equilibration_steps=7500,
                measurement_steps=r['stopping']['maximum_measurement_steps'],
                measurement_ps=r['stopping']['maximum_measurement_ps'],
                ensemble='NPT', thermostat_ps=.3, barostat_ps=3., pressure_bar=0.,
                momentum_removal_interval_ps=.3, preparation_reused=True,
                execution=execution, parent_selection='Seeded identity ordering within temperature and original role; no outcomes',
                retained_observables=['positions', 'velocities', 'box', 'atom_ids', 'timesteps'],
                interpretation='Velocity reshoots from original liquid preparations, not continuations of trajectory snapshots')
            write_json(directory/'input_metadata.json', metadata)
            write_json(directory/'technical/launch_config.json', config)
            (directory/'source.in.lammps').write_text(source_input(config, r, directory))
            r['input_sha256'] = {str(p.relative_to(directory)): sha256(p)
                                for p in directory.rglob('*') if p.is_file()}
            records.append(r)
    # Mix temperatures/roles and both shots in every lane. Do not promote an
    # outcome-dependent subset if the wall-clock budget truncates the queue.
    order = sorted(range(len(records)), key=lambda i: key(c['selection_seed'], records[i]['run_id'], 'queue'))
    assignments = [order[i::c['workers']] for i in range(c['workers'])]
    plan = dict(protocol='al_dense_half_stop_010ps_v1', sampling_protocol=SAMPLING,
        root=str(root), launch_root=str(launch), config=config, shooting_config=c, runs=records,
        assignments=assignments, parents=len(parents), roles=dict(Counter(r['split'] for r in records)),
        created_at=now(), source_manifest_sha256=sha256(Path(c['source_launch'])/'manifest.json'),
        stopping_manifest_sha256=sha256(Path(c['stopping_launch'])/'manifest.json'))
    write_json(launch/'manifest.json', plan)
    (launch/'manifest.sha256').write_text(sha256(launch/'manifest.json')+'\n')
    (root/'manifest.json').symlink_to(launch/'manifest.json')
    register_entry(c['campaign_id'], dict(root='simulation_runs', path=c['campaign_id'], kind='simulation',
        dependencies=sorted({r['parent_dataset'] for r in records}|{'al-main-010ps-20260927'}),
        metadata=dict(materials=['Al'], potential_ids=['al-lee2003-meam'], role='raw_dynamics',
            classification='research', source_origin='shooting', sampling_protocol=SAMPLING,
            title='Additional dense Al velocity reshoots on V100',
            description=f'{len(records)} planned shots, {len(parents)} existing melts; native NPT, 2 fs integration, 0.10 ps Q/V.',
            ancestry='Inherited melt identities and original roles; no new independent preparations.',
            evidence=[str(launch/'manifest.json'), str(launch/'submission.json'), str(launch/'status.json')])))
    write_json(launch/'status.json', dict(state='prepared', planned=len(records)))
    return launch


def environment(launch):
    c = manifest_at(launch)['shooting_config']
    profile = machine()['execution'][c['runtime_profile']]
    cuda = profile['cuda_root']; compiler = profile['cxx']
    cache = storage_path('scratch')/'software'/c['campaign_id']
    cache.mkdir(parents=True, exist_ok=True)
    return dict(PCM_PROJECT_ROOT=str(REPO), PYTHONPATH=str(launch/'code'),
        CUDA_HOME=cuda, CXX=compiler, CC=profile['cc'],
        CPATH=profile['cuda_include_path'],
        CUDAHOSTCXX=compiler, TORCH_CUDA_ARCH_LIST='7.0',
        PATH=cuda+'/bin:'+str(Path(profile['python']).parent)+':/usr/bin:/bin',
        TORCH_EXTENSIONS_DIR=str(cache/'extensions'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        TRITON_CACHE_DIR=str(cache/'triton'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OVITO_THREAD_COUNT='1', QT_QPA_PLATFORM='offscreen',
        MAX_JOBS='4', TORCHINDUCTOR_COMPILE_THREADS='4', UCX_TLS='tcp,self'), profile


def submit(launch):
    plan = manifest_at(launch); c = plan['shooting_config']
    if (launch/'submission.json').exists():
        raise FileExistsError('Campaign already submitted')
    freeze_code(launch)
    env, profile = environment(launch)
    command = [profile['python'], '-u', '-m', 'src.simulation.campaigns.dense_al_shooting']
    base = ['#!/bin/bash', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
            f'#SBATCH --chdir={launch}/code', '#SBATCH --no-requeue']
    body = ['set -euo pipefail', 'export '+shlex.join([k+'='+v for k,v in env.items()])]
    def script(action, options, arguments=()):
        return '\n'.join(base+options+body+['exec '+shlex.join(command+[action, '--launch', str(launch), *arguments]), ''])
    common = ['#SBATCH --partition='+c['partition'], '#SBATCH --cpus-per-task=4',
              '#SBATCH --gres=gpu:1', '#SBATCH --mem=24G']
    audit_job = sbatch(script('audit', common+['#SBATCH --job-name=al-shot-V100-audit',
        '#SBATCH --time='+c['audit_walltime'], f'#SBATCH --output={launch}/technical/audit-%j.log']), launch/'audit.sbatch')
    receipt = dict(state='audit_submitted', audit_job=audit_job, submitted_at=now(),
        runtime=profile, workers=c['workers'], walltime=c['walltime'], planned=len(plan['runs']))
    write_json(launch/'submission.json', receipt)
    job = sbatch(script('worker', common+['#SBATCH --job-name=al-dense-V100-shoot',
        f'#SBATCH --time={c["walltime"]}', f'#SBATCH --array=0-{c["workers"]-1}%{c["workers"]}',
        f'#SBATCH --dependency=afterok:{audit_job}', '#SBATCH --kill-on-invalid-dep=yes',
        f'#SBATCH --output={launch}/technical/worker-%A_%a.log']), launch/'workers.sbatch')
    receipt.update(state='workers_submitted', worker_job=job); write_json(launch/'submission.json', receipt)
    collect_job = sbatch(script('collect', ['#SBATCH --partition=CPU', '#SBATCH --cpus-per-task=1',
        '#SBATCH --mem=8G', '#SBATCH --time=04:00:00', '#SBATCH --job-name=al-shot-collect',
        f'#SBATCH --dependency=afterany:{audit_job}:{job}', f'#SBATCH --output={launch}/technical/collect-%j.log']),
        launch/'collect.sbatch')
    receipt.update(state='submitted', collect_job=collect_job)
    write_json(launch/'submission.json', receipt)
    return receipt


def check_runtime(launch):
    import torch
    import triton
    verify_code(launch)
    if (torch.__version__, triton.__version__, torch.cuda.get_device_capability()) != ('2.6.0+cu118', '3.2.0', (7, 0)):
        raise RuntimeError('This execution profile requires PyTorch 2.6/cu118, Triton 3.2 and a V100')


def audit(launch):
    check_runtime(launch)
    plan = manifest_at(launch); c = plan['shooting_config']; results = []
    for temperature in c['audit_temperatures_K']:
        r = next(r for r in plan['runs'] if r['temperature_K'] == temperature)
        out = launch/'technical/audit'/f'T{temperature}'
        subprocess.run([sys.executable, '-u', '-m', 'src.research.meam_gpu.npt', 'validate',
            '--directory', str(Path(plan['root'])/r['run_dir']), '--output', str(out),
            '--steps', str(c['audit_steps'])], check=True)
        results.append(json.loads((out/'validation.json').read_text()))
    if any(x['state'] != 'passed' for x in results):
        raise ValueError('V100 full-cell adapter audit failed')
    write_json(launch/'technical/audit/validation.json', dict(state='passed', results=results,
        seconds_per_step=max(x['metrics']['gpu_seconds']/x['steps'] for x in results),
        code_inventory_sha256=sha256(launch/'code_sha256.json'), completed_at=now()))


def worker(launch):
    check_runtime(launch)
    plan = manifest_at(launch); c = plan['shooting_config']
    gate = json.loads((launch/'technical/audit/validation.json').read_text())
    if gate['state'] != 'passed' or gate['code_inventory_sha256'] != sha256(launch/'code_sha256.json'):
        raise ValueError('Production requires a matching passed V100 audit')
    index = int(os.environ['SLURM_ARRAY_TASK_ID']); start = time.monotonic()
    completed = []; skipped = []
    for i in plan['assignments'][index]:
        r = deepcopy(plan['runs'][i])
        remaining = c['walltime_seconds'] - (time.monotonic()-start) - c['cleanup_reserve_seconds']
        steps = r['stopping']['maximum_measurement_steps'] + plan['config']['equilibration_steps']
        # Include compilation, monitoring, conversion/publication and a generous
        # throughput margin. Queue truncation is by the predeclared random order.
        estimate = 2*gate['seconds_per_step']*steps + 1800
        if remaining < estimate:
            skipped.append(r['run_id']); continue
        r['execution_timeout_seconds'] = max(1, int(remaining-900))
        run_source(plan, r)
        completed.append(r['run_id'])
        write_json(launch/f'technical/lane-{index}.json', dict(state='running', completed=completed,
            not_started=skipped, elapsed_seconds=time.monotonic()-start))
    write_json(launch/f'technical/lane-{index}.json', dict(state='finished', completed=completed,
        not_started=skipped, elapsed_seconds=time.monotonic()-start))


def collect(launch):
    plan = manifest_at(launch); rows = []
    for r in plan['runs']:
        directory = Path(plan['root'])/r['run_dir']; path = directory/'status.json'
        status = json.loads(path.read_text()) if path.exists() else dict(state='not_started')
        if status['state'] == 'running':
            status.update(state='failed', error='Worker ended before publication; retain partial trajectory and native restart',
                          finished_at=now())
            write_json(path, status)
        if status['state'] == 'failed':
            archive = storage_path('archive')/'simulations'/(r['run_id']+'-failed')
            if not archive.exists():
                publish_record(directory, r, failed=True)
        rows.append(dict(run_id=r['run_id'], source_id=r['source_id'], shot=r['shot_index'],
            split=r['split'], ancestor=r['root_lineage'], temperature_K=r['temperature_K'], state=status['state']))
    counts = dict(Counter(r['state'] for r in rows))
    result = dict(state='complete' if counts.get('complete') == len(rows) else 'incomplete',
        counts=counts, runs=rows, updated_at=now())
    write_json(launch/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'submit', 'audit', 'worker', 'collect'])
    p.add_argument('--config', type=Path)
    p.add_argument('--launch', type=Path)
    args = p.parse_args()
    if args.action == 'prepare': result = str(prepare(load_json(args.config)))
    else: result = globals()[args.action](args.launch.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
