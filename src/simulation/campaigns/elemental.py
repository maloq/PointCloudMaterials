"""Explicit Ti source-then-branches and archived Ta position-branch protocols."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def structure(path, cutoff):
    from ovito.io import import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    pipeline = import_file(str(path))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(rmsd_cutoff=cutoff))
    data = pipeline.compute()
    types = np.asarray(data.particles['Structure Type'])
    counts = np.bincount(types, minlength=4)
    return {'atom_count': len(types), 'ptm_rmsd_cutoff': cutoff,
            'fractions': {name: float(counts[i] / len(types))
                          for i, name in enumerate(('other', 'fcc', 'hcp', 'bcc'))},
            'crystal_fraction': float(np.sum(counts[1:4]) / len(types))}


def assess(root, step):
    config = json.loads((root / 'config.json').read_text())
    source = root / 'source'
    result = structure(source / f'snapshot_{step}.lammpstrj', config['ptm_rmsd_cutoff'])
    if result['atom_count'] != config['atom_count']:
        raise RuntimeError(f'Lost Ti atoms at source step {step}: {result}')
    result.update(step=step, time_ps=step * config['timestep_ps'])
    write_json(source / f'ptm_{step}.json', result)
    previous = step - config['assessment_steps']
    stop = False
    if previous >= config['assessment_steps'] and step >= 6 * config['assessment_steps']:
        earlier = json.loads((source / f'ptm_{previous}.json').read_text())
        stop = min(result['crystal_fraction'], earlier['crystal_fraction']) >= config['complete_crystal_fraction']
    write_json(root / 'source_progress.json', {**result, 'crystallization_complete': stop})
    # A unique include is published only on success. A failed shell assessment
    # therefore makes LAMMPS fail reading this file instead of ignoring an error.
    (source / f'decision_{step}.lammps').write_text(f'variable crystallized equal {int(stop)}\n')


def pair(config):
    return '\n'.join(config['pair_commands'])


def controls(config):
    return f'''mass 1 {config['mass_g_mol']}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {config['timestep_ps']}
thermo {config['dump_every_steps']}
thermo_style custom step time atoms temp press lx ly lz pe ke etotal density
thermo_modify format float %.16g flush yes lost error
'''


def npt(config, temperature):
    return f"fix ensemble all npt temp {temperature} {temperature} {config['thermostat_ps']} iso 0 0 {config['barostat_ps']}\n"


def dump(config):
    return f'''dump trajectory all custom {config['dump_every_steps']} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line "%d %d %.9g %.9g %.9g"
'''


def execute(config, directory, text, marker):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'in.lammps').write_text(text)
    command = [config['mpiexec'], '-n', str(len(config['cpus'])), '-bind-to',
               'user:' + ','.join(map(str, config['cpus'])), config['lammps'],
               '-in', 'in.lammps', '-log', 'log.lammps']
    environment = os.environ.copy()
    environment.update(LD_LIBRARY_PATH=str(Path(sys.prefix) / 'lib'), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MPIR_CVAR_CH4_NETMOD='ofi', FI_PROVIDER='tcp',
                       PYTHONPATH=str(REPO), QT_QPA_PLATFORM='offscreen')
    with (directory / 'stdout.log').open('xb') as output:
        try:
            subprocess.run(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"LAMMPS failed in {directory}; inspect {directory / 'stdout.log'}") from error
    if marker not in (directory / 'log.lammps').read_text().splitlines():
        raise RuntimeError(f'LAMMPS did not reach {marker}: {directory}')


def complete_trajectory(config, directory, steps, origin):
    restart = directory / 'final.restart.bin'
    if restart.stat().st_size == 0:
        raise RuntimeError(f'Empty final restart: {restart}')
    metadata = {'state': 'dynamics_complete', 'material': config['material'],
                'atom_count': config['atom_count'], 'steps': steps,
                'dump_every_steps': config['dump_every_steps'],
                'frame_count': steps // config['dump_every_steps'] + 1,
                'timestep_ps': config['timestep_ps'], 'origin': origin,
                'final_restart_sha256': sha256(restart), 'potential': config['potential_files']}
    write_json(directory / 'metadata.json', metadata)
    command = [sys.executable, str(REPO / 'scripts/convert_trajectory.py'),
               'elemental', str(directory), '--storage-dtype', config['position_storage_dtype']]
    if config['delete_verified_source_text']:
        command.append('--delete-source')
    subprocess.run(command, check=True)
    write_json(directory / 'outcome.json', {**metadata, 'state': 'complete'})


def verify_completed_branch(config, directory, branch):
    """Only reuse checksum-verified complete data from this exact Ta protocol."""
    from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory

    outcome = json.loads((directory / 'outcome.json').read_text())
    expected = {'state': 'complete', 'material': config['material'],
                'atom_count': config['atom_count'], 'steps': config['branch_steps'],
                'dump_every_steps': config['dump_every_steps'],
                'timestep_ps': config['timestep_ps'], 'origin': branch,
                'potential': config['potential_files']}
    for key, value in expected.items():
        if outcome[key] != value:
            raise RuntimeError(f'Completed branch {key} differs from requested protocol: {directory}')
    if sha256(directory / 'final.restart.bin') != outcome['final_restart_sha256']:
        raise RuntimeError(f'Completed restart checksum mismatch: {directory}')
    report = json.loads((directory / 'binary_conversion.json').read_text())
    binary = TemporalLAMMPSBinaryTrajectory.load(report['binary_path'])
    expected_steps = np.arange(0, config['branch_steps'] + 1, config['dump_every_steps'])
    if binary.atom_count != config['atom_count'] or not np.array_equal(binary.timesteps, expected_steps):
        raise RuntimeError(f'Completed binary atom count or timeline mismatch: {directory}')
    report = json.loads((directory / 'binary_conversion.json').read_text())
    if report['state'] != 'complete' or binary.verify_checksums() != report['checksums']:
        raise RuntimeError(f'Completed binary conversion checksums mismatch: {directory}')
    print(f'Verified completed branch; skipping dynamics: {directory}', flush=True)


def archive_failed_status(path):
    previous = json.loads(path.read_text())
    if previous['state'] != 'failed':
        raise RuntimeError(f'Resume requires a failed status, found {previous["state"]}: {path}')
    archive = path.parent / 'resume_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive.mkdir(parents=True)
    shutil.copy2(path, archive / path.name)
    return archive


def run_branch(config, root, branch):
    directory = root / 'branches' / branch['name']
    source = branch['source']
    if sha256(source) != branch['source_sha256']:
        raise RuntimeError(f'Branch source changed: {source}')
    text = f'''units metal
atom_style atomic
boundary p p p
region placeholder block 0 1 0 1 0 1 units box
create_box 1 placeholder
read_dump {source} {branch['source_step']} x y z box yes add yes replace no trim no scaled no wrapped yes
reset_timestep 0
{pair(config)}
{controls(config)}
velocity all create {config['temperature_K']} {branch['velocity_seed']} mom yes rot no dist gaussian loop all
'''
    if config['protocol'] == 'ta-position-branches':
        text += 'fix remove_drift all momentum 100 linear 1 1 1\n'
    text += npt(config, config['temperature_K']) + dump(config)
    text += f'''restart {config['checkpoint_steps']} checkpoint.*.restart.bin
run {config['branch_steps']}
write_restart final.restart.bin
print "BRANCH_COMPLETE"
'''
    execute(config, directory, text, 'BRANCH_COMPLETE')
    complete_trajectory(config, directory, config['branch_steps'], branch)


def run_selected_branch(config_path, parents_path, index):
    """Run one immutable position-conditioned parent, without launching a source."""
    from src.experiment_runner.tracking import tracked_run

    config = json.loads(config_path.read_text())
    parents = json.loads(parents_path.read_text())
    branch = parents['branches'][index]
    if config['protocol'] != 'ti-source-then-branches' or config['material'] != 'Ti':
        raise ValueError('Selected-parent execution requires the Ti position-branch protocol')
    for item in config['potential_files']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError(f'Potential changed: {item}')
    if 'SLURM_JOB_ID' in os.environ:
        ranks = int(os.environ['SLURM_CPUS_PER_TASK'])
        allocated = sorted(os.sched_getaffinity(0))
        if len(allocated) < ranks:
            raise RuntimeError(f'Slurm granted {ranks} CPUs but process affinity is {allocated}')
        config['cpus'] = allocated[:ranks]
        # Single-node MPICH ranks inherit this Slurm step's allocation.
        os.environ['HYDRA_BOOTSTRAP'] = 'fork'
    root = Path(config['output_root'])
    directory = root / 'branches' / branch['name']
    if directory.exists():
        raise FileExistsError(f'Refusing to rerun an existing selected branch: {directory}')
    with tracked_run(directory, kind='simulation', configs=[config_path, parents_path],
                     command=[sys.executable, *sys.argv]):
        write_json(directory / 'config.json', config)
        run_branch(config, root, branch)


def ti_source(config, root):
    melt = root / 'melt'
    nx, ny, nz = config['repetitions_xyz']
    text = f'''units metal
atom_style atomic
boundary p p p
lattice bcc {config['lattice_constant_A']}
region box block 0 {nx} 0 {ny} 0 {nz}
create_box 1 box
create_atoms 1 box
{pair(config)}
{controls(config)}
velocity all create {config['melt_temperature_K']} {config['melt_seed']} mom yes rot no dist gaussian loop all
'''
    text += npt(config, config['melt_temperature_K']).replace('fix ensemble ', 'fix melt_ensemble ')
    text += f'''compute diffusion all msd com yes
fix diffusion_log all ave/time 100 10 1000 c_diffusion[4] file msd.dat
restart {config['checkpoint_steps']} checkpoint.*.restart.bin
run {config['melt_steps']}
write_dump all custom liquid.lammpstrj id type x y z modify sort id
write_restart liquid.restart.bin
print "MELT_COMPLETE"
'''
    execute(config, melt, text, 'MELT_COMPLETE')
    liquid = structure(melt / 'liquid.lammpstrj', config['ptm_rmsd_cutoff'])
    msd = np.loadtxt(melt / 'msd.dat')
    liquid['msd_growth_last_half_A2'] = float(msd[-1, 1] - msd[len(msd) // 2, 1])
    if liquid['atom_count'] != config['atom_count'] or liquid['crystal_fraction'] > config['max_liquid_crystal_fraction'] or liquid['msd_growth_last_half_A2'] < config['minimum_liquid_msd_growth_A2']:
        raise RuntimeError(f'Ti melt is not a validated diffusive liquid: {liquid}')
    write_json(root / 'liquid_validation.json', liquid)
    source = root / 'source'
    source.mkdir()
    text = f'''units metal
atom_style atomic
read_restart {melt / 'liquid.restart.bin'}
reset_timestep 0
{pair(config)}
{controls(config)}
velocity all scale {config['temperature_K']}
'''
    text += npt(config, config['temperature_K']) + dump(config)
    text += f'''write_dump all custom snapshot_0.lammpstrj id type x y z modify sort id
shell {sys.executable} -m src.simulation.campaigns.elemental assess-ti {root} 0
include decision_0.lammps
variable cycle loop {config['max_source_steps'] // config['assessment_steps']}
label source_loop
run {config['assessment_steps']}
write_dump all custom snapshot_$(step:%.0f).lammpstrj id type x y z modify sort id
write_restart checkpoint.$(step:%.0f).restart.bin
shell {sys.executable} -m src.simulation.campaigns.elemental assess-ti {root} $(step:%.0f)
include decision_$(step:%.0f).lammps
if "${{crystallized}} == 1" then "jump SELF source_complete"
next cycle
jump SELF source_loop
print "SOURCE_LIMIT_REACHED_WITHOUT_COMPLETE_CRYSTALLIZATION"
quit 1
label source_complete
write_restart final.restart.bin
print "SOURCE_CRYSTALLIZATION_COMPLETE"
'''
    execute(config, source, text, 'SOURCE_CRYSTALLIZATION_COMPLETE')
    progress = json.loads((root / 'source_progress.json').read_text())
    samples = sorted((json.loads(p.read_text()) for p in source.glob('ptm_*.json')), key=lambda r: r['step'])
    branches = []
    # Select distinct chronological parents only after the full source is known.
    # These are deliberately outcome-conditioned stages, not unbiased nucleation samples.
    previous = -1
    for index, fraction in enumerate(config['branch_target_fractions']):
        candidates = [r for r in samples if r['step'] > previous]
        remaining = len(config['branch_target_fractions']) - index - 1
        if remaining:
            candidates = candidates[:-remaining]
        chosen = min(candidates, key=lambda r: abs(r['crystal_fraction'] - fraction))
        previous = chosen['step']
        path = source / f'snapshot_{previous}.lammpstrj'
        branches.append({'name': f'parent_{previous:09d}', 'source': str(path),
                         'source_sha256': sha256(path), 'source_step': previous,
                         'velocity_seed': config['branch_velocity_seeds'][index],
                         'target_crystal_fraction': fraction, 'observed_structure': chosen,
                         'root_lineage': str(source)})
    write_json(root / 'selected_branches.json', branches)
    complete_trajectory(config, source, progress['step'], {'protocol': 'continuous melt-quench source'})
    return branches


def run(config_path, *, resume_ta=False):
    config = json.loads(config_path.read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    if resume_ta:
        if config['protocol'] != 'ta-position-branches':
            raise ValueError('--resume-ta only resumes complete archived-position Ta branches')
        previous = json.loads((root / 'config.json').read_text())
        # Storage retention may change on recovery; scientific settings may not.
        for key, value in previous.items():
            if key not in {'delete_verified_source_text', 'position_storage_dtype'} and config[key] != value:
                raise RuntimeError(f'Resume changed existing campaign setting {key}: {root}')
        archive_failed_status(root / 'status.json')
    elif (root / 'status.json').exists():
        raise FileExistsError(f'Refusing to overwrite a previous campaign: {root}')
    if config['protocol'] not in ('ti-source-then-branches', 'ta-position-branches'):
        raise ValueError(f"Unknown explicit protocol: {config['protocol']}")
    for item in config['potential_files']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError(f'Potential changed: {item}')
    if config['branch_steps'] % config['dump_every_steps']:
        raise ValueError('Branch duration must be divisible by dump cadence')
    if shutil.disk_usage(root).free < config['required_free_bytes']:
        raise RuntimeError(f"Need {config['required_free_bytes']} free bytes at {root}")
    write_json(root / 'config.json', config)
    status = {'state': 'running', 'host': socket.gethostname(), 'pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_branches': []}
    write_json(root / 'status.json', status)
    try:
        branches = ti_source(config, root) if config['protocol'] == 'ti-source-then-branches' else config['branches']
        for branch in branches:
            status['current_branch'] = branch['name']
            write_json(root / 'status.json', status)
            directory = root / 'branches' / branch['name']
            if resume_ta and directory.exists():
                verify_completed_branch(config, directory, branch)
            else:
                run_branch(config, root, branch)
            status['completed_branches'].append(branch['name'])
            write_json(root / 'status.json', status)
        status.update(state='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / 'status.json', status)
    except BaseException as error:
        status.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        write_json(root / 'status.json', status)
        raise


def sequence(ta_config, ti_config, *, resume_ta=False):
    """Finish all Ta work before starting the Ti source, using the same CPUs."""
    configs = [json.loads(path.read_text()) for path in (ta_config, ti_config)]
    if [c['protocol'] for c in configs] != ['ta-position-branches', 'ti-source-then-branches']:
        raise ValueError('Sequence requires Ta position branches followed by a Ti crystallization source')
    root = Path(configs[0]['output_root']).parent
    if Path(configs[1]['output_root']).parent != root:
        raise ValueError('The Ta/Ti sequence must share one output parent')
    status_path = root / 'sequence_status.json'
    if resume_ta:
        archive_failed_status(status_path)
    elif status_path.exists():
        raise FileExistsError(f'Refusing to duplicate an existing sequence: {status_path}')
    status = {'state': 'running', 'order': ['Ta', 'Ti'], 'pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_campaigns': []}
    try:
        for path, config in zip((ta_config, ti_config), configs):
            status['current_campaign'] = config['material']
            write_json(status_path, status)
            directory = Path(config['output_root'])
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(REPO / 'scripts/run_lammps_campaign.py'),
                       'elemental', 'run', '--config', str(path)]
            resuming_campaign = resume_ta and config['material'] == 'Ta'
            if resuming_campaign:
                command.append('--resume-ta')
            with (directory / 'runner.log').open('ab' if resuming_campaign else 'xb') as output:
                subprocess.run(command,
                               cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
            status['completed_campaigns'].append(config['material'])
        status.update(state='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
    except BaseException as error:
        status.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        write_json(status_path, status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('run')
    command.add_argument('--config', required=True, type=Path)
    command.add_argument('--resume-ta', action='store_true')
    command = sub.add_parser('branch')
    command.add_argument('--config', required=True, type=Path)
    command.add_argument('--parents', required=True, type=Path)
    command.add_argument('--index', required=True, type=int)
    command = sub.add_parser('assess-ti')
    command.add_argument('root', type=Path)
    command.add_argument('step', type=int)
    command = sub.add_parser('sequence')
    command.add_argument('--ta-config', required=True, type=Path)
    command.add_argument('--ti-config', required=True, type=Path)
    command.add_argument('--resume-ta', action='store_true',
                         help='Verify and skip completed Ta branches after a failed sequence.')
    args = parser.parse_args(argv)
    if args.action == 'run':
        from src.experiment_runner.tracking import tracked_run
        config = json.loads(args.config.read_text())
        with tracked_run(Path(config['output_root']), kind='simulation', configs=[args.config],
                         command=[sys.executable, *sys.argv]):
            run(args.config.resolve(), resume_ta=args.resume_ta)
    elif args.action == 'branch':
        run_selected_branch(args.config.resolve(), args.parents.resolve(), args.index)
    elif args.action == 'assess-ti':
        assess(args.root.resolve(), args.step)
    else:
        sequence(args.ta_config.resolve(), args.ti_config.resolve(), resume_ta=args.resume_ta)


if __name__ == '__main__':
    main()
