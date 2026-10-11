"""Resume unfinished appearance quenches with bounded, explicitly recorded execution changes."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.project_runtime.paths import resolve_path
from .data import read
from .relaxed_data import cell


def submit(path):
    c = read(path)
    base = resolve_path(c['appearance_output']) / 'technical'
    root = base / c['recovery_name']
    if root.exists():
        raise FileExistsError(f'Recovery already prepared: {root}')
    p = read(base / 'relaxation-plan.json')
    if digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']:
        raise ValueError('Frozen relaxation plan changed')
    old_producer = base / 'code/src/research/birth_prediction/relaxed_data.py'
    if sha(old_producer) != p['cell_producer']:
        raise ValueError('Original frozen cell producer changed')
    cache = resolve_path(p['config']['cache'])
    pending = [t for t in p['tasks'] if not (cache / 'cells' / t['id'] / 'complete.json').exists()]
    isolated = [t for t in pending if t['id'] in c['isolated_tasks']]
    remaining = [t for t in pending if t['id'] not in c['isolated_tasks']]
    lanes = [[t] for t in isolated] + [remaining[i::c['parallel_lanes']] for i in range(c['parallel_lanes'])]
    lanes = [items for items in lanes if items]
    if not lanes:
        raise ValueError('All cells are already complete; run the existing seal')
    # Retain every original scientific dependency. Only the recorded execution
    # timeout extension and this scheduler adapter differ from the old bundle.
    frozen = root / 'code'
    shutil.copytree(base / 'code', frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    repository = Path(__file__).resolve().parents[3]
    for relative in ('src/research/birth_prediction/relaxed_data.py',
                     'src/research/birth_prediction/appearance_resume.py',
                     'src/data/relaxation_index.py', 'src/data/relaxation_inventory.py'):
        shutil.copy2(repository / relative, frozen / relative)
    write_json(frozen / 'config.json', c)
    write_json(root / 'plan.json', p)
    assignment = dict(plan_identity=p['identity'], original_cell_producer=p['cell_producer'],
        recovery_cell_producer=sha(frozen / 'src/research/birth_prediction/relaxed_data.py'),
        driver_sha256=sha(frozen / 'src/research/birth_prediction/appearance_resume.py'),
        reuse_implementation={name: sha(frozen / name) for name in
            ('src/data/relaxation_index.py', 'src/data/relaxation_inventory.py')},
        pending=len(pending), total=len(p['tasks']), lanes=lanes,
        recovery=dict(frame_timeout_seconds=c['frame_timeout_seconds'],
            scientific_settings='Original potential, precision, ranks, FIRE, timestep, force tolerance and iteration limits unchanged'))
    assignment['identity'] = digest(assignment)
    write_json(root / 'assignment.json', assignment)
    machine = yaml.safe_load((base / 'machine-relaxation.yaml').read_text())
    machine['catalog'] = str(frozen / 'configs/datasets.json')
    (root / 'machine.yaml').write_text(yaml.safe_dump(machine))
    record = dict(identity=assignment['identity'], jobs={}, code=str(frozen), remaining_cells=len(pending))
    queue = SlurmQueue(root, ExecutionBundle(frozen), 'src.research.birth_prediction.appearance_resume',
        dict(PCM_PROJECT_ROOT=str(frozen), PCM_MACHINE_CONFIG=str(root / 'machine.yaml'),
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1'),
        root / 'launch.json', record, 'APPEARANCE-RESUME')
    with queue.submission():
        job = queue.submit('worker', [f'--array=0-{len(lanes)-1}',
            f'--cpus-per-task={p["config"]["ranks"]}', '--mem=8G', f'--time={c["time_limit"]}'])
        subprocess.run(['scontrol', 'update', f'JobId={c["seal_job"]}', f'Dependency=afterok:{job}'], check=True)
        record['seal_job'] = c['seal_job']
        write_json(root / 'launch.json', record)
    return record


def worker(path, index):
    c = read(path)
    root = Path(path).resolve().parent.parent
    assignment, p = read(root / 'assignment.json'), read(root / 'plan.json')
    if (digest({k: v for k, v in assignment.items() if k != 'identity'}) != assignment['identity']
            or digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']
            or p['identity'] != assignment['plan_identity']
            or sha(Path(__file__)) != assignment['driver_sha256']
            or sha(Path(__file__).with_name('relaxed_data.py')) != assignment['recovery_cell_producer']):
        raise ValueError('Recovery producer or scientific plan changed')
    for name, expected in assignment['reuse_implementation'].items():
        if sha(root / 'code' / name) != expected:
            raise ValueError(f'Recovery reuse implementation changed: {name}')
    with recorded_stage(root / f'worker-{index}.json', job=os.environ.get('SLURM_JOB_ID'),
                        completed=0, total=len(assignment['lanes'][index])) as progress:
        for number, task in enumerate(assignment['lanes'][index]):
            progress.update(task=task['id'])
            cell(p, task, frame_timeout_seconds=c['frame_timeout_seconds'])
            progress.update(completed=number+1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('submit', 'worker'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    if args.stage == 'submit':
        print(submit(args.config))
    else:
        worker(args.config, args.index)


if __name__ == '__main__':
    main()
