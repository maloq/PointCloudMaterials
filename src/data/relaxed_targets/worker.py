"""Resumable, lock-coordinated full-cell minimization and fixed-identity TDA."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.liquid_structure import persistence_image
from src.data.conversion.relaxation import read_relaxed
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import load_json, machine, REPO
from src.simulation.relaxation import relax_frame, sha256
from .plan import digest, save


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def clouds(observed, relaxed, lengths, centers):
    """Both inputs are box-relative; select nearest 80 once, in observed geometry."""
    observed = np.mod(observed, lengths)
    relaxed = np.mod(relaxed, lengths)
    _, neighbors = cKDTree(observed, boxsize=lengths).query(observed[centers], k=80)
    if not np.array_equal(neighbors[:, 0], centers):
        raise ValueError('The selected center must be the unique first observed neighbor')
    result = []
    for points in (observed, relaxed):
        offsets = points[neighbors] - points[centers, None]
        offsets -= lengths * np.rint(offsets / lengths)
        result.append(offsets.astype(np.float32))
    return *result, neighbors


class AbsolutePositions:
    """The shooting producer stores box-relative positions; LAMMPS needs absolute."""
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __getitem__(self, frame):
        return self.trajectory.positions[frame].astype(np.float64) + self.trajectory.box_low[frame]


def settings(cfg, ranks):
    execution = machine()['execution']
    os.environ.update(execution['mpi_environment'])
    command = [part.format(ranks=ranks) for part in execution['mpi_launcher']]
    potential = cfg['potential_files']
    for path, key in zip(potential, ('library_sha256', 'parameter_sha256'), strict=True):
        if sha256(path) != cfg['potential_sha256'][key]:
            raise ValueError(f'Wrong generating-potential file: {path}')
    return dict(**cfg['relaxation'], lammps_command=command + [execution['lammps']],
        potential_files=potential, pair_commands=['pair_style meam',
        f'pair_coeff * * {potential[0]} Al {potential[1]} Al'])


def checked_receipt(directory):
    receipt = json.loads((directory/'complete.json').read_text())
    if sha256(directory/'targets.npz') != receipt['targets_sha256']:
        raise ValueError(f'Corrupt relaxed targets: {directory}')
    archive = Path(receipt['relaxation_archive'])
    if not (archive/'conversion.json').is_file():
        raise ValueError(f'Missing archived relaxation: {archive}')
    verify_archive(archive)
    return receipt


def verify_archive(directory):
    for name, expected in json.loads((directory/'archive_checksums.json').read_text()).items():
        if sha256(directory/name) != expected:
            raise ValueError(f'Archived relaxation changed: {directory/name}')


def publish(work, destination):
    """Verify every archived file before allowing the completion receipt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    shutil.copytree(work, temporary, dirs_exist_ok=True)
    hashes = {str(p.relative_to(work)): sha256(p) for p in work.rglob('*') if p.is_file()}
    for name, expected in hashes.items():
        if sha256(temporary/name) != expected:
            raise ValueError(f'Archive verification failed: {temporary/name}')
    save(temporary/'archive_checksums.json', hashes)
    if destination.exists():
        raise FileExistsError(f'Archive destination already exists: {destination}')
    temporary.rename(destination)


def produce(cfg, source, frame, relaxation, implementation):
    trajectory = ShootingBinaryTrajectory.load(source['trajectory'])
    if sha256(trajectory.root/'manifest.json') != source['manifest_sha256']:
        raise ValueError(f'Trajectory changed since planning: {trajectory.root}')
    observed = trajectory.positions[frame].astype(np.float64)
    low = trajectory.box_low[frame].astype(np.float64)
    lengths = trajectory.box_high[frame].astype(np.float64) - low
    centers = np.searchsorted(trajectory.atom_ids, source['center_atom_ids'])
    np.testing.assert_array_equal(trajectory.atom_ids[centers], source['center_atom_ids'])
    cell = hashlib.sha256()
    for values in (np.mod(observed, lengths), low, lengths, trajectory.atom_ids,
                   trajectory.atom_types, np.asarray(source['center_atom_ids'], np.int64)):
        cell.update(np.ascontiguousarray(values).tobytes())
    # Exact duplicate observed cells with identical centers share one target.
    key = digest([cell.hexdigest(), relaxation, implementation])
    directory = Path(cfg['cache'])/'frames'/key
    with lock(directory/'writer.lock') as acquired:
        if not acquired:
            return None
        if (directory/'complete.json').exists():
            return checked_receipt(directory)
        started = time.monotonic()
        work_root = Path(cfg['scratch'])/key
        work_root.mkdir(parents=True, exist_ok=True)
        attempts = sorted(work_root.glob('attempt_*'))
        if attempts and (attempts[-1]/'metadata.json').exists():
            work = attempts[-1]
        else:
            work = work_root/f'attempt_{len(attempts):03d}'
            absolute = SimpleNamespace(**vars(trajectory), atom_count=trajectory.atom_count)
            absolute.positions = AbsolutePositions(trajectory)
            try:
                relax_frame(absolute, frame, work, relaxation)
            except Exception:
                # Stopped failures are data too: preserve inputs/logs on STORE.
                failure_archive = Path(cfg['archive'])/'failures'/key/work.name
                if not failure_archive.exists():
                    publish(work, failure_archive)
                raise
        metadata = json.loads((work/'metadata.json').read_text())
        force = metadata['fmax_eV_per_A']
        if metadata['settings'] != relaxation or not np.isfinite(force) or force > relaxation['force_tolerance']:
            raise ValueError(f'Relaxation settings/force invalid: {work}')
        target = directory/'targets.npz'
        pending = directory/'targets_pending.json'
        if pending.exists():
            if sha256(target) != json.loads(pending.read_text())['sha256']:
                raise ValueError(f'Interrupted target write is corrupt: {target}')
        else:
            relaxed, _ = read_relaxed(work)
            hot, cold, neighbors = clouds(observed, relaxed-low, lengths, centers)
            hot_tda = np.stack([persistence_image(p) for p in hot])
            relaxed_tda = np.stack([persistence_image(p) for p in cold])
            if not np.isfinite(relaxed_tda).all() or not np.isfinite(hot_tda).all():
                raise ValueError(f'Nonfinite TDA for {source["id"]} frame {frame}')
            temporary = directory/'targets.tmp.npz'
            np.savez_compressed(temporary, center_atom_ids=trajectory.atom_ids[centers],
                neighbor_atom_ids=trajectory.atom_ids[neighbors], observed_clouds=hot,
                relaxed_clouds=cold, instantaneous_tda=hot_tda, relaxed_tda=relaxed_tda)
            temporary.replace(target)
            save(pending, dict(sha256=sha256(target)))
        # Descriptor inputs were saved in centered float32 before global float16
        # conversion. The converter verifies rounding/checksums before deletion.
        if not (work/'conversion.json').exists():
            with (work/'conversion_stdout.log').open('w') as log:
                subprocess.run([sys.executable, str(REPO/'scripts/convert_trajectory.py'),
                    'relaxation', str(work), '--delete-source', '--local-cloud-dtype', 'float32'], check=True, stdout=log,
                    stderr=subprocess.STDOUT, cwd=REPO)
        archive = Path(cfg['archive'])/'frames'/key
        if not archive.exists():
            publish(work, archive)
        verify_archive(archive)
        receipt = dict(state='complete', key=key, completed_at=now(), windows=len(centers),
            targets_sha256=sha256(target), targets=str(target), relaxation_archive=str(archive),
            implementation_sha256=implementation, relaxation=metadata,
            local_cloud_dtype='float32', target_dtype='float32', seconds=time.monotonic()-started,
            protocol='Full periodic fixed-cell FIRE; observed nearest-80 identities retained after minimization; raw 144D persistence image.')
        save(directory/'complete.json', receipt)
        return receipt


def implementation_hashes():
    paths = [Path(__file__), Path(__file__).with_name('plan.py'),
        REPO/'src/simulation/relaxation.py', REPO/'src/analysis/liquid_structure.py',
        REPO/'src/data/conversion/relaxation.py']
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def run(config_path, worker, hours, ranks, max_tasks=None, retry_failed=False):
    cfg = load_json(config_path)
    technical = Path(cfg['output'])/'technical'
    plan = json.loads((technical/'plan.json').read_text())
    from src.project_runtime.paths import portable_config
    if plan['config_signature'] != digest(portable_config(cfg)):
        raise ValueError('Worker configuration differs from the frozen target plan')
    relaxation = settings(cfg, ranks)
    implementation = implementation_hashes()
    implementation['lammps_binary'] = sha256(relaxation['lammps_command'][-1])
    binding = dict(implementation=implementation, relaxation=relaxation)
    with lock(technical/'binding.lock') as acquired:
        if not acquired:
            raise RuntimeError('Another worker is initializing; retry after it starts')
        path = technical/'implementation.json'
        if path.exists() and json.loads(path.read_text()) != binding:
            raise ValueError('Implementation/backend changed; use a new release, not mixed targets')
        save(path, binding)
    status_path = technical/'workers'/f'{worker}.json'
    completed = failed = 0
    deadline = time.monotonic() + hours*3600
    status = dict(state='running', worker=worker, pid=os.getpid(), slurm_job=os.environ.get('SLURM_JOB_ID'),
        started_at=now(), completed=0, failed=0)
    save(status_path, status)
    for task in plan['tasks']:
        if time.monotonic() + cfg['relaxation']['frame_timeout_seconds'] + 60 > deadline:
            break
        if max_tasks is not None and completed >= max_tasks:
            break
        destination = technical/'tasks'/f'{task["id"]}.json'
        failure = technical/'failures'/f'{task["id"]}.json'
        if destination.exists() or (failure.exists() and not retry_failed):
            continue
        with lock(technical/'locks'/f'{task["id"]}.lock') as acquired:
            if not acquired or destination.exists():
                continue
            source = plan['sources'][task['source_index']]
            status.update(task=task, source=source['id'], updated_at=now())
            save(status_path, status)
            try:
                receipt = produce(cfg, source, task['frame'], relaxation, implementation)
                if receipt is None:
                    continue
                save(destination, dict(**task, state='complete', target_key=receipt['key'],
                    targets=receipt['targets'], targets_sha256=receipt['targets_sha256'],
                    windows=receipt['windows'], completed_at=now(), worker=worker))
                completed += 1
                print(f'COMPLETE {task["id"]} {source["id"]} frame={task["frame"]} windows={receipt["windows"]}', flush=True)
            except Exception:
                failed += 1
                record = dict(**task, source=source['id'], state='failed', worker=worker,
                              failed_at=now(), traceback=traceback.format_exc())
                save(failure, record)
                print(record['traceback'], file=sys.stderr, flush=True)
                if failed >= cfg['max_worker_failures']:
                    status.update(state='failed', failed=failed, completed=completed, updated_at=now())
                    save(status_path, status)
                    raise RuntimeError(f'{failed} target failures; inspect {technical/"failures"}')
            status.update(completed=completed, failed=failed, updated_at=now())
            save(status_path, status)
    total_done = len(list((technical/'tasks').glob('*.json')))
    status.update(state='complete' if total_done == len(plan['tasks']) else 'partial_resumable',
                  updated_at=now(), campaign_completed=total_done, campaign_total=len(plan['tasks']))
    save(status_path, status)
    return status


def status(config_path):
    cfg = load_json(config_path)
    technical = Path(cfg['output'])/'technical'
    plan = json.loads((technical/'plan.json').read_text())
    done = [json.loads(p.read_text()) for p in (technical/'tasks').glob('*.json')]
    failures = list((technical/'failures').glob('*.json'))
    successful = {d['id'] for d in done}
    return dict(planned=plan['counts'], completed_frames=len(done),
        completed_windows=sum(d['windows'] for d in done),
        unique_relaxed_cells=len({d['target_key'] for d in done}),
        unresolved_failures=sum(p.stem not in successful for p in failures),
        workers=[json.loads(p.read_text()) for p in (technical/'workers').glob('*.json')])
