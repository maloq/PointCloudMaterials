"""Shared relaxation lookup with explicit recipe, precision and atom membership.

The index is metadata, not an experiment split or a label cache. Historical
float16 cells and pre-quantization float32 patches remain distinct products.
New native LAMMPS checkpoints permit arbitrary future patches without another
minimization. File locks and atomic publication coordinate independent jobs.
"""
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import storage_path


CONFIG = 'configs/relaxation_index.json'


def root():
    return storage_path('archive') / 'relaxation-index/v1'


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(key):
    path = root() / 'locks' / (key + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@lru_cache(maxsize=64)
def _file_digest(path, size, mtime_ns):
    return sha(path)


def current_digest(path):
    p = Path(path); stat = p.stat()
    return _file_digest(str(p.resolve()), stat.st_size, stat.st_mtime_ns)


def engine_command(settings):
    """Split the two actual repository launch forms: direct LAMMPS and MPI."""
    command = settings['lammps_command']
    if Path(command[0]).name in ('mpiexec', 'mpirun'):
        if command[1] not in ('-n', '-np'):
            raise ValueError(f'Unsupported MPI launch form: {command}')
        return command[3], [Path(command[0]).name, *command[1:3]], command[4:]
    return command[0], [], command[1:]


def recipe(settings, potential_checksums=None, engine_sha256=None):
    potentials = potential_checksums if potential_checksums is not None else {
        p: current_digest(p) for p in settings['potential_files']}
    executable, launcher, flags = engine_command(settings)
    if engine_sha256 is None:
        engine = shutil.which(executable)
        if engine is None:
            raise FileNotFoundError(f'LAMMPS executable not found: {executable}')
        engine_sha256 = current_digest(engine)
    pairs = '\n'.join(settings['pair_commands'])
    for path, checksum in sorted(potentials.items(), key=lambda item: -len(item[0])):
        pairs = pairs.replace(path, 'sha256:' + checksum)
    # Backend/rank count can change the basin reached. Timeouts cannot change
    # an accepted converged result; budgets and restart seeds remain explicit.
    result = dict(protocol='fixed_periodic_cell_norm_inf_v1', mass=settings['mass'],
        minimizer=settings.get('minimizer', 'cg'), force_tolerance=settings['force_tolerance'],
        max_iterations=settings['max_iterations'], max_evaluations=settings['max_evaluations'],
        pair_commands=pairs, potential_sha256=sorted(potentials.values()),
        engine_sha256=engine_sha256, launcher=launcher, engine_flags=flags,
        restart_sha256=settings.get('restart_sha256'))
    if result['minimizer'] == 'fire':
        result['timestep_ps'] = settings['timestep_ps']
    return result


def identity(meta, spec):
    return digest(dict(input_sha256=meta['input_sha256'], recipe=spec))


def source_key(meta):
    return digest({k: meta[k] for k in ('source_manifest_sha256', 'source_frame', 'source_timestep')})


def alias(meta, spec):
    return digest(dict(source=source_key(meta), recipe=spec))


def validate_metadata(meta, path):
    if (meta['state'] != 'relaxed' or not np.isfinite(meta['fmax_eV_per_A'])
            or not np.isfinite(meta['energy_eV'])
            or meta['fmax_eV_per_A'] > meta['settings']['force_tolerance']):
        raise ValueError(f'Unconverged/invalid relaxation: {path}')
    if meta['protocol'] != ('Full periodic cell, fixed box, generating potential; '
            'infinity-norm force convergence, no isolated-patch relaxation.'):
        raise ValueError(f'Unrecognized relaxation protocol: {path}')


def register(entry):
    destination = root() / 'entries' / (entry['id'] + '.json')
    with lock('entry-' + entry['id']):
        if destination.exists():
            old = read(destination)
            if old != entry:
                raise ValueError(f'Relaxation index entry changed: {destination}')
        else:
            write_json(destination, entry)
        # Unknown historical engines remain discoverable but not exact matches.
        pointer = root() / 'sources' / source_key(entry['metadata']) / (entry['id'] + '.json')
        if not pointer.exists():
            write_json(pointer, dict(entry=str(destination)))
    return entry


def candidates(meta, settings=None, *, spec=None):
    if spec is None and settings is not None:
        spec = recipe(settings)
    result = []
    for pointer in sorted((root() / 'sources' / source_key(meta)).glob('*.json')):
        entry = read(read(pointer)['entry'])
        if spec is None or entry['recipe'] == spec:
            result.append(entry)
    return result


def available(entry):
    return all(Path(f).is_file() for f in entry['files'])


def verify(entry):
    validate_metadata(entry['metadata'], entry['path'])
    for name, expected in entry['files'].items():
        if sha(name) != expected:
            raise ValueError(f'Changed indexed relaxation payload: {name}')
    return entry


def full_cell(key):
    folder = root() / 'cells' / key
    if not (folder / 'complete.json').is_file():
        return None
    entry = read(folder / 'complete.json')
    verify(entry)
    register(entry)  # recover a crash between checkpoint publication and indexing
    return folder, entry


def publish_cell(directory, spec):
    """Publish one native checkpoint; coordinate archives still use float16."""
    directory = Path(directory); meta = read(directory / 'metadata.json')
    validate_metadata(meta, directory)
    if recipe(meta['settings'], meta['potential_checksums'], spec['engine_sha256']) != spec:
        raise ValueError(f'Potential/recipe changed during relaxation: {directory}')
    key = identity(meta, spec); target = root() / 'cells' / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=key + '.', dir=target.parent))
    try:
        shutil.copyfile(directory / 'relaxed.restart', temporary / 'relaxed.restart')
        if sha(temporary / 'relaxed.restart') != sha(directory / 'relaxed.restart'):
            raise ValueError(f'Native checkpoint copy differs: {directory}')
        write_json(temporary / 'metadata.json', meta)
        entry = dict(id='cell-' + key, key=key, path=str(target), metadata=meta, recipe=spec,
            precision='native_restart', metadata_sha256=sha(temporary/'metadata.json'),
            payload=str(target/'relaxed.restart'), files={str(target/n):sha(temporary/n)
                for n in ('relaxed.restart', 'metadata.json')})
        write_json(temporary / 'complete.json', entry)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    register(entry)
    # The verified shared native checkpoint is the durable copy.
    (directory / 'relaxed.restart').unlink()
    return entry


def recover_native(directory):
    """Finish a publication interrupted after convergence, before conversion."""
    directory = Path(directory)
    if not (directory/'relaxed.restart').is_file():
        return
    meta = read(directory/'metadata.json')
    spec = meta['index_recipe']; key = identity(meta, spec)
    with lock('cell-' + key):
        hit = full_cell(key)
        if hit is None:
            publish_cell(directory, spec)
        else:
            _, entry = hit
            if sha(directory/'relaxed.restart') != sha(entry['payload']):
                raise ValueError(f'Conflicting native checkpoints for one input/recipe: {directory}')
            (directory/'relaxed.restart').unlink()


def materialize_cell(hit, trajectory, frame, directory, settings):
    """Read a native checkpoint and dump coordinates, with no MD or minimization."""
    folder, entry = hit; directory = Path(directory)
    meta = entry['metadata']
    if meta['atom_count'] != trajectory.atom_count:
        raise ValueError(f'Shared relaxed cell shape differs: {folder}')
    for name in ('box_low', 'box_high'):
        np.testing.assert_array_equal(meta[name], getattr(trajectory, name)[frame])
    text = f'''units metal
atom_style atomic
read_restart {folder / 'relaxed.restart'}
{chr(10).join(settings['pair_commands'])}
reset_timestep {int(trajectory.timesteps[frame])}
write_dump all custom relaxed.dump id type x y z modify sort id format line "%d %d %.17g %.17g %.17g"
'''
    (directory/'reuse.lammps').write_text(text)
    with (directory/'reuse.log').open('w') as log:
        subprocess.run([*settings['lammps_command'], '-in', 'reuse.lammps', '-log', 'none'],
            cwd=directory, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            check=True, timeout=settings['frame_timeout_seconds'])
    meta = dict(meta, source=str(trajectory.root), source_frame=int(frame),
        source_timestep=int(trajectory.timesteps[frame]),
        source_manifest_sha256=sha(trajectory.root/'manifest.json'), settings=settings,
        input_sha256=sha(directory/'input.data'),
        seconds=0., reuse=dict(index_entry=entry['id'], index_cell=str(folder),
                              key=entry['key'], original_seconds=meta['seconds']))
    write_json(directory/'metadata.json', meta)
    from src.data.conversion.relaxation import read_relaxed
    read_relaxed(directory)  # validates timeline, identities, box, and shape
    return meta


def _patch_values(entry):
    with np.load(entry['payload'], allow_pickle=False) as data:
        ids = data[entry['center_field']].reshape(-1)
        neighbors = data['neighbor_atom_ids'].reshape(len(ids), -1)
        xyz = data[entry['position_field']].reshape(len(ids), neighbors.shape[1], 3)
    if xyz.dtype != np.float32 or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid float32 patch payload: {entry["payload"]}')
    if (neighbors[:, 0] != ids).any() or np.any(xyz[:, 0]):
        raise ValueError(f'Patch centers/membership differ: {entry["payload"]}')
    return ids, neighbors, xyz


def patch_lookup(meta, settings, centers, neighbor_ids):
    """Reuse exact ordered memberships; output order belongs to the caller.

A request is served from one physical relaxation, never a mixture of minima.
Missing artifacts are not hits; changed checksums are errors.
"""
    centers = np.asarray(centers); neighbor_ids = np.asarray(neighbor_ids)
    if centers.ndim != 1 or neighbor_ids.ndim != 2 or len(centers) != len(neighbor_ids):
        raise ValueError('Expected center IDs [C] and ordered neighbor IDs [C,K]')
    for entry in candidates(meta, settings):
        if entry['precision'] != 'saved_float32_patches' or not available(entry):
            continue
        verify(entry)
        ids, neighbors, xyz = _patch_values(entry)
        if neighbors.shape[1] != neighbor_ids.shape[1]:
            continue
        selected = []
        for atom, membership in zip(centers, neighbor_ids, strict=True):
            matches = [i for i in np.flatnonzero(ids == atom)
                       if np.array_equal(neighbors[i], membership)]
            if not matches:
                break
            selected.append(matches[0])
        if len(selected) == len(centers):
            return xyz[selected].copy(), entry
    return None


def cell_lookup(meta, settings, *, precision):
    """Explicit precision required: a float16 cell is never a float32 patch."""
    if precision not in ('archived_float16_cell', 'native_restart'):
        raise ValueError(f'Unknown cell precision: {precision}')
    for entry in candidates(meta, settings):
        if entry['precision'] == precision and available(entry):
            return verify(entry)
    return None


def load_cell(meta, settings, *, precision):
    """Load the verified box-relative float16 trajectory (one frame)."""
    if precision != 'archived_float16_cell':
        raise ValueError('load_cell requires precision="archived_float16_cell"; use relax_frame for native precision')
    entry = cell_lookup(meta, settings, precision=precision)
    if entry is None:
        return None
    from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
    return TemporalLAMMPSBinaryTrajectory.load(entry['payload']), entry


def ensure_patches(trajectory, frame, settings, centers, neighbor_ids):
    """Experiment-independent compute-only-missing interface.

trajectory uses absolute positions, as relax_frame does. centers/neighbor_ids
are exact atom IDs chosen in the original geometry, with center first.
Returns (float32 offsets, index entry), preserving the caller's row order.
"""
    frame = int(frame)
    centers = np.asarray(centers); neighbors = np.asarray(neighbor_ids)
    if centers.ndim != 1 or neighbors.ndim != 2 or len(centers) == 0 or len(centers) != len(neighbors):
        raise ValueError('Nonempty center IDs [C] and neighbor IDs [C,K] required')
    if not np.array_equal(neighbors[:, 0], centers):
        raise ValueError('First neighbor must be the center')
    rows = np.searchsorted(trajectory.atom_ids, neighbors)
    np.testing.assert_array_equal(trajectory.atom_ids[rows], neighbors)
    meta = dict(source_manifest_sha256=sha(trajectory.root/'manifest.json'),
                source_frame=int(frame), source_timestep=int(trajectory.timesteps[frame]))
    spec = recipe(settings)
    with lock('request-' + alias(meta, spec)):
        hit = patch_lookup(meta, settings, centers, neighbors)
        if hit is not None:
            return hit
        from src.simulation.relaxation import relax_frame
        from src.data.conversion.relaxation import read_relaxed, convert
        from src.data.relaxed_targets.worker import publish
        from src.data.relaxation_inventory import import_archive
        key = digest(dict(source=meta, recipe=spec, centers=centers.tolist(), neighbors=neighbors.tolist()))
        target = root()/'patches'/key
        if target.exists():
            import_archive(target)
            hit = patch_lookup(meta, settings, centers, neighbors)
            if hit is None:
                raise ValueError(f'Published patch request is not reusable: {target}')
            return hit
        work_root = storage_path('scratch')/'relaxation-index'/key
        work_root.mkdir(parents=True, exist_ok=True)
        attempts = sorted(work_root.glob('attempt-*'))
        work = attempts[-1] if attempts and (attempts[-1]/'metadata.json').exists() else work_root/f'attempt-{len(attempts):03d}'
        if not (work/'metadata.json').exists():
            relax_frame(trajectory, frame, work, settings)
        if not (work/'conversion.json').exists():
            x, _ = read_relaxed(work)
            lengths = np.asarray(trajectory.box_high[frame],np.float64)-trajectory.box_low[frame]
            cold = x[rows] - x[rows[:, :1]]
            cold -= lengths * np.rint(cold / lengths)
            np.savez_compressed(work/'patches.npz', positions=cold.astype(np.float32),
                                center_atom_ids=centers, neighbor_atom_ids=neighbors)
            convert(work, delete_source=True, local_cloud_dtype='float32')
        publish(work, target)
        import_archive(target)
        hit = patch_lookup(meta, settings, centers, neighbors)
        if hit is None:
            raise RuntimeError(f'New relaxed patches absent from shared index: {target}')
        shutil.rmtree(work_root)
        return hit


def main():
    from src.data.relaxation_inventory import main as inventory_main
    inventory_main()


if __name__ == '__main__':
    main()
