"""Corrupt elemental dumps must never become complete trajectory artifacts."""
import json

import numpy as np
import pytest

from src.data_utils.conversion.cli import main
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory


def frame(step, second_id=2):
    return f'''ITEM: TIMESTEP
{step}
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-1 9
-1 9
-1 9
ITEM: ATOMS id type x y z
1 1 0 1 2
{second_id} 1 3 4 5
'''


@pytest.fixture
def branch(tmp_path):
    (tmp_path / 'metadata.json').write_text(json.dumps({
        'state': 'dynamics_complete', 'frame_count': 2, 'atom_count': 2,
        'dump_every_steps': 100, 'material': 'Ti'}))
    return tmp_path


def test_public_elemental_conversion_preserves_source_and_box_convention(branch):
    source = branch / 'trajectory.lammpstrj'
    source.write_text(frame(0) + frame(100))
    main(['elemental', str(branch), '--storage-dtype', 'float32'])
    binary = TemporalLAMMPSBinaryTrajectory.load(branch / 'trajectory_binary_float32')
    np.testing.assert_array_equal(binary.positions[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(binary.timesteps, [0, 100])
    binary.verify_checksums()
    assert source.is_file()
    assert not (branch / 'conversion_positions.npy').exists()


@pytest.mark.parametrize('dump', [frame(0) + frame(100, 3), frame(0) + frame(101),
                                  frame(0), frame(0) + frame(100) + frame(200)])
def test_reject_corrupt_or_wrong_horizon_dump(branch, dump):
    source = branch / 'trajectory.lammpstrj'
    source.write_text(dump)
    with pytest.raises(RuntimeError):
        main(['elemental', str(branch), '--storage-dtype', 'float32'])
    assert source.is_file()
    assert not (branch / 'trajectory_binary_float32').exists()


def test_failed_ta_prevents_ti_from_starting(tmp_path, monkeypatch):
    import subprocess
    from src.simulation.campaigns import elemental

    paths = []
    for material, protocol in [('Ta', 'ta-position-branches'), ('Ti', 'ti-source-then-branches')]:
        config = tmp_path / f'{material}.json'
        config.write_text(json.dumps({'protocol': protocol, 'material': material,
                                      'output_root': str(tmp_path / material)}))
        paths.append(config)

    def failed_dynamics(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(elemental.subprocess, 'run', failed_dynamics)
    with pytest.raises(subprocess.CalledProcessError):
        elemental.sequence(*paths)
    status = json.loads((tmp_path / 'sequence_status.json').read_text())
    assert status['state'] == 'failed'
    assert status['current_campaign'] == 'Ta'
    assert status['completed_campaigns'] == []
    assert not (tmp_path / 'Ti').exists()


@pytest.mark.parametrize('damage', [None, 'positions', 'seed'])
def test_resume_verifies_completed_binary_and_parent(branch, damage):
    from src.simulation.campaigns.elemental import sha256, verify_completed_branch

    config = {'material': 'Ta', 'atom_count': 2, 'branch_steps': 100,
              'dump_every_steps': 100, 'timestep_ps': 0.002, 'potential_files': []}
    parent = {'name': 'parent', 'velocity_seed': 17}
    metadata = json.loads((branch / 'metadata.json').read_text())
    metadata['material'] = 'Ta'
    (branch / 'metadata.json').write_text(json.dumps(metadata))
    (branch / 'trajectory.lammpstrj').write_text(frame(0) + frame(100))
    main(['elemental', str(branch), '--delete-source', '--storage-dtype', 'float32'])
    restart = branch / 'final.restart.bin'
    restart.write_bytes(b'test restart')
    outcome = {'state': 'complete', 'material': 'Ta', 'atom_count': 2,
               'steps': 100, 'dump_every_steps': 100, 'timestep_ps': 0.002,
               'origin': parent, 'potential': [], 'final_restart_sha256': sha256(restart)}
    (branch / 'outcome.json').write_text(json.dumps(outcome))
    if damage == 'positions':
        values = np.load(branch / 'trajectory_binary_float32/positions.npy', mmap_mode='r+')
        values[0, 0, 0] += 1
        values.flush()
        del values
    if damage == 'seed':
        parent = {**parent, 'velocity_seed': 18}
    if damage is None:
        verify_completed_branch(config, branch, parent)
    else:
        with pytest.raises(RuntimeError):
            verify_completed_branch(config, branch, parent)
    assert not (branch / 'trajectory.lammpstrj').exists()


def test_float16_rounding_and_periodic_reader(branch):
    from src.data_utils.temporal_lammps_dataset import _sanitize_periodic_points
    (branch / 'trajectory.lammpstrj').write_text(
        (frame(0) + frame(100)).replace('1 1 0 1 2', '1 1 8.999 1.1234 2'))
    main(['elemental', str(branch), '--delete-source'])
    binary = TemporalLAMMPSBinaryTrajectory.load(branch / 'trajectory_binary_float16')
    assert binary.positions.dtype == np.float16
    assert binary.positions[0, 0, 0] == 10
    decoded = _sanitize_periodic_points(binary.positions[0], binary.box_high[0] - binary.box_low[0])
    assert decoded.dtype == np.float32
    assert decoded[0, 0] == 0
    report = json.loads((branch / 'binary_conversion.json').read_text())
    assert 0 < report['quantization']['max_abs_error_A'] < 0.004
    binary.verify_checksums()
    assert not (branch / 'trajectory.lammpstrj').exists()


def test_binary_float16_migration_preserves_arrays_and_legacy_path(branch):
    from src.data_utils.conversion.position_storage import compress
    from src.data_utils.temporal_lammps_binary import resolve_temporal_lammps_artifact
    (branch / 'trajectory.lammpstrj').write_text(frame(0) + frame(100))
    main(['elemental', str(branch), '--storage-dtype', 'float32', '--delete-source'])
    old_path = branch / 'trajectory_binary_float32'
    report = compress(old_path, delete_source=True)
    assert old_path.is_symlink()
    binary = TemporalLAMMPSBinaryTrajectory.load(old_path)
    assert binary.positions.dtype == np.float16
    assert report['original_deleted']
    assert resolve_temporal_lammps_artifact(branch / 'trajectory.lammpstrj') == binary.root
    assert binary.verify_checksums() == report['checksums']
    saved = json.loads((branch / 'binary_conversion.json').read_text())
    assert saved['checksums'] == report['checksums']


def test_selected_branch_uses_slurm_affinity(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from src.simulation.campaigns import elemental
    from src.experiment_runner import tracking

    @contextmanager
    def tracked(directory, **kwargs):
        directory.mkdir(parents=True)
        yield

    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'material': 'Ti', 'protocol': 'ti-source-then-branches',
                                 'potential_files': [], 'cpus': [0, 1],
                                 'output_root': str(tmp_path / 'output')}))
    parents = tmp_path / 'parents.json'
    parents.write_text(json.dumps({'branches': [{'name': 'parent_0'}]}))
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    monkeypatch.setattr(elemental.os, 'sched_getaffinity', lambda pid: {12, 13})
    monkeypatch.setattr(tracking, 'tracked_run', tracked)
    observed = []
    monkeypatch.setattr(elemental, 'run_branch', lambda c, r, b: observed.append(c['cpus']))
    elemental.run_selected_branch(config, parents, 0)
    assert observed == [[12, 13]]
    assert elemental.os.environ['HYDRA_BOOTSTRAP'] == 'fork'
    with pytest.raises(FileExistsError):
        elemental.run_selected_branch(config, parents, 0)
