"""New lineages, unchanged dynamics and paired precision integrity."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from src.simulation.campaigns.memory_sources import specifications, source_input
from src.data.conversion.memory_pair import convert, paired_errors, PROTOCOL
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256


def test_splits_seeds_and_physical_kernel_are_predeclared():
    config = json.loads(Path('configs/simulation/predictive_memory_precision.json').read_text())
    records = specifications(config, 'fixture')
    assert Counter(r['split'] for r in records) == {'train': 6, 'val': 2, 'sealed_test': 4}
    assert len({r[key] for r in records for key in ('preparation_seed', 'velocity_seed')}) == 24
    assert len({r['root_lineage'] for r in records}) == 12
    for record in records:
        text = source_input(config, record)
        assert 'timestep 0.003' in text and '0.3 iso 0 0 3' in text
        assert 'fix remove_drift all momentum 100 linear 1 1 1' in text
        assert text.index('run 5000') < text.index('reset_timestep 0')
        assert 'dump trajectory all custom 25' in text
        assert 'run 64000' in text and '%.17g' in text
        measurement = text.split('reset_timestep 0')[1]
        assert 'velocity all create' not in measurement and 'halt' not in measurement
    assert 64000*.003 == 192. and 25*.003 == .075


def dump_fixture(root, extra=False):
    source = root/'trajectory.lammpstrj'
    with source.open('w') as stream:
        for frame, step in enumerate([0, 25, 50] if extra else [0, 25]):
            stream.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n-1 109\n0 110\n0 110\nITEM: ATOMS id type x y z vx vy vz\n')
            stream.write(f'1 1 {82.12345+frame*.01} 2 3 1.234567 0 0\n2 1 4 5 6 -2.345678 1 0\n')
    (root/'metadata.json').write_text(json.dumps(dict(protocol=PROTOCOL, state='dynamics_complete',
        measurement_steps=25, sample_interval_steps=25, atom_count=2, root_lineage='fixture',
        split='train', timestep_ps=.003, source_sha256=sha256(source))))
    return source


def test_paired_conversion_retains_reference_and_verifies_rounding_before_deletion(tmp_path):
    original = dump_fixture(tmp_path)
    report = convert(tmp_path, delete_source=True)
    assert report['source_deleted'] and not original.exists()
    full = ShootingBinaryTrajectory.load(tmp_path/'trajectory_binary_float32')
    half = ShootingBinaryTrajectory.load(tmp_path/'trajectory_binary_float16')
    assert full.storage_dtype == np.float32 and half.storage_dtype == np.float16
    assert report['quantization']['positions']['max_abs'] > .001
    assert report['quantization']['velocities']['rms'] > 0
    full.verify_checksums(); half.verify_checksums()
    np.testing.assert_array_equal(full.positions.astype(np.float16), half.positions)
    # Correctly include the original box origin in consumer coordinates.
    assert full.positions[0, 0, 0] == pytest.approx(83.12345, abs=1e-5)


def test_conversion_rejects_extra_frames_and_modified_complete_source(tmp_path):
    source = dump_fixture(tmp_path, extra=True)
    with pytest.raises(ValueError, match='timeline'):
        convert(tmp_path)
    assert source.exists() and not (tmp_path/'trajectory_binary_float32').exists()
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError, match='differs from completed'):
        convert(tmp_path)


def test_pair_verification_rejects_corrupted_identity(tmp_path):
    dump_fixture(tmp_path); convert(tmp_path)
    full = ShootingBinaryTrajectory.load(tmp_path/'trajectory_binary_float32')
    half = ShootingBinaryTrajectory.load(tmp_path/'trajectory_binary_float16')
    from types import SimpleNamespace
    corrupted = SimpleNamespace(**{name: getattr(half, name) for name in
        ('atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high', 'positions', 'velocities')})
    corrupted.atom_ids = np.array([2, 1])
    with pytest.raises(ValueError, match='atom_ids'):
        paired_errors(full, corrupted)
