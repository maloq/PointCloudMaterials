"""Physical-time sampling, periodic identity retention and safe target reuse."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.liquid_structure import persistence_image
from src.data.relaxed_targets.plan import ancestry, coarse_first, spaced_frames, save
from src.data.relaxed_targets.worker import AbsolutePositions, checked_receipt, clouds, lock, publish
from src.simulation.relaxation import sha256


def test_physical_sampling_does_not_append_event_endpoint():
    steps = np.array([0, 1, 10, 100, 1000, 2000, 2700])
    assert spaced_frames(steps, 3, 0, 72, 3) == [0, 4, 5]
    with pytest.raises(ValueError, match='does not contain'):
        spaced_frames(steps, 3, 0, 72, 1.5)
    frames = list(range(151))
    assert sorted(coarse_first(frames)) == frames
    assert coarse_first(frames)[:3] == [0, 76, 38]


def test_shooting_ancestry_joins_derived_campaigns():
    assert ancestry('source_group_00/campaign/replica_002') == ancestry('campaign/replica_002')
    assert ancestry('campaign/replica_002') != ancestry('campaign/replica_003')


def test_box_origin_adapter_and_fixed_observed_neighbors():
    rng = np.random.default_rng(31)
    hot = rng.uniform(0, 20, size=(120, 3))
    hot[0] = [.2, .3, .4]
    hot[1] = [19.9, .3, .4]
    low = np.array([-18., 3., 7.])
    trajectory = SimpleNamespace(positions=hot[None], box_low=low[None])
    np.testing.assert_array_equal(AbsolutePositions(trajectory)[0], hot+low)
    observed, _, selected = clouds(hot, hot, np.ones(3)*20, np.array([0]))
    assert selected[0, 0] == 0
    np.testing.assert_allclose(observed[0, 1], [-.3, 0., 0.], atol=1e-6)
    outsider = next(i for i in range(120) if i not in selected)
    cold = hot.copy()
    cold[outsider] = hot[0]+[.01, .01, .01]
    first, relaxed, identities = clouds(hot, cold, np.ones(3)*20, np.array([0]))
    assert outsider not in identities
    np.testing.assert_array_equal(first, relaxed)
    assert first.dtype == np.float32 and first.shape == (1, 80, 3)
    offset = np.array([9., 14., 18.])
    moved = clouds((hot+offset)%20, (cold+offset)%20, np.ones(3)*20, np.array([0]))
    np.testing.assert_allclose(moved[0], first, atol=1e-6)
    target = persistence_image(relaxed[0])
    assert target.shape == (144,) and target.dtype == np.float32 and np.isfinite(target).all()


def test_archive_and_target_receipt_detect_corruption(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    save(work/'conversion.json', {'state':'complete'})
    (work/'data').write_bytes(b'preserve me')
    archive = tmp_path/'archive'
    publish(work, archive)
    assert (archive/'data').read_bytes() == b'preserve me'
    directory = tmp_path/'targets'; directory.mkdir()
    np.savez(directory/'targets.npz', relaxed_tda=np.ones((1, 144), np.float32))
    save(directory/'complete.json', dict(targets_sha256=sha256(directory/'targets.npz'), relaxation_archive=str(archive)))
    assert checked_receipt(directory)['relaxation_archive'] == str(archive)
    with (directory/'targets.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Corrupt'):
        checked_receipt(directory)


def test_lock_excludes_other_worker_and_releases(tmp_path):
    with lock(tmp_path/'cell.lock') as first:
        assert first
        with lock(tmp_path/'cell.lock') as second:
            assert not second
    with lock(tmp_path/'cell.lock') as released:
        assert released


def test_resume_after_conversion_keeps_prequantization_targets(tmp_path, monkeypatch):
    from src.data.relaxed_targets import worker
    from src.data.conversion.relaxation import convert
    from src.data.trajectories.shooting import ShootingBinaryTrajectory
    root = tmp_path/'trajectory'; root.mkdir()
    save(root/'manifest.json', {})
    hot = np.random.default_rng(33).uniform(0, 20, (1, 100, 3)).astype(np.float32)
    low = np.array([[-13., 7., 2.]], np.float32)
    trajectory = ShootingBinaryTrajectory(root, {'atom_count':100}, hot, np.zeros_like(hot),
        np.array([1500]), low, low+20, np.arange(1, 101), np.ones(100, np.int32))
    monkeypatch.setattr(ShootingBinaryTrajectory, 'load', lambda _: trajectory)
    minimizations = []

    def minimize(absolute, frame, directory, settings):
        minimizations.append(frame)
        directory.mkdir()
        x = absolute.positions[frame]
        np.testing.assert_allclose(x, hot[0]+low[0], atol=1e-6)
        (directory/'input.data').write_text('full cell input')
        metadata = dict(state='relaxed', atom_count=100, source_timestep=1500,
            box_low=low[0].tolist(), box_high=(low[0]+20).tolist(), settings=settings,
            fmax_eV_per_A=.005)
        save(directory/'metadata.json', metadata)
        with (directory/'relaxed.dump').open('w') as handle:
            handle.write('ITEM: TIMESTEP\n1500\nITEM: NUMBER OF ATOMS\n100\nITEM: BOX BOUNDS pp pp pp\n')
            for a, b in zip(low[0], low[0]+20):
                handle.write(f'{a} {b}\n')
            handle.write('ITEM: ATOMS id type x y z\n')
            np.savetxt(handle, np.column_stack((trajectory.atom_ids, trajectory.atom_types, x)),
                       fmt=['%d', '%d', '%.17g', '%.17g', '%.17g'])

    monkeypatch.setattr(worker, 'relax_frame', minimize)
    monkeypatch.setattr(worker.subprocess, 'run', lambda command, **_: convert(command[3], True, 'float32'))
    source = dict(trajectory=str(root), manifest_sha256=sha256(root/'manifest.json'), center_atom_ids=[1,2], id='fixture')
    cfg = {k:str(tmp_path/k) for k in ('cache', 'scratch', 'archive')}
    relaxation = {'force_tolerance':.01}
    original_publish = worker.publish
    monkeypatch.setattr(worker, 'publish', lambda *_: (_ for _ in ()).throw(OSError('archive unavailable')))
    with pytest.raises(OSError, match='archive unavailable'):
        worker.produce(cfg, source, 0, relaxation, {'test':'same-code'})
    monkeypatch.setattr(worker, 'publish', original_publish)
    result = worker.produce(cfg, source, 0, relaxation, {'test':'same-code'})
    again = worker.produce(cfg, source, 0, relaxation, {'test':'same-code'})
    assert result == again and minimizations == [0]
    with np.load(result['targets']) as arrays:
        assert arrays['relaxed_clouds'].dtype == np.float32
        np.testing.assert_array_equal(arrays['relaxed_tda'], arrays['instantaneous_tda'])
