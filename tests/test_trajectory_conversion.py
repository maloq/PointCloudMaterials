"""Exercise conversion through the public CLI with repository-format fixtures."""
import hashlib
import json

import numpy as np
import pytest

from src.data_utils.conversion.cli import main
from src.data_utils.conversion.audit_temporal import audit_campaign
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset


@pytest.fixture
def temporal_campaign(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete"}))
    replica = tmp_path / "replica_0"
    replica.mkdir()
    positions = np.array([[[1, 2, 3], [4, 5, 6]], [[1.1, 2, 3], [4.1, 5, 6]]], dtype=np.float32)
    archive = replica / "trajectory.npz"
    np.savez(archive, step=np.array([0, 100]), positions_A=positions,
             cell_vectors_A=np.tile(np.eye(3) * 10, (2, 1, 1)))
    (replica / "analysis.json").write_text(json.dumps({
        "artifacts_sha256": {"trajectory.npz": hashlib.sha256(archive.read_bytes()).hexdigest()}
    }))
    main(["export-npz-dump", str(archive), str(replica / "trajectory.lammpstrj")])
    return tmp_path, replica, positions


def test_temporal_conversion_retains_then_deletes_verified_source(temporal_campaign):
    root, replica, positions = temporal_campaign
    source = replica / "trajectory.lammpstrj"
    main(["temporal", str(root)])
    assert source.is_file()
    binary = TemporalLAMMPSBinaryTrajectory.load(replica / "trajectory_binary_float32")
    np.testing.assert_array_equal(binary.positions, positions)
    assert not audit_campaign(root)["all_sources_deleted"]
    main(["audit-temporal", str(root)])
    with pytest.raises(RuntimeError, match="still contains text"):
        main(["audit-temporal", str(root), "--require-source-deleted"])
    main(["temporal", str(root), "--delete-source"])
    assert not source.exists()
    assert (replica / "trajectory.npz").is_file()
    assert audit_campaign(root, require_source_deleted=True)["all_sources_deleted"]
    frame, _, _ = TemporalLAMMPSDumpDataset.load_dump_frame_positions(source, frame_index=1)
    np.testing.assert_array_equal(frame, positions[1])
    main(["temporal", str(root), "--delete-source"])


def test_corrupt_temporal_binary_does_not_delete_source(temporal_campaign):
    root, replica, _ = temporal_campaign
    main(["temporal", str(root)])
    positions_path = replica / "trajectory_binary_float32/positions.npy"
    values = np.load(positions_path, mmap_mode="r+")
    values[0, 0, 0] += 1
    values.flush()
    del values
    with pytest.raises(RuntimeError, match="checksum"):
        main(["temporal", str(root), "--delete-source"])
    assert (replica / "trajectory.lammpstrj").is_file()


def test_changed_coordinate_archive_does_not_delete_source(temporal_campaign):
    root, replica, _ = temporal_campaign
    with (replica / "trajectory.npz").open("ab") as stream:
        stream.write(b"changed archive")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        main(["temporal", str(root), "--delete-source"])
    assert (replica / "trajectory.lammpstrj").is_file()


@pytest.mark.parametrize('delete_source', [False, True])
def test_elemental_conversion_preserves_coordinates_before_optional_text_removal(tmp_path, delete_source):
    from src.data_utils.conversion.elemental import convert
    from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory

    metadata = dict(state='dynamics_complete', frame_count=2, atom_count=2, dump_every_steps=50)
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    source = tmp_path / 'trajectory.lammpstrj'
    source.write_text(''.join(
        f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n'
        'ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n'
        'ITEM: ATOMS id type x y z\n1 1 11 2 3\n2 1 4 -1 6\n'
        for step in (0, 50)
    ))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = convert(tmp_path, delete_source=delete_source)
    binary = TemporalLAMMPSBinaryTrajectory.load(report['binary_path'])
    assert binary.verify_checksums() == report['checksums']
    np.testing.assert_array_equal(binary.positions, [[[1, 2, 3], [4, 9, 6]]] * 2)
    np.testing.assert_array_equal(binary.timesteps, [0, 50])
    assert report['source_sha256'] == source_hash
    assert source.exists() is (not delete_source)
    assert report['raw_text_preserved'] is (not delete_source)
    assert not (tmp_path / 'conversion_positions.npy').exists()
