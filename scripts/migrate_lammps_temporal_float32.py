#!/usr/bin/env python3
"""Replace strictly complete non-shooting LAMMPS text dumps with verified binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.temporal_lammps_binary import (  # noqa: E402
    FORMAT_NAME,
    TemporalLAMMPSBinaryTrajectory,
    binary_directory_sizes,
    binary_path_for_dump,
    write_temporal_lammps_binary,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
)


_TEXT_FILENAME = "trajectory.lammpstrj"
_BINARY_DIRNAME = "trajectory_binary_float32"
_REPORT_FILENAME = "binary_migration_float32.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse migration temporary file: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for frame_index in range(int(values.shape[0])):
        digest.update(np.ascontiguousarray(values[frame_index]).tobytes())
    return digest.hexdigest()


def _load_unarchived_prefix_positions(
    source: Path,
    *,
    prefix_frame_count: int,
    atom_count: int,
) -> np.ndarray:
    """Reproduce the repository producer's conversion for pre-measurement frames."""

    prefix = np.empty((prefix_frame_count, atom_count, 3), dtype=np.float32)
    expected_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    with source.open("r", encoding="utf-8") as handle:
        for frame_index in range(prefix_frame_count):
            marker = handle.readline()
            if marker != "ITEM: TIMESTEP\n":
                raise RuntimeError(
                    f"{source}: expected timestep header for prefix frame {frame_index}, "
                    f"got {marker.rstrip()!r}."
                )
            timestep = int(handle.readline())
            if handle.readline() != "ITEM: NUMBER OF ATOMS\n":
                raise RuntimeError(f"{source}: missing atom-count header at step {timestep}.")
            observed_atoms = int(handle.readline())
            if observed_atoms != atom_count:
                raise RuntimeError(
                    f"{source}: prefix step {timestep} has {observed_atoms} atoms, "
                    f"expected {atom_count}."
                )
            bounds_header = handle.readline()
            if bounds_header != "ITEM: BOX BOUNDS pp pp pp\n":
                raise RuntimeError(
                    f"{source}: unsupported prefix box header at step {timestep}: "
                    f"{bounds_header.rstrip()!r}."
                )
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()] for _ in range(3)],
                dtype=np.float64,
            )
            atom_header = handle.readline()
            if atom_header != "ITEM: ATOMS id type x y z\n":
                raise RuntimeError(
                    f"{source}: unsupported prefix atom columns at step {timestep}: "
                    f"{atom_header.rstrip()!r}."
                )
            table = np.loadtxt(handle, max_rows=atom_count)
            if table.shape != (atom_count, 5):
                raise RuntimeError(
                    f"{source}: prefix atom table at step {timestep} has shape={table.shape}, "
                    f"expected={(atom_count, 5)}."
                )
            ids = table[:, 0].astype(np.int64)
            if not np.array_equal(ids, expected_ids):
                raise RuntimeError(
                    f"{source}: prefix atom IDs are not exactly 1..{atom_count} at "
                    f"step {timestep}."
                )
            atom_types = table[:, 1].astype(np.int32)
            if not np.array_equal(atom_types, np.ones(atom_count, dtype=np.int32)):
                raise RuntimeError(
                    f"{source}: prefix atom types are not the repository-required Al type 1 "
                    f"at step {timestep}."
                )
            prefix[frame_index] = (
                table[:, 2:5] - bounds[:, 0][None, :]
            ).astype(np.float32)
    return prefix


def _completed_replicas(campaign_root: Path) -> tuple[Path, ...]:
    status = _load_json(campaign_root / "status.json")
    if status.get("state") != "complete":
        raise RuntimeError(
            f"Refusing to migrate a non-complete campaign: root={campaign_root}, "
            f"state={status.get('state')!r}."
        )
    replicas = {
        path.parent.resolve()
        for path in campaign_root.glob(f"**/{_TEXT_FILENAME}")
        if path.is_file()
    }
    replicas.update(
        path.parent.resolve()
        for path in campaign_root.glob(f"**/{_BINARY_DIRNAME}")
        if path.is_dir()
    )
    if not replicas:
        raise FileNotFoundError(
            f"Complete campaign contains neither text nor migrated trajectory artifacts: {campaign_root}"
        )
    return tuple(sorted(replicas))


def _load_verified_coordinate_archive(
    replica_dir: Path,
) -> tuple[Path, str, np.ndarray, np.ndarray, np.ndarray]:
    analysis_path = replica_dir / "analysis.json"
    analysis = _load_json(analysis_path)
    artifact_hashes = analysis.get("artifacts_sha256")
    if not isinstance(artifact_hashes, dict):
        raise TypeError(
            f"Completed replica analysis has no artifacts_sha256 object: {analysis_path}"
        )
    expected_archive_sha256 = artifact_hashes.get("trajectory.npz")
    if not isinstance(expected_archive_sha256, str) or len(expected_archive_sha256) != 64:
        raise ValueError(
            f"Completed replica analysis has no valid trajectory.npz SHA-256: {analysis_path}"
        )
    archive_path = replica_dir / "trajectory.npz"
    observed_archive_sha256 = _sha256_file(archive_path)
    if observed_archive_sha256 != expected_archive_sha256:
        raise RuntimeError(
            f"Coordinate archive checksum mismatch: path={archive_path}, "
            f"expected={expected_archive_sha256}, observed={observed_archive_sha256}."
        )
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"step", "positions_A", "cell_vectors_A"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Coordinate archive {archive_path} is missing arrays {missing}.")
        step = np.asarray(archive["step"], dtype=np.int64)
        positions = np.asarray(archive["positions_A"])
        cells = np.asarray(archive["cell_vectors_A"], dtype=np.float64)
    if positions.dtype != np.dtype("float32"):
        raise TypeError(
            f"Repository coordinate archive positions must be float32, got "
            f"dtype={positions.dtype}, path={archive_path}."
        )
    return archive_path, observed_archive_sha256, step, positions, cells


def _validate_archive_against_dump(
    *,
    source: Path,
    step: np.ndarray,
    positions: np.ndarray,
    cells: np.ndarray,
) -> tuple[Any, np.ndarray, str]:
    scan = TemporalLAMMPSDumpDataset.scan_dump_file(source)
    if tuple(scan.atom_columns) != ("id", "type", "x", "y", "z"):
        raise ValueError(
            f"Non-shooting migration requires repository position columns "
            f"('id','type','x','y','z'), got {scan.atom_columns} in {source}."
        )
    if positions.ndim != 3 or positions.shape[1:] != (scan.num_atoms, 3):
        raise ValueError(
            f"Coordinate archive has invalid position shape={positions.shape}; expected "
            f"(frames, {scan.num_atoms}, 3), source={source}."
        )
    archive_frame_count = int(positions.shape[0])
    prefix_frame_count = int(scan.frame_count) - archive_frame_count
    if prefix_frame_count < 0:
        raise ValueError(
            f"Coordinate archive has more frames than its text source: archive="
            f"{archive_frame_count}, dump={scan.frame_count}, source={source}."
        )
    if step.shape != (archive_frame_count,):
        raise ValueError(
            f"Coordinate archive step shape disagrees with archive positions: "
            f"step={step.shape}, position_frames={archive_frame_count}, source={source}."
        )
    archived_dump_steps = np.asarray(
        scan.timesteps[prefix_frame_count:], dtype=np.int64
    )
    relative_dump_steps = archived_dump_steps - int(archived_dump_steps[0])
    if not np.array_equal(step, relative_dump_steps):
        raise ValueError(
            f"Coordinate archive cadence disagrees with text dump timesteps: source={source}."
        )
    if cells.shape != (archive_frame_count, 3, 3):
        raise ValueError(
            f"Coordinate archive cells have shape={cells.shape}, expected="
            f"{(archive_frame_count, 3, 3)} in {source.parent / 'trajectory.npz'}."
        )
    diagonal_cells = np.zeros_like(cells)
    diagonal_cells[:, np.arange(3), np.arange(3)] = cells[
        :, np.arange(3), np.arange(3)
    ]
    if not np.array_equal(cells, diagonal_cells):
        raise ValueError(f"Migration requires orthogonal repository cells: {source.parent}.")
    box_lengths = np.asarray(
        scan.box_high[prefix_frame_count:] - scan.box_low[prefix_frame_count:],
        dtype=np.float64,
    )
    cell_lengths = cells[:, np.arange(3), np.arange(3)]
    if not np.allclose(box_lengths, cell_lengths, rtol=0.0, atol=2.0e-5):
        raise ValueError(
            f"Coordinate archive cell lengths disagree with text dump bounds: {source}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"Coordinate archive positions contain non-finite values: {source.parent}.")
    if prefix_frame_count == 0:
        complete_positions = np.array(positions, dtype=np.float32, copy=True)
    else:
        prefix = _load_unarchived_prefix_positions(
            source,
            prefix_frame_count=prefix_frame_count,
            atom_count=scan.num_atoms,
        )
        complete_positions = np.concatenate((prefix, positions), axis=0)
    if complete_positions.shape != (scan.frame_count, scan.num_atoms, 3):
        raise RuntimeError(
            f"Combined trajectory has shape={complete_positions.shape}, expected="
            f"{(scan.frame_count, scan.num_atoms, 3)} for {source}."
        )
    complete_lengths = np.asarray(scan.box_high - scan.box_low, dtype=np.float32)
    for frame_index in range(scan.frame_count):
        frame_lengths = complete_lengths[frame_index]
        wrapped = np.mod(complete_positions[frame_index], frame_lengths[None, :]).astype(
            np.float32, copy=False
        )
        complete_positions[frame_index] = np.minimum(
            wrapped,
            np.nextafter(frame_lengths, np.zeros_like(frame_lengths))[None, :],
        )
    return scan, complete_positions, _array_sha256(complete_positions)


def _migration_document(
    *,
    binary: TemporalLAMMPSBinaryTrajectory,
    source_record: dict[str, Any],
    archive_record: dict[str, Any],
    checksums: dict[str, str],
    migrated_at: str,
) -> dict[str, Any]:
    sizes = binary_directory_sizes(binary.root)
    return {
        "format": FORMAT_NAME,
        "schema_version": 1,
        "state": "complete",
        "migrated_at": migrated_at,
        "binary_path": _BINARY_DIRNAME,
        "storage_dtype": "float32",
        "frame_count": binary.frame_count,
        "atom_count": binary.atom_count,
        "binary_size_bytes": sizes["apparent_bytes"],
        "binary_allocated_bytes": sizes["allocated_bytes"],
        "array_sha256": checksums,
        "source_lammpstrj": source_record,
        "coordinate_archive": archive_record,
    }


def _record_analysis_artifact(replica_dir: Path, report: dict[str, Any]) -> None:
    analysis_path = replica_dir / "analysis.json"
    analysis = _load_json(analysis_path)
    analysis["trajectory_binary_artifact"] = report
    _write_json_atomic(analysis_path, analysis)


def _migrate_replica(replica_dir: Path, *, delete_source: bool) -> dict[str, Any]:
    source = replica_dir / _TEXT_FILENAME
    target = binary_path_for_dump(source)
    if target.name != _BINARY_DIRNAME:
        raise RuntimeError(f"Unexpected canonical binary name for {source}: {target.name}")
    report_path = replica_dir / _REPORT_FILENAME
    if target.is_dir():
        binary = TemporalLAMMPSBinaryTrajectory.load(target)
        checksums = binary.verify_checksums()
        source_record = dict(binary.manifest["source"]["trajectory_lammpstrj"])
        archive_record = dict(binary.manifest["source"]["coordinate_archive"])
        if source.is_file():
            if source.stat().st_size != int(source_record["size_bytes"]):
                raise RuntimeError(f"Residual text source size changed after migration: {source}")
            if _sha256_file(source) != str(source_record["sha256"]):
                raise RuntimeError(f"Residual text source checksum changed after migration: {source}")
            if delete_source:
                source.unlink()
                _fsync_directory(replica_dir)
        source_record["deleted"] = not source.exists()
        if source_record["deleted"] and source_record.get("deleted_at") is None:
            source_record["deleted_at"] = _utc_now()
        report = _migration_document(
            binary=binary,
            source_record=source_record,
            archive_record=archive_record,
            checksums=checksums,
            migrated_at=str(binary.manifest["created_at"]),
        )
        _write_json_atomic(report_path, report)
        _record_analysis_artifact(replica_dir, report)
        return report

    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Replica has neither a source text dump nor a complete binary: {replica_dir}"
        )
    archive_path, archive_sha256, step, positions, cells = _load_verified_coordinate_archive(
        replica_dir
    )
    archive_positions_sha256 = _array_sha256(positions)
    scan, complete_positions, positions_sha256 = _validate_archive_against_dump(
        source=source,
        step=step,
        positions=positions,
        cells=cells,
    )
    source_stat = source.stat()
    source_sha256 = _sha256_file(source)
    source_record = {
        "path": str(source),
        "size_bytes": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
        "sha256": source_sha256,
        "deleted": False,
        "deleted_at": None,
    }
    archive_record = {
        "path": str(archive_path),
        "size_bytes": int(archive_path.stat().st_size),
        "sha256": archive_sha256,
        "positions_float32_sha256": archive_positions_sha256,
        "archived_frame_count": int(positions.shape[0]),
        "unarchived_prefix_frame_count": int(scan.frame_count - positions.shape[0]),
    }
    migrated_at = _utc_now()
    binary = write_temporal_lammps_binary(
        target,
        positions=complete_positions,
        timesteps=np.asarray(scan.timesteps, dtype=np.int64),
        box_low=np.asarray(scan.box_low, dtype=np.float32),
        box_high=np.asarray(scan.box_high, dtype=np.float32),
        atom_ids=np.arange(1, scan.num_atoms + 1, dtype=np.int64),
        atom_types=np.ones(scan.num_atoms, dtype=np.int32),
        atom_columns=tuple(scan.atom_columns),
        source={
            "trajectory_lammpstrj": source_record,
            "coordinate_archive": archive_record,
        },
        provenance={
            "migration_script": str(Path(__file__).resolve()),
            "campaign_completion_required": True,
            "coordinate_source": "repository-produced checksum-bound trajectory.npz",
        },
    )
    checksums = binary.verify_checksums()
    if checksums["positions"] != positions_sha256:
        raise RuntimeError(
            f"Stored positions are not identical to the checksum-bound coordinate archive: {target}"
        )
    report = _migration_document(
        binary=binary,
        source_record=source_record,
        archive_record=archive_record,
        checksums=checksums,
        migrated_at=migrated_at,
    )
    _write_json_atomic(report_path, report)
    _record_analysis_artifact(replica_dir, report)
    _fsync_directory(replica_dir)
    if delete_source:
        source.unlink()
        _fsync_directory(replica_dir)
        source_record["deleted"] = True
        source_record["deleted_at"] = _utc_now()
        report = _migration_document(
            binary=binary,
            source_record=source_record,
            archive_record=archive_record,
            checksums=checksums,
            migrated_at=migrated_at,
        )
        _write_json_atomic(report_path, report)
        _record_analysis_artifact(replica_dir, report)
        _fsync_directory(replica_dir)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign_roots",
        nargs="+",
        type=Path,
        help="Strictly complete campaign or temperature roots to migrate.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each trajectory.lammpstrj only after complete binary verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.delete_source:
        raise ValueError(
            "This migration requires explicit --delete-source acknowledgement; "
            "no source files were changed."
        )
    campaign_reports: list[dict[str, Any]] = []
    for raw_root in args.campaign_roots:
        campaign_root = raw_root.expanduser().resolve()
        replicas = _completed_replicas(campaign_root)
        print(
            f"[temporal-migration] campaign={campaign_root} replicas={len(replicas)}",
            flush=True,
        )
        reports: list[dict[str, Any]] = []
        for replica_index, replica_dir in enumerate(replicas):
            print(
                f"[temporal-migration] replica={replica_index + 1}/{len(replicas)} "
                f"path={replica_dir}",
                flush=True,
            )
            reports.append(_migrate_replica(replica_dir, delete_source=True))
        campaign_report = {
            "format": FORMAT_NAME,
            "schema_version": 1,
            "state": "complete",
            "completed_at": _utc_now(),
            "campaign_root": str(campaign_root),
            "replica_count": len(reports),
            "source_size_bytes": sum(
                int(report["source_lammpstrj"]["size_bytes"]) for report in reports
            ),
            "binary_size_bytes": sum(int(report["binary_size_bytes"]) for report in reports),
            "all_sources_deleted": all(
                bool(report["source_lammpstrj"]["deleted"]) for report in reports
            ),
        }
        _write_json_atomic(
            campaign_root / "temporal_binary_migration_float32.json", campaign_report
        )
        campaign_reports.append(campaign_report)
    total_source = sum(int(report["source_size_bytes"]) for report in campaign_reports)
    total_binary = sum(int(report["binary_size_bytes"]) for report in campaign_reports)
    print(
        json.dumps(
            {
                "state": "complete",
                "campaign_count": len(campaign_reports),
                "replica_count": sum(int(report["replica_count"]) for report in campaign_reports),
                "source_size_bytes": total_source,
                "binary_size_bytes": total_binary,
                "freed_bytes": total_source - total_binary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
