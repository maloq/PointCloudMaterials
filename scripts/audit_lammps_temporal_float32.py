#!/usr/bin/env python3
"""Audit completed non-shooting float32 migrations after text deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.temporal_lammps_binary import (  # noqa: E402
    TemporalLAMMPSBinaryTrajectory,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _audit_root(root: Path) -> dict[str, int | str | bool]:
    status = _load_json(root / "status.json")
    if status.get("state") != "complete":
        raise RuntimeError(
            f"Audit root is not complete: root={root}, state={status.get('state')!r}."
        )
    residual_text = sorted(root.glob("**/trajectory.lammpstrj"))
    if residual_text:
        raise RuntimeError(
            f"Completed migration root still contains text trajectories: {residual_text}"
        )
    interrupted_builds = sorted(root.glob("**/.trajectory_binary_float32.building-*"))
    if interrupted_builds:
        raise RuntimeError(
            f"Completed migration root contains interrupted binary builds: {interrupted_builds}"
        )
    binary_dirs = sorted(
        path for path in root.glob("**/trajectory_binary_float32") if path.is_dir()
    )
    if not binary_dirs:
        raise RuntimeError(f"Completed migration root has no binary trajectories: {root}")

    source_bytes = 0
    binary_bytes = 0
    for binary_dir in binary_dirs:
        replica_dir = binary_dir.parent
        report = _load_json(replica_dir / "binary_migration_float32.json")
        if report.get("state") != "complete":
            raise RuntimeError(f"Migration report is not complete: {replica_dir}")
        source_record = report["source_lammpstrj"]
        if not bool(source_record["deleted"]):
            raise RuntimeError(f"Migration report does not record source deletion: {replica_dir}")
        original_path = Path(str(source_record["path"])).resolve()
        if original_path.exists():
            raise RuntimeError(f"Deleted source path still exists: {original_path}")

        binary = TemporalLAMMPSBinaryTrajectory.load(binary_dir)
        binary.verify_checksums()
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(original_path)
        if (
            scan.frame_count != binary.frame_count
            or scan.num_atoms != binary.atom_count
            or tuple(scan.atom_columns) != tuple(binary.manifest["atom_columns"])
        ):
            raise RuntimeError(
                f"Transparent reader scan disagrees with binary metadata: {binary_dir}"
            )
        archive_record = report["coordinate_archive"]
        archive_path = Path(str(archive_record["path"])).resolve()
        observed_archive_sha256 = _sha256_file(archive_path)
        if observed_archive_sha256 != str(archive_record["sha256"]):
            raise RuntimeError(
                f"Coordinate archive checksum changed after migration: {archive_path}"
            )
        analysis = _load_json(replica_dir / "analysis.json")
        if analysis.get("trajectory_binary_artifact", {}).get("state") != "complete":
            raise RuntimeError(
                f"Replica analysis does not identify the complete binary artifact: {replica_dir}"
            )
        source_bytes += int(source_record["size_bytes"])
        binary_bytes += int(report["binary_size_bytes"])

    return {
        "campaign_root": str(root),
        "state": "complete",
        "replica_count": len(binary_dirs),
        "source_size_bytes": source_bytes,
        "binary_size_bytes": binary_bytes,
        "freed_bytes": source_bytes - binary_bytes,
        "all_sources_deleted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_roots", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [_audit_root(path.expanduser().resolve()) for path in args.campaign_roots]
    print(
        json.dumps(
            {
                "state": "complete",
                "campaign_count": len(reports),
                "replica_count": sum(int(report["replica_count"]) for report in reports),
                "source_size_bytes": sum(
                    int(report["source_size_bytes"]) for report in reports
                ),
                "binary_size_bytes": sum(
                    int(report["binary_size_bytes"]) for report in reports
                ),
                "freed_bytes": sum(int(report["freed_bytes"]) for report in reports),
                "campaigns": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
