#!/usr/bin/env python3
"""Atomically move large raw LAMMPS artifacts out of a completed campaign."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_NAMES = ("trajectory.lammpstrj", "final.restart.bin")
REPORT_NAME = "lammps_artifacts_archive.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON document is missing: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(document).__name__}.")
    return document


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary JSON path: {temporary}.")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def archive(campaign_root: Path, archive_root: Path) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    if archive_root == campaign_root or campaign_root in archive_root.parents:
        raise ValueError(
            f"Archive must be outside the upload campaign directory: "
            f"campaign={campaign_root}, archive={archive_root}."
        )
    campaign = _load_json(campaign_root / "manifest.json")
    conversion = _load_json(campaign_root / "binary_conversion_float16.json")
    if campaign["state"] != "complete" or int(campaign["completed_branch_count"]) != 18:
        raise RuntimeError(f"Simulation campaign is not complete: {campaign_root}.")
    if conversion["state"] != "complete" or int(conversion["branch_count"]) != 18:
        raise RuntimeError(f"Float16 conversion campaign is not complete: {campaign_root}.")
    campaign_report_path = campaign_root / REPORT_NAME
    if campaign_report_path.exists() or archive_root.exists():
        raise FileExistsError(
            f"Refusing to merge with a prior archive: report={campaign_report_path}, "
            f"archive={archive_root}."
        )

    entries: list[dict[str, Any]] = []
    for branch in campaign["branches"]:
        material = str(branch["material"])
        snapshot = str(branch["snapshot"])
        branch_dir = campaign_root / "branches" / material / snapshot
        binary_manifest = _load_json(branch_dir / "trajectory_binary_float16" / "manifest.json")
        if binary_manifest["state"] != "complete" or binary_manifest["storage_dtype"] != "float16":
            raise RuntimeError(f"Verified float16 replacement is missing: {material}/{snapshot}.")
        for filename in ARTIFACT_NAMES:
            source = branch_dir / filename
            destination = archive_root / "branches" / material / snapshot / filename
            if not source.is_file():
                raise FileNotFoundError(f"Raw LAMMPS artifact is missing: {source}.")
            stat = source.stat()
            entries.append(
                {
                    "material": material,
                    "snapshot": snapshot,
                    "filename": filename,
                    "original_path": str(source),
                    "archive_path": str(destination),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                }
            )

    archive_root.mkdir(parents=True)
    if archive_root.stat().st_dev != campaign_root.stat().st_dev:
        raise OSError(
            f"Campaign and archive are on different filesystems; refusing a non-atomic move: "
            f"campaign_device={campaign_root.stat().st_dev}, "
            f"archive_device={archive_root.stat().st_dev}."
        )
    for entry in entries:
        destination = Path(entry["archive_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Archive target already exists: {destination}.")

    report: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "started_at_utc": _utc_now(),
        "campaign_root": str(campaign_root),
        "archive_root": str(archive_root),
        "same_filesystem_atomic_rename": True,
        "artifact_names": list(ARTIFACT_NAMES),
        "artifact_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "artifacts": entries,
        "moved_artifact_count": 0,
    }
    _write_json_atomic(campaign_report_path, report)
    _write_json_atomic(archive_root / REPORT_NAME, report)
    try:
        for index, entry in enumerate(entries):
            source = Path(entry["original_path"])
            destination = Path(entry["archive_path"])
            source.rename(destination)
            moved_stat = destination.stat()
            observed = {
                "size_bytes": int(moved_stat.st_size),
                "mtime_ns": int(moved_stat.st_mtime_ns),
                "device": int(moved_stat.st_dev),
                "inode": int(moved_stat.st_ino),
            }
            expected = {name: int(entry[name]) for name in observed}
            if source.exists() or observed != expected:
                raise RuntimeError(
                    f"Artifact move validation failed: source={source}, destination={destination}, "
                    f"expected={expected}, observed={observed}."
                )
            report["moved_artifact_count"] = index + 1
            print(
                f"[archive] moved {index + 1}/{len(entries)} "
                f"{entry['material']}/{entry['snapshot']}/{entry['filename']}",
                flush=True,
            )
    except BaseException as error:
        report.update(
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(campaign_report_path, report)
        _write_json_atomic(archive_root / REPORT_NAME, report)
        raise

    report.update({"state": "complete", "completed_at_utc": _utc_now()})
    _write_json_atomic(campaign_report_path, report)
    _write_json_atomic(archive_root / REPORT_NAME, report)
    print(
        f"[archive] complete: artifacts={len(entries)}, bytes={report['total_bytes']}, "
        f"archive={archive_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive(args.campaign, args.archive)


if __name__ == "__main__":
    main()
