#!/usr/bin/env python3
"""Safely replace complete shooting text dumps with verified float32 binaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.shooting_binary import (  # noqa: E402
    FORMAT_NAME,
    SCHEMA_VERSION,
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    convert_shooting_trajectory,
)
from src.data_utils.shooting_dataset import (  # noqa: E402
    validate_complete_shooting_branch,
)


_BINARY_DIRNAME = "trajectory_binary_float32"
_TEXT_FILENAME = "trajectory.lammpstrj"
_REPORT_FILENAME = "binary_migration_float32.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_object(path: Path) -> dict[str, Any]:
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
        raise FileExistsError(
            f"Refusing to reuse an existing migration metadata temporary file: {temporary}"
        )
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
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_timesteps(manifest: dict[str, Any]) -> tuple[int, ...]:
    protocol = manifest["protocol"]
    interval = int(protocol["sample_interval_steps"])
    run_steps = int(protocol["run_steps"])
    timesteps = tuple(range(0, run_steps + 1, interval))
    if len(timesteps) != int(protocol["expected_frame_count"]):
        raise RuntimeError(
            f"Campaign timestep contract is inconsistent: run_steps={run_steps}, "
            f"interval={interval}, generated_frames={len(timesteps)}, "
            f"expected_frames={protocol['expected_frame_count']}."
        )
    return timesteps


def _validate_binary_for_migration(
    binary: ShootingBinaryTrajectory,
    *,
    source_size_bytes: int,
    source_sha256: str,
    expected_timesteps: tuple[int, ...],
    atom_count: int,
) -> dict[str, str]:
    if binary.storage_dtype != np.dtype("float32"):
        raise RuntimeError(
            f"Migration requires float32 binary storage, got {binary.storage_dtype.name}: "
            f"{binary.root}."
        )
    if binary.atom_count != atom_count or tuple(binary.timesteps.tolist()) != expected_timesteps:
        raise RuntimeError(
            f"Migrated binary violates the campaign contract: root={binary.root}, "
            f"expected_atoms={atom_count}, observed_atoms={binary.atom_count}, "
            f"expected_timesteps=[{expected_timesteps[0]}, {expected_timesteps[-1]}], "
            f"observed_timesteps=[{int(binary.timesteps[0])}, {int(binary.timesteps[-1])}]."
        )
    if int(binary.manifest["source"]["size_bytes"]) != source_size_bytes:
        raise RuntimeError(
            f"Binary source size provenance differs from the complete outcome: "
            f"binary={binary.root}, expected={source_size_bytes}, "
            f"observed={binary.manifest['source']['size_bytes']}."
        )
    recorded_source_sha256 = str(binary.manifest["provenance"].get("source_sha256", ""))
    if recorded_source_sha256 != source_sha256:
        raise RuntimeError(
            f"Binary source SHA-256 provenance mismatch: binary={binary.root}, "
            f"expected={source_sha256}, observed={recorded_source_sha256}."
        )
    checksums = binary.verify_checksums()
    source_semantic = binary.manifest["source"]["semantic_float32_sha256"]
    if (
        checksums["positions"] != source_semantic["positions"]
        or checksums["velocities"] != source_semantic["velocities"]
    ):
        raise RuntimeError(
            f"Float32 binary is not semantically identical to the source arrays: {binary.root}."
        )
    return checksums


def _write_branch_outcome(
    branch_dir: Path, outcome: dict[str, Any]
) -> None:
    _write_json_atomic(branch_dir / "outcome.json", outcome)
    status_path = branch_dir / "status.json"
    if status_path.is_file():
        status = _load_json_object(status_path)
        if status.get("state") != "complete" or status.get("branch_id") != outcome["branch_id"]:
            raise RuntimeError(
                f"Completed branch status disagrees with its outcome before migration: "
                f"{status_path}."
            )
        _write_json_atomic(status_path, outcome)
    _fsync_directory(branch_dir)


def _artifact_document(
    *,
    sizes: dict[str, int],
    source_size_bytes: int,
    source_sha256: str,
    checksums: dict[str, str],
    migrated_at: str,
    source_deleted: bool,
    source_deleted_at: str | None,
) -> dict[str, Any]:
    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "path": _BINARY_DIRNAME,
        "storage_dtype": "float32",
        "size_bytes": sizes["apparent_bytes"],
        "allocated_bytes": sizes["allocated_bytes"],
        "array_sha256": checksums,
        "migrated_at": migrated_at,
        "source_lammpstrj": {
            "path": _TEXT_FILENAME,
            "size_bytes": source_size_bytes,
            "sha256": source_sha256,
            "deleted": source_deleted,
            "deleted_at": source_deleted_at,
        },
    }


def _migrate_branch(
    root: Path,
    manifest: dict[str, Any],
    branch: dict[str, Any],
    outcome: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    branch_dir = root / str(branch["branch_dir"])
    source = branch_dir / _TEXT_FILENAME
    target = branch_dir / _BINARY_DIRNAME
    artifact = outcome.get("trajectory_artifact")
    if artifact is not None:
        validate_complete_shooting_branch(root, manifest, branch, outcome)
        source_record = artifact["source_lammpstrj"]
        expected_source_size = int(source_record["size_bytes"])
        expected_source_sha256 = str(source_record["sha256"])
        binary = ShootingBinaryTrajectory.load(target)
        checksums = _validate_binary_for_migration(
            binary,
            source_size_bytes=expected_source_size,
            source_sha256=expected_source_sha256,
            expected_timesteps=_expected_timesteps(manifest),
            atom_count=int(manifest["atom_count"]),
        )
        if source.is_file():
            if source.stat().st_size != expected_source_size:
                raise RuntimeError(
                    f"Residual text source size changed after migration metadata was written: "
                    f"{source}."
                )
            if _sha256_file(source) != expected_source_sha256:
                raise RuntimeError(
                    f"Residual text source checksum changed after migration metadata was written: "
                    f"{source}."
                )
            source.unlink()
            _fsync_directory(branch_dir)
        if source.exists():
            raise RuntimeError(f"Text source still exists after explicit deletion: {source}")
        if not bool(source_record.get("deleted")):
            updated = dict(outcome)
            updated_artifact = dict(artifact)
            updated_source = dict(source_record)
            updated_source["deleted"] = True
            updated_source["deleted_at"] = _utc_now()
            updated_artifact["source_lammpstrj"] = updated_source
            updated["trajectory_artifact"] = updated_artifact
            _write_branch_outcome(branch_dir, updated)
            outcome = updated
        return "already_migrated", {
            "branch_index": int(branch["branch_index"]),
            "branch_id": str(branch["branch_id"]),
            "binary_size_bytes": int(artifact["size_bytes"]),
            "source_size_bytes": expected_source_size,
            "source_sha256": expected_source_sha256,
            "array_sha256": checksums,
        }

    validate_complete_shooting_branch(root, manifest, branch, outcome)
    source_size_bytes = int(outcome["trajectory_size_bytes"])
    source_sha256 = _sha256_file(source)
    print(
        f"[shooting-migration] converting campaign={root.name} "
        f"branch={branch['branch_index']}/{len(manifest['branches']) - 1}",
        flush=True,
    )
    if target.exists():
        binary = ShootingBinaryTrajectory.load(target)
    else:
        binary = convert_shooting_trajectory(
            source,
            target,
            timesteps=_expected_timesteps(manifest),
            atom_count=int(manifest["atom_count"]),
            storage_dtype="float32",
            provenance={
                "campaign_manifest": str(root / "manifest.json"),
                "branch_index": int(branch["branch_index"]),
                "branch_id": str(branch["branch_id"]),
                "outcome_completed_at": outcome.get("completed_at"),
                "source_sha256": source_sha256,
            },
        )
    checksums = _validate_binary_for_migration(
        binary,
        source_size_bytes=source_size_bytes,
        source_sha256=source_sha256,
        expected_timesteps=_expected_timesteps(manifest),
        atom_count=int(manifest["atom_count"]),
    )
    sizes = binary_directory_sizes(target)
    migrated_at = _utc_now()
    updated = dict(outcome)
    updated["trajectory_artifact"] = _artifact_document(
        sizes=sizes,
        source_size_bytes=source_size_bytes,
        source_sha256=source_sha256,
        checksums=checksums,
        migrated_at=migrated_at,
        source_deleted=False,
        source_deleted_at=None,
    )
    _write_branch_outcome(branch_dir, updated)

    source.unlink()
    _fsync_directory(branch_dir)
    if source.exists():
        raise RuntimeError(f"Text source still exists after explicit deletion: {source}")
    updated["trajectory_artifact"]["source_lammpstrj"]["deleted"] = True
    updated["trajectory_artifact"]["source_lammpstrj"]["deleted_at"] = _utc_now()
    _write_branch_outcome(branch_dir, updated)
    validate_complete_shooting_branch(root, manifest, branch, updated)
    print(
        f"[shooting-migration] verified-and-deleted campaign={root.name} "
        f"branch={branch['branch_index']} freed_bytes={source_size_bytes - sizes['apparent_bytes']}",
        flush=True,
    )
    return "migrated", {
        "branch_index": int(branch["branch_index"]),
        "branch_id": str(branch["branch_id"]),
        "migrated_at": migrated_at,
        "binary_size_bytes": sizes["apparent_bytes"],
        "binary_allocated_bytes": sizes["allocated_bytes"],
        "source_size_bytes": source_size_bytes,
        "source_sha256": source_sha256,
        "array_sha256": checksums,
    }


def _migrate_branch_worker(
    root: Path, manifest: dict[str, Any], branch: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    outcome = _load_json_object(root / str(branch["branch_dir"]) / "outcome.json")
    return _migrate_branch(root, manifest, branch, outcome)


def _migrate_campaign(root: Path, *, workers: int = 1) -> dict[str, Any]:
    manifest = _load_json_object(root / "manifest.json")
    if manifest.get("campaign_type") != "position_conditioned_langevin_nvt_shooting":
        raise ValueError(
            f"Unsupported campaign_type={manifest.get('campaign_type')!r}: "
            f"{root / 'manifest.json'}."
        )
    lock_path = root / "binary_migration_float32.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another float32 migration process holds the campaign lock: {lock_path}."
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(
            json.dumps(
                {"pid": os.getpid(), "hostname": os.uname().nodename, "started_at": _utc_now()}
            )
            + "\n"
        )
        lock.flush()
        os.fsync(lock.fileno())

        report_path = root / _REPORT_FILENAME
        prior_entries: dict[int, dict[str, Any]] = {}
        started_at = _utc_now()
        if report_path.is_file():
            previous = _load_json_object(report_path)
            started_at = str(previous["started_at"])
            prior_entries = {
                int(entry["branch_index"]): entry for entry in previous["branches"]
            }

        worker_count = int(workers)
        if worker_count <= 0 or worker_count > 8:
            raise ValueError(f"workers must be within [1, 8], got {worker_count}.")
        migrated_now = 0
        already_migrated = 0
        incomplete = 0
        complete_branches: list[dict[str, Any]] = []
        for branch in manifest["branches"]:
            branch_dir = root / str(branch["branch_dir"])
            outcome_path = branch_dir / "outcome.json"
            if not outcome_path.is_file():
                incomplete += 1
                continue
            outcome = _load_json_object(outcome_path)
            if outcome.get("state") != "complete":
                incomplete += 1
                continue
            complete_branches.append(branch)

        if worker_count == 1:
            results = [
                _migrate_branch_worker(root, manifest, branch)
                for branch in complete_branches
            ]
        else:
            results = []
            process_batch_size = worker_count * 4
            for start in range(0, len(complete_branches), process_batch_size):
                branch_batch = complete_branches[start : start + process_batch_size]
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    futures = [
                        executor.submit(_migrate_branch_worker, root, manifest, branch)
                        for branch in branch_batch
                    ]
                    results.extend(future.result() for future in futures)

        for action, entry in results:
            if action == "migrated":
                migrated_now += 1
            else:
                already_migrated += 1
            prior_entries[int(entry["branch_index"])] = entry

        final_entries = [prior_entries[index] for index in sorted(prior_entries)]
        report = {
            "schema_version": 1,
            "state": "complete_for_current_outcomes",
            "campaign_root": str(root),
            "started_at": started_at,
            "updated_at": _utc_now(),
            "intended_branch_count": len(manifest["branches"]),
            "migrated_complete_branch_count": len(final_entries),
            "incomplete_branch_count": incomplete,
            "migrated_now_count": migrated_now,
            "already_migrated_count": already_migrated,
            "worker_count": worker_count,
            "source_size_bytes": sum(
                int(entry["source_size_bytes"]) for entry in final_entries
            ),
            "binary_size_bytes": sum(
                int(entry["binary_size_bytes"]) for entry in final_entries
            ),
            "freed_apparent_bytes": sum(
                int(entry["source_size_bytes"]) - int(entry["binary_size_bytes"])
                for entry in final_entries
            ),
            "branches": final_entries,
        }
        _write_json_atomic(report_path, report)
        print(
            f"[shooting-migration] campaign-complete root={root} "
            f"migrated={len(final_entries)} incomplete={incomplete} report={report_path}",
            flush=True,
        )
        return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert strict-complete shooting trajectories to verified float32 binary "
            "directories and explicitly delete only their branch-root text dumps."
        )
    )
    parser.add_argument("--campaign-root", action="append", required=True, type=Path)
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        required=True,
        help="Required acknowledgement that verified trajectory.lammpstrj files are deleted.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    roots = tuple(path.expanduser().resolve() for path in args.campaign_root)
    if len(set(roots)) != len(roots):
        raise ValueError(f"Duplicate --campaign-root values: {[str(root) for root in roots]}.")
    reports = [_migrate_campaign(root, workers=args.workers) for root in roots]
    print(
        json.dumps(
            {
                "campaign_count": len(reports),
                "migrated_complete_branch_count": sum(
                    report["migrated_complete_branch_count"] for report in reports
                ),
                "incomplete_branch_count": sum(
                    report["incomplete_branch_count"] for report in reports
                ),
                "freed_apparent_bytes": sum(
                    report["freed_apparent_bytes"] for report in reports
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
