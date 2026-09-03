#!/usr/bin/env python3
"""Wait for one local shooting campaign, then run a second campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.synthetic.atomistic.lammps_shooting import run_local_campaign  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local-MPI shooting campaign after another one completes."
    )
    parser.add_argument("--wait-campaign-root", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", default=30, type=int)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object, got {type(document).__name__}.")
    return document


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    wait_root = args.wait_campaign_root.expanduser().resolve()
    root = args.campaign_root.expanduser().resolve()
    poll_seconds = int(args.poll_seconds)
    if poll_seconds <= 0:
        raise ValueError(f"poll-seconds must be > 0, got {poll_seconds}.")
    if "SLURM_JOB_ID" in os.environ:
        raise RuntimeError(
            "The local follow-up driver refuses to run inside Slurm; "
            f"detected SLURM_JOB_ID={os.environ['SLURM_JOB_ID']!r}."
        )

    followup_lock_path = root / "local_followup.lock"
    with followup_lock_path.open("a+", encoding="utf-8") as followup_lock:
        try:
            fcntl.flock(followup_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another follow-up driver holds the execution lock: {followup_lock_path}."
            ) from error
        started_at = datetime.now(timezone.utc).isoformat()
        followup_lock.seek(0)
        followup_lock.truncate()
        followup_lock.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": os.uname().nodename,
                    "started_at": started_at,
                    "wait_campaign_root": str(wait_root),
                    "campaign_root": str(root),
                },
                sort_keys=True,
            )
            + "\n"
        )
        followup_lock.flush()
        _write_json_atomic(
            root / "status.json",
            {
                "schema_version": 1,
                "state": "waiting",
                "updated_at": started_at,
                "execution_mode": "local_mpiexec_followup",
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "wait_campaign_root": str(wait_root),
            },
        )

        wait_lock_path = wait_root / "local_campaign.lock"
        with wait_lock_path.open("a+", encoding="utf-8") as wait_lock:
            while True:
                try:
                    fcntl.flock(wait_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(poll_seconds)

            summary_path = wait_root / "summary.json"
            try:
                summary = _load_json(summary_path)
                if summary.get("state") != "complete" or summary.get("branch_count") != 40:
                    raise RuntimeError(
                        f"Prerequisite campaign summary is not strictly complete: {summary_path}; "
                        f"state={summary.get('state')!r}, branch_count={summary.get('branch_count')!r}."
                    )
            except Exception as error:
                _write_json_atomic(
                    root / "status.json",
                    {
                        "schema_version": 1,
                        "state": "failed",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "execution_mode": "local_mpiexec_followup",
                        "hostname": os.uname().nodename,
                        "pid": os.getpid(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise

        run_local_campaign(root, start_index=0)


if __name__ == "__main__":
    main()
