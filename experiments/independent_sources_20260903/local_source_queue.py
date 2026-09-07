#!/usr/bin/env python3
"""Run prepared 510/520 K Al sources locally, with a fixed overnight cutoff."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import traceback

import psutil

REPOSITORY = Path(__file__).resolve().parents[2]
WORKER = REPOSITORY / "experiments/independent_sources_20260903/independent_meam_510_520K_sources.py"
PREPARED_FILES = {"metadata.json", "melt.in.lammps", "source.in.lammps"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_unsubmitted(root: Path, spec: dict) -> None:
    """Use the records written by this campaign's submit_next_wave producer."""
    records = [json.loads(line) for line in (root / "slurm/submission_chain.jsonl").read_text().splitlines()]
    records.append(json.loads((root / "slurm/active_submission.json").read_text()))
    for record in records:
        match = re.fullmatch(r"(\d+)-(\d+)%(\d+)", record["array_spec"])
        if match is None:
            raise ValueError(f"Unexpected source array specification: {record!r}")
        first, last, _ = map(int, match.groups())
        if first <= spec["run_index"] <= last:
            raise RuntimeError(f"Source {spec['run_index']} was submitted to Slurm: {record!r}")
    run_dir = root / spec["run_dir"]
    observed = {path.name for path in run_dir.iterdir()}
    if observed != PREPARED_FILES:
        raise RuntimeError(f"Source directory is not pristine: {run_dir}; files={sorted(observed)}")


def signal_process(process: psutil.Process, signum: int) -> None:
    try:
        process.send_signal(signum)
    except psutil.NoSuchProcess:
        print(f"{now()} Process {process.pid} has already exited.", flush=True)


def stop_tree(pid: int, grace_seconds: float) -> None:
    # MPICH and the installed LAMMPS Python wrapper create separate sessions.
    # Freeze each parent before enumerating its children, preventing new forks.
    processes = []

    def freeze(parent: psutil.Process) -> None:
        try:
            parent.suspend()
            processes.append(parent)
            children = parent.children()
        except psutil.NoSuchProcess:
            print(f"{now()} Process {parent.pid} exited before cutoff capture.", flush=True)
            return
        for child in children:
            freeze(child)

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"{now()} Worker {pid} has already exited.", flush=True)
        return
    freeze(root)
    print(f"{now()} Stopping worker tree rooted at {pid}: {len(processes)} processes.", flush=True)
    for child in reversed(processes):
        signal_process(child, signal.SIGTERM)
        signal_process(child, signal.SIGCONT)
    # Leave the root's exit status for subprocess.Popen.wait() to reap.
    _, alive = psutil.wait_procs(processes[1:], timeout=grace_seconds)
    if root.is_running() and root.status() != psutil.STATUS_ZOMBIE:
        alive.append(root)
    for child in alive:
        signal_process(child, signal.SIGKILL)


def wait_for_worker(process: subprocess.Popen, deadline: float, heartbeat, grace_seconds: float = 20.0) -> bool:
    """Return True on cutoff; allow TERM cleanup before the hard deadline."""
    while process.poll() is None:
        remaining = deadline - time.time() - grace_seconds
        if remaining <= 0:
            print(f"{now()} Cutoff reached; terminating worker tree {process.pid}.", flush=True)
            stop_tree(process.pid, grace_seconds)
            process.wait()
            return True
        try:
            process.wait(timeout=min(30.0, remaining))
        except subprocess.TimeoutExpired:
            heartbeat()
    return False


def mark_interrupted(run_dir: Path, reason: str) -> None:
    path = run_dir / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    if status.get("state") == "complete":
        return
    status.update(
        state="interrupted", updated_at=now(), reason=reason,
        partial_artifacts_preserved=True,
        restart_files=[str(p) for p in sorted(run_dir.glob("*restart*.bin"))],
        recovery_note="Partial sources require explicit recovery; the source runner refuses to overwrite them.",
    )
    write_json(path, status)


def run(args: argparse.Namespace) -> None:
    root = args.campaign_root.resolve()
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time() + 20:
        raise ValueError(f"Deadline must be in the future and include its UTC offset: {args.deadline}")
    if "SLURM_JOB_ID" in os.environ:
        raise RuntimeError("This queue must run outside Slurm.")
    manifest_bytes = (root / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != (root / "manifest.sha256").read_text().split()[0]:
        raise RuntimeError(f"Immutable campaign manifest checksum failed: {root}")
    manifest = json.loads(manifest_bytes)
    specs = [manifest["runs"][index] for index in args.task_indices]
    if len(set(args.task_indices)) != len(specs):
        raise ValueError(f"Duplicate queue tasks: {args.task_indices}")
    for index, spec in zip(args.task_indices, specs):
        if index < 0 or spec["run_index"] != index:
            raise ValueError(f"Invalid source task mapping: {index}, {spec}")
        check_unsubmitted(root, spec)
    plan = {
        "campaign_root": str(root), "manifest_sha256": digest,
        "host": socket.gethostname(), "deadline": cutoff.isoformat(),
        "mpi_ranks": manifest["execution"]["mpi_ranks_per_run"],
        "task_indices": args.task_indices, "sources": specs,
        "submission_check": "Campaign submission_chain.jsonl and active_submission.json, rechecked before each launch; no live squeue access on this host.",
        "cutoff_policy": "Freeze and capture the complete worker process tree, SIGTERM 20 seconds before deadline, then SIGKILL for remaining processes. Preserve partial artifacts.",
        "estimated_seconds_per_source": 7200,
    }
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    environment = os.environ.copy()
    environment.update(PATH=f"{sys.prefix}/bin:{environment['PATH']}", OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    status = {"state": "running", "started_at": now(), "pid": os.getpid(),
              "host": socket.gethostname(), "deadline": cutoff.isoformat(), "tasks": []}

    def heartbeat() -> None:
        status["updated_at"] = now()
        write_json(output / "status.json", status)

    process = None
    run_dir = None
    try:
        heartbeat()
        for spec in specs:
            if time.time() >= cutoff.timestamp() - 20:
                status["state"] = "stopped_at_deadline"
                break
            check_unsubmitted(root, spec)
            run_dir = root / spec["run_dir"]
            index = spec["run_index"]
            log_path = output / f"source_{index:03d}.log"
            command = [sys.executable, "-u", str(WORKER), "run-local-task",
                       "--campaign-root", str(root), "--task-index", str(index)]
            print(f"{now()} Starting source {index} at {spec['temperature_K']} K; log={log_path}", flush=True)
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=REPOSITORY, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                task = {"task_index": index, "run_dir": str(run_dir), "state": "running",
                        "pid": process.pid, "started_at": now(), "log": str(log_path)}
                status["tasks"].append(task)
                heartbeat()
                stopped = wait_for_worker(process, cutoff.timestamp(), heartbeat)
            task.update(returncode=process.returncode, finished_at=now())
            if stopped:
                mark_interrupted(run_dir, f"Local queue cutoff: {cutoff.isoformat()}")
                task["state"] = json.loads((run_dir / "status.json").read_text())["state"]
                status["state"] = "stopped_at_deadline"
                break
            if process.returncode != 0:
                task["state"] = "failed"
                raise RuntimeError(f"Source {index} exited {process.returncode}; inspect {log_path}")
            outcome = json.loads((run_dir / "outcome.json").read_text())
            if outcome["state"] != "complete":
                raise RuntimeError(f"Source {index} returned without a complete outcome: {run_dir}")
            task["state"] = "complete"
            print(f"{now()} Source {index} complete: {outcome['frame_count']} verified frames.", flush=True)
            process = None
            heartbeat()
        else:
            status["state"] = "complete"
        status["finished_at"] = now()
        heartbeat()
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as error:
        if process is not None:
            was_running = process.poll() is None
            stop_tree(process.pid, grace_seconds=2.0)
            process.wait()
            if run_dir is not None and was_running:
                mark_interrupted(run_dir, f"Queue failure: {error!r}")
                status["tasks"][-1].update(state="interrupted", returncode=process.returncode, finished_at=now())
        status.update(state="failed", finished_at=now(), error=repr(error), traceback=traceback.format_exc())
        heartbeat()
        raise


def handle_termination(signum, frame) -> None:
    raise InterruptedError(f"Queue received signal {signum}; stopping its worker tree.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-indices", type=int, nargs="+", required=True)
    parser.add_argument("--deadline", required=True, help="ISO timestamp with UTC offset")
    parser.add_argument("--check-only", action="store_true")
    signal.signal(signal.SIGTERM, handle_termination)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
