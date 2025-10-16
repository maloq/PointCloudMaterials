#!/usr/bin/env python3
"""Launch ablation experiments on SLURM by submitting one job per ablation value."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ablation import (  # noqa: E402
    _as_list,
    _format_value_for_path,
    _normalize_value,
    _json_safe,
    _value_to_display,
    aggregate_and_save_tables,
    load_ablation_config,
    MetricSpec,
    prepare_output_root,
    validate_ablation_config,
)
from omegaconf import DictConfig, OmegaConf  # noqa: E402

WORKER_SCRIPT = REPO_ROOT / "scripts" / "run_ablation_worker.py"
SBATCH_TEMPLATE = REPO_ROOT / "scripts" / "ablation_script_single.sh"


def _combine_flag_args(argv: List[str], flag: str) -> List[str]:
    combined: List[str] = []
    skip = False
    for idx, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token == flag:
            if idx + 1 >= len(argv):
                raise SystemExit(f"{flag} requires an argument.")
            combined.append(f"{flag}={argv[idx + 1]}")
            skip = True
        else:
            combined.append(token)
    return combined


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    if argv is None:
        argv_list = list(sys.argv[1:])
    else:
        argv_list = list(argv)
    argv_list = _combine_flag_args(argv_list, "--sbatch-arg")
    argv_list = _combine_flag_args(argv_list, "--srun-arg")
    parser = argparse.ArgumentParser(description="Launch ablation sweeps as separate sbatch jobs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the ablation YAML configuration.",
    )
    parser.add_argument(
        "--sbatch-arg",
        dest="sbatch_arg",
        action="append",
        default=[],
        help="Additional argument to pass to sbatch (specify multiple times).",
    )
    parser.add_argument(
        "--srun-arg",
        dest="sbatch_arg",
        action="append",
        help="Deprecated alias for --sbatch-arg.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Reserved for backward compatibility (ignored in sbatch mode).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polling running jobs for completion.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip aggregation even after all jobs finish.",
    )
    parser.add_argument("--wait", action="store_true", default=True, help="Wait for jobs (default).")
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Do not wait for jobs.")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Skip submission and aggregate metrics for the most recent output directory.",
    )
    return parser.parse_args(argv_list)


def _load_sbatch_defaults_from_script(script_path: Path) -> List[str]:
    """Extract a subset of SBATCH directives to reuse as default sbatch arguments."""
    if not script_path.exists():
        return []

    desired_keys = {
        "partition",
        "gres",
        "cpus-per-task",
        "mem",
        "gpus-per-node",
        "gpus",
        "nodes",
        "ntasks",
        "time",
        "account",
        "qos",
    }
    pattern = re.compile(r"^#SBATCH\s+--(?P<key>[^\s=]+)(?:[=\s]+(?P<value>[^#\s]+))?")
    defaults: Dict[str, Optional[str]] = {}

    for raw_line in script_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("#SBATCH"):
            continue
        cleaned = line.split("#", 1)[0].strip()
        match = pattern.match(cleaned)
        if not match:
            continue
        key = match.group("key")
        if key not in desired_keys:
            continue
        value = match.group("value")
        defaults[f"--{key}"] = value

    result: List[str] = []
    for flag, value in defaults.items():
        if value is None or value == "":
            result.append(flag)
        else:
            result.append(f"{flag}={value}")
    return result


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _detect_partition(flags: Sequence[str]) -> str:
    default_partition = "V100"
    for token in flags:
        if token.startswith("--partition="):
            value = token.split("=", 1)[1].strip()
            if value:
                return value
    return default_partition


def _flag_present(flags: Sequence[str], name: str) -> bool:
    prefix = f"--{name}"
    for token in flags:
        if token == prefix or token.startswith(prefix + "="):
            return True
    return False


def _default_root_and_run(cfg: DictConfig) -> Tuple[Path, str]:
    output_cfg = cfg.output if "output" in cfg and cfg.output is not None else DictConfig({})
    root_dir = Path(output_cfg.get("root_dir", "output/ablations")).expanduser()
    run_name = output_cfg.get("run_name")
    if not run_name:
        run_name = f"{cfg.variable.override.replace('.', '_')}_ablation"
    return root_dir, run_name


def _find_latest_output_dir(cfg: DictConfig) -> Path:
    root_dir, run_name = _default_root_and_run(cfg)
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"No ablation output directory found at {root_dir}")
    prefix = f"{run_name}_"
    candidates = [
        path for path in root_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed runs matching prefix '{prefix}' were found in {root_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _resume_latest_run(config_path: Path, cfg: DictConfig) -> None:
    latest_dir = _find_latest_output_dir(cfg)
    print(f"[Ablation] Resuming aggregation for latest output: {latest_dir}")  # noqa: T201
    try:
        from scripts import recover_ablation_results as recover  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Failed to import recover_ablation_results. Ensure project scripts are available.") from exc

    recover.main(["--output-root", str(latest_dir), "--config", str(config_path)])


def _write_manifest(manifest_path: Path, tasks: Sequence[Dict[str, Any]]) -> None:
    payload = [
        {
            "index": task["index"],
            "value": task["value_serializable"],
            "value_label": task["value_label"],
            "value_display": task["value_display"],
            "run_dir": str(task["run_dir"]),
            "worker_args": list(map(str, task.get("worker_args", []))),
            "sbatch_command": list(map(str, task.get("sbatch_command", []))),
            "wrap_command": task.get("wrap_command"),
            "job_id": task.get("job_id"),
            "status": task.get("status"),
            "error": task.get("error"),
        }
        for task in tasks
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


ACTIVE_JOB_STATES = {
    "PENDING",
    "CONFIGURING",
    "COMPLETING",
    "RUNNING",
    "SUSPENDED",
    "RESV_DEL_HOLD",
    "RESIZING",
    "STAGE_OUT",
    "STOPPED",
    "PREEMPTED",
}

FINAL_STATE_RETRIES = 12


def _sanitize_job_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_.")
    if not cleaned:
        cleaned = "ablation"
    return cleaned[:128]


def _build_wrap_command(worker_args: Sequence[str]) -> str:
    repo = shlex.quote(str(REPO_ROOT))
    python = shlex.quote(sys.executable)
    worker = " ".join(shlex.quote(token) for token in worker_args)
    return f"cd {repo} && {python} {worker}"


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)


def _parse_job_id(output: str) -> Optional[str]:
    text = output.strip()
    if not text:
        return None
    # sbatch --parsable returns the job ID (possibly with suffix)
    return text.splitlines()[0].strip()


def _query_squeue(job_ids: Sequence[str]) -> Dict[str, str]:
    if not job_ids:
        return {}
    joined = ",".join(job_ids)
    command = ["squeue", "-h", "-j", joined, "-o", "%i|%T"]
    result = _run_command(command)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"squeue failed (code {result.returncode}): {stderr}")

    mapping: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, _, state = line.partition("|")
        mapping[job_id.strip()] = state.strip()
    return mapping


def _query_sacct(job_id: str) -> Optional[str]:
    command = ["sacct", "-n", "-P", "-b", "-j", job_id, "-o", "JobID,State"]
    result = _run_command(command)
    if result.returncode != 0:
        return None
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.strip().split("|")
        if len(parts) < 2:
            continue
        job_field = parts[0].split(".", 1)[0]
        if job_field == job_id.split(".", 1)[0]:
            return parts[1]
    return None


def _wait_for_jobs(job_map: Dict[str, Dict[str, Any]], poll_interval: float) -> Dict[str, str]:
    remaining: Dict[str, Dict[str, Any]] = dict(job_map)
    final_states: Dict[str, str] = {}
    retries: Dict[str, int] = {job_id: 0 for job_id in remaining}

    while remaining:
        time.sleep(poll_interval)
        ids = list(remaining.keys())
        try:
            squeue_states = _query_squeue(ids)
        except RuntimeError as exc:
            print(f"[Ablation] Warning: {exc}. Retrying.", file=sys.stderr)  # noqa: T201
            continue

        for job_id in list(remaining):
            state = squeue_states.get(job_id)
            if state:
                normalized = state.upper()
                if normalized in ACTIVE_JOB_STATES:
                    continue
                final_states[job_id] = state
                remaining.pop(job_id, None)
                continue

            sacct_state = _query_sacct(job_id)
            if sacct_state:
                final_states[job_id] = sacct_state
                remaining.pop(job_id, None)
                continue

            retries[job_id] += 1
            if retries[job_id] >= FINAL_STATE_RETRIES:
                final_states[job_id] = "UNKNOWN"
                remaining.pop(job_id, None)

    return final_states


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.max_parallel:
        print("[Ablation] --max-parallel is ignored when submitting with sbatch.", file=sys.stderr)  # noqa: T201
    config_path = Path(args.config).expanduser().resolve()
    ablation_cfg = load_ablation_config(config_path)
    validate_ablation_config(ablation_cfg)
    if args.resume_latest:
        _resume_latest_run(config_path, ablation_cfg)
        return
    output_root, run_name = prepare_output_root(ablation_cfg)
    default_sbatch_args = _load_sbatch_defaults_from_script(SBATCH_TEMPLATE)
    if default_sbatch_args:
        print(  # noqa: T201
            f"[Ablation] Default sbatch args from {SBATCH_TEMPLATE.name}: {' '.join(default_sbatch_args)}"
        )
        requested_gpu_flags = [
            token for token in default_sbatch_args if token.startswith("--gres=") or token.startswith("--gpus")
        ]
        if requested_gpu_flags:
            print(f"[Ablation] GPU request detected: {' '.join(requested_gpu_flags)}")  # noqa: T201
    if not args.dry_run:
        resolved_payload = OmegaConf.to_container(ablation_cfg, resolve=True)
        with (output_root / "ablation_config_resolved.json").open("w", encoding="utf-8") as handle:
            json.dump(resolved_payload, handle, indent=2, sort_keys=True)

    override_key = ablation_cfg.variable.override
    values = _as_list(ablation_cfg.variable.get("values"))
    slurm_log_dir = output_root / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Ablation] SLURM logs will be written to {slurm_log_dir}")  # noqa: T201

    extra_sbatch_args = list(args.sbatch_arg or [])
    auto_sbatch_args: List[str] = []
    required_defaults = {
        "gpus-per-node": "1",
        "gpus": "1",
        "nodes": "1",
        "cpus-per-task": "8",
        "mem": "32GB",
        "partition": "H100",
    }
    existing_flags = [*default_sbatch_args, *extra_sbatch_args]
    for key, value in required_defaults.items():
        if not _flag_present(existing_flags + auto_sbatch_args, key):
            auto_sbatch_args.append(f"--{key}={value}")

    combined_gpu_flags = [*existing_flags, *auto_sbatch_args]
    requested_partition = _detect_partition(combined_gpu_flags)
    print(f"[Ablation] Requested partition: {requested_partition}")  # noqa: T201
    gpu_info_reported = False

    tasks: List[Dict[str, Any]] = []
    for idx, raw_value in enumerate(values):
        normalized = _normalize_value(raw_value)
        normalized_json = _json_safe(normalized)
        fallback_label = f"value_{idx + 1:02d}"
        value_label = _format_value_for_path(normalized, fallback=fallback_label)
        value_display = _value_to_display(normalized)
        run_dir = output_root / f"{override_key.replace('.', '_')}_{value_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        worker_args = [
            str(WORKER_SCRIPT),
            "--config",
            str(config_path),
            "--value-index",
            str(idx),
            "--override-key",
            override_key,
            "--output-root",
            str(output_root),
            "--run-output-dir",
            str(run_dir),
            "--run-name",
            run_name,
        ]
        wrap_command = _build_wrap_command(worker_args)

        job_name = _sanitize_job_name(f"{run_name}-{value_label}")
        stdout_path = slurm_log_dir / f"{job_name}_%j.out"
        stderr_path = slurm_log_dir / f"{job_name}_%j.err"

        sbatch_command = [
            "sbatch",
            "--parsable",
            *default_sbatch_args,
            *extra_sbatch_args,
            *auto_sbatch_args,
            f"--job-name={job_name}",
            f"--output={stdout_path}",
            f"--error={stderr_path}",
            f"--chdir={REPO_ROOT}",
            "--wrap",
            wrap_command,
        ]

        if not gpu_info_reported:
            gpu_flags = [
                token
                for token in sbatch_command
                if token.startswith("--gres") or token.startswith("--gpus")
            ]
            if gpu_flags:
                print(  # noqa: T201
                    f"[Ablation] GPU flags for sbatch: {' '.join(gpu_flags)} (partition: {requested_partition})"
                )
            else:
                print(  # noqa: T201
                    f"[Ablation] No GPU-related sbatch flags detected in sbatch command; defaulting to partition {requested_partition}."
                )
            gpu_info_reported = True

        tasks.append(
            {
                "index": idx,
                "value": raw_value,
                "value_serializable": normalized_json,
                "value_label": value_label,
                "value_display": value_display,
                "run_dir": run_dir,
                "worker_args": worker_args,
                "wrap_command": wrap_command,
                "sbatch_command": sbatch_command,
                "status": "pending",
            }
        )

    manifest_path = output_root / "jobs_manifest.json"
    _write_manifest(manifest_path, tasks)

    print(f"[Ablation] Launching {len(tasks)} sbatch jobs. Output root: {output_root}")  # noqa: T201
    if args.dry_run:
        for task in tasks:
            print(" ".join(shlex.quote(token) for token in task["sbatch_command"]))  # noqa: T201
        return

    job_map: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        command = task["sbatch_command"]
        result = _run_command(command)
        if result.returncode != 0:
            task["status"] = f"submission_failed({result.returncode})"
            task["error"] = result.stderr.strip() or "sbatch submission failed"
            print(  # noqa: T201
                f"[Ablation] sbatch submission failed for {task['value_label']}: {task['error']}"
            )
            continue
        job_id = _parse_job_id(result.stdout)
        if not job_id:
            task["status"] = "submission_failed(unknown_job_id)"
            task["error"] = result.stdout.strip() or "unknown sbatch response"
            print(  # noqa: T201
                f"[Ablation] sbatch submission returned unexpected output for {task['value_label']}: {result.stdout}"
            )
            continue
        task["job_id"] = job_id
        task["status"] = "submitted"
        job_map[job_id] = task
        print(f"[Ablation] Submitted job {job_id} for {task['value_label']} ({task['value_display']}).")  # noqa: T201

    _write_manifest(manifest_path, tasks)

    if not args.wait:
        print("[Ablation] Not waiting for sbatch jobs (--no-wait). Skipping aggregation.")  # noqa: T201
        return

    if job_map:
        final_states = _wait_for_jobs(job_map, args.poll_interval)
        for job_id, state in final_states.items():
            task = job_map[job_id]
            task["status"] = state
    else:
        print("[Ablation] No jobs were submitted successfully.")  # noqa: T201

    _write_manifest(manifest_path, tasks)

    statuses = {task.get("status") for task in tasks}
    print(f"[Ablation] Job statuses: {statuses}")  # noqa: T201

    incomplete = [
        task
        for task in tasks
        if task.get("status") and not str(task["status"]).upper().startswith("COMPLETED")
    ]
    if incomplete:
        print(  # noqa: T201
            "[Ablation] Warning: some jobs did not finish with COMPLETED status; aggregation may be partial."
        )

    if args.no_aggregate:
        print("[Ablation] Skipping aggregation by request (--no-aggregate).")  # noqa: T201
        return

    runs_for_aggregation: List[Dict[str, Any]] = []
    metrics_cfg = ablation_cfg.metrics
    final_metric_names = _as_list(metrics_cfg.final)
    best_specs = [MetricSpec(name=entry.name, mode=entry.mode) for entry in _as_list(metrics_cfg.get("best", []))]

    for task in tasks:
        run_dir = task["run_dir"]
        final_metrics = _read_json(run_dir / "final_metrics.json") or {}
        best_metrics = _read_json(run_dir / "best_metrics.json") or {}
        history_path = run_dir / "metrics_history.jsonl"
        training_dir_file = run_dir / "training_run_dir.txt"

        run_record = {
            "value": task["value_serializable"],
            "value_display": task["value_display"],
            "value_label": task["value_label"],
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "training_dir": _read_text(training_dir_file),
            "history_path": str(history_path.relative_to(output_root)) if history_path.exists() else None,
        }
        runs_for_aggregation.append(run_record)

    aggregate_and_save_tables(
        runs=runs_for_aggregation,
        output_dir=output_root,
        final_metric_names=final_metric_names,
        best_specs=best_specs,
        variable_name=override_key,
    )
    print("[Ablation] Aggregation complete.")  # noqa: T201


if __name__ == "__main__":
    main(sys.argv[1:])
