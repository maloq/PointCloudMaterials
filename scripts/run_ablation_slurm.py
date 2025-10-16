#!/usr/bin/env python3
"""Launch ablation experiments in parallel via SLURM srun."""

from __future__ import annotations

import argparse
import json
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
from omegaconf import OmegaConf  # noqa: E402

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
    argv_list = _combine_flag_args(argv_list, "--srun-arg")
    parser = argparse.ArgumentParser(description="Launch ablation sweeps using parallel srun jobs.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the ablation YAML configuration.",
    )
    parser.add_argument(
        "--srun-arg",
        action="append",
        default=[],
        help="Additional argument to pass to srun (specify multiple times).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Maximum number of concurrent srun processes (0 means no limit).",
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
    return parser.parse_args(argv_list)


def _load_srun_defaults_from_sbatch(script_path: Path) -> List[str]:
    """Extract a subset of SBATCH directives to reuse as default srun arguments."""
    if not script_path.exists():
        return []

    desired_keys = {
        "partition",
        "gres",
        "cpus-per-task",
        "mem",
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


def _launch_command(command: Sequence[str], workdir: Path) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=str(workdir))


def _poll_processes(processes: List[Dict[str, Any]]) -> None:
    still_running: List[Dict[str, Any]] = []
    for entry in processes:
        proc: subprocess.Popen = entry["proc"]
        ret = proc.poll()
        if ret is None:
            still_running.append(entry)
        else:
            entry["returncode"] = ret
            print(f"[srun] command finished with code {ret}: {' '.join(entry['command'])}")  # noqa: T201
    processes[:] = still_running


def _wait_for_processes(processes: List[Dict[str, Any]], poll_interval: float) -> None:
    while processes:
        _poll_processes(processes)
        if processes:
            time.sleep(poll_interval)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    ablation_cfg = load_ablation_config(config_path)
    validate_ablation_config(ablation_cfg)
    output_root, run_name = prepare_output_root(ablation_cfg)
    default_srun_args = _load_srun_defaults_from_sbatch(SBATCH_TEMPLATE)
    if default_srun_args:
        print(  # noqa: T201
            f"[Ablation] Default srun args from {SBATCH_TEMPLATE.name}: {' '.join(default_srun_args)}"
        )
    if not args.dry_run:
        resolved_payload = OmegaConf.to_container(ablation_cfg, resolve=True)
        with (output_root / "ablation_config_resolved.json").open("w", encoding="utf-8") as handle:
            json.dump(resolved_payload, handle, indent=2, sort_keys=True)

    override_key = ablation_cfg.variable.override
    values = _as_list(ablation_cfg.variable.values)

    tasks: List[Dict[str, Any]] = []
    for idx, raw_value in enumerate(values):
        normalized = _normalize_value(raw_value)
        normalized_json = _json_safe(normalized)
        fallback_label = f"value_{idx + 1:02d}"
        value_label = _format_value_for_path(normalized, fallback=fallback_label)
        value_display = _value_to_display(normalized)
        run_dir = output_root / f"{override_key.replace('.', '_')}_{value_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "srun",
            *default_srun_args,
            *args.srun_arg,
            sys.executable,
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

        tasks.append(
            {
                "index": idx,
                "value": raw_value,
                "value_serializable": normalized_json,
                "value_label": value_label,
                "value_display": value_display,
                "run_dir": run_dir,
                "command": command,
                "status": "pending",
            }
        )

    manifest_path = output_root / "jobs_manifest.json"
    manifest_payload = [
        {
            "index": task["index"],
            "value": task["value_serializable"],
            "value_label": task["value_label"],
            "value_display": task["value_display"],
            "command": list(map(str, task["command"])),
        }
        for task in tasks
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)

    print(f"[Ablation] Launching {len(tasks)} srun jobs. Output root: {output_root}")  # noqa: T201
    if args.dry_run:
        for task in tasks:
            print(" ".join(task["command"]))  # noqa: T201
        return

    running: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for task in tasks:
        while args.max_parallel and len(running) >= args.max_parallel:
            _poll_processes(running)
            if len(running) >= args.max_parallel:
                time.sleep(args.poll_interval)

        print(f"[Ablation] srun launch: {' '.join(task['command'])}")  # noqa: T201
        proc = _launch_command(task["command"], REPO_ROOT)
        task_entry = dict(task)
        task_entry["proc"] = proc
        running.append(task_entry)
        results.append(task_entry)

    if args.wait:
        _wait_for_processes(running, args.poll_interval)
    else:
        print("[Ablation] Not waiting for srun jobs (--no-wait). Skipping aggregation.")  # noqa: T201
        return

    for task in results:
        ret = task.get("returncode")
        if ret is None and "proc" in task:
            ret = task["proc"].wait()
            task["returncode"] = ret
        task["status"] = "completed" if ret == 0 else f"failed({ret})"

    statuses = {task["status"] for task in results}
    print(f"[Ablation] Job statuses: {statuses}")  # noqa: T201

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
