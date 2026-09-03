#!/usr/bin/env python3
"""Slurm campaign for nine new velocity-initialized homogeneous MEAM runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_lammps_homogeneous_campaign import _write_json_atomic  # noqa: E402
from scripts.run_lammps_unseeded_meam_ensemble import (  # noqa: E402
    prepare as prepare_ensemble,
    run as run_ensemble,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
)
from src.temporal_vamp.simulation_catalog import (  # noqa: E402
    discover_simulation_catalog,
    validate_dump_scan,
)


SCHEMA_VERSION = 1
RUN_SPECS = (
    (400.0, 35911),
    (400.0, 35923),
    (400.0, 35933),
    (450.0, 35951),
    (450.0, 35963),
    (450.0, 35977),
    (500.0, 35993),
    (500.0, 36007),
    (500.0, 36013),
)
MEASUREMENT_STEPS = 200_000
TIMESTEP_FS = 3.0
EQUILIBRATION_STEPS = 5_000
SAMPLE_INTERVAL_STEPS = 1_000
MPI_RANKS = 48
ARRAY_CONCURRENCY = 3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, execute, or summarize nine new MEAM source runs."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", required=True, type=Path)
    prepare_parser.add_argument("--output-root", required=True, type=Path)
    task_parser = subparsers.add_parser("run-task")
    task_parser.add_argument("--output-root", required=True, type=Path)
    task_parser.add_argument("--task-index", required=True, type=int)
    submit_parser = subparsers.add_parser("submit-next-wave")
    submit_parser.add_argument("--output-root", required=True, type=Path)
    submit_parser.add_argument("--start-index", required=True, type=int)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def _run_name(run_index: int, temperature_K: float, velocity_seed: int) -> str:
    return f"run_{run_index:03d}_T{temperature_K:g}_velocity_{velocity_seed}"


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object.")
    return document


def _write_status(output_root: Path, state: str, **details: object) -> None:
    _write_json_atomic(
        output_root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _write_slurm_scripts(output_root: Path) -> None:
    slurm_root = output_root / "slurm"
    runner = REPOSITORY_ROOT / "scripts/run_lammps_unseeded_meam_source_followup.py"
    common = f"""set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
"""
    (slurm_root / "run_task.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_source
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks={MPI_RANKS}
#SBATCH --ntasks-per-node={MPI_RANKS}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output={slurm_root}/%A_%a.out
#SBATCH --error={slurm_root}/%A_%a.err

{common}python {runner} run-task --output-root {output_root} \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}"
""",
        encoding="utf-8",
    )
    (slurm_root / "submit_wave.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_src_ctl
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output={slurm_root}/controller_%j.out
#SBATCH --error={slurm_root}/controller_%j.err

{common}python {runner} submit-next-wave --output-root {output_root} \\
  --start-index "${{SOURCE_START}}"
""",
        encoding="utf-8",
    )
    (slurm_root / "summarize.sbatch").write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_src_sum
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output={slurm_root}/summary_%j.out
#SBATCH --error={slurm_root}/summary_%j.err

{common}python {runner} summarize --output-root {output_root}
""",
        encoding="utf-8",
    )


def prepare(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Source campaign output exists and will not be overwritten: {output_root}."
        )
    all_seeds = tuple(seed for _, seed in RUN_SPECS)
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError(f"Source-run velocity seeds are not unique: {all_seeds}.")
    output_root.mkdir(parents=True)
    (output_root / "runs").mkdir()
    (output_root / "slurm").mkdir()
    runs: list[dict[str, object]] = []
    for run_index, (temperature_K, velocity_seed) in enumerate(RUN_SPECS):
        run_name = _run_name(run_index, temperature_K, velocity_seed)
        run_root = output_root / "runs" / run_name
        prepare_ensemble(
            source_root,
            run_root,
            (velocity_seed,),
            MEASUREMENT_STEPS,
            temperature_K,
            TIMESTEP_FS,
            EQUILIBRATION_STEPS,
            SAMPLE_INTERVAL_STEPS,
            MPI_RANKS,
        )
        runs.append(
            {
                "run_index": run_index,
                "run_name": run_name,
                "run_root": str(run_root.relative_to(output_root)),
                "temperature_K": temperature_K,
                "velocity_seed": velocity_seed,
            }
        )
    _write_json_atomic(
        output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "nine additional independent dynamical source runs for the shooting "
                "predictor parent pool"
            ),
            "independence_definition": (
                "shared validated liquid positions and cell; globally unique "
                "Maxwell-Boltzmann velocity initialization per NPT trajectory"
            ),
            "source_campaign": str(source_root),
            "run_count": len(runs),
            "runs_by_temperature": {
                f"{temperature_K:g}": sum(
                    run["temperature_K"] == temperature_K for run in runs
                )
                for temperature_K in (400.0, 450.0, 500.0)
            },
            "candidate_parent_configurations_per_nucleated_run": 2,
            "protocol": {
                "atom_count": 70_304,
                "ensemble": "NPT",
                "measurement_steps": MEASUREMENT_STEPS,
                "measurement_duration_ps": MEASUREMENT_STEPS
                * TIMESTEP_FS
                / 1000.0,
                "timestep_fs": TIMESTEP_FS,
                "equilibration_steps": EQUILIBRATION_STEPS,
                "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
            },
            "execution": {
                "launcher": "Slurm srun PMI2",
                "partition": "CPU",
                "mpi_ranks_per_run": MPI_RANKS,
                "memory_per_run": "24G",
                "time_limit_per_run": "04:00:00",
                "array_concurrency": ARRAY_CONCURRENCY,
            },
            "runs": runs,
        },
    )
    _write_slurm_scripts(output_root)
    _write_status(
        output_root,
        "prepared",
        completed_run_count=0,
        pending_run_count=len(runs),
    )


def run_task(output_root: Path, task_index: int) -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Source run-task requires a Slurm allocation.")
    manifest = _load_json(output_root / "manifest.json")
    runs = manifest["runs"]
    if not isinstance(runs, list) or task_index < 0 or task_index >= len(runs):
        raise IndexError(f"task_index={task_index} is outside [0, {len(runs)}).")
    run = runs[task_index]
    if not isinstance(run, dict) or int(run["run_index"]) != task_index:
        raise RuntimeError(f"Manifest task mapping is invalid at index {task_index}: {run}.")
    slurm_tasks = int(os.environ.get("SLURM_NTASKS", "0"))
    if slurm_tasks != MPI_RANKS:
        raise RuntimeError(
            f"Slurm allocated SLURM_NTASKS={slurm_tasks}, expected {MPI_RANKS}."
        )
    run_ensemble(
        output_root / str(run["run_root"]),
        (int(run["velocity_seed"]),),
        MEASUREMENT_STEPS,
        None,
        float(run["temperature_K"]),
        TIMESTEP_FS,
        EQUILIBRATION_STEPS,
        SAMPLE_INTERVAL_STEPS,
        MPI_RANKS,
    )


def _active_submission_conflicts(output_root: Path) -> list[str]:
    active_path = output_root / "slurm" / "active_submission.json"
    if not active_path.is_file():
        return []
    active = _load_json(active_path)
    job_ids = [str(active["array_job_id"]), str(active["successor_job_id"])]
    queued = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"],
        check=True,
        text=True,
        capture_output=True,
    )
    current_job_id = os.environ.get("SLURM_JOB_ID")
    return sorted(
        {
            line.strip()
            for line in queued.stdout.splitlines()
            if line.strip() and line.strip() != current_job_id
        }
    )


def submit_next_wave(output_root: Path, start_index: int) -> dict[str, object]:
    manifest = _load_json(output_root / "manifest.json")
    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise TypeError(f"{output_root / 'manifest.json'}: runs must be nonempty.")
    if start_index < 0 or start_index >= len(runs):
        raise IndexError(f"start_index={start_index} is outside [0, {len(runs)}).")
    conflicts = _active_submission_conflicts(output_root)
    if conflicts:
        raise RuntimeError(
            f"Refusing duplicate submission while prior campaign jobs remain active: {conflicts}."
        )
    wave_size = int(manifest["execution"]["array_concurrency"])
    stop_index = min(start_index + wave_size - 1, len(runs) - 1)
    array_spec = f"{start_index}-{stop_index}%{wave_size}"
    array_submission = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={array_spec}",
            str(output_root / "slurm" / "run_task.sbatch"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    array_job_id = array_submission.stdout.strip()
    if not array_job_id.isdigit():
        raise RuntimeError(
            f"Slurm returned invalid array ID for {array_spec}: "
            f"stdout={array_submission.stdout!r}, stderr={array_submission.stderr!r}."
        )
    if stop_index + 1 < len(runs):
        successor_kind = "controller"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            f"--export=ALL,SOURCE_START={stop_index + 1}",
            str(output_root / "slurm" / "submit_wave.sbatch"),
        ]
    else:
        successor_kind = "summary"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            str(output_root / "slurm" / "summarize.sbatch"),
        ]
    successor_submission = subprocess.run(
        successor_command,
        check=True,
        text=True,
        capture_output=True,
    )
    successor_job_id = successor_submission.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(
            f"Slurm returned invalid {successor_kind} ID: "
            f"stdout={successor_submission.stdout!r}, "
            f"stderr={successor_submission.stderr!r}."
        )
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "submitting_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_spec": array_spec,
        "array_job_id": array_job_id,
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
    }
    with (output_root / "slurm" / "submission_chain.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json_atomic(output_root / "slurm" / "active_submission.json", record)
    _write_status(
        output_root,
        "submitted",
        active_array_spec=array_spec,
        active_array_job_id=array_job_id,
        successor_kind=successor_kind,
        successor_job_id=successor_job_id,
    )
    print(json.dumps(record, indent=2))
    return record


def summarize(output_root: Path) -> dict[str, object]:
    manifest = _load_json(output_root / "manifest.json")
    runs = manifest["runs"]
    if not isinstance(runs, list):
        raise TypeError(f"{output_root / 'manifest.json'}: runs must be a list.")
    missing_or_incomplete: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError(f"Invalid run record in manifest: {run!r}.")
        run_root = output_root / str(run["run_root"])
        status_path = run_root / "status.json"
        if not status_path.is_file() or _load_json(status_path).get("state") != "complete":
            missing_or_incomplete.append(str(run["run_name"]))
            continue
        sub_summary = _load_json(run_root / "campaign_summary.json")
        if (
            sub_summary.get("completed_replica_count") != 1
            or sub_summary.get("requested_velocity_seeds")
            != [int(run["velocity_seed"])]
        ):
            raise RuntimeError(f"Invalid completed-run summary: {run_root}.")
    if missing_or_incomplete:
        raise RuntimeError(
            f"Cannot summarize: {len(missing_or_incomplete)} runs are incomplete: "
            f"{missing_or_incomplete}."
        )

    first_run_manifest = _load_json(
        output_root / str(runs[0]["run_root"]) / "manifest.json"
    )
    entries = discover_simulation_catalog(
        output_root,
        campaign_globs=("runs/run_*",),
        cache_root=output_root / "catalog_cache",
        required_atom_count=70_304,
        required_potential_parameter_sha256=str(
            first_run_manifest["potential"]["parameter_sha256"]
        ),
        required_crystal_seed=None,
        require_periodic=True,
    )
    if len(entries) != len(runs):
        raise RuntimeError(
            f"Catalog found {len(entries)} completed trajectories, expected {len(runs)}."
        )
    observed_pairs = {
        (entry.metadata.temperature_K, entry.metadata.velocity_seed) for entry in entries
    }
    if observed_pairs != set(RUN_SPECS):
        raise RuntimeError(
            f"Catalog temperature/seed pairs differ from the design: {observed_pairs}."
        )
    for entry in entries:
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(entry.trajectory_path)
        validate_dump_scan(entry.metadata, scan, entry.trajectory_path)

    nucleated = sum(entry.metadata.nucleation_observed for entry in entries)
    eligible = sum(
        entry.metadata.nucleation_observed
        and entry.metadata.nucleation_time_ps is not None
        and entry.metadata.nucleation_time_ps >= 12.0
        for entry in entries
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_run_count": len(entries),
        "runs_by_temperature": {
            f"{temperature_K:g}": sum(
                entry.metadata.temperature_K == temperature_K for entry in entries
            )
            for temperature_K in (400.0, 450.0, 500.0)
        },
        "nucleation_observed_count": int(nucleated),
        "shooting_parent_eligible_run_count": int(eligible),
        "candidate_parent_configuration_count": 2 * int(eligible),
    }
    _write_json_atomic(output_root / "campaign_summary.json", summary)
    status_details = {
        key: value
        for key, value in summary.items()
        if key not in {"schema_version", "state"}
    }
    _write_status(output_root, "complete", **status_details)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = _arguments()
    output_root = _resolve(args.output_root)
    try:
        if args.action == "prepare":
            prepare(_resolve(args.source_root), output_root)
        elif args.action == "run-task":
            run_task(output_root, args.task_index)
        elif args.action == "submit-next-wave":
            submit_next_wave(output_root, args.start_index)
        elif args.action == "summarize":
            summarize(output_root)
        else:
            raise AssertionError(f"Unhandled action {args.action!r}.")
    except BaseException:
        if output_root.is_dir() and args.action == "summarize":
            _write_status(output_root, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
