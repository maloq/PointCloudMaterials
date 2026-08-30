#!/usr/bin/env python3
"""Run multiple unseeded 2NN-MEAM temperature ensembles sequentially."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_lammps_homogeneous_campaign import _sha256, _write_json_atomic
from scripts.run_lammps_unseeded_meam_ensemble import (
    _validate_controls,
    prepare as prepare_ensemble,
    run as run_ensemble,
)


SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential all-core homogeneous-crystallization temperature sweep."
    )
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wait-for-status", type=Path)
    parser.add_argument(
        "--temperatures-K", type=float, nargs="+", default=(400.0, 450.0, 550.0, 600.0)
    )
    parser.add_argument(
        "--velocity-seeds",
        type=int,
        nargs="+",
        default=(35831, 35839, 35851, 35863, 35869, 35879),
    )
    parser.add_argument("--measurement-steps", type=int, default=200000)
    parser.add_argument("--timestep-fs", type=float, default=3.0)
    parser.add_argument("--equilibration-steps", type=int, default=5000)
    parser.add_argument("--sample-interval-steps", type=int, default=1000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _temperature_name(temperature_K: float) -> str:
    if not temperature_K.is_integer():
        raise ValueError(
            f"Temperature directory labels require integer kelvin, got {temperature_K}."
        )
    return f"temperature_{int(temperature_K)}K"


def _validate_campaign(
    temperatures_K: tuple[float, ...],
    seeds: tuple[int, ...],
    measurement_steps: int,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
) -> None:
    if not temperatures_K or len(set(temperatures_K)) != len(temperatures_K):
        raise ValueError(
            f"A non-empty sequence of unique temperatures is required, got "
            f"{temperatures_K}."
        )
    if any(value <= 0.0 for value in temperatures_K):
        raise ValueError(
            f"Temperatures must be positive, got {temperatures_K}."
        )
    if len(seeds) != 6:
        raise ValueError(f"Exactly six velocity seeds are required, got {seeds}.")
    for temperature_K in temperatures_K:
        _validate_controls(
            seeds,
            measurement_steps,
            temperature_K,
            timestep_fs,
            equilibration_steps,
            sample_interval_steps,
        )
        _temperature_name(temperature_K)


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


def _wait_for_prior_campaign(
    output_root: Path, status_path: Path | None
) -> None:
    if status_path is None:
        return
    if not status_path.is_file():
        raise FileNotFoundError(f"Wait status file is absent: {status_path}.")
    last_reported_at = 0.0
    while True:
        with status_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        state = document["state"]
        if state == "complete":
            print(f"Prior campaign is complete: {status_path}", flush=True)
            return
        if state in {"failed", "superseded"}:
            raise RuntimeError(
                f"Prior campaign ended with state={state}; refusing to start the "
                f"queued temperature campaign: {status_path}; status={document}."
            )
        now = time.monotonic()
        if now - last_reported_at >= 600.0 or last_reported_at == 0.0:
            print(
                f"Waiting for prior temperature campaign; current state={state}",
                flush=True,
            )
            last_reported_at = now
        _write_status(
            output_root,
            "waiting_for_prior_campaign",
            prior_status=str(status_path),
            prior_state=state,
        )
        time.sleep(30.0)


def prepare(
    source_root: Path,
    output_root: Path,
    temperatures_K: tuple[float, ...],
    seeds: tuple[int, ...],
    measurement_steps: int,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Temperature campaign output exists and will not be overwritten: "
            f"{output_root}."
        )
    output_root.mkdir(parents=True)
    _write_json_atomic(
        output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "paired-temperature comparison of spontaneous homogeneous "
                "crystallization in unseeded 70,304-atom Al"
            ),
            "source_campaign": str(source_root),
            "source_validation_sha256": _sha256(
                source_root / "source_validation.json"
            ),
            "temperatures_K": list(temperatures_K),
            "velocity_seeds": list(seeds),
            "paired_velocity_seed_design": True,
            "replicas_per_temperature": len(seeds),
            "total_replica_count": len(temperatures_K) * len(seeds),
            "timestep_fs": timestep_fs,
            "measurement_duration_ps": measurement_steps
            * timestep_fs
            / 1000.0,
            "coordinate_sample_interval_ps": sample_interval_steps
            * timestep_fs
            / 1000.0,
            "per_atom_frame_fields": {
                "positions_A": "angstrom",
                "velocities_A_per_ps": "angstrom/picosecond",
            },
            "execution": {
                "temperature_order": list(temperatures_K),
                "strictly_sequential_all_replicas": True,
                "physical_cores_per_replica": 48,
            },
        },
    )
    for temperature_K in temperatures_K:
        prepare_ensemble(
            source_root,
            output_root / _temperature_name(temperature_K),
            seeds,
            measurement_steps,
            temperature_K,
            timestep_fs,
            equilibration_steps,
            sample_interval_steps,
        )
    _write_status(
        output_root,
        "prepared",
        completed_temperatures_K=[],
        pending_temperatures_K=list(temperatures_K),
    )


def _completed_temperature_summary(temperature_root: Path) -> dict[str, object]:
    with (temperature_root / "campaign_summary.json").open(
        encoding="utf-8"
    ) as handle:
        document = json.load(handle)
    return {
        "temperature_K": int(temperature_root.name.removeprefix("temperature_").removesuffix("K")),
        "completed_replica_count": document["completed_replica_count"],
        "nucleation_observed_count": sum(
            bool(replica["nucleation_observed"]) for replica in document["replicas"]
        ),
        "campaign_summary": str(
            Path(temperature_root.name) / "campaign_summary.json"
        ),
    }


def run(
    output_root: Path,
    temperatures_K: tuple[float, ...],
    seeds: tuple[int, ...],
    measurement_steps: int,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
    wait_for_status: Path | None,
) -> None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Temperature campaign is not prepared; missing {manifest_path}."
        )
    _wait_for_prior_campaign(output_root, wait_for_status)
    completed_temperatures: list[float] = []
    temperature_summaries: list[dict[str, object]] = []
    for temperature_K in temperatures_K:
        temperature_root = output_root / _temperature_name(temperature_K)
        _write_status(
            output_root,
            "running_temperature",
            active_temperature_K=temperature_K,
            completed_temperatures_K=completed_temperatures,
            pending_temperatures_K=[
                value for value in temperatures_K if value not in completed_temperatures
            ],
            active_temperature_status=str(temperature_root / "status.json"),
        )
        print(
            f"temperature {temperature_K:.0f} K: starting six sequential replicas",
            flush=True,
        )
        run_ensemble(
            temperature_root,
            seeds,
            measurement_steps,
            None,
            temperature_K,
            timestep_fs,
            equilibration_steps,
            sample_interval_steps,
        )
        completed_temperatures.append(temperature_K)
        temperature_summaries.append(
            _completed_temperature_summary(temperature_root)
        )
        _write_json_atomic(
            output_root / "campaign_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "completed_temperature_count": len(completed_temperatures),
                "completed_replica_count": len(completed_temperatures) * len(seeds),
                "temperature_summaries": temperature_summaries,
            },
        )
    _write_status(
        output_root,
        "complete",
        completed_temperatures_K=completed_temperatures,
        campaign_summary="campaign_summary.json",
    )


def main() -> None:
    args = _arguments()
    source_root = _resolve(args.source_root)
    output_root = _resolve(args.output_root)
    wait_for_status = (
        _resolve(args.wait_for_status) if args.wait_for_status is not None else None
    )
    temperatures_K = tuple(args.temperatures_K)
    seeds = tuple(args.velocity_seeds)
    _validate_campaign(
        temperatures_K,
        seeds,
        args.measurement_steps,
        args.timestep_fs,
        args.equilibration_steps,
        args.sample_interval_steps,
    )
    try:
        if args.action == "prepare":
            prepare(
                source_root,
                output_root,
                temperatures_K,
                seeds,
                args.measurement_steps,
                args.timestep_fs,
                args.equilibration_steps,
                args.sample_interval_steps,
            )
        else:
            run(
                output_root,
                temperatures_K,
                seeds,
                args.measurement_steps,
                args.timestep_fs,
                args.equilibration_steps,
                args.sample_interval_steps,
                wait_for_status,
            )
    except BaseException:
        if output_root.is_dir():
            _write_status(output_root, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
