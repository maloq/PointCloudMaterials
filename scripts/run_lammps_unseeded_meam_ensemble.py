#!/usr/bin/env python3
"""Run a resumable sequential ensemble from a validated unseeded MEAM liquid."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_lammps_homogeneous_campaign import _sha256, _write_json_atomic
from scripts.run_lammps_unseeded_meam_crystallization import (
    EXPECTED_ATOM_COUNT,
    Settings,
    _production_input,
    _run_lammps,
    _write_status,
    analyze,
)


SCHEMA_VERSION = 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run sequential unseeded 2NN-MEAM replicas from one "
            "validated liquid configuration."
        )
    )
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wait-for-status", type=Path)
    parser.add_argument(
        "--velocity-seeds",
        type=int,
        nargs="+",
        default=(35831, 35839, 35851, 35863, 35869, 35879, 35897),
    )
    parser.add_argument("--target-temperature-K", type=float, default=500.0)
    parser.add_argument("--measurement-steps", type=int, default=200000)
    parser.add_argument("--timestep-fs", type=float, default=3.0)
    parser.add_argument("--equilibration-steps", type=int, default=5000)
    parser.add_argument("--sample-interval-steps", type=int, default=1000)
    parser.add_argument("--mpi-ranks", type=int, default=48)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _template_settings(
    source_root: Path,
    output_root: Path,
    velocity_seed: int,
    measurement_steps: int,
    target_temperature_K: float,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
    mpi_ranks: int = 48,
) -> Settings:
    return Settings(
        output_root=output_root,
        library_potential=source_root / "potential" / "Lee2003_Al.library.meam",
        parameter_potential=source_root / "potential" / "Lee2003_Al.meam",
        velocity_seed=velocity_seed,
        preparation_seed=24681357,
        target_temperature_K=target_temperature_K,
        melt_temperature_K=1325.0,
        pressure_GPa=0.0,
        timestep_fs=timestep_fs,
        thermostat_time_fs=300.0,
        barostat_time_fs=3000.0,
        lattice_constant_A=4.05,
        repetitions=26,
        melt_steps=100000,
        equilibration_steps=equilibration_steps,
        measurement_steps=measurement_steps,
        sample_interval=sample_interval_steps,
        mpi_ranks=mpi_ranks,
    )


def _validate_controls(
    seeds: tuple[int, ...],
    measurement_steps: int,
    target_temperature_K: float,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
    mpi_ranks: int = 48,
) -> None:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(
            f"A non-empty set of unique velocity seeds is required, got {seeds}."
        )
    if any(seed <= 0 for seed in seeds):
        raise ValueError(f"Velocity seeds must be positive, got {seeds}.")
    if timestep_fs <= 0.0:
        raise ValueError(f"timestep_fs must be positive, got {timestep_fs}.")
    if measurement_steps <= 0 or equilibration_steps <= 0:
        raise ValueError(
            "measurement_steps and equilibration_steps must be positive, got "
            f"{measurement_steps} and {equilibration_steps}."
        )
    measurement_duration_ps = measurement_steps * timestep_fs / 1000.0
    if not math.isclose(measurement_duration_ps, 600.0, abs_tol=1.0e-12):
        raise ValueError(
            "This campaign requires exactly 600 ps of measurement, got "
            f"{measurement_duration_ps} ps from measurement_steps={measurement_steps} "
            f"and timestep_fs={timestep_fs}."
        )
    equilibration_duration_ps = equilibration_steps * timestep_fs / 1000.0
    if not math.isclose(equilibration_duration_ps, 15.0, abs_tol=1.0e-12):
        raise ValueError(
            "This campaign requires exactly 15 ps of equilibration, got "
            f"{equilibration_duration_ps} ps."
        )
    if sample_interval_steps <= 0:
        raise ValueError(
            f"sample_interval_steps must be positive, got {sample_interval_steps}."
        )
    for name, steps in (
        ("measurement_steps", measurement_steps),
        ("equilibration_steps", equilibration_steps),
    ):
        if steps % sample_interval_steps != 0:
            raise ValueError(
                f"{name}={steps} must be divisible by sample_interval_steps="
                f"{sample_interval_steps}."
            )
    if target_temperature_K <= 0.0:
        raise ValueError(
            f"target_temperature_K must be positive, got {target_temperature_K}."
        )
    if mpi_ranks <= 0:
        raise ValueError(f"mpi_ranks must be positive, got {mpi_ranks}.")


def prepare(
    source_root: Path,
    output_root: Path,
    seeds: tuple[int, ...],
    measurement_steps: int,
    target_temperature_K: float,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
    mpi_ranks: int = 48,
) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Ensemble output root exists and will not be overwritten: {output_root}."
        )
    required = (
        source_root / "manifest.json",
        source_root / "source_validation.json",
        source_root / "prepared_liquid.lammps.data",
        source_root / "potential" / "Lee2003_Al.library.meam",
        source_root / "potential" / "Lee2003_Al.meam",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Validated source campaign is incomplete; missing {missing}."
        )
    with (source_root / "manifest.json").open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    with (source_root / "source_validation.json").open(encoding="utf-8") as handle:
        source_validation = json.load(handle)
    if source_manifest["atom_count"] != EXPECTED_ATOM_COUNT:
        raise RuntimeError(
            f"Source atom_count={source_manifest['atom_count']}, expected "
            f"{EXPECTED_ATOM_COUNT}."
        )
    if source_manifest["crystal_seed"] is not None:
        raise RuntimeError(
            f"Source manifest declares crystal_seed={source_manifest['crystal_seed']}."
        )
    if source_validation["crystalline_fraction"] != 0.0:
        raise RuntimeError(
            "Source must contain zero PTM-crystalline atoms, got crystalline_fraction="
            f"{source_validation['crystalline_fraction']}."
        )
    if source_validation["largest_crystalline_cluster_atoms"] != 0:
        raise RuntimeError(
            "Source must contain no crystalline cluster, got largest cluster="
            f"{source_validation['largest_crystalline_cluster_atoms']}."
        )

    output_root.mkdir(parents=True)
    (output_root / "potential").mkdir()
    (output_root / "replicas").mkdir()
    shutil.copy2(
        source_root / "prepared_liquid.lammps.data",
        output_root / "prepared_liquid.lammps.data",
    )
    shutil.copy2(
        source_root / "source_validation.json", output_root / "source_validation.json"
    )
    for name in ("Lee2003_Al.library.meam", "Lee2003_Al.meam"):
        shutil.copy2(source_root / "potential" / name, output_root / "potential" / name)
    _write_json_atomic(
        output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                f"{len(seeds)}-replica qualitative spontaneous homogeneous "
                "crystallization ensemble"
            ),
            "atom_count": EXPECTED_ATOM_COUNT,
            "crystal_seed": None,
            "shared_liquid_source": {
                "source_campaign": str(source_root),
                "prepared_liquid_sha256": _sha256(
                    source_root / "prepared_liquid.lammps.data"
                ),
                "source_validation_sha256": _sha256(
                    source_root / "source_validation.json"
                ),
                "coordinate_configuration_shared_across_replicas": True,
                "variation": "independent Maxwell-Boltzmann momenta only",
            },
            "potential": source_manifest["potential"],
            "protocol": {
                "velocity_seeds": list(seeds),
                "temperature_K": target_temperature_K,
                "pressure_GPa": 0.0,
                "timestep_fs": timestep_fs,
                "equilibration_steps": equilibration_steps,
                "equilibration_duration_ps": equilibration_steps
                * timestep_fs
                / 1000.0,
                "measurement_steps": measurement_steps,
                "measurement_duration_ps": measurement_steps
                * timestep_fs
                / 1000.0,
                "sample_interval_steps": sample_interval_steps,
                "sample_interval_ps": sample_interval_steps
                * timestep_fs
                / 1000.0,
                "per_atom_frame_fields": {
                    "positions_A": "angstrom",
                    "velocities_A_per_ps": "angstrom/picosecond",
                },
            },
            "execution": {
                "strictly_sequential_replicas": True,
                "mpi_ranks_per_replica": mpi_ranks,
                "physical_cores_per_replica": mpi_ranks,
            },
        },
    )
    template = _template_settings(
        output_root,
        output_root,
        seeds[0],
        measurement_steps,
        target_temperature_K,
        timestep_fs,
        equilibration_steps,
        sample_interval_steps,
        mpi_ranks,
    )
    _write_status(
        template,
        "prepared",
        completed_replicas=[],
        pending_velocity_seeds=list(seeds),
    )


def _wait_for_prior_campaign(settings: Settings, status_path: Path | None) -> None:
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
        if state == "failed":
            raise RuntimeError(
                f"Prior campaign failed; refusing to compete for its cores: "
                f"{status_path}; status={document}."
            )
        now = time.monotonic()
        if now - last_reported_at >= 600.0 or last_reported_at == 0.0:
            print(
                f"Waiting for prior campaign before starting the replica "
                f"queue; current state={state}",
                flush=True,
            )
            last_reported_at = now
        _write_status(
            settings,
            "waiting_for_prior_campaign",
            prior_status=str(status_path),
            prior_state=state,
        )
        time.sleep(30.0)


def _completed_elapsed_seconds(settings: Settings, replica_dir: Path) -> float:
    elapsed_path = replica_dir / "md_elapsed_seconds.json"
    if elapsed_path.is_file():
        with elapsed_path.open(encoding="utf-8") as handle:
            elapsed = float(json.load(handle)["elapsed_seconds"])
        if elapsed <= 0.0:
            raise RuntimeError(f"{elapsed_path}: elapsed_seconds must be positive.")
        return elapsed
    log_path = replica_dir / "lammps.log"
    matches = re.findall(
        r"^Loop time of ([0-9.eE+-]+) on .* for ([0-9]+) steps ",
        log_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    completed_steps = sum(int(steps) for _, steps in matches)
    if completed_steps != settings.total_production_steps:
        raise RuntimeError(
            f"{log_path}: completed loop steps total {completed_steps}, expected "
            f"{settings.total_production_steps}."
        )
    elapsed = sum(float(seconds) for seconds, _ in matches)
    _write_json_atomic(elapsed_path, {"elapsed_seconds": elapsed})
    return elapsed


def _write_summary(
    output_root: Path,
    seeds: tuple[int, ...],
    completed: list[dict[str, object]],
    elapsed_seconds: dict[str, float],
    mpi_ranks: int = 48,
) -> None:
    _write_json_atomic(
        output_root / "campaign_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "strictly_sequential_replicas": True,
            "physical_cores_per_replica": mpi_ranks,
            "requested_velocity_seeds": list(seeds),
            "completed_replica_count": len(completed),
            "md_elapsed_seconds": elapsed_seconds,
            "replicas": completed,
        },
    )


def run(
    output_root: Path,
    seeds: tuple[int, ...],
    measurement_steps: int,
    wait_for_status: Path | None,
    target_temperature_K: float,
    timestep_fs: float,
    equilibration_steps: int,
    sample_interval_steps: int,
    mpi_ranks: int = 48,
) -> None:
    required = (
        output_root / "manifest.json",
        output_root / "source_validation.json",
        output_root / "prepared_liquid.lammps.data",
        output_root / "potential" / "Lee2003_Al.library.meam",
        output_root / "potential" / "Lee2003_Al.meam",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared ensemble is incomplete; missing {missing}.")
    template = _template_settings(
        output_root,
        output_root,
        seeds[0],
        measurement_steps,
        target_temperature_K,
        timestep_fs,
        equilibration_steps,
        sample_interval_steps,
        mpi_ranks,
    )
    _wait_for_prior_campaign(template, wait_for_status)

    completed: list[dict[str, object]] = []
    elapsed_seconds: dict[str, float] = {}
    for index, seed in enumerate(seeds):
        settings = replace(template, velocity_seed=seed)
        replica_name = f"replica_{index:03d}_velocity_{seed}"
        replica_dir = output_root / "replicas" / replica_name
        analysis_path = replica_dir / "analysis.json"
        if analysis_path.is_file():
            with analysis_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if document.get("velocity_random_seed") != seed:
                raise RuntimeError(
                    f"{analysis_path}: velocity seed does not match {seed}."
                )
            elapsed_seconds[replica_name] = _completed_elapsed_seconds(
                settings, replica_dir
            )
            completed.append(document)
            print(f"{replica_name}: already complete; preserving artifacts", flush=True)
            continue

        if replica_dir.exists():
            elapsed_seconds[replica_name] = _completed_elapsed_seconds(
                settings, replica_dir
            )
            print(
                f"{replica_name}: completed MD found; resuming analysis", flush=True
            )
        else:
            replica_dir.mkdir()
            (replica_dir / "in.lammps").write_text(
                _production_input(settings), encoding="utf-8"
            )
            _write_status(
                settings,
                "running_replica",
                active_replica=replica_name,
                replica_index=index,
                completed_replicas=[item["replica_name"] for item in completed],
                pending_velocity_seeds=list(seeds[index:]),
            )
            print(
                f"{replica_name}: starting 600 ps unseeded run on all "
                f"{mpi_ranks} physical cores",
                flush=True,
            )
            elapsed_seconds[replica_name] = _run_lammps(
                settings, replica_dir, "in.lammps"
            )
            _write_json_atomic(
                replica_dir / "md_elapsed_seconds.json",
                {"elapsed_seconds": elapsed_seconds[replica_name]},
            )
        _write_status(
            settings,
            "analyzing_replica",
            active_replica=replica_name,
            completed_replicas=[item["replica_name"] for item in completed],
        )
        document = analyze(settings, replica_dir)
        document["replica_name"] = replica_name
        _write_json_atomic(analysis_path, document)
        completed.append(document)
        _write_summary(output_root, seeds, completed, elapsed_seconds, mpi_ranks)
        print(f"{replica_name}: MD, analysis, and plots complete", flush=True)

    _write_summary(output_root, seeds, completed, elapsed_seconds, mpi_ranks)
    _write_status(
        template,
        "complete",
        completed_replicas=[item["replica_name"] for item in completed],
        campaign_summary="campaign_summary.json",
    )


def main() -> None:
    args = _arguments()
    source_root = _resolve(args.source_root)
    output_root = _resolve(args.output_root)
    wait_for_status = (
        _resolve(args.wait_for_status) if args.wait_for_status is not None else None
    )
    seeds = tuple(args.velocity_seeds)
    _validate_controls(
        seeds,
        args.measurement_steps,
        args.target_temperature_K,
        args.timestep_fs,
        args.equilibration_steps,
        args.sample_interval_steps,
        args.mpi_ranks,
    )
    try:
        if args.action == "prepare":
            prepare(
                source_root,
                output_root,
                seeds,
                args.measurement_steps,
                args.target_temperature_K,
                args.timestep_fs,
                args.equilibration_steps,
                args.sample_interval_steps,
                args.mpi_ranks,
            )
        else:
            run(
                output_root,
                seeds,
                args.measurement_steps,
                wait_for_status,
                args.target_temperature_K,
                args.timestep_fs,
                args.equilibration_steps,
                args.sample_interval_steps,
                args.mpi_ranks,
            )
    except BaseException:
        if output_root.is_dir():
            settings = _template_settings(
                output_root,
                output_root,
                seeds[0],
                args.measurement_steps,
                args.target_temperature_K,
                args.timestep_fs,
                args.equilibration_steps,
                args.sample_interval_steps,
                args.mpi_ranks,
            )
            _write_status(settings, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
