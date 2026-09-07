#!/usr/bin/env python3
"""Run the fixed-15 ps predictive-dynamics campaign and exact continuations.

This parameter-locked specialization reuses the validated fixed-horizon
producer without changing any completed 48 ps campaign.  New branches use
``fix temp/csld`` because LAMMPS stores that fix's random-number state in a
binary restart; the older ``fix langevin`` protocol cannot restart exactly.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_masses, atomic_numbers


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.simulation.campaigns import predictive_dynamics as campaign  # noqa: E402
from src.data_utils.shooting_binary import (  # noqa: E402
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    compose_shooting_binary_trajectories,
    convert_shooting_trajectory,
)
from src.data_utils.shooting_dataset import validate_complete_shooting_branch  # noqa: E402
from src.data_utils.synthetic.atomistic.lammps_shooting import (  # noqa: E402
    _lammps_command,
    _lammps_environment,
)
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset  # noqa: E402


DESIGN_NAME = "predictive_dynamics_fixed15_float32_exact_csld"
CAMPAIGN_SEED = 20_260_904
TIMESTEP_FS = 3.0
DURATION_PS = 15.0
PREDICTION_DURATION_PS = 12.0
AUGMENTATION_BUFFER_PS = 3.0
RUN_STEPS = 5_000
SAMPLE_INTERVAL_STEPS = 100
EXPECTED_TIMESTEPS = tuple(range(0, RUN_STEPS + 1, SAMPLE_INTERVAL_STEPS))
EXPECTED_FRAME_COUNT = len(EXPECTED_TIMESTEPS)
THERMOSTAT_TIME_FS = 300.0
EXTENSION_STOP_PS = 24.0
EXTENSION_STOP_STEP = 8_000
EXTENSION_TIMESTEPS = tuple(range(RUN_STEPS + SAMPLE_INTERVAL_STEPS, EXTENSION_STOP_STEP + 1, SAMPLE_INTERVAL_STEPS))
TOPUP_ROOT = Path(
    "/home/ids/vmorozov/simulations/"
    "al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904"
)
SMOKE_ROOT = Path(
    "/home/ids/vmorozov/simulations/"
    "al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904"
)
SNAPSHOT = campaign.DEFAULT_SNAPSHOT


def _protocol() -> dict[str, Any]:
    return {
        "ensemble": "fixed-cell stochastic canonical NVT",
        "thermostat_style": "temp/csld",
        "timestep_fs": TIMESTEP_FS,
        "duration_ps": DURATION_PS,
        "prediction_duration_ps": PREDICTION_DURATION_PS,
        "augmentation_buffer_ps": AUGMENTATION_BUFFER_PS,
        "run_steps": RUN_STEPS,
        "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
        "sample_interval_ps": SAMPLE_INTERVAL_STEPS * TIMESTEP_FS / 1000.0,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "thermostat_time_fs": THERMOSTAT_TIME_FS,
        "dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
        "independent_momenta_per_branch": False,
        "storage_dtype": "float32",
        "transient_text_dump_deleted_after_verified_conversion": True,
        "mandatory_fixed_horizon_after_basin_arrival": True,
        "continuation_restart": {
            "state": "required",
            "exact_rng_state": True,
            "same_mpi_rank_count_required": True,
            "mpi_ranks": 24,
            "first_extension_stop_ps": EXTENSION_STOP_PS,
            "first_extension_stop_step": EXTENSION_STOP_STEP,
        },
    }


def _execution(wave_size: int) -> dict[str, Any]:
    if wave_size <= 0 or wave_size > 6:
        raise ValueError(f"wave_size must be within [1, 6], got {wave_size}.")
    return {
        "mpi_ranks_per_branch": 24,
        "launcher": "srun_pmi2",
        "partition": "CPU",
        "time_limit": "01:00:00",
        "memory": "24G",
        "array_concurrency": wave_size,
        "wave_size": wave_size,
        "normal_qos_max_submitted_jobs": 10,
    }


def _scientific_contract() -> dict[str, Any]:
    return {
        "exact_source_restart": False,
        "exact_short_to_extension_restart": True,
        "exact_extension_requires_same_mpi_ranks": 24,
        "interpretation": (
            "Fixed-cell stochastic canonical futures conditioned on immutable positions, "
            "cell, history, shooting temperature, sampled momentum, and thermostat stream."
        ),
        "no_equilibration_after_branching": True,
        "first_passage_does_not_stop_fixed_trajectory": True,
        "bootstrap_unit": "root_source_lineage",
        "overlapping_windows_are_correlated": True,
    }


def render_lammps_input(
    *,
    parent_id: str,
    branch_id: str,
    temperature_K: float,
    velocity_seed: int,
    thermostat_seed: int,
    timestep_fs: float,
    thermostat_time_fs: float,
    sample_interval_steps: int,
    run_steps: int,
) -> str:
    if run_steps != RUN_STEPS or timestep_fs != TIMESTEP_FS:
        raise ValueError(
            f"The 15 ps producer requires run_steps={RUN_STEPS} and "
            f"timestep_fs={TIMESTEP_FS}, got {run_steps} and {timestep_fs}."
        )
    mass = atomic_masses[atomic_numbers["Al"]]
    return f"""# Fixed-15 ps position-conditioned branch generated by PointCloudMaterials.
# temp/csld is used because its RNG state is stored for exact same-rank continuation.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../parents/{parent_id}/parent.lammps.data

mass 1 {mass:.12g}
pair_style meam
pair_coeff * * ../../potential/Lee2003_Al.library.meam Al ../../potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {timestep_fs / 1000.0:.12g}
velocity all create {temperature_K:.12g} {velocity_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all temp/csld {temperature_K:.12g} {temperature_K:.12g} {thermostat_time_fs / 1000.0:.12g} {thermostat_seed}

thermo {sample_interval_steps}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {sample_interval_steps} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g %.9g %.9g %.9g\"

print \"SHOOTING_15PS_BEGIN {branch_id} PARENT {parent_id}\"
run {run_steps}
write_restart final.restart.bin
print \"SHOOTING_15PS_COMPLETE {branch_id} PARENT {parent_id}\"
"""


def _configure() -> None:
    campaign.DESIGN_NAME = DESIGN_NAME
    campaign.CAMPAIGN_SEED = CAMPAIGN_SEED
    campaign.TIMESTEP_FS = TIMESTEP_FS
    campaign.DURATION_PS = DURATION_PS
    campaign.RUN_STEPS = RUN_STEPS
    campaign.SAMPLE_INTERVAL_STEPS = SAMPLE_INTERVAL_STEPS
    campaign.EXPECTED_TIMESTEPS = EXPECTED_TIMESTEPS
    campaign.EXPECTED_FRAME_COUNT = EXPECTED_FRAME_COUNT
    campaign.THERMOSTAT_TIME_FS = THERMOSTAT_TIME_FS
    campaign.DEFAULT_TOPUP_ROOT = TOPUP_ROOT
    campaign.DEFAULT_SMOKE_ROOT = SMOKE_ROOT
    campaign._protocol = _protocol
    campaign._execution = _execution
    campaign._scientific_contract = _scientific_contract
    campaign.render_lammps_input = render_lammps_input


def _rewrite_generated_slurm_scripts(root: Path, *, job_prefix: str) -> None:
    old_runner = Path(campaign.__file__).resolve()
    new_runner = Path(__file__).resolve()
    for path in (
        root / "slurm/run_branch.sbatch",
        root / "slurm/submit_wave.sbatch",
        root / "slurm/summarize.sbatch",
    ):
        text = path.read_text(encoding="utf-8")
        if str(old_runner) in text:
            text = text.replace(str(old_runner), str(new_runner))
        elif str(new_runner) not in text:
            raise RuntimeError(
                f"Generated Slurm script references neither the base nor 15 ps runner: {path}."
            )
        lines = [
            f"#SBATCH --job-name={job_prefix}" if line.startswith("#SBATCH --job-name=") else line
            for line in text.splitlines()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    continuation = root / "slurm/run_continuation_smoke.sbatch"
    continuation.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={job_prefix}_cont
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output={root}/slurm/continuation_%j.out
#SBATCH --error={root}/slurm/continuation_%j.err

set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
python {Path(__file__).resolve()} continuation-smoke --campaign-root {root} --branch-index 0
""",
        encoding="utf-8",
    )
    continuation.chmod(0o750)
    sequential = root / "slurm/run_sequential_subset.sbatch"
    sequential.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={job_prefix}_seq
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output={root}/slurm/sequential_%j.out
#SBATCH --error={root}/slurm/sequential_%j.err

set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
: "${{PREDICTIVE_START:?PREDICTIVE_START must be set}}"
: "${{PREDICTIVE_STOP:?PREDICTIVE_STOP must be set}}"
python {Path(__file__).resolve()} run-slurm-subset --campaign-root {root} --start-index "${{PREDICTIVE_START}}" --stop-index "${{PREDICTIVE_STOP}}" --summarize-after
""",
        encoding="utf-8",
    )
    sequential.chmod(0o750)


def _finalize_prepared_manifest(root: Path, manifest: dict[str, Any], *, job_prefix: str) -> None:
    protocol = _protocol()
    protocol["independent_momenta_per_branch"] = (
        manifest["design_kind"] == "legacy_40_parent_topup_from_12_to_16"
    )
    manifest["protocol"] = protocol
    manifest["scientific_contract"] = _scientific_contract()
    manifest["window_index"] = {
        "primary_horizons_ps": [3.0, 6.0, 12.0],
        "auxiliary_horizons_ps": [1.2, 9.0],
        "latest_start_ps": 3.0,
        "stride_ps": 0.6,
        "loader_ablation_stride_ps": 1.2,
        "maximum_prediction_horizon_ps": 12.0,
        "include_authoritative_existing_snapshot_branches": False,
    }
    manifest["extension_contract"] = {
        "short_branch_completion_independent_of_extension": True,
        "short_index": "short_15ps_branches.json",
        "extension_index": "extended_24ps_branches.json",
        "event_extension_index": "extended_event_branches.json",
        "thermostat_style": "temp/csld",
        "same_mpi_ranks_required": 24,
        "extension_output_timesteps": list(EXTENSION_TIMESTEPS),
        "composed_output_timesteps": list(range(0, EXTENSION_STOP_STEP + 1, SAMPLE_INTERVAL_STEPS)),
    }
    campaign._write_json_atomic(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(
        campaign._sha256_file(root / "manifest.json") + "  manifest.json\n", encoding="ascii"
    )
    _rewrite_generated_slurm_scripts(root, job_prefix=job_prefix)


def prepare_topup(snapshot: Path, root: Path, *, wave_size: int) -> dict[str, Any]:
    _configure()
    manifest = campaign.prepare_topup(snapshot, root, wave_size=wave_size)
    _finalize_prepared_manifest(root.resolve(), manifest, job_prefix="al15_topup")
    return manifest


def prepare_smoke(snapshot: Path, root: Path, *, wave_size: int) -> dict[str, Any]:
    _configure()
    manifest = campaign.prepare_smoke(snapshot, root, wave_size=wave_size)
    _finalize_prepared_manifest(root.resolve(), manifest, job_prefix="al15_smoke")
    return manifest


def _write_short_index(root: Path) -> dict[str, Any]:
    manifest = campaign._load_json(root / "manifest.json")
    records: list[dict[str, Any]] = []
    for branch in manifest["branches"]:
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        outcome = campaign._load_json(outcome_path)
        validate_complete_shooting_branch(root, manifest, branch, outcome)
        records.append(
            {
                "branch_index": int(branch["branch_index"]),
                "branch_id": branch["branch_id"],
                "parent_id": branch["parent_id"],
                "root_source_lineage_id": branch["root_source_lineage_id"],
                "source_split": branch["source_split"],
                "temperature_K": branch["temperature_K"],
                "branch_dir": branch["branch_dir"],
                "outcome_sha256": campaign._sha256_file(outcome_path),
                "trajectory_manifest_sha256": campaign._sha256_file(
                    root / str(branch["branch_dir"]) / "trajectory_binary_float32/manifest.json"
                ),
                "restart_sha256": outcome["restart_sha256"],
            }
        )
    result = {
        "schema_version": 1,
        "state": "complete",
        "campaign_root": str(root),
        "branch_count": len(records),
        "duration_ps": DURATION_PS,
        "records": records,
    }
    campaign._write_json_atomic(root / "short_15ps_branches.json", result)
    for name in ("extended_24ps_branches.json", "extended_event_branches.json"):
        path = root / name
        if not path.exists():
            campaign._write_json_atomic(
                path,
                {
                    "schema_version": 1,
                    "state": "complete",
                    "campaign_root": str(root),
                    "branch_count": 0,
                    "records": [],
                },
            )
    return result


def summarize(root: Path) -> dict[str, Any]:
    _configure()
    summary = campaign.summarize_campaign(root)
    short_index = _write_short_index(root.resolve())
    summary["short_15ps_index"] = "short_15ps_branches.json"
    summary["short_15ps_branch_count"] = short_index["branch_count"]
    summary["extended_24ps_index"] = "extended_24ps_branches.json"
    summary["extended_event_index"] = "extended_event_branches.json"
    campaign._write_json_atomic(root.resolve() / "summary.json", summary)
    campaign._write_json_atomic(root.resolve() / "status.json", summary)
    return summary


def submit_next_wave(root: Path, start_index: int) -> dict[str, Any]:
    root = root.expanduser().resolve()
    sequential_path = root / "slurm/sequential_submission.json"
    if sequential_path.is_file():
        sequential = campaign._load_json(sequential_path)
        job_id = str(sequential["job_id"])
        queued = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%A"],
            check=True,
            text=True,
            capture_output=True,
        )
        if queued.stdout.strip():
            raise RuntimeError(
                f"Refusing a duplicate wave while sequential job {job_id} is active: "
                f"{sequential_path}."
            )
    return campaign.submit_next_wave(root, start_index)


def run_slurm_subset(
    root: Path,
    start_index: int,
    stop_index: int,
    *,
    summarize_after: bool,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = _verify_manifest(root)
    if os.environ.get("SLURM_NTASKS") != "24":
        raise RuntimeError(
            "Sequential 15 ps execution requires one 24-rank Slurm allocation; "
            f"observed SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r}."
        )
    start = int(start_index)
    stop = int(stop_index)
    if start < 0 or stop < start or stop >= len(manifest["branches"]):
        raise IndexError(
            f"Requested inclusive branch range [{start}, {stop}] is outside "
            f"[0, {len(manifest['branches']) - 1}]."
        )
    lock_path = root / "slurm_sequential_execution.lock"
    record_path = root / "slurm_sequential_execution.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another sequential driver holds {lock_path}.") from error
        base = {
            "schema_version": 1,
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "mpi_ranks": 24,
            "start_index": start,
            "stop_index": stop,
            "started_at": campaign._utc_now(),
        }
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(base, sort_keys=True) + "\n")
        lock.flush()
        os.fsync(lock.fileno())
        campaign._write_json_atomic(record_path, {**base, "state": "running"})
        current = start
        try:
            for current in range(start, stop + 1):
                campaign.run_task(root, current)
                campaign._write_json_atomic(
                    record_path,
                    {
                        **base,
                        "state": "running",
                        "updated_at": campaign._utc_now(),
                        "last_completed_index": current,
                    },
                )
            result: dict[str, Any] = {
                **base,
                "state": "complete",
                "completed_at": campaign._utc_now(),
                "last_completed_index": stop,
            }
            campaign._write_json_atomic(record_path, result)
            if summarize_after:
                result["summary"] = summarize(root)
            return result
        except BaseException as error:
            failed = {
                **base,
                "state": "failed",
                "failed_at": campaign._utc_now(),
                "failed_branch_index": current,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
            }
            campaign._write_json_atomic(record_path, failed)
            raise


def _verify_manifest(root: Path) -> dict[str, Any]:
    manifest = campaign._load_json(root / "manifest.json")
    expected = (root / "manifest.sha256").read_text(encoding="ascii").split()[0]
    observed = campaign._sha256_file(root / "manifest.json")
    if observed != expected:
        raise RuntimeError(
            f"Immutable 15 ps manifest changed: expected={expected}, observed={observed}, "
            f"path={root / 'manifest.json'}."
        )
    protocol = manifest["protocol"]
    if (
        protocol.get("thermostat_style") != "temp/csld"
        or int(protocol["run_steps"]) != RUN_STEPS
        or int(protocol["expected_frame_count"]) != EXPECTED_FRAME_COUNT
        or int(manifest["execution"]["mpi_ranks_per_branch"]) != 24
    ):
        raise RuntimeError(f"Campaign does not satisfy the exact 15 ps contract: {root}.")
    return manifest


def _render_extension_input(branch: dict[str, Any]) -> str:
    temperature = float(branch["temperature_K"])
    seed = int(branch["thermostat_seed"])
    return f"""# Exact same-rank continuation from 15 to 24 ps.
log extension.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_restart ../final.restart.bin

pair_style meam
pair_coeff * * ../../../potential/Lee2003_Al.library.meam Al ../../../potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {TIMESTEP_FS / 1000.0:.12g}

fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all temp/csld {temperature:.12g} {temperature:.12g} {THERMOSTAT_TIME_FS / 1000.0:.12g} {seed}

thermo {SAMPLE_INTERVAL_STEPS}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {SAMPLE_INTERVAL_STEPS} extension_15_to_24ps.lammpstrj id type x y z vx vy vz
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g %.9g %.9g %.9g\"

print \"EXTENSION_BEGIN {branch['branch_id']} FROM_STEP {RUN_STEPS}\"
run {EXTENSION_STOP_STEP - RUN_STEPS}
write_restart final_24ps.restart.bin
print \"EXTENSION_COMPLETE {branch['branch_id']} AT_STEP {EXTENSION_STOP_STEP}\"
"""


def _update_extension_index(root: Path, record: dict[str, Any]) -> None:
    path = root / "extended_24ps_branches.json"
    lock_path = root / "extended_24ps_branches.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            document = campaign._load_json(path)
            records = list(document["records"])
        else:
            records = []
        records = [value for value in records if value["branch_id"] != record["branch_id"]]
        records.append(record)
        records.sort(key=lambda value: int(value["branch_index"]))
        campaign._write_json_atomic(
            path,
            {
                "schema_version": 1,
                "state": "complete",
                "campaign_root": str(root),
                "branch_count": len(records),
                "records": records,
            },
        )


def _require_exact_continuation(root: Path, manifest: dict[str, Any]) -> Path:
    """Require measured continuation acceptance before producing indexed extensions."""
    if manifest["design_kind"] == "single_parent_preproduction_smoke_8x2":
        smoke_root = root
    else:
        gate_path = root / "extension_gate.json"
        gate = campaign._load_json(gate_path)
        if gate["exact_15_to_24ps_extensions_allowed"] is not True:
            raise RuntimeError(f"Exact extensions are blocked by {gate_path}; run the continuation audit first.")
        smoke_root = Path(gate["smoke_campaign_root"])
    proof_path = smoke_root / "continuation_smoke_test.json"
    if not proof_path.is_file():
        raise RuntimeError(f"Exact continuation has no passing smoke comparison: {proof_path}.")
    proof = campaign._load_json(proof_path)
    if not (
        proof["state"] == "complete"
        and proof["positions_bitwise_equal_after_text_quantization"] is True
        and proof["velocities_bitwise_equal_after_text_quantization"] is True
        and proof["mpi_ranks"] == 24
        and proof["thermostat_style"] == "temp/csld"
        and proof["short_duration_ps"] == DURATION_PS
        and proof["extended_duration_ps"] == EXTENSION_STOP_PS
    ):
        raise RuntimeError(f"Exact continuation smoke did not satisfy the 15-to-24 ps contract: {proof_path}.")
    return proof_path


def extend_branch(
    root: Path,
    branch_index: int,
    *,
    selection_reason: str,
    selection_probability: float,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    proof = _require_exact_continuation(root, _verify_manifest(root))
    record = _run_extension(
        root, branch_index, selection_reason=selection_reason,
        selection_probability=selection_probability,
    )
    accepted = dict(record, classification="exact_continuation",
                    acceptance_proof=str(proof), acceptance_proof_sha256=campaign._sha256_file(proof))
    _update_extension_index(root, accepted)
    return accepted


def _run_extension(
    root: Path,
    branch_index: int,
    *,
    selection_reason: str,
    selection_probability: float,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = _verify_manifest(root)
    index = int(branch_index)
    if index < 0 or index >= len(manifest["branches"]):
        raise IndexError(f"branch_index={index} is outside [0, {len(manifest['branches'])}).")
    if os.environ.get("SLURM_NTASKS") != "24":
        raise RuntimeError(
            "Exact temp/csld continuation requires a 24-rank Slurm allocation; "
            f"observed SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r}."
        )
    probability = float(selection_probability)
    if not (0.0 < probability <= 1.0):
        raise ValueError(
            f"selection_probability must be within (0, 1], got {selection_probability}."
        )
    branch = manifest["branches"][index]
    branch_dir = root / str(branch["branch_dir"])
    short_outcome = campaign._load_json(branch_dir / "outcome.json")
    validate_complete_shooting_branch(root, manifest, branch, short_outcome)
    extension_dir = branch_dir / "extension_15_to_24ps"
    outcome_path = extension_dir / "outcome.json"
    if outcome_path.is_file():
        outcome = campaign._load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Existing extension outcome is not complete: {outcome_path}.")
        return outcome
    if extension_dir.exists():
        raise RuntimeError(
            f"Partial extension directory exists and must be archived explicitly: {extension_dir}."
        )
    extension_dir.mkdir()
    (extension_dir / "in.lammps").write_text(
        _render_extension_input(branch), encoding="utf-8"
    )
    campaign._write_json_atomic(
        extension_dir / "status.json",
        {
            "schema_version": 1,
            "state": "running",
            "started_at": campaign._utc_now(),
            "branch_id": branch["branch_id"],
            "selection_reason": selection_reason,
            "selection_probability": probability,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mpi_ranks": 24,
        },
    )
    command = _lammps_command(mpi_ranks=24, launcher="srun_pmi2")
    started = time.monotonic()
    stdout_path = extension_dir / "extension.stdout.log"
    try:
        with stdout_path.open("wb") as stdout:
            completed = subprocess.run(
                command,
                cwd=extension_dir,
                env=_lammps_environment(),
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"LAMMPS extension failed with return code {completed.returncode}; "
                f"inspect {stdout_path}."
            )
        source = extension_dir / "extension_15_to_24ps.lammpstrj"
        restart = extension_dir / "final_24ps.restart.bin"
        for artifact in (source, restart, extension_dir / "extension.lammps.log"):
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"Required extension artifact is absent or empty: {artifact}.")
        source_size = source.stat().st_size
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(source)
        raw_extension_timesteps = (RUN_STEPS, *EXTENSION_TIMESTEPS)
        if (
            scan.num_atoms != campaign.ATOM_COUNT
            or tuple(scan.timesteps.tolist()) != raw_extension_timesteps
            or tuple(scan.atom_columns) != tuple(manifest["protocol"]["dump_columns"])
        ):
            raise RuntimeError(
                f"Extension dump violates the 15→24 ps contract: atoms={scan.num_atoms}, "
                f"frames={scan.frame_count}, timesteps={scan.timesteps.tolist()}."
            )
        source_sha256 = campaign._sha256_file(source)
        extension_binary = convert_shooting_trajectory(
            source,
            extension_dir / "trajectory_binary_float32",
            timesteps=EXTENSION_TIMESTEPS,
            atom_count=campaign.ATOM_COUNT,
            storage_dtype="float32",
            provenance={
                "branch_id": branch["branch_id"],
                "segment": "15_to_24ps",
                "source_sha256": source_sha256,
            },
        )
        extension_binary.verify_checksums()
        short_binary = ShootingBinaryTrajectory.load(branch_dir / "trajectory_binary_float32")
        composed = compose_shooting_binary_trajectories(
            (short_binary, extension_binary),
            extension_dir / "composed_0_to_24ps_float32",
            timesteps=range(0, EXTENSION_STOP_STEP + 1, SAMPLE_INTERVAL_STEPS),
            storage_dtype="float32",
            provenance={
                "branch_id": branch["branch_id"],
                "short_outcome_sha256": campaign._sha256_file(branch_dir / "outcome.json"),
                "extension_source_sha256": source_sha256,
                "exact_same_rank_temp_csld_continuation": False,
                "classification": "unvalidated_restart_diagnostic",
            },
        )
        composed.verify_checksums()
        if source.stat().st_size != source_size or campaign._sha256_file(source) != source_sha256:
            raise RuntimeError(f"Extension text dump changed during verified conversion: {source}.")
        source.unlink()
        record = {
            "schema_version": 1,
            "state": "complete",
            "completed_at": campaign._utc_now(),
            "branch_index": index,
            "branch_id": branch["branch_id"],
            "selection_reason": selection_reason,
            "selection_probability": probability,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mpi_ranks": 24,
            "thermostat_style": "temp/csld",
            "elapsed_seconds": time.monotonic() - started,
            "first_timestep": EXTENSION_TIMESTEPS[0],
            "last_timestep": EXTENSION_TIMESTEPS[-1],
            "frame_count": len(EXTENSION_TIMESTEPS),
            "source_text_size_bytes": source_size,
            "source_text_sha256": source_sha256,
            "source_text_deleted": True,
            "extension_binary_path": "trajectory_binary_float32",
            "extension_binary_allocated_bytes": binary_directory_sizes(extension_binary.root)["allocated_bytes"],
            "composed_binary_path": "composed_0_to_24ps_float32",
            "composed_frame_count": composed.frame_count,
            "composed_last_timestep": int(composed.timesteps[-1]),
            "restart_path": "final_24ps.restart.bin",
            "restart_sha256": campaign._sha256_file(restart),
            "short_outcome_unchanged_sha256": campaign._sha256_file(branch_dir / "outcome.json"),
        }
        campaign._write_json_atomic(outcome_path, record)
        campaign._write_json_atomic(extension_dir / "status.json", record)
        return record
    except BaseException as error:
        campaign._write_json_atomic(
            extension_dir / "status.json",
            {
                "schema_version": 1,
                "state": "failed",
                "failed_at": campaign._utc_now(),
                "branch_id": branch["branch_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
            },
        )
        raise


def _render_uninterrupted_24ps_input(branch: dict[str, Any]) -> str:
    temperature = float(branch["temperature_K"])
    parent_id = str(branch["parent_id"])
    velocity_seed = int(branch["velocity_seed"])
    thermostat_seed = int(branch["thermostat_seed"])
    mass = atomic_masses[atomic_numbers["Al"]]
    return f"""# Uninterrupted 24 ps comparator for the continuation smoke test.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../parents/{parent_id}/parent.lammps.data

mass 1 {mass:.12g}
pair_style meam
pair_coeff * * ../../potential/Lee2003_Al.library.meam Al ../../potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {TIMESTEP_FS / 1000.0:.12g}
velocity all create {temperature:.12g} {velocity_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all temp/csld {temperature:.12g} {temperature:.12g} {THERMOSTAT_TIME_FS / 1000.0:.12g} {thermostat_seed}

thermo {SAMPLE_INTERVAL_STEPS}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {SAMPLE_INTERVAL_STEPS} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g %.9g %.9g %.9g\"

print \"UNINTERRUPTED_24PS_BEGIN {branch['branch_id']}\"
run {RUN_STEPS}
write_restart midpoint_15ps.restart.bin
run {EXTENSION_STOP_STEP - RUN_STEPS}
write_restart final_24ps.restart.bin
print \"UNINTERRUPTED_24PS_COMPLETE {branch['branch_id']}\"
"""


def continuation_smoke(root: Path, branch_index: int) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = _verify_manifest(root)
    if manifest["design_kind"] != "single_parent_preproduction_smoke_8x2":
        raise RuntimeError(f"Continuation smoke requires the 16-branch smoke campaign: {root}.")
    summary = campaign._load_json(root / "summary.json")
    if summary.get("state") != "complete" or int(summary["complete_outcome_count"]) != 16:
        raise RuntimeError(f"All 16 mandatory smoke branches must complete first: {root / 'summary.json'}.")
    index = int(branch_index)
    branch = manifest["branches"][index]
    result_path = root / "continuation_smoke_test.json"
    if result_path.is_file():
        _require_exact_continuation(root, manifest)
        return campaign._load_json(result_path)

    extension = _run_extension(
        root,
        index,
        selection_reason="preregistered_exact_continuation_smoke",
        selection_probability=1.0,
    )
    branch_dir = root / str(branch["branch_dir"])
    comparison_dir = root / "branches" / f"continuation_smoke_uninterrupted_{index:04d}"
    if comparison_dir.exists():
        raise RuntimeError(
            f"Partial uninterrupted comparator exists and must be archived explicitly: {comparison_dir}."
        )
    comparison_dir.mkdir()
    (comparison_dir / "in.lammps").write_text(
        _render_uninterrupted_24ps_input(branch), encoding="utf-8"
    )
    command = _lammps_command(mpi_ranks=24, launcher="srun_pmi2")
    stdout_path = comparison_dir / "lammps.stdout.log"
    try:
        with stdout_path.open("wb") as stdout:
            completed = subprocess.run(
                command,
                cwd=comparison_dir,
                env=_lammps_environment(),
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Uninterrupted 24 ps comparator failed with return code {completed.returncode}; "
                f"inspect {stdout_path}."
            )
        text_path = comparison_dir / "trajectory.lammpstrj"
        expected = tuple(range(0, EXTENSION_STOP_STEP + 1, SAMPLE_INTERVAL_STEPS))
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(text_path)
        if scan.num_atoms != campaign.ATOM_COUNT or tuple(scan.timesteps.tolist()) != expected:
            raise RuntimeError(
                f"Uninterrupted comparator violates the 24 ps contract: atoms={scan.num_atoms}, "
                f"timesteps={scan.timesteps.tolist()}."
            )
        text_sha256 = campaign._sha256_file(text_path)
        uninterrupted = convert_shooting_trajectory(
            text_path,
            comparison_dir / "trajectory_binary_float32",
            timesteps=expected,
            atom_count=campaign.ATOM_COUNT,
            storage_dtype="float32",
            provenance={
                "purpose": "uninterrupted_24ps_continuation_comparator",
                "branch_id": branch["branch_id"],
                "source_sha256": text_sha256,
            },
        )
        uninterrupted.verify_checksums()
        composed = ShootingBinaryTrajectory.load(
            branch_dir / "extension_15_to_24ps/composed_0_to_24ps_float32"
        )
        composed.verify_checksums()
        position_equal = bool(np.array_equal(uninterrupted.positions, composed.positions))
        velocity_equal = bool(np.array_equal(uninterrupted.velocities, composed.velocities))
        maximum_position_difference = float(
            np.max(np.abs(np.asarray(uninterrupted.positions) - np.asarray(composed.positions)))
        )
        maximum_velocity_difference = float(
            np.max(np.abs(np.asarray(uninterrupted.velocities) - np.asarray(composed.velocities)))
        )
        if not position_equal or not velocity_equal:
            raise RuntimeError(
                "The exact continuation smoke test diverged from uninterrupted temp/csld "
                f"dynamics: max_position_difference_A={maximum_position_difference}, "
                f"max_velocity_difference_A_per_ps={maximum_velocity_difference}. "
                "Production submission remains blocked pending an explicit numerical audit."
            )
        text_size = text_path.stat().st_size
        if campaign._sha256_file(text_path) != text_sha256:
            raise RuntimeError(f"Uninterrupted text trajectory changed during conversion: {text_path}.")
        text_path.unlink()
        result = {
            "schema_version": 1,
            "state": "complete",
            "completed_at": campaign._utc_now(),
            "campaign_root": str(root),
            "branch_index": index,
            "branch_id": branch["branch_id"],
            "mpi_ranks": 24,
            "thermostat_style": "temp/csld",
            "short_duration_ps": DURATION_PS,
            "extended_duration_ps": EXTENSION_STOP_PS,
            "frame_count": uninterrupted.frame_count,
            "positions_bitwise_equal_after_text_quantization": position_equal,
            "velocities_bitwise_equal_after_text_quantization": velocity_equal,
            "maximum_position_difference_A": maximum_position_difference,
            "maximum_velocity_difference_A_per_ps": maximum_velocity_difference,
            "short_outcome_sha256": extension["short_outcome_unchanged_sha256"],
            "extension_outcome_sha256": campaign._sha256_file(
                branch_dir / "extension_15_to_24ps/outcome.json"
            ),
            "uninterrupted_text_size_bytes": text_size,
            "uninterrupted_text_sha256": text_sha256,
            "uninterrupted_text_deleted": True,
        }
        campaign._write_json_atomic(result_path, result)
        accepted = dict(extension, classification="exact_continuation",
                        acceptance_proof=str(result_path),
                        acceptance_proof_sha256=campaign._sha256_file(result_path))
        _update_extension_index(root, accepted)
        return result
    except BaseException as error:
        campaign._write_json_atomic(
            root / "continuation_smoke_test_failed.json",
            {
                "schema_version": 1,
                "state": "failed",
                "failed_at": campaign._utc_now(),
                "branch_index": index,
                "branch_id": branch["branch_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "production_submission_blocked": True,
                "partial_artifacts_preserved": True,
            },
        )
        raise


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    topup = subparsers.add_parser("prepare-topup")
    topup.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    topup.add_argument("--campaign-root", type=Path, default=TOPUP_ROOT)
    topup.add_argument("--wave-size", type=int, default=1)
    smoke = subparsers.add_parser("prepare-smoke")
    smoke.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    smoke.add_argument("--campaign-root", type=Path, default=SMOKE_ROOT)
    smoke.add_argument("--wave-size", type=int, default=2)
    task = subparsers.add_parser("run-task")
    task.add_argument("--campaign-root", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    sequential = subparsers.add_parser("run-slurm-subset")
    sequential.add_argument("--campaign-root", type=Path, required=True)
    sequential.add_argument("--start-index", type=int, required=True)
    sequential.add_argument("--stop-index", type=int, required=True)
    sequential.add_argument("--summarize-after", action="store_true")
    submit = subparsers.add_parser("submit-next-wave")
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--start-index", type=int, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, required=True)
    archive = subparsers.add_parser("archive-partial")
    archive.add_argument("--campaign-root", type=Path, required=True)
    archive.add_argument("--branch-index", type=int, required=True)
    archive.add_argument("--label", required=True)
    extend = subparsers.add_parser("extend-branch")
    extend.add_argument("--campaign-root", type=Path, required=True)
    extend.add_argument("--branch-index", type=int, required=True)
    extend.add_argument("--selection-reason", required=True)
    extend.add_argument("--selection-probability", type=float, required=True)
    continuation = subparsers.add_parser("continuation-smoke")
    continuation.add_argument("--campaign-root", type=Path, required=True)
    continuation.add_argument("--branch-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _arguments(argv)
    _configure()
    if args.command == "prepare-topup":
        result = prepare_topup(args.snapshot, args.campaign_root, wave_size=args.wave_size)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "prepare-smoke":
        result = prepare_smoke(args.snapshot, args.campaign_root, wave_size=args.wave_size)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "run-task":
        campaign.run_task(args.campaign_root, args.task_index)
    elif args.command == "run-slurm-subset":
        print(
            json.dumps(
                run_slurm_subset(
                    args.campaign_root,
                    args.start_index,
                    args.stop_index,
                    summarize_after=args.summarize_after,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "submit-next-wave":
        submit_next_wave(args.campaign_root, args.start_index)
    elif args.command == "summarize":
        print(json.dumps(summarize(args.campaign_root), indent=2, sort_keys=True))
    elif args.command == "archive-partial":
        print(campaign.archive_partial(args.campaign_root, args.branch_index, label=args.label))
    elif args.command == "extend-branch":
        print(
            json.dumps(
                extend_branch(
                    args.campaign_root,
                    args.branch_index,
                    selection_reason=args.selection_reason,
                    selection_probability=args.selection_probability,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "continuation-smoke":
        print(
            json.dumps(
                continuation_smoke(args.campaign_root, args.branch_index),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}.")


if __name__ == "__main__":
    main()
