#!/usr/bin/env python3
"""Continue completed nested branches into a physical 24 ps fixed-horizon atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.shooting_binary import (  # noqa: E402
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    compose_shooting_binary_trajectories,
    convert_shooting_trajectory,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
)


SCHEMA_VERSION = 1
EXPECTED_SOURCE_CAMPAIGN_TYPE = (
    "transition_balanced_nested_langevin_nvt_shooting_pilot"
)
CAMPAIGN_TYPE = "fixed_horizon_compatibility_from_nested_first_passage"
EXPECTED_ATOM_COUNT = 70_304
TIMESTEP_FS = 3.0
FIXED_HORIZONS_PS = (6.0, 12.0, 24.0)
FIXED_DURATION_PS = FIXED_HORIZONS_PS[-1]
FIXED_DURATION_STEPS = int(round(FIXED_DURATION_PS * 1000.0 / TIMESTEP_FS))
SAMPLE_INTERVAL_STEPS = 100
FIXED_TIMESTEPS = tuple(
    range(0, FIXED_DURATION_STEPS + 1, SAMPLE_INTERVAL_STEPS)
)
STORAGE_DTYPE = "float16"
MPI_RANKS = 48
BATCH_SIZE = 12
THERMOSTAT_TIME_FS = 300.0
LIBRARY_SHA256 = "f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e"
PARAMETER_SHA256 = "b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuation_seed(branch_index: int) -> int:
    digest = hashlib.sha256(
        f"nested-fixed-horizon-continuation:{branch_index}:20260902".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") % 899_999_999 + 1


def render_continuation_input(
    *,
    source_restart: Path,
    temperature_K: float,
    thermostat_seed: int,
    first_step: int,
) -> str:
    remaining_steps = FIXED_DURATION_STEPS - int(first_step)
    if remaining_steps <= 0 or first_step % SAMPLE_INTERVAL_STEPS != 0:
        raise ValueError(
            f"Continuation requires an aligned step below {FIXED_DURATION_STEPS}, "
            f"got first_step={first_step}."
        )
    return f"""# Physical Markov continuation of a completed nested first-passage branch.
log continuation.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_restart {source_restart}

pair_style meam
pair_coeff * * ../../potential/Lee2003_Al.library.meam Al ../../potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {TIMESTEP_FS / 1000.0:.12g}

# Positions and velocities come from the first-passage restart. The fresh noise
# stream is a valid continuation of the Markov Langevin process; velocities are
# deliberately not resampled.
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all langevin {temperature_K:.12g} {temperature_K:.12g} {THERMOSTAT_TIME_FS / 1000.0:.12g} {thermostat_seed} zero yes
thermo {SAMPLE_INTERVAL_STEPS}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump continuation all custom {SAMPLE_INTERVAL_STEPS} continuation.lammpstrj id type x y z vx vy vz
dump_modify continuation first yes sort id format line "%d %d %.9g %.9g %.9g %.9g %.9g %.9g"

print "NESTED_FIXED_HORIZON_CONTINUATION_BEGIN"
run 0
run {remaining_steps}
undump continuation
write_restart final.restart.bin
print "NESTED_FIXED_HORIZON_CONTINUATION_COMPLETE"
"""


def _write_slurm_scripts(root: Path) -> None:
    runner = Path(__file__).resolve()
    common = f"""set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
"""
    slurm = root / "slurm"
    task = slurm / "run_batch.sbatch"
    task.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_fix24
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks={MPI_RANKS}
#SBATCH --ntasks-per-node={MPI_RANKS}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output={slurm}/batch_%j.out
#SBATCH --error={slurm}/batch_%j.err

{common}: "${{COMPAT_START:?COMPAT_START must be set}}"
: "${{COMPAT_STOP:?COMPAT_STOP must be set}}"
python {runner} run-batch --campaign-root {root} --start-index "${{COMPAT_START}}" --stop-index "${{COMPAT_STOP}}"
""",
        encoding="utf-8",
    )
    task.chmod(0o750)
    controller = slurm / "submit_batch.sbatch"
    controller.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_fixctl
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output={slurm}/controller_%j.out
#SBATCH --error={slurm}/controller_%j.err

{common}: "${{COMPAT_START:?COMPAT_START must be set}}"
python {runner} submit-next-batch --campaign-root {root} --start-index "${{COMPAT_START}}"
""",
        encoding="utf-8",
    )
    controller.chmod(0o750)
    summary = slurm / "summarize.sbatch"
    summary.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_fixsum
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output={slurm}/summary_%j.out
#SBATCH --error={slurm}/summary_%j.err

{common}python {runner} summarize --campaign-root {root}
""",
        encoding="utf-8",
    )
    summary.chmod(0o750)


def prepare_campaign(
    source_campaign_root: str | Path, campaign_root: str | Path
) -> dict[str, Any]:
    source_root = Path(source_campaign_root).expanduser().resolve()
    root = Path(campaign_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Compatibility campaign already exists: {root}")
    source_manifest = _load_json(source_root / "manifest.json")
    source_summary = _load_json(source_root / "summary.json")
    if source_manifest.get("campaign_type") != EXPECTED_SOURCE_CAMPAIGN_TYPE:
        raise ValueError(
            f"Unsupported nested source type={source_manifest.get('campaign_type')!r}: "
            f"{source_root}."
        )
    if source_summary.get("state") != "complete":
        raise RuntimeError(f"Nested source summary is not complete: {source_root / 'summary.json'}.")
    if int(source_manifest["atom_count"]) != EXPECTED_ATOM_COUNT:
        raise RuntimeError(
            f"Nested source atom count changed: {source_manifest['atom_count']}."
        )
    potential_files = (
        ("Lee2003_Al.library.meam", LIBRARY_SHA256),
        ("Lee2003_Al.meam", PARAMETER_SHA256),
    )
    for filename, expected_sha256 in potential_files:
        path = source_root / "potential" / filename
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Nested source potential checksum changed: path={path}, "
                f"expected={expected_sha256}, observed={observed_sha256}."
            )

    root.mkdir(parents=True)
    (root / "branches").mkdir()
    (root / "potential").mkdir()
    (root / "slurm").mkdir()
    for filename, _ in potential_files:
        shutil.copy2(source_root / "potential" / filename, root / "potential" / filename)

    branches: list[dict[str, Any]] = []
    continuation_count = 0
    first_passage_counts: Counter[str] = Counter()
    for source_branch in source_manifest["branches"]:
        branch_index = int(source_branch["branch_index"])
        source_branch_dir = source_root / str(source_branch["branch_dir"])
        source_outcome_path = source_branch_dir / "outcome.json"
        source_outcome = _load_json(source_outcome_path)
        if source_outcome.get("state") != "complete":
            raise RuntimeError(f"Nested source branch is not complete: {source_outcome_path}.")
        source_binary = ShootingBinaryTrajectory.load(
            Path(str(source_outcome["trajectory_artifact"]["path"]))
        )
        source_binary.verify_checksums()
        last_step = int(source_outcome["last_timestep"])
        available_fixed = tuple(step for step in FIXED_TIMESTEPS if step <= last_step)
        missing_existing = [
            step for step in available_fixed if step not in set(source_binary.timesteps.tolist())
        ]
        if missing_existing:
            raise RuntimeError(
                f"Nested source lacks regular 0.3 ps frames before stopping: "
                f"branch={source_branch['branch_id']}, missing={missing_existing}."
            )
        continuation_required = last_step < FIXED_DURATION_STEPS
        continuation_count += int(continuation_required)
        outcome_name = source_outcome["first_passage_outcome"]
        first_passage_counts[
            "censored" if outcome_name is None else str(outcome_name)
        ] += 1
        branch_dir = root / "branches" / str(source_branch["branch_id"])
        branch_dir.mkdir()
        branch = {
            **source_branch,
            "branch_dir": str(branch_dir.relative_to(root)),
            "source_branch_dir": str(source_branch_dir),
            "source_outcome_path": str(source_outcome_path),
            "source_outcome_sha256": _sha256_file(source_outcome_path),
            "source_binary_path": str(source_binary.root),
            "source_last_timestep": last_step,
            "continuation_required": continuation_required,
            "continuation_from_timestep": last_step if continuation_required else None,
            "continuation_thermostat_seed": (
                _continuation_seed(branch_index) if continuation_required else None
            ),
            "fixed_duration_ps": FIXED_DURATION_PS,
            "fixed_duration_steps": FIXED_DURATION_STEPS,
            "fixed_horizons_ps": list(FIXED_HORIZONS_PS),
            "storage_dtype": STORAGE_DTYPE,
        }
        _write_json_atomic(branch_dir / "metadata.json", branch)
        if continuation_required:
            restart = source_branch_dir / "final.restart.bin"
            if not restart.is_file() or restart.stat().st_size == 0:
                raise RuntimeError(
                    f"Nested source continuation restart is missing: {restart}."
                )
            (branch_dir / "continuation.in.lammps").write_text(
                render_continuation_input(
                    source_restart=restart,
                    temperature_K=float(source_branch["temperature_K"]),
                    thermostat_seed=int(branch["continuation_thermostat_seed"]),
                    first_step=last_step,
                ),
                encoding="utf-8",
            )
        branches.append(branch)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "prepared",
        "created_at": _utc_now(),
        "campaign_type": CAMPAIGN_TYPE,
        "source_campaign_root": str(source_root),
        "source_campaign_manifest_sha256": _sha256_file(source_root / "manifest.json"),
        "source_campaign_summary_sha256": _sha256_file(source_root / "summary.json"),
        "scientific_contract": {
            "no_fabricated_frames": (
                "Every frame is either from the original nested LAMMPS path or a LAMMPS "
                "continuation from its saved first-passage restart."
            ),
            "first_passage_labels": (
                "Original outcomes and times are copied without reclassification."
            ),
            "continuation_randomness": (
                "Positions and velocities are retained at the restart; only the Markov "
                "Langevin noise stream receives a new checksum-derived seed."
            ),
        },
        "atom_count": EXPECTED_ATOM_COUNT,
        "protocol": {
            "ensemble": "fixed-cell Langevin NVT",
            "timestep_fs": TIMESTEP_FS,
            "duration_ps": FIXED_DURATION_PS,
            "run_steps": FIXED_DURATION_STEPS,
            "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
            "sample_interval_ps": SAMPLE_INTERVAL_STEPS * TIMESTEP_FS / 1000.0,
            "expected_frame_count": len(FIXED_TIMESTEPS),
            "fixed_horizons_ps": list(FIXED_HORIZONS_PS),
            "dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
            "storage_dtype": STORAGE_DTYPE,
        },
        "execution": {
            "partition": "CPU",
            "mpi_ranks_per_continuation": MPI_RANKS,
            "memory": "24G",
            "time_limit": "06:00:00",
            "branches_per_sequential_batch": BATCH_SIZE,
            "concurrent_batches": 1,
        },
        "counts": {
            "parents": len(source_manifest["parents"]),
            "branches": len(branches),
            "branches_requiring_lammps_continuation": continuation_count,
            "branches_reusing_existing_24ps_path": len(branches) - continuation_count,
            "first_passage_outcomes": dict(first_passage_counts),
        },
        "parents": source_manifest["parents"],
        "branches": branches,
    }
    _write_json_atomic(root / "manifest.json", manifest)
    _write_json_atomic(
        root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "updated_at": _utc_now(),
            "complete_branch_count": 0,
            "pending_branch_count": len(branches),
        },
    )
    _write_slurm_scripts(root)
    return manifest


def _lammps_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MPIR_CVAR_CH4_NETMOD": "ofi",
            "FI_PROVIDER": "tcp",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    environment["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + (
        f":{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else ""
    )
    return environment


def _run_lammps(branch_dir: Path) -> float:
    lmp = Path(sys.prefix) / "bin" / "lmp"
    srun = shutil.which("srun")
    if "SLURM_JOB_ID" not in os.environ or srun is None or not lmp.is_file():
        raise RuntimeError(
            f"Nested continuation requires Slurm and pointnet LAMMPS: "
            f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')!r}, srun={srun!r}, lmp={lmp}."
        )
    command = [
        srun,
        "--mpi=pmi2",
        "--nodes=1",
        f"--ntasks={MPI_RANKS}",
        f"--ntasks-per-node={MPI_RANKS}",
        "--cpus-per-task=1",
        "--cpu-bind=cores",
        "--kill-on-bad-exit=1",
        str(lmp),
        "-in",
        "continuation.in.lammps",
    ]
    started = time.monotonic()
    with (branch_dir / "continuation.stdout.log").open("wb") as stdout:
        completed = subprocess.run(
            command,
            cwd=branch_dir,
            env=_lammps_environment(),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nested continuation LAMMPS failed with return code {completed.returncode}: "
            f"branch_dir={branch_dir}, log={branch_dir / 'continuation.stdout.log'}."
        )
    return elapsed


def run_branch(campaign_root: str | Path, task_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    index = int(task_index)
    if not isinstance(branches, list) or index < 0 or index >= len(branches):
        raise IndexError(f"task_index={index} is invalid for {root / 'manifest.json'}.")
    branch = branches[index]
    branch_dir = root / str(branch["branch_dir"])
    outcome_path = branch_dir / "outcome.json"
    if outcome_path.is_file():
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Existing compatibility outcome is not complete: {outcome_path}.")
        print(f"Compatibility branch {branch['branch_id']} is already complete.", flush=True)
        return outcome
    partial_names = [
        name
        for name in (
            "continuation.lammpstrj",
            "continuation_binary_float16",
            "trajectory_binary_float16",
            "continuation.lammps.log",
            "continuation.stdout.log",
            "final.restart.bin",
            "status.json",
        )
        if (branch_dir / name).exists()
    ]
    interrupted_builds = sorted(branch_dir.glob(".*.building-*"))
    if partial_names or interrupted_builds:
        raise RuntimeError(
            f"Compatibility branch has partial artifacts; preserve/archive before retry: "
            f"branch={branch['branch_id']}, artifacts={partial_names}, "
            f"building={[path.name for path in interrupted_builds]}."
        )
    _write_json_atomic(
        branch_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "updated_at": _utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
        },
    )
    try:
        source_outcome_path = Path(str(branch["source_outcome_path"]))
        if _sha256_file(source_outcome_path) != branch["source_outcome_sha256"]:
            raise RuntimeError(f"Nested source outcome changed: {source_outcome_path}.")
        source_outcome = _load_json(source_outcome_path)
        original = ShootingBinaryTrajectory.load(Path(str(branch["source_binary_path"])))
        original.verify_checksums()
        segments = [original]
        elapsed_seconds = 0.0
        continuation_source_size = 0
        continuation_source_sha256: str | None = None
        if bool(branch["continuation_required"]):
            allocated = int(os.environ.get("SLURM_NTASKS", "0"))
            if allocated != MPI_RANKS:
                raise RuntimeError(
                    f"SLURM_NTASKS={allocated}, expected {MPI_RANKS} for continuation."
                )
            elapsed_seconds = _run_lammps(branch_dir)
            text_path = branch_dir / "continuation.lammpstrj"
            scan = TemporalLAMMPSDumpDataset.scan_dump_file(text_path)
            expected_continuation_steps = np.arange(
                int(branch["continuation_from_timestep"]),
                FIXED_DURATION_STEPS + 1,
                SAMPLE_INTERVAL_STEPS,
                dtype=np.int64,
            )
            if (
                scan.num_atoms != EXPECTED_ATOM_COUNT
                or tuple(scan.atom_columns)
                != ("id", "type", "x", "y", "z", "vx", "vy", "vz")
                or not np.array_equal(scan.timesteps, expected_continuation_steps)
            ):
                raise RuntimeError(
                    f"Continuation dump violates the fixed-horizon contract: "
                    f"branch={branch['branch_id']}, atoms={scan.num_atoms}, "
                    f"columns={scan.atom_columns}, timesteps={scan.timesteps.tolist()}."
                )
            continuation_source_size = text_path.stat().st_size
            continuation_source_sha256 = _sha256_file(text_path)
            continuation = convert_shooting_trajectory(
                text_path,
                branch_dir / "continuation_binary_float16",
                timesteps=tuple(int(value) for value in expected_continuation_steps),
                atom_count=EXPECTED_ATOM_COUNT,
                storage_dtype=STORAGE_DTYPE,
                provenance={
                    "campaign_type": CAMPAIGN_TYPE,
                    "branch_id": branch["branch_id"],
                    "segment": "post-first-passage Markov continuation",
                },
            )
            continuation.verify_checksums()
            final_restart = branch_dir / "final.restart.bin"
            if not final_restart.is_file() or final_restart.stat().st_size == 0:
                raise RuntimeError(f"Continuation restart is missing: {final_restart}.")
            segments.append(continuation)

        target = branch_dir / "trajectory_binary_float16"
        composed = compose_shooting_binary_trajectories(
            segments,
            target,
            timesteps=FIXED_TIMESTEPS,
            storage_dtype=STORAGE_DTYPE,
            provenance={
                "campaign_type": CAMPAIGN_TYPE,
                "branch_id": branch["branch_id"],
                "source_campaign_root": manifest["source_campaign_root"],
                "source_first_passage_outcome_sha256": branch[
                    "source_outcome_sha256"
                ],
                "continuation_from_timestep": branch["continuation_from_timestep"],
                "continuation_thermostat_seed": branch[
                    "continuation_thermostat_seed"
                ],
            },
        )
        if (
            composed.storage_dtype != np.dtype(STORAGE_DTYPE)
            or composed.atom_count != EXPECTED_ATOM_COUNT
            or not np.array_equal(composed.timesteps, FIXED_TIMESTEPS)
        ):
            raise RuntimeError(
                f"Composed trajectory failed validation: branch={branch['branch_id']}."
            )
        sizes = binary_directory_sizes(target)
        if bool(branch["continuation_required"]):
            # The composed trajectory is self-contained, but ``continuation`` still
            # owns read-only numpy memmaps here. Deleting their directory on NFS
            # creates .nfs files and can fail after the scientifically complete
            # trajectory has already been written. Keep the small intermediate
            # binary as provenance and only remove the much larger text dump.
            (branch_dir / "continuation.lammpstrj").unlink()
        outcome = {
            **branch,
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "completed_at": _utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "elapsed_seconds": elapsed_seconds,
            "first_passage_outcome": source_outcome["first_passage_outcome"],
            "first_passage_onset_timestep": source_outcome[
                "first_passage_onset_timestep"
            ],
            "first_passage_confirmation_timestep": source_outcome[
                "first_passage_confirmation_timestep"
            ],
            "first_passage_time_ps": source_outcome["first_passage_time_ps"],
            "censored": source_outcome["censored"],
            "frame_count": composed.frame_count,
            "first_timestep": int(composed.timesteps[0]),
            "last_timestep": int(composed.timesteps[-1]),
            "trajectory_artifact": {
                "format": composed.manifest["format"],
                "path": str(target),
                "storage_dtype": STORAGE_DTYPE,
                "frame_count": composed.frame_count,
                "apparent_size_bytes": sizes["apparent_bytes"],
                "allocated_size_bytes": sizes["allocated_bytes"],
            },
            "continuation_source_lammpstrj": (
                None
                if not branch["continuation_required"]
                else {
                    "size_bytes": continuation_source_size,
                    "sha256": continuation_source_sha256,
                    "deleted": True,
                }
            ),
            "continuation_binary_intermediate": (
                None
                if not branch["continuation_required"]
                else {
                    "path": str(branch_dir / "continuation_binary_float16"),
                    "preserved": True,
                }
            ),
        }
        _write_json_atomic(outcome_path, outcome)
        _write_json_atomic(branch_dir / "status.json", outcome)
        print(json.dumps(outcome, indent=2, sort_keys=True), flush=True)
        return outcome
    except BaseException as error:
        _write_json_atomic(
            branch_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "updated_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
            },
        )
        raise


def recover_completed_compositions(campaign_root: str | Path) -> dict[str, int]:
    """Finalize branches that failed only during post-composition NFS cleanup.

    This recovery never invokes LAMMPS and never reconstructs frames. It accepts
    only the exact historical cleanup failure, verifies every self-contained
    composed binary and its immutable first-passage source, and writes outcomes
    only after all recoverable branches have passed validation.
    """

    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("campaign_type") != CAMPAIGN_TYPE:
        raise ValueError(
            f"Unsupported compatibility campaign_type={manifest.get('campaign_type')!r}: "
            f"{root / 'manifest.json'}."
        )

    planned: list[tuple[Path, dict[str, Any]]] = []
    already_complete = 0
    for branch in manifest["branches"]:
        branch_dir = root / str(branch["branch_dir"])
        outcome_path = branch_dir / "outcome.json"
        if outcome_path.is_file():
            outcome = _load_json(outcome_path)
            if outcome.get("state") != "complete":
                raise RuntimeError(
                    f"Existing recovery outcome is not complete: {outcome_path}."
                )
            already_complete += 1
            continue
        if not bool(branch["continuation_required"]):
            raise RuntimeError(
                "A branch that reused an existing 24 ps path has no outcome and "
                f"cannot be handled by cleanup recovery: {branch['branch_id']}."
            )
        status_path = branch_dir / "status.json"
        status = _load_json(status_path)
        error = str(status.get("error", ""))
        if (
            status.get("state") != "failed"
            or status.get("error_type") != "OSError"
            or "Directory not empty" not in error
            or "continuation_binary_float16" not in error
        ):
            raise RuntimeError(
                "Refusing to recover a branch whose failure was not the known "
                f"post-composition NFS cleanup error: branch={branch['branch_id']}, "
                f"status={status_path}."
            )

        source_outcome_path = Path(str(branch["source_outcome_path"]))
        if _sha256_file(source_outcome_path) != branch["source_outcome_sha256"]:
            raise RuntimeError(f"Nested source outcome changed: {source_outcome_path}.")
        source_outcome = _load_json(source_outcome_path)
        target = branch_dir / "trajectory_binary_float16"
        composed = ShootingBinaryTrajectory.load(target)
        composed.verify_checksums()
        if (
            composed.storage_dtype != np.dtype(STORAGE_DTYPE)
            or composed.atom_count != EXPECTED_ATOM_COUNT
            or not np.array_equal(composed.timesteps, FIXED_TIMESTEPS)
        ):
            raise RuntimeError(
                "Recovery candidate violates the fixed-horizon binary contract: "
                f"branch={branch['branch_id']}, dtype={composed.storage_dtype.name}, "
                f"atoms={composed.atom_count}, timesteps={composed.timesteps.tolist()}."
            )
        provenance = composed.manifest.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("campaign_type") != CAMPAIGN_TYPE
            or provenance.get("branch_id") != branch["branch_id"]
            or provenance.get("source_first_passage_outcome_sha256")
            != branch["source_outcome_sha256"]
        ):
            raise RuntimeError(
                f"Recovery binary provenance disagrees with branch={branch['branch_id']}."
            )
        final_restart = branch_dir / "final.restart.bin"
        if not final_restart.is_file() or final_restart.stat().st_size == 0:
            raise RuntimeError(f"Recovery restart is missing: {final_restart}.")
        sizes = binary_directory_sizes(target)
        recovered_at = _utc_now()
        outcome = {
            **branch,
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "completed_at": str(composed.manifest["created_at"]),
            "slurm_job_id": status.get("slurm_job_id"),
            "hostname": None,
            "elapsed_seconds": None,
            "first_passage_outcome": source_outcome["first_passage_outcome"],
            "first_passage_onset_timestep": source_outcome[
                "first_passage_onset_timestep"
            ],
            "first_passage_confirmation_timestep": source_outcome[
                "first_passage_confirmation_timestep"
            ],
            "first_passage_time_ps": source_outcome["first_passage_time_ps"],
            "censored": source_outcome["censored"],
            "frame_count": composed.frame_count,
            "first_timestep": int(composed.timesteps[0]),
            "last_timestep": int(composed.timesteps[-1]),
            "trajectory_artifact": {
                "format": composed.manifest["format"],
                "path": str(target),
                "storage_dtype": STORAGE_DTYPE,
                "frame_count": composed.frame_count,
                "apparent_size_bytes": sizes["apparent_bytes"],
                "allocated_size_bytes": sizes["allocated_bytes"],
            },
            "continuation_source_lammpstrj": {
                "deleted": True,
                "size_bytes": None,
                "sha256": None,
                "unavailable_reason": (
                    "worker deleted the verified text dump before the historical "
                    "NFS intermediate-cleanup failure"
                ),
            },
            "continuation_binary_intermediate": {
                "path": str(branch_dir / "continuation_binary_float16"),
                "preserved": False,
                "unavailable_reason": (
                    "historical cleanup removed its files before NFS reported an "
                    "open-memmap directory"
                ),
            },
            "recovery": {
                "recovered_at": recovered_at,
                "reason": "post-composition NFS cleanup failure",
                "composed_binary_checksums_verified": True,
                "frames_reconstructed": False,
                "lammps_rerun": False,
                "previous_status": str(status_path),
            },
        }
        planned.append((branch_dir, outcome))

    for branch_dir, outcome in planned:
        _write_json_atomic(branch_dir / "outcome.json", outcome)
        _write_json_atomic(branch_dir / "status.json", outcome)
    result = {"already_complete": already_complete, "recovered": len(planned)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_batch(campaign_root: str | Path, start_index: int, stop_index: int) -> None:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branch_count = len(manifest["branches"])
    start = int(start_index)
    stop = int(stop_index)
    if start < 0 or stop < start or stop >= branch_count:
        raise IndexError(f"Invalid compatibility batch [{start}, {stop}] of {branch_count}.")
    failures: list[dict[str, Any]] = []
    for index in range(start, stop + 1):
        try:
            run_branch(root, index)
        except BaseException as error:
            failures.append(
                {"branch_index": index, "error_type": type(error).__name__, "error": str(error)}
            )
            traceback.print_exc()
    if os.environ.get("COMPAT_AUTO_CHAIN") == "1":
        _submit_following_job(root, current_stop=stop, branch_count=branch_count)
    if failures:
        raise RuntimeError(f"Compatibility batch completed with failures: {failures}.")


def _submit_following_job(root: Path, *, current_stop: int, branch_count: int) -> None:
    current_job_id = os.environ.get("SLURM_JOB_ID")
    if current_job_id is None or not current_job_id.isdigit():
        raise RuntimeError(
            "Automatic compatibility chaining requires a numeric SLURM_JOB_ID."
        )
    next_start = int(current_stop) + 1
    if next_start < int(branch_count):
        next_stop = min(next_start + BATCH_SIZE - 1, int(branch_count) - 1)
        successor_kind = "worker"
        command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{current_job_id}",
            (
                "--export=ALL,COMPAT_AUTO_CHAIN=1,"
                f"COMPAT_START={next_start},COMPAT_STOP={next_stop}"
            ),
            str(root / "slurm" / "run_batch.sbatch"),
        ]
    else:
        next_stop = None
        successor_kind = "summary"
        command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{current_job_id}",
            str(root / "slurm" / "summarize.sbatch"),
        ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    successor_job_id = result.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(
            f"Invalid auto-chained compatibility job ID: {result.stdout!r}."
        )
    record = {
        "submitted_at": _utc_now(),
        "submitting_job_id": current_job_id,
        "completed_batch_stop": int(current_stop),
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
        "successor_batch_start": next_start if successor_kind == "worker" else None,
        "successor_batch_stop": next_stop,
    }
    with (root / "slurm" / "submission_chain.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json_atomic(root / "slurm" / "active_submission.json", record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


def _active_submission_conflicts(root: Path) -> list[str]:
    path = root / "slurm" / "active_submission.json"
    if not path.is_file():
        return []
    active = _load_json(path)
    job_ids = [str(active["worker_job_id"]), str(active["successor_job_id"])]
    queued = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"],
        check=True,
        text=True,
        capture_output=True,
    )
    current = os.environ.get("SLURM_JOB_ID")
    return sorted(
        {
            line.strip()
            for line in queued.stdout.splitlines()
            if line.strip() and line.strip() != current
        }
    )


def submit_next_batch(campaign_root: str | Path, start_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branch_count = len(manifest["branches"])
    start = int(start_index)
    if start < 0 or start >= branch_count:
        raise IndexError(f"start_index={start} is outside [0, {branch_count}).")
    conflicts = _active_submission_conflicts(root)
    if conflicts:
        raise RuntimeError(
            f"Refusing duplicate compatibility submission; active jobs={conflicts}."
        )
    stop = min(start + BATCH_SIZE - 1, branch_count - 1)
    worker_result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--export=ALL,COMPAT_START={start},COMPAT_STOP={stop}",
            str(root / "slurm" / "run_batch.sbatch"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    worker_job_id = worker_result.stdout.strip()
    if not worker_job_id.isdigit():
        raise RuntimeError(f"Invalid Slurm compatibility worker ID: {worker_result.stdout!r}.")
    if stop + 1 < branch_count:
        successor_kind = "controller"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{worker_job_id}",
            f"--export=ALL,COMPAT_START={stop + 1}",
            str(root / "slurm" / "submit_batch.sbatch"),
        ]
    else:
        successor_kind = "summary"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{worker_job_id}",
            str(root / "slurm" / "summarize.sbatch"),
        ]
    successor_result = subprocess.run(
        successor_command, check=True, text=True, capture_output=True
    )
    successor_job_id = successor_result.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(
            f"Invalid compatibility {successor_kind} ID: {successor_result.stdout!r}."
        )
    record = {
        "submitted_at": _utc_now(),
        "submitting_job_id": os.environ.get("SLURM_JOB_ID"),
        "batch_start": start,
        "batch_stop": stop,
        "worker_job_id": worker_job_id,
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
    }
    with (root / "slurm" / "submission_chain.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json_atomic(root / "slurm" / "active_submission.json", record)
    _write_json_atomic(
        root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "submitted",
            "updated_at": _utc_now(),
            **record,
        },
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def summarize_campaign(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    outcomes: list[dict[str, Any]] = []
    missing: list[str] = []
    for branch in manifest["branches"]:
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        if not outcome_path.is_file():
            missing.append(str(branch["branch_id"]))
            continue
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Compatibility outcome is not complete: {outcome_path}.")
        if _sha256_file(Path(str(branch["source_outcome_path"]))) != branch[
            "source_outcome_sha256"
        ]:
            raise RuntimeError(
                f"Nested source outcome changed for branch={branch['branch_id']}."
            )
        binary = ShootingBinaryTrajectory.load(
            Path(str(outcome["trajectory_artifact"]["path"]))
        )
        binary.verify_checksums()
        if (
            binary.storage_dtype != np.dtype(STORAGE_DTYPE)
            or binary.atom_count != EXPECTED_ATOM_COUNT
            or not np.array_equal(binary.timesteps, FIXED_TIMESTEPS)
        ):
            raise RuntimeError(
                f"Compatibility binary contract changed: branch={branch['branch_id']}."
            )
        outcomes.append(outcome)
    if missing:
        raise RuntimeError(
            f"Cannot summarize incomplete compatibility campaign: "
            f"missing={len(missing)}, first={missing[:10]}."
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": _utc_now(),
        "parent_count": int(manifest["counts"]["parents"]),
        "branch_count": len(outcomes),
        "physically_continued_branch_count": sum(
            bool(outcome["continuation_required"]) for outcome in outcomes
        ),
        "existing_path_branch_count": sum(
            not bool(outcome["continuation_required"]) for outcome in outcomes
        ),
        "counts_by_temperature": {
            f"{temperature:g}": sum(
                float(outcome["temperature_K"]) == temperature for outcome in outcomes
            )
            for temperature in (400.0, 450.0, 500.0)
        },
        "first_passage_outcome_counts": dict(
            Counter(
                "censored"
                if outcome["first_passage_outcome"] is None
                else str(outcome["first_passage_outcome"])
                for outcome in outcomes
            )
        ),
        "fixed_horizons_ps": list(FIXED_HORIZONS_PS),
        "frame_count_per_branch": len(FIXED_TIMESTEPS),
        "storage_dtype": STORAGE_DTYPE,
        "apparent_trajectory_bytes": sum(
            int(outcome["trajectory_artifact"]["apparent_size_bytes"])
            for outcome in outcomes
        ),
    }
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-campaign-root", type=Path, required=True)
    prepare.add_argument("--campaign-root", type=Path, required=True)
    task = commands.add_parser("run-task")
    task.add_argument("--campaign-root", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    batch = commands.add_parser("run-batch")
    batch.add_argument("--campaign-root", type=Path, required=True)
    batch.add_argument("--start-index", type=int, required=True)
    batch.add_argument("--stop-index", type=int, required=True)
    submit = commands.add_parser("submit-next-batch")
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--start-index", type=int, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, required=True)
    recover = commands.add_parser("recover-completed-compositions")
    recover.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "prepare":
        manifest = prepare_campaign(args.source_campaign_root, args.campaign_root)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    elif args.action == "run-task":
        run_branch(args.campaign_root, args.task_index)
    elif args.action == "run-batch":
        run_batch(args.campaign_root, args.start_index, args.stop_index)
    elif args.action == "submit-next-batch":
        submit_next_batch(args.campaign_root, args.start_index)
    elif args.action == "summarize":
        summarize_campaign(args.campaign_root)
    elif args.action == "recover-completed-compositions":
        recover_completed_compositions(args.campaign_root)
    else:
        raise AssertionError(f"Unhandled action {args.action!r}.")


if __name__ == "__main__":
    main()
