#!/usr/bin/env python3
"""Generate split-safe, independently melted Al source histories for screening."""

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
from ase.build import bulk
from ase.io import write


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_lammps_homogeneous_campaign import (  # noqa: E402
    PRESSURE_BAR_TO_GPA,
    _read_lammps_dump,
    _read_thermodynamic_log,
)
from scripts.run_lammps_unseeded_meam_crystallization import (  # noqa: E402
    _liquid_validation,
)
from src.data_utils.shooting_binary import (  # noqa: E402
    ShootingBinaryTrajectory,
    convert_shooting_trajectory,
)
from src.data_utils.synthetic.atomistic.transition_analysis import (  # noqa: E402
    CRYSTALLINE_STRUCTURE_TYPES,
    STRUCTURE_NAMES,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
)


SCHEMA_VERSION = 1
EXPECTED_ATOM_COUNT = 70_304
CAMPAIGN_SEED = 20_260_902
TEMPERATURES_K = (400.0, 450.0, 500.0)
RUNS_PER_TEMPERATURE = 30
SPLIT_COUNTS_PER_TEMPERATURE = {
    "optimization": 18,
    "model_selection": 6,
    "final_validation": 6,
}
BOUNDARY_BANDS = {
    400.0: (25, 55),
    450.0: (25, 40),
    500.0: (18, 30),
}
TIMESTEP_FS = 3.0
MELT_TEMPERATURE_K = 1325.0
MELT_STEPS = 100_000
EQUILIBRATION_STEPS = 5_000
MEASUREMENT_STEPS = 200_000
SAMPLE_INTERVAL_STEPS = 250
THERMOSTAT_TIME_FS = 300.0
BAROSTAT_TIME_FS = 3000.0
PRESSURE_GPA = 0.0
MPI_RANKS = 48
ARRAY_CONCURRENCY = 5
STORAGE_DTYPE = "float16"
LIBRARY_SHA256 = "f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e"
PARAMETER_SHA256 = "b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a"
RUNTIME_ARTIFACTS = (
    "melt.lammps.log",
    "melt.stdout.log",
    "melt_validation.lammpstrj",
    "prepared_liquid.lammps.data",
    "melt_final.restart.bin",
    "melt.restart.1.bin",
    "melt.restart.2.bin",
    "equilibration.lammps.log",
    "measurement.lammps.log",
    "source.stdout.log",
    "trajectory.lammpstrj",
    "trajectory_binary_float16",
    "source.restart.1.bin",
    "source.restart.2.bin",
    "final.restart.bin",
    "source_validation.json",
    "crystallization_progress.npz",
    "thermodynamics.npz",
    "outcome.json",
    "status.json",
)


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


def _seed(temperature_K: float, source_index: int, role: str) -> int:
    digest = hashlib.sha256(
        f"{CAMPAIGN_SEED}:{temperature_K:g}:{source_index}:{role}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") % 899_999_999 + 1


def source_run_specs() -> tuple[dict[str, Any], ...]:
    split_sequence = tuple(
        split
        for split, count in SPLIT_COUNTS_PER_TEMPERATURE.items()
        for _ in range(count)
    )
    if len(split_sequence) != RUNS_PER_TEMPERATURE:
        raise RuntimeError(
            f"Source split counts sum to {len(split_sequence)}, expected "
            f"{RUNS_PER_TEMPERATURE}."
        )
    specs: list[dict[str, Any]] = []
    for temperature_K in TEMPERATURES_K:
        for source_index, source_split in enumerate(split_sequence):
            run_index = len(specs)
            preparation_seed = _seed(temperature_K, source_index, "melt")
            velocity_seed = _seed(temperature_K, source_index, "quench")
            run_id = (
                f"source_{run_index:03d}_T{temperature_K:g}_{source_split}_"
                f"melt{preparation_seed}_velocity{velocity_seed}"
            )
            specs.append(
                {
                    "run_index": run_index,
                    "source_index_within_temperature": source_index,
                    "run_id": run_id,
                    "run_dir": f"runs/{run_id}",
                    "temperature_K": temperature_K,
                    "source_split": source_split,
                    "preparation_seed": preparation_seed,
                    "velocity_seed": velocity_seed,
                    "boundary_cluster_min_atoms": BOUNDARY_BANDS[temperature_K][0],
                    "boundary_cluster_max_atoms": BOUNDARY_BANDS[temperature_K][1],
                }
            )
    seeds = [
        int(spec[key])
        for spec in specs
        for key in ("preparation_seed", "velocity_seed")
    ]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Generated melt and quench seeds are not globally unique.")
    return tuple(specs)


def _pair_commands() -> str:
    return """pair_style meam
pair_coeff * * ../../potential/Lee2003_Al.library.meam Al ../../potential/Lee2003_Al.meam Al"""


def render_melt_input(spec: dict[str, Any]) -> str:
    pressure_bar = PRESSURE_GPA / PRESSURE_BAR_TO_GPA
    return f"""# Independent full-box melt for {spec['run_id']}.
log melt.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../initial_fcc.lammps.data

mass 1 26.9815
{_pair_commands()}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {TIMESTEP_FS / 1000.0:.12g}

velocity all create {MELT_TEMPERATURE_K:.12g} {spec['preparation_seed']} mom yes rot no dist gaussian
fix remove_drift all momentum 100 linear 1 1 1
fix melt all npt temp {MELT_TEMPERATURE_K:.12g} {MELT_TEMPERATURE_K:.12g} {THERMOSTAT_TIME_FS / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {BAROSTAT_TIME_FS / 1000.0:.12g}
thermo 1000
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
restart 25000 melt.restart.1.bin melt.restart.2.bin

print "INDEPENDENT_SOURCE_MELT_BEGIN {spec['run_id']}"
run {MELT_STEPS}
restart 0
write_data prepared_liquid.lammps.data
write_dump all custom melt_validation.lammpstrj id type x y z modify sort id format line "%d %d %.9g %.9g %.9g"
write_restart melt_final.restart.bin
print "INDEPENDENT_SOURCE_MELT_COMPLETE {spec['run_id']}"
"""


def render_source_input(spec: dict[str, Any]) -> str:
    pressure_bar = PRESSURE_GPA / PRESSURE_BAR_TO_GPA
    return f"""# Independently melted undercooled source history for {spec['run_id']}.
log equilibration.lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data prepared_liquid.lammps.data

mass 1 26.9815
{_pair_commands()}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {TIMESTEP_FS / 1000.0:.12g}

velocity all create {spec['temperature_K']:.12g} {spec['velocity_seed']} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {spec['temperature_K']:.12g} {spec['temperature_K']:.12g} {THERMOSTAT_TIME_FS / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {BAROSTAT_TIME_FS / 1000.0:.12g}
thermo {SAMPLE_INTERVAL_STEPS}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes

print "INDEPENDENT_SOURCE_EQUILIBRATION_BEGIN {spec['run_id']}"
run {EQUILIBRATION_STEPS}
reset_timestep 0
log measurement.lammps.log
dump trajectory all custom {SAMPLE_INTERVAL_STEPS} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory first yes sort id format line "%d %d %.9g %.9g %.9g %.9g %.9g %.9g"
restart 25000 source.restart.1.bin source.restart.2.bin

print "INDEPENDENT_SOURCE_MEASUREMENT_BEGIN {spec['run_id']}"
run 0
run {MEASUREMENT_STEPS}
restart 0
undump trajectory
write_restart final.restart.bin
print "INDEPENDENT_SOURCE_MEASUREMENT_COMPLETE {spec['run_id']}"
"""


def _write_slurm_scripts(root: Path) -> None:
    runner = Path(__file__).resolve()
    common = f"""set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
"""
    slurm = root / "slurm"
    task = slurm / "run_source.sbatch"
    task.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_ind_source
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks={MPI_RANKS}
#SBATCH --ntasks-per-node={MPI_RANKS}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output={slurm}/%A_%a.out
#SBATCH --error={slurm}/%A_%a.err

{common}python {runner} run-task --campaign-root {root} --task-index "${{SLURM_ARRAY_TASK_ID}}"
""",
        encoding="utf-8",
    )
    task.chmod(0o750)
    controller = slurm / "submit_wave.sbatch"
    controller.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_ind_src_ctl
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output={slurm}/controller_%j.out
#SBATCH --error={slurm}/controller_%j.err

{common}: "${{SOURCE_START:?SOURCE_START must identify the next source index}}"
python {runner} submit-next-wave --campaign-root {root} --start-index "${{SOURCE_START}}"
""",
        encoding="utf-8",
    )
    controller.chmod(0o750)
    summary = slurm / "summarize.sbatch"
    summary.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_ind_src_sum
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output={slurm}/summary_%j.out
#SBATCH --error={slurm}/summary_%j.err

{common}python {runner} summarize --campaign-root {root}
""",
        encoding="utf-8",
    )
    summary.chmod(0o750)


def prepare_campaign(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Independent source campaign already exists: {root}")
    library = REPOSITORY_ROOT / "datasets/potentials/Lee2003_Al.library.meam"
    parameters = REPOSITORY_ROOT / "datasets/potentials/Lee2003_Al.meam"
    for path, expected in ((library, LIBRARY_SHA256), (parameters, PARAMETER_SHA256)):
        if _sha256_file(path) != expected:
            raise RuntimeError(f"2NN-MEAM checksum mismatch: path={path}, expected={expected}.")
    specs = source_run_specs()
    root.mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "potential").mkdir()
    (root / "slurm").mkdir()
    shutil.copy2(library, root / "potential" / library.name)
    shutil.copy2(parameters, root / "potential" / parameters.name)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((26, 26, 26))
    if len(atoms) != EXPECTED_ATOM_COUNT:
        raise RuntimeError(f"FCC producer made {len(atoms)} atoms, expected {EXPECTED_ATOM_COUNT}.")
    write(
        root / "initial_fcc.lammps.data",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=("Al",),
    )
    for spec in specs:
        run_dir = root / str(spec["run_dir"])
        run_dir.mkdir()
        (run_dir / "melt.in.lammps").write_text(render_melt_input(spec), encoding="utf-8")
        (run_dir / "source.in.lammps").write_text(render_source_input(spec), encoding="utf-8")
        _write_json_atomic(run_dir / "metadata.json", spec)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "prepared",
        "created_at": _utc_now(),
        "campaign_type": "independently_melted_boundary_parent_sources",
        "scientific_contract": {
            "structural_independence": (
                "Every source has its own 300 ps full-box melt trajectory from a unique "
                "preparation seed before its independently seeded undercooling history."
            ),
            "split_unit": "source run and every screened parent and future descended from it",
            "screening_futures_reused_for_evaluation": False,
        },
        "atom_count": EXPECTED_ATOM_COUNT,
        "temperatures_K": list(TEMPERATURES_K),
        "counts": {
            "runs": len(specs),
            "runs_per_temperature": RUNS_PER_TEMPERATURE,
            "runs_by_split": dict(Counter(str(spec["source_split"]) for spec in specs)),
        },
        "protocol": {
            "melt_temperature_K": MELT_TEMPERATURE_K,
            "melt_steps": MELT_STEPS,
            "melt_duration_ps": MELT_STEPS * TIMESTEP_FS / 1000.0,
            "equilibration_steps": EQUILIBRATION_STEPS,
            "equilibration_duration_ps": EQUILIBRATION_STEPS * TIMESTEP_FS / 1000.0,
            "measurement_steps": MEASUREMENT_STEPS,
            "measurement_duration_ps": MEASUREMENT_STEPS * TIMESTEP_FS / 1000.0,
            "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
            "sample_interval_ps": SAMPLE_INTERVAL_STEPS * TIMESTEP_FS / 1000.0,
            "timestep_fs": TIMESTEP_FS,
            "trajectory_storage_dtype": STORAGE_DTYPE,
            "trajectory_fields": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
            "ptm_rmsd_cutoff": 0.1,
            "cluster_connectivity_cutoff_A": 3.5,
            "nucleation_cluster_atoms": 100,
            "nucleation_persistence_frames": 3,
            "boundary_bands": {f"{t:g}": list(BOUNDARY_BANDS[t]) for t in TEMPERATURES_K},
        },
        "execution": {
            "partition": "CPU",
            "mpi_ranks_per_run": MPI_RANKS,
            "memory_per_run": "24G",
            "time_limit_per_run": "06:00:00",
            "array_concurrency": ARRAY_CONCURRENCY,
        },
        "potential": {
            "library_sha256": LIBRARY_SHA256,
            "parameter_sha256": PARAMETER_SHA256,
        },
        "runs": list(specs),
    }
    _write_json_atomic(root / "manifest.json", manifest)
    _write_json_atomic(
        root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "updated_at": _utc_now(),
            "complete_run_count": 0,
            "pending_run_count": len(specs),
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


def _run_lammps(run_dir: Path, input_name: str, stdout_name: str) -> float:
    lmp = Path(sys.prefix) / "bin" / "lmp"
    srun = shutil.which("srun")
    if "SLURM_JOB_ID" not in os.environ or srun is None or not lmp.is_file():
        raise RuntimeError(
            f"Independent source MD requires Slurm, srun, and pointnet LAMMPS; "
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
        input_name,
    ]
    started = time.monotonic()
    with (run_dir / stdout_name).open("wb") as stdout:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=_lammps_environment(),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed with return code {completed.returncode}: "
            f"run_dir={run_dir}, input={input_name}, log={run_dir / stdout_name}."
        )
    return elapsed


def _ptm_progress(trajectory_path: Path, expected_steps: np.ndarray) -> dict[str, np.ndarray]:
    try:
        from ovito.io import import_file
        from ovito.modifiers import (
            ClusterAnalysisModifier,
            PolyhedralTemplateMatchingModifier,
        )
    except ImportError as exc:
        raise ImportError("Source-history PTM analysis requires OVITO in pointnet.") from exc
    pipeline = import_file(str(trajectory_path), sort_particles=True)
    if int(pipeline.source.num_frames) != len(expected_steps):
        raise RuntimeError(
            f"OVITO saw {pipeline.source.num_frames} frames, expected {len(expected_steps)}: "
            f"{trajectory_path}."
        )
    fractions = np.empty((len(expected_steps), len(STRUCTURE_NAMES)), dtype=np.float32)
    cluster_counts = np.empty(len(expected_steps), dtype=np.int64)
    largest = np.empty(len(expected_steps), dtype=np.int64)
    for frame_index, expected_step in enumerate(expected_steps):
        data = pipeline.compute(frame_index)
        observed_step = int(data.attributes.get("Timestep", -1))
        if observed_step != int(expected_step) or int(data.particles.count) != EXPECTED_ATOM_COUNT:
            raise RuntimeError(
                f"PTM source frame contract changed: frame={frame_index}, "
                f"expected_step={int(expected_step)}, observed_step={observed_step}, "
                f"atoms={int(data.particles.count)}, path={trajectory_path}."
            )
        ptm = PolyhedralTemplateMatchingModifier()
        ptm.rmsd_cutoff = 0.1
        data.apply(ptm)
        structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        counts = np.bincount(structure_types, minlength=len(STRUCTURE_NAMES))[
            : len(STRUCTURE_NAMES)
        ]
        fractions[frame_index] = counts / EXPECTED_ATOM_COUNT
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
        clusters = ClusterAnalysisModifier(cutoff=3.5, only_selected=True, sort_by_size=True)
        data.apply(clusters)
        cluster_counts[frame_index] = int(data.attributes["ClusterAnalysis.cluster_count"])
        largest[frame_index] = int(data.attributes["ClusterAnalysis.largest_size"])
        if frame_index % 40 == 0 or frame_index + 1 == len(expected_steps):
            print(
                f"PTM_PROGRESS frame={frame_index + 1}/{len(expected_steps)} "
                f"step={int(expected_step)} largest_cluster={int(largest[frame_index])}",
                flush=True,
            )
    crystalline_fraction = np.sum(fractions[:, 1:4], axis=1, dtype=np.float32)
    return {
        "step": np.asarray(expected_steps, dtype=np.int64),
        "time_ps": np.asarray(expected_steps, dtype=np.float64) * TIMESTEP_FS / 1000.0,
        "structure_names": np.asarray(STRUCTURE_NAMES),
        "structure_fractions": fractions,
        "crystalline_fraction": crystalline_fraction,
        "crystalline_cluster_count": cluster_counts,
        "largest_crystalline_cluster_atoms": largest,
    }


def _first_persistent_nucleation(largest: np.ndarray) -> tuple[int | None, int | None]:
    above = largest >= 100
    for confirmation in range(2, len(above)):
        if bool(np.all(above[confirmation - 2 : confirmation + 1])):
            return confirmation - 2, confirmation
    return None, None


def run_source_task(campaign_root: str | Path, task_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or task_index < 0 or task_index >= len(runs):
        raise IndexError(f"task_index={task_index} is outside [0, {len(runs) if isinstance(runs, list) else 0}).")
    spec = runs[task_index]
    if not isinstance(spec, dict) or int(spec["run_index"]) != task_index:
        raise RuntimeError(f"Invalid source task mapping at index {task_index}: {spec!r}.")
    run_dir = root / str(spec["run_dir"])
    outcome_path = run_dir / "outcome.json"
    if outcome_path.is_file():
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Existing outcome is not complete: {outcome_path}.")
        print(f"Source {spec['run_id']} is already complete; leaving it unchanged.")
        return outcome
    partial = [name for name in RUNTIME_ARTIFACTS if (run_dir / name).exists()]
    if partial:
        raise RuntimeError(
            f"Source {spec['run_id']} has partial artifacts {partial}; archive them before retry."
        )
    allocated = int(os.environ.get("SLURM_NTASKS", "0"))
    if allocated != MPI_RANKS:
        raise RuntimeError(f"SLURM_NTASKS={allocated}, expected {MPI_RANKS} for {run_dir}.")
    status_path = run_dir / "status.json"
    _write_json_atomic(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "updated_at": _utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": os.uname().nodename,
        },
    )
    try:
        melt_elapsed = _run_lammps(run_dir, "melt.in.lammps", "melt.stdout.log")
        melt_steps, melt_positions, melt_cells = _read_lammps_dump(
            run_dir / "melt_validation.lammpstrj"
        )
        if melt_steps.tolist() != [MELT_STEPS]:
            raise RuntimeError(
                f"Melt validation step changed: observed={melt_steps.tolist()}, expected={[MELT_STEPS]}."
            )
        melt_validation = _liquid_validation(melt_positions[0], melt_cells[0])
        melt_validation.update(
            {
                "preparation_seed": int(spec["preparation_seed"]),
                "elapsed_seconds": melt_elapsed,
            }
        )
        _write_json_atomic(run_dir / "source_validation.json", melt_validation)

        source_elapsed = _run_lammps(run_dir, "source.in.lammps", "source.stdout.log")
        trajectory_path = run_dir / "trajectory.lammpstrj"
        scan = TemporalLAMMPSDumpDataset.scan_dump_file(trajectory_path)
        expected_steps = np.arange(
            0, MEASUREMENT_STEPS + 1, SAMPLE_INTERVAL_STEPS, dtype=np.int64
        )
        if scan.num_atoms != EXPECTED_ATOM_COUNT or tuple(scan.atom_columns) != (
            "id", "type", "x", "y", "z", "vx", "vy", "vz"
        ):
            raise RuntimeError(
                f"Source dump contract changed: atoms={scan.num_atoms}, "
                f"columns={scan.atom_columns}, path={trajectory_path}."
            )
        if not np.array_equal(scan.timesteps, expected_steps):
            raise RuntimeError(
                f"Source timesteps changed: observed={scan.timesteps.tolist()}, "
                f"expected={expected_steps.tolist()}, path={trajectory_path}."
            )
        progress = _ptm_progress(trajectory_path, expected_steps)
        with (run_dir / "crystallization_progress.npz").open("wb") as handle:
            np.savez(handle, **progress)

        thermo = _read_thermodynamic_log(run_dir / "measurement.lammps.log")
        missing_thermo = [int(step) for step in expected_steps if int(step) not in thermo]
        if missing_thermo:
            raise RuntimeError(
                f"Measurement thermodynamics are missing steps {missing_thermo}: {run_dir}."
            )
        thermo_values = np.asarray([thermo[int(step)] for step in expected_steps], dtype=np.float64)
        with (run_dir / "thermodynamics.npz").open("wb") as handle:
            np.savez(
                handle,
                step=expected_steps,
                temperature_K=thermo_values[:, 0].astype(np.float32),
                pressure_GPa=(thermo_values[:, 1] * PRESSURE_BAR_TO_GPA).astype(np.float32),
                volume_A3=thermo_values[:, 2],
                potential_energy_eV_per_atom=thermo_values[:, 3] / EXPECTED_ATOM_COUNT,
                number_density_atoms_per_A3=EXPECTED_ATOM_COUNT / thermo_values[:, 2],
            )

        binary_dir = run_dir / "trajectory_binary_float16"
        binary = convert_shooting_trajectory(
            trajectory_path,
            binary_dir,
            timesteps=tuple(int(step) for step in expected_steps),
            atom_count=EXPECTED_ATOM_COUNT,
            storage_dtype=STORAGE_DTYPE,
            provenance={
                "campaign_type": manifest["campaign_type"],
                "run_id": spec["run_id"],
                "structural_independence": manifest["scientific_contract"][
                    "structural_independence"
                ],
            },
        )
        binary.verify_checksums()
        for required in (run_dir / "final.restart.bin", run_dir / "melt_final.restart.bin"):
            if not required.is_file() or required.stat().st_size == 0:
                raise RuntimeError(f"Required completed source restart is missing: {required}.")

        largest = progress["largest_crystalline_cluster_atoms"]
        onset_index, confirmation_index = _first_persistent_nucleation(largest)
        band_min = int(spec["boundary_cluster_min_atoms"])
        band_max = int(spec["boundary_cluster_max_atoms"])
        band_indices = np.flatnonzero((largest >= band_min) & (largest <= band_max))
        source_size = trajectory_path.stat().st_size
        source_sha256 = _sha256_file(trajectory_path)
        trajectory_path.unlink()
        for name in (
            "melt.restart.1.bin",
            "melt.restart.2.bin",
            "source.restart.1.bin",
            "source.restart.2.bin",
        ):
            path = run_dir / name
            if path.exists():
                path.unlink()
        outcome = {
            **spec,
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "completed_at": _utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "melt_elapsed_seconds": melt_elapsed,
            "source_elapsed_seconds": source_elapsed,
            "melt_validation": melt_validation,
            "frame_count": binary.frame_count,
            "first_timestep": int(binary.timesteps[0]),
            "last_timestep": int(binary.timesteps[-1]),
            "trajectory_artifact": {
                "format": binary.manifest["format"],
                "path": str(binary_dir),
                "storage_dtype": STORAGE_DTYPE,
                "source_lammpstrj": {
                    "path": str(trajectory_path),
                    "size_bytes": source_size,
                    "sha256": source_sha256,
                    "deleted": True,
                },
            },
            "progress_artifact": {
                "path": str(run_dir / "crystallization_progress.npz"),
                "sha256": _sha256_file(run_dir / "crystallization_progress.npz"),
            },
            "thermodynamics_artifact": {
                "path": str(run_dir / "thermodynamics.npz"),
                "sha256": _sha256_file(run_dir / "thermodynamics.npz"),
            },
            "nucleation_observed": onset_index is not None,
            "nucleation_onset_step": (
                None if onset_index is None else int(expected_steps[onset_index])
            ),
            "nucleation_confirmation_step": (
                None if confirmation_index is None else int(expected_steps[confirmation_index])
            ),
            "nucleation_onset_time_ps": (
                None
                if onset_index is None
                else float(expected_steps[onset_index] * TIMESTEP_FS / 1000.0)
            ),
            "boundary_candidate_frame_count": int(len(band_indices)),
            "boundary_candidate_steps": [int(expected_steps[index]) for index in band_indices],
            "final_restart_size_bytes": (run_dir / "final.restart.bin").stat().st_size,
            "input_sha256": {
                "melt": _sha256_file(run_dir / "melt.in.lammps"),
                "source": _sha256_file(run_dir / "source.in.lammps"),
            },
        }
        _write_json_atomic(outcome_path, outcome)
        _write_json_atomic(status_path, outcome)
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return outcome
    except BaseException as error:
        _write_json_atomic(
            status_path,
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


def _active_submission_conflicts(root: Path) -> list[str]:
    path = root / "slurm" / "active_submission.json"
    if not path.is_file():
        return []
    active = _load_json(path)
    ids = [str(active["array_job_id"]), str(active["successor_job_id"])]
    queued = subprocess.run(
        ["squeue", "-h", "-j", ",".join(ids), "-o", "%A"],
        check=True,
        text=True,
        capture_output=True,
    )
    current = os.environ.get("SLURM_JOB_ID")
    return sorted(
        {line.strip() for line in queued.stdout.splitlines() if line.strip() != current}
    )


def submit_next_wave(campaign_root: str | Path, start_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or start_index < 0 or start_index >= len(runs):
        raise IndexError(f"start_index={start_index} is invalid for source campaign.")
    conflicts = _active_submission_conflicts(root)
    if conflicts:
        raise RuntimeError(f"Refusing duplicate source submission; active jobs={conflicts}.")
    stop_index = min(start_index + ARRAY_CONCURRENCY - 1, len(runs) - 1)
    array_spec = f"{start_index}-{stop_index}%{ARRAY_CONCURRENCY}"
    array_result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={array_spec}",
            str(root / "slurm" / "run_source.sbatch"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    array_job_id = array_result.stdout.strip()
    if not array_job_id.isdigit():
        raise RuntimeError(f"Invalid Slurm source array ID: {array_result.stdout!r}.")
    if stop_index + 1 < len(runs):
        successor_kind = "controller"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            f"--export=ALL,SOURCE_START={stop_index + 1}",
            str(root / "slurm" / "submit_wave.sbatch"),
        ]
    else:
        successor_kind = "summary"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            str(root / "slurm" / "summarize.sbatch"),
        ]
    successor_result = subprocess.run(
        successor_command, check=True, text=True, capture_output=True
    )
    successor_job_id = successor_result.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(f"Invalid source {successor_kind} ID: {successor_result.stdout!r}.")
    record = {
        "submitted_at": _utc_now(),
        "submitting_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_spec": array_spec,
        "array_job_id": array_job_id,
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
    }
    with (root / "slurm" / "submission_chain.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json_atomic(root / "slurm" / "active_submission.json", record)
    _write_json_atomic(
        root / "status.json",
        {"schema_version": SCHEMA_VERSION, "state": "submitted", "updated_at": _utc_now(), **record},
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def summarize_campaign(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise TypeError(f"Source manifest runs must be a list: {root / 'manifest.json'}.")
    outcomes: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in runs:
        run_dir = root / str(spec["run_dir"])
        path = run_dir / "outcome.json"
        if not path.is_file():
            missing.append(str(spec["run_id"]))
            continue
        outcome = _load_json(path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Source outcome is not complete: {path}.")
        binary = ShootingBinaryTrajectory.load(outcome["trajectory_artifact"]["path"])
        binary.verify_checksums()
        if binary.storage_dtype != np.dtype(STORAGE_DTYPE):
            raise RuntimeError(f"Source binary dtype changed: {binary.root}.")
        for key in ("progress_artifact", "thermodynamics_artifact"):
            artifact = outcome[key]
            artifact_path = Path(artifact["path"])
            if _sha256_file(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"Source artifact checksum mismatch: {artifact_path}.")
        restart = run_dir / "final.restart.bin"
        if not restart.is_file() or restart.stat().st_size == 0:
            raise RuntimeError(f"Completed source restart is missing: {restart}.")
        outcomes.append(outcome)
    if missing:
        raise RuntimeError(
            f"Cannot summarize independent sources: missing={len(missing)}, first={missing[:10]}."
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": _utc_now(),
        "run_count": len(outcomes),
        "runs_by_temperature": {
            f"{temperature:g}": sum(o["temperature_K"] == temperature for o in outcomes)
            for temperature in TEMPERATURES_K
        },
        "runs_by_split": dict(Counter(str(o["source_split"]) for o in outcomes)),
        "nucleated_run_count": sum(bool(o["nucleation_observed"]) for o in outcomes),
        "boundary_candidate_frame_count": sum(
            int(o["boundary_candidate_frame_count"]) for o in outcomes
        ),
        "boundary_candidate_frames_by_temperature": {
            f"{temperature:g}": sum(
                int(o["boundary_candidate_frame_count"])
                for o in outcomes
                if o["temperature_K"] == temperature
            )
            for temperature in TEMPERATURES_K
        },
        "trajectory_storage_dtype": STORAGE_DTYPE,
    }
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--campaign-root", required=True, type=Path)
    task = subparsers.add_parser("run-task")
    task.add_argument("--campaign-root", required=True, type=Path)
    task.add_argument("--task-index", required=True, type=int)
    submit = subparsers.add_parser("submit-next-wave")
    submit.add_argument("--campaign-root", required=True, type=Path)
    submit.add_argument("--start-index", required=True, type=int)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--campaign-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "prepare":
        manifest = prepare_campaign(args.campaign_root)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    elif args.action == "run-task":
        run_source_task(args.campaign_root, args.task_index)
    elif args.action == "submit-next-wave":
        submit_next_wave(args.campaign_root, args.start_index)
    elif args.action == "summarize":
        summarize_campaign(args.campaign_root)
    else:
        raise AssertionError(f"Unhandled action {args.action!r}.")


if __name__ == "__main__":
    main()
