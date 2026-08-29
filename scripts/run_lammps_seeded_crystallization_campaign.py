#!/usr/bin/env python3
"""Prepare and run a qualitative 70k-atom seeded Al crystallization campaign.

The source is constructed under the same Mishin EAM potential used for dynamics:
an exact FCC system is melted everywhere except for a spherical nucleus, quenched,
and validated before use.  Replicas run strictly sequentially; each LAMMPS process
uses four MPI ranks with twelve OpenMP threads per rank (all 48 physical CPU cores).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.plot_homogeneous_checkpoint import _plot_dashboard  # noqa: E402
from scripts.run_lammps_homogeneous_campaign import (  # noqa: E402
    EXPECTED_ATOM_COUNT,
    EXPECTED_POTENTIAL_SHA256,
    PRESSURE_BAR_TO_GPA,
    _read_lammps_dump,
    _read_thermodynamic_log,
    _sha256,
    _write_json_atomic,
)
from src.data_utils.synthetic.atomistic.homogeneous_analysis import (  # noqa: E402
    HomogeneousCrystallizationAnalysis,
    analyze_homogeneous_crystallization,
    write_homogeneous_progress_visualization,
    write_homogeneous_rdf_visualization,
)
from src.data_utils.synthetic.atomistic.simulation import (  # noqa: E402
    ThermodynamicTrace,
    validate_thermodynamic_trace,
)
from src.data_utils.synthetic.atomistic.transition_analysis import (  # noqa: E402
    STRUCTURE_NAMES,
    write_structure_slice_visualization,
)


SCHEMA_VERSION = 1
CHEMICAL_SYMBOL = "Al"


@dataclass(frozen=True)
class Settings:
    output_root: Path
    potential: Path
    seeds: tuple[int, ...]
    preparation_seed: int
    target_temperature_K: float
    pressure_GPa: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    lattice_constant_A: float
    repetitions: int
    nucleus_radius_A: float
    anchored_core_radius_A: float
    melt_temperature_K: float
    melt_steps: int
    quench_steps: int
    liquid_hold_steps: int
    anchor_steps: int
    measurement_steps: int
    sample_interval: int
    mpi_ranks: int
    omp_threads_per_rank: int

    @property
    def atom_count(self) -> int:
        return 4 * self.repetitions**3

    @property
    def preparation_steps(self) -> int:
        return self.melt_steps + self.quench_steps + self.liquid_hold_steps

    @property
    def replica_steps(self) -> int:
        return self.anchor_steps + self.measurement_steps


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run an EAM-native seeded Al crystallization campaign. "
            "Preparation and replicas use all 48 physical cores."
        )
    )
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--potential",
        type=Path,
        default=Path("datasets/potentials/Al99.eam.alloy"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(35803, 35831, 35839, 35851),
    )
    parser.add_argument("--preparation-seed", type=int, default=24681357)
    parser.add_argument("--target-temperature-K", type=float, default=700.0)
    parser.add_argument("--pressure-GPa", type=float, default=0.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--thermostat-time-fs", type=float, default=100.0)
    parser.add_argument("--barostat-time-fs", type=float, default=500.0)
    parser.add_argument("--lattice-constant-A", type=float, default=4.13)
    parser.add_argument("--repetitions", type=int, default=26)
    parser.add_argument("--nucleus-radius-A", type=float, default=20.0)
    parser.add_argument("--anchored-core-radius-A", type=float, default=15.0)
    parser.add_argument("--melt-temperature-K", type=float, default=1600.0)
    parser.add_argument("--melt-steps", type=int, default=30000)
    parser.add_argument("--quench-steps", type=int, default=10000)
    parser.add_argument("--liquid-hold-steps", type=int, default=5000)
    parser.add_argument("--anchor-steps", type=int, default=2000)
    parser.add_argument("--measurement-steps", type=int, default=110000)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--omp-threads-per-rank", type=int, default=12)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _settings(args: argparse.Namespace) -> Settings:
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be non-empty and unique, got {seeds}.")
    settings = Settings(
        output_root=_resolve(args.output_root),
        potential=_resolve(args.potential),
        seeds=seeds,
        preparation_seed=args.preparation_seed,
        target_temperature_K=args.target_temperature_K,
        pressure_GPa=args.pressure_GPa,
        timestep_fs=args.timestep_fs,
        thermostat_time_fs=args.thermostat_time_fs,
        barostat_time_fs=args.barostat_time_fs,
        lattice_constant_A=args.lattice_constant_A,
        repetitions=args.repetitions,
        nucleus_radius_A=args.nucleus_radius_A,
        anchored_core_radius_A=args.anchored_core_radius_A,
        melt_temperature_K=args.melt_temperature_K,
        melt_steps=args.melt_steps,
        quench_steps=args.quench_steps,
        liquid_hold_steps=args.liquid_hold_steps,
        anchor_steps=args.anchor_steps,
        measurement_steps=args.measurement_steps,
        sample_interval=args.sample_interval,
        mpi_ranks=args.mpi_ranks,
        omp_threads_per_rank=args.omp_threads_per_rank,
    )
    positive = {
        key: value
        for key, value in settings.__dict__.items()
        if key
        in {
            "target_temperature_K",
            "timestep_fs",
            "thermostat_time_fs",
            "barostat_time_fs",
            "lattice_constant_A",
            "repetitions",
            "nucleus_radius_A",
            "anchored_core_radius_A",
            "melt_temperature_K",
            "melt_steps",
            "quench_steps",
            "liquid_hold_steps",
            "anchor_steps",
            "measurement_steps",
            "sample_interval",
            "mpi_ranks",
            "omp_threads_per_rank",
        }
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Positive campaign controls required, got {invalid}.")
    if settings.atom_count != EXPECTED_ATOM_COUNT:
        raise ValueError(
            f"repetitions={settings.repetitions} creates {settings.atom_count} atoms, "
            f"but this campaign requires exactly {EXPECTED_ATOM_COUNT}."
        )
    if settings.anchored_core_radius_A >= settings.nucleus_radius_A:
        raise ValueError(
            "anchored_core_radius_A must be smaller than nucleus_radius_A, got "
            f"{settings.anchored_core_radius_A} >= {settings.nucleus_radius_A}."
        )
    for name, steps in (
        ("melt_steps", settings.melt_steps),
        ("quench_steps", settings.quench_steps),
        ("liquid_hold_steps", settings.liquid_hold_steps),
        ("anchor_steps", settings.anchor_steps),
        ("measurement_steps", settings.measurement_steps),
    ):
        if steps % settings.sample_interval != 0:
            raise ValueError(
                f"{name}={steps} must be divisible by sample_interval="
                f"{settings.sample_interval}."
            )
    if settings.mpi_ranks * settings.omp_threads_per_rank != 48:
        raise ValueError(
            "This host has 48 physical cores; require mpi_ranks * "
            f"omp_threads_per_rank == 48, got {settings.mpi_ranks} * "
            f"{settings.omp_threads_per_rank}."
        )
    if settings.melt_temperature_K <= settings.target_temperature_K:
        raise ValueError(
            f"melt_temperature_K={settings.melt_temperature_K} must exceed "
            f"target_temperature_K={settings.target_temperature_K}."
        )
    return settings


def _write_status(settings: Settings, state: str, **details: object) -> None:
    _write_json_atomic(
        settings.output_root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _lammps_environment(settings: Settings) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MPIR_CVAR_CH4_NETMOD": "ofi",
            "FI_PROVIDER": "tcp",
            "OMP_NUM_THREADS": str(settings.omp_threads_per_rank),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    environment["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + (
        f":{environment['LD_LIBRARY_PATH']}"
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    return environment


def _lammps_command(settings: Settings, input_name: str) -> list[str]:
    mpiexec = Path(sys.prefix) / "bin" / "mpiexec"
    lmp = Path(sys.prefix) / "bin" / "lmp"
    for path in (mpiexec, lmp):
        if not path.is_file():
            raise FileNotFoundError(f"Required pointnet executable is absent: {path}.")
    return [
        str(mpiexec),
        "-n",
        str(settings.mpi_ranks),
        "-bind-to",
        f"core:{settings.omp_threads_per_rank}",
        "-map-by",
        "numa",
        str(lmp),
        "-sf",
        "omp",
        "-pk",
        "omp",
        str(settings.omp_threads_per_rank),
        "-in",
        input_name,
    ]


def _run_lammps(settings: Settings, directory: Path, input_name: str) -> float:
    stdout_path = directory / "lammps.stdout.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout:
        completed = subprocess.run(
            _lammps_command(settings, input_name),
            cwd=directory,
            env=_lammps_environment(settings),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed in {directory} with return code {completed.returncode}. "
            f"Inspect {stdout_path}."
        )
    return elapsed


def _completed_md_elapsed_seconds(settings: Settings, replica_dir: Path) -> float:
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
    if completed_steps != settings.replica_steps:
        raise RuntimeError(
            f"{log_path}: completed LAMMPS loop steps total {completed_steps}, "
            f"expected {settings.replica_steps}; refusing to treat this replica as "
            "complete."
        )
    elapsed = sum(float(seconds) for seconds, _ in matches)
    _write_json_atomic(elapsed_path, {"elapsed_seconds": elapsed})
    return elapsed


def _preparation_input(settings: Settings, box_length_A: float) -> str:
    center = box_length_A / 2.0
    pressure_bar = settings.pressure_GPa / PRESSURE_BAR_TO_GPA
    return f"""# EAM-native liquid preparation around a retained FCC nucleus.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../initial_fcc.lammps.data

pair_style eam/alloy
pair_coeff * * ../potential/Al99.eam.alloy Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings.timestep_fs / 1000.0:.12g}

region nucleus sphere {center:.12g} {center:.12g} {center:.12g} {settings.nucleus_radius_A:.12g} units box
group nucleus region nucleus
group mobile subtract all nucleus
velocity all set 0.0 0.0 0.0
velocity mobile create {settings.melt_temperature_K:.12g} {settings.preparation_seed} mom yes rot no dist gaussian
fix integrate mobile nve
fix hold nucleus setforce 0.0 0.0 0.0
fix thermalize mobile langevin {settings.melt_temperature_K:.12g} {settings.melt_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} {settings.preparation_seed + 2} zero yes

thermo {settings.sample_interval}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
dump preparation all custom 5000 liquid_preparation.lammpstrj id type x y z
dump_modify preparation sort id format line \"%d %d %.9g %.9g %.9g\"

print \"EAM_NATIVE_MELT_BEGIN\"
run {settings.melt_steps}
unfix thermalize
fix thermalize mobile langevin {settings.melt_temperature_K:.12g} {settings.target_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} {settings.preparation_seed + 4} zero yes
print \"EAM_NATIVE_QUENCH_BEGIN\"
run {settings.quench_steps}
unfix thermalize
fix thermalize mobile langevin {settings.target_temperature_K:.12g} {settings.target_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} {settings.preparation_seed + 6} zero yes
print \"EAM_NATIVE_TARGET_HOLD_BEGIN pressure_target_bar={pressure_bar:.12g}\"
run {settings.liquid_hold_steps}
write_data ../seeded_source.lammps.data
print \"EAM_NATIVE_SEEDED_SOURCE_COMPLETE\"
"""


def _ptm_seed_validation(
    positions_A: np.ndarray,
    cell_A: np.ndarray,
    *,
    settings: Settings,
) -> dict[str, object]:
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import (
        ClusterAnalysisModifier,
        PolyhedralTemplateMatchingModifier,
    )

    lengths_A = np.diag(cell_A)
    wrapped = np.mod(np.asarray(positions_A, dtype=np.float64), lengths_A)
    center_A = lengths_A / 2.0
    displacement_A = wrapped - center_A
    displacement_A -= np.rint(displacement_A / lengths_A) * lengths_A
    radius_A = np.linalg.norm(displacement_A, axis=1)
    atoms = Atoms(
        CHEMICAL_SYMBOL * len(wrapped), positions=wrapped, cell=cell_A, pbc=True
    )
    data = ase_to_ovito(atoms)
    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = 0.10
    data.apply(ptm)
    structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
    crystalline = np.isin(structure_types, (1, 2, 3))
    data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
    data.apply(
        ClusterAnalysisModifier(cutoff=3.5, only_selected=True, sort_by_size=True)
    )
    core = radius_A <= settings.anchored_core_radius_A
    far_liquid = radius_A >= settings.nucleus_radius_A + 8.0
    neighbor_distance_A = cKDTree(wrapped, boxsize=lengths_A).query(
        wrapped, k=2
    )[0][:, 1]
    result = {
        "atom_count": len(wrapped),
        "ptm_rmsd_cutoff": 0.10,
        "total_crystalline_fraction": float(np.mean(crystalline)),
        "core_atom_count": int(np.sum(core)),
        "core_crystalline_fraction": float(np.mean(crystalline[core])),
        "far_liquid_atom_count": int(np.sum(far_liquid)),
        "far_liquid_crystalline_fraction": float(
            np.mean(crystalline[far_liquid])
        ),
        "largest_crystalline_cluster_atoms": int(
            data.attributes["ClusterAnalysis.largest_size"]
        ),
        "minimum_pair_distance_A": float(np.min(neighbor_distance_A)),
    }
    failures = []
    if result["core_crystalline_fraction"] < 0.85:
        failures.append("core_crystalline_fraction < 0.85")
    if result["far_liquid_crystalline_fraction"] > 0.01:
        failures.append("far_liquid_crystalline_fraction > 0.01")
    if result["largest_crystalline_cluster_atoms"] < 500:
        failures.append("largest_crystalline_cluster_atoms < 500")
    if result["minimum_pair_distance_A"] < 1.8:
        failures.append("minimum_pair_distance_A < 1.8")
    if failures:
        raise RuntimeError(
            f"Prepared seed/liquid validation failed: {failures}; metrics={result}."
        )
    return result


def prepare(settings: Settings) -> None:
    if settings.output_root.exists():
        raise FileExistsError(
            f"Output root exists and will not be overwritten: {settings.output_root}."
        )
    if _sha256(settings.potential) != EXPECTED_POTENTIAL_SHA256:
        raise RuntimeError(
            f"{settings.potential}: potential SHA-256 does not match the validated "
            "Mishin Al99 file."
        )
    settings.output_root.mkdir(parents=True)
    (settings.output_root / "potential").mkdir()
    preparation_dir = settings.output_root / "preparation"
    preparation_dir.mkdir()
    (settings.output_root / "replicas").mkdir()
    shutil.copy2(
        settings.potential, settings.output_root / "potential" / "Al99.eam.alloy"
    )

    atoms = bulk(
        CHEMICAL_SYMBOL,
        "fcc",
        a=settings.lattice_constant_A,
        cubic=True,
    ).repeat((settings.repetitions,) * 3)
    if len(atoms) != EXPECTED_ATOM_COUNT:
        raise RuntimeError(
            f"FCC producer created {len(atoms)} atoms, expected {EXPECTED_ATOM_COUNT}."
        )
    box_length_A = float(atoms.cell.lengths()[0])
    write(
        settings.output_root / "initial_fcc.lammps.data",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=(CHEMICAL_SYMBOL,),
    )
    (preparation_dir / "in.lammps").write_text(
        _preparation_input(settings, box_length_A), encoding="utf-8"
    )
    _write_json_atomic(
        settings.output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "qualitative seeded crystallization growth, not homogeneous "
                "nucleation-rate estimation"
            ),
            "atom_count": settings.atom_count,
            "potential": {
                "name": "Mishin et al. 1999 Al EAM alloy",
                "file": "potential/Al99.eam.alloy",
                "sha256": EXPECTED_POTENTIAL_SHA256,
            },
            "source_preparation": {
                "construction": (
                    "26x26x26 conventional FCC cells; retain a spherical FCC nucleus "
                    "while melting and quenching every atom outside it under EAM"
                ),
                "preparation_seed": settings.preparation_seed,
                "initial_lattice_constant_A": settings.lattice_constant_A,
                "nucleus_radius_A": settings.nucleus_radius_A,
                "melt_temperature_K": settings.melt_temperature_K,
                "melt_steps": settings.melt_steps,
                "quench_target_temperature_K": settings.target_temperature_K,
                "quench_steps": settings.quench_steps,
                "liquid_hold_steps": settings.liquid_hold_steps,
                "ensemble": "fixed-cell NVE plus Langevin thermostat on non-nucleus atoms",
            },
            "replica_protocol": {
                "seeds": list(settings.seeds),
                "target_temperature_K": settings.target_temperature_K,
                "pressure_GPa": settings.pressure_GPa,
                "timestep_fs": settings.timestep_fs,
                "anchored_core_radius_A": settings.anchored_core_radius_A,
                "anchor_steps": settings.anchor_steps,
                "measurement_steps": settings.measurement_steps,
                "sample_interval": settings.sample_interval,
                "ensemble": "Nose-Hoover NPT",
            },
            "execution": {
                "strictly_sequential_replicas": True,
                "mpi_ranks": settings.mpi_ranks,
                "openmp_threads_per_rank": settings.omp_threads_per_rank,
                "physical_cores_per_replica": (
                    settings.mpi_ranks * settings.omp_threads_per_rank
                ),
            },
        },
    )
    _write_status(settings, "preparing_eam_liquid_and_seed")
    elapsed = _run_lammps(settings, preparation_dir, "in.lammps")
    steps, positions_A, cells_A = _read_lammps_dump(
        preparation_dir / "liquid_preparation.lammpstrj"
    )
    if int(steps[-1]) != settings.preparation_steps:
        raise RuntimeError(
            f"Preparation dump ended at step {int(steps[-1])}, expected "
            f"{settings.preparation_steps}."
        )
    validation = _ptm_seed_validation(
        positions_A[-1], cells_A[-1], settings=settings
    )
    validation["preparation_elapsed_seconds"] = elapsed
    validation["preparation_steps_per_second"] = settings.preparation_steps / elapsed
    _write_json_atomic(settings.output_root / "source_validation.json", validation)
    _write_status(
        settings,
        "prepared",
        preparation_elapsed_seconds=elapsed,
        source_validation="source_validation.json",
    )
    print(
        f"Prepared and validated {settings.output_root / 'seeded_source.lammps.data'}: "
        f"{json.dumps(validation, sort_keys=True)}",
        flush=True,
    )


def _replica_name(index: int, seed: int) -> str:
    return f"replica_{index:03d}_seed_{seed}"


def _replica_input(settings: Settings, name: str, seed: int) -> str:
    box_length_A = settings.lattice_constant_A * settings.repetitions
    center = box_length_A / 2.0
    pressure_bar = settings.pressure_GPa / PRESSURE_BAR_TO_GPA
    return f"""# Sequential all-core seeded crystallization replica.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../seeded_source.lammps.data

pair_style eam/alloy
pair_coeff * * ../../potential/Al99.eam.alloy Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings.timestep_fs / 1000.0:.12g}
velocity all create {settings.target_temperature_K:.12g} {seed} mom yes rot no dist gaussian

region anchored_core sphere {center:.12g} {center:.12g} {center:.12g} {settings.anchored_core_radius_A:.12g} units box
group anchored_core region anchored_core
fix ensemble all npt temp {settings.target_temperature_K:.12g} {settings.target_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {settings.barostat_time_fs / 1000.0:.12g}
fix anchor anchored_core spring/self 5.0
fix remove_drift all momentum 100 linear 1 1 1

thermo {settings.sample_interval}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
dump trajectory all custom {settings.sample_interval} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g\"
restart 10000 restart.1.bin restart.2.bin

print \"SEEDED_REPLICA {name} SEED {seed} CORE_STABILIZATION_BEGIN\"
run {settings.anchor_steps}
unfix anchor
print \"SEEDED_REPLICA {name} SEED {seed} UNBIASED_GROWTH_BEGIN\"
run {settings.measurement_steps}
print \"SEEDED_REPLICA {name} SEED {seed} COMPLETE\"
"""


def _load_full_trace(replica_dir: Path) -> ThermodynamicTrace:
    step, positions_A, cell_vectors_A = _read_lammps_dump(
        replica_dir / "trajectory.lammpstrj"
    )
    thermo = _read_thermodynamic_log(replica_dir / "lammps.log")
    missing = [int(value) for value in step if int(value) not in thermo]
    if missing:
        raise RuntimeError(
            f"{replica_dir / 'lammps.log'}: missing thermo rows for {missing}."
        )
    values = np.asarray([thermo[int(value)] for value in step], dtype=np.float64)
    trace = ThermodynamicTrace(
        step=step,
        temperature_K=values[:, 0],
        pressure_GPa=values[:, 1] * PRESSURE_BAR_TO_GPA,
        volume_A3=values[:, 2],
        potential_energy_eV_per_atom=values[:, 3] / EXPECTED_ATOM_COUNT,
        positions_A=positions_A,
        cell_vectors_A=cell_vectors_A,
    )
    validate_thermodynamic_trace(
        trace,
        atom_count=EXPECTED_ATOM_COUNT,
        context=f"seeded LAMMPS trajectory {replica_dir}",
    )
    return trace


def _measurement_trace(full: ThermodynamicTrace, settings: Settings) -> ThermodynamicTrace:
    mask = full.step >= settings.anchor_steps
    return ThermodynamicTrace(
        step=full.step[mask] - settings.anchor_steps,
        temperature_K=full.temperature_K[mask],
        pressure_GPa=full.pressure_GPa[mask],
        volume_A3=full.volume_A3[mask],
        potential_energy_eV_per_atom=full.potential_energy_eV_per_atom[mask],
        positions_A=full.positions_A[mask],
        cell_vectors_A=full.cell_vectors_A[mask],
    )


def _write_overview(path: Path, visualizations: Path, name: str) -> None:
    panels = (
        ("thermodynamics and crystal growth", "checkpoint_dashboard.png"),
        ("PTM and connected-cluster growth", "crystallization_progress.png"),
        ("structure slices", "structure_slice.png"),
        ("total RDF", "total_rdf.png"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18.0, 13.0), constrained_layout=True)
    for axis, (title, filename) in zip(axes.flat, panels):
        axis.imshow(plt.imread(visualizations / filename))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"LAMMPS EAM seeded crystallization: {name}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _analysis_document(
    analysis: HomogeneousCrystallizationAnalysis,
    *,
    name: str,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "replica_name": name,
        "random_seed": seed,
        "interpretation": (
            "seeded crystal-growth trajectory; initial cluster is deliberately "
            "supercritical and must not be interpreted as spontaneous nucleation"
        ),
        "initial_crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "final_crystalline_fraction": float(analysis.crystalline_fraction[-1]),
        "net_crystalline_fraction_change": float(
            analysis.crystalline_fraction[-1] - analysis.crystalline_fraction[0]
        ),
        "initial_largest_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[0]
        ),
        "final_largest_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[-1]
        ),
        "maximum_cluster_atoms": int(
            np.max(analysis.largest_crystalline_cluster_atoms)
        ),
        "ptm_rmsd_cutoff": analysis.ptm_rmsd_cutoff,
    }


def analyze_replica(settings: Settings, index: int, seed: int) -> dict[str, object]:
    name = _replica_name(index, seed)
    replica_dir = settings.output_root / "replicas" / name
    full = _load_full_trace(replica_dir)
    expected = np.arange(
        0, settings.replica_steps + 1, settings.sample_interval, dtype=np.int64
    )
    if not np.array_equal(full.step, expected):
        raise RuntimeError(
            f"{replica_dir}: expected saved steps {expected.tolist()}, got "
            f"{full.step.tolist()}."
        )
    trace = _measurement_trace(full, settings)
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol=CHEMICAL_SYMBOL,
        timestep_fs=settings.timestep_fs,
        ptm_rmsd_cutoff=0.10,
        crystalline_cluster_cutoff_A=3.5,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=3,
        rdf_cutoff_A=8.0,
        rdf_bins=160,
        progress=lambda message: print(f"{name}: {message}", flush=True),
    )
    with (replica_dir / "trajectory.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            step=trace.step,
            temperature_K=trace.temperature_K,
            pressure_GPa=trace.pressure_GPa,
            volume_A3=trace.volume_A3,
            potential_energy_eV_per_atom=trace.potential_energy_eV_per_atom,
            positions_A=trace.positions_A,
            cell_vectors_A=trace.cell_vectors_A,
        )
    with (replica_dir / "crystallization_progress.npz").open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            structure_names=np.asarray(STRUCTURE_NAMES),
            structure_fractions=analysis.structure_fractions,
            crystalline_fraction=analysis.crystalline_fraction,
            crystalline_cluster_count=analysis.crystalline_cluster_count,
            largest_crystalline_cluster_atoms=analysis.largest_crystalline_cluster_atoms,
        )
    with (replica_dir / "total_rdf.npz").open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            distance_A=analysis.rdf_distance_A,
            g_r=analysis.rdf_g_r,
        )

    visualizations = replica_dir / "visualizations"
    visualizations.mkdir(exist_ok=True)
    write_homogeneous_progress_visualization(
        visualizations / "crystallization_progress.png",
        trace=trace,
        analysis=analysis,
        temperature_K=settings.target_temperature_K,
        pressure_GPa=settings.pressure_GPa,
        simulation_title="Seeded FCC crystal growth in undercooled EAM Al",
    )
    write_homogeneous_rdf_visualization(
        visualizations / "total_rdf.png",
        analysis=analysis,
        temperature_K=settings.target_temperature_K,
    )
    write_structure_slice_visualization(
        visualizations / "structure_slice.png",
        trace=trace,
        chemical_symbol=CHEMICAL_SYMBOL,
        timestep_fs=settings.timestep_fs,
        reference_planes_fractional=(),
        simulation_name=f"seeded EAM crystallization {name}",
        temperature_K=settings.target_temperature_K,
        ptm_rmsd_cutoff=0.10,
    )
    online = {
        "measurement_step": analysis.step,
        "crystalline_fraction": analysis.crystalline_fraction,
        "crystalline_cluster_count": analysis.crystalline_cluster_count,
        "largest_crystalline_cluster_atoms": (
            analysis.largest_crystalline_cluster_atoms
        ),
    }
    _plot_dashboard(
        visualizations / "checkpoint_dashboard.png",
        trace=full,
        online=online,
        checkpoint_steps=(settings.replica_steps,),
        completed_global_step=settings.replica_steps,
        replica_name=name,
        model_name="Mishin-1999-Al-EAM seeded FCC growth",
        chemical_symbol=CHEMICAL_SYMBOL,
        timestep_fs=settings.timestep_fs,
        equilibration_steps=settings.anchor_steps,
        planned_measurement_steps=settings.measurement_steps,
        sample_interval=settings.sample_interval,
        target_temperature_K=settings.target_temperature_K,
        target_pressure_GPa=settings.pressure_GPa,
        maximum_liquid_crystalline_fraction=None,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=3,
        simulation_kind="seeded crystallization-growth",
    )
    _write_overview(visualizations / "visualization_overview.png", visualizations, name)
    document = _analysis_document(analysis, name=name, seed=seed)
    document["artifacts_sha256"] = {
        "trajectory.npz": _sha256(replica_dir / "trajectory.npz"),
        "crystallization_progress.npz": _sha256(
            replica_dir / "crystallization_progress.npz"
        ),
        "total_rdf.npz": _sha256(replica_dir / "total_rdf.npz"),
        **{
            f"visualizations/{path.name}": _sha256(path)
            for path in sorted(visualizations.glob("*.png"))
        },
    }
    _write_json_atomic(replica_dir / "analysis.json", document)
    return document


def run(settings: Settings) -> None:
    required = (
        settings.output_root / "manifest.json",
        settings.output_root / "source_validation.json",
        settings.output_root / "seeded_source.lammps.data",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Campaign preparation is incomplete; missing required artifacts {missing}."
        )
    with (settings.output_root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_seeds = manifest["replica_protocol"]["seeds"]
    if expected_seeds != list(settings.seeds):
        raise RuntimeError(
            f"Prepared campaign seeds are {expected_seeds}, requested {list(settings.seeds)}."
        )
    completed: list[dict[str, object]] = []
    elapsed_seconds: dict[str, float] = {}
    for index, seed in enumerate(settings.seeds):
        name = _replica_name(index, seed)
        replica_dir = settings.output_root / "replicas" / name
        analysis_path = replica_dir / "analysis.json"
        if analysis_path.is_file():
            with analysis_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if document.get("replica_name") != name or document.get("random_seed") != seed:
                raise RuntimeError(
                    f"{analysis_path}: identity does not match replica {name}, seed {seed}."
                )
            elapsed_seconds[name] = _completed_md_elapsed_seconds(
                settings, replica_dir
            )
            completed.append(document)
            print(f"{name}: already complete; keeping existing artifacts", flush=True)
            continue

        if replica_dir.exists():
            elapsed_seconds[name] = _completed_md_elapsed_seconds(
                settings, replica_dir
            )
            print(
                f"{name}: MD is complete; resuming at analysis and visualizations",
                flush=True,
            )
        else:
            replica_dir.mkdir()
            (replica_dir / "in.lammps").write_text(
                _replica_input(settings, name, seed), encoding="utf-8"
            )
            _write_status(
                settings,
                "running_replica",
                active_replica=name,
                replica_index=index,
                completed_replicas=[item["replica_name"] for item in completed],
            )
            print(
                f"{name}: starting sequential MD on all 48 physical cores", flush=True
            )
            elapsed_seconds[name] = _run_lammps(
                settings, replica_dir, "in.lammps"
            )
            _write_json_atomic(
                replica_dir / "md_elapsed_seconds.json",
                {"elapsed_seconds": elapsed_seconds[name]},
            )
        _write_status(
            settings,
            "analyzing_replica",
            active_replica=name,
            completed_replicas=[item["replica_name"] for item in completed],
        )
        document = analyze_replica(settings, index, seed)
        completed.append(document)
        print(f"{name}: MD, analysis, and visualizations complete", flush=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "strictly_sequential_replicas": True,
        "physical_cores_per_replica": 48,
        "md_elapsed_seconds": elapsed_seconds,
        "steps_per_second": {
            name: settings.replica_steps / elapsed
            for name, elapsed in elapsed_seconds.items()
        },
        "replicas": completed,
    }
    _write_json_atomic(settings.output_root / "campaign_summary.json", summary)
    _write_status(
        settings,
        "complete",
        campaign_summary="campaign_summary.json",
        md_elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    args = _arguments()
    settings = _settings(args)
    try:
        if args.action == "prepare":
            prepare(settings)
        else:
            run(settings)
    except BaseException:
        if settings.output_root.is_dir():
            _write_status(settings, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
