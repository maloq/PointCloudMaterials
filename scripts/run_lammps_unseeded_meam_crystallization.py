#!/usr/bin/env python3
"""Run qualitative homogeneous crystallization of 70k-atom Al with 2NN-MEAM.

The complete periodic FCC box is first melted at 1325 K.  The validated liquid is
then instantaneously undercooled to 500 K and evolved without crystalline seeds,
positional restraints, surfaces, or impurities.  A numeric random seed is used only
to generate Maxwell-Boltzmann velocities.
"""

from __future__ import annotations

import argparse
import json
import os
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
from ase.build import bulk
from ase.io import write


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.plot_homogeneous_checkpoint import _plot_dashboard  # noqa: E402
from scripts.run_lammps_homogeneous_campaign import (  # noqa: E402
    EXPECTED_ATOM_COUNT,
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
    library_potential: Path
    parameter_potential: Path
    velocity_seed: int
    preparation_seed: int
    target_temperature_K: float
    melt_temperature_K: float
    pressure_GPa: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    lattice_constant_A: float
    repetitions: int
    melt_steps: int
    equilibration_steps: int
    measurement_steps: int
    sample_interval: int
    mpi_ranks: int

    @property
    def atom_count(self) -> int:
        return 4 * self.repetitions**3

    @property
    def total_production_steps(self) -> int:
        return self.equilibration_steps + self.measurement_steps


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melt and spontaneously crystallize unseeded 70k-atom Al."
    )
    parser.add_argument("action", choices=("prepare", "run", "all"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--library-potential",
        type=Path,
        default=Path("datasets/potentials/Lee2003_Al.library.meam"),
    )
    parser.add_argument(
        "--parameter-potential",
        type=Path,
        default=Path("datasets/potentials/Lee2003_Al.meam"),
    )
    parser.add_argument("--velocity-seed", type=int, default=35803)
    parser.add_argument("--preparation-seed", type=int, default=24681357)
    parser.add_argument("--target-temperature-K", type=float, default=500.0)
    parser.add_argument("--melt-temperature-K", type=float, default=1325.0)
    parser.add_argument("--pressure-GPa", type=float, default=0.0)
    parser.add_argument("--timestep-fs", type=float, default=3.0)
    parser.add_argument("--thermostat-time-fs", type=float, default=300.0)
    parser.add_argument("--barostat-time-fs", type=float, default=3000.0)
    parser.add_argument("--lattice-constant-A", type=float, default=4.05)
    parser.add_argument("--repetitions", type=int, default=26)
    parser.add_argument("--melt-steps", type=int, default=100000)
    parser.add_argument("--equilibration-steps", type=int, default=5000)
    parser.add_argument("--measurement-steps", type=int, default=333000)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--mpi-ranks", type=int, default=48)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings(
        output_root=_resolve(args.output_root),
        library_potential=_resolve(args.library_potential),
        parameter_potential=_resolve(args.parameter_potential),
        velocity_seed=args.velocity_seed,
        preparation_seed=args.preparation_seed,
        target_temperature_K=args.target_temperature_K,
        melt_temperature_K=args.melt_temperature_K,
        pressure_GPa=args.pressure_GPa,
        timestep_fs=args.timestep_fs,
        thermostat_time_fs=args.thermostat_time_fs,
        barostat_time_fs=args.barostat_time_fs,
        lattice_constant_A=args.lattice_constant_A,
        repetitions=args.repetitions,
        melt_steps=args.melt_steps,
        equilibration_steps=args.equilibration_steps,
        measurement_steps=args.measurement_steps,
        sample_interval=args.sample_interval,
        mpi_ranks=args.mpi_ranks,
    )
    if settings.atom_count != EXPECTED_ATOM_COUNT:
        raise ValueError(
            f"repetitions={settings.repetitions} creates {settings.atom_count} atoms, "
            f"expected {EXPECTED_ATOM_COUNT}."
        )
    positive = {
        "target_temperature_K": settings.target_temperature_K,
        "melt_temperature_K": settings.melt_temperature_K,
        "timestep_fs": settings.timestep_fs,
        "thermostat_time_fs": settings.thermostat_time_fs,
        "barostat_time_fs": settings.barostat_time_fs,
        "lattice_constant_A": settings.lattice_constant_A,
        "melt_steps": settings.melt_steps,
        "equilibration_steps": settings.equilibration_steps,
        "measurement_steps": settings.measurement_steps,
        "sample_interval": settings.sample_interval,
        "mpi_ranks": settings.mpi_ranks,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Positive simulation controls required, got {invalid}.")
    if settings.melt_temperature_K <= settings.target_temperature_K:
        raise ValueError(
            f"melt_temperature_K={settings.melt_temperature_K} must exceed "
            f"target_temperature_K={settings.target_temperature_K}."
        )
    for name, steps in (
        ("melt_steps", settings.melt_steps),
        ("equilibration_steps", settings.equilibration_steps),
        ("measurement_steps", settings.measurement_steps),
    ):
        if steps % settings.sample_interval != 0:
            raise ValueError(
                f"{name}={steps} must be divisible by sample_interval="
                f"{settings.sample_interval}."
            )
    if settings.mpi_ranks != 48:
        raise ValueError(
            f"This all-core MEAM run requires exactly 48 MPI ranks, got "
            f"{settings.mpi_ranks}."
        )
    for path in (settings.library_potential, settings.parameter_potential):
        if not path.is_file():
            raise FileNotFoundError(f"Required 2NN-MEAM potential file is absent: {path}.")
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
        "core:1",
        "-map-by",
        "numa",
        str(lmp),
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
            env=_lammps_environment(),
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


def _pair_commands(library: str, parameters: str) -> str:
    return f"""pair_style meam
pair_coeff * * {library} Al {parameters} Al"""


def _preparation_input(settings: Settings) -> str:
    pressure_bar = settings.pressure_GPa / PRESSURE_BAR_TO_GPA
    return f"""# Fully melt every atom; no retained crystalline seed.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../initial_fcc.lammps.data

mass 1 26.9815
{_pair_commands('../potential/Lee2003_Al.library.meam', '../potential/Lee2003_Al.meam')}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings.timestep_fs / 1000.0:.12g}

velocity all create {settings.melt_temperature_K:.12g} {settings.preparation_seed} mom yes rot no dist gaussian
fix remove_drift all momentum 100 linear 1 1 1
fix melt all npt temp {settings.melt_temperature_K:.12g} {settings.melt_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {settings.barostat_time_fs / 1000.0:.12g}

thermo {settings.sample_interval}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
dump validation all custom {settings.melt_steps} melt_validation.lammpstrj id type x y z
dump_modify validation sort id format line \"%d %d %.9g %.9g %.9g\"
restart 25000 restart.1.bin restart.2.bin

print \"UNSEEDED_FULL_MELT_BEGIN\"
run {settings.melt_steps}
write_data ../prepared_liquid.lammps.data
print \"UNSEEDED_FULL_MELT_COMPLETE\"
"""


def _production_input(settings: Settings) -> str:
    pressure_bar = settings.pressure_GPa / PRESSURE_BAR_TO_GPA
    return f"""# Homogeneous crystallization from a fully melted periodic liquid.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../prepared_liquid.lammps.data

mass 1 26.9815
{_pair_commands('../../potential/Lee2003_Al.library.meam', '../../potential/Lee2003_Al.meam')}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings.timestep_fs / 1000.0:.12g}

# This numeric seed changes velocities only; it does not create a crystal seed.
velocity all create {settings.target_temperature_K:.12g} {settings.velocity_seed} mom yes rot no dist gaussian
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {settings.target_temperature_K:.12g} {settings.target_temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {settings.barostat_time_fs / 1000.0:.12g}

thermo {settings.sample_interval}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
restart 25000 restart.1.bin restart.2.bin

print \"UNSEEDED_UNDERCOOLING_EQUILIBRATION_BEGIN\"
run {settings.equilibration_steps}
dump trajectory all custom {settings.sample_interval} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g\"
print \"UNSEEDED_HOMOGENEOUS_MEASUREMENT_BEGIN\"
run {settings.measurement_steps}
print \"UNSEEDED_HOMOGENEOUS_MEASUREMENT_COMPLETE\"
"""


def _single_frame_trace(positions_A: np.ndarray, cell_A: np.ndarray) -> ThermodynamicTrace:
    volume_A3 = float(np.linalg.det(cell_A))
    return ThermodynamicTrace(
        step=np.asarray([0], dtype=np.int64),
        temperature_K=np.asarray([0.0]),
        pressure_GPa=np.asarray([0.0]),
        volume_A3=np.asarray([volume_A3]),
        potential_energy_eV_per_atom=np.asarray([0.0]),
        positions_A=positions_A[None].astype(np.float32),
        cell_vectors_A=cell_A[None].astype(np.float64),
    )


def _liquid_validation(
    positions_A: np.ndarray, cell_A: np.ndarray
) -> dict[str, object]:
    trace = _single_frame_trace(positions_A, cell_A)
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol=CHEMICAL_SYMBOL,
        timestep_fs=1.0,
        ptm_rmsd_cutoff=0.10,
        crystalline_cluster_cutoff_A=3.5,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=1,
        rdf_cutoff_A=8.0,
        rdf_bins=160,
        progress=lambda message: print(f"melt_validation: {message}", flush=True),
    )
    result = {
        "atom_count": int(positions_A.shape[0]),
        "crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "largest_crystalline_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[0]
        ),
        "cell_vectors_A": cell_A.tolist(),
    }
    if result["crystalline_fraction"] >= 0.01:
        raise RuntimeError(
            "The fully melted source failed the <1% crystalline-fraction requirement: "
            f"{result}."
        )
    if result["largest_crystalline_cluster_atoms"] >= 100:
        raise RuntimeError(
            "The fully melted source already contains a threshold-sized crystalline "
            f"cluster: {result}."
        )
    return result


def prepare(settings: Settings) -> None:
    if settings.output_root.exists():
        raise FileExistsError(
            f"Output root exists and will not be overwritten: {settings.output_root}."
        )
    settings.output_root.mkdir(parents=True)
    (settings.output_root / "potential").mkdir()
    preparation_dir = settings.output_root / "preparation"
    preparation_dir.mkdir()
    (settings.output_root / "replicas").mkdir()
    shutil.copy2(
        settings.library_potential,
        settings.output_root / "potential" / "Lee2003_Al.library.meam",
    )
    shutil.copy2(
        settings.parameter_potential,
        settings.output_root / "potential" / "Lee2003_Al.meam",
    )

    atoms = bulk(
        CHEMICAL_SYMBOL, "fcc", a=settings.lattice_constant_A, cubic=True
    ).repeat((settings.repetitions,) * 3)
    if len(atoms) != EXPECTED_ATOM_COUNT:
        raise RuntimeError(
            f"FCC producer created {len(atoms)} atoms, expected {EXPECTED_ATOM_COUNT}."
        )
    write(
        settings.output_root / "initial_fcc.lammps.data",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=(CHEMICAL_SYMBOL,),
    )
    (preparation_dir / "in.lammps").write_text(
        _preparation_input(settings), encoding="utf-8"
    )
    _write_json_atomic(
        settings.output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "qualitative spontaneous homogeneous crystallization",
            "atom_count": settings.atom_count,
            "crystal_seed": None,
            "surfaces": None,
            "impurities": None,
            "boundary_conditions": "periodic in x, y, and z",
            "potential": {
                "name": "Lee-Shim-Baskes 2003 Al 2NN-MEAM",
                "library_file": "potential/Lee2003_Al.library.meam",
                "library_sha256": _sha256(settings.library_potential),
                "parameter_file": "potential/Lee2003_Al.meam",
                "parameter_sha256": _sha256(settings.parameter_potential),
                "source": (
                    "NIST Interatomic Potentials Repository entry "
                    "2003--Lee-B-J-Shim-J-H-Baskes-M-I--Al"
                ),
            },
            "protocol": {
                "melt_temperature_K": settings.melt_temperature_K,
                "melt_steps": settings.melt_steps,
                "melt_duration_ps": settings.melt_steps
                * settings.timestep_fs
                / 1000.0,
                "target_temperature_K": settings.target_temperature_K,
                "pressure_GPa": settings.pressure_GPa,
                "equilibration_steps": settings.equilibration_steps,
                "measurement_steps": settings.measurement_steps,
                "measurement_duration_ps": settings.measurement_steps
                * settings.timestep_fs
                / 1000.0,
                "timestep_fs": settings.timestep_fs,
                "velocity_seed": settings.velocity_seed,
                "velocity_seed_interpretation": (
                    "Maxwell-Boltzmann momenta only; not a crystalline seed"
                ),
            },
            "execution": {
                "mpi_ranks": settings.mpi_ranks,
                "physical_cores": 48,
            },
        },
    )
    _write_status(settings, "melting_full_box")
    print("preparation: melting every atom at 1325 K on 48 cores", flush=True)
    elapsed = _run_lammps(settings, preparation_dir, "in.lammps")
    steps, positions_A, cells_A = _read_lammps_dump(
        preparation_dir / "melt_validation.lammpstrj"
    )
    if int(steps[-1]) != settings.melt_steps:
        raise RuntimeError(
            f"Melt validation trajectory ends at step {int(steps[-1])}, expected "
            f"{settings.melt_steps}."
        )
    validation = _liquid_validation(positions_A[-1], cells_A[-1])
    validation["melt_elapsed_seconds"] = elapsed
    _write_json_atomic(settings.output_root / "source_validation.json", validation)
    _write_status(
        settings,
        "prepared",
        source_validation="source_validation.json",
        melt_elapsed_seconds=elapsed,
    )
    print(f"preparation: validated fully liquid source: {validation}", flush=True)


def _load_full_trace(settings: Settings, replica_dir: Path) -> ThermodynamicTrace:
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
        context=f"unseeded 2NN-MEAM trajectory {replica_dir}",
    )
    expected = np.arange(
        settings.equilibration_steps,
        settings.total_production_steps + 1,
        settings.sample_interval,
        dtype=np.int64,
    )
    if not np.array_equal(trace.step, expected):
        raise RuntimeError(
            f"{replica_dir}: expected saved steps {expected.tolist()}, got "
            f"{trace.step.tolist()}."
        )
    return trace


def _measurement_trace(settings: Settings, full: ThermodynamicTrace) -> ThermodynamicTrace:
    return ThermodynamicTrace(
        step=full.step - settings.equilibration_steps,
        temperature_K=full.temperature_K,
        pressure_GPa=full.pressure_GPa,
        volume_A3=full.volume_A3,
        potential_energy_eV_per_atom=full.potential_energy_eV_per_atom,
        positions_A=full.positions_A,
        cell_vectors_A=full.cell_vectors_A,
    )


def _write_overview(path: Path, visualizations: Path) -> None:
    panels = (
        ("thermodynamics and crystallization", "checkpoint_dashboard.png"),
        ("PTM and connected-cluster evolution", "crystallization_progress.png"),
        ("structure slices", "structure_slice.png"),
        ("total RDF", "total_rdf.png"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18.0, 13.0), constrained_layout=True)
    for axis, (title, filename) in zip(axes.flat, panels):
        axis.imshow(plt.imread(visualizations / filename))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Unseeded homogeneous crystallization of 70,304-atom Al")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _analysis_document(
    analysis: HomogeneousCrystallizationAnalysis,
    *,
    settings: Settings,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "replica_name": f"replica_000_velocity_{settings.velocity_seed}",
        "velocity_random_seed": settings.velocity_seed,
        "crystal_seed": None,
        "interpretation": (
            "spontaneous homogeneous crystallization from a validated fully melted "
            "periodic liquid; no retained crystal, restraint, surface, or impurity"
        ),
        "initial_crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "final_crystalline_fraction": float(analysis.crystalline_fraction[-1]),
        "initial_largest_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[0]
        ),
        "final_largest_cluster_atoms": int(
            analysis.largest_crystalline_cluster_atoms[-1]
        ),
        "maximum_cluster_atoms": int(
            np.max(analysis.largest_crystalline_cluster_atoms)
        ),
        "nucleation_observed": analysis.nucleation_observed,
        "nucleation_step": analysis.nucleation_step,
        "nucleation_time_ps": analysis.nucleation_time_ps,
        "confirmation_step": analysis.confirmation_step,
        "confirmation_time_ps": analysis.confirmation_time_ps,
        "nucleus_size_threshold_atoms": analysis.nucleus_size_threshold_atoms,
        "threshold_persistence_frames": analysis.threshold_persistence_frames,
        "ptm_rmsd_cutoff": analysis.ptm_rmsd_cutoff,
    }


def analyze(settings: Settings, replica_dir: Path) -> dict[str, object]:
    full = _load_full_trace(settings, replica_dir)
    trace = _measurement_trace(settings, full)
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
        progress=lambda message: print(f"analysis: {message}", flush=True),
    )
    if analysis.crystalline_fraction[0] >= 0.01:
        raise RuntimeError(
            "The measurement starts from a non-liquid configuration: initial "
            f"crystalline fraction={analysis.crystalline_fraction[0]:.6f}."
        )
    if analysis.largest_crystalline_cluster_atoms[0] >= 100:
        raise RuntimeError(
            "The measurement starts with a threshold-sized crystal: initial largest "
            f"cluster={analysis.largest_crystalline_cluster_atoms[0]} atoms."
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
    visualizations.mkdir()
    write_homogeneous_progress_visualization(
        visualizations / "crystallization_progress.png",
        trace=trace,
        analysis=analysis,
        temperature_K=settings.target_temperature_K,
        pressure_GPa=settings.pressure_GPa,
        simulation_title="Spontaneous homogeneous crystallization in 2NN-MEAM Al",
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
        simulation_name="unseeded homogeneous 2NN-MEAM Al",
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
        trace=trace,
        online=online,
        checkpoint_steps=(settings.measurement_steps,),
        completed_global_step=settings.measurement_steps,
        replica_name=f"replica_000_velocity_{settings.velocity_seed}",
        model_name="Lee-Shim-Baskes-2003-Al-2NN-MEAM",
        chemical_symbol=CHEMICAL_SYMBOL,
        timestep_fs=settings.timestep_fs,
        equilibration_steps=0,
        planned_measurement_steps=settings.measurement_steps,
        sample_interval=settings.sample_interval,
        target_temperature_K=settings.target_temperature_K,
        target_pressure_GPa=settings.pressure_GPa,
        maximum_liquid_crystalline_fraction=0.01,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=3,
        simulation_kind="unseeded homogeneous crystallization",
    )
    _write_overview(
        visualizations / "visualization_overview.png", visualizations
    )
    document = _analysis_document(analysis, settings=settings)
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
        settings.output_root / "prepared_liquid.lammps.data",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Unseeded liquid preparation is incomplete; missing {missing}."
        )
    replica_name = f"replica_000_velocity_{settings.velocity_seed}"
    replica_dir = settings.output_root / "replicas" / replica_name
    if replica_dir.exists():
        raise FileExistsError(
            f"Replica output exists and will not be overwritten: {replica_dir}."
        )
    replica_dir.mkdir()
    (replica_dir / "in.lammps").write_text(
        _production_input(settings), encoding="utf-8"
    )
    _write_status(
        settings,
        "running_unseeded_production",
        active_replica=replica_name,
        physical_cores=48,
    )
    print(
        f"{replica_name}: starting 999 ps unseeded measurement on 48 cores",
        flush=True,
    )
    elapsed = _run_lammps(settings, replica_dir, "in.lammps")
    _write_json_atomic(
        replica_dir / "md_elapsed_seconds.json", {"elapsed_seconds": elapsed}
    )
    _write_status(
        settings,
        "analyzing",
        active_replica=replica_name,
        md_elapsed_seconds=elapsed,
    )
    document = analyze(settings, replica_dir)
    _write_json_atomic(
        settings.output_root / "campaign_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "md_elapsed_seconds": elapsed,
            "steps_per_second": settings.total_production_steps / elapsed,
            "analysis": document,
        },
    )
    _write_status(
        settings,
        "complete",
        campaign_summary="campaign_summary.json",
        nucleation_observed=document["nucleation_observed"],
        nucleation_time_ps=document["nucleation_time_ps"],
    )


def main() -> None:
    args = _arguments()
    settings = _settings(args)
    try:
        if args.action in ("prepare", "all"):
            prepare(settings)
        if args.action in ("run", "all"):
            run(settings)
    except BaseException:
        if settings.output_root.is_dir():
            _write_status(settings, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
