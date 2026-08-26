#!/usr/bin/env python3
"""Run a small, fast LAMMPS ensemble from the repository's 70k-atom Al liquid.

The four MD replicas run concurrently.  Full PTM/cluster/RDF analysis is deferred
until all replicas finish, matching the existing MACE campaign's separation between
MD and expensive structural analysis.
"""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.plot_homogeneous_checkpoint import _plot_dashboard  # noqa: E402
from src.data_utils.synthetic.atomistic.homogeneous_analysis import (  # noqa: E402
    HomogeneousCrystallizationAnalysis,
    ReplicaObservation,
    analyze_homogeneous_crystallization,
    analyze_replica_survival,
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


EXPECTED_ATOM_COUNT = 70_304
EXPECTED_POTENTIAL_SHA256 = (
    "60c8a085be79d273324ab421f5b1447578fef55c1acfc6492c0999f15ee8a284"
)
PRESSURE_BAR_TO_GPA = 1.0e-4
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CampaignSettings:
    output_root: Path
    source_trajectory: Path
    source_frame_step: int
    potential: Path
    seeds: tuple[int, ...]
    temperature_K: float
    pressure_GPa: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    equilibration_steps: int
    measurement_steps: int
    sample_interval: int
    threads_per_replica: int
    cpu_sets: tuple[str, ...]

    @property
    def total_steps(self) -> int:
        return self.equilibration_steps + self.measurement_steps


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent 70,304-atom Al crystallization replicas with the Mishin "
            "EAM potential, then create the same structural visualizations as the MACE "
            "homogeneous-crystallization workflow."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--source-trajectory",
        type=Path,
        default=Path(
            "output/synthetic_data/al_liquid_source_70304_mpa_seed12345_500K/"
            "replica_000_bulk_liquid/trajectory.npz"
        ),
    )
    parser.add_argument("--source-frame-step", type=int, default=3000)
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
    parser.add_argument("--temperature-K", type=float, default=500.0)
    parser.add_argument("--pressure-GPa", type=float, default=0.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--thermostat-time-fs", type=float, default=100.0)
    parser.add_argument("--barostat-time-fs", type=float, default=500.0)
    parser.add_argument("--equilibration-steps", type=int, default=5000)
    parser.add_argument("--measurement-steps", type=int, default=110000)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--threads-per-replica", type=int, default=12)
    parser.add_argument(
        "--cpu-sets",
        nargs="+",
        default=("0-11", "12-23", "24-35", "36-47"),
        help="One taskset CPU list per replica.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create inputs but do not start LAMMPS.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _settings(args: argparse.Namespace) -> CampaignSettings:
    seeds = tuple(args.seeds)
    cpu_sets = tuple(args.cpu_sets)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must be a non-empty unique list, got {seeds}.")
    if len(cpu_sets) != len(seeds):
        raise ValueError(
            f"--cpu-sets must contain one entry per seed; got {len(cpu_sets)} CPU "
            f"sets for {len(seeds)} seeds."
        )
    positive_values = {
        "temperature_K": args.temperature_K,
        "timestep_fs": args.timestep_fs,
        "thermostat_time_fs": args.thermostat_time_fs,
        "barostat_time_fs": args.barostat_time_fs,
        "equilibration_steps": args.equilibration_steps,
        "measurement_steps": args.measurement_steps,
        "sample_interval": args.sample_interval,
        "threads_per_replica": args.threads_per_replica,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Campaign controls must be positive, got {invalid}.")
    if args.equilibration_steps % args.sample_interval != 0:
        raise ValueError(
            f"equilibration_steps={args.equilibration_steps} must be divisible by "
            f"sample_interval={args.sample_interval}."
        )
    if args.measurement_steps % args.sample_interval != 0:
        raise ValueError(
            f"measurement_steps={args.measurement_steps} must be divisible by "
            f"sample_interval={args.sample_interval}."
        )
    return CampaignSettings(
        output_root=_resolve(args.output_root),
        source_trajectory=_resolve(args.source_trajectory),
        source_frame_step=args.source_frame_step,
        potential=_resolve(args.potential),
        seeds=seeds,
        temperature_K=args.temperature_K,
        pressure_GPa=args.pressure_GPa,
        timestep_fs=args.timestep_fs,
        thermostat_time_fs=args.thermostat_time_fs,
        barostat_time_fs=args.barostat_time_fs,
        equilibration_steps=args.equilibration_steps,
        measurement_steps=args.measurement_steps,
        sample_interval=args.sample_interval,
        threads_per_replica=args.threads_per_replica,
        cpu_sets=cpu_sets,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_status(settings: CampaignSettings, state: str, **details: object) -> None:
    _write_json_atomic(
        settings.output_root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _load_source(settings: CampaignSettings) -> Atoms:
    if not settings.source_trajectory.is_file():
        raise FileNotFoundError(
            f"Required repository liquid trajectory is absent: "
            f"{settings.source_trajectory}."
        )
    with np.load(settings.source_trajectory) as stored:
        matches = np.flatnonzero(stored["step"] == settings.source_frame_step)
        if len(matches) != 1:
            raise RuntimeError(
                f"{settings.source_trajectory}: expected exactly one frame at step "
                f"{settings.source_frame_step}, found {matches.tolist()}."
            )
        frame = int(matches[0])
        positions_A = np.asarray(stored["positions_A"][frame], dtype=np.float64)
        cell_A = np.asarray(stored["cell_vectors_A"][frame], dtype=np.float64)
    if positions_A.shape != (EXPECTED_ATOM_COUNT, 3):
        raise RuntimeError(
            f"{settings.source_trajectory}: expected positions shape "
            f"({EXPECTED_ATOM_COUNT}, 3), got {positions_A.shape}."
        )
    off_diagonal = cell_A.copy()
    off_diagonal[np.diag_indices(3)] = 0.0
    if np.any(off_diagonal != 0.0):
        raise RuntimeError(
            f"{settings.source_trajectory}: LAMMPS campaign expects the repository's "
            f"orthogonal liquid box; got cell={cell_A.tolist()}."
        )
    return Atoms("Al" * EXPECTED_ATOM_COUNT, positions=positions_A, cell=cell_A, pbc=True)


def _lammps_input(settings: CampaignSettings, replica_name: str, seed: int) -> str:
    pressure_bar = settings.pressure_GPa / PRESSURE_BAR_TO_GPA
    return f"""# Generated by scripts/run_lammps_homogeneous_campaign.py
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../source.lammps.data

mass 1 {atomic_masses[atomic_numbers['Al']]:.12g}
pair_style eam/alloy
pair_coeff * * ../../potential/Al99.eam.alloy Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {settings.timestep_fs / 1000.0:.12g}
velocity all create {settings.temperature_K:.12g} {seed} mom yes rot no dist gaussian
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {settings.temperature_K:.12g} {settings.temperature_K:.12g} {settings.thermostat_time_fs / 1000.0:.12g} iso {pressure_bar:.12g} {pressure_bar:.12g} {settings.barostat_time_fs / 1000.0:.12g}

thermo {settings.sample_interval}
thermo_style custom step temp press vol pe
thermo_modify format float %.16g flush yes
dump trajectory all custom {settings.sample_interval} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g\"
restart 10000 restart.1.bin restart.2.bin

print \"CAMPAIGN_REPLICA {replica_name} SEED {seed}\"
run {settings.total_steps}
print \"CAMPAIGN_COMPLETE {replica_name} SEED {seed}\"
"""


def _replica_name(index: int, seed: int) -> str:
    return f"replica_{index:03d}_seed_{seed}"


def prepare_campaign(settings: CampaignSettings) -> None:
    if settings.output_root.exists():
        raise FileExistsError(
            f"Campaign output already exists and will not be overwritten: "
            f"{settings.output_root}."
        )
    potential_sha256 = _sha256(settings.potential)
    if potential_sha256 != EXPECTED_POTENTIAL_SHA256:
        raise RuntimeError(
            f"{settings.potential}: expected Mishin Al99 EAM SHA-256 "
            f"{EXPECTED_POTENTIAL_SHA256}, got {potential_sha256}."
        )
    settings.output_root.mkdir(parents=True)
    (settings.output_root / "potential").mkdir()
    (settings.output_root / "replicas").mkdir()
    source_atoms = _load_source(settings)
    write(
        settings.output_root / "source.lammps.data",
        source_atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=("Al",),
    )
    shutil.copy2(
        settings.potential,
        settings.output_root / "potential" / "Al99.eam.alloy",
    )

    replicas = []
    for index, (seed, cpu_set) in enumerate(zip(settings.seeds, settings.cpu_sets)):
        name = _replica_name(index, seed)
        replica_dir = settings.output_root / "replicas" / name
        replica_dir.mkdir()
        (replica_dir / "in.lammps").write_text(
            _lammps_input(settings, name, seed), encoding="utf-8"
        )
        replicas.append(
            {
                "replica_name": name,
                "random_seed": seed,
                "cpu_set": cpu_set,
            }
        )

    _write_json_atomic(
        settings.output_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "qualitative fast crystallization ensemble; classical EAM results are "
                "not numerically interchangeable with the MACE trajectories"
            ),
            "source": {
                "trajectory": str(settings.source_trajectory),
                "trajectory_sha256": _sha256(settings.source_trajectory),
                "frame_step": settings.source_frame_step,
                "shared_coordinate_configuration": True,
                "atom_count": EXPECTED_ATOM_COUNT,
            },
            "potential": {
                "name": "Mishin et al. 1999 Al EAM alloy",
                "file": "potential/Al99.eam.alloy",
                "sha256": potential_sha256,
                "pair_style": "eam/alloy/omp via the LAMMPS OPENMP suffix",
                "source_url": (
                    "https://www.ctcms.nist.gov/potentials/entry/"
                    "1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al/"
                ),
            },
            "protocol": {
                "ensemble": "LAMMPS Nose-Hoover NPT",
                "temperature_K": settings.temperature_K,
                "pressure_GPa": settings.pressure_GPa,
                "timestep_fs": settings.timestep_fs,
                "thermostat_time_fs": settings.thermostat_time_fs,
                "barostat_time_fs": settings.barostat_time_fs,
                "equilibration_steps": settings.equilibration_steps,
                "measurement_steps": settings.measurement_steps,
                "sample_interval": settings.sample_interval,
            },
            "execution": {
                "threads_per_replica": settings.threads_per_replica,
                "concurrent_replicas": len(settings.seeds),
                "deferred_structural_analysis": True,
            },
            "replicas": replicas,
        },
    )
    _write_status(settings, "prepared", replicas=replicas)


def _lammps_environment(settings: CampaignSettings) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(settings.threads_per_replica)
    environment["OMP_PROC_BIND"] = "close"
    environment["OMP_PLACES"] = "cores"
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + (
        f":{environment['LD_LIBRARY_PATH']}"
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    return environment


def run_lammps_replicas(settings: CampaignSettings) -> dict[str, float]:
    lmp = Path(sys.prefix) / "bin" / "lmp"
    if not lmp.is_file():
        raise FileNotFoundError(
            f"LAMMPS executable is absent from the active environment: {lmp}."
        )
    environment = _lammps_environment(settings)
    processes: list[tuple[str, subprocess.Popen[bytes], object, float]] = []
    for index, (seed, cpu_set) in enumerate(zip(settings.seeds, settings.cpu_sets)):
        name = _replica_name(index, seed)
        replica_dir = settings.output_root / "replicas" / name
        stdout_path = replica_dir / "lammps.stdout.log"
        stdout = stdout_path.open("wb")
        command = [
            "taskset",
            "-c",
            cpu_set,
            str(lmp),
            "-sf",
            "omp",
            "-pk",
            "omp",
            str(settings.threads_per_replica),
            "-in",
            "in.lammps",
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=replica_dir,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((name, process, stdout, started))

    _write_status(
        settings,
        "running_md",
        replicas=[
            {"replica_name": name, "pid": process.pid}
            for name, process, _, _ in processes
        ],
    )
    failures: dict[str, int] = {}
    elapsed_seconds: dict[str, float] = {}
    for name, process, stdout, started in processes:
        return_code = process.wait()
        stdout.close()
        elapsed_seconds[name] = time.monotonic() - started
        if return_code != 0:
            failures[name] = return_code
    if failures:
        raise RuntimeError(
            f"LAMMPS replicas failed with return codes {failures}. Inspect each "
            f"replica's lammps.stdout.log under {settings.output_root / 'replicas'}."
        )
    return elapsed_seconds


def _read_lammps_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: list[int] = []
    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    expected_ids = np.arange(1, EXPECTED_ATOM_COUNT + 1, dtype=np.int64)
    with path.open("r", encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker != "ITEM: TIMESTEP\n":
                raise RuntimeError(
                    f"{path}: expected 'ITEM: TIMESTEP', got {marker.rstrip()!r}."
                )
            step = int(handle.readline())
            if handle.readline() != "ITEM: NUMBER OF ATOMS\n":
                raise RuntimeError(f"{path}: missing atom-count header at step {step}.")
            atom_count = int(handle.readline())
            if atom_count != EXPECTED_ATOM_COUNT:
                raise RuntimeError(
                    f"{path}: step {step} has {atom_count} atoms, expected "
                    f"{EXPECTED_ATOM_COUNT}."
                )
            bounds_header = handle.readline()
            if bounds_header != "ITEM: BOX BOUNDS pp pp pp\n":
                raise RuntimeError(
                    f"{path}: unsupported box header at step {step}: "
                    f"{bounds_header.rstrip()!r}."
                )
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()] for _ in range(3)],
                dtype=np.float64,
            )
            atom_header = handle.readline()
            if atom_header != "ITEM: ATOMS id type x y z\n":
                raise RuntimeError(
                    f"{path}: unsupported atom columns at step {step}: "
                    f"{atom_header.rstrip()!r}."
                )
            table = np.loadtxt(handle, max_rows=atom_count)
            if table.shape != (atom_count, 5):
                raise RuntimeError(
                    f"{path}: step {step} atom table has shape {table.shape}, expected "
                    f"({atom_count}, 5)."
                )
            ids = table[:, 0].astype(np.int64)
            if not np.array_equal(ids, expected_ids):
                raise RuntimeError(
                    f"{path}: atom IDs are not the exact sorted sequence 1..{atom_count} "
                    f"at step {step}."
                )
            box_lengths = bounds[:, 1] - bounds[:, 0]
            frame_positions = table[:, 2:5] - bounds[:, 0][None, :]
            steps.append(step)
            positions.append(frame_positions.astype(np.float32))
            cells.append(np.diag(box_lengths))
    if not steps:
        raise RuntimeError(f"{path}: trajectory contains no frames.")
    return (
        np.asarray(steps, dtype=np.int64),
        np.stack(positions),
        np.stack(cells),
    )


def _read_thermodynamic_log(path: Path) -> dict[int, tuple[float, float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_fields = ("Step", "Temp", "Press", "Volume", "PotEng")
    header_indices = [
        index for index, line in enumerate(lines) if tuple(line.split()) == header_fields
    ]
    if not header_indices:
        raise RuntimeError(
            f"{path}: did not find the configured thermo columns {header_fields}."
        )
    start = header_indices[-1] + 1
    rows: dict[int, tuple[float, float, float, float]] = {}
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("Loop time of"):
            break
        fields = stripped.split()
        if len(fields) != 5:
            continue
        try:
            step = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            continue
        rows[step] = values  # type: ignore[assignment]
    if not rows:
        raise RuntimeError(f"{path}: thermo block contains no numeric samples.")
    return rows


def _load_full_trace(replica_dir: Path) -> ThermodynamicTrace:
    step, positions_A, cell_vectors_A = _read_lammps_dump(
        replica_dir / "trajectory.lammpstrj"
    )
    thermo = _read_thermodynamic_log(replica_dir / "lammps.log")
    missing = [int(value) for value in step if int(value) not in thermo]
    if missing:
        raise RuntimeError(
            f"{replica_dir / 'lammps.log'}: missing thermodynamic rows for dump "
            f"steps {missing}."
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
        context=f"LAMMPS trajectory {replica_dir}",
    )
    return trace


def _measurement_trace(
    full_trace: ThermodynamicTrace, settings: CampaignSettings
) -> ThermodynamicTrace:
    mask = full_trace.step >= settings.equilibration_steps
    return ThermodynamicTrace(
        step=full_trace.step[mask] - settings.equilibration_steps,
        temperature_K=full_trace.temperature_K[mask],
        pressure_GPa=full_trace.pressure_GPa[mask],
        volume_A3=full_trace.volume_A3[mask],
        potential_energy_eV_per_atom=full_trace.potential_energy_eV_per_atom[mask],
        positions_A=full_trace.positions_A[mask],
        cell_vectors_A=full_trace.cell_vectors_A[mask],
    )


def _analysis_document(
    analysis: HomogeneousCrystallizationAnalysis,
    *,
    replica_name: str,
    random_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "replica_name": replica_name,
        "random_seed": random_seed,
        "nucleation_observed": analysis.nucleation_observed,
        "nucleation_step": analysis.nucleation_step,
        "nucleation_time_ps": analysis.nucleation_time_ps,
        "confirmation_step": analysis.confirmation_step,
        "confirmation_time_ps": analysis.confirmation_time_ps,
        "initial_crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "final_crystalline_fraction": float(analysis.crystalline_fraction[-1]),
        "maximum_crystalline_fraction": float(np.max(analysis.crystalline_fraction)),
        "maximum_cluster_atoms": int(
            np.max(analysis.largest_crystalline_cluster_atoms)
        ),
        "ptm_rmsd_cutoff": analysis.ptm_rmsd_cutoff,
        "nucleus_size_threshold_atoms": analysis.nucleus_size_threshold_atoms,
        "threshold_persistence_frames": analysis.threshold_persistence_frames,
    }


def _write_overview(path: Path, visualization_dir: Path, replica_name: str) -> None:
    images = (
        ("thermodynamics and crystallinity", "checkpoint_dashboard.png"),
        ("crystallization progress", "crystallization_progress.png"),
        ("structure slices", "structure_slice.png"),
        ("total RDF", "total_rdf.png"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18.0, 13.0), constrained_layout=True)
    for axis, (title, filename) in zip(axes.flat, images):
        axis.imshow(plt.imread(visualization_dir / filename))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"LAMMPS EAM homogeneous crystallization: {replica_name}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_replica(
    settings: CampaignSettings, index: int, seed: int
) -> ReplicaObservation:
    name = _replica_name(index, seed)
    replica_dir = settings.output_root / "replicas" / name
    print(f"{name}: loading LAMMPS dump", flush=True)
    full_trace = _load_full_trace(replica_dir)
    expected_steps = np.arange(
        0, settings.total_steps + 1, settings.sample_interval, dtype=np.int64
    )
    if not np.array_equal(full_trace.step, expected_steps):
        raise RuntimeError(
            f"{replica_dir}: expected saved steps {expected_steps.tolist()}, got "
            f"{full_trace.step.tolist()}."
        )
    trace = _measurement_trace(full_trace, settings)
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol="Al",
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
            largest_crystalline_cluster_atoms=(
                analysis.largest_crystalline_cluster_atoms
            ),
        )
    with (replica_dir / "total_rdf.npz").open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            distance_A=analysis.rdf_distance_A,
            g_r=analysis.rdf_g_r,
        )

    visualization_dir = replica_dir / "visualizations"
    visualization_dir.mkdir()
    write_homogeneous_progress_visualization(
        visualization_dir / "crystallization_progress.png",
        trace=trace,
        analysis=analysis,
        temperature_K=settings.temperature_K,
        pressure_GPa=settings.pressure_GPa,
    )
    write_homogeneous_rdf_visualization(
        visualization_dir / "total_rdf.png",
        analysis=analysis,
        temperature_K=settings.temperature_K,
    )
    write_structure_slice_visualization(
        visualization_dir / "structure_slice.png",
        trace=trace,
        chemical_symbol="Al",
        timestep_fs=settings.timestep_fs,
        reference_planes_fractional=(),
        simulation_name=f"LAMMPS EAM homogeneous crystallization {name}",
        temperature_K=settings.temperature_K,
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
        visualization_dir / "checkpoint_dashboard.png",
        trace=full_trace,
        online=online,
        checkpoint_steps=(settings.total_steps,),
        completed_global_step=settings.total_steps,
        replica_name=name,
        model_name="Mishin-1999-Al-EAM",
        chemical_symbol="Al",
        timestep_fs=settings.timestep_fs,
        equilibration_steps=settings.equilibration_steps,
        planned_measurement_steps=settings.measurement_steps,
        sample_interval=settings.sample_interval,
        target_temperature_K=settings.temperature_K,
        target_pressure_GPa=settings.pressure_GPa,
        maximum_liquid_crystalline_fraction=0.15,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=3,
    )
    _write_overview(
        visualization_dir / "visualization_overview.png", visualization_dir, name
    )
    visualizations = {}
    for path in sorted(visualization_dir.glob("*.png")):
        visualizations[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json_atomic(
        visualization_dir / "visualization_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "replica_name": name,
            "visualizations": visualizations,
        },
    )
    document = _analysis_document(
        analysis, replica_name=name, random_seed=seed
    )
    document["artifacts_sha256"] = {
        "trajectory.npz": _sha256(replica_dir / "trajectory.npz"),
        "crystallization_progress.npz": _sha256(
            replica_dir / "crystallization_progress.npz"
        ),
        "total_rdf.npz": _sha256(replica_dir / "total_rdf.npz"),
    }
    _write_json_atomic(replica_dir / "analysis.json", document)
    observation_time_ps = (
        float(analysis.nucleation_time_ps)
        if analysis.nucleation_observed
        else settings.measurement_steps * settings.timestep_fs / 1000.0
    )
    print(f"{name}: analysis and visualizations complete", flush=True)
    return ReplicaObservation(
        replica_name=name,
        random_seed=seed,
        event_observed=analysis.nucleation_observed,
        observation_time_ps=observation_time_ps,
    )


def analyze_campaign(
    settings: CampaignSettings, elapsed_seconds: dict[str, float]
) -> None:
    _write_status(settings, "analyzing", md_elapsed_seconds=elapsed_seconds)
    observations = tuple(
        analyze_replica(settings, index, seed)
        for index, seed in enumerate(settings.seeds)
    )
    survival = analyze_replica_survival(observations)
    _write_json_atomic(
        settings.output_root / "campaign_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "md_elapsed_seconds": elapsed_seconds,
            "steps_per_second": {
                name: settings.total_steps / elapsed
                for name, elapsed in elapsed_seconds.items()
            },
            "survival_analysis": survival.to_dict(),
        },
    )
    with (settings.output_root / "survival_curve.npz").open("wb") as handle:
        np.savez(
            handle,
            time_ps=survival.time_ps,
            replicas_at_risk=survival.replicas_at_risk,
            events=survival.events,
            censored=survival.censored,
            survival_probability=survival.survival_probability,
        )
    _write_status(
        settings,
        "complete",
        md_elapsed_seconds=elapsed_seconds,
        campaign_summary="campaign_summary.json",
    )


def main() -> None:
    args = _arguments()
    settings = _settings(args)
    try:
        prepare_campaign(settings)
        if args.prepare_only:
            print(f"Prepared {settings.output_root}", flush=True)
            return
        elapsed_seconds = run_lammps_replicas(settings)
        analyze_campaign(settings, elapsed_seconds)
        print(f"Campaign complete: {settings.output_root}", flush=True)
    except BaseException:
        if settings.output_root.is_dir():
            _write_status(settings, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
