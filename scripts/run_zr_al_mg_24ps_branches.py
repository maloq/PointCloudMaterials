#!/usr/bin/env python3
"""Run the six archived thermal snapshots of Zr, Al, and Mg for 24 ps.

The campaign contains 18 sequential LAMMPS branches: one branch for each of
the six repository-owned ``initial_configurations`` of each material.  Every
branch uses isotropic Nose-Hoover NPT at the material's published isothermal
crystallization temperature and zero external pressure.

The source dumps contain positions and periodic boxes, but no velocities or
Nose-Hoover extended state.  These trajectories are therefore new
position-conditioned paths and not exact continuations of the source MD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
POTENTIAL_ROOT = REPOSITORY_ROOT / "datasets/potentials"
EXPECTED_COLUMNS = ("id", "type", "x", "y", "z")
SCHEMA_VERSION = 1
DURATION_PS = 24.0
SAMPLE_INTERVAL_PS = 0.1
MPI_RANKS = 32
TARGET_PRESSURE_BAR = 0.0
LAMMPS_ERROR_RE = re.compile(r"^ERROR(?: on proc \d+)?:", flags=re.MULTILINE)
LOOP_TIME_RE = re.compile(r"Loop time of ([0-9.eE+-]+) on \d+ procs for")


@dataclass(frozen=True)
class Source:
    relative_path: str
    timestep: int


@dataclass(frozen=True)
class Material:
    symbol: str
    mass: float
    temperature_K: float
    timestep_fs: float
    atom_count: int
    sources: dict[str, Source]
    potential_files: tuple[str, ...]
    potential_sha256: tuple[str, ...]
    pair_commands: tuple[str, ...]
    measured_steps_per_second: float
    potential_provenance: str


MATERIALS = {
    "Al": Material(
        symbol="Al",
        mass=26.981_538_5,
        temperature_K=650.0,
        timestep_fs=1.0,
        atom_count=1_048_576,
        sources={
            label: Source(
                f"datasets/Al/inherent_configurations_off/initial_configurations/{label}.pos",
                timestep,
            )
            for label, timestep in (
                ("166ps", 1_660_000),
                ("170ps", 1_700_000),
                ("174ps", 1_740_000),
                ("175ps", 1_750_000),
                ("177ps", 1_770_000),
                ("240ps", 2_400_000),
            )
        },
        potential_files=("Al1.eam.fs",),
        potential_sha256=(
            "768a9ad9b0cda57f36523b5d247942130101b26b0cbbd9d30c7bd7e1decc7ae3",
        ),
        pair_commands=("pair_style eam/fs", "pair_coeff * * {potential_0} Al"),
        measured_steps_per_second=9.640,
        potential_provenance=(
            "Mendelev, Kramer, Becker, and Asta, Philosophical Magazine 88, "
            "1723-1750 (2008), DOI 10.1080/14786430802206482; NIST Al1.eam.fs."
        ),
    ),
    "Mg": Material(
        symbol="Mg",
        mass=24.305,
        temperature_K=600.0,
        timestep_fs=1.0,
        atom_count=1_048_576,
        sources={
            label: Source(f"datasets/Mg/initial_configurations/{label}.pos", timestep)
            for label, timestep in (
                ("940ps", 940_000),
                ("960ps", 960_000),
                ("980ps", 980_000),
                ("990ps", 990_000),
                ("1000ps", 1_000_000),
                ("1500ps", 1_500_000),
            )
        },
        potential_files=("Mg1.eam.fs",),
        potential_sha256=(
            "0ceed5387f16d0cb7f4a2088fc4665e27708dc28e3e5b3a6c2e01ac0528faba2",
        ),
        pair_commands=("pair_style eam/fs", "pair_coeff * * {potential_0} Mg"),
        measured_steps_per_second=9.839,
        potential_provenance=(
            "Wilson and Mendelev, Journal of Chemical Physics 144, 144707 "
            "(2016), DOI 10.1063/1.4946032; NIST Mg1.eam.fs."
        ),
    ),
    "Zr": Material(
        symbol="Zr",
        mass=91.224,
        temperature_K=1250.0,
        timestep_fs=2.0,
        atom_count=1_024_000,
        sources={
            label: Source(f"datasets/Zr/initial_configurations/{label}.pos", timestep)
            for label, timestep in (
                ("160ps", 700_000),
                ("200ps", 720_000),
                ("240ps", 740_000),
                ("280ps", 760_000),
                ("310ps", 775_000),
                ("1560ps", 1_100_000),
            )
        },
        potential_files=("Becker2020_Zr.library.meam", "Becker2020_Zr.meam"),
        potential_sha256=(
            "576bae0feb53fee85ecb95d73c62c7c682d81a70acb766b8bfd08985ae180871",
            "aa676488bc8118477d6597ee664d601617396dc32b6b637446a4f44d4e28e778",
        ),
        pair_commands=(
            "pair_style meam/c",
            "pair_coeff * * {potential_0} Zr {potential_1} Zr",
        ),
        measured_steps_per_second=1.428,
        potential_provenance=(
            "2NN-MEAM reconstructed from Becker et al., Physical Review B 102, "
            "104205 (2020), DOI 10.1103/PhysRevB.102.104205. The original "
            "machine-readable potential was unavailable."
        ),
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(document).__name__}.")
    return document


def _seed(material: str, snapshot: str, stream: str) -> int:
    digest = hashlib.sha256(f"pcm-24ps-npt:{material}:{snapshot}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 899_999_999 + 1


def _read_source_header(path: Path, material: Material, source: Source) -> list[list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        header = [handle.readline().rstrip("\r\n") for _ in range(9)]
    if header[0] != "ITEM: TIMESTEP" or int(header[1]) != source.timestep:
        raise ValueError(
            f"Unexpected timestep in {path}: expected={source.timestep}, header={header[:2]!r}."
        )
    if header[2] != "ITEM: NUMBER OF ATOMS" or int(header[3]) != material.atom_count:
        raise ValueError(
            f"Unexpected atom count in {path}: expected={material.atom_count}, "
            f"header={header[2:4]!r}."
        )
    if header[4] != "ITEM: BOX BOUNDS pp pp pp":
        raise ValueError(f"Expected periodic orthorhombic bounds in {path}, got {header[4]!r}.")
    bounds = [[float(value) for value in line.split()] for line in header[5:8]]
    if any(len(axis) != 2 or axis[1] <= axis[0] for axis in bounds):
        raise ValueError(f"Invalid box bounds in {path}: {bounds!r}.")
    if tuple(header[8].split()[2:]) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected atom columns in {path}: expected={EXPECTED_COLUMNS}, got={header[8]!r}."
        )
    return bounds


def _exact_steps(duration_ps: float, timestep_fs: float) -> int:
    value = duration_ps * 1000.0 / timestep_fs
    rounded = round(value)
    if abs(value - rounded) > 1.0e-9:
        raise ValueError(
            f"Duration {duration_ps} ps is not an exact multiple of timestep {timestep_fs} fs."
        )
    return rounded


def _render_lammps_input(
    material: Material,
    source_path: Path,
    source: Source,
    campaign_potentials: tuple[Path, ...],
    velocity_seed: int,
    duration_ps: float = DURATION_PS,
    sample_interval_ps: float = SAMPLE_INTERVAL_PS,
) -> str:
    timestep_ps = material.timestep_fs / 1000.0
    run_steps = _exact_steps(duration_ps, material.timestep_fs)
    sample_steps = _exact_steps(sample_interval_ps, material.timestep_fs)
    thermostat_damping_ps = 100.0 * timestep_ps
    barostat_damping_ps = 1000.0 * timestep_ps
    replacements = {
        f"potential_{index}": str(path.resolve())
        for index, path in enumerate(campaign_potentials)
    }
    pair_text = "\n".join(command.format(**replacements) for command in material.pair_commands)
    return f"""# 24 ps position-conditioned {material.symbol} dynamics.
# New ambient-pressure NPT path; not an exact continuation of the source trajectory.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic

region placeholder block 0 1 0 1 0 1 units box
create_box 1 placeholder
read_dump {source_path.resolve()} {source.timestep} x y z box yes add yes replace no trim no scaled no wrapped yes
reset_timestep 0

mass 1 {material.mass:.12g}
{pair_text}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {timestep_ps:.12g}
velocity all create {material.temperature_K:.12g} {velocity_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix ensemble all npt temp {material.temperature_K:.12g} {material.temperature_K:.12g} {thermostat_damping_ps:.12g} iso {TARGET_PRESSURE_BAR:.12g} {TARGET_PRESSURE_BAR:.12g} {barostat_damping_ps:.12g}

thermo {sample_steps}
thermo_style custom step time atoms temp press lx ly lz pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {sample_steps} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line "%d %d %.9g %.9g %.9g"

print "{material.symbol.upper()}_24PS_BRANCH_BEGIN"
run {run_steps}
write_restart final.restart.bin
print "{material.symbol.upper()}_24PS_BRANCH_COMPLETE"
"""


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MPIR_CVAR_CH4_NETMOD": "ofi",
            "FI_PROVIDER": "tcp",
        }
    )
    prefix_library = str(Path(sys.prefix) / "lib")
    environment["LD_LIBRARY_PATH"] = prefix_library + (
        f":{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else ""
    )
    return environment


def _executables() -> tuple[Path, Path]:
    lammps = Path(sys.prefix) / "bin/lmp"
    mpiexec = Path(sys.prefix) / "bin/mpiexec"
    missing = [str(path) for path in (lammps, mpiexec) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing LAMMPS/MPICH executables in active environment: {missing}. "
            f"Run with {REPOSITORY_ROOT / '.venv/bin/python'}."
        )
    return lammps, mpiexec


def prepare(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Campaign root already exists: {output_root}.")
    lammps, mpiexec = _executables()
    temporary_root = output_root.with_name(f".{output_root.name}.preparing")
    if temporary_root.exists():
        raise FileExistsError(f"Stale preparation directory exists: {temporary_root}.")

    branch_records: list[dict[str, object]] = []
    estimated_total_bytes = 0
    estimated_total_seconds = 0.0
    try:
        (temporary_root / "potential").mkdir(parents=True)
        for material in MATERIALS.values():
            for filename, expected_hash in zip(
                material.potential_files, material.potential_sha256, strict=True
            ):
                source_potential = POTENTIAL_ROOT / filename
                if not source_potential.is_file():
                    raise FileNotFoundError(f"Required potential is missing: {source_potential}.")
                observed_hash = _sha256(source_potential)
                if observed_hash != expected_hash:
                    raise ValueError(
                        f"Potential hash mismatch for {source_potential}: "
                        f"expected={expected_hash}, observed={observed_hash}."
                    )
                shutil.copy2(source_potential, temporary_root / "potential" / filename)

        for material in MATERIALS.values():
            run_steps = _exact_steps(DURATION_PS, material.timestep_fs)
            sample_steps = _exact_steps(SAMPLE_INTERVAL_PS, material.timestep_fs)
            expected_frames = run_steps // sample_steps + 1
            campaign_potentials = tuple(
                output_root / "potential" / filename for filename in material.potential_files
            )
            for snapshot, source in material.sources.items():
                source_path = REPOSITORY_ROOT / source.relative_path
                if not source_path.is_file():
                    raise FileNotFoundError(f"Required source snapshot is missing: {source_path}.")
                bounds = _read_source_header(source_path, material, source)
                branch_dir = temporary_root / "branches" / material.symbol / snapshot
                branch_dir.mkdir(parents=True)
                velocity_seed = _seed(material.symbol, snapshot, "velocity")
                (branch_dir / "in.lammps").write_text(
                    _render_lammps_input(
                        material,
                        source_path,
                        source,
                        campaign_potentials,
                        velocity_seed,
                    ),
                    encoding="utf-8",
                )
                estimated_frame_bytes = material.atom_count * 43 + 4096
                estimated_trajectory_bytes = estimated_frame_bytes * expected_frames
                estimated_total_bytes += estimated_trajectory_bytes + material.atom_count * 96
                estimated_seconds = run_steps / material.measured_steps_per_second
                estimated_total_seconds += estimated_seconds
                record: dict[str, object] = {
                    "schema_version": SCHEMA_VERSION,
                    "state": "prepared",
                    "material": material.symbol,
                    "snapshot": snapshot,
                    "source": str(source_path.resolve()),
                    "source_timestep": source.timestep,
                    "source_sha256": _sha256(source_path),
                    "source_box_bounds_A": bounds,
                    "atom_count": material.atom_count,
                    "temperature_K": material.temperature_K,
                    "timestep_fs": material.timestep_fs,
                    "duration_ps": DURATION_PS,
                    "duration_steps": run_steps,
                    "sample_interval_ps": SAMPLE_INTERVAL_PS,
                    "sample_interval_steps": sample_steps,
                    "expected_frames": expected_frames,
                    "velocity_seed": velocity_seed,
                    "estimated_trajectory_bytes": estimated_trajectory_bytes,
                    "estimated_runtime_seconds": estimated_seconds,
                    "prepared_at_utc": _utc_now(),
                }
                _write_json_atomic(branch_dir / "metadata.json", record)
                branch_records.append(record)

        free_bytes = shutil.disk_usage(output_root.parent).free
        required_bytes = int(estimated_total_bytes * 1.08)
        if free_bytes < required_bytes:
            raise OSError(
                f"Insufficient disk for 18 branches: free={free_bytes}, "
                f"estimated_with_8_percent_margin={required_bytes}, output={output_root}."
            )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "prepared_at_utc": _utc_now(),
            "output_root": str(output_root),
            "branch_count": len(branch_records),
            "material_order": list(MATERIALS),
            "protocol": {
                "ensemble": "isotropic Nose-Hoover NPT",
                "pressure_bar": TARGET_PRESSURE_BAR,
                "duration_ps": DURATION_PS,
                "sample_interval_ps": SAMPLE_INTERVAL_PS,
                "mpi_ranks": MPI_RANKS,
                "execution": "sequential branches",
                "material_temperature_K": {
                    material.symbol: material.temperature_K for material in MATERIALS.values()
                },
                "material_timestep_fs": {
                    material.symbol: material.timestep_fs for material in MATERIALS.values()
                },
                "thermostat_damping_timesteps": 100,
                "barostat_damping_timesteps": 1000,
            },
            "scientific_scope": (
                "New position-conditioned paths from all six archived thermal coordinate "
                "snapshots per material. Source velocities and Nose-Hoover state are absent; "
                "these are not exact restarts."
            ),
            "protocol_provenance": (
                "Becker et al., Scientific Reports 12, 3195 (2022), DOI "
                "10.1038/s41598-022-06963-5, and the preceding Zr study, Physical Review E "
                "105, 045304 (2022), DOI 10.1103/PhysRevE.105.045304."
            ),
            "known_zr_compatibility_limit": (
                "The reconstructed Zr potential gives a large negative initial pressure for "
                "the archived boxes. Zr branches will contain barostat relaxation and must "
                "remain labeled provisional."
            ),
            "potentials": {
                material.symbol: {
                    "files": list(material.potential_files),
                    "sha256": list(material.potential_sha256),
                    "provenance": material.potential_provenance,
                }
                for material in MATERIALS.values()
            },
            "execution": {
                "lammps": str(lammps),
                "mpiexec": str(mpiexec),
                "mpi_ranks": MPI_RANKS,
            },
            "estimates": {
                "trajectory_and_restart_bytes": estimated_total_bytes,
                "required_bytes_with_margin": required_bytes,
                "free_bytes_at_prepare": free_bytes,
                "runtime_seconds": estimated_total_seconds,
                "runtime_hours": estimated_total_seconds / 3600.0,
                "basis": "Measured full-system CPU rates on this computer.",
            },
            "branches": [
                {"material": record["material"], "snapshot": record["snapshot"]}
                for record in branch_records
            ],
        }
        _write_json_atomic(temporary_root / "manifest.json", manifest)
        temporary_root.replace(output_root)
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


def _scan_trajectory(path: Path, metadata: dict[str, object]) -> dict[str, object]:
    expected_steps = list(
        range(
            0,
            int(metadata["duration_steps"]) + 1,
            int(metadata["sample_interval_steps"]),
        )
    )
    expected_atoms = int(metadata["atom_count"])
    marker = b"ITEM: TIMESTEP\n"
    observed_steps: list[int] = []
    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            raise RuntimeError(f"Trajectory is empty: {path}.")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            position = 0
            while True:
                found = data.find(marker, position)
                if found < 0:
                    break
                cursor = found
                lines: list[str] = []
                for _ in range(9):
                    line_end = data.find(b"\n", cursor)
                    if line_end < 0:
                        raise RuntimeError(
                            f"Truncated trajectory header at byte {cursor} in {path}."
                        )
                    lines.append(data[cursor:line_end].decode("ascii").rstrip("\r"))
                    cursor = line_end + 1
                if lines[0] != "ITEM: TIMESTEP" or lines[2] != "ITEM: NUMBER OF ATOMS":
                    raise RuntimeError(f"Malformed frame header at byte {found} in {path}.")
                if int(lines[3]) != expected_atoms:
                    raise RuntimeError(
                        f"Atom count changed at step {lines[1]} in {path}: "
                        f"expected={expected_atoms}, observed={lines[3]}."
                    )
                if lines[4] != "ITEM: BOX BOUNDS pp pp pp":
                    raise RuntimeError(f"Boundary contract changed in {path}: {lines[4]!r}.")
                if tuple(lines[8].split()[2:]) != EXPECTED_COLUMNS:
                    raise RuntimeError(f"Column contract changed in {path}: {lines[8]!r}.")
                observed_steps.append(int(lines[1]))
                position = cursor
            if data[-1:] != b"\n":
                raise RuntimeError(f"Trajectory does not end with a complete line: {path}.")
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"Trajectory timestep contract failed for {path}: expected_count={len(expected_steps)}, "
            f"observed_count={len(observed_steps)}, observed_tail={observed_steps[-5:]}."
        )
    return {
        "frame_count": len(observed_steps),
        "first_timestep": observed_steps[0],
        "last_timestep": observed_steps[-1],
        "atom_count": expected_atoms,
        "columns": list(EXPECTED_COLUMNS),
    }


def preflight(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("state") != "prepared":
        raise RuntimeError(
            f"Preflight requires prepared campaign state, got {manifest.get('state')!r}."
        )
    lammps, mpiexec = _executables()
    results: list[dict[str, object]] = []
    for material in MATERIALS.values():
        snapshot = next(iter(material.sources))
        source = material.sources[snapshot]
        source_path = REPOSITORY_ROOT / source.relative_path
        campaign_potentials = tuple(
            output_root / "potential" / filename for filename in material.potential_files
        )
        with tempfile.TemporaryDirectory(prefix=f"pcm-{material.symbol.lower()}-npt-preflight-") as raw:
            workdir = Path(raw)
            input_path = workdir / "in.lammps"
            input_path.write_text(
                _render_lammps_input(
                    material,
                    source_path,
                    source,
                    campaign_potentials,
                    _seed(material.symbol, snapshot, "velocity"),
                    duration_ps=0.1,
                    sample_interval_ps=0.1,
                ),
                encoding="utf-8",
            )
            command = [str(mpiexec), "-n", str(MPI_RANKS), str(lammps), "-in", "in.lammps"]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed_seconds = time.monotonic() - started
            output = completed.stdout
            marker = f"{material.symbol.upper()}_24PS_BRANCH_COMPLETE"
            if completed.returncode != 0 or LAMMPS_ERROR_RE.search(output) or marker not in output:
                errors = [line for line in output.splitlines() if line.startswith("ERROR")]
                raise RuntimeError(
                    f"{material.symbol} NPT preflight failed: return_code={completed.returncode}, "
                    f"completion_marker={marker in output}, errors={errors[-5:]}, "
                    f"output_tail={output[-4000:]!r}."
                )
            metadata = {
                "duration_steps": _exact_steps(0.1, material.timestep_fs),
                "sample_interval_steps": _exact_steps(0.1, material.timestep_fs),
                "atom_count": material.atom_count,
            }
            scan = _scan_trajectory(workdir / "trajectory.lammpstrj", metadata)
            thermo_rows: list[dict[str, float | int]] = []
            for line in output.splitlines():
                fields = line.split()
                if len(fields) != 11:
                    continue
                try:
                    row = {
                        "step": int(fields[0]),
                        "time_ps": float(fields[1]),
                        "temperature_K": float(fields[3]),
                        "pressure_bar": float(fields[4]),
                        "box_length_x_A": float(fields[5]),
                    }
                except ValueError:
                    continue
                if 0 <= row["step"] <= int(metadata["duration_steps"]):
                    thermo_rows.append(row)
            if len(thermo_rows) < 2:
                raise RuntimeError(
                    f"Could not parse initial/final thermo rows from {material.symbol} preflight."
                )
            dangerous_match = re.search(r"Dangerous builds = (\d+)", output)
            if dangerous_match is None or int(dangerous_match.group(1)) != 0:
                raise RuntimeError(
                    f"{material.symbol} preflight did not report zero dangerous builds: "
                    f"match={dangerous_match.group(0) if dangerous_match else None!r}."
                )
            loop_matches = LOOP_TIME_RE.findall(output)
            if not loop_matches:
                raise RuntimeError(f"{material.symbol} preflight did not report a LAMMPS loop time.")
            results.append(
                {
                    "material": material.symbol,
                    "snapshot": snapshot,
                    "duration_ps": 0.1,
                    "elapsed_seconds": elapsed_seconds,
                    "lammps_loop_seconds": float(loop_matches[-1]),
                    "initial": thermo_rows[0],
                    "final": thermo_rows[-1],
                    "dangerous_neighbor_builds": 0,
                    "trajectory_scan": scan,
                }
            )
            print(
                f"{material.symbol} preflight passed: "
                f"P0={thermo_rows[0]['pressure_bar']:.3f} bar, "
                f"P1={thermo_rows[-1]['pressure_bar']:.3f} bar, "
                f"T1={thermo_rows[-1]['temperature_K']:.3f} K",
                flush=True,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": _utc_now(),
        "mpi_ranks": MPI_RANKS,
        "results": results,
    }
    _write_json_atomic(output_root / "preflight.json", report)
    manifest["preflight"] = {
        "state": "passed",
        "report": "preflight.json",
        "completed_at_utc": report["completed_at_utc"],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps(report, indent=2), flush=True)


def _run_branch(output_root: Path, material: Material, snapshot: str) -> None:
    branch_dir = output_root / "branches" / material.symbol / snapshot
    metadata_path = branch_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    if metadata.get("state") == "complete":
        print(f"{material.symbol}/{snapshot}: already complete", flush=True)
        return
    if metadata.get("state") != "prepared":
        raise RuntimeError(
            f"Branch is not runnable: {metadata_path} has state={metadata.get('state')!r}."
        )
    artifacts = [
        branch_dir / filename
        for filename in ("trajectory.lammpstrj", "final.restart.bin", "lammps.log", "stdout.log")
    ]
    partial = [path.name for path in artifacts if path.exists()]
    if partial:
        raise RuntimeError(f"Refusing to overwrite partial branch files in {branch_dir}: {partial}.")
    free_bytes = shutil.disk_usage(output_root).free
    required_bytes = int(metadata["estimated_trajectory_bytes"] * 1.08)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk for {material.symbol}/{snapshot}: free={free_bytes}, "
            f"required_with_margin={required_bytes}."
        )

    lammps, mpiexec = _executables()
    command = [str(mpiexec), "-n", str(MPI_RANKS), str(lammps), "-in", "in.lammps"]
    metadata.update(
        {
            "state": "running",
            "started_at_utc": _utc_now(),
            "command": command,
        }
    )
    _write_json_atomic(metadata_path, metadata)
    completion_marker = f"{material.symbol.upper()}_24PS_BRANCH_COMPLETE"
    saw_completion = False
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        with (branch_dir / "stdout.log").open("x", encoding="utf-8") as stdout:
            process = subprocess.Popen(
                command,
                cwd=branch_dir,
                env=_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("LAMMPS stdout pipe was not created.")
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                if completion_marker in line:
                    saw_completion = True
                fields = line.split()
                if fields:
                    try:
                        step = int(fields[0])
                    except ValueError:
                        continue
                    progress_steps = _exact_steps(1.0, material.timestep_fs)
                    if step % progress_steps == 0:
                        print(f"{material.symbol}/{snapshot}: {line.strip()}", flush=True)
            return_code = process.wait()
        output = (branch_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if return_code != 0 or LAMMPS_ERROR_RE.search(output) or not saw_completion:
            errors = [line for line in output.splitlines() if line.startswith("ERROR")]
            raise RuntimeError(
                f"LAMMPS failed for {material.symbol}/{snapshot}: return_code={return_code}, "
                f"completion_marker={saw_completion}, errors={errors[-5:]}. "
                f"Inspect {branch_dir / 'stdout.log'}."
            )
        trajectory = branch_dir / "trajectory.lammpstrj"
        restart = branch_dir / "final.restart.bin"
        missing = [
            str(path) for path in (trajectory, restart) if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                f"LAMMPS completed {material.symbol}/{snapshot} without artifacts: {missing}."
            )
        scan = _scan_trajectory(trajectory, metadata)
        loop_matches = LOOP_TIME_RE.findall(output)
        metadata.update(
            {
                "state": "complete",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "lammps_loop_seconds": float(loop_matches[-1]) if loop_matches else None,
                "trajectory": {
                    "path": "trajectory.lammpstrj",
                    "size_bytes": trajectory.stat().st_size,
                    "scan": scan,
                },
                "restart": {
                    "path": "final.restart.bin",
                    "size_bytes": restart.stat().st_size,
                },
            }
        )
        _write_json_atomic(metadata_path, metadata)
        print(f"{material.symbol}/{snapshot}: complete", flush=True)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        metadata.update(
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(metadata_path, metadata)
        raise


def run(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("state") == "complete":
        print(f"Campaign is already complete: {output_root}", flush=True)
        return
    if manifest.get("state") not in {"prepared", "running"}:
        raise RuntimeError(
            f"Campaign is not runnable: state={manifest.get('state')!r} in {manifest_path}."
        )
    manifest.update({"state": "running", "started_at_utc": manifest.get("started_at_utc", _utc_now())})
    _write_json_atomic(manifest_path, manifest)
    started = time.monotonic()
    try:
        for material in MATERIALS.values():
            for snapshot in material.sources:
                _run_branch(output_root, material, snapshot)
    except BaseException as error:
        manifest.update(
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(manifest_path, manifest)
        runner_path = output_root / "runner.json"
        if runner_path.is_file():
            runner = _load_json(runner_path)
            runner.update(
                {
                    "state": "failed",
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_json_atomic(runner_path, runner)
        raise
    branch_metadata = [
        _load_json(output_root / "branches" / material.symbol / snapshot / "metadata.json")
        for material in MATERIALS.values()
        for snapshot in material.sources
    ]
    manifest.update(
        {
            "state": "complete",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "completed_branch_count": sum(
                record.get("state") == "complete" for record in branch_metadata
            ),
            "actual_trajectory_bytes": sum(
                int(record["trajectory"]["size_bytes"]) for record in branch_metadata
            ),
            "actual_restart_bytes": sum(
                int(record["restart"]["size_bytes"]) for record in branch_metadata
            ),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    runner_path = output_root / "runner.json"
    if runner_path.is_file():
        runner = _load_json(runner_path)
        runner.update({"state": "complete", "completed_at_utc": _utc_now()})
        _write_json_atomic(runner_path, runner)
    print(json.dumps(manifest, indent=2), flush=True)


def _pid_is_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def launch(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("state") != "prepared":
        raise RuntimeError(
            f"Detached launch requires prepared state, got {manifest.get('state')!r}."
        )
    preflight_record = manifest.get("preflight")
    if not isinstance(preflight_record, dict) or preflight_record.get("state") != "passed":
        raise RuntimeError(
            f"Detached launch requires a passed preflight in {manifest_path}; "
            f"got {preflight_record!r}."
        )
    runner_path = output_root / "runner.json"
    if runner_path.exists():
        raise FileExistsError(f"Runner record already exists: {runner_path}.")
    log_path = output_root / "campaign_runner.log"
    command = [str(Path(sys.executable)), str(SCRIPT_PATH), "_run-detached", str(output_root)]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runner = {
        "schema_version": SCHEMA_VERSION,
        "state": "running",
        "pid": process.pid,
        "process_group": process.pid,
        "launched_at_utc": _utc_now(),
        "command": command,
        "log": str(log_path),
    }
    _write_json_atomic(runner_path, runner)
    manifest.update({"state": "running", "started_at_utc": _utc_now()})
    _write_json_atomic(manifest_path, manifest)
    time.sleep(0.5)
    if process.poll() is not None:
        raise RuntimeError(
            f"Detached campaign exited immediately with code {process.returncode}; inspect {log_path}."
        )
    print(json.dumps(runner, indent=2), flush=True)


def status(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    manifest = _load_json(output_root / "manifest.json")
    states: dict[str, int] = {}
    active: list[dict[str, object]] = []
    completed_bytes = 0
    for material in MATERIALS.values():
        for snapshot in material.sources:
            metadata_path = output_root / "branches" / material.symbol / snapshot / "metadata.json"
            record = _load_json(metadata_path)
            state = str(record["state"])
            states[state] = states.get(state, 0) + 1
            if state == "complete":
                completed_bytes += int(record["trajectory"]["size_bytes"])
            elif state == "running":
                log_path = metadata_path.parent / "lammps.log"
                last_step = None
                if log_path.is_file():
                    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        fields = line.split()
                        if fields:
                            try:
                                step = int(fields[0])
                            except ValueError:
                                continue
                            if 0 <= step <= int(record["duration_steps"]):
                                last_step = step
                active.append(
                    {
                        "material": material.symbol,
                        "snapshot": snapshot,
                        "last_step": last_step,
                        "total_steps": record["duration_steps"],
                    }
                )
    runner_path = output_root / "runner.json"
    runner = _load_json(runner_path) if runner_path.is_file() else None
    runner_pid = int(runner["pid"]) if runner is not None else -1
    document = {
        "campaign": str(output_root),
        "state": manifest.get("state"),
        "branch_states": states,
        "active": active,
        "completed_trajectory_bytes": completed_bytes,
        "free_bytes": shutil.disk_usage(output_root).free,
        "runner": runner,
        "runner_process_alive": _pid_is_alive(runner_pid),
    }
    print(json.dumps(document, indent=2), flush=True)


def stop(output_root: Path) -> None:
    output_root = output_root.expanduser().resolve()
    runner_path = output_root / "runner.json"
    runner = _load_json(runner_path)
    pid = int(runner["pid"])
    if runner.get("state") != "running" or not _pid_is_alive(pid):
        raise RuntimeError(
            f"No live campaign runner to stop: state={runner.get('state')!r}, pid={pid}."
        )
    os.killpg(pid, signal.SIGTERM)
    runner.update({"state": "stopped_by_user", "stopped_at_utc": _utc_now()})
    _write_json_atomic(runner_path, runner)
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update({"state": "stopped_by_user", "stopped_at_utc": _utc_now()})
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"stopped_process_group": pid, "campaign": str(output_root)}, indent=2))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "preflight", "run", "launch", "status", "stop"):
        command = subparsers.add_parser(action)
        command.add_argument("output", type=Path)
    hidden = subparsers.add_parser("_run-detached", help=argparse.SUPPRESS)
    hidden.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "prepare":
        prepare(args.output)
    elif args.action == "preflight":
        preflight(args.output)
    elif args.action in {"run", "_run-detached"}:
        run(args.output)
    elif args.action == "launch":
        launch(args.output)
    elif args.action == "status":
        status(args.output)
    elif args.action == "stop":
        stop(args.output)
    else:
        raise AssertionError(f"Unhandled action: {args.action!r}.")


if __name__ == "__main__":
    main()
