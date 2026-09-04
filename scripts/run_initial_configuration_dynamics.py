#!/usr/bin/env python3
"""Run position-conditioned 10 ps LAMMPS trajectories for Al, Mg, or Ta.

The archived ``initial_configurations`` contain positions and periodic boxes, but
not velocities or Nose-Hoover thermostat/barostat state.  Consequently, this
script creates a new fixed-cell Langevin-NVT trajectory from each selected
position snapshot; it does not claim to continue the original NPT trajectory.

The script is standalone: it imports only the Python standard library.  Material
temperatures, integration timesteps, source files, and EAM potentials are the
concrete values used by the repository-owned datasets and their source paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
POTENTIAL_ROOT = REPOSITORY_ROOT / "datasets/potentials"
EXPECTED_COLUMNS = ("id", "type", "x", "y", "z")
SCHEMA_VERSION = 1
DEFAULT_DURATION_PS = 10.0
DEFAULT_SAMPLE_INTERVAL_PS = 0.1
DEFAULT_MPI_RANKS = 32
LAMMPS_ERROR_RE = re.compile(r"^ERROR(?: on proc \d+)?:", flags=re.MULTILINE)


@dataclass(frozen=True)
class Potential:
    filename: str
    pair_style: str
    sha256: str
    download_url: str
    citation: str


@dataclass(frozen=True)
class Source:
    relative_path: str
    timestep: int
    atom_count: int


@dataclass(frozen=True)
class Material:
    symbol: str
    mass: float
    temperature_K: float
    timestep_fs: float
    lattice_style: str
    lattice_constant_A: float
    potential: Potential
    sources: dict[str, Source]


MATERIALS = {
    "Al": Material(
        symbol="Al",
        mass=26.981_538_5,
        temperature_K=650.0,
        timestep_fs=1.0,
        lattice_style="fcc",
        lattice_constant_A=4.05,
        potential=Potential(
            filename="Al1.eam.fs",
            pair_style="eam/fs",
            sha256="768a9ad9b0cda57f36523b5d247942130101b26b0cbbd9d30c7bd7e1decc7ae3",
            download_url=(
                "https://www.ctcms.nist.gov/potentials/Download/"
                "2008--Mendelev-M-I-Kramer-M-J-Becker-C-A-Asta-M--Al/1/Al1.eam.fs"
            ),
            citation=(
                "M. I. Mendelev, M. J. Kramer, C. A. Becker, and M. Asta, "
                "Philosophical Magazine 88, 1723-1750 (2008), "
                "doi:10.1080/14786430802206482."
            ),
        ),
        sources={
            "166ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/166ps.pos",
                1_660_000,
                1_048_576,
            ),
            "170ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/170ps.pos",
                1_700_000,
                1_048_576,
            ),
            "174ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/174ps.pos",
                1_740_000,
                1_048_576,
            ),
            "175ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/175ps.pos",
                1_750_000,
                1_048_576,
            ),
            "177ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/177ps.pos",
                1_770_000,
                1_048_576,
            ),
            "240ps": Source(
                "datasets/Al/inherent_configurations_off/initial_configurations/240ps.pos",
                2_400_000,
                1_048_576,
            ),
        },
    ),
    "Mg": Material(
        symbol="Mg",
        mass=24.305,
        temperature_K=600.0,
        timestep_fs=1.0,
        lattice_style="hcp",
        lattice_constant_A=3.21,
        potential=Potential(
            filename="Mg1.eam.fs",
            pair_style="eam/fs",
            sha256="0ceed5387f16d0cb7f4a2088fc4665e27708dc28e3e5b3a6c2e01ac0528faba2",
            download_url=(
                "https://www.ctcms.nist.gov/potentials/Download/"
                "2016--Wilson-S-R-Mendelev-M-I--Mg/1/Mg1.eam.fs"
            ),
            citation=(
                "S. R. Wilson and M. I. Mendelev, Journal of Chemical Physics 144, "
                "144707 (2016), doi:10.1063/1.4946032."
            ),
        ),
        sources={
            "940ps": Source("datasets/Mg/initial_configurations/940ps.pos", 940_000, 1_048_576),
            "960ps": Source("datasets/Mg/initial_configurations/960ps.pos", 960_000, 1_048_576),
            "980ps": Source("datasets/Mg/initial_configurations/980ps.pos", 980_000, 1_048_576),
            "990ps": Source("datasets/Mg/initial_configurations/990ps.pos", 990_000, 1_048_576),
            "1000ps": Source(
                "datasets/Mg/initial_configurations/1000ps.pos", 1_000_000, 1_048_576
            ),
            "1500ps": Source(
                "datasets/Mg/initial_configurations/1500ps.pos", 1_500_000, 1_048_576
            ),
        },
    ),
    "Ta": Material(
        symbol="Ta",
        mass=180.947_88,
        temperature_K=1900.0,
        timestep_fs=2.0,
        lattice_style="bcc",
        lattice_constant_A=3.30,
        potential=Potential(
            filename="Ta.lammps.eam",
            pair_style="eam/alloy",
            sha256="8908993117f2502ed48bd31b737556719c3f7f11d1e7b7213eb257cd1ca42386",
            download_url=(
                "https://web.archive.org/web/20170715234655id_/"
                "https://sites.google.com/site/eampotentials/Ta/Ta.lammps.eam"
                "?attredirects=0&d=1"
            ),
            citation=(
                "L. Zhong, J. Wang, H. Sheng, Z. Zhang, and S. X. Mao, Nature 512, "
                "177-180 (2014), doi:10.1038/nature13617; author-published EAM table."
            ),
        ),
        sources={
            "2.7ns": Source("datasets/Ta/initial_configurations/2.7ns.pos", 1_350_000, 10_000_422),
            "2.8ns": Source("datasets/Ta/initial_configurations/2.8ns.pos", 1_400_000, 10_000_422),
            "2.9ns": Source("datasets/Ta/initial_configurations/2.9ns.pos", 1_450_000, 10_000_422),
            "3.0ns": Source("datasets/Ta/initial_configurations/3.0ns.pos", 1_500_000, 10_000_422),
            "3.60ns": Source(
                "datasets/Ta/initial_configurations/3.60ns.pos", 1_800_000, 10_000_422
            ),
            "model_1m": Source(
                "datasets/Ta/initial_configurations/model_1m.pos", 1_480_000, 1_024_000
            ),
        },
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
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema in {path}: expected={SCHEMA_VERSION}, "
            f"observed={document.get('schema_version')!r}."
        )
    return document


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a finite positive number, got {value!r}.")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def _seed(value: str) -> int:
    parsed = _positive_int(value)
    if parsed >= 900_000_000:
        raise argparse.ArgumentTypeError(
            f"LAMMPS random seeds must be below 900000000, got {value!r}."
        )
    return parsed


def _default_seed(material: str, snapshot: str, stream: str) -> int:
    payload = f"PointCloudMaterials:initial-dynamics:v1:{material}:{snapshot}:{stream}"
    integer = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")
    return integer % 899_999_999 + 1


def _material(value: str) -> str:
    normalized = value.capitalize()
    if normalized not in MATERIALS:
        raise argparse.ArgumentTypeError(
            f"Unknown material {value!r}; expected one of {', '.join(MATERIALS)}."
        )
    return normalized


def _material_selection(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(MATERIALS)
    selected = [_material(value) for value in values]
    if len(selected) != len(set(selected)):
        raise ValueError(f"Materials must be unique, got {selected}.")
    return selected


def _exact_steps(duration_ps: float, timestep_fs: float, description: str) -> int:
    steps_float = duration_ps * 1000.0 / timestep_fs
    steps = round(steps_float)
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{description}={duration_ps} ps is not an exact multiple of "
            f"timestep={timestep_fs} fs; computed_steps={steps_float}."
        )
    return steps


def _read_source_header(path: Path, expected: Source) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required initial configuration is missing: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        header = [handle.readline().rstrip("\r\n") for _ in range(9)]
    if header[0] != "ITEM: TIMESTEP" or int(header[1]) != expected.timestep:
        raise ValueError(
            f"Unexpected source timestep in {path}: expected={expected.timestep}, "
            f"header={header[:2]!r}."
        )
    if header[2] != "ITEM: NUMBER OF ATOMS" or int(header[3]) != expected.atom_count:
        raise ValueError(
            f"Unexpected atom count in {path}: expected={expected.atom_count}, "
            f"header={header[2:4]!r}."
        )
    if header[4] != "ITEM: BOX BOUNDS pp pp pp":
        raise ValueError(f"Expected a periodic orthorhombic box in {path}, got {header[4]!r}.")
    bounds = [[float(value) for value in row.split()] for row in header[5:8]]
    if any(len(axis) != 2 or axis[1] <= axis[0] for axis in bounds):
        raise ValueError(f"Invalid box bounds in {path}: {bounds!r}.")
    columns = tuple(header[8].split()[2:])
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Expected atom columns {EXPECTED_COLUMNS} in {path}, got {columns}.")
    return bounds


def _resolve_executable(requested: str | None, name: str) -> Path:
    if requested:
        candidate = Path(requested).expanduser()
        resolved = candidate.resolve() if candidate.is_file() else shutil.which(requested)
    else:
        environment_candidate = Path(sys.prefix) / "bin" / name
        resolved = environment_candidate if environment_candidate.is_file() else shutil.which(name)
    if resolved is None or not Path(resolved).is_file():
        raise FileNotFoundError(
            f"Could not resolve executable {requested or name!r}. Activate the LAMMPS "
            f"environment or pass --{name}."
        )
    return Path(resolved).resolve()


def _potential_path(material: Material) -> Path:
    path = POTENTIAL_ROOT / material.potential.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required {material.symbol} potential is missing: {path}. Run "
            f"'{sys.executable} {SCRIPT_PATH} install-potentials {material.symbol}'."
        )
    observed = _sha256(path)
    if observed != material.potential.sha256:
        raise ValueError(
            f"Potential checksum mismatch for {path}: expected={material.potential.sha256}, "
            f"observed={observed}. Refusing to simulate with an unidentified interaction model."
        )
    return path


def install_potentials(material_names: list[str]) -> None:
    POTENTIAL_ROOT.mkdir(parents=True, exist_ok=True)
    for name in _material_selection(material_names):
        material = MATERIALS[name]
        target = POTENTIAL_ROOT / material.potential.filename
        if target.exists():
            observed = _sha256(target)
            if observed != material.potential.sha256:
                raise ValueError(
                    f"Refusing to overwrite mismatched potential {target}: "
                    f"expected={material.potential.sha256}, observed={observed}."
                )
            print(f"{name}: already installed and checksum-valid: {target}", flush=True)
            continue
        request = urllib.request.Request(
            material.potential.download_url,
            headers={"User-Agent": "PointCloudMaterials-potential-installer/1"},
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=POTENTIAL_ROOT, prefix=f".{target.name}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                with urllib.request.urlopen(request, timeout=120) as response:
                    shutil.copyfileobj(response, temporary)
            observed = _sha256(temporary_path)
            if observed != material.potential.sha256:
                raise ValueError(
                    f"Downloaded {name} potential has the wrong checksum: "
                    f"expected={material.potential.sha256}, observed={observed}, "
                    f"url={material.potential.download_url}."
                )
            temporary_path.replace(target)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        print(f"{name}: installed {target}", flush=True)


def list_sources() -> None:
    rows: list[dict[str, object]] = []
    for name, material in MATERIALS.items():
        for label, source in material.sources.items():
            path = REPOSITORY_ROOT / source.relative_path
            rows.append(
                {
                    "material": name,
                    "snapshot": label,
                    "path": str(path),
                    "exists": path.is_file(),
                    "source_timestep": source.timestep,
                    "atom_count": source.atom_count,
                    "temperature_K": material.temperature_K,
                    "new_timestep_fs": material.timestep_fs,
                    "source_size_bytes": path.stat().st_size if path.is_file() else None,
                }
            )
    print(json.dumps(rows, indent=2))


def _lammps_path(path: Path) -> str:
    resolved = path.resolve()
    if any(character.isspace() for character in str(resolved)):
        raise ValueError(f"LAMMPS paths cannot contain whitespace: {resolved}.")
    return str(resolved)


def _pair_commands(material: Material, potential_path: Path) -> str:
    return (
        f"pair_style {material.potential.pair_style}\n"
        f"pair_coeff * * {_lammps_path(potential_path)} {material.symbol}"
    )


def _render_lammps_input(
    *,
    material: Material,
    source_path: Path,
    source: Source,
    potential_path: Path,
    duration_steps: int,
    sample_interval_steps: int,
    velocity_seed: int,
    thermostat_seed: int,
) -> str:
    thermostat_time_ps = 100.0 * material.timestep_fs / 1000.0
    marker = f"{material.symbol.upper()}_INITIAL_DYNAMICS"
    return f"""# Generated by {SCRIPT_PATH.name}.
# New position-conditioned fixed-cell Langevin-NVT path; not an exact NPT restart.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic

region placeholder block 0 1 0 1 0 1 units box
create_box 1 placeholder
read_dump {_lammps_path(source_path)} {source.timestep} x y z box yes add yes replace no trim no scaled no wrapped yes
reset_timestep 0

mass 1 {material.mass:.12g}
{_pair_commands(material, potential_path)}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {material.timestep_fs / 1000.0:.12g}
velocity all create {material.temperature_K:.12g} {velocity_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all langevin {material.temperature_K:.12g} {material.temperature_K:.12g} {thermostat_time_ps:.12g} {thermostat_seed} zero yes

thermo {sample_interval_steps}
thermo_style custom step time atoms temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {sample_interval_steps} trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format line "%d %d %.9g %.9g %.9g"

print "{marker}_BEGIN"
run {duration_steps}
write_restart final.restart.bin
print "{marker}_COMPLETE"
"""


def prepare_campaign(args: argparse.Namespace) -> dict[str, object]:
    material = MATERIALS[args.material]
    if args.snapshot not in material.sources:
        raise KeyError(
            f"Unknown {args.material} snapshot {args.snapshot!r}; expected one of "
            f"{list(material.sources)}."
        )
    source = material.sources[args.snapshot]
    velocity_seed = (
        args.velocity_seed
        if args.velocity_seed is not None
        else _default_seed(material.symbol, args.snapshot, "velocity")
    )
    thermostat_seed = (
        args.thermostat_seed
        if args.thermostat_seed is not None
        else _default_seed(material.symbol, args.snapshot, "thermostat")
    )
    if velocity_seed == thermostat_seed:
        raise ValueError(
            f"Velocity and thermostat seeds must differ, got {velocity_seed} for both streams."
        )
    source_path = REPOSITORY_ROOT / source.relative_path
    bounds = _read_source_header(source_path, source)
    installed_potential = _potential_path(material)
    duration_steps = _exact_steps(args.duration_ps, material.timestep_fs, "duration")
    sample_interval_steps = _exact_steps(
        args.sample_interval_ps, material.timestep_fs, "sample_interval"
    )
    if duration_steps % sample_interval_steps != 0:
        raise ValueError(
            f"Duration steps ({duration_steps}) must be divisible by sample interval steps "
            f"({sample_interval_steps})."
        )
    lammps = _resolve_executable(args.lmp, "lmp")
    mpiexec = _resolve_executable(args.mpiexec, "mpiexec") if args.mpi_ranks > 1 else None
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Campaign directory already exists and will not be overwritten: {output_root}.")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare.", dir=output_root.parent)
    )
    try:
        potential_dir = temporary_root / "potential"
        run_dir = temporary_root / "run"
        potential_dir.mkdir()
        run_dir.mkdir()
        campaign_potential = potential_dir / installed_potential.name
        final_campaign_potential = output_root / "potential" / installed_potential.name
        shutil.copy2(installed_potential, campaign_potential)
        input_path = run_dir / "in.lammps"
        input_path.write_text(
            _render_lammps_input(
                material=material,
                source_path=source_path,
                source=source,
                potential_path=final_campaign_potential,
                duration_steps=duration_steps,
                sample_interval_steps=sample_interval_steps,
                velocity_seed=velocity_seed,
                thermostat_seed=thermostat_seed,
            ),
            encoding="utf-8",
        )
        expected_frames = duration_steps // sample_interval_steps + 1
        source_bytes = source_path.stat().st_size
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "prepared_at_utc": _utc_now(),
            "output_root": str(output_root),
            "material": material.symbol,
            "snapshot": args.snapshot,
            "source": {
                "path": str(source_path.resolve()),
                "sha256": _sha256(source_path),
                "timestep": source.timestep,
                "atom_count": source.atom_count,
                "box_bounds_A": bounds,
                "columns": list(EXPECTED_COLUMNS),
                "size_bytes": source_bytes,
            },
            "protocol": {
                "ensemble": "fixed-cell Langevin NVT",
                "temperature_K": material.temperature_K,
                "timestep_fs": material.timestep_fs,
                "duration_ps": args.duration_ps,
                "duration_steps": duration_steps,
                "sample_interval_ps": args.sample_interval_ps,
                "sample_interval_steps": sample_interval_steps,
                "expected_frame_count": expected_frames,
                "thermostat_time_fs": 100.0 * material.timestep_fs,
                "velocity_seed": velocity_seed,
                "thermostat_seed": thermostat_seed,
            },
            "potential": {
                "file": f"potential/{campaign_potential.name}",
                "sha256": material.potential.sha256,
                "pair_style": material.potential.pair_style,
                "download_url": material.potential.download_url,
                "citation": material.potential.citation,
            },
            "execution": {
                "lammps": str(lammps),
                "mpiexec": str(mpiexec) if mpiexec is not None else None,
                "mpi_ranks": args.mpi_ranks,
            },
            "scientific_scope": (
                "New position-conditioned fixed-cell Langevin-NVT trajectory initialized "
                "from archived finite-temperature coordinates. The source has no velocities "
                "or thermostat/barostat state, so this is not an exact continuation of the "
                "paper's Nose-Hoover NPT trajectory."
            ),
            "protocol_provenance": (
                "S. Becker, E. Devijver, R. Molinier, and N. Jakse, Scientific "
                "Reports 12, 3195 (2022), doi:10.1038/s41598-022-06963-5: "
                "Tiso=650/600/1900 K for Al/Mg/Ta and timestep=1/1/2 fs."
            ),
            "source_timestep_note": (
                "The Al dump timestep is preserved exactly but is ten times the timestep "
                "implied by the filename and the paper's stated 1 fs integration step."
                if material.symbol == "Al"
                else "The dump timestep agrees with the filename at the paper's stated timestep."
            ),
            "estimated_trajectory_bytes": source_bytes * expected_frames,
        }
        _write_json_atomic(temporary_root / "manifest.json", manifest)
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root)
        raise
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_DYNAMIC"] = "FALSE"
    prefix_library = str(Path(sys.prefix) / "lib")
    environment["LD_LIBRARY_PATH"] = prefix_library + (
        f":{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else ""
    )
    return environment


def _scan_trajectory(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    protocol = manifest["protocol"]
    source = manifest["source"]
    if not isinstance(protocol, dict) or not isinstance(source, dict):
        raise TypeError(f"Malformed protocol/source records in {path.parent.parent / 'manifest.json'}.")
    expected_steps = list(
        range(
            0,
            int(protocol["duration_steps"]) + 1,
            int(protocol["sample_interval_steps"]),
        )
    )
    expected_atoms = int(source["atom_count"])
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
                line_start = found
                lines: list[str] = []
                cursor = line_start
                for _ in range(9):
                    line_end = data.find(b"\n", cursor)
                    if line_end < 0:
                        raise RuntimeError(f"Truncated trajectory header at byte {cursor} in {path}.")
                    lines.append(data[cursor:line_end].decode("ascii").rstrip("\r"))
                    cursor = line_end + 1
                if lines[0] != "ITEM: TIMESTEP" or lines[2] != "ITEM: NUMBER OF ATOMS":
                    raise RuntimeError(f"Malformed trajectory frame header at byte {found} in {path}.")
                if int(lines[3]) != expected_atoms:
                    raise RuntimeError(
                        f"Atom count changed in {path} at step {lines[1]}: "
                        f"expected={expected_atoms}, observed={lines[3]}."
                    )
                if lines[4] != "ITEM: BOX BOUNDS pp pp pp":
                    raise RuntimeError(
                        f"Boundary contract changed in {path} at step {lines[1]}: {lines[4]!r}."
                    )
                if tuple(lines[8].split()[2:]) != EXPECTED_COLUMNS:
                    raise RuntimeError(
                        f"Column contract changed in {path} at step {lines[1]}: {lines[8]!r}."
                    )
                observed_steps.append(int(lines[1]))
                position = cursor
            if data[-1:] != b"\n":
                raise RuntimeError(f"Trajectory does not end on a complete line: {path}.")
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"Trajectory timestep contract failed for {path}: expected_count={len(expected_steps)}, "
            f"observed_count={len(observed_steps)}, expected={expected_steps}, "
            f"observed={observed_steps}."
        )
    return {
        "frame_count": len(observed_steps),
        "first_timestep": observed_steps[0],
        "last_timestep": observed_steps[-1],
        "atom_count": expected_atoms,
        "columns": list(EXPECTED_COLUMNS),
    }


def run_campaign(campaign_root: Path) -> dict[str, object]:
    campaign_root = campaign_root.expanduser().resolve()
    manifest_path = campaign_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared campaign manifest is missing: {manifest_path}.")
    manifest = _load_json(manifest_path)
    if manifest.get("state") == "complete":
        print(f"Campaign is already complete: {campaign_root}", flush=True)
        return manifest
    if manifest.get("state") != "prepared":
        raise RuntimeError(
            f"Campaign is not runnable: {manifest_path} has state={manifest.get('state')!r}."
        )
    run_dir = campaign_root / "run"
    artifacts = [
        run_dir / name
        for name in ("trajectory.lammpstrj", "final.restart.bin", "lammps.log", "stdout.log")
    ]
    partial = [path.name for path in artifacts if path.exists()]
    if partial:
        raise RuntimeError(f"Refusing to overwrite partial artifacts in {run_dir}: {partial}.")
    estimated_bytes = int(manifest["estimated_trajectory_bytes"])
    free_bytes = shutil.disk_usage(campaign_root).free
    required_bytes = math.ceil(estimated_bytes * 1.10)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space for estimated text trajectory: free={free_bytes} bytes, "
            f"required_with_10_percent_margin={required_bytes} bytes, "
            f"campaign={campaign_root}."
        )
    execution = manifest["execution"]
    if not isinstance(execution, dict):
        raise TypeError(f"Malformed execution record in {manifest_path}.")
    lammps = Path(str(execution["lammps"]))
    if not lammps.is_file():
        raise FileNotFoundError(f"Recorded LAMMPS executable is missing: {lammps}.")
    mpi_ranks = int(execution["mpi_ranks"])
    if mpi_ranks == 1:
        command = [str(lammps), "-in", "in.lammps"]
    else:
        mpiexec = Path(str(execution["mpiexec"]))
        if not mpiexec.is_file():
            raise FileNotFoundError(f"Recorded MPI launcher is missing: {mpiexec}.")
        command = [str(mpiexec), "-n", str(mpi_ranks), str(lammps), "-in", "in.lammps"]
    manifest["state"] = "running"
    manifest["started_at_utc"] = _utc_now()
    manifest["command"] = command
    _write_json_atomic(manifest_path, manifest)
    stdout_path = run_dir / "stdout.log"
    started = time.monotonic()
    material = str(manifest["material"])
    completion_marker = f"{material.upper()}_INITIAL_DYNAMICS_COMPLETE"
    saw_completion = False
    try:
        process: subprocess.Popen[str] | None = None
        try:
            with stdout_path.open("x", encoding="utf-8") as stdout:
                process = subprocess.Popen(
                    command,
                    cwd=run_dir,
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
                    if len(fields) == 9:
                        try:
                            step = int(fields[0])
                        except ValueError:
                            continue
                        protocol = manifest["protocol"]
                        if not isinstance(protocol, dict):
                            raise TypeError(f"Malformed protocol record in {manifest_path}.")
                        progress_steps = round(1000.0 / float(protocol["timestep_fs"]))
                        if step % progress_steps == 0:
                            print(f"{material}: {line.strip()}", flush=True)
                return_code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
        output = stdout_path.read_text(encoding="utf-8", errors="replace")
        if return_code != 0 or LAMMPS_ERROR_RE.search(output) or not saw_completion:
            errors = [line for line in output.splitlines() if line.startswith("ERROR")]
            raise RuntimeError(
                f"LAMMPS failed: return_code={return_code}, completion_marker={saw_completion}, "
                f"errors={errors[-5:]}. Inspect {stdout_path} and {run_dir / 'lammps.log'}."
            )
        trajectory = run_dir / "trajectory.lammpstrj"
        restart = run_dir / "final.restart.bin"
        missing = [str(path) for path in (trajectory, restart) if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"LAMMPS completed without required artifacts: {missing}.")
        scan = _scan_trajectory(trajectory, manifest)
        manifest.update(
            {
                "state": "complete",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "trajectory": {
                    "path": "run/trajectory.lammpstrj",
                    "size_bytes": trajectory.stat().st_size,
                    "scan": scan,
                },
                "restart": {
                    "path": "run/final.restart.bin",
                    "size_bytes": restart.stat().st_size,
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)
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
        raise
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_campaign(campaign_root: Path) -> dict[str, object]:
    campaign_root = campaign_root.expanduser().resolve()
    manifest = _load_json(campaign_root / "manifest.json")
    if manifest.get("state") != "prepared":
        raise RuntimeError(
            f"Only a prepared campaign can be launched; observed state={manifest.get('state')!r}."
        )
    runner_path = campaign_root / "runner.json"
    runner_log = campaign_root / "campaign_runner.log"
    if runner_path.exists() or runner_log.exists():
        raise FileExistsError(
            f"Runner metadata/log already exists and will not be overwritten: "
            f"{runner_path}, {runner_log}."
        )
    _write_json_atomic(
        runner_path,
        {"schema_version": SCHEMA_VERSION, "state": "launching", "created_at_utc": _utc_now()},
    )
    try:
        with runner_log.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT_PATH), "_run-detached", str(campaign_root)],
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
            "launched_at_utc": _utc_now(),
            "log": str(runner_log),
        }
        _write_json_atomic(runner_path, runner)
        time.sleep(0.25)
        if process.poll() is not None:
            raise RuntimeError(
                f"Detached runner exited immediately with return code {process.returncode}; "
                f"inspect {runner_log}."
            )
    except BaseException as error:
        _write_json_atomic(
            runner_path,
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    print(json.dumps(runner, indent=2), flush=True)
    return runner


def _run_detached(campaign_root: Path) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    runner_path = campaign_root / "runner.json"
    try:
        run_campaign(campaign_root)
    except BaseException as error:
        runner = _load_json(runner_path)
        runner.update(
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(runner_path, runner)
        raise
    runner = _load_json(runner_path)
    runner.update({"state": "complete", "completed_at_utc": _utc_now()})
    _write_json_atomic(runner_path, runner)


def status_campaign(campaign_root: Path) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    manifest = _load_json(campaign_root / "manifest.json")
    runner_path = campaign_root / "runner.json"
    runner = _load_json(runner_path) if runner_path.is_file() else None
    pid = int(runner.get("pid", -1)) if runner is not None else -1
    document = {
        "campaign": str(campaign_root),
        "state": manifest.get("state"),
        "material": manifest.get("material"),
        "snapshot": manifest.get("snapshot"),
        "runner": runner,
        "runner_process_alive": _pid_is_alive(pid) if pid > 0 else False,
    }
    print(json.dumps(document, indent=2))


def stop_campaign(campaign_root: Path) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    runner_path = campaign_root / "runner.json"
    runner = _load_json(runner_path)
    pid = int(runner.get("pid", -1))
    if runner.get("state") != "running" or pid <= 0 or not _pid_is_alive(pid):
        raise RuntimeError(
            f"No live detached runner to stop: state={runner.get('state')!r}, pid={pid}."
        )
    os.killpg(pid, signal.SIGTERM)
    runner.update({"state": "stopped_by_user", "stopped_at_utc": _utc_now()})
    _write_json_atomic(runner_path, runner)
    manifest_path = campaign_root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update({"state": "stopped_by_user", "stopped_at_utc": _utc_now()})
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"stopped_process_group": pid, "campaign": str(campaign_root)}, indent=2))


def check_potentials(material_names: list[str], requested_lammps: str | None) -> None:
    lammps = _resolve_executable(requested_lammps, "lmp")
    for name in _material_selection(material_names):
        material = MATERIALS[name]
        potential = _potential_path(material)
        input_text = f"""units metal
atom_style atomic
boundary p p p
lattice {material.lattice_style} {material.lattice_constant_A}
region box block 0 2 0 2 0 2 units lattice
create_box 1 box
create_atoms 1 box
mass 1 {material.mass:.12g}
{_pair_commands(material, potential)}
thermo 1
thermo_style custom step atoms pe press
run 0
print "{material.symbol.upper()}_POTENTIAL_CHECK_COMPLETE"
"""
        with tempfile.TemporaryDirectory(prefix=f"{name.lower()}_potential_check_") as temporary:
            result = subprocess.run(
                [str(lammps), "-log", "none"],
                input=input_text,
                cwd=temporary,
                env=_environment(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        marker = f"{material.symbol.upper()}_POTENTIAL_CHECK_COMPLETE"
        if result.returncode != 0 or LAMMPS_ERROR_RE.search(result.stdout) or marker not in result.stdout:
            raise RuntimeError(
                f"{name} potential smoke test failed: return_code={result.returncode}, "
                f"output={result.stdout[-4000:]!r}."
            )
        print(f"{name}: LAMMPS potential check passed ({material.potential.pair_style}).", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    install = subparsers.add_parser("install-potentials")
    install.add_argument("materials", nargs="+", help="Al, Mg, Ta, or the single value 'all'.")

    subparsers.add_parser("list-sources")

    check = subparsers.add_parser("check-potentials")
    check.add_argument("materials", nargs="+", help="Al, Mg, Ta, or the single value 'all'.")
    check.add_argument("--lmp")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("material", type=_material)
    prepare.add_argument("snapshot")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--duration-ps", type=_positive_float, default=DEFAULT_DURATION_PS)
    prepare.add_argument(
        "--sample-interval-ps", type=_positive_float, default=DEFAULT_SAMPLE_INTERVAL_PS
    )
    prepare.add_argument("--mpi-ranks", type=_positive_int, default=DEFAULT_MPI_RANKS)
    prepare.add_argument(
        "--velocity-seed",
        type=_seed,
        help="Override the deterministic material/snapshot-specific velocity seed.",
    )
    prepare.add_argument(
        "--thermostat-seed",
        type=_seed,
        help="Override the deterministic material/snapshot-specific thermostat seed.",
    )
    prepare.add_argument("--lmp")
    prepare.add_argument("--mpiexec")

    for action in ("run", "launch", "status", "stop"):
        command = subparsers.add_parser(action)
        command.add_argument("campaign", type=Path)

    hidden = subparsers.add_parser("_run-detached", help=argparse.SUPPRESS)
    hidden.add_argument("campaign", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.action == "install-potentials":
        install_potentials(args.materials)
    elif args.action == "list-sources":
        list_sources()
    elif args.action == "check-potentials":
        check_potentials(args.materials, args.lmp)
    elif args.action == "prepare":
        prepare_campaign(args)
    elif args.action == "run":
        run_campaign(args.campaign)
    elif args.action == "launch":
        launch_campaign(args.campaign)
    elif args.action == "status":
        status_campaign(args.campaign)
    elif args.action == "stop":
        stop_campaign(args.campaign)
    elif args.action == "_run-detached":
        _run_detached(args.campaign)
    else:
        raise AssertionError(f"Unhandled action {args.action!r}.")


if __name__ == "__main__":
    main()
