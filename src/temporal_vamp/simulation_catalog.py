from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data_utils.temporal_lammps_binary import BINARY_SUFFIX


@dataclass(frozen=True)
class SimulationMetadata:
    campaign_name: str
    replica_name: str
    atom_count: int
    temperature_K: float
    pressure_GPa: float
    timestep_fs: float
    equilibration_steps: int
    measurement_steps: int
    sample_interval_steps: int
    sample_interval_ps: float
    frame_count: int
    first_dump_timestep: int
    last_dump_timestep: int
    velocity_seed: int
    crystal_seed: int | None
    boundary_conditions: tuple[str, str, str]
    ensemble: str
    potential_name: str
    potential_library_sha256: str
    potential_parameter_sha256: str
    prepared_liquid_sha256: str | None
    nucleation_observed: bool
    nucleation_time_ps: float | None
    initial_crystalline_fraction: float
    final_crystalline_fraction: float
    progress_steps: tuple[int, ...]
    progress_times_ps: tuple[float, ...]
    structure_names: tuple[str, ...]
    structure_fractions: tuple[tuple[float, ...], ...]
    crystalline_fraction: tuple[float, ...]
    crystalline_cluster_count: tuple[int, ...]
    largest_crystalline_cluster_atoms: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class CatalogEntry:
    trajectory_path: Path
    run_id: str
    cache_dir: Path
    metadata: SimulationMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_path": str(self.trajectory_path),
            "run_id": self.run_id,
            "cache_dir": str(self.cache_dir),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class _LAMMPSProtocol:
    units: str
    boundary_conditions: tuple[str, str, str]
    timestep_fs: float
    temperature_K: float
    pressure_GPa: float
    velocity_seed: int
    ensemble: str
    equilibration_steps: int
    measurement_steps: int
    sample_interval_steps: int
    dump_columns: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required simulation metadata file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Simulation metadata is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _single_match(pattern: str, text: str, *, field: str, path: Path) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {field} declaration in {path}, found {len(matches)}."
        )
    return matches[0]


def _parse_lammps_protocol(path: Path) -> _LAMMPSProtocol:
    if not path.is_file():
        raise FileNotFoundError(f"Required LAMMPS input is missing: {path}")
    text = path.read_text(encoding="utf-8")
    units = _single_match(r"^\s*units\s+(\S+)\s*$", text, field="units", path=path).group(1)
    if units != "metal":
        raise ValueError(
            f"Temporal catalog currently supports repository simulations using LAMMPS metal units; "
            f"got units={units!r} in {path}."
        )
    boundary = tuple(
        _single_match(
            r"^\s*boundary\s+(\S+)\s+(\S+)\s+(\S+)\s*$",
            text,
            field="boundary",
            path=path,
        ).groups()
    )
    timestep_ps = float(
        _single_match(
            r"^\s*timestep\s+([0-9.eE+-]+)\s*$", text, field="timestep", path=path
        ).group(1)
    )
    velocity = _single_match(
        r"^\s*velocity\s+all\s+create\s+([0-9.eE+-]+)\s+(\d+)\b.*$",
        text,
        field="velocity all create",
        path=path,
    )
    ensemble = _single_match(
        r"^\s*fix\s+\S+\s+all\s+(npt)\s+temp\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+[0-9.eE+-]+\s+iso\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)\s+[0-9.eE+-]+\s*$",
        text,
        field="NPT ensemble",
        path=path,
    )
    temperature_start = float(ensemble.group(2))
    temperature_stop = float(ensemble.group(3))
    pressure_start_bar = float(ensemble.group(4))
    pressure_stop_bar = float(ensemble.group(5))
    velocity_temperature = float(velocity.group(1))
    if temperature_start != temperature_stop or temperature_start != velocity_temperature:
        raise ValueError(
            f"Expected a constant target temperature matching velocity initialization in {path}; "
            f"got velocity={velocity_temperature}, NPT=({temperature_start}, {temperature_stop}) K."
        )
    if pressure_start_bar != pressure_stop_bar:
        raise ValueError(
            f"Expected constant NPT pressure in {path}, got ({pressure_start_bar}, "
            f"{pressure_stop_bar}) bar."
        )
    dump = _single_match(
        r"^\s*dump\s+\S+\s+all\s+custom\s+(\d+)\s+trajectory\.lammpstrj\s+(.+?)\s*$",
        text,
        field="trajectory dump",
        path=path,
    )
    runs = [int(match.group(1)) for match in re.finditer(r"^\s*run\s+(\d+)\s*$", text, re.MULTILINE)]
    if len(runs) != 2:
        raise ValueError(
            f"Expected exactly two run commands (equilibration and measurement) in {path}, "
            f"found {runs}."
        )
    return _LAMMPSProtocol(
        units=units,
        boundary_conditions=boundary,
        timestep_fs=timestep_ps * 1000.0,
        temperature_K=temperature_start,
        pressure_GPa=pressure_start_bar / 10000.0,
        velocity_seed=int(velocity.group(2)),
        ensemble=ensemble.group(1),
        equilibration_steps=runs[0],
        measurement_steps=runs[1],
        sample_interval_steps=int(dump.group(1)),
        dump_columns=tuple(dump.group(2).split()),
    )


def _find_simulation_manifest(replica_dir: Path, campaign_dir: Path) -> tuple[Path, dict[str, Any]]:
    current = replica_dir.parent
    while True:
        candidate = current / "manifest.json"
        if candidate.is_file():
            manifest = _load_json(candidate)
            if all(key in manifest for key in ("atom_count", "potential", "protocol")):
                return candidate, manifest
        if current == campaign_dir:
            break
        if campaign_dir not in current.parents:
            break
        current = current.parent
    raise FileNotFoundError(
        "Could not find a simulation manifest containing atom_count, potential, and protocol "
        f"between replica={replica_dir} and campaign={campaign_dir}."
    )


def _manifest_temperature(protocol: dict[str, Any], path: Path) -> float:
    if "temperature_K" in protocol:
        return float(protocol["temperature_K"])
    if "target_temperature_K" in protocol:
        return float(protocol["target_temperature_K"])
    raise KeyError(
        f"Simulation protocol in {path} has neither temperature_K nor target_temperature_K."
    )


def _assert_equal(actual: Any, expected: Any, *, field: str, context: Path) -> None:
    if actual != expected:
        raise ValueError(
            f"Simulation metadata disagreement for {field} in {context}: "
            f"manifest/progress={actual!r}, LAMMPS={expected!r}."
        )


def _assert_float_close(actual: float, expected: float, *, field: str, context: Path) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Simulation metadata disagreement for {field} in {context}: "
            f"manifest/progress={actual!r}, LAMMPS={expected!r}."
        )


def load_simulation_metadata(trajectory_path: str | Path, campaign_dir: str | Path) -> SimulationMetadata:
    trajectory = Path(trajectory_path).expanduser().resolve()
    campaign = Path(campaign_dir).expanduser().resolve()
    replica_dir = trajectory.parent
    analysis_path = replica_dir / "analysis.json"
    progress_path = replica_dir / "crystallization_progress.npz"
    input_path = replica_dir / "in.lammps"
    analysis = _load_json(analysis_path)
    manifest_path, manifest = _find_simulation_manifest(replica_dir, campaign)
    lammps = _parse_lammps_protocol(input_path)

    if not progress_path.is_file():
        raise FileNotFoundError(f"Required crystallization progress metadata is missing: {progress_path}")
    with np.load(progress_path, allow_pickle=False) as progress:
        required_arrays = {
            "step",
            "time_ps",
            "structure_names",
            "structure_fractions",
            "crystalline_fraction",
            "crystalline_cluster_count",
            "largest_crystalline_cluster_atoms",
        }
        missing = sorted(required_arrays.difference(progress.files))
        if missing:
            raise KeyError(f"{progress_path} is missing required arrays {missing}.")
        steps = np.asarray(progress["step"], dtype=np.int64)
        times_ps = np.asarray(progress["time_ps"], dtype=np.float64)
        structure_names = np.asarray(progress["structure_names"]).astype(str)
        structure_fractions = np.asarray(progress["structure_fractions"], dtype=np.float64)
        crystalline_fraction = np.asarray(progress["crystalline_fraction"], dtype=np.float64)
        crystalline_cluster_count = np.asarray(
            progress["crystalline_cluster_count"], dtype=np.int64
        )
        largest_cluster = np.asarray(
            progress["largest_crystalline_cluster_atoms"], dtype=np.int64
        )
    if (
        steps.ndim != 1
        or times_ps.shape != steps.shape
        or crystalline_fraction.shape != steps.shape
        or crystalline_cluster_count.shape != steps.shape
        or largest_cluster.shape != steps.shape
        or structure_names.ndim != 1
        or structure_fractions.shape != (steps.size, structure_names.size)
    ):
        raise ValueError(
            f"Progress arrays have inconsistent shapes in {progress_path}; got "
            f"step={steps.shape}, time_ps={times_ps.shape}, "
            f"structure_names={structure_names.shape}, "
            f"structure_fractions={structure_fractions.shape}, "
            f"crystalline_fraction={crystalline_fraction.shape}, "
            f"crystalline_cluster_count={crystalline_cluster_count.shape}, "
            f"largest_cluster={largest_cluster.shape}."
        )
    expected_steps = np.arange(
        0,
        lammps.measurement_steps + lammps.sample_interval_steps,
        lammps.sample_interval_steps,
        dtype=np.int64,
    )
    if not np.array_equal(steps, expected_steps):
        raise ValueError(
            f"Progress steps in {progress_path} do not match LAMMPS measurement/dump cadence; "
            f"expected {expected_steps.size} values from 0 to {lammps.measurement_steps} "
            f"by {lammps.sample_interval_steps}, got {steps.size} values from "
            f"{int(steps[0])} to {int(steps[-1])}."
        )
    expected_times_ps = steps.astype(np.float64) * lammps.timestep_fs / 1000.0
    if not np.allclose(times_ps, expected_times_ps, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Progress physical times in {progress_path} disagree with step*timestep."
        )

    protocol = manifest["protocol"]
    potential = manifest["potential"]
    _assert_float_close(_manifest_temperature(protocol, manifest_path), lammps.temperature_K, field="temperature_K", context=manifest_path)
    _assert_float_close(float(protocol["pressure_GPa"]), lammps.pressure_GPa, field="pressure_GPa", context=manifest_path)
    _assert_float_close(float(protocol["timestep_fs"]), lammps.timestep_fs, field="timestep_fs", context=manifest_path)
    _assert_equal(int(protocol["equilibration_steps"]), lammps.equilibration_steps, field="equilibration_steps", context=manifest_path)
    _assert_equal(int(protocol["measurement_steps"]), lammps.measurement_steps, field="measurement_steps", context=manifest_path)
    if "sample_interval_steps" in protocol:
        _assert_equal(int(protocol["sample_interval_steps"]), lammps.sample_interval_steps, field="sample_interval_steps", context=manifest_path)
    _assert_equal(int(analysis["velocity_random_seed"]), lammps.velocity_seed, field="velocity_seed", context=analysis_path)
    _assert_equal(analysis["crystal_seed"], manifest["crystal_seed"], field="crystal_seed", context=analysis_path)
    required_dump_columns = ("id", "type", "x", "y", "z")
    _assert_equal(lammps.dump_columns, required_dump_columns, field="dump_columns", context=input_path)

    sample_interval_ps = lammps.sample_interval_steps * lammps.timestep_fs / 1000.0
    if "sample_interval_ps" in protocol:
        _assert_float_close(float(protocol["sample_interval_ps"]), sample_interval_ps, field="sample_interval_ps", context=manifest_path)
    initial_fraction = float(crystalline_fraction[0])
    final_fraction = float(crystalline_fraction[-1])
    _assert_float_close(float(analysis["initial_crystalline_fraction"]), initial_fraction, field="initial_crystalline_fraction", context=analysis_path)
    _assert_float_close(float(analysis["final_crystalline_fraction"]), final_fraction, field="final_crystalline_fraction", context=analysis_path)

    shared_liquid = manifest.get("shared_liquid_source", {})
    return SimulationMetadata(
        campaign_name=campaign.name,
        replica_name=str(analysis["replica_name"]),
        atom_count=int(manifest["atom_count"]),
        temperature_K=lammps.temperature_K,
        pressure_GPa=lammps.pressure_GPa,
        timestep_fs=lammps.timestep_fs,
        equilibration_steps=lammps.equilibration_steps,
        measurement_steps=lammps.measurement_steps,
        sample_interval_steps=lammps.sample_interval_steps,
        sample_interval_ps=sample_interval_ps,
        frame_count=int(steps.size),
        first_dump_timestep=lammps.equilibration_steps,
        last_dump_timestep=lammps.equilibration_steps + lammps.measurement_steps,
        velocity_seed=lammps.velocity_seed,
        crystal_seed=analysis["crystal_seed"],
        boundary_conditions=lammps.boundary_conditions,
        ensemble=lammps.ensemble,
        potential_name=str(potential["name"]),
        potential_library_sha256=str(potential["library_sha256"]),
        potential_parameter_sha256=str(potential["parameter_sha256"]),
        prepared_liquid_sha256=shared_liquid.get("prepared_liquid_sha256"),
        nucleation_observed=bool(analysis["nucleation_observed"]),
        nucleation_time_ps=(
            None if analysis["nucleation_time_ps"] is None else float(analysis["nucleation_time_ps"])
        ),
        initial_crystalline_fraction=initial_fraction,
        final_crystalline_fraction=final_fraction,
        progress_steps=tuple(int(value) for value in steps),
        progress_times_ps=tuple(float(value) for value in times_ps),
        structure_names=tuple(str(value) for value in structure_names),
        structure_fractions=tuple(
            tuple(float(value) for value in row) for row in structure_fractions
        ),
        crystalline_fraction=tuple(float(value) for value in crystalline_fraction),
        crystalline_cluster_count=tuple(int(value) for value in crystalline_cluster_count),
        largest_crystalline_cluster_atoms=tuple(int(value) for value in largest_cluster),
    )


def discover_simulation_catalog(
    root: str | Path,
    *,
    campaign_globs: Sequence[str],
    cache_root: str | Path,
    required_atom_count: int,
    required_potential_parameter_sha256: str,
    required_crystal_seed: int | None,
    require_periodic: bool,
) -> tuple[CatalogEntry, ...]:
    data_root = Path(root).expanduser().resolve()
    resolved_cache_root = Path(cache_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Simulation catalog root is missing: {data_root}")
    campaigns: list[Path] = []
    for pattern in campaign_globs:
        matched = sorted(path.resolve() for path in data_root.glob(str(pattern)) if path.is_dir())
        if not matched:
            raise FileNotFoundError(
                f"Simulation campaign glob {pattern!r} matched no directories under {data_root}."
            )
        campaigns.extend(matched)
    unique_campaigns = sorted(set(campaigns))
    entries: list[CatalogEntry] = []
    seen_trajectories: set[Path] = set()
    for campaign in unique_campaigns:
        text_trajectories = sorted(campaign.glob("**/trajectory.lammpstrj"))
        binary_trajectories = sorted(
            path
            for path in campaign.glob(f"**/trajectory{BINARY_SUFFIX}")
            if path.is_dir()
        )
        by_replica: dict[Path, Path] = {
            trajectory.parent.resolve(): trajectory.resolve()
            for trajectory in text_trajectories
        }
        for trajectory in binary_trajectories:
            by_replica[trajectory.parent.resolve()] = trajectory.resolve()
        trajectories = [by_replica[replica] for replica in sorted(by_replica)]
        if not trajectories:
            raise FileNotFoundError(
                f"Campaign contains no trajectory.lammpstrj files or verified "
                f"trajectory{BINARY_SUFFIX} directories: {campaign}"
            )
        for trajectory in trajectories:
            trajectory = trajectory.resolve()
            if trajectory in seen_trajectories:
                raise ValueError(f"Trajectory was discovered by multiple campaign globs: {trajectory}")
            seen_trajectories.add(trajectory)
            metadata = load_simulation_metadata(trajectory, campaign)
            if metadata.atom_count != int(required_atom_count):
                raise ValueError(
                    f"Catalog trajectory {trajectory} has atom_count={metadata.atom_count}; "
                    f"required {required_atom_count}."
                )
            if metadata.potential_parameter_sha256 != str(required_potential_parameter_sha256):
                raise ValueError(
                    f"Catalog trajectory {trajectory} uses potential parameter hash "
                    f"{metadata.potential_parameter_sha256}; required "
                    f"{required_potential_parameter_sha256}."
                )
            if metadata.crystal_seed != required_crystal_seed:
                raise ValueError(
                    f"Catalog trajectory {trajectory} has crystal_seed={metadata.crystal_seed}; "
                    f"required {required_crystal_seed}."
                )
            if require_periodic and metadata.boundary_conditions != ("p", "p", "p"):
                raise ValueError(
                    f"Catalog trajectory {trajectory} is not fully periodic: "
                    f"boundary={metadata.boundary_conditions}."
                )
            relative_replica = trajectory.parent.relative_to(data_root)
            run_id = "/".join(relative_replica.parts)
            cache_dir = resolved_cache_root.joinpath(*relative_replica.parts)
            entries.append(
                CatalogEntry(
                    trajectory_path=trajectory,
                    run_id=run_id,
                    cache_dir=cache_dir,
                    metadata=metadata,
                )
            )
    if not entries:
        raise ValueError(f"Simulation catalog resolved zero trajectories under {data_root}.")
    return tuple(entries)


def validate_dump_scan(metadata: SimulationMetadata, scan: Any, trajectory_path: Path) -> None:
    if int(scan.num_atoms) != metadata.atom_count:
        raise ValueError(
            f"Dump atom count disagrees with metadata for {trajectory_path}: "
            f"dump={scan.num_atoms}, metadata={metadata.atom_count}."
        )
    if int(scan.frame_count) != metadata.frame_count:
        raise ValueError(
            f"Dump frame count disagrees with progress metadata for {trajectory_path}: "
            f"dump={scan.frame_count}, metadata={metadata.frame_count}."
        )
    expected_timesteps = metadata.first_dump_timestep + np.asarray(
        metadata.progress_steps, dtype=np.int64
    )
    if not np.array_equal(np.asarray(scan.timesteps, dtype=np.int64), expected_timesteps):
        raise ValueError(
            f"Dump timesteps disagree with equilibration/progress metadata for {trajectory_path}."
        )
    if tuple(scan.atom_columns) != ("id", "type", "x", "y", "z"):
        raise ValueError(
            f"Dump columns are not the required tracked Cartesian coordinates for {trajectory_path}: "
            f"{scan.atom_columns}."
        )


__all__ = [
    "CatalogEntry",
    "SimulationMetadata",
    "discover_simulation_catalog",
    "load_simulation_metadata",
    "validate_dump_scan",
]
