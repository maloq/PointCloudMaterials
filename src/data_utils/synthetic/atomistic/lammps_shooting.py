"""Position-conditioned LAMMPS shooting ensembles from archived MD frames.

The repository's archived homogeneous-crystallization trajectories contain
coordinates and cells, but not velocities or the internal state of the NPT
thermostat/barostat.  Consequently this module deliberately implements new
fixed-cell Langevin-NVT shots with independently sampled momenta.  These are
conditional future ensembles from a recorded position, not exact restarts of
the original NPT path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.temporal_vamp.simulation_catalog import (
    CatalogEntry,
    discover_simulation_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = 1
EXPECTED_ATOM_COUNT = 70_304
LAMMPS_MAX_SEED = 900_000_000


@dataclass(frozen=True)
class ShootingConfig:
    config_path: Path
    output_root: Path
    source_root: Path
    source_campaign_globs: tuple[str, ...]
    temperatures_K: tuple[float, ...]
    expected_source_counts: dict[str, int]
    parent_offsets_ps: tuple[float, ...]
    branches_per_parent: int
    campaign_seed: int
    validation_source_velocity_seeds: tuple[int, ...]
    library_potential: Path
    parameter_potential: Path
    library_sha256: str
    parameter_sha256: str
    timestep_fs: float
    duration_ps: float
    sample_interval_steps: int
    thermostat_time_fs: float
    mpi_ranks: int
    launcher: str
    partition: str
    time_limit: str
    memory: str
    array_concurrency: int

    @property
    def run_steps(self) -> int:
        return int(round(self.duration_ps * 1000.0 / self.timestep_fs))

    @property
    def expected_frame_count(self) -> int:
        return self.run_steps // self.sample_interval_steps + 1


def _mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], *, context: str, path: Path
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise KeyError(f"{path}: unsupported keys in {context}: {unknown}.")


def _repo_path(value: Any, *, context: str, path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path}: {context} must be a non-empty path string.")
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = REPOSITORY_ROOT / resolved
    return resolved.resolve()


def _positive_float(value: Any, *, context: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be an explicit number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path}: {context} must be finite and > 0, got {result}.")
    return result


def _positive_integer(value: Any, *, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{path}: {context} must be a positive integer, got {value!r}.")
    return value


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object, got {type(document).__name__}.")
    return document


def load_shooting_config(path: str | Path) -> ShootingConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: top-level configuration must be a mapping.")
    _reject_unknown(
        raw,
        {"campaign", "sources", "potential", "dynamics", "execution"},
        context="top level",
        path=config_path,
    )
    campaign = _mapping(raw, "campaign", config_path)
    sources = _mapping(raw, "sources", config_path)
    potential = _mapping(raw, "potential", config_path)
    dynamics = _mapping(raw, "dynamics", config_path)
    execution = _mapping(raw, "execution", config_path)
    _reject_unknown(
        campaign,
        {"output_root", "branches_per_parent", "campaign_seed", "validation_source_velocity_seeds"},
        context="campaign",
        path=config_path,
    )
    _reject_unknown(
        sources,
        {"root", "campaign_globs", "temperatures_K", "expected_source_counts", "parent_offsets_ps"},
        context="sources",
        path=config_path,
    )
    _reject_unknown(
        potential,
        {"library_file", "parameter_file", "library_sha256", "parameter_sha256"},
        context="potential",
        path=config_path,
    )
    _reject_unknown(
        dynamics,
        {"timestep_fs", "duration_ps", "sample_interval_steps", "thermostat_time_fs"},
        context="dynamics",
        path=config_path,
    )
    _reject_unknown(
        execution,
        {
            "mpi_ranks",
            "launcher",
            "partition",
            "time_limit",
            "memory",
            "array_concurrency",
        },
        context="execution",
        path=config_path,
    )

    campaign_globs = sources.get("campaign_globs")
    temperatures = sources.get("temperatures_K")
    source_counts = sources.get("expected_source_counts")
    offsets = sources.get("parent_offsets_ps")
    validation_seeds = campaign.get("validation_source_velocity_seeds")
    if not isinstance(campaign_globs, list) or not all(
        isinstance(value, str) and value for value in campaign_globs
    ):
        raise TypeError(f"{config_path}: sources.campaign_globs must be a non-empty list of strings.")
    if not campaign_globs:
        raise ValueError(f"{config_path}: sources.campaign_globs cannot be empty.")
    if not isinstance(temperatures, list) or not temperatures:
        raise TypeError(f"{config_path}: sources.temperatures_K must be a non-empty list.")
    temperatures_K = tuple(
        _positive_float(value, context="sources.temperatures_K[]", path=config_path)
        for value in temperatures
    )
    if len(set(temperatures_K)) != len(temperatures_K):
        raise ValueError(f"{config_path}: sources.temperatures_K must be unique.")
    if not isinstance(source_counts, dict) or set(source_counts) != {
        f"{temperature:g}" for temperature in temperatures_K
    }:
        raise ValueError(
            f"{config_path}: sources.expected_source_counts keys must exactly match "
            f"temperatures_K; got {source_counts!r}."
        )
    expected_source_counts = {
        key: _positive_integer(value, context=f"sources.expected_source_counts.{key}", path=config_path)
        for key, value in source_counts.items()
    }
    if not isinstance(offsets, list) or not offsets:
        raise TypeError(f"{config_path}: sources.parent_offsets_ps must be a non-empty list.")
    parent_offsets_ps = tuple(float(value) for value in offsets)
    if any(not math.isfinite(value) or value >= 0.0 for value in parent_offsets_ps):
        raise ValueError(
            f"{config_path}: every parent offset must be finite and strictly before nucleation; "
            f"got {parent_offsets_ps}."
        )
    if len(set(parent_offsets_ps)) != len(parent_offsets_ps):
        raise ValueError(f"{config_path}: sources.parent_offsets_ps must be unique.")
    if not isinstance(validation_seeds, list) or not validation_seeds or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in validation_seeds
    ):
        raise TypeError(
            f"{config_path}: campaign.validation_source_velocity_seeds must be a non-empty integer list."
        )

    sha_values = {}
    for key in ("library_sha256", "parameter_sha256"):
        value = potential.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise TypeError(f"{config_path}: potential.{key} must be 64 lowercase hexadecimal characters.")
        sha_values[key] = value
    for key in ("partition", "time_limit", "memory"):
        value = execution.get(key)
        if not isinstance(value, str) or not value:
            raise TypeError(f"{config_path}: execution.{key} must be a non-empty string.")
    launcher = execution.get("launcher")
    if launcher != "srun_pmi2":
        raise ValueError(
            f"{config_path}: execution.launcher must be 'srun_pmi2' for the cluster's "
            f"Slurm-managed conda MPICH launch, got {launcher!r}."
        )

    result = ShootingConfig(
        config_path=config_path,
        output_root=_repo_path(campaign.get("output_root"), context="campaign.output_root", path=config_path),
        source_root=_repo_path(sources.get("root"), context="sources.root", path=config_path),
        source_campaign_globs=tuple(campaign_globs),
        temperatures_K=temperatures_K,
        expected_source_counts=expected_source_counts,
        parent_offsets_ps=parent_offsets_ps,
        branches_per_parent=_positive_integer(campaign.get("branches_per_parent"), context="campaign.branches_per_parent", path=config_path),
        campaign_seed=_positive_integer(campaign.get("campaign_seed"), context="campaign.campaign_seed", path=config_path),
        validation_source_velocity_seeds=tuple(int(value) for value in validation_seeds),
        library_potential=_repo_path(potential.get("library_file"), context="potential.library_file", path=config_path),
        parameter_potential=_repo_path(potential.get("parameter_file"), context="potential.parameter_file", path=config_path),
        library_sha256=sha_values["library_sha256"],
        parameter_sha256=sha_values["parameter_sha256"],
        timestep_fs=_positive_float(dynamics.get("timestep_fs"), context="dynamics.timestep_fs", path=config_path),
        duration_ps=_positive_float(dynamics.get("duration_ps"), context="dynamics.duration_ps", path=config_path),
        sample_interval_steps=_positive_integer(dynamics.get("sample_interval_steps"), context="dynamics.sample_interval_steps", path=config_path),
        thermostat_time_fs=_positive_float(dynamics.get("thermostat_time_fs"), context="dynamics.thermostat_time_fs", path=config_path),
        mpi_ranks=_positive_integer(execution.get("mpi_ranks"), context="execution.mpi_ranks", path=config_path),
        launcher=launcher,
        partition=str(execution["partition"]),
        time_limit=str(execution["time_limit"]),
        memory=str(execution["memory"]),
        array_concurrency=_positive_integer(execution.get("array_concurrency"), context="execution.array_concurrency", path=config_path),
    )
    if result.run_steps * result.timestep_fs / 1000.0 != result.duration_ps:
        raise ValueError(
            f"{config_path}: duration_ps={result.duration_ps} is not an exact integer number "
            f"of timestep_fs={result.timestep_fs} steps."
        )
    if result.run_steps % result.sample_interval_steps != 0:
        raise ValueError(
            f"{config_path}: run_steps={result.run_steps} must be divisible by "
            f"sample_interval_steps={result.sample_interval_steps}."
        )
    if len(set(result.validation_source_velocity_seeds)) != len(
        result.validation_source_velocity_seeds
    ):
        raise ValueError(f"{config_path}: validation source velocity seeds must be unique.")
    return result


def select_parent_frame_indices(
    frame_times_ps: np.ndarray,
    *,
    nucleation_time_ps: float,
    offsets_ps: tuple[float, ...],
) -> tuple[int, ...]:
    times = np.asarray(frame_times_ps, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Expected strictly increasing 1D frame times, got shape={times.shape}.")
    indices: list[int] = []
    half_interval = float(np.min(np.diff(times))) / 2.0 + 1.0e-9
    for offset in offsets_ps:
        target = float(nucleation_time_ps) + float(offset)
        if target < times[0] or target > times[-1]:
            raise ValueError(
                f"Parent target time {target} ps (nucleation={nucleation_time_ps}, "
                f"offset={offset}) is outside [{times[0]}, {times[-1]}] ps."
            )
        index = int(np.argmin(np.abs(times - target)))
        if abs(float(times[index]) - target) > half_interval:
            raise ValueError(
                f"No source frame is sufficiently close to target={target} ps; nearest is "
                f"frame {index} at {times[index]} ps."
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"Parent offsets {offsets_ps} resolve to duplicate frame indices {indices}."
        )
    return tuple(indices)


def branch_random_seeds(campaign_seed: int, parent_index: int, shot_index: int) -> tuple[int, int]:
    state = np.random.SeedSequence(
        [int(campaign_seed), int(parent_index), int(shot_index)]
    ).generate_state(2, dtype=np.uint32)
    velocity_seed = int(state[0] % (LAMMPS_MAX_SEED - 1)) + 1
    thermostat_seed = int(state[1] % (LAMMPS_MAX_SEED - 1)) + 1
    if velocity_seed == thermostat_seed:
        thermostat_seed = thermostat_seed % (LAMMPS_MAX_SEED - 1) + 1
    return velocity_seed, thermostat_seed


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
    return f"""# Position-conditioned shooting branch generated by PointCloudMaterials.
# This is a new fixed-cell Langevin-NVT path, not an exact NPT restart.
log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_data ../../parents/{parent_id}/parent.lammps.data

mass 1 {atomic_masses[atomic_numbers['Al']]:.12g}
pair_style meam
pair_coeff * * ../../potential/Lee2003_Al.library.meam Al ../../potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {timestep_fs / 1000.0:.12g}
velocity all create {temperature_K:.12g} {velocity_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all langevin {temperature_K:.12g} {temperature_K:.12g} {thermostat_time_fs / 1000.0:.12g} {thermostat_seed} zero yes

thermo {sample_interval_steps}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
dump trajectory all custom {sample_interval_steps} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory sort id format line \"%d %d %.9g %.9g %.9g %.9g %.9g %.9g\"

print \"SHOOTING_BEGIN {branch_id} PARENT {parent_id}\"
run {run_steps}
write_restart final.restart.bin
print \"SHOOTING_COMPLETE {branch_id} PARENT {parent_id}\"
"""


def _source_npz(entry: CatalogEntry) -> Path:
    path = entry.trajectory_path.with_suffix(".npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"Catalog trajectory has no repository-produced coordinate archive: {path}."
        )
    return path


def _source_split(entry: CatalogEntry, config: ShootingConfig) -> str:
    return (
        "validation"
        if entry.metadata.velocity_seed in config.validation_source_velocity_seeds
        else "train"
    )


def _selected_sources(config: ShootingConfig) -> tuple[CatalogEntry, ...]:
    catalog = discover_simulation_catalog(
        config.source_root,
        campaign_globs=config.source_campaign_globs,
        cache_root=config.output_root / "source_catalog_cache",
        required_atom_count=EXPECTED_ATOM_COUNT,
        required_potential_parameter_sha256=config.parameter_sha256,
        required_crystal_seed=None,
        require_periodic=True,
    )
    selected = tuple(
        entry for entry in catalog if entry.metadata.temperature_K in config.temperatures_K
    )
    observed_counts = {
        f"{temperature:g}": sum(
            entry.metadata.temperature_K == temperature for entry in selected
        )
        for temperature in config.temperatures_K
    }
    if observed_counts != config.expected_source_counts:
        raise RuntimeError(
            f"Selected source counts do not match the checksum-bound campaign design: "
            f"observed={observed_counts}, expected={config.expected_source_counts}."
        )
    for entry in selected:
        if not entry.metadata.nucleation_observed or entry.metadata.nucleation_time_ps is None:
            raise RuntimeError(
                f"Selected transition source has no detected nucleation event: {entry.run_id}."
            )
    return tuple(sorted(selected, key=lambda item: item.run_id))


def prepare_campaign(config: ShootingConfig) -> dict[str, Any]:
    if config.output_root.exists():
        raise FileExistsError(
            f"Shooting campaign output already exists and will not be overwritten: {config.output_root}."
        )
    for potential_path, expected_hash in (
        (config.library_potential, config.library_sha256),
        (config.parameter_potential, config.parameter_sha256),
    ):
        if not potential_path.is_file():
            raise FileNotFoundError(f"Required 2NN-MEAM potential is absent: {potential_path}.")
        observed_hash = _sha256(potential_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Potential checksum mismatch for {potential_path}: expected={expected_hash}, "
                f"observed={observed_hash}."
            )

    sources = _selected_sources(config)
    config.output_root.mkdir(parents=True)
    for name in ("parents", "branches", "potential", "slurm"):
        (config.output_root / name).mkdir()
    shutil.copy2(config.library_potential, config.output_root / "potential" / config.library_potential.name)
    shutil.copy2(config.parameter_potential, config.output_root / "potential" / config.parameter_potential.name)
    shutil.copy2(config.config_path, config.output_root / "campaign_config.yaml")

    parents: list[dict[str, Any]] = []
    source_documents: list[dict[str, Any]] = []
    for source_index, entry in enumerate(sources):
        metadata = entry.metadata
        assert metadata.nucleation_time_ps is not None
        frame_indices = select_parent_frame_indices(
            np.asarray(metadata.progress_times_ps),
            nucleation_time_ps=metadata.nucleation_time_ps,
            offsets_ps=config.parent_offsets_ps,
        )
        archive_path = _source_npz(entry)
        archive_sha256 = _sha256(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {"step", "positions_A", "cell_vectors_A"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"{archive_path}: source archive is missing arrays {missing}.")
            steps = np.asarray(archive["step"], dtype=np.int64)
            if not np.array_equal(steps, np.asarray(metadata.progress_steps, dtype=np.int64)):
                raise RuntimeError(
                    f"{archive_path}: coordinate steps disagree with crystallization progress metadata."
                )
            for offset_index, (offset_ps, frame_index) in enumerate(
                zip(config.parent_offsets_ps, frame_indices)
            ):
                positions_A = np.asarray(archive["positions_A"][frame_index], dtype=np.float64)
                cell_A = np.asarray(archive["cell_vectors_A"][frame_index], dtype=np.float64)
                if positions_A.shape != (EXPECTED_ATOM_COUNT, 3) or cell_A.shape != (3, 3):
                    raise RuntimeError(
                        f"{archive_path}: invalid parent shapes at frame={frame_index}: "
                        f"positions={positions_A.shape}, cell={cell_A.shape}."
                    )
                off_diagonal = cell_A.copy()
                off_diagonal[np.diag_indices(3)] = 0.0
                if np.any(np.abs(off_diagonal) > 1.0e-10):
                    raise RuntimeError(
                        f"{archive_path}: LAMMPS shooting currently requires the repository's "
                        f"orthogonal cells; got cell={cell_A.tolist()}."
                    )
                parent_index = len(parents)
                phase = f"pre_nucleation_{abs(offset_ps):g}ps"
                parent_id = (
                    f"parent_{parent_index:03d}_T{metadata.temperature_K:g}_"
                    f"v{metadata.velocity_seed}_{phase}"
                )
                parent_dir = config.output_root / "parents" / parent_id
                parent_dir.mkdir()
                atoms = Atoms(
                    "Al" * EXPECTED_ATOM_COUNT,
                    positions=positions_A,
                    cell=cell_A,
                    pbc=True,
                )
                data_path = parent_dir / "parent.lammps.data"
                write(
                    data_path,
                    atoms,
                    format="lammps-data",
                    atom_style="atomic",
                    specorder=("Al",),
                )
                parent = {
                    "parent_index": parent_index,
                    "parent_id": parent_id,
                    "source_index": source_index,
                    "source_run_id": entry.run_id,
                    "source_split": _source_split(entry, config),
                    "source_velocity_seed": metadata.velocity_seed,
                    "temperature_K": metadata.temperature_K,
                    "nucleation_time_ps": metadata.nucleation_time_ps,
                    "parent_offset_ps": offset_ps,
                    "phase": phase,
                    "source_frame_index": frame_index,
                    "source_frame_step": int(steps[frame_index]),
                    "source_frame_time_ps": float(metadata.progress_times_ps[frame_index]),
                    "source_crystalline_fraction": float(metadata.crystalline_fraction[frame_index]),
                    "source_largest_crystalline_cluster_atoms": int(
                        metadata.largest_crystalline_cluster_atoms[frame_index]
                    ),
                    "data_file": str(data_path.relative_to(config.output_root)),
                    "data_sha256": _sha256(data_path),
                }
                _write_json_atomic(parent_dir / "metadata.json", parent)
                parents.append(parent)
        source_documents.append(
            {
                "source_index": source_index,
                "run_id": entry.run_id,
                "trajectory_lammpstrj": str(entry.trajectory_path),
                "trajectory_npz": str(archive_path),
                "trajectory_npz_sha256": archive_sha256,
                "temperature_K": metadata.temperature_K,
                "velocity_seed": metadata.velocity_seed,
                "nucleation_time_ps": metadata.nucleation_time_ps,
                "split": _source_split(entry, config),
            }
        )

    branches: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for parent in parents:
        for shot_index in range(config.branches_per_parent):
            branch_index = len(branches)
            velocity_seed, thermostat_seed = branch_random_seeds(
                config.campaign_seed, int(parent["parent_index"]), shot_index
            )
            for seed in (velocity_seed, thermostat_seed):
                if seed in observed_seeds:
                    raise RuntimeError(f"Random-seed collision while preparing branch {branch_index}: {seed}.")
                observed_seeds.add(seed)
            branch_id = f"branch_{branch_index:04d}_{parent['parent_id']}_shot_{shot_index:02d}"
            branch_dir = config.output_root / "branches" / branch_id
            branch_dir.mkdir()
            (branch_dir / "in.lammps").write_text(
                render_lammps_input(
                    parent_id=str(parent["parent_id"]),
                    branch_id=branch_id,
                    temperature_K=float(parent["temperature_K"]),
                    velocity_seed=velocity_seed,
                    thermostat_seed=thermostat_seed,
                    timestep_fs=config.timestep_fs,
                    thermostat_time_fs=config.thermostat_time_fs,
                    sample_interval_steps=config.sample_interval_steps,
                    run_steps=config.run_steps,
                ),
                encoding="utf-8",
            )
            branch = {
                "branch_index": branch_index,
                "branch_id": branch_id,
                "branch_dir": str(branch_dir.relative_to(config.output_root)),
                "parent_index": parent["parent_index"],
                "parent_id": parent["parent_id"],
                "source_run_id": parent["source_run_id"],
                "source_split": parent["source_split"],
                "source_velocity_seed": parent["source_velocity_seed"],
                "temperature_K": parent["temperature_K"],
                "phase": parent["phase"],
                "shot_index": shot_index,
                "velocity_seed": velocity_seed,
                "thermostat_seed": thermostat_seed,
            }
            _write_json_atomic(branch_dir / "metadata.json", branch)
            branches.append(branch)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_type": "position_conditioned_langevin_nvt_shooting",
        "scientific_contract": {
            "exact_restart": False,
            "reason": (
                "Archived sources contain positions and cells but not velocities or serialized "
                "NPT thermostat/barostat state."
            ),
            "interpretation": (
                "Independent fixed-cell Langevin-NVT futures conditioned on each archived "
                "position under the same 2NN-MEAM Hamiltonian."
            ),
            "no_equilibration_after_branching": True,
        },
        "atom_count": EXPECTED_ATOM_COUNT,
        "source_config": {
            "root": str(config.source_root),
            "campaign_globs": list(config.source_campaign_globs),
            "temperatures_K": list(config.temperatures_K),
            "parent_offsets_ps": list(config.parent_offsets_ps),
            "validation_source_velocity_seeds": list(config.validation_source_velocity_seeds),
        },
        "potential": {
            "name": "Lee-Shim-Baskes 2003 Al 2NN-MEAM",
            "library_file": "potential/Lee2003_Al.library.meam",
            "library_sha256": config.library_sha256,
            "parameter_file": "potential/Lee2003_Al.meam",
            "parameter_sha256": config.parameter_sha256,
        },
        "protocol": {
            "ensemble": "fixed-cell Langevin NVT",
            "timestep_fs": config.timestep_fs,
            "duration_ps": config.duration_ps,
            "run_steps": config.run_steps,
            "sample_interval_steps": config.sample_interval_steps,
            "sample_interval_ps": config.sample_interval_steps * config.timestep_fs / 1000.0,
            "expected_frame_count": config.expected_frame_count,
            "thermostat_time_fs": config.thermostat_time_fs,
            "dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
            "independent_momenta_per_branch": True,
        },
        "execution": {
            "mpi_ranks_per_branch": config.mpi_ranks,
            "launcher": config.launcher,
            "partition": config.partition,
            "time_limit": config.time_limit,
            "memory": config.memory,
            "array_concurrency": config.array_concurrency,
        },
        "counts": {
            "sources": len(source_documents),
            "parents": len(parents),
            "branches": len(branches),
            "branches_by_split": {
                split: sum(branch["source_split"] == split for branch in branches)
                for split in ("train", "validation")
            },
        },
        "sources": source_documents,
        "parents": parents,
        "branches": branches,
    }
    _write_json_atomic(config.output_root / "manifest.json", manifest)
    _write_json_atomic(
        config.output_root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "branch_count": len(branches),
        },
    )
    _write_slurm_script(config, len(branches))
    return manifest


def _write_slurm_script(config: ShootingConfig, branch_count: int) -> None:
    script_path = config.output_root / "slurm" / "run_branch.sbatch"
    script_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_shoot
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={config.mpi_ranks}
#SBATCH --ntasks-per-node={config.mpi_ranks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={config.memory}
#SBATCH --time={config.time_limit}
#SBATCH --output={config.output_root}/slurm/%A_%a.out
#SBATCH --error={config.output_root}/slurm/%A_%a.err

set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
python {REPOSITORY_ROOT}/scripts/run_lammps_meam_shooting_campaign.py run-task \\
  --campaign-root {config.output_root} \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}"
""",
        encoding="utf-8",
    )
    script_path.chmod(0o750)
    (config.output_root / "slurm" / "array_spec.txt").write_text(
        f"0-{branch_count - 1}%{config.array_concurrency}\n", encoding="utf-8"
    )
    summary_path = config.output_root / "slurm" / "summarize.sbatch"
    summary_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_shoot_summary
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output={config.output_root}/slurm/%j_summary.out
#SBATCH --error={config.output_root}/slurm/%j_summary.err

set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
python {REPOSITORY_ROOT}/scripts/run_lammps_meam_shooting_campaign.py summarize \\
  --campaign-root {config.output_root}
""",
        encoding="utf-8",
    )
    summary_path.chmod(0o750)
    controller_path = config.output_root / "slurm" / "submit_wave.sbatch"
    controller_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_meam_shoot_submit
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output={config.output_root}/slurm/%j_submit.out
#SBATCH --error={config.output_root}/slurm/%j_submit.err

set -euo pipefail
: "${{SHOOT_START:?SHOOT_START must identify the first branch in this wave}}"
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
python {REPOSITORY_ROOT}/scripts/run_lammps_meam_shooting_campaign.py submit-next-wave \\
  --campaign-root {config.output_root} \\
  --start-index "${{SHOOT_START}}"
""",
        encoding="utf-8",
    )
    controller_path.chmod(0o750)


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


def _lammps_command(*, mpi_ranks: int, launcher: str) -> list[str]:
    lmp = Path(sys.prefix) / "bin" / "lmp"
    if not lmp.is_file():
        raise FileNotFoundError(f"Required pointnet LAMMPS executable is absent: {lmp}.")
    if launcher != "srun_pmi2":
        raise ValueError(f"Unsupported LAMMPS launcher {launcher!r}; expected 'srun_pmi2'.")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            "The srun_pmi2 launcher requires a Slurm allocation, but SLURM_JOB_ID is absent. "
            "Submit slurm/run_branch.sbatch instead of running a production branch locally."
        )
    srun = shutil.which("srun")
    if srun is None:
        raise FileNotFoundError("The configured srun_pmi2 launcher requires srun on PATH.")
    return [
        srun,
        "--mpi=pmi2",
        "--nodes=1",
        f"--ntasks={mpi_ranks}",
        f"--ntasks-per-node={mpi_ranks}",
        "--cpus-per-task=1",
        "--cpu-bind=cores",
        "--kill-on-bad-exit=1",
        str(lmp),
        "-in",
        "in.lammps",
    ]


def _materialize_missing_branch_input(
    root: Path, manifest: dict[str, Any], branch: dict[str, Any]
) -> Path:
    """Recreate immutable branch inputs from the campaign manifest when absent."""
    branch_dir = root / str(branch["branch_dir"])
    if branch_dir.exists():
        required = (branch_dir / "metadata.json", branch_dir / "in.lammps")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Incomplete branch input directory {branch_dir}: missing {missing}. "
                "Inspect it before resubmitting; existing directories are never repaired in place."
            )
        return branch_dir

    protocol = manifest["protocol"]
    branch_dir.mkdir()
    (branch_dir / "in.lammps").write_text(
        render_lammps_input(
            parent_id=str(branch["parent_id"]),
            branch_id=str(branch["branch_id"]),
            temperature_K=float(branch["temperature_K"]),
            velocity_seed=int(branch["velocity_seed"]),
            thermostat_seed=int(branch["thermostat_seed"]),
            timestep_fs=float(protocol["timestep_fs"]),
            thermostat_time_fs=float(protocol["thermostat_time_fs"]),
            sample_interval_steps=int(protocol["sample_interval_steps"]),
            run_steps=int(protocol["run_steps"]),
        ),
        encoding="utf-8",
    )
    _write_json_atomic(branch_dir / "metadata.json", branch)
    return branch_dir


def run_branch(campaign_root: str | Path, task_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest["branches"]
    if not isinstance(branches, list):
        raise TypeError(f"{root / 'manifest.json'}: branches must be a list.")
    index = int(task_index)
    if index < 0 or index >= len(branches):
        raise IndexError(f"task_index={index} is outside [0, {len(branches)}).")
    branch = branches[index]
    branch_dir = root / branch["branch_dir"]
    outcome_path = branch_dir / "outcome.json"
    if outcome_path.is_file():
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Existing outcome is not complete: {outcome_path}.")
        print(f"Branch {branch['branch_id']} is already complete; leaving artifacts unchanged.")
        return outcome
    branch_dir = _materialize_missing_branch_input(root, manifest, branch)
    partial = [
        path.name
        for path in (branch_dir / "trajectory.lammpstrj", branch_dir / "final.restart.bin")
        if path.exists()
    ]
    if partial:
        raise RuntimeError(
            f"Branch {branch['branch_id']} has partial artifacts {partial} but no complete outcome. "
            f"Inspect {branch_dir} before explicitly removing or archiving the partial run."
        )
    status_path = branch_dir / "status.json"
    _write_json_atomic(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    mpi_ranks = int(manifest["execution"]["mpi_ranks_per_branch"])
    launcher = str(manifest["execution"]["launcher"])
    command = _lammps_command(mpi_ranks=mpi_ranks, launcher=launcher)
    running_status = _load_json(status_path)
    running_status["launcher"] = launcher
    running_status["command"] = command
    _write_json_atomic(status_path, running_status)
    started = time.monotonic()
    stdout_path = branch_dir / "lammps.stdout.log"
    with stdout_path.open("wb") as stdout:
        completed = subprocess.run(
            command,
            cwd=branch_dir,
            env=_lammps_environment(),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_seconds = time.monotonic() - started
    if completed.returncode != 0:
        _write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "return_code": completed.returncode,
                "stdout": str(stdout_path),
            },
        )
        raise RuntimeError(
            f"LAMMPS branch {branch['branch_id']} failed with return code "
            f"{completed.returncode}; inspect {stdout_path}."
        )

    trajectory = branch_dir / "trajectory.lammpstrj"
    restart = branch_dir / "final.restart.bin"
    for artifact in (trajectory, restart, branch_dir / "lammps.log"):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(
                f"LAMMPS reported success but required artifact is absent or empty: {artifact}."
            )
    scan = TemporalLAMMPSDumpDataset.scan_dump_file(trajectory)
    expected_frame_count = int(manifest["protocol"]["expected_frame_count"])
    expected_steps = np.arange(
        0,
        int(manifest["protocol"]["run_steps"]) + 1,
        int(manifest["protocol"]["sample_interval_steps"]),
        dtype=np.int64,
    )
    if (
        scan.num_atoms != int(manifest["atom_count"])
        or scan.frame_count != expected_frame_count
        or tuple(scan.atom_columns) != tuple(manifest["protocol"]["dump_columns"])
        or not np.array_equal(scan.timesteps, expected_steps)
    ):
        raise RuntimeError(
            f"Completed dump validation failed for {trajectory}: atoms={scan.num_atoms}, "
            f"frames={scan.frame_count}, columns={scan.atom_columns}, "
            f"timesteps=[{scan.timesteps[0]}, {scan.timesteps[-1]}]."
        )
    outcome = {
        **branch,
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "launcher": launcher,
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "frame_count": scan.frame_count,
        "first_timestep": int(scan.timesteps[0]),
        "last_timestep": int(scan.timesteps[-1]),
        "trajectory_size_bytes": trajectory.stat().st_size,
        "restart_size_bytes": restart.stat().st_size,
    }
    _write_json_atomic(outcome_path, outcome)
    _write_json_atomic(status_path, outcome)
    print(json.dumps(outcome, indent=2))
    return outcome


def summarize_campaign(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    outcomes: list[dict[str, Any]] = []
    missing: list[str] = []
    for branch in manifest["branches"]:
        path = root / branch["branch_dir"] / "outcome.json"
        if not path.is_file():
            missing.append(str(branch["branch_id"]))
            continue
        outcome = _load_json(path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Branch outcome is not complete: {path}.")
        outcomes.append(outcome)
    if missing:
        raise RuntimeError(
            f"Cannot summarize an incomplete campaign: {len(missing)} branches have no "
            f"complete outcome; first missing={missing[:10]}."
        )
    elapsed = np.asarray([outcome["elapsed_seconds"] for outcome in outcomes], dtype=np.float64)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "branch_count": len(outcomes),
        "branch_counts_by_temperature_and_split": {
            f"{temperature:g}K_{split}": sum(
                outcome["temperature_K"] == temperature and outcome["source_split"] == split
                for outcome in outcomes
            )
            for temperature in manifest["source_config"]["temperatures_K"]
            for split in ("train", "validation")
        },
        "elapsed_seconds": {
            "minimum": float(elapsed.min()),
            "median": float(np.median(elapsed)),
            "maximum": float(elapsed.max()),
            "sum": float(elapsed.sum()),
        },
        "trajectory_size_bytes": int(
            sum(outcome["trajectory_size_bytes"] for outcome in outcomes)
        ),
    }
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "status.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def submit_next_wave(campaign_root: str | Path, start_index: int) -> dict[str, Any]:
    """Submit one QOS-sized array wave and its dependent successor controller.

    The successor uses ``afterany`` so an isolated node or launcher failure cannot
    strand every later branch.  The final strict summary still fails if any branch
    lacks a validated complete outcome.
    """
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest["branches"]
    if not isinstance(branches, list) or not branches:
        raise TypeError(f"{root / 'manifest.json'}: branches must be a non-empty list.")
    start = int(start_index)
    if start < 0 or start >= len(branches):
        raise IndexError(f"start_index={start} is outside [0, {len(branches)}).")
    wave_size = int(manifest["execution"]["array_concurrency"])
    stop = min(start + wave_size - 1, len(branches) - 1)
    array_spec = f"{start}-{stop}%{wave_size}"
    array_submission = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={array_spec}",
            str(root / "slurm" / "run_branch.sbatch"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    array_job_id = array_submission.stdout.strip()
    if not array_job_id.isdigit():
        raise RuntimeError(
            f"Slurm returned an invalid array job ID for wave {array_spec}: "
            f"stdout={array_submission.stdout!r}, stderr={array_submission.stderr!r}."
        )
    if stop + 1 < len(branches):
        successor_kind = "controller"
        successor_submission = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterany:{array_job_id}",
                f"--export=ALL,SHOOT_START={stop + 1}",
                str(root / "slurm" / "submit_wave.sbatch"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    else:
        successor_kind = "summary"
        successor_submission = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterany:{array_job_id}",
                str(root / "slurm" / "summarize.sbatch"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    successor_job_id = successor_submission.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(
            f"Slurm returned an invalid {successor_kind} job ID after wave {array_spec}: "
            f"stdout={successor_submission.stdout!r}, stderr={successor_submission.stderr!r}."
        )
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "submitting_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_spec": array_spec,
        "array_job_id": array_job_id,
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
    }
    chain_path = root / "slurm" / "submission_chain.jsonl"
    with chain_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json_atomic(root / "slurm" / "active_submission.json", record)
    print(json.dumps(record, indent=2))
    return record


__all__ = [
    "ShootingConfig",
    "branch_random_seeds",
    "load_shooting_config",
    "prepare_campaign",
    "render_lammps_input",
    "run_branch",
    "select_parent_frame_indices",
    "submit_next_wave",
    "summarize_campaign",
]
