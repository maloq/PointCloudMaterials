"""Transition-balanced nested LAMMPS shooting with online PTM stopping.

This producer is deliberately separate from ``lammps_shooting``.  The original
campaign sampled one independent momentum and one independent Langevin stream
per branch at fixed time offsets.  Here a parent is selected by its connected
PTM cluster size, each momentum seed is reused across several thermostat seeds,
and a single uninterrupted LAMMPS process stops after persistent basin arrival.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

from src.data_utils.shooting_binary import (
    ShootingBinaryTrajectory,
    convert_shooting_trajectory,
)
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.data_utils.synthetic.atomistic.transition_analysis import (
    CRYSTALLINE_STRUCTURE_TYPES,
)
from src.temporal_vamp.simulation_catalog import (
    CatalogEntry,
    discover_simulation_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = 1
EXPECTED_ATOM_COUNT = 70_304
LAMMPS_MAX_SEED = 900_000_000
TEMPERATURES_K = (400.0, 450.0, 500.0)
SPLIT_NAMES = ("optimization", "model_selection", "final_validation")
NESTED_STORAGE_DTYPE = "float16"
FIXED_HORIZONS_PS = (6.0, 12.0, 24.0)


@dataclass(frozen=True)
class SourceGroup:
    root: Path
    campaign_globs: tuple[str, ...]


@dataclass(frozen=True)
class TemperatureProtocol:
    temperature_K: float
    expected_source_count: int
    expected_basin_a_max_cluster_atoms: int
    maximum_duration_ps: float


@dataclass(frozen=True)
class NestedShootingConfig:
    config_path: Path
    output_root: Path
    campaign_seed: int
    source_groups: tuple[SourceGroup, ...]
    temperatures: tuple[TemperatureProtocol, ...]
    calibration_duration_ps: float
    calibration_quantile: float
    transition_parents_per_temperature: int
    basin_control_parents_per_temperature: int
    maximum_parents_per_source_run: int
    momentum_samples_per_parent: int
    thermostat_futures_per_momentum: int
    crystal_basin_min_cluster_atoms: int
    basin_persistence_frames: int
    ptm_rmsd_cutoff: float
    crystalline_cluster_cutoff_A: float
    timestep_fs: float
    thermostat_time_fs: float
    monitor_interval_ps: float
    library_potential: Path
    parameter_potential: Path
    library_sha256: str
    parameter_sha256: str
    mpi_ranks: int
    partition: str
    time_limit: str
    memory: str
    array_concurrency: int

    @property
    def monitor_interval_steps(self) -> int:
        return int(round(self.monitor_interval_ps * 1000.0 / self.timestep_fs))


@dataclass(frozen=True)
class CandidateFrame:
    entry: CatalogEntry
    source_run_id: str
    source_split: str
    frame_index: int
    source_step: int
    source_time_ps: float
    crystalline_fraction: float
    largest_cluster_atoms: int
    basin_role: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object, got {type(document).__name__}.")
    return document


def _write_json_atomic(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _resolve_path(value: object, *, context: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{config_path}: {context} must be a nonempty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _positive_int(value: object, *, context: str, config_path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{config_path}: {context} must be a positive integer, got {value!r}.")
    return value


def _positive_float(value: object, *, context: str, config_path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{config_path}: {context} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{config_path}: {context} must be finite and positive, got {result}.")
    return result


def _required_mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key} must be a mapping.")
    return value


def load_nested_shooting_config(path: str | Path) -> NestedShootingConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: top-level configuration must be a mapping.")
    expected_top = {
        "campaign",
        "sources",
        "selection",
        "futures",
        "basins",
        "dynamics",
        "potential",
        "execution",
    }
    if set(raw) != expected_top:
        raise KeyError(
            f"{config_path}: top-level keys must be exactly {sorted(expected_top)}, "
            f"got {sorted(raw)}."
        )
    campaign = _required_mapping(raw, "campaign", config_path)
    sources = _required_mapping(raw, "sources", config_path)
    selection = _required_mapping(raw, "selection", config_path)
    futures = _required_mapping(raw, "futures", config_path)
    basins = _required_mapping(raw, "basins", config_path)
    dynamics = _required_mapping(raw, "dynamics", config_path)
    potential = _required_mapping(raw, "potential", config_path)
    execution = _required_mapping(raw, "execution", config_path)

    source_group_values = sources.get("groups")
    if not isinstance(source_group_values, list) or not source_group_values:
        raise TypeError(f"{config_path}: sources.groups must be a nonempty list.")
    source_groups: list[SourceGroup] = []
    for index, value in enumerate(source_group_values):
        if not isinstance(value, dict) or set(value) != {"root", "campaign_globs"}:
            raise TypeError(
                f"{config_path}: sources.groups[{index}] must contain only root and "
                "campaign_globs."
            )
        globs = value["campaign_globs"]
        if not isinstance(globs, list) or not globs or not all(
            isinstance(item, str) and item for item in globs
        ):
            raise TypeError(
                f"{config_path}: sources.groups[{index}].campaign_globs must be a "
                "nonempty string list."
            )
        source_groups.append(
            SourceGroup(
                root=_resolve_path(
                    value["root"],
                    context=f"sources.groups[{index}].root",
                    config_path=config_path,
                ),
                campaign_globs=tuple(globs),
            )
        )

    temperature_values = sources.get("temperatures")
    expected_temperature_keys = {f"{temperature:g}" for temperature in TEMPERATURES_K}
    if not isinstance(temperature_values, dict) or set(temperature_values) != expected_temperature_keys:
        raise KeyError(
            f"{config_path}: sources.temperatures keys must be exactly "
            f"{sorted(expected_temperature_keys)}."
        )
    temperatures: list[TemperatureProtocol] = []
    for temperature_K in TEMPERATURES_K:
        key = f"{temperature_K:g}"
        value = temperature_values[key]
        required_keys = {
            "expected_source_count",
            "expected_basin_a_max_cluster_atoms",
            "maximum_duration_ps",
        }
        if not isinstance(value, dict) or set(value) != required_keys:
            raise KeyError(
                f"{config_path}: sources.temperatures.{key} keys must be exactly "
                f"{sorted(required_keys)}."
            )
        temperatures.append(
            TemperatureProtocol(
                temperature_K=temperature_K,
                expected_source_count=_positive_int(
                    value["expected_source_count"],
                    context=f"sources.temperatures.{key}.expected_source_count",
                    config_path=config_path,
                ),
                expected_basin_a_max_cluster_atoms=_positive_int(
                    value["expected_basin_a_max_cluster_atoms"],
                    context=(
                        f"sources.temperatures.{key}."
                        "expected_basin_a_max_cluster_atoms"
                    ),
                    config_path=config_path,
                ),
                maximum_duration_ps=_positive_float(
                    value["maximum_duration_ps"],
                    context=f"sources.temperatures.{key}.maximum_duration_ps",
                    config_path=config_path,
                ),
            )
        )

    calibration_quantile = _positive_float(
        selection.get("liquid_calibration_quantile"),
        context="selection.liquid_calibration_quantile",
        config_path=config_path,
    )
    if calibration_quantile >= 1.0:
        raise ValueError(
            f"{config_path}: selection.liquid_calibration_quantile must be below 1."
        )
    ptm_rmsd_cutoff = _positive_float(
        basins.get("ptm_rmsd_cutoff"),
        context="basins.ptm_rmsd_cutoff",
        config_path=config_path,
    )
    if ptm_rmsd_cutoff > 1.0:
        raise ValueError(f"{config_path}: basins.ptm_rmsd_cutoff is normalized and must be <= 1.")

    hash_values: dict[str, str] = {}
    for key in ("library_sha256", "parameter_sha256"):
        value = potential.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise TypeError(f"{config_path}: potential.{key} must be lowercase SHA-256.")
        hash_values[key] = value

    result = NestedShootingConfig(
        config_path=config_path,
        output_root=_resolve_path(
            campaign.get("output_root"),
            context="campaign.output_root",
            config_path=config_path,
        ),
        campaign_seed=_positive_int(
            campaign.get("campaign_seed"),
            context="campaign.campaign_seed",
            config_path=config_path,
        ),
        source_groups=tuple(source_groups),
        temperatures=tuple(temperatures),
        calibration_duration_ps=_positive_float(
            selection.get("liquid_calibration_duration_ps"),
            context="selection.liquid_calibration_duration_ps",
            config_path=config_path,
        ),
        calibration_quantile=calibration_quantile,
        transition_parents_per_temperature=_positive_int(
            selection.get("transition_parents_per_temperature"),
            context="selection.transition_parents_per_temperature",
            config_path=config_path,
        ),
        basin_control_parents_per_temperature=_positive_int(
            selection.get("basin_control_parents_per_temperature"),
            context="selection.basin_control_parents_per_temperature",
            config_path=config_path,
        ),
        maximum_parents_per_source_run=_positive_int(
            selection.get("maximum_parents_per_source_run"),
            context="selection.maximum_parents_per_source_run",
            config_path=config_path,
        ),
        momentum_samples_per_parent=_positive_int(
            futures.get("momentum_samples_per_parent"),
            context="futures.momentum_samples_per_parent",
            config_path=config_path,
        ),
        thermostat_futures_per_momentum=_positive_int(
            futures.get("thermostat_futures_per_momentum"),
            context="futures.thermostat_futures_per_momentum",
            config_path=config_path,
        ),
        crystal_basin_min_cluster_atoms=_positive_int(
            basins.get("crystal_min_cluster_atoms"),
            context="basins.crystal_min_cluster_atoms",
            config_path=config_path,
        ),
        basin_persistence_frames=_positive_int(
            basins.get("persistence_frames"),
            context="basins.persistence_frames",
            config_path=config_path,
        ),
        ptm_rmsd_cutoff=ptm_rmsd_cutoff,
        crystalline_cluster_cutoff_A=_positive_float(
            basins.get("cluster_connectivity_cutoff_A"),
            context="basins.cluster_connectivity_cutoff_A",
            config_path=config_path,
        ),
        timestep_fs=_positive_float(
            dynamics.get("timestep_fs"),
            context="dynamics.timestep_fs",
            config_path=config_path,
        ),
        thermostat_time_fs=_positive_float(
            dynamics.get("thermostat_time_fs"),
            context="dynamics.thermostat_time_fs",
            config_path=config_path,
        ),
        monitor_interval_ps=_positive_float(
            dynamics.get("monitor_interval_ps"),
            context="dynamics.monitor_interval_ps",
            config_path=config_path,
        ),
        library_potential=_resolve_path(
            potential.get("library_file"),
            context="potential.library_file",
            config_path=config_path,
        ),
        parameter_potential=_resolve_path(
            potential.get("parameter_file"),
            context="potential.parameter_file",
            config_path=config_path,
        ),
        library_sha256=hash_values["library_sha256"],
        parameter_sha256=hash_values["parameter_sha256"],
        mpi_ranks=_positive_int(
            execution.get("mpi_ranks"),
            context="execution.mpi_ranks",
            config_path=config_path,
        ),
        partition=str(execution.get("partition")),
        time_limit=str(execution.get("time_limit")),
        memory=str(execution.get("memory")),
        array_concurrency=_positive_int(
            execution.get("array_concurrency"),
            context="execution.array_concurrency",
            config_path=config_path,
        ),
    )
    if not result.partition or not result.time_limit or not result.memory:
        raise ValueError(f"{config_path}: execution strings cannot be empty.")
    if abs(result.monitor_interval_steps * result.timestep_fs / 1000.0 - result.monitor_interval_ps) > 1e-12:
        raise ValueError(
            f"{config_path}: monitor_interval_ps={result.monitor_interval_ps} is not an "
            f"integer number of timestep_fs={result.timestep_fs} steps."
        )
    for protocol in result.temperatures:
        run_steps = int(round(protocol.maximum_duration_ps * 1000.0 / result.timestep_fs))
        if run_steps * result.timestep_fs / 1000.0 != protocol.maximum_duration_ps:
            raise ValueError(
                f"{config_path}: {protocol.maximum_duration_ps} ps at "
                f"{protocol.temperature_K:g} K is not an integer step count."
            )
        if run_steps % result.monitor_interval_steps != 0:
            raise ValueError(
                f"{config_path}: maximum duration at {protocol.temperature_K:g} K must "
                "contain a whole number of monitor intervals."
            )
    return result


def nested_random_seeds(
    campaign_seed: int,
    parent_index: int,
    momentum_index: int,
    thermostat_index: int,
) -> tuple[int, int]:
    momentum_state = np.random.SeedSequence(
        [int(campaign_seed), int(parent_index), int(momentum_index), 0x4D4F4D]
    ).generate_state(1, dtype=np.uint32)
    thermostat_state = np.random.SeedSequence(
        [
            int(campaign_seed),
            int(parent_index),
            int(momentum_index),
            int(thermostat_index),
            0x54484552,
        ]
    ).generate_state(1, dtype=np.uint32)
    momentum_seed = int(momentum_state[0] % (LAMMPS_MAX_SEED - 1)) + 1
    thermostat_seed = int(thermostat_state[0] % (LAMMPS_MAX_SEED - 1)) + 1
    if momentum_seed == thermostat_seed:
        thermostat_seed = thermostat_seed % (LAMMPS_MAX_SEED - 1) + 1
    return momentum_seed, thermostat_seed


def multirate_output_steps(
    *, timestep_fs: float, maximum_duration_ps: float
) -> tuple[int, ...]:
    targets_ps: list[float] = []
    targets_ps.extend(np.arange(0.0, 0.3000000001, 0.03).tolist())
    targets_ps.extend(np.arange(0.4, 3.0000000001, 0.1).tolist())
    if maximum_duration_ps > 3.0:
        targets_ps.extend(
            np.arange(3.3, maximum_duration_ps + 1.0e-10, 0.3).tolist()
        )
    steps = tuple(
        sorted(
            {
                int(round(time_ps * 1000.0 / timestep_fs))
                for time_ps in targets_ps
                if time_ps <= maximum_duration_ps + 1.0e-10
            }
        )
    )
    maximum_step = int(round(maximum_duration_ps * 1000.0 / timestep_fs))
    if not steps or steps[0] != 0 or steps[-1] != maximum_step:
        raise RuntimeError(
            f"Multirate schedule does not span [0, {maximum_step}]: first/last="
            f"{steps[:1]}/{steps[-1:]}."
        )
    if any(next_step <= step for step, next_step in zip(steps, steps[1:])):
        raise RuntimeError(f"Multirate output steps are not strictly increasing: {steps}.")
    return steps


def _source_npz(entry: CatalogEntry) -> Path:
    path = entry.trajectory_path.parent / "trajectory.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Source catalog entry has no trajectory.npz: {path}")
    return path


def _discover_sources(config: NestedShootingConfig) -> tuple[tuple[str, CatalogEntry], ...]:
    discovered: list[tuple[str, CatalogEntry]] = []
    seen_replica_dirs: set[Path] = set()
    for group_index, group in enumerate(config.source_groups):
        entries = discover_simulation_catalog(
            group.root,
            campaign_globs=group.campaign_globs,
            cache_root=config.output_root / "source_catalog_cache" / f"group_{group_index:02d}",
            required_atom_count=EXPECTED_ATOM_COUNT,
            required_potential_parameter_sha256=config.parameter_sha256,
            required_crystal_seed=None,
            require_periodic=True,
        )
        for entry in entries:
            replica_dir = entry.trajectory_path.parent.resolve()
            if replica_dir in seen_replica_dirs:
                raise RuntimeError(f"Source replica was selected more than once: {replica_dir}")
            seen_replica_dirs.add(replica_dir)
            source_run_id = f"source_group_{group_index:02d}/{entry.run_id}"
            discovered.append((source_run_id, entry))
    selected = [
        item for item in discovered if item[1].metadata.temperature_K in TEMPERATURES_K
    ]
    observed_counts = {
        f"{temperature:g}": sum(
            entry.metadata.temperature_K == temperature for _, entry in selected
        )
        for temperature in TEMPERATURES_K
    }
    expected_counts = {
        f"{protocol.temperature_K:g}": protocol.expected_source_count
        for protocol in config.temperatures
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Nested shooting source counts differ from the checksum-bound design: "
            f"observed={observed_counts}, expected={expected_counts}."
        )
    for source_run_id, entry in selected:
        if not entry.metadata.nucleation_observed or entry.metadata.nucleation_time_ps is None:
            raise RuntimeError(
                f"Transition-parent source did not nucleate: {source_run_id}."
            )
        _source_npz(entry)
    return tuple(sorted(selected, key=lambda item: item[0]))


def _source_splits(
    sources: tuple[tuple[str, CatalogEntry], ...], campaign_seed: int
) -> dict[str, str]:
    result: dict[str, str] = {}
    for temperature_index, temperature_K in enumerate(TEMPERATURES_K):
        run_ids = [
            source_run_id
            for source_run_id, entry in sources
            if entry.metadata.temperature_K == temperature_K
        ]
        ranked = sorted(
            run_ids,
            key=lambda run_id: hashlib.sha256(
                f"{campaign_seed}|{temperature_K:g}|{run_id}".encode("utf-8")
            ).hexdigest(),
        )
        train_count = int(round(0.70 * len(ranked)))
        remaining = len(ranked) - train_count
        model_count = remaining // 2 + (
            1 if remaining % 2 == 1 and temperature_index % 2 == 1 else 0
        )
        final_count = remaining - model_count
        if min(train_count, model_count, final_count) < 1:
            raise RuntimeError(
                f"At least three sources are required at {temperature_K:g} K to create "
                f"source-grouped splits; got {len(ranked)}."
            )
        for index, run_id in enumerate(ranked):
            if index < train_count:
                split = "optimization"
            elif index < train_count + model_count:
                split = "model_selection"
            else:
                split = "final_validation"
            result[run_id] = split
    return result


def _calibrate_basin_a(
    config: NestedShootingConfig,
    sources: tuple[tuple[str, CatalogEntry], ...],
) -> dict[float, dict[str, Any]]:
    calibration: dict[float, dict[str, Any]] = {}
    for protocol in config.temperatures:
        samples: list[int] = []
        source_sample_counts: dict[str, int] = {}
        for source_run_id, entry in sources:
            if entry.metadata.temperature_K != protocol.temperature_K:
                continue
            times = np.asarray(entry.metadata.progress_times_ps, dtype=np.float64)
            clusters = np.asarray(
                entry.metadata.largest_crystalline_cluster_atoms, dtype=np.int64
            )
            selected = clusters[times <= config.calibration_duration_ps + 1.0e-12]
            if selected.size == 0:
                raise RuntimeError(
                    f"No liquid-calibration frames at {protocol.temperature_K:g} K in "
                    f"{source_run_id}."
                )
            samples.extend(int(value) for value in selected)
            source_sample_counts[source_run_id] = int(selected.size)
        threshold = int(math.ceil(float(np.quantile(samples, config.calibration_quantile))))
        if threshold != protocol.expected_basin_a_max_cluster_atoms:
            raise RuntimeError(
                f"Temperature-calibrated basin-A threshold changed at "
                f"{protocol.temperature_K:g} K: observed={threshold}, "
                f"expected={protocol.expected_basin_a_max_cluster_atoms}."
            )
        if threshold >= config.crystal_basin_min_cluster_atoms:
            raise RuntimeError(
                f"Basin A and B overlap at {protocol.temperature_K:g} K: "
                f"A<={threshold}, B>={config.crystal_basin_min_cluster_atoms}."
            )
        calibration[protocol.temperature_K] = {
            "temperature_K": protocol.temperature_K,
            "sample_definition": (
                "repository source progress frames from 0 ps through "
                f"{config.calibration_duration_ps:g} ps inclusive"
            ),
            "sample_count": len(samples),
            "source_sample_counts": source_sample_counts,
            "quantile": config.calibration_quantile,
            "quantile_value_atoms": float(
                np.quantile(samples, config.calibration_quantile)
            ),
            "basin_a_max_cluster_atoms": threshold,
            "minimum_atoms": int(np.min(samples)),
            "maximum_atoms": int(np.max(samples)),
        }
    return calibration


def _candidate_pool(
    config: NestedShootingConfig,
    sources: tuple[tuple[str, CatalogEntry], ...],
    splits: dict[str, str],
    calibration: dict[float, dict[str, Any]],
) -> dict[float, list[CandidateFrame]]:
    pools: dict[float, list[CandidateFrame]] = {temperature: [] for temperature in TEMPERATURES_K}
    for source_run_id, entry in sources:
        metadata = entry.metadata
        temperature_K = metadata.temperature_K
        basin_a = int(calibration[temperature_K]["basin_a_max_cluster_atoms"])
        steps = np.asarray(metadata.progress_steps, dtype=np.int64)
        times = np.asarray(metadata.progress_times_ps, dtype=np.float64)
        fractions = np.asarray(metadata.crystalline_fraction, dtype=np.float64)
        clusters = np.asarray(metadata.largest_crystalline_cluster_atoms, dtype=np.int64)
        assert metadata.nucleation_time_ps is not None
        for frame_index in range(len(steps)):
            if times[frame_index] >= metadata.nucleation_time_ps:
                continue
            cluster = int(clusters[frame_index])
            if basin_a < cluster < config.crystal_basin_min_cluster_atoms:
                pools[temperature_K].append(
                    CandidateFrame(
                        entry=entry,
                        source_run_id=source_run_id,
                        source_split=splits[source_run_id],
                        frame_index=frame_index,
                        source_step=int(steps[frame_index]),
                        source_time_ps=float(times[frame_index]),
                        crystalline_fraction=float(fractions[frame_index]),
                        largest_cluster_atoms=cluster,
                        basin_role="transition_candidate",
                    )
                )
    return pools


def _stable_key(candidate: CandidateFrame, campaign_seed: int) -> str:
    return hashlib.sha256(
        (
            f"{campaign_seed}|{candidate.source_run_id}|{candidate.source_step}|"
            f"{candidate.basin_role}"
        ).encode("utf-8")
    ).hexdigest()


def _select_transition_candidates(
    candidates: list[CandidateFrame],
    *,
    count: int,
    maximum_per_source: int,
    basin_a_max: int,
    basin_b_min: int,
    campaign_seed: int,
) -> list[CandidateFrame]:
    if not candidates:
        raise RuntimeError("Structural candidate pool is empty.")
    targets = np.linspace(basin_a_max + 1, basin_b_min - 1, count)
    remaining = list(candidates)
    selected: list[CandidateFrame] = []
    source_counts: dict[str, int] = {}
    for target in targets:
        eligible = [
            candidate
            for candidate in remaining
            if source_counts.get(candidate.source_run_id, 0) < maximum_per_source
        ]
        if not eligible:
            raise RuntimeError(
                f"Cannot select {count} transition parents with at most "
                f"{maximum_per_source} per source; selected={len(selected)}."
            )
        chosen = min(
            eligible,
            key=lambda candidate: (
                source_counts.get(candidate.source_run_id, 0),
                abs(candidate.largest_cluster_atoms - float(target)),
                _stable_key(candidate, campaign_seed),
            ),
        )
        selected.append(chosen)
        source_counts[chosen.source_run_id] = source_counts.get(chosen.source_run_id, 0) + 1
        remaining.remove(chosen)
    return selected


def _select_basin_controls(
    *,
    temperature_K: float,
    sources: tuple[tuple[str, CatalogEntry], ...],
    splits: dict[str, str],
    basin_a_max: int,
    basin_b_min: int,
    count_each: int,
    campaign_seed: int,
    source_counts: dict[str, int],
    maximum_per_source: int,
) -> list[CandidateFrame]:
    liquid: list[CandidateFrame] = []
    crystal: list[CandidateFrame] = []
    for source_run_id, entry in sources:
        if entry.metadata.temperature_K != temperature_K:
            continue
        metadata = entry.metadata
        steps = np.asarray(metadata.progress_steps, dtype=np.int64)
        times = np.asarray(metadata.progress_times_ps, dtype=np.float64)
        fractions = np.asarray(metadata.crystalline_fraction, dtype=np.float64)
        clusters = np.asarray(metadata.largest_crystalline_cluster_atoms, dtype=np.int64)
        liquid_indices = np.flatnonzero(clusters <= basin_a_max)
        crystal_indices = np.flatnonzero(clusters >= basin_b_min)
        if liquid_indices.size:
            frame_index = int(liquid_indices[0])
            liquid.append(
                CandidateFrame(
                    entry=entry,
                    source_run_id=source_run_id,
                    source_split=splits[source_run_id],
                    frame_index=frame_index,
                    source_step=int(steps[frame_index]),
                    source_time_ps=float(times[frame_index]),
                    crystalline_fraction=float(fractions[frame_index]),
                    largest_cluster_atoms=int(clusters[frame_index]),
                    basin_role="liquid_control",
                )
            )
        if crystal_indices.size:
            frame_index = int(crystal_indices[-1])
            crystal.append(
                CandidateFrame(
                    entry=entry,
                    source_run_id=source_run_id,
                    source_split=splits[source_run_id],
                    frame_index=frame_index,
                    source_step=int(steps[frame_index]),
                    source_time_ps=float(times[frame_index]),
                    crystalline_fraction=float(fractions[frame_index]),
                    largest_cluster_atoms=int(clusters[frame_index]),
                    basin_role="crystal_control",
                )
            )
    if len(liquid) < count_each or len(crystal) < count_each:
        raise RuntimeError(
            f"Insufficient basin controls at {temperature_K:g} K: "
            f"liquid={len(liquid)}, crystal={len(crystal)}, requested={count_each}."
        )
    selected: list[CandidateFrame] = []
    for role_candidates in (liquid, crystal):
        for _ in range(count_each):
            eligible = [
                candidate
                for candidate in role_candidates
                if candidate not in selected
                and source_counts.get(candidate.source_run_id, 0) < maximum_per_source
            ]
            if not eligible:
                raise RuntimeError(
                    f"Cannot select {count_each} controls of role "
                    f"{role_candidates[0].basin_role!r} at {temperature_K:g} K "
                    f"without exceeding {maximum_per_source} parents per source."
                )
            chosen = min(
                eligible,
                key=lambda candidate: (
                    source_counts.get(candidate.source_run_id, 0),
                    _stable_key(candidate, campaign_seed),
                ),
            )
            selected.append(chosen)
            source_counts[chosen.source_run_id] = source_counts.get(chosen.source_run_id, 0) + 1
    return selected


def render_nested_lammps_input(
    *,
    parent_id: str,
    branch_id: str,
    temperature_K: float,
    momentum_seed: int,
    thermostat_seed: int,
    timestep_fs: float,
    thermostat_time_fs: float,
    monitor_interval_steps: int,
    maximum_steps: int,
) -> str:
    monitor_iterations = maximum_steps // monitor_interval_steps
    monitor_runner = REPOSITORY_ROOT / "scripts/run_lammps_meam_nested_shooting_campaign.py"
    pointnet_python = Path("/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python")
    return f"""# Transition-balanced nested shooting branch generated by PointCloudMaterials.
# Identical momentum_seed values deliberately identify fixed-(X,v) thermostat futures.
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
velocity all create {temperature_K:.12g} {momentum_seed} mom yes rot no dist gaussian loop all
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve

thermo {monitor_interval_steps}
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes

variable output_file file output_steps.txt
variable next_output equal next(output_file)
dump trajectory all custom {monitor_interval_steps} trajectory.lammpstrj id type x y z vx vy vz
dump_modify trajectory every v_next_output first yes sort id format line "%d %d %.9g %.9g %.9g %.9g %.9g %.9g"

variable temperature_file file temperature_steps.txt
variable next_temperature equal next(temperature_file)
fix sampled_temperature all print v_next_temperature "$(step) $(temp)" file sampled_temperature.tsv screen no title "step temperature_K"

print "NESTED_SHOOTING_BEGIN {branch_id} PARENT {parent_id}"
run 0
print "$(step) $(temp)" file initial_temperature.txt screen no
write_dump all custom initial_state.lammpstrj id type x y z vx vy vz fx fy fz modify sort id format line "%d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g"
fix thermostat all langevin {temperature_K:.12g} {temperature_K:.12g} {thermostat_time_fs / 1000.0:.12g} {thermostat_seed} zero yes

variable monitor_iteration loop {monitor_iterations}
label monitor_loop
run {monitor_interval_steps}
write_dump all custom monitor_frame.lammpstrj id type x y z modify sort id format line "%d %d %.9g %.9g %.9g"
shell /bin/rm -f monitor_decision.txt
shell {pointnet_python} {monitor_runner} monitor-frame --branch-dir .
variable stop_decision file monitor_decision.txt
if "${{stop_decision}} == 1" then "jump SELF basin_reached"
variable stop_decision delete
next monitor_iteration
jump SELF monitor_loop

label maximum_duration
print "NESTED_SHOOTING_CENSORED {branch_id}"
jump SELF finalize

label basin_reached
variable stop_decision delete
variable monitor_iteration delete
print "NESTED_SHOOTING_BASIN_REACHED {branch_id}"
# First passage is now fixed, but the Markov trajectory remains scientifically
# valid after basin arrival. Continue it without further classification so the
# saved path always contains the configured fixed-horizon point clouds.
variable continuation_steps equal {maximum_steps}-step
if "${{continuation_steps}} > 0" then "run ${{continuation_steps}}"
variable continuation_steps delete

label finalize
# At maximum duration the file variables have consumed their sentinels.  Detach
# the consumers before write_restart initializes the system again; otherwise
# fix print tries to evaluate next(temperature_file) after LAMMPS has deleted
# the exhausted file variable.
unfix sampled_temperature
undump trajectory
write_restart final.restart.bin
print "NESTED_SHOOTING_COMPLETE {branch_id} PARENT {parent_id}"
"""


def _write_schedule(
    path: Path,
    output_steps: tuple[int, ...],
    sentinel: int,
    monitor_interval_steps: int,
) -> None:
    # LAMMPS evaluates a variable dump/fix cadence once at the beginning of
    # every ``run`` command.  ``next(file_variable)`` consumes that value even
    # when the run begins before the requested output step.  Duplicate the
    # first scheduled step after every monitor-block boundary so that this
    # control evaluation and the actual output evaluation see the same step.
    first_after_boundary = {
        min(step for step in output_steps if step > boundary)
        for boundary in range(0, output_steps[-1], monitor_interval_steps)
    }
    values: list[int] = []
    for step in output_steps[1:]:
        if step in first_after_boundary:
            values.append(step)
        values.append(step)
    values.append(sentinel)
    path.write_text("".join(f"{step}\n" for step in values), encoding="utf-8")


def _write_slurm_scripts(config: NestedShootingConfig) -> None:
    root = config.output_root
    runner = REPOSITORY_ROOT / "scripts/run_lammps_meam_nested_shooting_campaign.py"
    common = f"""set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
"""
    task_path = root / "slurm" / "run_branch.sbatch"
    task_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_shoot
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={config.mpi_ranks}
#SBATCH --ntasks-per-node={config.mpi_ranks}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem={config.memory}
#SBATCH --time={config.time_limit}
#SBATCH --output={root}/slurm/%A_%a.out
#SBATCH --error={root}/slurm/%A_%a.err

{common}python {runner} run-task --campaign-root {root} --task-index "${{SLURM_ARRAY_TASK_ID}}"
""",
        encoding="utf-8",
    )
    task_path.chmod(0o750)
    controller_path = root / "slurm" / "submit_wave.sbatch"
    controller_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_ctl
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output={root}/slurm/controller_%j.out
#SBATCH --error={root}/slurm/controller_%j.err

{common}: "${{NESTED_START:?NESTED_START must identify the first branch}}"
python {runner} submit-next-wave --campaign-root {root} --start-index "${{NESTED_START}}"
""",
        encoding="utf-8",
    )
    controller_path.chmod(0o750)
    summary_path = root / "slurm" / "summarize.sbatch"
    summary_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=al_nested_summary
#SBATCH --partition={config.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output={root}/slurm/summary_%j.out
#SBATCH --error={root}/slurm/summary_%j.err

{common}python {runner} summarize --campaign-root {root}
""",
        encoding="utf-8",
    )
    summary_path.chmod(0o750)


def prepare_nested_campaign(config: NestedShootingConfig) -> dict[str, Any]:
    if config.output_root.exists():
        raise FileExistsError(
            f"Nested shooting output exists and will not be overwritten: {config.output_root}"
        )
    shortest_duration = min(
        protocol.maximum_duration_ps for protocol in config.temperatures
    )
    if shortest_duration < FIXED_HORIZONS_PS[-1]:
        raise RuntimeError(
            "Nested maximum durations must retain every fixed-horizon target: "
            f"shortest_duration_ps={shortest_duration}, "
            f"fixed_horizons_ps={FIXED_HORIZONS_PS}."
        )
    for potential_path, expected_sha256 in (
        (config.library_potential, config.library_sha256),
        (config.parameter_potential, config.parameter_sha256),
    ):
        if not potential_path.is_file():
            raise FileNotFoundError(f"Required MEAM potential is missing: {potential_path}")
        observed = _sha256_file(potential_path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"Potential checksum mismatch for {potential_path}: "
                f"expected={expected_sha256}, observed={observed}."
            )

    sources = _discover_sources(config)
    splits = _source_splits(sources, config.campaign_seed)
    calibration = _calibrate_basin_a(config, sources)
    candidate_pools = _candidate_pool(config, sources, splits, calibration)

    selected: list[CandidateFrame] = []
    selection_counts_by_source: dict[str, int] = {}
    for protocol in config.temperatures:
        temperature_K = protocol.temperature_K
        transition = _select_transition_candidates(
            candidate_pools[temperature_K],
            count=config.transition_parents_per_temperature,
            maximum_per_source=config.maximum_parents_per_source_run,
            basin_a_max=protocol.expected_basin_a_max_cluster_atoms,
            basin_b_min=config.crystal_basin_min_cluster_atoms,
            campaign_seed=config.campaign_seed,
        )
        for candidate in transition:
            selection_counts_by_source[candidate.source_run_id] = (
                selection_counts_by_source.get(candidate.source_run_id, 0) + 1
            )
        controls = _select_basin_controls(
            temperature_K=temperature_K,
            sources=sources,
            splits=splits,
            basin_a_max=protocol.expected_basin_a_max_cluster_atoms,
            basin_b_min=config.crystal_basin_min_cluster_atoms,
            count_each=config.basin_control_parents_per_temperature,
            campaign_seed=config.campaign_seed,
            source_counts=selection_counts_by_source,
            maximum_per_source=config.maximum_parents_per_source_run,
        )
        selected.extend(transition)
        selected.extend(controls)

    if max(selection_counts_by_source.values()) > config.maximum_parents_per_source_run:
        raise RuntimeError(
            f"Parent selection exceeded the source-run cap: {selection_counts_by_source}."
        )

    root = config.output_root
    root.mkdir(parents=True)
    for name in ("parents", "branches", "potential", "slurm"):
        (root / name).mkdir()
    shutil.copy2(config.library_potential, root / "potential" / config.library_potential.name)
    shutil.copy2(config.parameter_potential, root / "potential" / config.parameter_potential.name)
    shutil.copy2(config.config_path, root / "campaign_config.yaml")

    selected.sort(
        key=lambda candidate: (
            candidate.entry.metadata.temperature_K,
            {"transition_candidate": 0, "liquid_control": 1, "crystal_control": 2}[
                candidate.basin_role
            ],
            candidate.largest_cluster_atoms,
            candidate.source_run_id,
        )
    )
    frames_by_archive: dict[Path, list[tuple[int, CandidateFrame]]] = {}
    for parent_index, candidate in enumerate(selected):
        frames_by_archive.setdefault(_source_npz(candidate.entry), []).append(
            (parent_index, candidate)
        )

    parents_by_index: dict[int, dict[str, Any]] = {}
    for archive_path, archive_candidates in frames_by_archive.items():
        archive_sha256 = _sha256_file(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {"step", "positions_A", "cell_vectors_A"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"{archive_path}: missing source arrays {missing}.")
            source_steps = np.asarray(archive["step"], dtype=np.int64)
            for parent_index, candidate in archive_candidates:
                if int(source_steps[candidate.frame_index]) != candidate.source_step:
                    raise RuntimeError(
                        f"{archive_path}: candidate progress step changed at frame "
                        f"{candidate.frame_index}."
                    )
                positions_A = np.asarray(
                    archive["positions_A"][candidate.frame_index], dtype=np.float64
                )
                cell_A = np.asarray(
                    archive["cell_vectors_A"][candidate.frame_index], dtype=np.float64
                )
                if positions_A.shape != (EXPECTED_ATOM_COUNT, 3) or cell_A.shape != (3, 3):
                    raise RuntimeError(
                        f"Invalid selected parent shapes in {archive_path}: "
                        f"positions={positions_A.shape}, cell={cell_A.shape}."
                    )
                off_diagonal = cell_A.copy()
                off_diagonal[np.diag_indices(3)] = 0.0
                if np.any(np.abs(off_diagonal) > 1.0e-10):
                    raise RuntimeError(
                        f"Nested LAMMPS shooting requires orthogonal source cells: {cell_A.tolist()}."
                    )
                temperature_K = candidate.entry.metadata.temperature_K
                parent_id = (
                    f"parent_{parent_index:03d}_T{temperature_K:g}_"
                    f"cluster{candidate.largest_cluster_atoms:05d}_{candidate.basin_role}"
                )
                parent_dir = root / "parents" / parent_id
                parent_dir.mkdir()
                data_path = parent_dir / "parent.lammps.data"
                atoms = Atoms(
                    "Al" * EXPECTED_ATOM_COUNT,
                    positions=positions_A,
                    cell=cell_A,
                    pbc=True,
                )
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
                    "basin_role": candidate.basin_role,
                    "temperature_K": temperature_K,
                    "source_run_id": candidate.source_run_id,
                    "source_split": candidate.source_split,
                    "source_velocity_seed": candidate.entry.metadata.velocity_seed,
                    "source_frame_index": candidate.frame_index,
                    "source_frame_step": candidate.source_step,
                    "source_frame_time_ps": candidate.source_time_ps,
                    "source_crystalline_fraction": candidate.crystalline_fraction,
                    "source_largest_crystalline_cluster_atoms": candidate.largest_cluster_atoms,
                    "source_coordinate_archive": str(archive_path),
                    "source_coordinate_archive_sha256": archive_sha256,
                    "data_file": str(data_path.relative_to(root)),
                    "data_sha256": _sha256_file(data_path),
                    "basin_a_max_cluster_atoms": int(
                        calibration[temperature_K]["basin_a_max_cluster_atoms"]
                    ),
                    "basin_b_min_cluster_atoms": config.crystal_basin_min_cluster_atoms,
                }
                _write_json_atomic(parent_dir / "metadata.json", parent)
                parents_by_index[parent_index] = parent

    parents = [parents_by_index[index] for index in range(len(selected))]
    maximum_duration_by_temperature = {
        protocol.temperature_K: protocol.maximum_duration_ps
        for protocol in config.temperatures
    }
    branches: list[dict[str, Any]] = []
    seen_momentum_seeds: set[int] = set()
    seen_thermostat_seeds: set[int] = set()
    for parent in parents:
        parent_index = int(parent["parent_index"])
        temperature_K = float(parent["temperature_K"])
        maximum_duration_ps = maximum_duration_by_temperature[temperature_K]
        maximum_steps = int(round(maximum_duration_ps * 1000.0 / config.timestep_fs))
        output_steps = multirate_output_steps(
            timestep_fs=config.timestep_fs,
            maximum_duration_ps=maximum_duration_ps,
        )
        sentinel = maximum_steps + config.monitor_interval_steps
        for momentum_index in range(config.momentum_samples_per_parent):
            expected_momentum_seed: int | None = None
            for thermostat_index in range(config.thermostat_futures_per_momentum):
                momentum_seed, thermostat_seed = nested_random_seeds(
                    config.campaign_seed,
                    parent_index,
                    momentum_index,
                    thermostat_index,
                )
                if expected_momentum_seed is None:
                    expected_momentum_seed = momentum_seed
                    if momentum_seed in seen_momentum_seeds:
                        raise RuntimeError(f"Nested momentum-seed collision: {momentum_seed}.")
                    seen_momentum_seeds.add(momentum_seed)
                elif momentum_seed != expected_momentum_seed:
                    raise RuntimeError(
                        f"Momentum seed changed across thermostat futures for parent={parent_index}, "
                        f"momentum_index={momentum_index}."
                    )
                if thermostat_seed in seen_thermostat_seeds:
                    raise RuntimeError(f"Nested thermostat-seed collision: {thermostat_seed}.")
                seen_thermostat_seeds.add(thermostat_seed)
                branch_index = len(branches)
                branch_id = (
                    f"branch_{branch_index:04d}_{parent['parent_id']}_"
                    f"momentum_{momentum_index:02d}_noise_{thermostat_index:02d}"
                )
                branch_dir = root / "branches" / branch_id
                branch_dir.mkdir()
                branch = {
                    "branch_index": branch_index,
                    "branch_id": branch_id,
                    "branch_dir": str(branch_dir.relative_to(root)),
                    "parent_index": parent_index,
                    "parent_id": parent["parent_id"],
                    "basin_role": parent["basin_role"],
                    "source_run_id": parent["source_run_id"],
                    "source_split": parent["source_split"],
                    "temperature_K": temperature_K,
                    "momentum_index": momentum_index,
                    "thermostat_index": thermostat_index,
                    "momentum_seed": momentum_seed,
                    "thermostat_seed": thermostat_seed,
                    "maximum_duration_ps": maximum_duration_ps,
                    "maximum_steps": maximum_steps,
                    "output_steps": list(output_steps),
                    "basin_a_max_cluster_atoms": parent[
                        "basin_a_max_cluster_atoms"
                    ],
                    "basin_b_min_cluster_atoms": config.crystal_basin_min_cluster_atoms,
                    "basin_persistence_frames": config.basin_persistence_frames,
                    "monitor_interval_steps": config.monitor_interval_steps,
                    "timestep_fs": config.timestep_fs,
                    "ptm_rmsd_cutoff": config.ptm_rmsd_cutoff,
                    "crystalline_cluster_cutoff_A": config.crystalline_cluster_cutoff_A,
                }
                _write_json_atomic(branch_dir / "metadata.json", branch)
                (branch_dir / "in.lammps").write_text(
                    render_nested_lammps_input(
                        parent_id=str(parent["parent_id"]),
                        branch_id=branch_id,
                        temperature_K=temperature_K,
                        momentum_seed=momentum_seed,
                        thermostat_seed=thermostat_seed,
                        timestep_fs=config.timestep_fs,
                        thermostat_time_fs=config.thermostat_time_fs,
                        monitor_interval_steps=config.monitor_interval_steps,
                        maximum_steps=maximum_steps,
                    ),
                    encoding="utf-8",
                )
                _write_schedule(
                    branch_dir / "output_steps.txt",
                    output_steps,
                    sentinel,
                    config.monitor_interval_steps,
                )
                _write_schedule(
                    branch_dir / "temperature_steps.txt",
                    output_steps,
                    sentinel,
                    config.monitor_interval_steps,
                )
                branches.append(branch)

    source_documents = [
        {
            "source_run_id": source_run_id,
            "source_split": splits[source_run_id],
            "trajectory_path": str(entry.trajectory_path),
            "coordinate_archive": str(_source_npz(entry)),
            "temperature_K": entry.metadata.temperature_K,
            "velocity_seed": entry.metadata.velocity_seed,
            "nucleation_time_ps": entry.metadata.nucleation_time_ps,
        }
        for source_run_id, entry in sources
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_type": "transition_balanced_nested_langevin_nvt_shooting_pilot",
        "scientific_contract": {
            "parent_coordinate": "connected PTM largest crystalline cluster atoms",
            "source_split_unit": "independent source run and every descendant",
            "nested_randomness": (
                "momentum_seed is shared by thermostat children; thermostat_seed is unique"
            ),
            "first_passage": (
                "single uninterrupted LAMMPS process checked every monitor interval; "
                "persistent A or B arrival fixes the label, then the same process continues "
                "to maximum duration so fixed-horizon frames remain available; a branch "
                "with no arrival by maximum duration is censored"
            ),
            "temporary_text_policy": (
                "LAMMPS text is a branch-local staging artifact and is deleted only after "
                f"the {NESTED_STORAGE_DTYPE} binary and observables pass validation"
            ),
        },
        "atom_count": EXPECTED_ATOM_COUNT,
        "calibration": {
            f"{temperature:g}": calibration[temperature]
            for temperature in TEMPERATURES_K
        },
        "basins": {
            "A": "temperature-calibrated largest-cluster upper threshold",
            "B_min_cluster_atoms": config.crystal_basin_min_cluster_atoms,
            "persistence_frames": config.basin_persistence_frames,
            "monitor_interval_ps": config.monitor_interval_ps,
            "ptm_rmsd_cutoff": config.ptm_rmsd_cutoff,
            "cluster_connectivity_cutoff_A": config.crystalline_cluster_cutoff_A,
        },
        "output_cadence": {
            "0_to_0.3_ps": "0.03 ps",
            "0.3_to_3_ps": (
                "0.1 ps targets rounded to the nearest 3 fs integration step"
            ),
            "after_3_ps": "0.3 ps",
            "exact_confirmed_basin_crossing_frame": True,
            "fixed_horizons_ps": list(FIXED_HORIZONS_PS),
            "storage": (
                "pointcloudmaterials.shooting_trajectory float16 memory-mapped vectors "
                "decoded to float32 by consumers"
            ),
        },
        "execution": {
            "partition": config.partition,
            "mpi_ranks_per_branch": config.mpi_ranks,
            "memory": config.memory,
            "time_limit": config.time_limit,
            "array_concurrency": config.array_concurrency,
        },
        "potential": {
            "library_sha256": config.library_sha256,
            "parameter_sha256": config.parameter_sha256,
        },
        "counts": {
            "sources": len(sources),
            "parents": len(parents),
            "transition_parents": sum(
                parent["basin_role"] == "transition_candidate" for parent in parents
            ),
            "control_parents": sum(
                parent["basin_role"] != "transition_candidate" for parent in parents
            ),
            "branches": len(branches),
            "branches_by_temperature": {
                f"{temperature:g}": sum(
                    branch["temperature_K"] == temperature for branch in branches
                )
                for temperature in TEMPERATURES_K
            },
            "branches_by_split": {
                split: sum(branch["source_split"] == split for branch in branches)
                for split in SPLIT_NAMES
            },
        },
        "sources": source_documents,
        "parents": parents,
        "branches": branches,
    }
    _write_json_atomic(root / "manifest.json", manifest)
    _write_json_atomic(
        root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "branch_count": len(branches),
        },
    )
    _write_slurm_scripts(config)
    return manifest


def evaluate_monitor_frame(branch_dir: str | Path) -> dict[str, Any]:
    root = Path(branch_dir).expanduser().resolve()
    metadata = _load_json(root / "metadata.json")
    frame_path = root / "monitor_frame.lammpstrj"
    if not frame_path.is_file() or frame_path.stat().st_size == 0:
        raise FileNotFoundError(f"Online PTM monitor frame is missing or empty: {frame_path}")
    try:
        from ovito.io import import_file
        from ovito.modifiers import (
            ClusterAnalysisModifier,
            PolyhedralTemplateMatchingModifier,
        )
    except ImportError as exc:
        raise ImportError(
            "Nested shooting online stopping requires OVITO in the pointnet environment."
        ) from exc

    pipeline = import_file(str(frame_path), sort_particles=True)
    data = pipeline.compute()
    observed_atom_count = int(data.particles.count)
    if observed_atom_count != EXPECTED_ATOM_COUNT:
        raise RuntimeError(
            f"Online PTM frame atom count changed: expected={EXPECTED_ATOM_COUNT}, "
            f"observed={observed_atom_count}, path={frame_path}."
        )
    observed_step = int(data.attributes.get("Timestep", -1))
    if observed_step < 0:
        raise RuntimeError(f"OVITO did not expose the LAMMPS timestep for {frame_path}.")
    monitor_interval_steps = int(metadata["monitor_interval_steps"])
    if observed_step <= 0 or observed_step % monitor_interval_steps != 0:
        raise RuntimeError(
            f"Online PTM frame step is not a positive monitor checkpoint: "
            f"step={observed_step}, interval={monitor_interval_steps}."
        )

    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = float(metadata["ptm_rmsd_cutoff"])
    data.apply(ptm)
    structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
    crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
    data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
    clusters = ClusterAnalysisModifier(
        cutoff=float(metadata["crystalline_cluster_cutoff_A"]),
        only_selected=True,
        sort_by_size=True,
    )
    data.apply(clusters)
    crystalline_fraction = float(np.mean(crystalline))
    largest_cluster_atoms = int(data.attributes["ClusterAnalysis.largest_size"])
    basin_a = largest_cluster_atoms <= int(metadata["basin_a_max_cluster_atoms"])
    basin_b = largest_cluster_atoms >= int(metadata["basin_b_min_cluster_atoms"])
    if basin_a and basin_b:
        raise RuntimeError(
            f"Basin definitions overlap for branch {metadata['branch_id']}: "
            f"cluster={largest_cluster_atoms}."
        )

    state_path = root / "online_monitor.json"
    if state_path.is_file():
        state = _load_json(state_path)
        if state.get("branch_id") != metadata["branch_id"]:
            raise RuntimeError(
                f"Online monitor state belongs to another branch: {state_path}."
            )
        if state.get("first_passage_outcome") is not None:
            raise RuntimeError(
                f"Online PTM monitor was invoked after a terminal basin event: {state_path}."
            )
        observations = state.get("observations")
        if not isinstance(observations, list) or not observations:
            raise RuntimeError(f"Online monitor state has invalid observations: {state_path}.")
        previous_step = int(observations[-1]["timestep"])
        if observed_step - previous_step != monitor_interval_steps:
            raise RuntimeError(
                f"Online PTM checkpoints are not contiguous: previous={previous_step}, "
                f"current={observed_step}, interval={monitor_interval_steps}."
            )
    else:
        if observed_step != monitor_interval_steps:
            raise RuntimeError(
                f"First online PTM checkpoint must be step {monitor_interval_steps}, "
                f"got {observed_step}."
            )
        state = {
            "schema_version": SCHEMA_VERSION,
            "branch_id": metadata["branch_id"],
            "monitor_interval_steps": monitor_interval_steps,
            "timestep_fs": float(metadata["timestep_fs"]),
            "basin_a_max_cluster_atoms": int(
                metadata["basin_a_max_cluster_atoms"]
            ),
            "basin_b_min_cluster_atoms": int(
                metadata["basin_b_min_cluster_atoms"]
            ),
            "persistence_frames": int(metadata["basin_persistence_frames"]),
            "consecutive_basin_a": 0,
            "consecutive_basin_b": 0,
            "first_passage_outcome": None,
            "first_passage_onset_timestep": None,
            "first_passage_confirmation_timestep": None,
            "observations": [],
        }
        observations = state["observations"]

    observation = {
        "timestep": observed_step,
        "time_ps": observed_step * float(metadata["timestep_fs"]) / 1000.0,
        "crystalline_fraction": crystalline_fraction,
        "largest_crystalline_cluster_atoms": largest_cluster_atoms,
        "basin_a": basin_a,
        "basin_b": basin_b,
    }
    observations.append(observation)
    state["consecutive_basin_a"] = (
        int(state["consecutive_basin_a"]) + 1 if basin_a else 0
    )
    state["consecutive_basin_b"] = (
        int(state["consecutive_basin_b"]) + 1 if basin_b else 0
    )
    persistence = int(metadata["basin_persistence_frames"])
    outcome: str | None = None
    if int(state["consecutive_basin_a"]) >= persistence:
        outcome = "basin_A_liquid"
    elif int(state["consecutive_basin_b"]) >= persistence:
        outcome = "basin_B_crystal"
    if outcome is not None:
        onset = int(observations[-persistence]["timestep"])
        state["first_passage_outcome"] = outcome
        state["first_passage_onset_timestep"] = onset
        state["first_passage_confirmation_timestep"] = observed_step
        state["first_passage_time_ps"] = (
            onset * float(metadata["timestep_fs"]) / 1000.0
        )
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(state_path, state)
    decision_path = root / "monitor_decision.txt"
    temporary_decision = root / f".monitor_decision.tmp-{os.getpid()}"
    temporary_decision.write_text("1\n" if outcome is not None else "0\n", encoding="ascii")
    temporary_decision.replace(decision_path)
    return state


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


def _lammps_command(mpi_ranks: int) -> list[str]:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Nested shooting production branches require a Slurm allocation.")
    lmp = Path(sys.prefix) / "bin" / "lmp"
    if not lmp.is_file():
        raise FileNotFoundError(f"pointnet LAMMPS executable is missing: {lmp}")
    srun = shutil.which("srun")
    if srun is None:
        raise FileNotFoundError("Nested shooting requires srun on PATH.")
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


def _load_initial_forces(path: Path, atom_count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    expected_line_count = 9 + atom_count
    if len(lines) != expected_line_count:
        raise RuntimeError(
            f"Initial-state dump line count changed: expected={expected_line_count}, "
            f"observed={len(lines)}, path={path}."
        )
    if lines[0] != "ITEM: TIMESTEP" or int(lines[1]) != 0:
        raise RuntimeError(f"Initial-state dump is not timestep zero: {path}.")
    if lines[2] != "ITEM: NUMBER OF ATOMS" or int(lines[3]) != atom_count:
        raise RuntimeError(f"Initial-state atom count changed: {path}.")
    if lines[4] != "ITEM: BOX BOUNDS pp pp pp":
        raise RuntimeError(f"Initial-state box is not orthogonal periodic: {path}.")
    expected_header = "ITEM: ATOMS id type x y z vx vy vz fx fy fz"
    if lines[8] != expected_header:
        raise RuntimeError(
            f"Initial-state columns changed: expected={expected_header!r}, "
            f"observed={lines[8]!r}."
        )
    table = np.fromstring("\n".join(lines[9:]), sep=" ", dtype=np.float64)
    if table.size != atom_count * 11:
        raise RuntimeError(
            f"Initial-state atom table is truncated: values={table.size}, path={path}."
        )
    table = table.reshape(atom_count, 11)
    ids = table[:, 0].astype(np.int64)
    order = np.argsort(ids, kind="mergesort")
    if not np.array_equal(ids[order], np.arange(1, atom_count + 1, dtype=np.int64)):
        raise RuntimeError(f"Initial-state IDs are not exactly 1..{atom_count}: {path}.")
    forces = table[:, 8:11].astype(np.float32)[order]
    if not np.all(np.isfinite(forces)):
        raise RuntimeError(f"Initial-state forces contain nonfinite values: {path}.")
    return forces


def _load_sampled_temperatures(
    branch_dir: Path, expected_timesteps: np.ndarray
) -> np.ndarray:
    initial_tokens = (branch_dir / "initial_temperature.txt").read_text(
        encoding="ascii"
    ).split()
    if len(initial_tokens) != 2 or int(initial_tokens[0]) != 0:
        raise RuntimeError(
            f"Invalid initial temperature record: {branch_dir / 'initial_temperature.txt'}."
        )
    records: dict[int, float] = {0: float(initial_tokens[1])}
    temperature_path = branch_dir / "sampled_temperature.tsv"
    lines = temperature_path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].split() != ["step", "temperature_K"]:
        raise RuntimeError(f"Invalid sampled-temperature header: {temperature_path}.")
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != 2:
            raise RuntimeError(f"Invalid sampled-temperature row {line!r} in {temperature_path}.")
        step = int(tokens[0])
        value = float(tokens[1])
        if step in records:
            raise RuntimeError(f"Duplicate sampled temperature at step={step}: {temperature_path}.")
        records[step] = value
    expected = [int(value) for value in expected_timesteps]
    missing = [step for step in expected if step not in records]
    extras = sorted(set(records).difference(expected))
    if missing or extras:
        raise RuntimeError(
            f"Sampled temperatures do not match stored frames: missing={missing}, extras={extras}, "
            f"path={temperature_path}."
        )
    values = np.asarray([records[step] for step in expected], dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError(f"Sampled temperatures are nonfinite or nonpositive: {temperature_path}.")
    return values


def _write_observables(
    path: Path,
    *,
    saved_timesteps: np.ndarray,
    temperatures_K: np.ndarray,
    initial_forces: np.ndarray,
    monitor_state: dict[str, Any],
    censored: bool,
) -> dict[str, Any]:
    observations = monitor_state["observations"]
    monitor_timesteps = np.asarray(
        [value["timestep"] for value in observations], dtype=np.int64
    )
    crystalline_fraction = np.asarray(
        [value["crystalline_fraction"] for value in observations], dtype=np.float32
    )
    largest_cluster_atoms = np.asarray(
        [value["largest_crystalline_cluster_atoms"] for value in observations],
        dtype=np.int64,
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            saved_timesteps=np.asarray(saved_timesteps, dtype=np.int64),
            temperature_K=np.asarray(temperatures_K, dtype=np.float32),
            initial_forces_eV_per_A=np.asarray(initial_forces, dtype=np.float32),
            monitor_timesteps=monitor_timesteps,
            ptm_crystalline_fraction=crystalline_fraction,
            largest_crystalline_cluster_atoms=largest_cluster_atoms,
            censored=np.asarray([censored], dtype=np.bool_),
        )
    temporary.replace(path)
    arrays = {
        "saved_timesteps": np.asarray(saved_timesteps, dtype=np.int64),
        "temperature_K": np.asarray(temperatures_K, dtype=np.float32),
        "initial_forces_eV_per_A": np.asarray(initial_forces, dtype=np.float32),
        "monitor_timesteps": monitor_timesteps,
        "ptm_crystalline_fraction": crystalline_fraction,
        "largest_crystalline_cluster_atoms": largest_cluster_atoms,
        "censored": np.asarray([censored], dtype=np.bool_),
    }
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrays": {
            name: {
                "shape": list(values.shape),
                "dtype": values.dtype.name,
                "sha256": _array_sha256(values),
            }
            for name, values in arrays.items()
        },
    }


def _validate_observables(path: Path, description: dict[str, Any]) -> None:
    if _sha256_file(path) != description["sha256"]:
        raise RuntimeError(f"Nested observables file checksum mismatch: {path}.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(description["arrays"]):
            raise RuntimeError(f"Nested observables arrays changed: {path}.")
        for name, expected in description["arrays"].items():
            values = np.asarray(archive[name])
            if list(values.shape) != expected["shape"] or values.dtype.name != expected["dtype"]:
                raise RuntimeError(
                    f"Nested observable {name!r} shape/dtype changed in {path}."
                )
            if _array_sha256(values) != expected["sha256"]:
                raise RuntimeError(
                    f"Nested observable {name!r} checksum mismatch in {path}."
                )


def run_nested_branch(campaign_root: str | Path, task_index: int) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    if not isinstance(branches, list):
        raise TypeError(f"{root / 'manifest.json'}: branches must be a list.")
    index = int(task_index)
    if index < 0 or index >= len(branches):
        raise IndexError(f"task_index={index} is outside [0, {len(branches)}).")
    branch = branches[index]
    branch_dir = root / str(branch["branch_dir"])
    outcome_path = branch_dir / "outcome.json"
    if outcome_path.is_file():
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Existing nested branch outcome is not complete: {outcome_path}.")
        print(f"Nested branch {branch['branch_id']} is already complete; leaving it unchanged.")
        return outcome
    partial_names = [
        name
        for name in (
            "trajectory.lammpstrj",
            "trajectory_binary_float16",
            "initial_state.lammpstrj",
            "monitor_frame.lammpstrj",
            "online_monitor.json",
            "monitor_decision.txt",
            "sampled_temperature.tsv",
            "initial_temperature.txt",
            "final.restart.bin",
            "observables.npz",
            "lammps.log",
            "lammps.stdout.log",
            "status.json",
        )
        if (branch_dir / name).exists()
    ]
    if partial_names:
        raise RuntimeError(
            f"Nested branch {branch['branch_id']} has partial artifacts {partial_names} but no "
            f"complete outcome. Archive the attempt before resubmitting: {branch_dir}."
        )

    status_path = branch_dir / "status.json"
    _write_json_atomic(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": os.uname().nodename,
        },
    )
    try:
        mpi_ranks = int(manifest["execution"]["mpi_ranks_per_branch"])
        allocated_tasks = int(os.environ.get("SLURM_NTASKS", "0"))
        if allocated_tasks != mpi_ranks:
            raise RuntimeError(
                f"Nested branch received SLURM_NTASKS={allocated_tasks}, expected {mpi_ranks}."
            )
        command = _lammps_command(mpi_ranks)
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
            raise RuntimeError(
                f"Nested LAMMPS branch {branch['branch_id']} failed with return code "
                f"{completed.returncode}; inspect {stdout_path}."
            )

        trajectory_path = branch_dir / "trajectory.lammpstrj"
        restart_path = branch_dir / "final.restart.bin"
        initial_state_path = branch_dir / "initial_state.lammpstrj"
        monitor_state_path = branch_dir / "online_monitor.json"
        for artifact in (
            trajectory_path,
            restart_path,
            initial_state_path,
            monitor_state_path,
            branch_dir / "lammps.log",
        ):
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(
                    f"Nested LAMMPS reported success but artifact is missing or empty: {artifact}."
                )

        scan = TemporalLAMMPSDumpDataset.scan_dump_file(trajectory_path)
        if scan.num_atoms != EXPECTED_ATOM_COUNT or tuple(scan.atom_columns) != (
            "id",
            "type",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
        ):
            raise RuntimeError(
                f"Nested dump contract changed: atoms={scan.num_atoms}, "
                f"columns={scan.atom_columns}, path={trajectory_path}."
            )
        monitor_state = _load_json(monitor_state_path)
        observations = monitor_state.get("observations")
        if not isinstance(observations, list) or not observations:
            raise RuntimeError(f"Nested monitor state has no observations: {monitor_state_path}.")
        last_step = int(scan.timesteps[-1])
        expected_steps = np.asarray(branch["output_steps"], dtype=np.int64)
        if not np.array_equal(scan.timesteps, expected_steps):
            raise RuntimeError(
                f"Nested dump does not match the multirate schedule: expected={expected_steps.tolist()}, "
                f"observed={scan.timesteps.tolist()}, path={trajectory_path}."
            )
        first_passage_outcome = monitor_state.get("first_passage_outcome")
        censored = first_passage_outcome is None
        if censored:
            if int(observations[-1]["timestep"]) != int(branch["maximum_steps"]):
                raise RuntimeError(
                    f"Censored nested monitor stopped before maximum duration: "
                    f"monitor={observations[-1]['timestep']}, "
                    f"maximum={branch['maximum_steps']}."
                )
        elif int(observations[-1]["timestep"]) != int(
            monitor_state["first_passage_confirmation_timestep"]
        ):
            raise RuntimeError(
                "Nested monitor did not stop on the confirmed basin frame: "
                f"monitor={observations[-1]['timestep']}, "
                f"confirmation={monitor_state['first_passage_confirmation_timestep']}."
            )
        if last_step != int(branch["maximum_steps"]):
            raise RuntimeError(
                f"Nested fixed-horizon trajectory stopped early: last={last_step}, "
                f"maximum={branch['maximum_steps']}."
            )

        source_size = trajectory_path.stat().st_size
        source_sha256 = _sha256_file(trajectory_path)
        binary_dir = branch_dir / "trajectory_binary_float16"
        binary = convert_shooting_trajectory(
            trajectory_path,
            binary_dir,
            timesteps=tuple(int(value) for value in scan.timesteps),
            atom_count=EXPECTED_ATOM_COUNT,
            storage_dtype=NESTED_STORAGE_DTYPE,
            provenance={
                "campaign_type": manifest["campaign_type"],
                "branch_id": branch["branch_id"],
                "source_lammpstrj": {
                    "path": str(trajectory_path),
                    "size_bytes": source_size,
                    "sha256": source_sha256,
                },
            },
        )
        binary.verify_checksums()
        initial_forces = _load_initial_forces(initial_state_path, EXPECTED_ATOM_COUNT)
        temperatures_K = _load_sampled_temperatures(branch_dir, scan.timesteps)
        observables_path = branch_dir / "observables.npz"
        observables = _write_observables(
            observables_path,
            saved_timesteps=scan.timesteps,
            temperatures_K=temperatures_K,
            initial_forces=initial_forces,
            monitor_state=monitor_state,
            censored=censored,
        )
        _validate_observables(observables_path, observables)

        for staging_path in (
            trajectory_path,
            initial_state_path,
            branch_dir / "monitor_frame.lammpstrj",
            branch_dir / "sampled_temperature.tsv",
            branch_dir / "initial_temperature.txt",
            branch_dir / "monitor_decision.txt",
        ):
            if staging_path.exists():
                staging_path.unlink()
        outcome = {
            **branch,
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "elapsed_seconds": elapsed_seconds,
            "input_artifact": {
                "path": str(branch_dir / "in.lammps"),
                "sha256": _sha256_file(branch_dir / "in.lammps"),
            },
            "frame_count": int(scan.frame_count),
            "first_timestep": int(scan.timesteps[0]),
            "last_timestep": last_step,
            "first_passage_outcome": first_passage_outcome,
            "first_passage_onset_timestep": monitor_state.get(
                "first_passage_onset_timestep"
            ),
            "first_passage_confirmation_timestep": monitor_state.get(
                "first_passage_confirmation_timestep"
            ),
            "first_passage_time_ps": monitor_state.get("first_passage_time_ps"),
            "censored": censored,
            "trajectory_artifact": {
                "format": "pointcloudmaterials.shooting_trajectory",
                "storage_dtype": NESTED_STORAGE_DTYPE,
                "path": str(binary_dir),
                "frame_count": binary.frame_count,
                "source_lammpstrj": {
                    "path": str(trajectory_path),
                    "size_bytes": source_size,
                    "sha256": source_sha256,
                    "deleted": True,
                },
            },
            "observables_artifact": observables,
            "restart_size_bytes": restart_path.stat().st_size,
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
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_artifacts_preserved": True,
            },
        )
        raise


def summarize_nested_campaign(campaign_root: str | Path) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    parents = manifest.get("parents")
    if not isinstance(branches, list) or not isinstance(parents, list):
        raise TypeError(f"{root / 'manifest.json'}: parents and branches must be lists.")
    outcomes: list[dict[str, Any]] = []
    missing: list[str] = []
    for branch in branches:
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        if not outcome_path.is_file():
            missing.append(str(branch["branch_id"]))
            continue
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Nested branch outcome is not complete: {outcome_path}.")
        input_artifact = outcome.get("input_artifact")
        if not isinstance(input_artifact, dict):
            raise RuntimeError(f"Nested outcome has no checksum-bound input: {outcome_path}.")
        input_path = Path(str(input_artifact["path"]))
        if _sha256_file(input_path) != input_artifact["sha256"]:
            raise RuntimeError(f"Nested branch input checksum changed: {input_path}.")
        binary = ShootingBinaryTrajectory.load(
            Path(str(outcome["trajectory_artifact"]["path"]))
        )
        binary.verify_checksums()
        if (
            binary.storage_dtype != np.dtype(NESTED_STORAGE_DTYPE)
            or int(binary.timesteps[-1]) != int(outcome["maximum_steps"])
            or binary.frame_count != int(outcome["frame_count"])
            or int(binary.timesteps[-1]) != int(outcome["last_timestep"])
        ):
            raise RuntimeError(
                f"Nested binary trajectory disagrees with outcome: {outcome_path}."
            )
        observables_description = outcome["observables_artifact"]
        _validate_observables(
            Path(str(observables_description["path"])), observables_description
        )
        outcomes.append(outcome)
    if missing:
        raise RuntimeError(
            f"Cannot summarize incomplete nested campaign: missing={len(missing)}, "
            f"first={missing[:10]}."
        )

    expected_children = int(
        manifest["counts"]["branches"] // manifest["counts"]["parents"]
    )
    parent_summaries: list[dict[str, Any]] = []
    for parent in parents:
        selected = [
            outcome
            for outcome in outcomes
            if int(outcome["parent_index"]) == int(parent["parent_index"])
        ]
        if len(selected) != expected_children:
            raise RuntimeError(
                f"Nested parent {parent['parent_id']} has {len(selected)} children; "
                f"expected {expected_children}."
            )
        by_momentum: dict[int, list[dict[str, Any]]] = {}
        for outcome in selected:
            by_momentum.setdefault(int(outcome["momentum_index"]), []).append(outcome)
        for momentum_index, children in by_momentum.items():
            momentum_seeds = {int(child["momentum_seed"]) for child in children}
            thermostat_seeds = {int(child["thermostat_seed"]) for child in children}
            if len(momentum_seeds) != 1 or len(thermostat_seeds) != len(children):
                raise RuntimeError(
                    f"Nested seed structure changed for parent={parent['parent_id']}, "
                    f"momentum_index={momentum_index}."
                )
        counts = {
            "basin_A_liquid": sum(
                outcome["first_passage_outcome"] == "basin_A_liquid"
                for outcome in selected
            ),
            "basin_B_crystal": sum(
                outcome["first_passage_outcome"] == "basin_B_crystal"
                for outcome in selected
            ),
            "censored": sum(bool(outcome["censored"]) for outcome in selected),
        }
        resolved = counts["basin_A_liquid"] + counts["basin_B_crystal"]
        p_B = None if resolved == 0 else counts["basin_B_crystal"] / resolved
        parent_summaries.append(
            {
                "parent_index": parent["parent_index"],
                "parent_id": parent["parent_id"],
                "temperature_K": parent["temperature_K"],
                "source_run_id": parent["source_run_id"],
                "source_split": parent["source_split"],
                "basin_role": parent["basin_role"],
                "source_largest_crystalline_cluster_atoms": parent[
                    "source_largest_crystalline_cluster_atoms"
                ],
                "counts": counts,
                "p_B_resolved": p_B,
                "retain_as_mixed_parent": (
                    parent["basin_role"] == "transition_candidate"
                    and p_B is not None
                    and 0.2 < p_B < 0.8
                ),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "branch_count": len(outcomes),
        "parent_count": len(parent_summaries),
        "outcome_counts": {
            outcome_name: sum(
                outcome["first_passage_outcome"] == outcome_name for outcome in outcomes
            )
            for outcome_name in ("basin_A_liquid", "basin_B_crystal")
        }
        | {"censored": sum(bool(outcome["censored"]) for outcome in outcomes)},
        "counts_by_temperature": {
            f"{temperature:g}": {
                "branches": sum(
                    outcome["temperature_K"] == temperature for outcome in outcomes
                ),
                "basin_A_liquid": sum(
                    outcome["temperature_K"] == temperature
                    and outcome["first_passage_outcome"] == "basin_A_liquid"
                    for outcome in outcomes
                ),
                "basin_B_crystal": sum(
                    outcome["temperature_K"] == temperature
                    and outcome["first_passage_outcome"] == "basin_B_crystal"
                    for outcome in outcomes
                ),
                "censored": sum(
                    outcome["temperature_K"] == temperature
                    and bool(outcome["censored"])
                    for outcome in outcomes
                ),
            }
            for temperature in TEMPERATURES_K
        },
        "counts_by_source_split": {
            split: sum(outcome["source_split"] == split for outcome in outcomes)
            for split in SPLIT_NAMES
        },
        "mixed_transition_parent_count": sum(
            bool(parent["retain_as_mixed_parent"]) for parent in parent_summaries
        ),
        "maximum_censoring_fraction_by_temperature": max(
            sum(
                outcome["temperature_K"] == temperature
                and bool(outcome["censored"])
                for outcome in outcomes
            )
            / sum(outcome["temperature_K"] == temperature for outcome in outcomes)
            for temperature in TEMPERATURES_K
        ),
        "parents": parent_summaries,
    }
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _active_submission_conflicts(root: Path) -> list[str]:
    path = root / "slurm" / "active_submission.json"
    if not path.is_file():
        return []
    active = _load_json(path)
    job_ids = [str(active["array_job_id"]), str(active["successor_job_id"])]
    queued = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"],
        check=True,
        text=True,
        capture_output=True,
    )
    current_job_id = os.environ.get("SLURM_JOB_ID")
    return sorted(
        {
            value.strip()
            for value in queued.stdout.splitlines()
            if value.strip() and value.strip() != current_job_id
        }
    )


def submit_next_nested_wave(
    campaign_root: str | Path, start_index: int
) -> dict[str, Any]:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    if not isinstance(branches, list) or not branches:
        raise TypeError(f"{root / 'manifest.json'}: branches must be nonempty.")
    start = int(start_index)
    if start < 0 or start >= len(branches):
        raise IndexError(f"start_index={start} is outside [0, {len(branches)}).")
    conflicts = _active_submission_conflicts(root)
    if conflicts:
        raise RuntimeError(
            f"Refusing duplicate nested submission while jobs remain active: {conflicts}."
        )
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
            f"Slurm returned invalid nested array ID: stdout={array_submission.stdout!r}, "
            f"stderr={array_submission.stderr!r}."
        )
    if stop + 1 < len(branches):
        successor_kind = "controller"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            f"--export=ALL,NESTED_START={stop + 1}",
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
    successor_submission = subprocess.run(
        successor_command,
        check=True,
        text=True,
        capture_output=True,
    )
    successor_job_id = successor_submission.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(
            f"Slurm returned invalid nested {successor_kind} ID: "
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **record,
        },
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


__all__ = [
    "NestedShootingConfig",
    "evaluate_monitor_frame",
    "load_nested_shooting_config",
    "multirate_output_steps",
    "nested_random_seeds",
    "prepare_nested_campaign",
    "render_nested_lammps_input",
    "run_nested_branch",
    "submit_next_nested_wave",
    "summarize_nested_campaign",
]
