#!/usr/bin/env python3
"""Manifest-driven fixed-48 ps predictive-dynamics shooting campaigns.

This runner is intentionally layered on the validated position-shooting
producer.  A task first completes the ordinary LAMMPS trajectory, then computes
the full PTM first-passage record, converts the dump to verified float32 arrays,
deletes only that verified transient text dump, and finally publishes the
canonical ``outcome.json``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.simulation.campaigns.independent_meam_source import _ptm_progress  # noqa: E402
from src.data_utils.shooting_binary import (  # noqa: E402
    FORMAT_NAME,
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    compose_shooting_binary_trajectories,
    convert_shooting_trajectory,
)
from src.data_utils.shooting_dataset import (  # noqa: E402
    resolve_shooting_trajectory_path,
    validate_complete_shooting_branch,
)
from src.data_utils.synthetic.atomistic.lammps_shooting import (  # noqa: E402
    render_lammps_input,
    run_branch as run_lammps_dynamics,
)


SCHEMA_VERSION = 1
CAMPAIGN_TYPE = "position_conditioned_langevin_nvt_shooting"
DESIGN_NAME = "predictive_dynamics_fixed48_float32"
CAMPAIGN_SEED = 20_260_903
ATOM_COUNT = 70_304
TIMESTEP_FS = 3.0
DURATION_PS = 48.0
RUN_STEPS = 16_000
SAMPLE_INTERVAL_STEPS = 100
EXPECTED_TIMESTEPS = tuple(range(0, RUN_STEPS + 1, SAMPLE_INTERVAL_STEPS))
EXPECTED_FRAME_COUNT = len(EXPECTED_TIMESTEPS)
THERMOSTAT_TIME_FS = 300.0
LAMMPS_MAX_SEED = 900_000_000
PERSISTENCE_FRAMES = 3
BASIN_B_MIN = 100
BASIN_A_MAX = {400.0: 19, 450.0: 20, 500.0: 16}
LIBRARY_SHA256 = "f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e"
PARAMETER_SHA256 = "b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a"
DEFAULT_SNAPSHOT = Path(
    "/home/ids/vmorozov/experiments/"
    "predictive_atlas_geoframe_v2_480branches_20260902/dataset_snapshot.json"
)
DEFAULT_TOPUP_ROOT = Path(
    "/home/ids/vmorozov/simulations/"
    "al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903"
)
DEFAULT_SMOKE_ROOT = Path(
    "/home/ids/vmorozov/simulations/"
    "al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903"
)
SMOKE_SOURCE_SNAPSHOT_BRANCH_INDEX = 4
SMOKE_PARENT_SOURCE_TIMESTEP = 4_000


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


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (LAMMPS_MAX_SEED - 1) + 1


def _verify_potential(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required MEAM potential file is missing: {path}")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"MEAM potential checksum mismatch: path={path}, expected={expected_sha256}, "
            f"observed={observed}."
        )


def _new_root(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing campaign root: {root}")
    root.mkdir(parents=True)
    for name in ("parents", "branches", "potential", "slurm"):
        (root / name).mkdir()


def _copy_potential(root: Path, source_root: Path | None = None) -> dict[str, Any]:
    if source_root is None:
        library = REPOSITORY_ROOT / "datasets/potentials/Lee2003_Al.library.meam"
        parameter = REPOSITORY_ROOT / "datasets/potentials/Lee2003_Al.meam"
    else:
        library = source_root / "potential/Lee2003_Al.library.meam"
        parameter = source_root / "potential/Lee2003_Al.meam"
    _verify_potential(library, LIBRARY_SHA256)
    _verify_potential(parameter, PARAMETER_SHA256)
    shutil.copy2(library, root / "potential" / library.name)
    shutil.copy2(parameter, root / "potential" / parameter.name)
    return {
        "name": "Lee-Shim-Baskes 2003 Al 2NN-MEAM",
        "library_file": "potential/Lee2003_Al.library.meam",
        "library_sha256": LIBRARY_SHA256,
        "parameter_file": "potential/Lee2003_Al.meam",
        "parameter_sha256": PARAMETER_SHA256,
    }


def _protocol() -> dict[str, Any]:
    return {
        "ensemble": "fixed-cell Langevin NVT",
        "timestep_fs": TIMESTEP_FS,
        "duration_ps": DURATION_PS,
        "run_steps": RUN_STEPS,
        "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
        "sample_interval_ps": 0.3,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "thermostat_time_fs": THERMOSTAT_TIME_FS,
        "dump_columns": ["id", "type", "x", "y", "z", "vx", "vy", "vz"],
        "independent_momenta_per_branch": False,
        "storage_dtype": "float32",
        "transient_text_dump_deleted_after_verified_conversion": True,
        "mandatory_fixed_horizon_after_basin_arrival": True,
    }


def _execution(wave_size: int) -> dict[str, Any]:
    if wave_size <= 0 or wave_size > 6:
        raise ValueError(f"wave_size must be within [1, 6], got {wave_size}.")
    return {
        "mpi_ranks_per_branch": 24,
        "launcher": "srun_pmi2",
        "partition": "CPU",
        "time_limit": "04:00:00",
        "memory": "24G",
        "array_concurrency": wave_size,
        "wave_size": wave_size,
        "normal_qos_max_submitted_jobs": 10,
    }


def _scientific_contract() -> dict[str, Any]:
    return {
        "exact_restart": False,
        "interpretation": (
            "Fixed-cell Langevin-NVT futures conditioned on immutable positions, cell, "
            "history, shooting temperature, sampled momentum, and thermostat stream."
        ),
        "no_equilibration_after_branching": True,
        "first_passage_does_not_stop_fixed_trajectory": True,
        "bootstrap_unit": "root_source_lineage",
        "overlapping_windows_are_correlated": True,
    }


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "manifest.json"
    _write_json_atomic(path, manifest)
    (root / "manifest.sha256").write_text(_sha256_file(path) + "  manifest.json\n", encoding="ascii")
    _write_json_atomic(
        root / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "prepared",
            "updated_at": _utc_now(),
            "branch_count": len(manifest["branches"]),
        },
    )


def _branch_input(root: Path, parent: dict[str, Any], branch: dict[str, Any]) -> None:
    branch_dir = root / str(branch["branch_dir"])
    branch_dir.mkdir()
    (branch_dir / "in.lammps").write_text(
        render_lammps_input(
            parent_id=str(parent["parent_id"]),
            branch_id=str(branch["branch_id"]),
            temperature_K=float(branch["temperature_K"]),
            velocity_seed=int(branch["velocity_seed"]),
            thermostat_seed=int(branch["thermostat_seed"]),
            timestep_fs=TIMESTEP_FS,
            thermostat_time_fs=THERMOSTAT_TIME_FS,
            sample_interval_steps=SAMPLE_INTERVAL_STEPS,
            run_steps=RUN_STEPS,
        ),
        encoding="utf-8",
    )
    _write_json_atomic(branch_dir / "metadata.json", branch)


def _write_slurm_scripts(root: Path, *, job_prefix: str) -> None:
    runner = Path(__file__).resolve()
    execution = _load_json(root / "manifest.json")["execution"]
    common = f"""set -euo pipefail
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH={REPOSITORY_ROOT}
"""
    task = root / "slurm/run_branch.sbatch"
    task.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={job_prefix}
#SBATCH --partition={execution['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks={execution['mpi_ranks_per_branch']}
#SBATCH --ntasks-per-node={execution['mpi_ranks_per_branch']}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem={execution['memory']}
#SBATCH --time={execution['time_limit']}
#SBATCH --output={root}/slurm/%A_%a.out
#SBATCH --error={root}/slurm/%A_%a.err

{common}python {runner} run-task --campaign-root {root} --task-index "${{SLURM_ARRAY_TASK_ID}}"
""",
        encoding="utf-8",
    )
    task.chmod(0o750)
    controller = root / "slurm/submit_wave.sbatch"
    controller.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={job_prefix}_ctl
#SBATCH --partition={execution['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output={root}/slurm/%j_controller.out
#SBATCH --error={root}/slurm/%j_controller.err

{common}: "${{PREDICTIVE_START:?PREDICTIVE_START must be set}}"
python {runner} submit-next-wave --campaign-root {root} --start-index "${{PREDICTIVE_START}}"
""",
        encoding="utf-8",
    )
    controller.chmod(0o750)
    summary = root / "slurm/summarize.sbatch"
    summary.write_text(
        f"""#!/bin/bash
#SBATCH --job-name={job_prefix}_sum
#SBATCH --partition={execution['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output={root}/slurm/%j_summary.out
#SBATCH --error={root}/slurm/%j_summary.err

{common}python {runner} summarize --campaign-root {root}
""",
        encoding="utf-8",
    )
    summary.chmod(0o750)


def _base_manifest(
    *,
    design_kind: str,
    parents: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    potential: dict[str, Any],
    wave_size: int,
) -> dict[str, Any]:
    temperatures = sorted({float(parent["temperature_K"]) for parent in parents})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "campaign_type": CAMPAIGN_TYPE,
        "design_name": DESIGN_NAME,
        "campaign_seed": CAMPAIGN_SEED,
        "design_kind": design_kind,
        "scientific_contract": _scientific_contract(),
        "atom_count": ATOM_COUNT,
        "source_config": {"temperatures_K": temperatures},
        "potential": potential,
        "protocol": _protocol(),
        "basins": {
            "A_max_cluster_atoms_by_temperature": {
                f"{temperature:g}": BASIN_A_MAX[temperature] for temperature in temperatures
            },
            "B_min_cluster_atoms": BASIN_B_MIN,
            "persistence_frames": PERSISTENCE_FRAMES,
            "ptm_rmsd_cutoff": 0.1,
            "cluster_connectivity_cutoff_A": 3.5,
        },
        "execution": _execution(wave_size),
        "counts": {
            "parents": len(parents),
            "branches": len(branches),
            "branches_by_split": dict(Counter(str(item["source_split"]) for item in branches)),
        },
        "parents": parents,
        "branches": branches,
    }


def prepare_topup(snapshot_path: Path, campaign_root: Path, *, wave_size: int) -> dict[str, Any]:
    snapshot_path = snapshot_path.expanduser().resolve()
    root = campaign_root.expanduser().resolve()
    snapshot = _load_json(snapshot_path)
    old_parents = snapshot.get("parents")
    old_branches = snapshot.get("branches")
    campaign_roots = snapshot.get("campaign_roots")
    if not isinstance(old_parents, list) or len(old_parents) != 40:
        raise RuntimeError(
            f"Authoritative snapshot must contain exactly 40 parents, got "
            f"{len(old_parents) if isinstance(old_parents, list) else type(old_parents).__name__}: "
            f"{snapshot_path}."
        )
    if not isinstance(old_branches, list) or len(old_branches) != 480:
        raise RuntimeError(
            f"Authoritative snapshot must contain exactly 480 branches, got "
            f"{len(old_branches) if isinstance(old_branches, list) else type(old_branches).__name__}: "
            f"{snapshot_path}."
        )
    if not isinstance(campaign_roots, list) or len(campaign_roots) != 4:
        raise RuntimeError(f"Expected four source campaign roots in {snapshot_path}.")
    source_root = Path(str(snapshot["campaign_root"])).resolve()
    source_manifest = _load_json(source_root / "manifest.json")
    if source_manifest.get("campaign_type") != CAMPAIGN_TYPE:
        raise RuntimeError(f"Unexpected source campaign type in {source_root / 'manifest.json'}.")

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_pairs: set[tuple[int, int]] = set()
    old_canonical: dict[str, int] = {}
    for branch in old_branches:
        parent_id = str(branch["parent_id"])
        pair = (int(branch["velocity_seed"]), int(branch["thermostat_seed"]))
        if pair in observed_pairs:
            raise RuntimeError(f"Duplicate seed pair in authoritative snapshot: {pair}.")
        observed_pairs.add(pair)
        by_parent[parent_id].append(branch)
    for parent in old_parents:
        parent_id = str(parent["parent_id"])
        values = by_parent[parent_id]
        if len(values) != 12:
            raise RuntimeError(f"Parent {parent_id} has {len(values)} old branches, expected 12.")
        for canonical_index, branch in enumerate(values):
            old_canonical[str(branch["branch_uid"])] = canonical_index

    _new_root(root)
    potential = _copy_potential(root, source_root)
    parents: list[dict[str, Any]] = []
    for value in old_parents:
        parent = dict(value)
        parent_id = str(parent["parent_id"])
        source_data = source_root / str(parent["data_file"])
        if _sha256_file(source_data) != str(parent["data_sha256"]):
            raise RuntimeError(f"Immutable source parent checksum changed: {source_data}.")
        parent_dir = root / "parents" / parent_id
        parent_dir.mkdir()
        target_data = parent_dir / "parent.lammps.data"
        shutil.copy2(source_data, target_data)
        parent["data_file"] = str(target_data.relative_to(root))
        parent["data_sha256"] = _sha256_file(target_data)
        parent["source_temperature_K"] = float(parent["temperature_K"])
        parent["shooting_temperature_K"] = float(parent["temperature_K"])
        parent["history_available"] = False
        parent["history_note"] = (
            "Legacy immutable parent retained for the broad fixed-horizon baseline; "
            "no dense 0.3 ps prehistory is fabricated."
        )
        _write_json_atomic(parent_dir / "metadata.json", parent)
        parents.append(parent)

    branches: list[dict[str, Any]] = []
    used_new_seeds: set[int] = set()
    for parent in parents:
        for canonical_shot_index in range(12, 16):
            parent_id = str(parent["parent_id"])
            velocity_seed = _seed(
                DESIGN_NAME, "topup", CAMPAIGN_SEED, parent_id, canonical_shot_index, "velocity"
            )
            thermostat_seed = _seed(
                DESIGN_NAME, "topup", CAMPAIGN_SEED, parent_id, canonical_shot_index, "thermostat"
            )
            pair = (velocity_seed, thermostat_seed)
            if pair in observed_pairs:
                raise RuntimeError(f"Generated top-up seed pair already exists: {pair}.")
            if velocity_seed == thermostat_seed or velocity_seed in used_new_seeds or thermostat_seed in used_new_seeds:
                raise RuntimeError(
                    f"Generated random seed collision for parent={parent_id}, "
                    f"shot={canonical_shot_index}: {pair}."
                )
            observed_pairs.add(pair)
            used_new_seeds.update(pair)
            branch_index = len(branches)
            branch_id = (
                f"branch_{branch_index:04d}_{parent_id}_canonical_shot_{canonical_shot_index:02d}"
            )
            branch = {
                "branch_index": branch_index,
                "branch_id": branch_id,
                "branch_dir": f"branches/{branch_id}",
                "parent_index": int(parent["parent_index"]),
                "parent_id": parent_id,
                "source_run_id": str(parent["source_run_id"]),
                "root_source_lineage_id": str(parent["source_run_id"]),
                "source_split": str(parent["source_split"]),
                "source_velocity_seed": int(parent["source_velocity_seed"]),
                "temperature_K": float(parent["temperature_K"]),
                "source_temperature_K": float(parent["temperature_K"]),
                "shooting_temperature_K": float(parent["temperature_K"]),
                "phase": str(parent["phase"]),
                "basin_role": "legacy_pre_nucleation",
                "shot_index": canonical_shot_index,
                "canonical_shot_index": canonical_shot_index,
                "momentum_index": canonical_shot_index,
                "thermostat_replica_index": 0,
                "momentum_group_id": f"{parent_id}__momentum_{canonical_shot_index:02d}",
                "velocity_seed": velocity_seed,
                "thermostat_seed": thermostat_seed,
            }
            _branch_input(root, parent, branch)
            branches.append(branch)

    manifest = _base_manifest(
        design_kind="legacy_40_parent_topup_from_12_to_16",
        parents=parents,
        branches=branches,
        potential=potential,
        wave_size=wave_size,
    )
    manifest["authoritative_existing_snapshot"] = {
        "path": str(snapshot_path),
        "sha256": _sha256_file(snapshot_path),
        "existing_parent_count": 40,
        "existing_branch_count": 480,
        "existing_canonical_shot_index_by_branch_uid": old_canonical,
        "source_campaign_roots": [str(Path(value).resolve()) for value in campaign_roots],
    }
    manifest["counts"]["new_futures_per_parent"] = 4
    manifest["counts"]["merged_futures_per_parent"] = 16
    manifest["protocol"]["independent_momenta_per_branch"] = True
    manifest["window_index"] = {
        "latest_start_ps": 24.0,
        "stride_ps": 1.2,
        "maximum_prediction_horizon_ps": 24.0,
        "include_authoritative_existing_snapshot_branches": True,
    }
    _write_manifest(root, manifest)
    _write_slurm_scripts(root, job_prefix="al_pred_topup")
    return manifest


def _write_parent_data(binary: ShootingBinaryTrajectory, timestep: int, path: Path) -> None:
    frame = binary.load_position_frames((timestep,))[timestep]
    if not np.array_equal(frame.atom_ids, np.arange(1, ATOM_COUNT + 1, dtype=np.int64)):
        raise RuntimeError(f"Source frame atom IDs changed before parent creation: {binary.root}.")
    if not np.all(frame.atom_types == 1):
        raise RuntimeError(f"Source frame contains non-Al atom types: {binary.root}.")
    atoms = Atoms(
        symbols=["Al"] * ATOM_COUNT,
        positions=np.asarray(frame.positions, dtype=np.float64),
        cell=np.diag(np.asarray(frame.box_lengths, dtype=np.float64)),
        pbc=True,
    )
    write(path, atoms, format="lammps-data", atom_style="atomic")


def _ptm_descriptor_from_binary(
    binary: ShootingBinaryTrajectory, timestep: int
) -> dict[str, Any]:
    try:
        from ovito.data import DataCollection, Particles, SimulationCell
        from ovito.modifiers import (
            ClusterAnalysisModifier,
            PolyhedralTemplateMatchingModifier,
        )
    except ImportError as exc:
        raise ImportError("Smoke parent characterization requires OVITO in pointnet.") from exc
    indices = np.flatnonzero(np.asarray(binary.timesteps) == timestep)
    if len(indices) != 1:
        raise RuntimeError(f"Expected one source frame at timestep={timestep}: {binary.root}.")
    frame_index = int(indices[0])
    data = DataCollection()
    particles = Particles(count=binary.atom_count)
    particles.create_property(
        "Position", data=np.asarray(binary.positions[frame_index], dtype=np.float64)
    )
    data.objects.append(particles)
    cell = SimulationCell(pbc=(True, True, True))
    lengths = np.asarray(
        binary.box_high[frame_index] - binary.box_low[frame_index], dtype=np.float64
    )
    cell[...] = (
        (lengths[0], 0.0, 0.0, 0.0),
        (0.0, lengths[1], 0.0, 0.0),
        (0.0, 0.0, lengths[2], 0.0),
    )
    data.objects.append(cell)
    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = 0.1
    data.apply(ptm)
    structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
    crystalline = np.isin(structure_types, (1, 2, 3))
    data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
    clusters = ClusterAnalysisModifier(cutoff=3.5, only_selected=True, sort_by_size=True)
    data.apply(clusters)
    return {
        "largest_crystalline_cluster_atoms": int(data.attributes["ClusterAnalysis.largest_size"]),
        "crystalline_fraction": float(np.mean(crystalline)),
        "crystalline_cluster_count": int(data.attributes["ClusterAnalysis.cluster_count"]),
        "ptm_rmsd_cutoff": 0.1,
        "cluster_connectivity_cutoff_A": 3.5,
        "source_binary_path": str(binary.root),
        "source_binary_manifest_sha256": _sha256_file(binary.root / "manifest.json"),
        "source_timestep": timestep,
    }


def prepare_smoke(snapshot_path: Path, campaign_root: Path, *, wave_size: int) -> dict[str, Any]:
    snapshot_path = snapshot_path.expanduser().resolve()
    root = campaign_root.expanduser().resolve()
    snapshot = _load_json(snapshot_path)
    branches_snapshot = snapshot.get("branches")
    if not isinstance(branches_snapshot, list) or len(branches_snapshot) != 480:
        raise RuntimeError(f"Smoke preparation requires the authoritative 480-branch snapshot.")
    source_branch = branches_snapshot[SMOKE_SOURCE_SNAPSHOT_BRANCH_INDEX]
    source_root = Path(str(source_branch["campaign_root"])).resolve()
    source_manifest = _load_json(source_root / "manifest.json")
    source_outcome = _load_json(
        source_root / str(source_branch["branch_dir"]) / "outcome.json"
    )
    if source_outcome.get("state") != "complete":
        raise RuntimeError(f"Smoke ancestor is not complete: {source_branch['branch_uid']}.")
    source_binary_path = resolve_shooting_trajectory_path(source_root, source_branch)
    source_binary = ShootingBinaryTrajectory.load(source_binary_path)
    source_binary.verify_checksums()
    if source_binary.storage_dtype != np.dtype("float32"):
        raise RuntimeError(f"Smoke ancestor must be float32, got {source_binary.storage_dtype}.")

    descriptor = _ptm_descriptor_from_binary(source_binary, SMOKE_PARENT_SOURCE_TIMESTEP)
    largest = int(descriptor["largest_crystalline_cluster_atoms"])
    fraction = float(descriptor["crystalline_fraction"])
    if not (BASIN_A_MAX[400.0] < largest < BASIN_B_MIN):
        raise RuntimeError(f"Selected smoke shifted state is not transitional: largest={largest}.")

    _new_root(root)
    potential = _copy_potential(root, source_root)
    parent_id = "smoke_shifted_parent_T400_ancestor_branch004_t12ps"
    parent_dir = root / "parents" / parent_id
    parent_dir.mkdir()
    history_timesteps = tuple(range(0, SMOKE_PARENT_SOURCE_TIMESTEP + 1, SAMPLE_INTERVAL_STEPS))
    history_path = parent_dir / "history_binary_float32"
    history = compose_shooting_binary_trajectories(
        (source_binary,),
        history_path,
        timesteps=history_timesteps,
        storage_dtype="float32",
        provenance={
            "purpose": "immutable_12ps_parent_history",
            "root_source_lineage_id": source_branch["source_run_id"],
            "ancestor_branch_uid": source_branch["branch_uid"],
            "relative_timestep_origin": SMOKE_PARENT_SOURCE_TIMESTEP,
        },
    )
    parent_data = parent_dir / "parent.lammps.data"
    _write_parent_data(source_binary, SMOKE_PARENT_SOURCE_TIMESTEP, parent_data)
    source_parent = next(
        value for value in snapshot["parents"] if value["parent_id"] == source_branch["parent_id"]
    )
    parent = {
        "parent_index": 0,
        "parent_id": parent_id,
        "data_file": str(parent_data.relative_to(root)),
        "data_sha256": _sha256_file(parent_data),
        "temperature_K": 400.0,
        "source_temperature_K": 400.0,
        "shooting_temperature_K": 400.0,
        "source_run_id": str(source_branch["source_run_id"]),
        "root_source_lineage_id": str(source_branch["source_run_id"]),
        "source_split": "optimization",
        "source_velocity_seed": int(source_branch["source_velocity_seed"]),
        "phase": "transition_candidate",
        "basin_role": "transition_candidate",
        "source_physical_time_ps": float(source_parent["source_frame_time_ps"]) + 12.0,
        "ancestor_branch_uid": str(source_branch["branch_uid"]),
        "ancestor_branch_relative_time_ps": 12.0,
        "source_largest_crystalline_cluster_atoms": largest,
        "source_crystalline_fraction": fraction,
        "source_descriptor": descriptor,
        "history_available": True,
        "history_artifact": {
            "path": str(history_path.relative_to(root)),
            "storage_dtype": history.storage_dtype.name,
            "frame_count": history.frame_count,
            "source_timesteps": list(history_timesteps),
            "relative_times_ps": [
                (step - SMOKE_PARENT_SOURCE_TIMESTEP) * TIMESTEP_FS / 1000.0
                for step in history_timesteps
            ],
            "required_relative_times_ps": [-12.0, -9.0, -6.0, -3.0, 0.0],
            "array_sha256": {
                name: str(description["sha256"])
                for name, description in history.manifest["arrays"].items()
            },
        },
    }
    _write_json_atomic(parent_dir / "metadata.json", parent)

    branches: list[dict[str, Any]] = []
    used_thermostat_seeds: set[int] = set()
    used_velocity_seeds: set[int] = set()
    for momentum_index in range(8):
        velocity_seed = _seed(DESIGN_NAME, "smoke", CAMPAIGN_SEED, parent_id, momentum_index, "velocity")
        if velocity_seed in used_velocity_seeds:
            raise RuntimeError(f"Smoke velocity-seed collision: {velocity_seed}.")
        used_velocity_seeds.add(velocity_seed)
        for thermostat_replica_index in range(2):
            thermostat_seed = _seed(
                DESIGN_NAME,
                "smoke",
                CAMPAIGN_SEED,
                parent_id,
                momentum_index,
                thermostat_replica_index,
                "thermostat",
            )
            if thermostat_seed == velocity_seed or thermostat_seed in used_thermostat_seeds:
                raise RuntimeError(f"Smoke thermostat-seed collision: {thermostat_seed}.")
            used_thermostat_seeds.add(thermostat_seed)
            branch_index = len(branches)
            branch_id = (
                f"branch_{branch_index:04d}_{parent_id}_momentum_{momentum_index:02d}_"
                f"noise_{thermostat_replica_index:02d}"
            )
            branch = {
                "branch_index": branch_index,
                "branch_id": branch_id,
                "branch_dir": f"branches/{branch_id}",
                "parent_index": 0,
                "parent_id": parent_id,
                "source_run_id": str(parent["source_run_id"]),
                "root_source_lineage_id": str(parent["root_source_lineage_id"]),
                "source_split": "optimization",
                "source_velocity_seed": int(parent["source_velocity_seed"]),
                "temperature_K": 400.0,
                "source_temperature_K": 400.0,
                "shooting_temperature_K": 400.0,
                "phase": "transition_candidate",
                "basin_role": "transition_candidate",
                "shot_index": branch_index,
                "canonical_shot_index": branch_index,
                "momentum_index": momentum_index,
                "thermostat_replica_index": thermostat_replica_index,
                "momentum_group_id": f"{parent_id}__momentum_{momentum_index:02d}",
                "velocity_seed": velocity_seed,
                "thermostat_seed": thermostat_seed,
            }
            _branch_input(root, parent, branch)
            branches.append(branch)

    manifest = _base_manifest(
        design_kind="single_parent_preproduction_smoke_8x2",
        parents=[parent],
        branches=branches,
        potential=potential,
        wave_size=wave_size,
    )
    manifest["smoke_test"] = {
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": _sha256_file(snapshot_path),
        "ancestor_branch_uid": source_branch["branch_uid"],
        "ancestor_branch_outcome_sha256": _sha256_file(
            source_root / str(source_branch["branch_dir"]) / "outcome.json"
        ),
        "selection_uses_present_descriptor_only": True,
    }
    manifest["window_index"] = {
        "latest_start_ps": 24.0,
        "stride_ps": 1.2,
        "maximum_prediction_horizon_ps": 24.0,
    }
    _write_manifest(root, manifest)
    _write_slurm_scripts(root, job_prefix="al_pred_smoke")
    return manifest


def _load_or_compute_progress(
    branch_dir: Path, expected_steps: np.ndarray
) -> tuple[dict[str, np.ndarray], Path]:
    path = branch_dir / "first_passage_progress.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            progress = {name: np.asarray(archive[name]) for name in archive.files}
        required = {
            "step",
            "time_ps",
            "structure_names",
            "structure_fractions",
            "crystalline_fraction",
            "crystalline_cluster_count",
            "largest_crystalline_cluster_atoms",
        }
        if set(progress) != required:
            raise RuntimeError(
                f"Existing first-passage progress has unexpected arrays: path={path}, "
                f"expected={sorted(required)}, observed={sorted(progress)}."
            )
        if not np.array_equal(progress["step"], expected_steps):
            raise RuntimeError(f"Existing first-passage timesteps changed: {path}.")
        return progress, path
    trajectory = branch_dir / "trajectory.lammpstrj"
    progress = _ptm_progress(trajectory, expected_steps)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **progress)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return progress, path


def _first_passage(
    largest: np.ndarray, *, basin_a_max: int, basin_b_min: int
) -> dict[str, Any]:
    values = np.asarray(largest, dtype=np.int64)
    if values.shape != (EXPECTED_FRAME_COUNT,):
        raise RuntimeError(
            f"First-passage cluster series must have shape {(EXPECTED_FRAME_COUNT,)}, "
            f"got {values.shape}."
        )
    event: tuple[int, int, str] | None = None
    for confirmation_index in range(PERSISTENCE_FRAMES - 1, len(values)):
        onset_index = confirmation_index - PERSISTENCE_FRAMES + 1
        recent = values[onset_index : confirmation_index + 1]
        if bool(np.all(recent <= basin_a_max)):
            event = (onset_index, confirmation_index, "dissolution_to_basin_A")
            break
        if bool(np.all(recent >= basin_b_min)):
            event = (onset_index, confirmation_index, "crystallization_to_basin_B")
            break
    if event is None:
        return {
            "event": "censored",
            "first_passage_outcome": "censored",
            "censored": True,
            "onset_timestep": None,
            "onset_time_ps": None,
            "confirmation_timestep": None,
            "confirmation_time_ps": None,
            "maximum_observation_time_ps": DURATION_PS,
        }
    onset_index, confirmation_index, event_name = event
    return {
        "event": event_name,
        "first_passage_outcome": (
            "basin_A_liquid" if event_name == "dissolution_to_basin_A" else "basin_B_crystal"
        ),
        "censored": False,
        "onset_timestep": int(EXPECTED_TIMESTEPS[onset_index]),
        "onset_time_ps": float(EXPECTED_TIMESTEPS[onset_index] * TIMESTEP_FS / 1000.0),
        "confirmation_timestep": int(EXPECTED_TIMESTEPS[confirmation_index]),
        "confirmation_time_ps": float(
            EXPECTED_TIMESTEPS[confirmation_index] * TIMESTEP_FS / 1000.0
        ),
        "maximum_observation_time_ps": DURATION_PS,
    }


def _postprocess_branch(root: Path, manifest: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    branch_dir = root / str(branch["branch_dir"])
    outcome_path = branch_dir / "outcome.json"
    dynamics_path = branch_dir / "dynamics_outcome.json"
    if outcome_path.is_file():
        outcome = _load_json(outcome_path)
        if outcome.get("state") == "complete" and outcome.get("design_name") == DESIGN_NAME:
            validate_complete_shooting_branch(root, manifest, branch, outcome)
            return outcome
        if outcome.get("state") != "complete":
            raise RuntimeError(f"Canonical branch outcome is not complete: {outcome_path}.")
        if dynamics_path.exists():
            raise RuntimeError(
                f"Both raw and canonical outcomes exist for an interrupted branch: {branch_dir}."
            )
        outcome_path.replace(dynamics_path)
    if not dynamics_path.is_file():
        raise FileNotFoundError(f"Raw dynamics outcome is missing before postprocessing: {dynamics_path}")
    dynamics = _load_json(dynamics_path)
    if dynamics.get("state") != "complete":
        raise RuntimeError(f"Raw dynamics outcome is not complete: {dynamics_path}.")
    status_path = branch_dir / "status.json"
    _write_json_atomic(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "state": "postprocessing",
            "updated_at": _utc_now(),
            "branch_id": branch["branch_id"],
            "partial_artifacts_are_not_data": True,
        },
    )
    expected_steps = np.asarray(EXPECTED_TIMESTEPS, dtype=np.int64)
    progress, progress_path = _load_or_compute_progress(branch_dir, expected_steps)
    temperature = float(branch["temperature_K"])
    basin_a_max = int(manifest["basins"]["A_max_cluster_atoms_by_temperature"][f"{temperature:g}"])
    first_passage = _first_passage(
        progress["largest_crystalline_cluster_atoms"],
        basin_a_max=basin_a_max,
        basin_b_min=int(manifest["basins"]["B_min_cluster_atoms"]),
    )

    source = branch_dir / "trajectory.lammpstrj"
    target = branch_dir / "trajectory_binary_float32"
    source_size = int(dynamics["trajectory_size_bytes"])
    source_sha256: str
    if target.exists():
        binary = ShootingBinaryTrajectory.load(target)
        source_record = binary.manifest.get("provenance", {})
        source_sha256 = str(source_record.get("source_sha256", ""))
        if len(source_sha256) != 64:
            raise RuntimeError(
                f"Interrupted binary lacks the transient source checksum: {target / 'manifest.json'}."
            )
    else:
        if not source.is_file() or source.stat().st_size != source_size:
            raise RuntimeError(
                f"Transient trajectory is missing or changed before float32 conversion: {source}."
            )
        source_sha256 = _sha256_file(source)
        binary = convert_shooting_trajectory(
            source,
            target,
            timesteps=EXPECTED_TIMESTEPS,
            atom_count=ATOM_COUNT,
            storage_dtype="float32",
            provenance={
                "design_name": DESIGN_NAME,
                "campaign_manifest": str(root / "manifest.json"),
                "branch_id": branch["branch_id"],
                "source_sha256": source_sha256,
            },
        )
    if (
        binary.storage_dtype != np.dtype("float32")
        or binary.atom_count != ATOM_COUNT
        or tuple(binary.timesteps.tolist()) != EXPECTED_TIMESTEPS
    ):
        raise RuntimeError(f"Converted branch violates the mandatory binary contract: {target}.")
    checksums = binary.verify_checksums()
    sizes = binary_directory_sizes(target)
    if source.exists():
        if source.stat().st_size != source_size or _sha256_file(source) != source_sha256:
            raise RuntimeError(f"Transient text trajectory changed after conversion: {source}.")
        source.unlink()
    if source.exists():
        raise RuntimeError(f"Transient text trajectory still exists after verified deletion: {source}.")

    initial_velocities = np.asarray(binary.velocities[0], dtype=np.float32)
    restart = branch_dir / "final.restart.bin"
    if not restart.is_file() or restart.stat().st_size <= 0:
        raise RuntimeError(f"Final restart is missing or empty: {restart}.")
    artifact = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "path": "trajectory_binary_float32",
        "storage_dtype": "float32",
        "size_bytes": int(sizes["apparent_bytes"]),
        "allocated_bytes": int(sizes["allocated_bytes"]),
        "array_sha256": checksums,
        "source_lammpstrj": {
            "path": "trajectory.lammpstrj",
            "size_bytes": source_size,
            "sha256": source_sha256,
            "deleted": True,
            "deleted_at": _utc_now(),
        },
    }
    final = {
        **dynamics,
        "state": "complete",
        "design_name": DESIGN_NAME,
        "postprocessing_completed_at": _utc_now(),
        "trajectory_artifact": artifact,
        "first_passage": first_passage,
        "first_passage_progress_artifact": {
            "path": progress_path.name,
            "size_bytes": progress_path.stat().st_size,
            "sha256": _sha256_file(progress_path),
            "arrays": {
                name: {"shape": list(values.shape), "dtype": values.dtype.name}
                for name, values in progress.items()
            },
        },
        "initial_velocity_field": {
            "trajectory_array": "trajectory_binary_float32/velocities.npy",
            "frame_index": 0,
            "timestep": 0,
            "shape": [ATOM_COUNT, 3],
            "dtype": "float32",
            "sha256": _array_sha256(initial_velocities),
            "momentum_group_id": branch["momentum_group_id"],
        },
        "restart_sha256": _sha256_file(restart),
    }
    _write_json_atomic(outcome_path, final)
    _write_json_atomic(status_path, final)
    validate_complete_shooting_branch(root, manifest, branch, final)
    return final


def run_task(
    campaign_root: Path,
    task_index: int,
    *,
    launcher_override: str | None = None,
    mpi_ranks_override: int | None = None,
) -> dict[str, Any]:
    root = campaign_root.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    if not isinstance(branches, list) or task_index < 0 or task_index >= len(branches):
        raise IndexError(f"task_index={task_index} is outside this campaign.")
    branch = branches[task_index]
    branch_dir = root / str(branch["branch_dir"])
    outcome_path = branch_dir / "outcome.json"
    dynamics_path = branch_dir / "dynamics_outcome.json"
    if outcome_path.is_file():
        existing = _load_json(outcome_path)
        if existing.get("state") == "complete" and existing.get("design_name") == DESIGN_NAME:
            validate_complete_shooting_branch(root, manifest, branch, existing)
            print(f"Branch {branch['branch_id']} is already strict-complete.", flush=True)
            return existing
    try:
        if not dynamics_path.is_file():
            run_lammps_dynamics(
                root,
                task_index,
                launcher_override=launcher_override,
                mpi_ranks_override=mpi_ranks_override,
            )
            raw = _load_json(outcome_path)
            if raw.get("state") != "complete":
                raise RuntimeError(f"LAMMPS producer did not publish a complete raw outcome: {outcome_path}.")
            outcome_path.replace(dynamics_path)
        result = _postprocess_branch(root, manifest, branch)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    except BaseException as error:
        _write_json_atomic(
            branch_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "updated_at": _utc_now(),
                "branch_id": branch["branch_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
                "partial_artifacts_are_not_data": True,
            },
        )
        raise


def _complete_outcome_count(root: Path, manifest: dict[str, Any]) -> int:
    return sum(
        (root / str(branch["branch_dir"]) / "outcome.json").is_file()
        and _load_json(root / str(branch["branch_dir"]) / "outcome.json").get("state")
        == "complete"
        for branch in manifest["branches"]
    )


def run_local_subset(
    campaign_root: Path,
    start_index: int,
    stop_index: int,
    *,
    mpi_ranks: int,
) -> dict[str, Any]:
    """Run an inclusive branch range sequentially on a non-Slurm CPU host."""
    root = campaign_root.expanduser().resolve()
    if "SLURM_JOB_ID" in os.environ:
        raise RuntimeError(
            "run-local-subset refuses to run inside Slurm; "
            f"detected SLURM_JOB_ID={os.environ['SLURM_JOB_ID']!r}."
        )
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    if not isinstance(branches, list) or not branches:
        raise TypeError(f"{root / 'manifest.json'}: branches must be a non-empty list.")
    start = int(start_index)
    stop = int(stop_index)
    if start < 0 or stop < start or stop >= len(branches):
        raise IndexError(
            f"Requested inclusive branch range [{start}, {stop}] is outside "
            f"[0, {len(branches) - 1}]."
        )
    ranks = int(mpi_ranks)
    if ranks <= 0:
        raise ValueError(f"mpi_ranks must be positive, got {ranks}.")
    if (root / "slurm/active_submission.json").exists():
        raise RuntimeError(
            "This campaign has a Slurm active_submission.json; refusing a local launch "
            f"without proving the recorded jobs are inactive: {root / 'slurm/active_submission.json'}."
        )

    lock_path = root / "local_execution.lock"
    record_path = root / "local_execution.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another local driver holds the campaign execution lock: {lock_path}."
            ) from error
        started_at = _utc_now()
        base_record = {
            "schema_version": SCHEMA_VERSION,
            "execution_mode": "local_mpiexec",
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "started_at": started_at,
            "start_index": start,
            "stop_index": stop,
            "mpi_ranks": ranks,
        }
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(base_record, sort_keys=True) + "\n")
        lock.flush()
        os.fsync(lock.fileno())
        _write_json_atomic(record_path, {**base_record, "state": "running"})
        _write_json_atomic(
            root / "status.json",
            {
                **base_record,
                "state": "local_subset_running",
                "updated_at": started_at,
                "complete_outcome_count": _complete_outcome_count(root, manifest),
                "branch_count": len(branches),
            },
        )
        current = start
        try:
            for current in range(start, stop + 1):
                run_task(
                    root,
                    current,
                    launcher_override="local_mpiexec",
                    mpi_ranks_override=ranks,
                )
                progress = {
                    **base_record,
                    "state": "running",
                    "updated_at": _utc_now(),
                    "last_completed_index": current,
                    "complete_outcome_count": _complete_outcome_count(root, manifest),
                }
                _write_json_atomic(record_path, progress)
                _write_json_atomic(
                    root / "status.json",
                    {
                        **progress,
                        "state": "local_subset_running",
                        "branch_count": len(branches),
                    },
                )
            complete_count = _complete_outcome_count(root, manifest)
            final = {
                **base_record,
                "state": "complete",
                "completed_at": _utc_now(),
                "last_completed_index": stop,
                "complete_outcome_count": complete_count,
            }
            _write_json_atomic(record_path, final)
            _write_json_atomic(
                root / "status.json",
                {
                    **final,
                    "state": "complete" if complete_count == len(branches) else "partially_complete",
                    "branch_count": len(branches),
                },
            )
            return final
        except BaseException as error:
            failed = {
                **base_record,
                "state": "failed",
                "failed_at": _utc_now(),
                "failed_branch_index": current,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_artifacts_preserved": True,
            }
            _write_json_atomic(record_path, failed)
            _write_json_atomic(
                root / "status.json",
                {
                    **failed,
                    "state": "local_subset_failed",
                    "branch_count": len(branches),
                    "complete_outcome_count": _complete_outcome_count(root, manifest),
                },
            )
            raise


def _window_index(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    specification = manifest.get("window_index")
    if not isinstance(specification, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "campaign_root": str(root),
            "window_count": 0,
            "windows": [],
            "note": "This top-up preserves legacy parents; dense windows remain indexed in their source campaigns.",
        }
    stride_steps = int(round(float(specification["stride_ps"]) * 1000.0 / TIMESTEP_FS))
    latest_steps = int(round(float(specification["latest_start_ps"]) * 1000.0 / TIMESTEP_FS))
    horizon_steps = int(
        round(float(specification["maximum_prediction_horizon_ps"]) * 1000.0 / TIMESTEP_FS)
    )
    if stride_steps % SAMPLE_INTERVAL_STEPS != 0 or latest_steps % SAMPLE_INTERVAL_STEPS != 0:
        raise RuntimeError(f"Window stride/start is not aligned to stored frames: {specification}.")
    start_steps = tuple(range(0, latest_steps + 1, stride_steps))
    trajectories = [
        {**branch, "window_branch_uid": str(branch["branch_id"]), "campaign_root": str(root)}
        for branch in manifest["branches"]
    ]
    if bool(specification.get("include_authoritative_existing_snapshot_branches", False)):
        snapshot = _load_json(Path(str(manifest["authoritative_existing_snapshot"]["path"])))
        trajectories = [
            {
                **branch,
                "root_source_lineage_id": branch["source_run_id"],
                "basin_role": "legacy_pre_nucleation",
                "window_branch_uid": branch["branch_uid"],
            }
            for branch in snapshot["branches"]
        ] + trajectories
    parent_history_duration = {
        str(parent["parent_id"]): (
            abs(float(parent["history_artifact"]["relative_times_ps"][0]))
            if bool(parent.get("history_available", False))
            else 0.0
        )
        for parent in manifest["parents"]
    }
    windows: list[dict[str, Any]] = []
    for branch in trajectories:
        branch_uid = str(branch["window_branch_uid"])
        branch_ids = [
            f"window__{branch_uid}__start_{start_step:05d}" for start_step in start_steps
        ]
        history_duration = parent_history_duration.get(str(branch["parent_id"]), 0.0)
        for start_step, window_id in zip(start_steps, branch_ids):
            future_stop = min(RUN_STEPS, start_step + horizon_steps)
            windows.append(
                {
                    "window_id": window_id,
                    "root_source_run_id": branch["root_source_lineage_id"],
                    "ancestor_parent_id": branch["parent_id"],
                    "ancestor_branch_id": branch_uid,
                    "ancestor_campaign_root": branch["campaign_root"],
                    "absolute_lineage_time_ps": start_step * TIMESTEP_FS / 1000.0,
                    "relative_start_time_ps": start_step * TIMESTEP_FS / 1000.0,
                    "start_timestep": start_step,
                    "available_prehistory_interval_ps": [
                        -history_duration - start_step * TIMESTEP_FS / 1000.0,
                        0.0,
                    ],
                    "available_future_interval_ps": [
                        0.0,
                        (RUN_STEPS - start_step) * TIMESTEP_FS / 1000.0,
                    ],
                    "indexed_target_interval_ps": [
                        0.0,
                        (future_stop - start_step) * TIMESTEP_FS / 1000.0,
                    ],
                    "temperature_K": branch["temperature_K"],
                    "phase": branch["phase"],
                    "basin_role": branch["basin_role"],
                    "split": branch["source_split"],
                    "realized_future_count": 1,
                    "separately_generated_conditional_law_future_count": (
                        16 if start_step == 0 else 0
                    ),
                    "overlap_group_id": f"trajectory__{branch_uid}",
                    "overlapping_future_window_ids": [
                        other_id
                        for other_step, other_id in zip(start_steps, branch_ids)
                        if other_id != window_id
                        and max(start_step, other_step)
                        <= min(future_stop, min(RUN_STEPS, other_step + horizon_steps))
                    ],
                    "independence_unit": branch["root_source_lineage_id"],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "campaign_root": str(root),
        "stride_ps": float(specification["stride_ps"]),
        "maximum_prediction_horizon_ps": float(specification["maximum_prediction_horizon_ps"]),
        "window_count": len(windows),
        "correlated_not_independent": True,
        "windows": windows,
    }


def _conditional_law_index(
    root: Path, manifest: dict[str, Any], outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        by_parent[str(outcome["parent_id"])].append(outcome)
    parents: list[dict[str, Any]] = []
    for parent in manifest["parents"]:
        parent_id = str(parent["parent_id"])
        values = sorted(by_parent[parent_id], key=lambda item: int(item["canonical_shot_index"]))
        expected = 4 if manifest["design_kind"] == "legacy_40_parent_topup_from_12_to_16" else 16
        if len(values) != expected:
            raise RuntimeError(f"Parent {parent_id} has {len(values)} new futures, expected {expected}.")
        entry: dict[str, Any] = {
            "parent_id": parent_id,
            "parent_index": int(parent["parent_index"]),
            "root_source_lineage_id": parent["root_source_lineage_id"]
            if "root_source_lineage_id" in parent
            else parent["source_run_id"],
            "split": parent["source_split"],
            "source_temperature_K": parent["source_temperature_K"],
            "shooting_temperature_K": parent["shooting_temperature_K"],
            "basin_role": parent.get("basin_role", "legacy_pre_nucleation"),
            "new_validated_future_count": len(values),
            "new_branch_ids": [value["branch_id"] for value in values],
            "new_canonical_shot_indices": [int(value["canonical_shot_index"]) for value in values],
        }
        if expected == 16:
            entry["validated_future_count"] = 16
            entry["momentum_groups"] = {
                group_id: [value["branch_id"] for value in values if value["momentum_group_id"] == group_id]
                for group_id in sorted({str(value["momentum_group_id"]) for value in values})
            }
        else:
            snapshot_spec = manifest["authoritative_existing_snapshot"]
            snapshot = _load_json(Path(str(snapshot_spec["path"])))
            canonical_by_uid = snapshot_spec["existing_canonical_shot_index_by_branch_uid"]
            old_values = [
                value for value in snapshot["branches"] if value["parent_id"] == parent_id
            ]
            if len(old_values) != 12:
                raise RuntimeError(
                    f"Authoritative snapshot now has {len(old_values)} old futures for "
                    f"parent={parent_id}, expected 12."
                )
            combined = [
                {
                    "branch_uid": old["branch_uid"],
                    "campaign_root": old["campaign_root"],
                    "branch_dir": old["branch_dir"],
                    "canonical_shot_index": int(canonical_by_uid[old["branch_uid"]]),
                    "velocity_seed": int(old["velocity_seed"]),
                    "thermostat_seed": int(old["thermostat_seed"]),
                    "origin": "authoritative_existing_snapshot",
                }
                for old in old_values
            ] + [
                {
                    "branch_uid": f"topup__{value['branch_id']}",
                    "campaign_root": str(root),
                    "branch_dir": value["branch_dir"],
                    "canonical_shot_index": int(value["canonical_shot_index"]),
                    "velocity_seed": int(value["velocity_seed"]),
                    "thermostat_seed": int(value["thermostat_seed"]),
                    "origin": "four_branch_topup",
                }
                for value in values
            ]
            if sorted(value["canonical_shot_index"] for value in combined) != list(range(16)):
                raise RuntimeError(f"Merged canonical shot indices are not 0..15: parent={parent_id}.")
            pairs = {
                (value["velocity_seed"], value["thermostat_seed"]) for value in combined
            }
            if len(pairs) != 16:
                raise RuntimeError(f"Merged seed pairs are not unique: parent={parent_id}.")
            for old in old_values:
                old_outcome = _load_json(
                    Path(str(old["campaign_root"])) / str(old["branch_dir"]) / "outcome.json"
                )
                if old_outcome.get("state") != "complete":
                    raise RuntimeError(
                        f"Authoritative old outcome is no longer complete: branch={old['branch_uid']}."
                    )
            entry["validated_future_count"] = 16
            entry["merged_futures"] = sorted(
                combined, key=lambda value: value["canonical_shot_index"]
            )
        parents.append(entry)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "campaign_root": str(root),
        "parent_count": len(parents),
        "parents": parents,
    }


def _validate_nested_seed_design(manifest: dict[str, Any]) -> None:
    if manifest["design_kind"] != "single_parent_preproduction_smoke_8x2":
        return
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in manifest["branches"]:
        by_parent[str(branch["parent_id"])].append(branch)
    for parent_id, values in by_parent.items():
        if len(values) != 16:
            raise RuntimeError(f"Nested parent {parent_id} has {len(values)} branches, expected 16.")
        momentum_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for branch in values:
            momentum_groups[int(branch["momentum_index"])].append(branch)
        if set(momentum_groups) != set(range(8)):
            raise RuntimeError(f"Nested parent {parent_id} does not have momentum indices 0..7.")
        for momentum_index, replicas in momentum_groups.items():
            if len(replicas) != 2 or len({int(item["velocity_seed"]) for item in replicas}) != 1:
                raise RuntimeError(
                    f"Momentum group {parent_id}/{momentum_index} is not exactly two shared-velocity futures."
                )
            if {int(item["thermostat_replica_index"]) for item in replicas} != {0, 1}:
                raise RuntimeError(f"Momentum group {parent_id}/{momentum_index} lacks noise replicas 0/1.")


def summarize_campaign(campaign_root: Path) -> dict[str, Any]:
    root = campaign_root.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    manifest_sha_path = root / "manifest.sha256"
    recorded_sha = manifest_sha_path.read_text(encoding="ascii").split()[0]
    if _sha256_file(root / "manifest.json") != recorded_sha:
        raise RuntimeError(f"Immutable campaign manifest checksum changed: {root / 'manifest.json'}.")
    branches = manifest.get("branches")
    if not isinstance(branches, list) or len(branches) != int(manifest["counts"]["branches"]):
        raise RuntimeError(f"Manifest branch count is inconsistent: {root / 'manifest.json'}.")
    seed_pairs: set[tuple[int, int]] = set()
    outcomes: list[dict[str, Any]] = []
    for branch in branches:
        pair = (int(branch["velocity_seed"]), int(branch["thermostat_seed"]))
        if pair in seed_pairs:
            raise RuntimeError(f"Duplicate seed pair in campaign manifest: {pair}.")
        seed_pairs.add(pair)
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        if not outcome_path.is_file():
            raise RuntimeError(f"Campaign is incomplete; missing outcome: {outcome_path}.")
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete" or outcome.get("design_name") != DESIGN_NAME:
            raise RuntimeError(f"Branch is not predictive-dynamics complete: {outcome_path}.")
        validate_complete_shooting_branch(root, manifest, branch, outcome)
        binary = ShootingBinaryTrajectory.load(
            root / str(branch["branch_dir"]) / "trajectory_binary_float32"
        )
        binary.verify_checksums()
        if _sha256_file(root / str(branch["branch_dir"]) / "final.restart.bin") != str(
            outcome["restart_sha256"]
        ):
            raise RuntimeError(f"Final restart checksum changed: branch={branch['branch_id']}.")
        progress_path = root / str(branch["branch_dir"]) / str(
            outcome["first_passage_progress_artifact"]["path"]
        )
        if _sha256_file(progress_path) != str(outcome["first_passage_progress_artifact"]["sha256"]):
            raise RuntimeError(f"First-passage progress checksum changed: {progress_path}.")
        initial_sha = _array_sha256(np.asarray(binary.velocities[0], dtype=np.float32))
        if initial_sha != str(outcome["initial_velocity_field"]["sha256"]):
            raise RuntimeError(f"Initial velocity field checksum changed: branch={branch['branch_id']}.")
        outcomes.append(outcome)
    _validate_nested_seed_design(manifest)

    law_index = _conditional_law_index(root, manifest, outcomes)
    window_index = _window_index(root, manifest)
    _write_json_atomic(root / "conditional_law_parents.json", law_index)
    _write_json_atomic(root / "overlapping_temporal_windows.json", window_index)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "completed_at": _utc_now(),
        "campaign_root": str(root),
        "design_kind": manifest["design_kind"],
        "parent_count": len(manifest["parents"]),
        "branch_count": len(outcomes),
        "complete_outcome_count": len(outcomes),
        "unique_seed_pair_count": len(seed_pairs),
        "temperature_counts": dict(
            Counter(f"{float(value['temperature_K']):g}" for value in outcomes)
        ),
        "split_counts": dict(Counter(str(value["source_split"]) for value in outcomes)),
        "first_passage_counts": dict(
            Counter(str(value["first_passage"]["event"]) for value in outcomes)
        ),
        "trajectory_binary_bytes": int(
            sum(int(value["trajectory_artifact"]["size_bytes"]) for value in outcomes)
        ),
        "restart_bytes": int(sum(int(value["restart_size_bytes"]) for value in outcomes)),
        "conditional_law_index": "conditional_law_parents.json",
        "overlapping_window_index": "overlapping_temporal_windows.json",
        "overlapping_window_count": int(window_index["window_count"]),
    }
    if manifest["design_kind"] == "legacy_40_parent_topup_from_12_to_16":
        summary["merged_existing_and_topup_branch_count"] = 640
        summary["merged_futures_per_parent"] = 16
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(root / "status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _active_submission_conflicts(root: Path) -> list[str]:
    path = root / "slurm/active_submission.json"
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
    return sorted({line.strip() for line in queued.stdout.splitlines() if line.strip() != current})


def _submitted_job_count() -> int:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("USER is unset; cannot enforce the normal-QOS submitted-job limit.")
    queued = subprocess.run(
        ["squeue", "-r", "-h", "-u", user, "-o", "%i"],
        check=True,
        text=True,
        capture_output=True,
    )
    return sum(bool(line.strip()) for line in queued.stdout.splitlines())


def submit_next_wave(campaign_root: Path, start_index: int) -> dict[str, Any]:
    root = campaign_root.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branches = manifest.get("branches")
    if not isinstance(branches, list) or start_index < 0 or start_index >= len(branches):
        raise IndexError(f"start_index={start_index} is outside this campaign.")
    local_record_path = root / "local_execution.json"
    if local_record_path.is_file():
        local_record = _load_json(local_record_path)
        if local_record.get("state") == "running":
            raise RuntimeError(
                "Refusing a Slurm submission while a local subset is running: "
                f"{local_record_path}; record={local_record}."
            )
    conflicts = _active_submission_conflicts(root)
    if conflicts:
        raise RuntimeError(f"Refusing a duplicate campaign submission; active jobs={conflicts}.")
    configured_wave_size = int(manifest["execution"]["wave_size"])
    current_jobs = _submitted_job_count()
    maximum_jobs = int(manifest["execution"]["normal_qos_max_submitted_jobs"])
    available_slots = maximum_jobs - current_jobs
    tasks_in_wave = min(
        configured_wave_size,
        len(branches) - start_index,
        available_slots - 1,
    )
    if tasks_in_wave <= 0:
        raise RuntimeError(
            "Cannot submit a shooting array and its required successor without exceeding "
            f"the normal-QOS job count: current={current_jobs}, limit={maximum_jobs}, "
            f"available={available_slots}. Wait for another job to leave the queue."
        )
    stop_index = start_index + tasks_in_wave - 1
    array_spec = f"{start_index}-{stop_index}%{tasks_in_wave}"
    array_result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={array_spec}",
            str(root / "slurm/run_branch.sbatch"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    array_job_id = array_result.stdout.strip()
    if not array_job_id.isdigit():
        raise RuntimeError(f"Invalid Slurm array ID: {array_result.stdout!r}.")
    if stop_index + 1 < len(branches):
        successor_kind = "controller"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            f"--export=ALL,PREDICTIVE_START={stop_index + 1}",
            str(root / "slurm/submit_wave.sbatch"),
        ]
    else:
        successor_kind = "summary"
        successor_command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            str(root / "slurm/summarize.sbatch"),
        ]
    successor_result = subprocess.run(
        successor_command, check=True, text=True, capture_output=True
    )
    successor_job_id = successor_result.stdout.strip()
    if not successor_job_id.isdigit():
        raise RuntimeError(f"Invalid Slurm {successor_kind} ID: {successor_result.stdout!r}.")
    record = {
        "submitted_at": _utc_now(),
        "submitting_job_id": os.environ.get("SLURM_JOB_ID"),
        "array_spec": array_spec,
        "array_job_id": array_job_id,
        "successor_kind": successor_kind,
        "successor_job_id": successor_job_id,
        "jobs_in_queue_before_submission": current_jobs,
        "normal_qos_limit": maximum_jobs,
        "configured_wave_size": configured_wave_size,
        "submitted_wave_size": tasks_in_wave,
    }
    with (root / "slurm/submission_chain.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_json_atomic(root / "slurm/active_submission.json", record)
    _write_json_atomic(
        root / "status.json",
        {"schema_version": SCHEMA_VERSION, "state": "submitted", "updated_at": _utc_now(), **record},
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return record


def archive_partial(campaign_root: Path, branch_index: int, *, label: str) -> Path:
    root = campaign_root.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    branch = manifest["branches"][branch_index]
    branch_dir = root / str(branch["branch_dir"])
    outcome_path = branch_dir / "outcome.json"
    if outcome_path.is_file() and _load_json(outcome_path).get("state") == "complete":
        raise RuntimeError(f"Refusing to archive or alter a complete branch: {branch_dir}.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = branch_dir / f"interrupted_attempt_{label}_{timestamp}"
    archive.mkdir()
    names = (
        "trajectory.lammpstrj",
        "final.restart.bin",
        "lammps.log",
        "lammps.stdout.log",
        "dynamics_outcome.json",
        "first_passage_progress.npz",
        "status.json",
        "trajectory_binary_float32",
    )
    moved: list[str] = []
    for name in names:
        path = branch_dir / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
            moved.append(name)
    for building in sorted(branch_dir.glob(".trajectory_binary_float32.building-*")):
        shutil.move(str(building), str(archive / building.name))
        moved.append(building.name)
    if not moved:
        archive.rmdir()
        raise RuntimeError(f"Branch has no partial runtime artifacts to archive: {branch_dir}.")
    _write_json_atomic(
        archive / "archive.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "branch_id": branch["branch_id"],
            "moved_artifacts": moved,
            "reason": label,
            "not_training_data": True,
        },
    )
    return archive


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    topup = subparsers.add_parser("prepare-topup")
    topup.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    topup.add_argument("--campaign-root", type=Path, default=DEFAULT_TOPUP_ROOT)
    topup.add_argument("--wave-size", type=int, default=2)
    smoke = subparsers.add_parser("prepare-smoke")
    smoke.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    smoke.add_argument("--campaign-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    smoke.add_argument("--wave-size", type=int, default=2)
    run = subparsers.add_parser("run-task")
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    local = subparsers.add_parser("run-local-subset")
    local.add_argument("--campaign-root", type=Path, required=True)
    local.add_argument("--start-index", type=int, required=True)
    local.add_argument("--stop-index", type=int, required=True)
    local.add_argument("--mpi-ranks", type=int, required=True)
    submit = subparsers.add_parser("submit-next-wave")
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--start-index", type=int, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, required=True)
    archive = subparsers.add_parser("archive-partial")
    archive.add_argument("--campaign-root", type=Path, required=True)
    archive.add_argument("--branch-index", type=int, required=True)
    archive.add_argument("--label", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _arguments(argv)
    if args.command == "prepare-topup":
        result = prepare_topup(args.snapshot, args.campaign_root, wave_size=args.wave_size)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "prepare-smoke":
        result = prepare_smoke(args.snapshot, args.campaign_root, wave_size=args.wave_size)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "run-task":
        run_task(args.campaign_root, args.task_index)
    elif args.command == "run-local-subset":
        run_local_subset(
            args.campaign_root,
            args.start_index,
            args.stop_index,
            mpi_ranks=args.mpi_ranks,
        )
    elif args.command == "submit-next-wave":
        submit_next_wave(args.campaign_root, args.start_index)
    elif args.command == "summarize":
        summarize_campaign(args.campaign_root)
    elif args.command == "archive-partial":
        print(archive_partial(args.campaign_root, args.branch_index, label=args.label))
    else:
        raise AssertionError(f"Unhandled command: {args.command}.")


if __name__ == "__main__":
    main()
