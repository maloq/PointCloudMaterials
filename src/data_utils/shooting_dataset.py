from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree


_SHOOTING_COLUMNS = ("id", "type", "x", "y", "z", "vx", "vy", "vz")


@dataclass(frozen=True)
class ShootingCampaignSnapshot:
    root: Path
    campaign_roots: tuple[Path, ...]
    manifest: dict[str, Any]
    parents: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    complete_outcome_count: int
    ignored_incomplete_count: int

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return tuple(str(parent["parent_id"]) for parent in self.parents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_root": str(self.root),
            "campaign_roots": [str(root) for root in self.campaign_roots],
            "campaign_type": self.manifest["campaign_type"],
            "selected_parent_count": len(self.parents),
            "selected_branch_count": len(self.branches),
            "complete_outcome_count_at_snapshot": self.complete_outcome_count,
            "ignored_incomplete_count_at_snapshot": self.ignored_incomplete_count,
            "parents": list(self.parents),
            "branches": list(self.branches),
        }


@dataclass(frozen=True)
class ShootingPositionFrame:
    timestep: int
    atom_ids: np.ndarray
    atom_types: np.ndarray
    positions: np.ndarray
    box_low: np.ndarray
    box_high: np.ndarray

    @property
    def box_lengths(self) -> np.ndarray:
        return self.box_high - self.box_low


@dataclass(frozen=True)
class ShootingFrame(ShootingPositionFrame):
    velocities: np.ndarray


@dataclass(frozen=True)
class PeriodicEnvironmentBatch:
    points: torch.Tensor
    context_points: torch.Tensor | None
    center_positions: np.ndarray
    context_center_offsets: np.ndarray | None
    context_center_atom_ids: np.ndarray | None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required shooting metadata file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def resolve_shooting_trajectory_path(
    campaign_root: str | Path, branch: dict[str, Any]
) -> Path:
    """Resolve the current trajectory artifact for a complete shooting branch."""

    root = Path(campaign_root).expanduser().resolve()
    branch_dir = root / str(branch["branch_dir"])
    outcome = _load_json(branch_dir / "outcome.json")
    artifact = outcome.get("trajectory_artifact")
    if artifact is None:
        raise RuntimeError(
            "Complete shooting branch has not been migrated to the required float32 "
            f"binary format: branch={branch['branch_id']}, outcome={branch_dir / 'outcome.json'}. "
            "Run scripts/migrate_lammps_shooting_float32.py for this campaign."
        )
    if not isinstance(artifact, dict):
        raise TypeError(
            f"trajectory_artifact must be a JSON object in {branch_dir / 'outcome.json'}, "
            f"got {type(artifact).__name__}."
        )
    from src.data_utils.shooting_binary import FORMAT_NAME

    if artifact.get("format") != FORMAT_NAME:
        raise ValueError(
            f"Unsupported trajectory artifact format for branch={branch['branch_id']}: "
            f"{artifact.get('format')!r}."
        )
    storage_dtype = str(artifact.get("storage_dtype", "float32"))
    expected_directory_name = f"trajectory_binary_{storage_dtype}"
    artifact_path = Path(str(artifact.get("path")))
    resolved_path = (
        artifact_path.resolve()
        if artifact_path.is_absolute()
        else (branch_dir / artifact_path).resolve()
    )
    if resolved_path.name != expected_directory_name:
        raise ValueError(
            f"Unexpected repository shooting-binary path for branch={branch['branch_id']}: "
            f"path={resolved_path}, storage_dtype={storage_dtype!r}, "
            f"expected_directory_name={expected_directory_name!r}."
        )
    return resolved_path


def validate_complete_shooting_branch(
    root: Path,
    manifest: dict[str, Any],
    branch: dict[str, Any],
    outcome: dict[str, Any],
) -> None:
    branch_id = str(branch["branch_id"])
    outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
    if outcome.get("state") != "complete":
        raise RuntimeError(
            f"Internal error: non-complete outcome reached strict validation: {outcome_path}."
        )
    for key in (
        "branch_index",
        "branch_id",
        "parent_index",
        "parent_id",
        "source_run_id",
        "source_split",
        "source_velocity_seed",
        "temperature_K",
        "phase",
        "shot_index",
        "velocity_seed",
        "thermostat_seed",
    ):
        if outcome.get(key) != branch[key]:
            raise RuntimeError(
                f"Completed outcome disagrees with manifest for branch={branch_id}, key={key!r}: "
                f"manifest={branch[key]!r}, outcome={outcome.get(key)!r}, path={outcome_path}."
            )

    protocol = manifest["protocol"]
    expected_last_timestep = int(protocol["run_steps"])
    if (
        int(outcome["frame_count"]) != int(protocol["expected_frame_count"])
        or int(outcome["first_timestep"]) != 0
        or int(outcome["last_timestep"]) != expected_last_timestep
    ):
        raise RuntimeError(
            f"Completed branch has invalid temporal contract: branch={branch_id}, "
            f"frames={outcome['frame_count']}, first={outcome['first_timestep']}, "
            f"last={outcome['last_timestep']}."
        )

    branch_dir = root / str(branch["branch_dir"])
    restart = branch_dir / "final.restart.bin"
    artifact = outcome.get("trajectory_artifact")
    if artifact is None:
        trajectory = branch_dir / "trajectory.lammpstrj"
        if not trajectory.is_file() or trajectory.stat().st_size <= 0:
            raise RuntimeError(
                f"Completed branch {branch_id} is missing a nonempty artifact: {trajectory}."
            )
        if int(trajectory.stat().st_size) != int(outcome["trajectory_size_bytes"]):
            raise RuntimeError(
                f"Completed branch artifact size changed after validation: branch={branch_id}, "
                f"path={trajectory}, outcome_size={outcome['trajectory_size_bytes']}, "
                f"observed_size={trajectory.stat().st_size}."
            )
    else:
        trajectory = resolve_shooting_trajectory_path(root, branch)
        from src.data_utils.shooting_binary import (
            ShootingBinaryTrajectory,
            binary_directory_sizes,
        )

        binary = ShootingBinaryTrajectory.load(trajectory)
        expected_timesteps = np.arange(
            0,
            int(protocol["run_steps"]) + 1,
            int(protocol["sample_interval_steps"]),
            dtype=np.int64,
        )
        expected_storage_dtype = np.dtype(str(artifact.get("storage_dtype", "float32")))
        if (
            binary.storage_dtype != expected_storage_dtype
            or binary.atom_count != int(manifest["atom_count"])
            or binary.frame_count != int(protocol["expected_frame_count"])
            or not np.array_equal(binary.timesteps, expected_timesteps)
        ):
            raise RuntimeError(
                f"Completed binary branch violates the campaign trajectory contract: "
                f"branch={branch_id}, dtype={binary.storage_dtype.name}, "
                f"expected_dtype={expected_storage_dtype.name}, "
                f"atoms={binary.atom_count}, frames={binary.frame_count}, "
                f"first={int(binary.timesteps[0])}, last={int(binary.timesteps[-1])}."
            )
        sizes = binary_directory_sizes(trajectory)
        if sizes["apparent_bytes"] != int(artifact["size_bytes"]):
            raise RuntimeError(
                f"Completed binary trajectory size changed after migration: branch={branch_id}, "
                f"outcome_size={artifact['size_bytes']}, "
                f"observed_size={sizes['apparent_bytes']}, path={trajectory}."
            )

    if not restart.is_file() or restart.stat().st_size <= 0:
        raise RuntimeError(
            f"Completed branch {branch_id} is missing a nonempty artifact: {restart}."
        )
    if int(restart.stat().st_size) != int(outcome["restart_size_bytes"]):
        raise RuntimeError(
            f"Completed branch artifact size changed after validation: branch={branch_id}, "
            f"path={restart}, outcome_size={outcome['restart_size_bytes']}, "
            f"observed_size={restart.stat().st_size}."
        )


def load_shooting_campaign_snapshot(
    campaign_root: str | Path,
    *,
    temperatures_K: Sequence[float],
    minimum_complete_branches_per_parent: int,
) -> ShootingCampaignSnapshot:
    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("campaign_type") != "position_conditioned_langevin_nvt_shooting":
        raise ValueError(
            f"Unsupported campaign_type={manifest.get('campaign_type')!r} in "
            f"{root / 'manifest.json'}."
        )
    protocol = manifest["protocol"]
    if tuple(protocol["dump_columns"]) != _SHOOTING_COLUMNS:
        raise ValueError(
            f"Shooting training requires dump columns {_SHOOTING_COLUMNS}, got "
            f"{tuple(protocol['dump_columns'])}."
        )
    minimum = int(minimum_complete_branches_per_parent)
    intended_per_parent = int(manifest["counts"]["branches"]) // int(
        manifest["counts"]["parents"]
    )
    if minimum <= 0 or minimum > intended_per_parent:
        raise ValueError(
            "minimum_complete_branches_per_parent must be within the manifest ensemble "
            f"size [1, {intended_per_parent}], got {minimum}."
        )
    selected_temperatures = {float(value) for value in temperatures_K}
    if not selected_temperatures:
        raise ValueError("shooting.data.temperatures_K must be non-empty.")

    manifest_parents = {
        str(parent["parent_id"]): parent for parent in manifest["parents"]
    }
    complete_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    complete_count = 0
    ignored_count = 0
    for branch in manifest["branches"]:
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        if not outcome_path.is_file():
            ignored_count += 1
            continue
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            ignored_count += 1
            continue
        validate_complete_shooting_branch(root, manifest, branch, outcome)
        complete_count += 1
        if float(branch["temperature_K"]) in selected_temperatures:
            complete_by_parent[str(branch["parent_id"])].append(branch)

    selected_parents: list[dict[str, Any]] = []
    selected_branches: list[dict[str, Any]] = []
    for parent_id, branches in complete_by_parent.items():
        if len(branches) < minimum:
            continue
        parent = manifest_parents[parent_id]
        ordered = sorted(branches, key=lambda item: int(item["shot_index"]))
        selected_parents.append(parent)
        selected_branches.extend(ordered)
    selected_parents.sort(key=lambda item: int(item["parent_index"]))
    selected_parent_ids = {str(parent["parent_id"]) for parent in selected_parents}
    selected_branches = [
        branch
        for branch in sorted(selected_branches, key=lambda item: int(item["branch_index"]))
        if str(branch["parent_id"]) in selected_parent_ids
    ]
    if not selected_parents:
        counts = Counter(
            (float(branch["temperature_K"]), str(branch["source_split"]))
            for branches in complete_by_parent.values()
            for branch in branches
        )
        raise RuntimeError(
            "No shooting parent satisfies the requested complete-branch threshold. "
            f"temperatures={sorted(selected_temperatures)}, minimum={minimum}, "
            f"available_complete_branch_counts={dict(counts)}."
        )
    splits = {str(parent["source_split"]) for parent in selected_parents}
    if splits != {"train", "validation"}:
        raise RuntimeError(
            "Shooting training requires selected complete parents in both source splits; "
            f"got splits={sorted(splits)}, temperatures={sorted(selected_temperatures)}."
        )
    return ShootingCampaignSnapshot(
        root=root,
        campaign_roots=(root,),
        manifest=manifest,
        parents=tuple(selected_parents),
        branches=tuple(selected_branches),
        complete_outcome_count=complete_count,
        ignored_incomplete_count=ignored_count,
    )


def load_shooting_campaigns_snapshot(
    campaign_roots: Sequence[str | Path],
    *,
    temperatures_K: Sequence[float],
    minimum_complete_branches_per_parent: int,
) -> ShootingCampaignSnapshot:
    roots = tuple(Path(value).expanduser().resolve() for value in campaign_roots)
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError(
            "data.campaign_roots must contain at least two distinct shooting roots; "
            f"got {[str(root) for root in roots]}."
        )
    component_snapshots = tuple(
        load_shooting_campaign_snapshot(
            root,
            temperatures_K=temperatures_K,
            minimum_complete_branches_per_parent=1,
        )
        for root in roots
    )
    reference = component_snapshots[0]
    parent_comparison_keys = (
        "parent_id",
        "source_run_id",
        "source_split",
        "source_velocity_seed",
        "temperature_K",
        "phase",
        "source_frame_index",
        "source_frame_step",
        "source_frame_time_ps",
        "data_sha256",
    )
    reference_parents = {
        str(parent["parent_id"]): parent for parent in reference.manifest["parents"]
    }
    for component in component_snapshots[1:]:
        if (
            int(component.manifest["atom_count"])
            != int(reference.manifest["atom_count"])
            or component.manifest["protocol"] != reference.manifest["protocol"]
        ):
            raise RuntimeError(
                "Shooting campaigns cannot be merged because their atom count or "
                f"protocol differs: reference={reference.root}, other={component.root}."
            )
        component_parents = {
            str(parent["parent_id"]): parent
            for parent in component.manifest["parents"]
        }
        if component_parents.keys() != reference_parents.keys():
            raise RuntimeError(
                "Shooting campaigns cannot be merged because their parent catalogs "
                f"differ: reference={reference.root}, other={component.root}."
            )
        for parent_id, reference_parent in reference_parents.items():
            other_parent = component_parents[parent_id]
            for key in parent_comparison_keys:
                if reference_parent.get(key) != other_parent.get(key):
                    raise RuntimeError(
                        "Shooting campaigns disagree on parent provenance: "
                        f"parent={parent_id}, key={key!r}, "
                        f"reference={reference_parent.get(key)!r}, "
                        f"other={other_parent.get(key)!r}, root={component.root}."
                    )

    complete_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_keys: set[tuple[str, int, int]] = set()
    for campaign_index, component in enumerate(component_snapshots):
        for branch in component.branches:
            seed_key = (
                str(branch["parent_id"]),
                int(branch["velocity_seed"]),
                int(branch["thermostat_seed"]),
            )
            if seed_key in seed_keys:
                raise RuntimeError(
                    "Shooting campaigns contain a duplicate parent/velocity/thermostat "
                    f"seed tuple: {seed_key}."
                )
            seed_keys.add(seed_key)
            merged_branch = dict(branch)
            merged_branch["campaign_root"] = str(component.root)
            merged_branch["campaign_index"] = campaign_index
            merged_branch["branch_uid"] = (
                f"campaign_{campaign_index:02d}__{branch['branch_id']}"
            )
            complete_by_parent[str(branch["parent_id"])].append(merged_branch)

    minimum = int(minimum_complete_branches_per_parent)
    if minimum <= 0:
        raise ValueError(
            f"minimum_complete_branches_per_parent must be positive, got {minimum}."
        )
    selected_parent_ids = {
        parent_id
        for parent_id, branches in complete_by_parent.items()
        if len(branches) >= minimum
    }
    selected_parents = tuple(
        parent
        for parent in reference.manifest["parents"]
        if str(parent["parent_id"]) in selected_parent_ids
        and float(parent["temperature_K"])
        in {float(value) for value in temperatures_K}
    )
    selected_branches = tuple(
        sorted(
            (
                branch
                for parent_id, branches in complete_by_parent.items()
                if parent_id in selected_parent_ids
                for branch in branches
            ),
            key=lambda item: (
                int(item["parent_index"]),
                int(item["campaign_index"]),
                int(item["branch_index"]),
            ),
        )
    )
    if not selected_parents:
        branch_counts = {
            parent_id: len(branches) for parent_id, branches in complete_by_parent.items()
        }
        raise RuntimeError(
            "No merged shooting parent satisfies the requested total complete-branch "
            f"threshold={minimum}; counts={branch_counts}."
        )
    splits = {str(parent["source_split"]) for parent in selected_parents}
    if splits != {"train", "validation"}:
        raise RuntimeError(
            "Merged shooting training requires parents in both source splits; "
            f"got {sorted(splits)}."
        )
    return ShootingCampaignSnapshot(
        root=reference.root,
        campaign_roots=roots,
        manifest=reference.manifest,
        parents=selected_parents,
        branches=selected_branches,
        complete_outcome_count=sum(
            snapshot.complete_outcome_count for snapshot in component_snapshots
        ),
        ignored_incomplete_count=sum(
            snapshot.ignored_incomplete_count for snapshot in component_snapshots
        ),
    )


def shooting_snapshot_sha256(snapshot: ShootingCampaignSnapshot) -> str:
    payload = json.dumps(
        {
            "campaign_root": str(snapshot.root),
            "campaign_roots": [str(root) for root in snapshot.campaign_roots],
            "campaign_type": snapshot.manifest["campaign_type"],
            "parents": list(snapshot.parents),
            "branches": list(snapshot.branches),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_periodic_environment_batch(
    frame: ShootingPositionFrame,
    *,
    center_atom_ids: np.ndarray,
    num_points: int,
    radius: float,
    spatial_context_center_count: int,
) -> PeriodicEnvironmentBatch:
    centers_requested = np.asarray(center_atom_ids, dtype=np.int64)
    positions = np.searchsorted(frame.atom_ids, centers_requested)
    if not np.array_equal(frame.atom_ids[positions], centers_requested):
        raise RuntimeError(
            f"Requested center atom IDs are absent at timestep={frame.timestep}."
        )
    frame_points = np.asarray(frame.positions, dtype=np.float32)
    box_lengths = np.asarray(frame.box_lengths, dtype=np.float32)
    tree = cKDTree(frame_points, boxsize=box_lengths, balanced_tree=False)
    centers = np.asarray(frame_points[positions], dtype=np.float32)
    _, neighbor_indices = tree.query(centers, k=int(num_points))
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    local = frame_points[neighbor_indices] - centers[:, None, :]
    local -= box_lengths[None, None, :] * np.round(
        local / box_lengths[None, None, :]
    )
    local = (local / float(radius)).astype(np.float32, copy=False)

    context_points: torch.Tensor | None = None
    context_center_offsets: np.ndarray | None = None
    context_center_atom_ids: np.ndarray | None = None
    context_count = int(spatial_context_center_count)
    if context_count > 0:
        available = neighbor_indices != positions[:, None]
        offsets64 = local.astype(np.float64) * float(radius)
        minimum_distance_squared = np.sum(offsets64**2, axis=2)
        minimum_distance_squared[~available] = -np.inf
        batch_rows = np.arange(centers.shape[0], dtype=np.int64)
        selected_slots = np.empty((centers.shape[0], context_count), dtype=np.int64)
        for context_slot in range(context_count):
            chosen = np.argmax(minimum_distance_squared, axis=1).astype(np.int64)
            if np.any(~np.isfinite(minimum_distance_squared[batch_rows, chosen])):
                raise RuntimeError(
                    f"Spatial-context FPS exhausted candidates at slot={context_slot}."
                )
            selected_slots[:, context_slot] = chosen
            selected_offset = offsets64[batch_rows, chosen]
            delta = offsets64 - selected_offset[:, None, :]
            delta -= box_lengths[None, None, :] * np.round(
                delta / box_lengths[None, None, :]
            )
            minimum_distance_squared = np.minimum(
                minimum_distance_squared, np.sum(delta**2, axis=2)
            )
            minimum_distance_squared[~available] = -np.inf
            minimum_distance_squared[batch_rows, chosen] = -np.inf
        context_indices = neighbor_indices[batch_rows[:, None], selected_slots]
        context_center_offsets = offsets64[
            batch_rows[:, None], selected_slots
        ].astype(np.float32, copy=False)
        context_center_atom_ids = frame.atom_ids[context_indices].astype(
            np.int64, copy=False
        )
        context_centers = frame_points[context_indices.reshape(-1)]
        _, context_neighbors = tree.query(context_centers, k=int(num_points))
        context_neighbors = np.asarray(context_neighbors, dtype=np.int64)
        context_local = frame_points[context_neighbors] - context_centers[:, None, :]
        context_local -= box_lengths[None, None, :] * np.round(
            context_local / box_lengths[None, None, :]
        )
        context_local = (context_local / float(radius)).astype(np.float32, copy=False)
        context_points = torch.from_numpy(
            context_local.reshape(
                centers.shape[0], context_count, int(num_points), 3
            )
        )

    return PeriodicEnvironmentBatch(
        points=torch.from_numpy(local),
        context_points=context_points,
        center_positions=centers + frame.box_low[None, :],
        context_center_offsets=context_center_offsets,
        context_center_atom_ids=context_center_atom_ids,
    )


__all__ = [
    "PeriodicEnvironmentBatch",
    "ShootingCampaignSnapshot",
    "ShootingFrame",
    "ShootingPositionFrame",
    "build_periodic_environment_batch",
    "load_shooting_campaign_snapshot",
    "load_shooting_campaigns_snapshot",
    "resolve_shooting_trajectory_path",
    "shooting_snapshot_sha256",
    "validate_complete_shooting_branch",
]
