"""First-passage committor baselines for the nested Al shooting campaign.

The nested campaign differs from the older fixed-horizon shooting data in two
important ways: trajectories stop at a persistent basin hit, and every parent
has a two-by-two momentum/thermostat hierarchy.  This module keeps those
semantics explicit.  It never interprets a partial branch as data and never
turns the four children of one parent into four independent structures.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data_utils.shooting_binary import FORMAT_NAME, ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import (
    ShootingPositionFrame,
    build_periodic_environment_batch,
)
from src.temporal_vamp.embeddings import FrozenEncoder
from src.temporal_vamp.shooting_dynamics import (
    VELOCITY_TOKEN_FEATURE_NAMES,
    invariant_velocity_token_features,
)


CAMPAIGN_TYPE = "transition_balanced_nested_langevin_nvt_shooting_pilot"
REGION_NAMES = ("nucleus", "interface", "background")
SPLIT_NAMES = ("optimization", "model_selection", "final_validation")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required nested-shooting JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NestedShootingSnapshot:
    root: Path
    manifest: dict[str, Any]
    parents: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    outcomes: tuple[dict[str, Any], ...]
    summary: dict[str, Any] | None

    @property
    def parent_index_to_position(self) -> dict[int, int]:
        return {
            int(parent["parent_index"]): position
            for position, parent in enumerate(self.parents)
        }


def _nested_binary_path(
    root: Path, branch: dict[str, Any], outcome: dict[str, Any]
) -> Path:
    artifact = outcome.get("trajectory_artifact")
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT_NAME:
        raise RuntimeError(
            "A complete nested branch must own a float32 shooting trajectory: "
            f"branch={branch['branch_id']}, artifact={artifact!r}."
        )
    expected = (root / str(branch["branch_dir"]) / "trajectory_binary_float32").resolve()
    observed = Path(str(artifact.get("path"))).expanduser().resolve()
    if observed != expected:
        raise RuntimeError(
            "Nested trajectory artifact path disagrees with the repository layout: "
            f"branch={branch['branch_id']}, expected={expected}, observed={observed}."
        )
    return expected


def load_nested_shooting_snapshot(
    campaign_root: str | Path,
    *,
    require_complete_campaign: bool,
) -> NestedShootingSnapshot:
    """Load only direct, authoritative complete outcomes from one campaign."""

    root = Path(campaign_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("campaign_type") != CAMPAIGN_TYPE:
        raise ValueError(
            f"Unsupported nested campaign_type={manifest.get('campaign_type')!r}: "
            f"{root / 'manifest.json'}."
        )
    parents = tuple(manifest["parents"])
    branches = tuple(manifest["branches"])
    counts = manifest["counts"]
    if len(parents) != int(counts["parents"]) or len(branches) != int(counts["branches"]):
        raise RuntimeError(
            "Nested manifest counts disagree with its catalogs: "
            f"parents={len(parents)}/{counts['parents']}, "
            f"branches={len(branches)}/{counts['branches']}."
        )
    splits_by_source: dict[str, set[str]] = defaultdict(set)
    for parent in parents:
        splits_by_source[str(parent["source_run_id"])].add(
            str(parent["source_split"])
        )
    leaked_sources = {
        source: sorted(splits)
        for source, splits in splits_by_source.items()
        if len(splits) != 1
    }
    if leaked_sources:
        raise RuntimeError(
            "Nested source trajectories cross data splits: "
            f"{leaked_sources}."
        )
    observed_splits = {
        next(iter(splits)) for splits in splits_by_source.values()
    }
    if observed_splits != set(SPLIT_NAMES):
        raise RuntimeError(
            "Nested campaign must expose optimization, model-selection, and final "
            f"source splits; observed={sorted(observed_splits)}."
        )
    outcomes: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for branch in branches:
        outcome_path = root / str(branch["branch_dir"]) / "outcome.json"
        if not outcome_path.is_file():
            incomplete.append(str(branch["branch_id"]))
            continue
        outcome = _load_json(outcome_path)
        if outcome.get("state") != "complete":
            incomplete.append(str(branch["branch_id"]))
            continue
        for key in (
            "branch_index",
            "branch_id",
            "parent_index",
            "parent_id",
            "source_run_id",
            "source_split",
            "temperature_K",
            "basin_role",
            "momentum_index",
            "momentum_seed",
            "thermostat_index",
            "thermostat_seed",
        ):
            if outcome.get(key) != branch[key]:
                raise RuntimeError(
                    "Nested outcome disagrees with the immutable manifest: "
                    f"branch={branch['branch_id']}, key={key}, "
                    f"manifest={branch[key]!r}, outcome={outcome.get(key)!r}."
                )
        _nested_binary_path(root, branch, outcome)
        outcomes.append(outcome)
    summary_path = root / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    if require_complete_campaign:
        if incomplete:
            raise RuntimeError(
                "Nested campaign is not complete; refusing to fit on a moving target: "
                f"complete={len(outcomes)}, intended={len(branches)}, "
                f"first_missing={incomplete[:10]}."
            )
        if summary is None or summary.get("state") != "complete":
            raise RuntimeError(
                f"Strict nested summary is missing or incomplete: {summary_path}."
            )
        if int(summary["branch_count"]) != len(branches):
            raise RuntimeError(
                f"Strict nested summary branch_count={summary['branch_count']} but "
                f"manifest has {len(branches)} branches."
            )
    return NestedShootingSnapshot(
        root=root,
        manifest=manifest,
        parents=parents,
        branches=branches,
        outcomes=tuple(sorted(outcomes, key=lambda value: int(value["branch_index"]))),
        summary=summary,
    )


@dataclass(frozen=True)
class NestedFeatureCache:
    path: Path
    manifest: dict[str, Any]
    parent_scalar_features: np.ndarray
    parent_force_features: np.ndarray
    token_z: np.ndarray
    token_parent_position: np.ndarray
    token_region: np.ndarray
    token_atom_id: np.ndarray
    velocity_token_features: np.ndarray
    velocity_parent_position: np.ndarray
    velocity_momentum_index: np.ndarray
    velocity_region: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "NestedFeatureCache":
        root = Path(path).expanduser().resolve()
        manifest = _load_json(root / "manifest.json")
        names = (
            "parent_scalar_features",
            "parent_force_features",
            "token_z",
            "token_parent_position",
            "token_region",
            "token_atom_id",
            "velocity_token_features",
            "velocity_parent_position",
            "velocity_momentum_index",
            "velocity_region",
        )
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in names
        }
        for name, values in arrays.items():
            expected = tuple(int(value) for value in manifest["array_shapes"][name])
            if values.shape != expected:
                raise RuntimeError(
                    f"Nested feature-cache shape changed for {name}: "
                    f"expected={expected}, observed={values.shape}, path={root}."
                )
        return cls(path=root, manifest=manifest, **arrays)


def _periodic_farthest_sample(
    atom_ids: np.ndarray,
    positions: np.ndarray,
    box_lengths: np.ndarray,
    count: int,
) -> np.ndarray:
    ids = np.asarray(atom_ids, dtype=np.int64)
    points = np.asarray(positions, dtype=np.float64)
    if ids.size <= int(count):
        return np.sort(ids)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    points = points[order]
    selected = np.empty(int(count), dtype=np.int64)
    selected[0] = 0
    delta = points - points[0]
    delta -= box_lengths[None, :] * np.round(delta / box_lengths[None, :])
    minimum_squared = np.sum(delta**2, axis=1)
    minimum_squared[0] = -np.inf
    for index in range(1, int(count)):
        chosen = int(np.argmax(minimum_squared))
        selected[index] = chosen
        delta = points - points[chosen]
        delta -= box_lengths[None, :] * np.round(delta / box_lengths[None, :])
        minimum_squared = np.minimum(minimum_squared, np.sum(delta**2, axis=1))
        minimum_squared[selected[: index + 1]] = -np.inf
    return np.sort(ids[selected])


def select_parent_region_atom_ids(
    *,
    all_atom_ids: np.ndarray,
    positions: np.ndarray,
    box_lengths: np.ndarray,
    cluster_labels: np.ndarray,
    nucleus_center_count: int,
    interface_center_count: int,
    background_center_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select fixed physical regions without using any future outcome."""

    atom_ids = np.asarray(all_atom_ids, dtype=np.int64)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    nucleus_all = atom_ids[labels == 1]
    if nucleus_all.size == 0:
        raise RuntimeError("PTM cluster analysis returned no largest-cluster atoms.")
    nucleus_indices = nucleus_all - 1
    nucleus = _periodic_farthest_sample(
        nucleus_all,
        positions[nucleus_indices],
        box_lengths,
        int(nucleus_center_count),
    )

    nucleus_tree = cKDTree(
        np.asarray(positions[nucleus_indices], dtype=np.float64),
        boxsize=np.asarray(box_lengths, dtype=np.float64),
        balanced_tree=False,
    )
    nearest_distance, _ = nucleus_tree.query(
        np.asarray(positions, dtype=np.float64), k=1
    )
    is_nucleus = labels == 1
    candidates = np.flatnonzero(~is_nucleus)
    interface_order = candidates[
        np.argsort(nearest_distance[candidates], kind="stable")
    ]
    if interface_order.size < int(interface_center_count):
        raise RuntimeError(
            "Not enough non-nucleus atoms for the requested interface centers: "
            f"available={interface_order.size}, requested={interface_center_count}."
        )
    interface_indices = interface_order[: int(interface_center_count)]
    interface = np.sort(atom_ids[interface_indices])

    excluded = np.zeros(atom_ids.size, dtype=bool)
    excluded[nucleus - 1] = True
    excluded[interface_indices] = True
    background_candidates = atom_ids[~excluded]
    if background_candidates.size < int(background_center_count):
        raise RuntimeError(
            "Not enough bulk atoms after excluding the nucleus and interface: "
            f"available={background_candidates.size}, requested={background_center_count}."
        )
    rng = np.random.default_rng(int(seed))
    background = np.sort(
        rng.choice(
            background_candidates,
            size=int(background_center_count),
            replace=False,
        )
    ).astype(np.int64, copy=False)
    return nucleus, interface, background


def _parent_ptm_clusters(
    path: Path, *, ptm_rmsd_cutoff: float, cluster_cutoff_A: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    try:
        from ovito.io import import_file
        from ovito.modifiers import (
            ClusterAnalysisModifier,
            PolyhedralTemplateMatchingModifier,
        )
    except ImportError as exc:
        raise ImportError(
            "Nested committor feature extraction requires OVITO in the pointnet environment."
        ) from exc

    data = import_file(str(path), sort_particles=True).compute()
    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = float(ptm_rmsd_cutoff)
    data.apply(ptm)
    structure_types = np.array(
        data.particles["Structure Type"][...], dtype=np.int32
    )
    crystalline = np.isin(structure_types, np.asarray([1, 2, 3], dtype=np.int32))
    data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
    data.apply(
        ClusterAnalysisModifier(
            cutoff=float(cluster_cutoff_A), only_selected=True, sort_by_size=True
        )
    )
    atom_ids = np.array(data.particles["Particle Identifier"][...], dtype=np.int64)
    cluster = np.array(data.particles["Cluster"][...], dtype=np.int64)
    return (
        atom_ids,
        cluster,
        float(crystalline.mean()),
        int(data.attributes["ClusterAnalysis.largest_size"]),
    )


def _representative_complete_branches(
    snapshot: NestedShootingSnapshot,
) -> dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]]:
    branches = {int(value["branch_index"]): value for value in snapshot.branches}
    selected: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for outcome in snapshot.outcomes:
        key = (int(outcome["parent_index"]), int(outcome["momentum_index"]))
        branch = branches[int(outcome["branch_index"])]
        current = selected.get(key)
        if current is None or int(branch["thermostat_index"]) < int(
            current[0]["thermostat_index"]
        ):
            selected[key] = (branch, outcome)
    return selected


def _trajectory_for(
    snapshot: NestedShootingSnapshot,
    pair: tuple[dict[str, Any], dict[str, Any]],
) -> ShootingBinaryTrajectory:
    branch, outcome = pair
    trajectory = ShootingBinaryTrajectory.load(
        _nested_binary_path(snapshot.root, branch, outcome)
    )
    if trajectory.storage_dtype != np.dtype("float32"):
        raise RuntimeError(
            f"Nested training requires float32 storage, got {trajectory.storage_dtype.name}: "
            f"{trajectory.root}."
        )
    return trajectory


@torch.inference_mode()
def extract_nested_feature_cache(
    snapshot: NestedShootingSnapshot,
    *,
    encoder: FrozenEncoder,
    cache_path: str | Path,
    num_points: int,
    radius: float,
    nucleus_center_count: int,
    interface_center_count: int,
    background_center_count: int,
    point_cloud_batch_size: int,
    selection_seed: int,
    force_recompute: bool,
) -> NestedFeatureCache:
    """Encode nucleus/interface/bulk tokens and phase-space descriptors."""

    target = Path(cache_path).expanduser().resolve()
    checkpoint_stat = encoder.checkpoint_path.stat()
    spec = {
        "version": 1,
        "campaign_root": str(snapshot.root),
        "manifest_sha256": _sha256_json(snapshot.manifest),
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "representation_source": encoder.representation_source,
        "encoder_repeats": int(encoder.repeats),
        "encoder_seed": int(encoder.seed),
        "num_points": int(num_points),
        "radius": float(radius),
        "nucleus_center_count": int(nucleus_center_count),
        "interface_center_count": int(interface_center_count),
        "background_center_count": int(background_center_count),
        "point_cloud_batch_size": int(point_cloud_batch_size),
        "selection_seed": int(selection_seed),
    }
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and not force_recompute:
        existing = _load_json(manifest_path)
        if existing["spec"] != spec:
            raise RuntimeError(
                f"Nested feature-cache specification changed: {target}. "
                "Choose a new cache path or set force_recompute=true."
            )
        return NestedFeatureCache.load(target)
    if target.exists() and not force_recompute:
        raise RuntimeError(
            f"Nested feature-cache exists without a complete manifest: {target}."
        )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    representatives = _representative_complete_branches(snapshot)
    parent_tokens: list[np.ndarray] = []
    parent_positions: list[np.ndarray] = []
    parent_regions: list[np.ndarray] = []
    parent_atom_ids: list[np.ndarray] = []
    velocity_tokens: list[np.ndarray] = []
    velocity_parents: list[np.ndarray] = []
    velocity_momenta: list[np.ndarray] = []
    velocity_regions: list[np.ndarray] = []
    scalar_features = np.empty((len(snapshot.parents), 3), dtype=np.float32)
    force_features = np.empty((len(snapshot.parents), len(REGION_NAMES), 4), dtype=np.float32)

    for parent_position, parent in enumerate(snapshot.parents):
        parent_index = int(parent["parent_index"])
        pairs = [representatives.get((parent_index, momentum)) for momentum in (0, 1)]
        if any(pair is None for pair in pairs):
            raise RuntimeError(
                "Every nested parent needs one complete branch for each momentum before "
                f"feature extraction: parent={parent['parent_id']}."
            )
        trajectories = [_trajectory_for(snapshot, pair) for pair in pairs if pair is not None]
        frames = [trajectory.load_frames([0])[0] for trajectory in trajectories]
        if not np.array_equal(frames[0].positions, frames[1].positions):
            raise RuntimeError(
                f"Nested momenta changed parent positions: parent={parent['parent_id']}."
            )
        parent_path = snapshot.root / str(parent["data_file"])
        ptm_atom_ids, cluster_labels, crystalline_fraction, largest_cluster = (
            _parent_ptm_clusters(
                parent_path,
                ptm_rmsd_cutoff=float(snapshot.manifest["basins"]["ptm_rmsd_cutoff"]),
                cluster_cutoff_A=float(
                    snapshot.manifest["basins"]["cluster_connectivity_cutoff_A"]
                ),
            )
        )
        if not np.array_equal(ptm_atom_ids, frames[0].atom_ids):
            raise RuntimeError(
                f"OVITO and binary atom order differ for parent={parent['parent_id']}."
            )
        expected_cluster = int(parent["source_largest_crystalline_cluster_atoms"])
        if largest_cluster != expected_cluster:
            raise RuntimeError(
                "Recomputed initial PTM cluster disagrees with parent selection: "
                f"parent={parent['parent_id']}, expected={expected_cluster}, "
                f"observed={largest_cluster}."
            )
        groups = select_parent_region_atom_ids(
            all_atom_ids=frames[0].atom_ids,
            positions=frames[0].positions,
            box_lengths=frames[0].box_lengths,
            cluster_labels=cluster_labels,
            nucleus_center_count=int(nucleus_center_count),
            interface_center_count=int(interface_center_count),
            background_center_count=int(background_center_count),
            seed=int(selection_seed) + parent_index,
        )
        center_ids = np.concatenate(groups)
        region = np.concatenate(
            [np.full(group.size, index, dtype=np.int8) for index, group in enumerate(groups)]
        )
        frame = ShootingPositionFrame(
            timestep=0,
            atom_ids=frames[0].atom_ids,
            atom_types=frames[0].atom_types,
            positions=frames[0].positions,
            box_low=frames[0].box_low,
            box_high=frames[0].box_high,
        )
        environments = build_periodic_environment_batch(
            frame,
            center_atom_ids=center_ids,
            num_points=int(num_points),
            radius=float(radius),
            spatial_context_center_count=0,
        )
        encoded_chunks = [
            encoder.encode(
                environments.points[start : start + int(point_cloud_batch_size)]
            ).cpu()
            for start in range(0, center_ids.size, int(point_cloud_batch_size))
        ]
        token_z = torch.cat(encoded_chunks, dim=0).numpy().astype(np.float32, copy=False)
        parent_tokens.append(token_z)
        parent_positions.append(np.full(center_ids.size, parent_position, dtype=np.int32))
        parent_regions.append(region)
        parent_atom_ids.append(center_ids)

        scalar_features[parent_position] = np.asarray(
            [
                float(parent["temperature_K"]),
                np.log1p(float(largest_cluster)),
                crystalline_fraction,
            ],
            dtype=np.float32,
        )
        representative_outcome = pairs[0][1]  # type: ignore[index]
        observable_path = Path(str(representative_outcome["observables_artifact"]["path"]))
        with np.load(observable_path, allow_pickle=False) as observables:
            forces = np.asarray(observables["initial_forces_eV_per_A"], dtype=np.float32)
        if forces.shape != (int(snapshot.manifest["atom_count"]), 3):
            raise RuntimeError(
                f"Initial force shape changed for parent={parent['parent_id']}: {forces.shape}."
            )
        for region_index, group in enumerate(groups):
            magnitude = np.linalg.norm(forces[group - 1], axis=1)
            force_features[parent_position, region_index] = np.asarray(
                [
                    magnitude.mean(),
                    magnitude.std(),
                    magnitude.max(),
                    np.sqrt(np.mean(magnitude**2)),
                ],
                dtype=np.float32,
            )

        positions = np.asarray(frames[0].positions, dtype=np.float32)
        box_lengths = np.asarray(frames[0].box_lengths, dtype=np.float32)
        tree = cKDTree(positions, boxsize=box_lengths, balanced_tree=False)
        center_indices = center_ids - 1
        _, neighbor_indices = tree.query(positions[center_indices], k=int(num_points))
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        relative = positions[neighbor_indices] - positions[center_indices, None]
        relative -= box_lengths[None, None, :] * np.round(
            relative / box_lengths[None, None, :]
        )
        relative = (relative / float(radius)).astype(np.float32, copy=False)
        for momentum_index, momentum_frame in enumerate(frames):
            descriptors = invariant_velocity_token_features(
                relative,
                np.asarray(momentum_frame.velocities[neighbor_indices], dtype=np.float32),
                np.asarray(momentum_frame.velocities[center_indices], dtype=np.float32),
            )
            velocity_tokens.append(descriptors)
            velocity_parents.append(
                np.full(center_ids.size, parent_position, dtype=np.int32)
            )
            velocity_momenta.append(
                np.full(center_ids.size, momentum_index, dtype=np.int8)
            )
            velocity_regions.append(region)
        print(
            f"[nested-committor] encoded parent {parent_position + 1}/{len(snapshot.parents)} "
            f"{parent['parent_id']} centers={center_ids.size} nucleus={groups[0].size}",
            flush=True,
        )

    arrays = {
        "parent_scalar_features": scalar_features,
        "parent_force_features": force_features,
        "token_z": np.concatenate(parent_tokens, axis=0),
        "token_parent_position": np.concatenate(parent_positions),
        "token_region": np.concatenate(parent_regions),
        "token_atom_id": np.concatenate(parent_atom_ids),
        "velocity_token_features": np.concatenate(velocity_tokens, axis=0),
        "velocity_parent_position": np.concatenate(velocity_parents),
        "velocity_momentum_index": np.concatenate(velocity_momenta),
        "velocity_region": np.concatenate(velocity_regions),
    }
    for name, values in arrays.items():
        temporary = target / f"{name}.tmp.npy"
        final = target / f"{name}.npy"
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, final)
    cache_manifest = {
        "state": "complete",
        "spec": spec,
        "region_names": list(REGION_NAMES),
        "scalar_feature_names": [
            "temperature_K",
            "log1p_largest_crystalline_cluster_atoms",
            "crystalline_fraction",
        ],
        "force_feature_names": ["mean_norm", "std_norm", "max_norm", "rms_norm"],
        "velocity_token_feature_names": list(VELOCITY_TOKEN_FEATURE_NAMES),
        "array_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "array_dtypes": {name: values.dtype.name for name, values in arrays.items()},
    }
    temporary_manifest = target / "manifest.tmp.json"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(cache_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_manifest, manifest_path)
    return NestedFeatureCache.load(target)


def _aggregate_projected_tokens(
    projected: np.ndarray,
    parent_position: np.ndarray,
    region: np.ndarray,
    *,
    parent_count: int,
) -> np.ndarray:
    dimension = int(projected.shape[1])
    output = np.empty((parent_count, len(REGION_NAMES), 2 * dimension), dtype=np.float64)
    for parent in range(parent_count):
        for region_index in range(len(REGION_NAMES)):
            selected = projected[
                (parent_position == parent) & (region == region_index)
            ]
            if selected.shape[0] == 0:
                raise RuntimeError(
                    f"No projected tokens for parent={parent}, region={REGION_NAMES[region_index]}."
                )
            output[parent, region_index] = np.concatenate(
                [selected.mean(axis=0), selected.std(axis=0)]
            )
    return output.reshape(parent_count, -1)


def _aggregate_velocity_tokens(
    projected: np.ndarray,
    parent_position: np.ndarray,
    momentum_index: np.ndarray,
    region: np.ndarray,
    *,
    parent_count: int,
) -> np.ndarray:
    dimension = int(projected.shape[1])
    output = np.empty(
        (parent_count, 2, len(REGION_NAMES), 2 * dimension), dtype=np.float64
    )
    for parent in range(parent_count):
        for momentum in (0, 1):
            for region_index in range(len(REGION_NAMES)):
                selected = projected[
                    (parent_position == parent)
                    & (momentum_index == momentum)
                    & (region == region_index)
                ]
                if selected.shape[0] == 0:
                    raise RuntimeError(
                        "No velocity tokens for "
                        f"parent={parent}, momentum={momentum}, "
                        f"region={REGION_NAMES[region_index]}."
                    )
                output[parent, momentum, region_index] = np.concatenate(
                    [selected.mean(axis=0), selected.std(axis=0)]
                )
    return output.reshape(parent_count, 2, -1)


@dataclass(frozen=True)
class _FeatureSet:
    sample_features: np.ndarray
    geoframe_pca: PCA | None
    velocity_pca: PCA | None
    feature_names: tuple[str, ...]


def _build_feature_set(
    cache: NestedFeatureCache,
    *,
    sample_parent: np.ndarray,
    sample_momentum: np.ndarray,
    fit_parent_mask: np.ndarray,
    kind: str,
    geoframe_dimension: int,
    velocity_dimension: int,
) -> _FeatureSet:
    parent_count = int(cache.parent_scalar_features.shape[0])
    scalar = np.asarray(cache.parent_scalar_features, dtype=np.float64)
    force = np.asarray(cache.parent_force_features, dtype=np.float64).reshape(parent_count, -1)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if kind == "temperature":
        parts.append(scalar[:, :1])
        names.append("temperature_K")
    elif kind in {
        "collective_variables",
        "collective_variables_force",
        "geoframe_cv",
        "geoframe_cv_force",
        "phase_space",
    }:
        parts.append(scalar)
        names.extend(cache.manifest["scalar_feature_names"])
    elif kind == "geoframe":
        parts.append(scalar[:, :1])
        names.append("temperature_K")
    else:
        raise ValueError(f"Unknown nested feature kind={kind!r}.")

    geoframe_pca: PCA | None = None
    if kind in {"geoframe", "geoframe_cv", "geoframe_cv_force", "phase_space"}:
        token_parent = np.asarray(cache.token_parent_position, dtype=np.int32)
        train_token = fit_parent_mask[token_parent]
        geoframe_pca = PCA(n_components=int(geoframe_dimension), svd_solver="full")
        geoframe_pca.fit(np.asarray(cache.token_z[train_token], dtype=np.float64))
        projected = geoframe_pca.transform(np.asarray(cache.token_z, dtype=np.float64))
        aggregated = _aggregate_projected_tokens(
            projected,
            token_parent,
            np.asarray(cache.token_region, dtype=np.int8),
            parent_count=parent_count,
        )
        parts.append(aggregated)
        names.extend(
            f"geoframe_{region}_{stat}_pc{component + 1}"
            for region in REGION_NAMES
            for stat in ("mean", "std")
            for component in range(int(geoframe_dimension))
        )
    if kind in {"collective_variables_force", "geoframe_cv_force", "phase_space"}:
        parts.append(force)
        names.extend(
            f"force_{region}_{feature}"
            for region in REGION_NAMES
            for feature in cache.manifest["force_feature_names"]
        )
    parent_features = np.concatenate(parts, axis=1)
    sample_parts = [parent_features[sample_parent]]

    velocity_pca: PCA | None = None
    if kind == "phase_space":
        velocity_parent = np.asarray(cache.velocity_parent_position, dtype=np.int32)
        train_velocity = fit_parent_mask[velocity_parent]
        velocity_pca = PCA(n_components=int(velocity_dimension), svd_solver="full")
        velocity_pca.fit(
            np.asarray(cache.velocity_token_features[train_velocity], dtype=np.float64)
        )
        velocity_projected = velocity_pca.transform(
            np.asarray(cache.velocity_token_features, dtype=np.float64)
        )
        velocity_aggregated = _aggregate_velocity_tokens(
            velocity_projected,
            velocity_parent,
            np.asarray(cache.velocity_momentum_index, dtype=np.int8),
            np.asarray(cache.velocity_region, dtype=np.int8),
            parent_count=parent_count,
        )
        sample_parts.append(velocity_aggregated[sample_parent, sample_momentum])
        names.extend(
            f"velocity_{region}_{stat}_pc{component + 1}"
            for region in REGION_NAMES
            for stat in ("mean", "std")
            for component in range(int(velocity_dimension))
        )
    return _FeatureSet(
        sample_features=np.concatenate(sample_parts, axis=1),
        geoframe_pca=geoframe_pca,
        velocity_pca=velocity_pca,
        feature_names=tuple(names),
    )


@dataclass(frozen=True)
class _FittedLogistic:
    scaler: StandardScaler
    model: LogisticRegression

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    C: float,
    prior_groups: np.ndarray,
    beta_prior: float,
) -> _FittedLogistic:
    scaler = StandardScaler().fit(features)
    fit_features = features
    fit_labels = labels
    sample_weight = np.ones(labels.size, dtype=np.float64)
    if float(beta_prior) > 0.0:
        representatives: list[np.ndarray] = []
        for group in np.unique(prior_groups):
            selected = features[prior_groups == group]
            if not np.allclose(selected, selected[:1], rtol=0.0, atol=1.0e-10):
                raise RuntimeError(
                    "Beta-binomial pseudocount group has non-identical state features: "
                    f"group={group}."
                )
            representatives.append(selected[0])
        prior_features = np.stack(representatives, axis=0)
        fit_features = np.concatenate(
            [features, prior_features, prior_features], axis=0
        )
        fit_labels = np.concatenate(
            [
                labels,
                np.zeros(prior_features.shape[0], dtype=labels.dtype),
                np.ones(prior_features.shape[0], dtype=labels.dtype),
            ]
        )
        sample_weight = np.concatenate(
            [
                sample_weight,
                np.full(prior_features.shape[0], float(beta_prior)),
                np.full(prior_features.shape[0], float(beta_prior)),
            ]
        )
    model = LogisticRegression(C=float(C), solver="lbfgs", max_iter=20_000)
    model.fit(
        scaler.transform(fit_features),
        fit_labels,
        sample_weight=sample_weight,
    )
    return _FittedLogistic(scaler=scaler, model=model)


def _binary_nll(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(log_loss(labels, np.clip(prediction, 1.0e-7, 1.0 - 1.0e-7), labels=[0, 1]))


def _metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    parent_position: np.ndarray,
) -> dict[str, Any]:
    if labels.size == 0:
        return {"resolved_branch_count": 0, "parent_count": 0}
    parent_values = np.unique(parent_position)
    empirical = np.asarray(
        [labels[parent_position == parent].mean() for parent in parent_values],
        dtype=np.float64,
    )
    predicted_parent = np.asarray(
        [prediction[parent_position == parent].mean() for parent in parent_values],
        dtype=np.float64,
    )
    auc = (
        float(roc_auc_score(labels, prediction))
        if np.unique(labels).size == 2
        else None
    )
    correlation = (
        float(np.corrcoef(empirical, predicted_parent)[0, 1])
        if parent_values.size >= 2
        and np.std(empirical) > 0.0
        and np.std(predicted_parent) > 0.0
        else None
    )
    return {
        "resolved_branch_count": int(labels.size),
        "parent_count": int(parent_values.size),
        "basin_A_count": int(np.sum(labels == 0)),
        "basin_B_count": int(np.sum(labels == 1)),
        "negative_log_likelihood": _binary_nll(labels, prediction),
        "branch_brier": float(np.mean((prediction - labels) ** 2)),
        "roc_auc": auc,
        "parent_pB_brier": float(np.mean((predicted_parent - empirical) ** 2)),
        "parent_pB_mae": float(np.mean(np.abs(predicted_parent - empirical))),
        "parent_pB_correlation": correlation,
    }


def _resolved_samples(
    snapshot: NestedShootingSnapshot,
) -> dict[str, np.ndarray]:
    parent_position = snapshot.parent_index_to_position
    parent_by_index = {
        int(parent["parent_index"]): parent for parent in snapshot.parents
    }
    outcomes = [value for value in snapshot.outcomes if not bool(value["censored"])]
    labels = np.asarray(
        [value["first_passage_outcome"] == "basin_B_crystal" for value in outcomes],
        dtype=np.int64,
    )
    return {
        "label": labels,
        "parent_position": np.asarray(
            [parent_position[int(value["parent_index"])] for value in outcomes],
            dtype=np.int32,
        ),
        "momentum_index": np.asarray(
            [int(value["momentum_index"]) for value in outcomes], dtype=np.int8
        ),
        "branch_index": np.asarray(
            [int(value["branch_index"]) for value in outcomes], dtype=np.int32
        ),
        "split": np.asarray([str(value["source_split"]) for value in outcomes]),
        "role": np.asarray([str(value["basin_role"]) for value in outcomes]),
        "source_run_id": np.asarray([str(value["source_run_id"]) for value in outcomes]),
        "parent_id": np.asarray([str(value["parent_id"]) for value in outcomes]),
        "temperature_K": np.asarray(
            [float(value["temperature_K"]) for value in outcomes], dtype=np.float64
        ),
        "parent_cluster_atoms": np.asarray(
            [
                int(
                    parent_by_index[int(value["parent_index"])][
                        "source_largest_crystalline_cluster_atoms"
                    ]
                )
                for value in outcomes
            ],
            dtype=np.int64,
        ),
    }


def _select_hyperparameters(
    cache: NestedFeatureCache,
    samples: dict[str, np.ndarray],
    *,
    kind: str,
    geoframe_dimensions: Sequence[int],
    velocity_dimensions: Sequence[int],
    logistic_C_values: Sequence[float],
    group_folds: int,
    beta_prior: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimization = samples["split"] == "optimization"
    resolved_indices = np.flatnonzero(optimization)
    parent_count = int(cache.parent_scalar_features.shape[0])
    fit_parent_mask = np.zeros(parent_count, dtype=bool)
    fit_parent_mask[np.unique(samples["parent_position"][optimization])] = True
    geo_grid = (
        tuple(int(value) for value in geoframe_dimensions)
        if kind in {"geoframe", "geoframe_cv", "geoframe_cv_force", "phase_space"}
        else (0,)
    )
    velocity_grid = (
        tuple(int(value) for value in velocity_dimensions)
        if kind == "phase_space"
        else (0,)
    )
    groups = samples["source_run_id"][optimization]
    prior_groups = samples["parent_position"][optimization].astype(np.int64)
    if kind == "phase_space":
        prior_groups = 2 * prior_groups + samples["momentum_index"][
            optimization
        ].astype(np.int64)
    unique_groups = np.unique(groups)
    folds = min(int(group_folds), int(unique_groups.size))
    if folds < 2:
        raise RuntimeError(
            f"Grouped hyperparameter selection needs at least two source runs, got {folds}."
        )
    splitter = GroupKFold(n_splits=folds)
    records: list[dict[str, Any]] = []
    for geo_dim in geo_grid:
        for velocity_dim in velocity_grid:
            feature_set = _build_feature_set(
                cache,
                sample_parent=samples["parent_position"],
                sample_momentum=samples["momentum_index"],
                fit_parent_mask=fit_parent_mask,
                kind=kind,
                geoframe_dimension=geo_dim,
                velocity_dimension=velocity_dim,
            )
            X = feature_set.sample_features[optimization]
            y = samples["label"][optimization]
            for C in logistic_C_values:
                fold_loss = 0.0
                fold_count = 0
                for train_local, validation_local in splitter.split(X, y, groups):
                    if np.unique(y[train_local]).size != 2:
                        raise RuntimeError(
                            "A grouped optimization fold contains one outcome class; "
                            f"kind={kind}, held_out_groups="
                            f"{np.unique(groups[validation_local]).tolist()}."
                        )
                    fitted = _fit_logistic(
                        X[train_local],
                        y[train_local],
                        C=float(C),
                        prior_groups=prior_groups[train_local],
                        beta_prior=float(beta_prior),
                    )
                    prediction = fitted.predict(X[validation_local])
                    fold_loss += (
                        _binary_nll(y[validation_local], prediction)
                        * validation_local.size
                    )
                    fold_count += int(validation_local.size)
                records.append(
                    {
                        "kind": kind,
                        "geoframe_dimension": geo_dim,
                        "velocity_dimension": velocity_dim,
                        "C": float(C),
                        "beta_prior_per_state": float(beta_prior),
                        "grouped_cv_negative_log_likelihood": fold_loss / fold_count,
                        "optimization_resolved_indices_sha256": hashlib.sha256(
                            resolved_indices.tobytes()
                        ).hexdigest(),
                    }
                )
    best = min(
        records,
        key=lambda value: (
            value["grouped_cv_negative_log_likelihood"],
            value["geoframe_dimension"] + value["velocity_dimension"],
            value["C"],
        ),
    )
    return best, records


def _fit_and_predict(
    cache: NestedFeatureCache,
    samples: dict[str, np.ndarray],
    *,
    kind: str,
    hyperparameters: dict[str, Any],
    fit_splits: Sequence[str],
    beta_prior: float,
) -> tuple[np.ndarray, _FeatureSet, _FittedLogistic]:
    fit_sample = np.isin(samples["split"], np.asarray(tuple(fit_splits)))
    parent_count = int(cache.parent_scalar_features.shape[0])
    fit_parent_mask = np.zeros(parent_count, dtype=bool)
    fit_parent_mask[np.unique(samples["parent_position"][fit_sample])] = True
    feature_set = _build_feature_set(
        cache,
        sample_parent=samples["parent_position"],
        sample_momentum=samples["momentum_index"],
        fit_parent_mask=fit_parent_mask,
        kind=kind,
        geoframe_dimension=int(hyperparameters["geoframe_dimension"]),
        velocity_dimension=int(hyperparameters["velocity_dimension"]),
    )
    prior_groups = samples["parent_position"][fit_sample].astype(np.int64)
    if kind == "phase_space":
        prior_groups = 2 * prior_groups + samples["momentum_index"][fit_sample].astype(
            np.int64
        )
    fitted = _fit_logistic(
        feature_set.sample_features[fit_sample],
        samples["label"][fit_sample],
        C=float(hyperparameters["C"]),
        prior_groups=prior_groups,
        beta_prior=float(beta_prior),
    )
    return fitted.predict(feature_set.sample_features), feature_set, fitted


def _outcome_inventory(snapshot: NestedShootingSnapshot) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        inventory[split] = {}
        for role in ("transition_candidate", "liquid_control", "crystal_control"):
            selected = [
                value
                for value in snapshot.outcomes
                if value["source_split"] == split and value["basin_role"] == role
            ]
            inventory[split][role] = {
                "branch_count": len(selected),
                "parent_count": len({value["parent_id"] for value in selected}),
                "basin_A_liquid": sum(
                    value["first_passage_outcome"] == "basin_A_liquid"
                    for value in selected
                ),
                "basin_B_crystal": sum(
                    value["first_passage_outcome"] == "basin_B_crystal"
                    for value in selected
                ),
                "censored": sum(bool(value["censored"]) for value in selected),
            }
    return inventory


def _nested_variance_diagnostics(snapshot: NestedShootingSnapshot) -> dict[str, Any]:
    by_momentum: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for outcome in snapshot.outcomes:
        by_momentum[
            (int(outcome["parent_index"]), int(outcome["momentum_index"]))
        ].append(outcome)
        by_parent[int(outcome["parent_index"])].append(outcome)

    resolved_noise_pairs = 0
    discordant_noise_pairs = 0
    momentum_pB: dict[int, dict[int, float]] = defaultdict(dict)
    for (parent_index, momentum_index), outcomes in by_momentum.items():
        resolved = [value for value in outcomes if not bool(value["censored"])]
        if len(outcomes) == 2 and len(resolved) == 2:
            resolved_noise_pairs += 1
            labels = [
                value["first_passage_outcome"] == "basin_B_crystal"
                for value in resolved
            ]
            discordant_noise_pairs += int(labels[0] != labels[1])
        if resolved:
            momentum_pB[parent_index][momentum_index] = float(
                np.mean(
                    [
                        value["first_passage_outcome"] == "basin_B_crystal"
                        for value in resolved
                    ]
                )
            )

    momentum_differences = [
        abs(values[0] - values[1])
        for values in momentum_pB.values()
        if set(values) == {0, 1}
    ]
    mixed_transition_parents = 0
    resolved_transition_parents = 0
    parent_by_index = {
        int(parent["parent_index"]): parent for parent in snapshot.parents
    }
    for parent_index, outcomes in by_parent.items():
        if parent_by_index[parent_index]["basin_role"] != "transition_candidate":
            continue
        resolved = [value for value in outcomes if not bool(value["censored"])]
        if not resolved:
            continue
        resolved_transition_parents += 1
        p_B = float(
            np.mean(
                [
                    value["first_passage_outcome"] == "basin_B_crystal"
                    for value in resolved
                ]
            )
        )
        mixed_transition_parents += int(0.2 < p_B < 0.8)
    return {
        "resolved_thermostat_noise_pair_count": resolved_noise_pairs,
        "discordant_thermostat_noise_pair_count": discordant_noise_pairs,
        "discordant_thermostat_noise_pair_fraction": (
            discordant_noise_pairs / resolved_noise_pairs
            if resolved_noise_pairs
            else None
        ),
        "parents_with_both_resolved_momentum_estimates": len(momentum_differences),
        "parents_with_different_momentum_estimates": sum(
            value > 0.0 for value in momentum_differences
        ),
        "mean_absolute_momentum_pB_difference": (
            float(np.mean(momentum_differences)) if momentum_differences else None
        ),
        "resolved_transition_parent_count": resolved_transition_parents,
        "mixed_transition_parent_count": mixed_transition_parents,
    }


def _save_model_arrays(
    output_dir: Path,
    model_name: str,
    prefix: str,
    feature_set: _FeatureSet,
    fitted: _FittedLogistic,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        f"{model_name}_{prefix}_scaler_mean": fitted.scaler.mean_,
        f"{model_name}_{prefix}_scaler_scale": fitted.scaler.scale_,
        f"{model_name}_{prefix}_coefficient": fitted.model.coef_,
        f"{model_name}_{prefix}_intercept": fitted.model.intercept_,
    }
    if feature_set.geoframe_pca is not None:
        arrays[f"{model_name}_{prefix}_geoframe_pca_mean"] = feature_set.geoframe_pca.mean_
        arrays[f"{model_name}_{prefix}_geoframe_pca_components"] = (
            feature_set.geoframe_pca.components_
        )
    if feature_set.velocity_pca is not None:
        arrays[f"{model_name}_{prefix}_velocity_pca_mean"] = feature_set.velocity_pca.mean_
        arrays[f"{model_name}_{prefix}_velocity_pca_components"] = (
            feature_set.velocity_pca.components_
        )
    return {"arrays": arrays, "feature_names": list(feature_set.feature_names)}


def _plot_data_and_models(
    output_dir: Path,
    snapshot: NestedShootingSnapshot,
    cache: NestedFeatureCache,
    samples: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    parent_pB = []
    censor_fraction = []
    cluster = []
    temperature = []
    role = []
    for parent_position, parent in enumerate(snapshot.parents):
        selected = samples["parent_position"] == parent_position
        parent_outcomes = [
            value
            for value in snapshot.outcomes
            if int(value["parent_index"]) == int(parent["parent_index"])
        ]
        parent_pB.append(
            float(samples["label"][selected].mean()) if np.any(selected) else np.nan
        )
        censor_fraction.append(
            float(np.mean([bool(value["censored"]) for value in parent_outcomes]))
        )
        cluster.append(int(parent["source_largest_crystalline_cluster_atoms"]))
        temperature.append(float(parent["temperature_K"]))
        role.append(str(parent["basin_role"]))
    figure, axes = plt.subplots(
        2, 1, figsize=(7.0, 7.0), sharex=True, constrained_layout=True
    )
    resolved = np.isfinite(parent_pB)
    scatter = axes[0].scatter(
        np.asarray(cluster)[resolved],
        np.asarray(parent_pB)[resolved],
        c=np.asarray(temperature)[resolved],
        cmap="viridis",
        s=55,
        vmin=min(temperature),
        vmax=max(temperature),
    )
    axes[0].set_ylabel("empirical resolved $p_B$ (4 futures maximum)")
    axes[1].scatter(
        cluster,
        censor_fraction,
        c=temperature,
        cmap="viridis",
        s=55,
        vmin=min(temperature),
        vmax=max(temperature),
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("initial largest crystalline cluster (atoms)")
    axes[1].set_ylabel("censored-future fraction")
    figure.colorbar(scatter, ax=axes, label="temperature (K)")
    figure.savefig(plots / "empirical_pB_vs_initial_cluster.png", dpi=180)
    plt.close(figure)


def _exact_final_parent_bootstrap(
    labels: np.ndarray,
    parent_position: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> dict[str, Any]:
    parents = np.unique(parent_position)
    if parents.size != 4:
        raise RuntimeError(
            "The repository nested pilot has exactly four final transition parents; "
            f"observed={parents.size}."
        )
    observed = float(
        np.mean((candidate_prediction - labels) ** 2)
        - np.mean((baseline_prediction - labels) ** 2)
    )
    differences: list[float] = []
    for sampled in itertools.product(parents.tolist(), repeat=parents.size):
        indices = np.concatenate(
            [np.flatnonzero(parent_position == parent) for parent in sampled]
        )
        differences.append(
            float(
                np.mean((candidate_prediction[indices] - labels[indices]) ** 2)
                - np.mean((baseline_prediction[indices] - labels[indices]) ** 2)
            )
        )
    values = np.asarray(differences, dtype=np.float64)
    return {
        "brier_difference_candidate_minus_temperature": observed,
        "exact_parent_bootstrap_95_interval": np.quantile(
            values, [0.025, 0.975]
        ).tolist(),
        "exact_parent_bootstrap_probability_candidate_better": float(
            np.mean(values < 0.0)
        ),
        "bootstrap_enumeration_count": int(values.size),
    }

    names = list(predictions)
    final_brier = [
        metrics["models"][name]["final_validation_transition"]["branch_brier"]
        for name in names
    ]
    final_nll = [
        metrics["models"][name]["final_validation_transition"][
            "negative_log_likelihood"
        ]
        for name in names
    ]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].bar(x, final_brier)
    axes[0].set_ylabel("final transition branch Brier")
    axes[1].bar(x, final_nll)
    axes[1].set_ylabel("final transition negative log likelihood")
    for axis in axes:
        axis.set_xticks(x, names, rotation=25, ha="right")
    figure.tight_layout()
    figure.savefig(plots / "final_transition_model_comparison.png", dpi=180)
    plt.close(figure)


def fit_nested_committor_models(
    snapshot: NestedShootingSnapshot,
    cache: NestedFeatureCache,
    *,
    output_dir: str | Path,
    geoframe_dimensions: Sequence[int],
    velocity_dimensions: Sequence[int],
    logistic_C_values: Sequence[float],
    group_folds: int,
) -> dict[str, Any]:
    """Fit calibrated finite-time first-passage classifiers and save all outputs."""

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    samples = _resolved_samples(snapshot)
    model_specs = {
        "temperature": ("temperature", 0.0),
        "collective_variables": ("collective_variables", 0.0),
        "collective_variables_jeffreys": ("collective_variables", 0.5),
        "collective_variables_force": ("collective_variables_force", 0.0),
        "collective_variables_force_jeffreys": (
            "collective_variables_force",
            0.5,
        ),
        "geoframe": ("geoframe", 0.0),
        "geoframe_plus_cv": ("geoframe_cv", 0.0),
        "geoframe_plus_cv_jeffreys": ("geoframe_cv", 0.5),
        "geoframe_plus_cv_force": ("geoframe_cv_force", 0.0),
        "phase_space": ("phase_space", 0.0),
        "phase_space_jeffreys": ("phase_space", 0.5),
    }
    metrics: dict[str, Any] = {
        "campaign_root": str(snapshot.root),
        "feature_cache": str(cache.path),
        "scientific_target": (
            "Probability of first persistent hit of basin_B_crystal before "
            "basin_A_liquid within the temperature-specific maximum duration; "
            "censored branches are excluded from binary likelihood."
        ),
        "outcome_inventory": _outcome_inventory(snapshot),
        "nested_variance_diagnostics": _nested_variance_diagnostics(snapshot),
        "resolved_branch_count": int(samples["label"].size),
        "censored_branch_count": int(sum(bool(value["censored"]) for value in snapshot.outcomes)),
        "models": {},
    }
    saved_arrays: dict[str, np.ndarray] = {
        name: values for name, values in samples.items() if values.dtype.kind != "U"
    }
    model_parameter_arrays: dict[str, np.ndarray] = {}
    predictions_for_plot: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}

    for model_name, (kind, beta_prior) in model_specs.items():
        best, cv_records = _select_hyperparameters(
            cache,
            samples,
            kind=kind,
            geoframe_dimensions=geoframe_dimensions,
            velocity_dimensions=velocity_dimensions,
            logistic_C_values=logistic_C_values,
            group_folds=int(group_folds),
            beta_prior=float(beta_prior),
        )
        selection_prediction, selection_features, selection_model = _fit_and_predict(
            cache,
            samples,
            kind=kind,
            hyperparameters=best,
            fit_splits=("optimization",),
            beta_prior=float(beta_prior),
        )
        final_prediction, final_features, final_model = _fit_and_predict(
            cache,
            samples,
            kind=kind,
            hyperparameters=best,
            fit_splits=("optimization", "model_selection"),
            beta_prior=float(beta_prior),
        )
        masks = {
            "optimization": samples["split"] == "optimization",
            "model_selection": samples["split"] == "model_selection",
            "final_validation_all": samples["split"] == "final_validation",
            "final_validation_transition": (
                (samples["split"] == "final_validation")
                & (samples["role"] == "transition_candidate")
            ),
        }
        model_metrics: dict[str, Any] = {
            "selected_hyperparameters": best,
            "beta_prior_per_state": float(beta_prior),
            "grouped_cv_candidates": cv_records,
            "model_selection_fit_contract": "optimization only",
            "final_validation_fit_contract": "optimization + model_selection",
        }
        for subset, mask in masks.items():
            prediction = (
                selection_prediction
                if subset in {"optimization", "model_selection"}
                else final_prediction
            )
            model_metrics[subset] = _metrics(
                samples["label"][mask],
                prediction[mask],
                samples["parent_position"][mask],
            )
        metrics["models"][model_name] = model_metrics
        saved_arrays[f"{model_name}_optimization_fit_prediction"] = selection_prediction
        saved_arrays[f"{model_name}_final_fit_prediction"] = final_prediction
        predictions_for_plot[model_name] = final_prediction
        for prefix, feature_set, fitted in (
            ("optimization_fit", selection_features, selection_model),
            ("final_fit", final_features, final_model),
        ):
            saved = _save_model_arrays(
                target, model_name, prefix, feature_set, fitted
            )
            model_parameter_arrays.update(saved["arrays"])
            model_metadata[f"{model_name}_{prefix}"] = {
                "feature_names": saved["feature_names"],
                "fit_splits": (
                    ["optimization"]
                    if prefix == "optimization_fit"
                    else ["optimization", "model_selection"]
                ),
            }
        print(
            f"[nested-committor] {model_name}: CV NLL="
            f"{best['grouped_cv_negative_log_likelihood']:.4f}, final transition Brier="
            f"{model_metrics['final_validation_transition'].get('branch_brier')}",
            flush=True,
        )

    final_transition = (
        (samples["split"] == "final_validation")
        & (samples["role"] == "transition_candidate")
    )
    baseline = predictions_for_plot["temperature"][final_transition]
    metrics["final_transition_uncertainty"] = {
        model_name: _exact_final_parent_bootstrap(
            samples["label"][final_transition],
            samples["parent_position"][final_transition],
            baseline,
            prediction[final_transition],
        )
        for model_name, prediction in predictions_for_plot.items()
        if model_name != "temperature"
    }

    np.savez(target / "coordinates_and_predictions.npz", **saved_arrays)
    np.savez(target / "model_parameters.npz", **model_parameter_arrays)
    with (target / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (target / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _plot_data_and_models(target, snapshot, cache, samples, predictions_for_plot, metrics)
    return metrics


__all__ = [
    "NestedFeatureCache",
    "NestedShootingSnapshot",
    "extract_nested_feature_cache",
    "fit_nested_committor_models",
    "load_nested_shooting_snapshot",
    "select_parent_region_atom_ids",
]
