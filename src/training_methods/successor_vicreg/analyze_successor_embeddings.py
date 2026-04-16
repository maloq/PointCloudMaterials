from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch

from src.analysis.clustering import prepare_clustering_features
from src.training_methods.successor_vicreg.artifact import SuccessorEmbeddingsArtifact
from src.training_methods.successor_vicreg.export_successor_embeddings import (
    load_successor_checkpoint,
)
from src.training_methods.successor_vicreg.module import SuccessorVICRegModule
from src.training_methods.vamp.common import ensure_dir, log_progress, save_json


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class ClusterTargetVarianceAccumulator:
    def __init__(self, *, cluster_count: int, target_dim: int) -> None:
        self.cluster_count = int(cluster_count)
        self.target_dim = int(target_dim)
        self.count = np.zeros((self.cluster_count,), dtype=np.int64)
        self.mean = np.zeros((self.cluster_count, self.target_dim), dtype=np.float64)
        self.m2 = np.zeros((self.cluster_count,), dtype=np.float64)

    def update(self, labels: np.ndarray, values: np.ndarray) -> None:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        values_arr = np.asarray(values, dtype=np.float32)
        if values_arr.ndim != 2:
            raise ValueError(
                "ClusterTargetVarianceAccumulator expects values with shape (B, D), "
                f"got {tuple(values_arr.shape)}."
            )
        if labels_arr.shape[0] != values_arr.shape[0]:
            raise ValueError(
                "ClusterTargetVarianceAccumulator labels/value length mismatch. "
                f"labels={labels_arr.shape[0]}, values={values_arr.shape[0]}."
            )
        for cluster_id in np.unique(labels_arr):
            cluster_int = int(cluster_id)
            if cluster_int < 0 or cluster_int >= self.cluster_count:
                continue
            members = values_arr[labels_arr == cluster_int].astype(np.float64, copy=False)
            if members.shape[0] == 0:
                continue
            batch_count = int(members.shape[0])
            batch_mean = members.mean(axis=0)
            centered = members - batch_mean[None, :]
            batch_m2 = float(np.square(centered).sum())

            running_count = int(self.count[cluster_int])
            if running_count == 0:
                self.count[cluster_int] = batch_count
                self.mean[cluster_int] = batch_mean
                self.m2[cluster_int] = batch_m2
                continue

            total_count = running_count + batch_count
            delta = batch_mean - self.mean[cluster_int]
            self.m2[cluster_int] = (
                self.m2[cluster_int]
                + batch_m2
                + float(np.square(delta).sum()) * running_count * batch_count / float(total_count)
            )
            self.mean[cluster_int] = self.mean[cluster_int] + delta * (batch_count / float(total_count))
            self.count[cluster_int] = total_count

    def summary(self) -> dict[str, object]:
        weighted_numerator = 0.0
        weighted_denominator = 0.0
        per_cluster: dict[str, dict[str, float | int]] = {}
        for cluster_id in range(self.cluster_count):
            count = int(self.count[cluster_id])
            if count <= 0:
                continue
            denom = float(count * self.target_dim)
            variance = float(self.m2[cluster_id] / denom)
            per_cluster[str(cluster_id)] = {
                "count": count,
                "mean_target_variance": variance,
            }
            weighted_numerator += float(self.m2[cluster_id])
            weighted_denominator += denom
        overall = float(weighted_numerator / weighted_denominator) if weighted_denominator > 0.0 else float("nan")
        return {
            "mean_discounted_future_field_variance": overall,
            "cluster_count_used": int(len(per_cluster)),
            "per_cluster": per_cluster,
        }


def _run_kmeans(features: np.ndarray, *, k: int, random_state: int) -> np.ndarray:
    estimator = KMeans(
        n_clusters=int(k),
        random_state=int(random_state),
        n_init=20,
    )
    return estimator.fit_predict(features).astype(np.int64, copy=False)


def _plot_selected_frame_snapshots(
    coords: np.ndarray,
    labels: np.ndarray,
    timesteps: np.ndarray,
    *,
    out_path: str | Path,
    title: str,
    max_frames: int,
) -> None:
    frame_count = int(coords.shape[0])
    snapshot_count = min(int(max_frames), frame_count)
    selected = np.linspace(0, frame_count - 1, num=snapshot_count, dtype=np.int64)
    ncols = min(3, snapshot_count)
    nrows = int(np.ceil(snapshot_count / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.8 * ncols, 4.6 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    cmap = plt.get_cmap("tab20")
    for ax, frame_idx in zip(axes_arr, selected.tolist(), strict=False):
        frame_labels = labels[int(frame_idx)]
        frame_coords = coords[int(frame_idx)]
        ax.scatter(
            frame_coords[:, 0],
            frame_coords[:, 1],
            s=8,
            c=cmap(frame_labels % 20),
            alpha=0.85,
            linewidths=0,
        )
        ax.set_title(f"frame={frame_idx} timestep={int(timesteps[int(frame_idx)])}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_arr[snapshot_count:]:
        ax.axis("off")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _build_frame_sample_indices(dataset, frame_to_slot: dict[int, int], frame_idx: int, atom_ids: np.ndarray) -> list[int]:
    atom_positions = np.searchsorted(dataset.atom_ids, atom_ids)
    if not np.array_equal(dataset.atom_ids[atom_positions], atom_ids):
        raise ValueError(
            "Artifact atom ids are not a subset of the lookup dataset atom ids. "
            f"frame_idx={frame_idx}."
        )
    frame_slot = frame_to_slot.get(int(frame_idx))
    if frame_slot is None:
        raise ValueError(f"Lookup dataset does not contain frame_idx={frame_idx}.")
    base = int(frame_slot) * int(dataset.center_count)
    return [base + int(atom_slot) for atom_slot in atom_positions.tolist()]


def _fetch_frame_neighborhoods(
    model: SuccessorVICRegModule,
    *,
    frame_idx: int,
    atom_ids: np.ndarray,
    atom_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = model._get_lookup_dataset()
    if model._lookup_frame_to_slot is None:
        raise RuntimeError("Lookup dataset frame-slot mapping is not available.")

    points_chunks = []
    atom_id_chunks = []
    for start in range(0, int(atom_ids.size), int(atom_chunk_size)):
        stop = min(int(atom_ids.size), start + int(atom_chunk_size))
        atom_chunk = np.asarray(atom_ids[start:stop], dtype=np.int64)
        sample_indices = _build_frame_sample_indices(
            dataset,
            model._lookup_frame_to_slot,
            int(frame_idx),
            atom_chunk,
        )
        batch = dataset.__getitems__(sample_indices)
        points = batch["points"]
        local_atom_ids = batch["local_atom_ids"]
        if not hasattr(points, "detach"):
            raise TypeError("Lookup dataset returned non-tensor points.")
        if points.ndim != 4 or points.shape[1] != 1 or points.shape[-1] != 3:
            raise ValueError(
                "Lookup dataset returned an unexpected neighborhood tensor shape. "
                f"Expected (B, 1, N, 3), got {tuple(points.shape)}."
            )
        if local_atom_ids.ndim != 3 or local_atom_ids.shape[1] != 1:
            raise ValueError(
                "Lookup dataset returned an unexpected local_atom_ids tensor shape. "
                f"Expected (B, 1, N), got {tuple(local_atom_ids.shape)}."
            )
        points_chunks.append(points[:, 0].detach().cpu().numpy().astype(np.float32, copy=False))
        atom_id_chunks.append(local_atom_ids[:, 0].detach().cpu().numpy().astype(np.int64, copy=False))
    return (
        np.concatenate(points_chunks, axis=0),
        np.concatenate(atom_id_chunks, axis=0),
    )


def _lookup_online_latents(
    model: SuccessorVICRegModule,
    required_pairs: list[tuple[int, int]],
    *,
    chunk_size: int,
) -> dict[tuple[int, int], np.ndarray]:
    if not required_pairs:
        return {}

    dataset = model._get_lookup_dataset()
    if model._lookup_frame_to_slot is None:
        raise RuntimeError("Lookup dataset frame-slot mapping is not available.")

    resolved: dict[tuple[int, int], np.ndarray] = {}
    for start in range(0, len(required_pairs), int(chunk_size)):
        chunk_pairs = required_pairs[start : start + int(chunk_size)]
        frame_indices = np.asarray([pair[0] for pair in chunk_pairs], dtype=np.int64)
        atom_ids = np.asarray([pair[1] for pair in chunk_pairs], dtype=np.int64)
        atom_positions = np.searchsorted(dataset.atom_ids, atom_ids)
        if not np.array_equal(dataset.atom_ids[atom_positions], atom_ids):
            raise ValueError(
                "Required neighborhood atom ids were not found in the lookup dataset. "
                f"missing_pairs={chunk_pairs[:8]}."
            )
        sample_indices = []
        for frame_idx, atom_slot in zip(frame_indices.tolist(), atom_positions.tolist(), strict=True):
            frame_slot = model._lookup_frame_to_slot.get(int(frame_idx))
            if frame_slot is None:
                raise ValueError(f"Lookup dataset does not contain frame_idx={frame_idx}.")
            sample_indices.append(int(frame_slot) * int(dataset.center_count) + int(atom_slot))
        fetched = dataset.__getitems__(sample_indices)
        points = fetched["points"]
        latents = model.encode_current_invariant(
            points[:, 0].to(device=model.device, dtype=model.dtype, non_blocking=True)
        )
        latents_np = latents.detach().cpu().numpy().astype(np.float32, copy=False)
        for latent_idx, pair in enumerate(chunk_pairs):
            resolved[(int(pair[0]), int(pair[1]))] = latents_np[latent_idx]
    return resolved


def _compute_future_field_for_frame(
    artifact: SuccessorEmbeddingsArtifact,
    model: SuccessorVICRegModule,
    *,
    frame_slot: int,
    atom_chunk_size: int,
    missing_latent_chunk_size: int,
) -> np.ndarray:
    frame_idx = int(artifact.frame_indices[frame_slot])
    frame_z = np.asarray(artifact.invariant_embeddings[frame_slot], dtype=np.float32)
    atom_ids = np.asarray(artifact.atom_ids, dtype=np.int64)
    atom_to_slot = {int(atom_id): int(slot) for slot, atom_id in enumerate(atom_ids.tolist())}
    latent_dim = int(frame_z.shape[1])
    future_field = np.empty((artifact.num_atoms, latent_dim), dtype=np.float32)

    _, local_atom_ids_frame = _fetch_frame_neighborhoods(
        model,
        frame_idx=frame_idx,
        atom_ids=atom_ids,
        atom_chunk_size=atom_chunk_size,
    )
    if local_atom_ids_frame.shape[0] != artifact.num_atoms:
        raise RuntimeError(
            "Fetched frame neighborhood count does not match the artifact atom count. "
            f"frame_idx={frame_idx}, expected={artifact.num_atoms}, got={local_atom_ids_frame.shape[0]}."
        )

    neighborhood_atom_ids_by_atom: list[list[int]] = []
    missing_pairs: set[tuple[int, int]] = set()
    for atom_slot in range(artifact.num_atoms):
        neighborhood_atom_ids = SuccessorVICRegModule._extract_neighborhood_atom_ids(
            local_atom_ids=torch.as_tensor(local_atom_ids_frame[atom_slot]),
        )
        neighborhood_atom_ids_by_atom.append(neighborhood_atom_ids)
        for neighbor_atom_id in neighborhood_atom_ids:
            if int(neighbor_atom_id) not in atom_to_slot:
                missing_pairs.add((frame_idx, int(neighbor_atom_id)))

    missing_latents = _lookup_online_latents(
        model,
        sorted(missing_pairs),
        chunk_size=missing_latent_chunk_size,
    )

    for atom_slot in range(artifact.num_atoms):
        neighborhood_atom_ids = neighborhood_atom_ids_by_atom[atom_slot]
        neighborhood_latents = np.stack(
            [
                frame_z[atom_to_slot[int(neighbor_atom_id)]]
                if int(neighbor_atom_id) in atom_to_slot
                else missing_latents[(frame_idx, int(neighbor_atom_id))]
                for neighbor_atom_id in neighborhood_atom_ids
            ],
            axis=0,
        )
        future_field[atom_slot] = np.mean(neighborhood_latents, axis=0, dtype=np.float32)
    return future_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster Successor-VICReg embeddings and compare raw z_inv against successor "
            "embeddings using a future-field variance metric."
        )
    )
    parser.add_argument("artifact", help="Successor embedding artifact NPZ.")
    parser.add_argument("--output-dir", default=None, help="Analysis output directory.")
    parser.add_argument("--k", type=int, default=8, help="KMeans cluster count.")
    parser.add_argument("--random-state", type=int, default=0, help="KMeans random seed.")
    parser.add_argument("--l2-normalize", action="store_true", help="L2-normalize features before clustering.")
    parser.add_argument("--standardize", action="store_true", help="Standardize features before clustering.")
    parser.add_argument("--pca-variance", type=float, default=0.98, help="PCA explained-variance target.")
    parser.add_argument("--pca-max-components", type=int, default=64, help="Maximum PCA components.")
    parser.add_argument("--snapshot-count", type=int, default=6, help="Number of snapshot frames to render.")
    parser.add_argument(
        "--feature-sets",
        default="raw_z,successor",
        help="Comma-separated subset of feature sets to cluster: raw_z,successor,concat.",
    )
    parser.add_argument(
        "--atom-chunk-size",
        type=int,
        default=1024,
        help="Per-frame neighborhood fetch chunk size.",
    )
    parser.add_argument(
        "--missing-latent-chunk-size",
        type=int,
        default=2048,
        help="Chunk size when encoding neighborhood latents missing from the artifact.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index for any missing-neighbor encoder calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = SuccessorEmbeddingsArtifact.load(args.artifact)
    if args.output_dir is None:
        artifact_path = Path(artifact.metadata.get("npz_path", args.artifact)).expanduser().resolve()
        output_dir = ensure_dir(artifact_path.parent / "analysis")
    else:
        output_dir = ensure_dir(args.output_dir)

    feature_names = [name.strip().lower() for name in str(args.feature_sets).split(",") if name.strip()]
    requested_feature_names = list(dict.fromkeys(feature_names))
    if not requested_feature_names:
        raise ValueError("At least one feature set must be selected for analysis.")

    available_feature_sets = {
        "raw_z": np.asarray(artifact.invariant_embeddings, dtype=np.float32),
        "successor": np.asarray(artifact.successor_embeddings, dtype=np.float32),
    }
    if bool(artifact.metadata.get("successor_use_concat_z_and_successor_for_clustering", False)) or "concat" in requested_feature_names:
        available_feature_sets["concat"] = np.concatenate(
            (
                np.asarray(artifact.invariant_embeddings, dtype=np.float32),
                np.asarray(artifact.successor_embeddings, dtype=np.float32),
            ),
            axis=2,
        )

    cluster_labels_by_feature: dict[str, np.ndarray] = {}
    cluster_summary: dict[str, object] = {}
    for feature_name in requested_feature_names:
        if feature_name not in available_feature_sets:
            raise ValueError(
                f"Unknown feature set {feature_name!r}. Available={sorted(available_feature_sets)}."
            )
        feature_grid = available_feature_sets[feature_name]
        flat_features = feature_grid.reshape(-1, feature_grid.shape[2])
        prepared_features, prep_info = prepare_clustering_features(
            flat_features,
            random_state=int(args.random_state),
            l2_normalize=bool(args.l2_normalize),
            standardize=bool(args.standardize),
            pca_variance=float(args.pca_variance),
            pca_max_components=int(args.pca_max_components),
        )
        labels = _run_kmeans(
            prepared_features,
            k=int(args.k),
            random_state=int(args.random_state),
        )
        labels_grid = labels.reshape(feature_grid.shape[:2])
        cluster_labels_by_feature[feature_name] = labels_grid
        cluster_summary[feature_name] = {
            "feature_dim": int(feature_grid.shape[2]),
            "prepared_dim": int(prepared_features.shape[1]),
            "cluster_count": int(args.k),
            "prep_info": prep_info,
        }
        _plot_selected_frame_snapshots(
            artifact.center_positions,
            labels_grid,
            artifact.timesteps,
            out_path=output_dir / f"{feature_name}_cluster_snapshots.png",
            title=f"{feature_name} clustering snapshots",
            max_frames=int(args.snapshot_count),
        )

    labels_payload = {
        feature_name: labels.astype(np.int16, copy=False)
        for feature_name, labels in cluster_labels_by_feature.items()
    }
    labels_payload["frame_indices"] = np.asarray(artifact.frame_indices, dtype=np.int64)
    labels_payload["atom_ids"] = np.asarray(artifact.atom_ids, dtype=np.int64)
    labels_payload["timesteps"] = np.asarray(artifact.timesteps, dtype=np.int64)
    labels_path = output_dir / "cluster_labels.npz"
    np.savez_compressed(labels_path, **labels_payload)

    checkpoint_path = artifact.metadata.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError(
            "Successor embedding artifact metadata is missing checkpoint_path, "
            "so future-field metrics cannot be computed."
        )
    model, _, _ = load_successor_checkpoint(
        checkpoint_path,
        cuda_device=int(args.cuda_device),
    )
    model.eval()

    frame_count = int(artifact.frame_count)
    horizon = int(artifact.metadata.get("successor_horizon_H", 0))
    gamma = float(artifact.metadata.get("successor_gamma", 0.0))
    frame_step = int(artifact.metadata.get("training_frame_stride", 1))
    if horizon <= 0:
        raise ValueError(
            "Artifact metadata must provide successor_horizon_H > 0 for future-field analysis, "
            f"got {horizon}."
        )
    if frame_step <= 0:
        raise ValueError(f"training_frame_stride must be > 0, got {frame_step}.")
    max_future_offset = horizon * frame_step
    if frame_count <= max_future_offset:
        raise ValueError(
            "Trajectory is too short for the configured successor horizon. "
            f"frame_count={frame_count}, successor_horizon_H={horizon}, "
            f"training_frame_stride={frame_step}."
        )
    if abs(gamma - float(model.successor_gamma)) > 1e-8:
        log_progress(
            "analyze_successor_embeddings",
            (
                "artifact successor_gamma differs from the restored checkpoint; "
                f"artifact={gamma}, checkpoint={model.successor_gamma}. Using the artifact value."
            ),
        )

    accumulators = {
        feature_name: ClusterTargetVarianceAccumulator(
            cluster_count=int(args.k),
            target_dim=int(artifact.successor_dim),
        )
        for feature_name in requested_feature_names
    }

    future_field_queue: deque[np.ndarray] = deque()
    discount_weights = np.asarray(
        [gamma ** delta_idx for delta_idx in range(horizon)],
        dtype=np.float32,
    )

    log_progress(
        "analyze_successor_embeddings",
        (
            f"computing future-field variance metric over {frame_count - max_future_offset} valid "
            f"frame slices (frame_step={frame_step})"
        ),
    )
    for reverse_frame_slot in range(frame_count - 1, -1, -1):
        future_field = _compute_future_field_for_frame(
            artifact,
            model,
            frame_slot=reverse_frame_slot,
            atom_chunk_size=int(args.atom_chunk_size),
            missing_latent_chunk_size=int(args.missing_latent_chunk_size),
        )
        if len(future_field_queue) >= max_future_offset:
            target = np.zeros_like(future_field_queue[0], dtype=np.float32)
            for delta_idx in range(horizon):
                queue_index = ((delta_idx + 1) * frame_step) - 1
                target += discount_weights[delta_idx] * future_field_queue[queue_index]
            for feature_name in requested_feature_names:
                labels_frame = cluster_labels_by_feature[feature_name][reverse_frame_slot]
                accumulators[feature_name].update(labels_frame, target)
        future_field_queue.appendleft(future_field)
        if len(future_field_queue) > max_future_offset:
            future_field_queue.pop()

    future_metric_summary = {
        feature_name: accumulator.summary()
        for feature_name, accumulator in accumulators.items()
    }

    summary = {
        "artifact_path": str(Path(args.artifact).expanduser().resolve()),
        "labels_path": str(labels_path),
        "frame_count": int(artifact.frame_count),
        "num_atoms": int(artifact.num_atoms),
        "invariant_dim": int(artifact.invariant_dim),
        "successor_dim": int(artifact.successor_dim),
        "successor_horizon_H": int(horizon),
        "successor_gamma": float(gamma),
        "training_frame_stride": int(frame_step),
        "k": int(args.k),
        "random_state": int(args.random_state),
        "feature_sets": cluster_summary,
        "future_field_metrics": future_metric_summary,
        "notes": [
            "The future-field metric is the mean within-cluster variance of the discounted future local field r.",
            "The future field is the mean latent over the full fixed-size local neighborhood at the target frame.",
            "When the embedding artifact does not include every neighborhood atom, missing latents are recomputed from the checkpoint on demand.",
        ],
    }
    summary_path = save_json(_to_jsonable(summary), output_dir / "summary.json")
    log_progress(
        "analyze_successor_embeddings",
        f"saved analysis summary to {summary_path}",
    )


if __name__ == "__main__":
    main()
