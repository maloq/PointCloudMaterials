from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from src.data_utils.data_load import PointCloudDataset
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.training_methods.spd.rot_heads import sixd_to_so3
from src.utils.spd_metrics import random_rotation_matrix
from src.utils.spd_utils import apply_rotation


def _normalize_channels(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def _channel_norm_weights(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1)
    return np.clip(norms, eps, None)


def _kabsch_rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal rotation aligning source vectors to target vectors (C x 3)."""
    cov = source.T @ target
    U, _, Vt = np.linalg.svd(cov, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def _rotation_angle_deg(rot: np.ndarray, eps: float = 1e-7) -> float:
    trace = np.trace(rot)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return float(np.degrees(np.arccos(cos_theta)))


def _rotation_geodesic_angle_rad(R1: np.ndarray, R2: np.ndarray, eps: float = 1e-7) -> float:
    trace = np.trace(R1.T @ R2)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return float(np.arccos(cos_theta))


def _random_rotation_matrices(num: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(num, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.empty((num, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _rotation_geodesic_angles_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = pred.transpose(-1, -2) @ target
    trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = 0.5 * (trace - 1.0)
    cos_clamped = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.arccos(cos_clamped)
    clamped = cos_clamped != cos_theta
    return angle, clamped


def kabsch_rotation_sanity_check(
    num_channels: int = 16,
    seed: int = 0,
) -> float:
    """Return geodesic angle (deg) between recovered and true rotation."""
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(num_channels, 3))
    R_true = _random_rotation_matrices(1, seed=seed + 1)[0]
    T = V @ R_true
    R_hat = _kabsch_rotation_from_vectors(V, T)
    return _rotation_angle_deg(R_hat.T @ R_true)


def _cluster_rotations_geodesic(rotations: np.ndarray, threshold_rad: float) -> np.ndarray:
    if rotations.size == 0:
        return np.empty((0,), dtype=int)
    labels = np.full((len(rotations),), -1, dtype=int)
    centers: list[np.ndarray] = []
    for idx, rot in enumerate(rotations):
        assigned = False
        for c_idx, center in enumerate(centers):
            if _rotation_geodesic_angle_rad(rot, center) <= threshold_rad:
                labels[idx] = c_idx
                assigned = True
                break
        if not assigned:
            centers.append(rot)
            labels[idx] = len(centers) - 1
    return labels


def compute_cluster_prototypes_and_alignments(
    eq_latents: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    n_iters: int = 5,
    min_cluster_size: int = 3,
    normalize_channels: bool = False,
    channel_weighting: str = "norm",
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Estimate per-cluster canonical equivariant prototypes and per-sample alignments."""
    if eq_latents.size == 0 or cluster_labels.size != len(eq_latents):
        return {
            "proto_eq": np.empty((0,)),
            "R_align": np.empty((0, 3, 3)),
            "residuals": np.empty((0,)),
            "cluster_sizes": np.empty((0,), dtype=int),
        }

    eq_latents = np.asarray(eq_latents)
    cluster_labels = np.asarray(cluster_labels).astype(int)
    num_samples = eq_latents.shape[0]
    num_channels = eq_latents.shape[1]
    num_clusters = int(cluster_labels.max()) + 1

    proto_eq = np.zeros((num_clusters, num_channels, 3), dtype=np.float32)
    R_align = np.tile(np.eye(3, dtype=np.float32), (num_samples, 1, 1))
    residuals = np.full((num_samples,), np.nan, dtype=np.float32)
    cluster_sizes = np.zeros((num_clusters,), dtype=int)

    for k in range(num_clusters):
        idx = np.where(cluster_labels == k)[0]
        cluster_sizes[k] = len(idx)
        if len(idx) == 0:
            continue

        V = eq_latents[idx].astype(np.float64)
        V_use = _normalize_channels(V, eps=eps) if normalize_channels else V
        proto = V_use[0].copy()

        if len(idx) >= min_cluster_size:
            for _ in range(max(1, int(n_iters))):
                aligned = np.empty_like(V_use)
                for j in range(len(idx)):
                    if channel_weighting == "norm":
                        weights = _channel_norm_weights(proto, eps=eps)
                        source = V_use[j] * weights[:, None]
                        target = proto * weights[:, None]
                    else:
                        source = V_use[j]
                        target = proto
                    R = _kabsch_rotation_from_vectors(source, target)
                    aligned[j] = V_use[j] @ R
                proto = aligned.mean(axis=0)
                if normalize_channels:
                    proto = _normalize_channels(proto, eps=eps)

            for j, sample_idx in enumerate(idx):
                if channel_weighting == "norm":
                    weights = _channel_norm_weights(proto, eps=eps)
                    source = V_use[j] * weights[:, None]
                    target = proto * weights[:, None]
                else:
                    source = V_use[j]
                    target = proto
                R = _kabsch_rotation_from_vectors(source, target)
                R_align[sample_idx] = R.astype(np.float32)
                aligned = V_use[j] @ R
                residuals[sample_idx] = float(
                    np.linalg.norm(aligned - proto) / (np.linalg.norm(proto) + eps)
                )
            proto_eq[k] = proto.astype(np.float32)
        else:
            proto = V_use.mean(axis=0)
            if normalize_channels:
                proto = _normalize_channels(proto, eps=eps)
            proto_eq[k] = proto.astype(np.float32)
            for j, sample_idx in enumerate(idx):
                aligned = V_use[j]
                residuals[sample_idx] = float(
                    np.linalg.norm(aligned - proto) / (np.linalg.norm(proto) + eps)
                )

    return {
        "proto_eq": proto_eq,
        "R_align": R_align,
        "residuals": residuals,
        "cluster_sizes": cluster_sizes,
    }


def _rotation_head_align_to_proto(
    model: BarlowTwinsModule,
    samples: torch.Tensor,
    proto: torch.Tensor,
    *,
    batch_size: int = 4096,
) -> torch.Tensor:
    pose_head = getattr(model, "pose_head", None)
    if pose_head is None:
        raise ValueError("Rotation-head alignment requested, but model.pose_head is missing.")
    if samples.numel() == 0:
        return torch.empty((0, 3, 3), device=samples.device, dtype=samples.dtype)

    rots = []
    proto_b = proto.unsqueeze(0)
    step = max(1, int(batch_size))
    for start in range(0, int(samples.shape[0]), step):
        stop = min(int(samples.shape[0]), start + step)
        chunk = samples[start:stop]
        proto_chunk = proto_b.expand(chunk.shape[0], -1, -1)
        # Match training convention: cov(target, source) -> R_source->target.
        cov = torch.einsum("bci,bcj->bij", proto_chunk, chunk)
        r6 = pose_head(cov)
        rots.append(sixd_to_so3(r6, eps=1e-6))
    return torch.cat(rots, dim=0)


def compute_cluster_prototypes_and_alignments_with_rotation_head(
    model: BarlowTwinsModule,
    eq_latents: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    n_iters: int = 5,
    min_cluster_size: int = 3,
    normalize_channels: bool = False,
    eps: float = 1e-6,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Head-based version of orientation analysis using `pose_head` alignments."""
    if eq_latents.size == 0 or cluster_labels.size != len(eq_latents):
        return {
            "proto_eq": np.empty((0,)),
            "R_align": np.empty((0, 3, 3)),
            "residuals": np.empty((0,)),
            "cluster_sizes": np.empty((0,), dtype=int),
        }

    pose_head = getattr(model, "pose_head", None)
    if pose_head is None:
        raise ValueError("Rotation-head alignment requested, but model.pose_head is missing.")

    eq_latents = np.asarray(eq_latents)
    cluster_labels = np.asarray(cluster_labels).astype(int)
    num_samples = eq_latents.shape[0]
    num_clusters = int(cluster_labels.max()) + 1

    if num_clusters <= 0:
        return {
            "proto_eq": np.empty((0,)),
            "R_align": np.empty((0, 3, 3)),
            "residuals": np.empty((0,)),
            "cluster_sizes": np.empty((0,), dtype=int),
        }

    device = model.device
    with torch.inference_mode():
        eq_tensor = torch.as_tensor(eq_latents, device=device, dtype=torch.float32)
        if hasattr(model, "_prepare_eq_latent"):
            eq_tensor = model._prepare_eq_latent(eq_tensor)
        if eq_tensor is None or eq_tensor.dim() != 3 or eq_tensor.shape[-1] != 3:
            raise ValueError(
                f"Expected equivariant latents shaped (N,C,3) after preparation. "
                f"Got {None if eq_tensor is None else tuple(eq_tensor.shape)}"
            )

        num_channels = int(eq_tensor.shape[1])
        proto_eq = np.zeros((num_clusters, num_channels, 3), dtype=np.float32)
        R_align = np.tile(np.eye(3, dtype=np.float32), (num_samples, 1, 1))
        residuals = np.full((num_samples,), np.nan, dtype=np.float32)
        cluster_sizes = np.zeros((num_clusters,), dtype=int)

        for k in range(num_clusters):
            idx = np.where(cluster_labels == k)[0]
            cluster_sizes[k] = len(idx)
            if len(idx) == 0:
                continue

            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            V = eq_tensor.index_select(0, idx_t)
            if normalize_channels:
                V = V / torch.linalg.norm(V, dim=-1, keepdim=True).clamp_min(float(eps))
            proto = V[0].clone()

            if len(idx) >= min_cluster_size:
                for _ in range(max(1, int(n_iters))):
                    rot = _rotation_head_align_to_proto(
                        model, V, proto, batch_size=batch_size
                    )
                    aligned = apply_rotation(V, rot)
                    proto = aligned.mean(dim=0)
                    if normalize_channels:
                        proto = proto / torch.linalg.norm(proto, dim=-1, keepdim=True).clamp_min(
                            float(eps)
                        )

                rot = _rotation_head_align_to_proto(
                    model, V, proto, batch_size=batch_size
                )
                aligned = apply_rotation(V, rot)
                denom = torch.linalg.norm(proto).clamp_min(float(eps))
                res = torch.linalg.norm(aligned - proto.unsqueeze(0), dim=(1, 2)) / denom

                R_align[idx] = rot.detach().cpu().numpy().astype(np.float32)
                residuals[idx] = res.detach().cpu().numpy().astype(np.float32)
                proto_eq[k] = proto.detach().cpu().numpy().astype(np.float32)
            else:
                proto = V.mean(dim=0)
                if normalize_channels:
                    proto = proto / torch.linalg.norm(proto, dim=-1, keepdim=True).clamp_min(
                        float(eps)
                    )
                diff = V - proto.unsqueeze(0)
                denom = torch.linalg.norm(proto).clamp_min(float(eps))
                res = torch.linalg.norm(diff, dim=(1, 2)) / denom
                residuals[idx] = res.detach().cpu().numpy().astype(np.float32)
                proto_eq[k] = proto.detach().cpu().numpy().astype(np.float32)

    return {
        "proto_eq": proto_eq,
        "R_align": R_align,
        "residuals": residuals,
        "cluster_sizes": cluster_sizes,
    }


def compute_self_symmetry_metrics(
    eq_latents: np.ndarray,
    *,
    num_probes: int = 64,
    beta: float = 10.0,
    delta: float = 0.05,
    mode_threshold_deg: float = 15.0,
    max_samples: int | None = None,
    seed: int = 0,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Estimate per-sample orientation ambiguity via random probe rotations."""
    if eq_latents.size == 0:
        return {
            "e_min": np.empty((0,)),
            "e_p10": np.empty((0,)),
            "e_median": np.empty((0,)),
            "entropy": np.empty((0,)),
            "n_eff": np.empty((0,)),
            "n_modes": np.empty((0,)),
            "mode_width_deg": np.empty((0,)),
            "computed_mask": np.empty((0,), dtype=bool),
        }

    eq_latents = np.asarray(eq_latents)
    num_samples = eq_latents.shape[0]
    probe_rots = _random_rotation_matrices(int(num_probes), seed=seed)
    threshold_rad = np.deg2rad(mode_threshold_deg)

    e_min = np.full((num_samples,), np.nan, dtype=np.float32)
    e_p10 = np.full((num_samples,), np.nan, dtype=np.float32)
    e_median = np.full((num_samples,), np.nan, dtype=np.float32)
    entropy = np.full((num_samples,), np.nan, dtype=np.float32)
    n_eff = np.full((num_samples,), np.nan, dtype=np.float32)
    n_modes = np.full((num_samples,), np.nan, dtype=np.float32)
    mode_width = np.full((num_samples,), np.nan, dtype=np.float32)

    compute_idx = _sample_indices(num_samples, max_samples)
    computed_mask = np.zeros((num_samples,), dtype=bool)

    for idx in compute_idx:
        V = eq_latents[idx].astype(np.float64)
        S = V.T @ V
        norm_S = np.linalg.norm(S)
        if norm_S < eps:
            e_vals = np.zeros((num_probes,), dtype=np.float64)
            p = np.full((num_probes,), 1.0 / num_probes, dtype=np.float64)
            e_min[idx] = 0.0
            e_p10[idx] = 0.0
            e_median[idx] = 0.0
            entropy[idx] = float(-np.sum(p * np.log(p + eps)))
            n_eff[idx] = float(np.exp(entropy[idx]))
            n_modes[idx] = 1.0
            mode_width[idx] = 180.0
            computed_mask[idx] = True
            continue

        QS = probe_rots @ S
        QSQT = QS @ np.transpose(probe_rots, (0, 2, 1))
        diff = QSQT - S
        e_vals = np.linalg.norm(diff.reshape(num_probes, -1), axis=1) / (norm_S + eps)

        e_min_val = float(np.min(e_vals))
        e_min[idx] = e_min_val
        e_p10[idx] = float(np.percentile(e_vals, 10))
        e_median[idx] = float(np.median(e_vals))

        weights = np.exp(-beta * e_vals)
        w_sum = float(weights.sum())
        if not np.isfinite(w_sum) or w_sum <= 0:
            weights = np.ones_like(weights)
            w_sum = float(weights.sum())
        probs = weights / w_sum
        entropy[idx] = float(-np.sum(probs * np.log(probs + eps)))
        n_eff[idx] = float(np.exp(entropy[idx]))

        thresh = min(e_min_val + delta, float(np.percentile(e_vals, 5)))
        sel = np.where(e_vals <= thresh)[0]
        if sel.size == 0:
            sel = np.array([int(np.argmin(e_vals))])
        rot_sel = probe_rots[sel]
        labels = _cluster_rotations_geodesic(rot_sel, threshold_rad=threshold_rad)
        n_modes[idx] = float(len(np.unique(labels)))
        if len(labels) == 0:
            mode_width[idx] = np.nan
        else:
            widths = []
            for mode in np.unique(labels):
                members = rot_sel[labels == mode]
                center = members[0]
                if len(members) == 1:
                    widths.append(0.0)
                    continue
                dists = [
                    _rotation_geodesic_angle_rad(member, center)
                    for member in members
                ]
                widths.append(float(np.mean(dists)))
            mode_width[idx] = float(np.degrees(np.mean(widths)))
        computed_mask[idx] = True

    return {
        "e_min": e_min,
        "e_p10": e_p10,
        "e_median": e_median,
        "entropy": entropy,
        "n_eff": n_eff,
        "n_modes": n_modes,
        "mode_width_deg": mode_width,
        "computed_mask": computed_mask,
    }


def cluster_orientations_by_cluster(
    rot_mats: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    threshold_deg: float = 15.0,
) -> np.ndarray:
    if rot_mats.size == 0 or cluster_labels.size != len(rot_mats):
        return np.empty((0,), dtype=int)
    rot_mats = np.asarray(rot_mats)
    cluster_labels = np.asarray(cluster_labels).astype(int)
    unique_clusters = np.unique(cluster_labels)
    labels_out = np.full((len(rot_mats),), -1, dtype=int)
    offset = 0
    threshold_rad = np.deg2rad(threshold_deg)
    for cluster in unique_clusters:
        idx = np.where(cluster_labels == cluster)[0]
        if idx.size == 0:
            continue
        labels = _cluster_rotations_geodesic(rot_mats[idx], threshold_rad=threshold_rad)
        if labels.size == 0:
            labels_out[idx] = offset
            offset += 1
        else:
            labels_out[idx] = labels + offset
            offset += int(labels.max()) + 1
    return labels_out


def cap_cluster_labels(
    labels: np.ndarray,
    *,
    max_clusters: int,
    other_label: int = -1,
) -> np.ndarray:
    """Keep the largest `max_clusters` labels; collapse the rest to `other_label`."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= max_clusters:
        return labels
    order = np.argsort(counts)[::-1]
    keep = set(unique[order[:max_clusters]].tolist())
    capped = labels.copy()
    mask = ~np.isin(capped, list(keep))
    capped[mask] = other_label
    return capped


def _build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    num_nodes = int(coords.shape[0])
    if num_nodes < 2:
        return np.empty((0, 2), dtype=np.int64)

    k_eff = max(1, min(int(k), num_nodes - 1))
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required for kNN-based grain segmentation.") from exc

    tree = cKDTree(coords)
    _, nn_idx = tree.query(coords, k=k_eff + 1)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    edge_set: set[tuple[int, int]] = set()
    for i in range(num_nodes):
        for j in nn_idx[i, 1:]:
            j = int(j)
            if j < 0 or j >= num_nodes or j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

    if not edge_set:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(edge_set), dtype=np.int64)


def _rotation_geodesic_angles_np(
    rot_a: np.ndarray,
    rot_b: np.ndarray,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    if rot_a.size == 0 or rot_b.size == 0:
        return np.empty((0,), dtype=np.float32)
    delta = np.matmul(np.transpose(rot_a, (0, 2, 1)), rot_b)
    trace = np.trace(delta, axis1=1, axis2=2)
    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return np.arccos(cos_theta).astype(np.float32)


def _connected_components_from_edges(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    if num_nodes <= 0:
        return np.empty((0,), dtype=int)

    parent = np.arange(num_nodes, dtype=np.int64)
    rank = np.zeros(num_nodes, dtype=np.int8)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    if edges.size:
        for a, b in np.asarray(edges, dtype=np.int64):
            ra = _find(int(a))
            rb = _find(int(b))
            if ra == rb:
                continue
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

    roots = np.empty((num_nodes,), dtype=np.int64)
    for idx in range(num_nodes):
        roots[idx] = _find(idx)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


def _relabel_nonnegative(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int, copy=False)
    out = np.full(labels.shape, -1, dtype=int)
    valid_mask = labels >= 0
    if not valid_mask.any():
        return out
    unique = np.unique(labels[valid_mask])
    remap = {int(old): int(new) for new, old in enumerate(unique.tolist())}
    out_valid = np.fromiter((remap[int(v)] for v in labels[valid_mask]), dtype=int)
    out[valid_mask] = out_valid
    return out


def segment_grains_with_pose_head(
    model: BarlowTwinsModule,
    inv_latents: np.ndarray,
    eq_latents: np.ndarray,
    motif_labels: np.ndarray,
    coords: np.ndarray,
    *,
    knn_k: int = 12,
    edge_weight_threshold: float = 0.35,
    orientation_tau_deg: float = 18.0,
    alpha_scale_quantile: float = 0.75,
    align_n_iters: int = 5,
    align_min_cluster_size: int = 3,
    align_normalize_channels: bool = True,
    min_grain_size: int = 1,
) -> Dict[str, Any]:
    """Orientation-aware grain segmentation with motif boundary preservation."""
    inv_latents = np.asarray(inv_latents)
    eq_latents = np.asarray(eq_latents)
    motif_labels = np.asarray(motif_labels).astype(int, copy=False)
    coords = np.asarray(coords, dtype=np.float32)

    num_nodes = int(inv_latents.shape[0]) if inv_latents.ndim >= 1 else 0
    empty = {
        "grain_labels": np.empty((0,), dtype=int),
        "alpha": np.empty((0,), dtype=np.float32),
        "orientation_residuals": np.empty((0,), dtype=np.float32),
        "metrics": {
            "num_nodes": int(num_nodes),
            "num_grains": 0,
            "status": "empty",
        },
    }
    if num_nodes < 2:
        return empty
    if inv_latents.ndim != 2:
        inv_latents = np.reshape(inv_latents, (num_nodes, -1))
    if coords.ndim != 2 or coords.shape[1] < 3:
        return {
            **empty,
            "metrics": {
                "num_nodes": int(num_nodes),
                "num_grains": 0,
                "status": "invalid_coords",
            },
        }
    if motif_labels.shape[0] != num_nodes or coords.shape[0] != num_nodes:
        return {
            **empty,
            "metrics": {
                "num_nodes": int(num_nodes),
                "num_grains": 0,
                "status": "shape_mismatch",
            },
        }

    coords3 = coords[:, :3].astype(np.float32, copy=False)
    edges = _build_knn_edges(coords3, k=knn_k)
    if edges.size == 0:
        singleton_labels = np.arange(num_nodes, dtype=int)
        return {
            "grain_labels": singleton_labels,
            "alpha": np.ones((num_nodes,), dtype=np.float32),
            "orientation_residuals": np.full((num_nodes,), np.nan, dtype=np.float32),
            "metrics": {
                "num_nodes": int(num_nodes),
                "num_edges_total": 0,
                "num_grains": int(num_nodes),
                "status": "no_edges",
            },
        }

    orientation_available = bool(
        getattr(model, "pose_head", None) is not None
        and eq_latents.ndim == 3
        and eq_latents.shape[0] == num_nodes
        and eq_latents.shape[-1] == 3
    )
    alpha = np.ones((num_nodes,), dtype=np.float32)
    residuals = np.full((num_nodes,), np.nan, dtype=np.float32)
    rot_align = np.tile(np.eye(3, dtype=np.float32), (num_nodes, 1, 1))

    if orientation_available:
        try:
            align = compute_cluster_prototypes_and_alignments_with_rotation_head(
                model,
                eq_latents,
                motif_labels,
                n_iters=max(1, int(align_n_iters)),
                min_cluster_size=max(2, int(align_min_cluster_size)),
                normalize_channels=bool(align_normalize_channels),
            )
            rot_candidate = np.asarray(align.get("R_align", rot_align), dtype=np.float32)
            if rot_candidate.shape == (num_nodes, 3, 3):
                rot_align = rot_candidate
            else:
                orientation_available = False

            res_candidate = np.asarray(
                align.get("residuals", residuals),
                dtype=np.float32,
            )
            if res_candidate.shape == (num_nodes,):
                residuals = res_candidate

            finite_res = residuals[np.isfinite(residuals)]
            q = float(np.clip(alpha_scale_quantile, 0.05, 0.95))
            if finite_res.size:
                scale = float(np.quantile(finite_res, q))
                scale = max(scale, 1e-6)
                scaled = np.clip(residuals / scale, 0.0, 10.0)
                alpha = np.exp(-(scaled ** 2)).astype(np.float32)
                alpha[~np.isfinite(alpha)] = 0.0
            else:
                alpha = np.zeros((num_nodes,), dtype=np.float32)
        except Exception:
            orientation_available = False
            alpha = np.ones((num_nodes,), dtype=np.float32)
            residuals = np.full((num_nodes,), np.nan, dtype=np.float32)
            rot_align = np.tile(np.eye(3, dtype=np.float32), (num_nodes, 1, 1))

    src = edges[:, 0]
    dst = edges[:, 1]
    same_motif = motif_labels[src] == motif_labels[dst]

    dz = inv_latents[src] - inv_latents[dst]
    dz2 = np.einsum("ij,ij->i", dz, dz).astype(np.float32)
    dz_ref = dz2[same_motif] if same_motif.any() else dz2
    dz_pos = dz_ref[dz_ref > 0]
    if dz_pos.size:
        sigma2 = float(np.median(dz_pos))
    elif dz_ref.size:
        sigma2 = float(np.mean(dz_ref))
    else:
        sigma2 = 1.0
    sigma2 = max(sigma2, 1e-8)
    motif_weight = np.exp(-dz2 / sigma2).astype(np.float32)

    theta = np.zeros((len(edges),), dtype=np.float32)
    if orientation_available:
        theta = _rotation_geodesic_angles_np(rot_align[src], rot_align[dst])
    tau_rad = max(np.deg2rad(max(1e-3, float(orientation_tau_deg))), 1e-6)
    orient_weight = np.exp(
        -((alpha[src] * alpha[dst]) * (theta ** 2)) / (tau_rad * tau_rad)
    ).astype(np.float32)

    edge_weight = motif_weight * orient_weight
    threshold = float(np.clip(edge_weight_threshold, 0.0, 1.0))
    keep_mask = same_motif & (edge_weight >= threshold)
    kept_edges = edges[keep_mask]

    grain_labels = _connected_components_from_edges(num_nodes, kept_edges)
    min_grain = max(1, int(min_grain_size))
    if min_grain > 1:
        unique, counts = np.unique(grain_labels, return_counts=True)
        small_labels = unique[counts < min_grain]
        if small_labels.size:
            small_mask = np.isin(grain_labels, small_labels)
            grain_labels = grain_labels.astype(int, copy=True)
            grain_labels[small_mask] = -1
    grain_labels = _relabel_nonnegative(grain_labels)

    valid_grains = grain_labels[grain_labels >= 0]
    grain_sizes = (
        np.unique(valid_grains, return_counts=True)[1]
        if valid_grains.size
        else np.empty((0,), dtype=int)
    )

    metrics: Dict[str, Any] = {
        "status": "ok",
        "num_nodes": int(num_nodes),
        "knn_k": int(max(1, min(int(knn_k), num_nodes - 1))),
        "num_edges_total": int(edges.shape[0]),
        "num_edges_same_motif": int(np.sum(same_motif)),
        "num_edges_kept": int(np.sum(keep_mask)),
        "edge_keep_fraction": float(np.mean(keep_mask)) if keep_mask.size else 0.0,
        "edge_weight_threshold": float(threshold),
        "sigma_latent": float(np.sqrt(sigma2)),
        "orientation_tau_deg": float(orientation_tau_deg),
        "orientation_available": float(1.0 if orientation_available else 0.0),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_median": float(np.median(alpha)),
        "num_grains": int(len(np.unique(valid_grains))),
        "grain_size_min": int(np.min(grain_sizes)) if grain_sizes.size else 0,
        "grain_size_median": float(np.median(grain_sizes)) if grain_sizes.size else 0.0,
        "grain_size_max": int(np.max(grain_sizes)) if grain_sizes.size else 0,
    }
    if orientation_available and theta.size:
        metrics["edge_theta_mean_deg"] = float(np.degrees(np.mean(theta)))
        metrics["edge_theta_median_deg"] = float(np.degrees(np.median(theta)))
    finite_res = residuals[np.isfinite(residuals)]
    if finite_res.size:
        metrics["orientation_residual_mean"] = float(np.mean(finite_res))
        metrics["orientation_residual_median"] = float(np.median(finite_res))

    return {
        "grain_labels": grain_labels.astype(int, copy=False),
        "alpha": alpha.astype(np.float32, copy=False),
        "orientation_residuals": residuals.astype(np.float32, copy=False),
        "metrics": metrics,
    }


def _looks_like_coords(value: Any) -> bool:
    if value is None:
        return False
    if torch.is_tensor(value):
        if value.ndim == 1 and value.shape[0] == 3:
            return True
        if value.ndim == 2 and value.shape[1] == 3:
            return True
        return False
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.shape[0] == 3:
            return True
        if value.ndim == 2 and value.shape[1] == 3:
            return True
    return False


def _extract_pc_phase_coords(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(batch, dict):
        pc = batch["points"]
        phase = batch.get("class_id", None)
        coords = batch.get("coords", None)
    elif isinstance(batch, (tuple, list)):
        pc = batch[0]
        phase = None
        coords = None
        if len(batch) > 1:
            second = batch[1]
            if _looks_like_coords(second):
                coords = second
            else:
                phase = second
        if len(batch) > 2:
            third = batch[2]
            if coords is None and _looks_like_coords(third):
                coords = third
            elif phase is None:
                phase = third
    else:
        pc = batch
        phase = None
        coords = None
    if phase is not None and not torch.is_tensor(phase):
        phase = torch.as_tensor(phase)
    if coords is not None and not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)
    return pc, phase, coords


def _extract_pc_and_phase(batch: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    pc, phase, _ = _extract_pc_phase_coords(batch)
    return pc, phase


def _unwrap_subset_indices(dataset: Any) -> Tuple[Any, list[int] | None]:
    indices: list[int] | None = None
    while isinstance(dataset, torch.utils.data.Subset):
        if indices is None:
            indices = list(dataset.indices)
        else:
            indices = [indices[i] for i in dataset.indices]
        dataset = dataset.dataset
    return dataset, indices


def build_real_coords_dataloader(
    cfg: DictConfig,
    dm: Any,
    use_train_data: bool,
    use_full_dataset: bool = False,
) -> torch.utils.data.DataLoader:
    data_cfg = cfg.data
    data_files = getattr(data_cfg, "data_files", None)
    if not data_files:
        raise ValueError("No dataset under data_files files provided")

    file_list = data_files
    if isinstance(file_list, ListConfig):
        file_list = list(file_list)
    if isinstance(file_list, str):
        file_list = [file_list]

    full_dataset = PointCloudDataset(
        root=data_cfg.data_path,
        data_files=file_list,
        radius=getattr(data_cfg, "radius", 8),
        sample_type=getattr(data_cfg, "sample_type", "regular"),
        overlap_fraction=getattr(data_cfg, "overlap_fraction", 0.0),
        n_samples=getattr(data_cfg, "n_samples", 1000),
        num_points=getattr(data_cfg, "num_points", 100),
        return_coords=True,
        pre_normalize=getattr(data_cfg, "pre_normalize", True),
        normalize=getattr(data_cfg, "normalize", True),
        sampling_method=getattr(data_cfg, "sampling_method", "drop_farthest"),
    )

    dataset = full_dataset
    if not use_full_dataset:
        target_dataset = dm.train_dataset if use_train_data else dm.test_dataset
        _, indices = _unwrap_subset_indices(target_dataset)
        if indices is not None:
            dataset = torch.utils.data.Subset(full_dataset, indices)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def gather_inference_batches(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int | None = 4,
    max_samples_total: int | None = None,
    collect_coords: bool = False,
    seed_base: int | None = 123,
    progress_every_batches: int = 25,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect inputs and latents from batches."""
    inv_latents, eq_latents, phases, coords_list = [], [], [], []
    collected = 0
    max_samples = None if max_samples_total is None else max(1, int(max_samples_total))
    every = max(1, int(progress_every_batches))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pc, phase, coords = _extract_pc_phase_coords(batch)
            pc = pc.to(device)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)

            if seed_base is None:
                z, _, eq_z = model(pc)
            else:
                z, _, eq_z = _seeded_forward(model, pc, seed_base + batch_idx)
            if z is None:
                continue

            z_cpu = z.detach().cpu()
            take = z_cpu.shape[0]
            if max_samples is not None:
                remaining = max_samples - collected
                if remaining <= 0:
                    break
                take = min(take, remaining)

            if take <= 0:
                break

            inv_latents.append(z_cpu[:take])
            if collect_coords and coords is not None:
                coords_t = coords.detach().cpu()
                if coords_t.ndim == 1:
                    coords_t = coords_t.unsqueeze(0)
                elif coords_t.ndim > 2:
                    coords_t = coords_t.view(coords_t.shape[0], -1)
                coords_list.append(coords_t[:take])
            if eq_z is not None:
                eq_latents.append(eq_z.detach().cpu()[:take])
            if phase is not None:
                phases.append(phase.detach().view(-1).cpu()[:take])

            collected += int(take)
            if verbose and ((batch_idx + 1) % every == 0 or take != z_cpu.shape[0]):
                print(
                    f"[analysis][collect] batch={batch_idx + 1} "
                    f"samples={collected}"
                    + (f"/{max_samples}" if max_samples is not None else "")
                )

            if max_samples is not None and collected >= max_samples:
                if verbose:
                    print(f"[analysis][collect] reached sample cap: {collected}")
                break

    def _cat(tensors):
        return torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,))

    def _cat_coords(tensors):
        if not tensors:
            return np.empty((0, 3), dtype=np.float32)
        return torch.cat(tensors, dim=0).numpy()

    return {
        "inv_latents": _cat(inv_latents),
        "eq_latents": _cat(eq_latents),
        "phases": _cat(phases),
        "coords": _cat_coords(coords_list),
    }


def _sample_indices(num_samples: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or num_samples <= max_samples:
        return np.arange(num_samples)
    rng = np.random.default_rng(0)
    return rng.choice(num_samples, size=max_samples, replace=False)


def _default_cluster_count(num_samples: int, fallback: int = 4) -> int:
    if num_samples < 2:
        return 0
    return max(2, min(fallback, num_samples // 2))


def _seeded_forward(model: BarlowTwinsModule, pc: torch.Tensor, seed: int):
    """Run forward pass with fixed random seed while preserving global RNG state."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if pc.is_cuda else None
    torch.manual_seed(seed)
    if pc.is_cuda:
        torch.cuda.manual_seed_all(seed)
    try:
        return model(pc)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def evaluate_latent_equivariance(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 2,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate equivariance in encoder outputs (seeded vs unseeded)."""
    eq_errors_seeded = []
    eq_errors_unseeded = []
    determinism_errors = []
    identity_errors = []

    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, _ = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)
            batch_size = pc.shape[0]

            rots = torch.stack(
                [
                    torch.tensor(random_rotation_matrix(), device=device, dtype=pc.dtype)
                    for _ in range(batch_size)
                ]
            )
            pc_rot = torch.einsum("bij,bnj->bni", rots, pc)
            identity = torch.eye(3, device=device, dtype=pc.dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )

            seed = 42 + batch_idx

            # Determinism check (same input, same seed)
            _, _, eq_z_1 = _seeded_forward(model, pc, seed)
            _, _, eq_z_2 = _seeded_forward(model, pc, seed)
            if eq_z_1 is not None and eq_z_2 is not None:
                det_diff = torch.linalg.norm(eq_z_1 - eq_z_2, dim=-1).mean(dim=1)
                determinism_errors.extend(det_diff.detach().cpu().numpy().tolist())

            # Identity rotation check (should be ~0)
            _, _, eq_z_orig = _seeded_forward(model, pc, seed)
            _, _, eq_z_id = _seeded_forward(model, pc, seed)
            if eq_z_orig is not None and eq_z_id is not None:
                expected_id = torch.einsum("bij,bcj->bci", identity, eq_z_orig)
                id_rel = torch.linalg.norm(eq_z_id - expected_id, dim=-1) / torch.linalg.norm(
                    expected_id, dim=-1
                ).clamp_min(1e-6)
                identity_errors.extend(id_rel.mean(dim=1).detach().cpu().numpy().tolist())

            # Random rotation WITH seed control
            _, _, eq_z = _seeded_forward(model, pc, seed)
            _, _, eq_z_rot = _seeded_forward(model, pc_rot, seed)
            if eq_z is not None and eq_z_rot is not None:
                expected_eq = torch.einsum("bij,bcj->bci", rots, eq_z)
                rel = torch.linalg.norm(eq_z_rot - expected_eq, dim=-1) / torch.linalg.norm(
                    expected_eq, dim=-1
                ).clamp_min(1e-6)
                eq_errors_seeded.extend(rel.mean(dim=1).detach().cpu().numpy().tolist())

            # Random rotation WITHOUT seed control
            _, _, eq_z_uns = model(pc)
            _, _, eq_z_rot_uns = model(pc_rot)
            if eq_z_uns is not None and eq_z_rot_uns is not None:
                expected_eq_uns = torch.einsum("bij,bcj->bci", rots, eq_z_uns)
                rel_uns = torch.linalg.norm(eq_z_rot_uns - expected_eq_uns, dim=-1) / torch.linalg.norm(
                    expected_eq_uns, dim=-1
                ).clamp_min(1e-6)
                eq_errors_unseeded.extend(rel_uns.mean(dim=1).detach().cpu().numpy().tolist())

    eq_seeded = np.asarray(eq_errors_seeded)
    eq_unseeded = np.asarray(eq_errors_unseeded)
    det_arr = np.asarray(determinism_errors)
    id_arr = np.asarray(identity_errors)

    print("\n" + "=" * 60)
    print("EQUIVARIANCE DIAGNOSTICS")
    print("=" * 60)
    print("1. DETERMINISM (same input, same seed):")
    print(f"   Mean abs diff: {det_arr.mean():.6e}" if det_arr.size else "   N/A")
    print("   Should be ~0 if model is deterministic with fixed seed")
    print()
    print("2. IDENTITY ROTATION (R=I, same seed):")
    print(f"   Mean rel error: {id_arr.mean():.6e}" if id_arr.size else "   N/A")
    print("   Should be ~0 (sanity check)")
    print()
    print("3. RANDOM ROTATION (WITH seed control):")
    print(f"   Latent rel error: {eq_seeded.mean():.4f}" if eq_seeded.size else "   N/A")
    print("   Tests true equivariance (FPS initialized same way)")
    print()
    print("4. RANDOM ROTATION (WITHOUT seed control):")
    print(f"   Latent rel error: {eq_unseeded.mean():.4f}" if eq_unseeded.size else "   N/A")
    print("   Includes non-determinism from FPS random init")
    print("=" * 60)

    nondet = float(eq_unseeded.mean() - eq_seeded.mean()) if eq_seeded.size and eq_unseeded.size else float("nan")
    if eq_seeded.size and eq_unseeded.size:
        print(f"\nNON-DETERMINISM CONTRIBUTION: {nondet:.4f}")
        if nondet > 0.1:
            print("  WARNING: Significant non-determinism detected (likely FPS init).")
        print()

    metrics = {
        "eq_latent_rel_error_mean": float(eq_seeded.mean()) if eq_seeded.size else float("nan"),
        "eq_latent_rel_error_median": float(np.median(eq_seeded)) if eq_seeded.size else float("nan"),
        "eq_latent_rel_error_unseeded": float(eq_unseeded.mean()) if eq_unseeded.size else float("nan"),
        "eq_latent_rel_error_unseeded_median": float(np.median(eq_unseeded)) if eq_unseeded.size else float("nan"),
        "eq_latent_determinism_mean": float(det_arr.mean()) if det_arr.size else float("nan"),
        "eq_latent_identity_mean": float(id_arr.mean()) if id_arr.size else float("nan"),
        "eq_latent_nondeterminism_contribution": nondet,
        "num_samples": int(eq_seeded.size),
    }
    return metrics, eq_seeded


def evaluate_pose_head_relative_rotation(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    *,
    max_batches: int = 2,
    seed_base: int = 7331,
) -> tuple[Dict[str, float], np.ndarray]:
    """Evaluate pose head on true relative-rotation pairs (x, R·x)."""
    pose_head = getattr(model, "pose_head", None)
    if pose_head is None:
        return {"pose_head_eval_available": 0.0, "num_samples": 0}, np.empty((0,), dtype=np.float32)

    angle_deg_all: list[np.ndarray] = []
    angle_sym_deg_all: list[np.ndarray] = []
    det_all: list[np.ndarray] = []
    clamp_all: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, _ = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)
            if pc.dim() != 3 or pc.shape[-1] != 3:
                continue

            bsz = int(pc.shape[0])
            if hasattr(model, "_random_rotation_matrices"):
                rot = model._random_rotation_matrices(bsz, device=pc.device, dtype=pc.dtype)
            else:
                rot = torch.stack(
                    [
                        torch.tensor(random_rotation_matrix(), device=pc.device, dtype=pc.dtype)
                        for _ in range(bsz)
                    ]
                )
            pc_rot = apply_rotation(pc, rot)

            seed = int(seed_base + batch_idx)
            if bool(getattr(model, "pose_seeded_forward", True)):
                if hasattr(model, "_seeded_encoder_output"):
                    _, eq_a = model._seeded_encoder_output(pc, seed)
                    _, eq_b = model._seeded_encoder_output(pc_rot, seed)
                else:
                    _, _, eq_a = _seeded_forward(model, pc, seed)
                    _, _, eq_b = _seeded_forward(model, pc_rot, seed)
            else:
                if hasattr(model, "_prepare_encoder_input") and hasattr(model, "_split_encoder_output"):
                    eq_a = model._split_encoder_output(model.encoder(model._prepare_encoder_input(pc)))[1]
                    eq_b = model._split_encoder_output(model.encoder(model._prepare_encoder_input(pc_rot)))[1]
                else:
                    _, _, eq_a = model(pc)
                    _, _, eq_b = model(pc_rot)

            if hasattr(model, "_prepare_eq_latent"):
                eq_a = model._prepare_eq_latent(eq_a)
                eq_b = model._prepare_eq_latent(eq_b)
            if eq_a is None or eq_b is None:
                continue
            if eq_a.dim() != 3 or eq_a.shape[-1] != 3 or eq_b.shape != eq_a.shape:
                continue

            cov = torch.einsum("bci,bcj->bij", eq_b, eq_a)
            rot_hat = sixd_to_so3(pose_head(cov), eps=1e-6)

            eps = float(getattr(model, "symmetry_angle_eps", 1e-6))
            if hasattr(model, "_rotation_geodesic_angles"):
                angles_id, clamped = model._rotation_geodesic_angles(
                    rot_hat.to(torch.float32),
                    rot.to(dtype=torch.float32),
                    eps=eps,
                )
            else:
                angles_id, clamped = _rotation_geodesic_angles_torch(
                    rot_hat.to(torch.float32),
                    rot.to(dtype=torch.float32),
                    eps=eps,
                )
            rad_to_deg = 180.0 / np.pi
            angle_deg_all.append((angles_id * rad_to_deg).detach().cpu().numpy().astype(np.float32))
            clamp_all.append(clamped.detach().cpu().numpy())
            det_all.append(torch.det(rot_hat.to(torch.float32)).detach().cpu().numpy().astype(np.float32))

            sym_mats = getattr(model, "symmetry_mats", None)
            if sym_mats is not None:
                sym_mats = sym_mats.to(device=pc.device, dtype=pc.dtype)
                rot_targets = rot.unsqueeze(1) @ sym_mats.unsqueeze(0)
                if hasattr(model, "_rotation_geodesic_angles"):
                    angles_all, _ = model._rotation_geodesic_angles(
                        rot_hat.to(torch.float32).unsqueeze(1),
                        rot_targets.to(dtype=torch.float32),
                        eps=eps,
                    )
                else:
                    angles_all, _ = _rotation_geodesic_angles_torch(
                        rot_hat.to(torch.float32).unsqueeze(1),
                        rot_targets.to(dtype=torch.float32),
                        eps=eps,
                    )
                theta_sym = angles_all.min(dim=1).values
                angle_sym_deg_all.append(
                    (theta_sym * rad_to_deg).detach().cpu().numpy().astype(np.float32)
                )

    angle_deg = (
        np.concatenate(angle_deg_all).astype(np.float32)
        if angle_deg_all
        else np.empty((0,), dtype=np.float32)
    )
    angle_sym_deg = (
        np.concatenate(angle_sym_deg_all).astype(np.float32)
        if angle_sym_deg_all
        else np.empty((0,), dtype=np.float32)
    )
    det_vals = (
        np.concatenate(det_all).astype(np.float32)
        if det_all
        else np.empty((0,), dtype=np.float32)
    )
    clamp_flags = np.concatenate(clamp_all) if clamp_all else np.empty((0,), dtype=bool)

    print("\n" + "=" * 60)
    print("POSE-HEAD RELATIVE ROTATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Samples evaluated: {int(angle_deg.size)}")
    if angle_deg.size:
        print(
            f"Raw angle (deg): mean={float(np.mean(angle_deg)):.3f}, "
            f"median={float(np.median(angle_deg)):.3f}, p90={float(np.percentile(angle_deg, 90)):.3f}"
        )
    if angle_sym_deg.size:
        print(
            f"Symmetry-min angle (deg): mean={float(np.mean(angle_sym_deg)):.3f}, "
            f"median={float(np.median(angle_sym_deg)):.3f}, "
            f"p90={float(np.percentile(angle_sym_deg, 90)):.3f}"
        )
    if det_vals.size:
        print(f"det(R_hat): mean={float(np.mean(det_vals)):.6f}")
    if clamp_flags.size:
        print(f"acos clamp fraction: {float(np.mean(clamp_flags)):.6f}")
    print("=" * 60)

    metrics: Dict[str, float] = {
        "pose_head_eval_available": 1.0,
        "num_samples": int(angle_deg.size),
        "pose_head_rel_angle_mean_deg": float(np.mean(angle_deg)) if angle_deg.size else float("nan"),
        "pose_head_rel_angle_median_deg": float(np.median(angle_deg)) if angle_deg.size else float("nan"),
        "pose_head_rel_angle_p90_deg": float(np.percentile(angle_deg, 90)) if angle_deg.size else float("nan"),
        "pose_head_rel_det_mean": float(np.mean(det_vals)) if det_vals.size else float("nan"),
        "pose_head_rel_clamp_frac": float(np.mean(clamp_flags)) if clamp_flags.size else float("nan"),
        "max_batches": int(max_batches),
    }
    if angle_sym_deg.size:
        metrics["pose_head_rel_angle_sym_mean_deg"] = float(np.mean(angle_sym_deg))
        metrics["pose_head_rel_angle_sym_median_deg"] = float(np.median(angle_sym_deg))
        metrics["pose_head_rel_angle_sym_p90_deg"] = float(np.percentile(angle_sym_deg, 90))

    return metrics, angle_deg


__all__ = [
    "_default_cluster_count",
    "_sample_indices",
    "build_real_coords_dataloader",
    "kabsch_rotation_sanity_check",
    "cluster_orientations_by_cluster",
    "cap_cluster_labels",
    "compute_cluster_prototypes_and_alignments",
    "compute_cluster_prototypes_and_alignments_with_rotation_head",
    "compute_self_symmetry_metrics",
    "evaluate_pose_head_relative_rotation",
    "evaluate_latent_equivariance",
    "gather_inference_batches",
    "segment_grains_with_pose_head",
]
