"""
Pairwise orientation quality analysis.

Workflow:
1) Predict pairwise relative rotations on a spatial kNN graph.
2) Synchronize these pairwise rotations into absolute SO(3) orientations.
3) Report edge residuals, triplet cycle consistency, and spatial smoothness.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from scipy.spatial import cKDTree
import sys
import os
sys.path.append(os.getcwd())
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.training_methods.spd.rot_heads import sixd_to_so3


def _project_to_so3_batch(mats: np.ndarray) -> np.ndarray:
    """Project batched 3x3 matrices to the closest proper rotations."""
    if mats.size == 0:
        return mats.astype(np.float32)
    U, _, Vt = np.linalg.svd(mats)
    R = np.matmul(U, Vt)
    det = np.linalg.det(R)
    neg = det < 0
    if np.any(neg):
        U_fix = U.copy()
        U_fix[neg, :, -1] *= -1.0
        R[neg] = np.matmul(U_fix[neg], Vt[neg])
    return R.astype(np.float32)


def _rotation_geodesic_deg_np(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    """Geodesic angle (degrees) between batched rotations."""
    if pred.size == 0:
        return np.empty((0,), dtype=np.float32)
    delta = np.matmul(np.transpose(pred, (0, 2, 1)), target)
    trace = np.trace(delta, axis1=1, axis2=2)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return np.degrees(np.arccos(cos_theta)).astype(np.float32)


def _subsample_nodes(
    num_nodes: int,
    max_nodes: int | None,
    seed: int,
) -> np.ndarray:
    if max_nodes is None or max_nodes <= 0 or num_nodes <= max_nodes:
        return np.arange(num_nodes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_nodes, size=int(max_nodes), replace=False)).astype(np.int64)


def _build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Return unique undirected kNN edges as (E,2) int64 array."""
    n = int(coords.shape[0])
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    k_eff = max(1, min(int(k), n - 1))
    tree = cKDTree(coords)
    _, nn_idx = tree.query(coords, k=k_eff + 1)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in nn_idx[i, 1:]:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))
    if not edge_set:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    return edges


def _predict_pairwise_relative_rotations(
    model: BarlowTwinsModule,
    eq_latents: np.ndarray,
    edges: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict R_ij (i->j) for edges using the pose head."""
    if edges.size == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    pose_head = getattr(model, "pose_head", None)
    if pose_head is None:
        raise ValueError("model.pose_head is required for pairwise orientation analysis.")

    device = model.device
    with torch.inference_mode():
        eq = torch.as_tensor(eq_latents, device=device, dtype=torch.float32)
        if hasattr(model, "_prepare_eq_latent"):
            eq = model._prepare_eq_latent(eq)
        if eq is None or eq.dim() != 3 or eq.shape[-1] != 3:
            raise ValueError(
                f"Expected eq latents shaped (N,C,3). Got {None if eq is None else tuple(eq.shape)}"
            )
        edge_t = torch.as_tensor(edges, device=device, dtype=torch.long)
        rots: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for start in range(0, int(edge_t.shape[0]), step):
            stop = min(int(edge_t.shape[0]), start + step)
            e = edge_t[start:stop]
            src = e[:, 0]
            dst = e[:, 1]
            eq_i = eq.index_select(0, src)
            eq_j = eq.index_select(0, dst)
            cov = torch.einsum("bci,bcj->bij", eq_j, eq_i)
            rot_ij = sixd_to_so3(pose_head(cov), eps=1e-6)
            rots.append(rot_ij.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rots, axis=0) if rots else np.empty((0, 3, 3), dtype=np.float32)


def synchronize_rotations_iterative(
    num_nodes: int,
    edges: np.ndarray,
    rot_ij: np.ndarray,
    *,
    num_iters: int = 30,
    tol_deg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative rotation synchronization from pairwise constraints."""
    if num_nodes <= 0:
        return np.empty((0, 3, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if edges.size == 0 or rot_ij.size == 0:
        eye = np.eye(3, dtype=np.float32)
        return np.tile(eye, (num_nodes, 1, 1)), np.empty((0,), dtype=np.float32)

    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    rel_to_i: list[list[np.ndarray]] = [[] for _ in range(num_nodes)]
    for (i, j), Rij in zip(edges, rot_ij):
        i = int(i)
        j = int(j)
        neighbors[i].append(j)
        rel_to_i[i].append(Rij.T)  # R_ji
        neighbors[j].append(i)
        rel_to_i[j].append(Rij)  # R_ij

    eye = np.eye(3, dtype=np.float32)
    R_abs = np.tile(eye, (num_nodes, 1, 1))
    updates: list[float] = []
    for _ in range(max(1, int(num_iters))):
        R_new = R_abs.copy()
        for i in range(num_nodes):
            if not neighbors[i]:
                continue
            nbr_idx = np.asarray(neighbors[i], dtype=np.int64)
            rel = np.stack(rel_to_i[i], axis=0).astype(np.float32)
            mats = np.matmul(rel, R_abs[nbr_idx])
            M = np.mean(mats, axis=0, keepdims=True)
            R_new[i] = _project_to_so3_batch(M)[0]
        step_deg = float(np.mean(_rotation_geodesic_deg_np(R_new, R_abs)))
        updates.append(step_deg)
        R_abs = R_new
        if step_deg <= float(tol_deg):
            break
    return R_abs.astype(np.float32), np.asarray(updates, dtype=np.float32)


def _edge_residuals_deg(
    R_abs: np.ndarray,
    edges: np.ndarray,
    rot_ij: np.ndarray,
) -> np.ndarray:
    if edges.size == 0:
        return np.empty((0,), dtype=np.float32)
    R_est = np.matmul(
        R_abs[edges[:, 1]],
        np.transpose(R_abs[edges[:, 0]], (0, 2, 1)),
    )
    return _rotation_geodesic_deg_np(R_est, rot_ij)


def _cycle_errors_deg(
    num_nodes: int,
    edges: np.ndarray,
    rot_ij: np.ndarray,
    *,
    max_triplets: int = 20000,
) -> np.ndarray:
    """Cycle consistency errors for i->j->k->i triplets."""
    if num_nodes < 3 or edges.size == 0:
        return np.empty((0,), dtype=np.float32)

    edge_map: dict[tuple[int, int], np.ndarray] = {}
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for (i, j), Rij in zip(edges, rot_ij):
        i = int(i)
        j = int(j)
        edge_map[(i, j)] = Rij
        edge_map[(j, i)] = Rij.T
        neighbors[i].add(j)
        neighbors[j].add(i)

    triplets: list[tuple[int, int, int]] = []
    limit = max(0, int(max_triplets))
    for i in range(num_nodes):
        ni = neighbors[i]
        if len(ni) < 2:
            continue
        for j in ni:
            if j <= i:
                continue
            common = ni.intersection(neighbors[j])
            for k in common:
                if k <= j:
                    continue
                triplets.append((i, j, k))
                if limit and len(triplets) >= limit:
                    break
            if limit and len(triplets) >= limit:
                break
        if limit and len(triplets) >= limit:
            break

    if not triplets:
        return np.empty((0,), dtype=np.float32)
    cyc = np.empty((len(triplets), 3, 3), dtype=np.float32)
    for idx, (i, j, k) in enumerate(triplets):
        Rij = edge_map[(i, j)]
        Rjk = edge_map[(j, k)]
        Rki = edge_map[(k, i)]
        cyc[idx] = Rij @ Rjk @ Rki
    identity = np.tile(np.eye(3, dtype=np.float32), (len(triplets), 1, 1))
    return _rotation_geodesic_deg_np(cyc, identity)


def _spatial_smoothness_deg(
    R_abs: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 6,
) -> np.ndarray:
    if len(coords) < 2:
        return np.empty((0,), dtype=np.float32)
    k_eff = max(1, min(int(k), len(coords) - 1))
    edges = _build_knn_edges(coords, k_eff)
    if edges.size == 0:
        return np.empty((0,), dtype=np.float32)
    R_rel = np.matmul(R_abs[edges[:, 1]], np.transpose(R_abs[edges[:, 0]], (0, 2, 1)))
    identity = np.tile(np.eye(3, dtype=np.float32), (len(edges), 1, 1))
    return _rotation_geodesic_deg_np(R_rel, identity)


def _nan_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
    }


def evaluate_pairwise_orientation_synchronization(
    model: BarlowTwinsModule,
    eq_latents: np.ndarray,
    coords: np.ndarray,
    *,
    knn_k: int = 12,
    max_nodes: int | None = 6000,
    sync_iters: int = 30,
    sync_tol_deg: float = 1e-3,
    smooth_k: int = 6,
    max_triplets: int = 20000,
    pair_batch_size: int = 4096,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Pairwise-to-absolute orientation validation:
    1) pose-head relative rotations on spatial neighbors,
    2) global SO(3) synchronization,
    3) triplet cycle consistency + spatial smoothness diagnostics.
    """
    eq_latents = np.asarray(eq_latents)
    coords = np.asarray(coords)
    if eq_latents.ndim != 3 or eq_latents.shape[-1] != 3:
        raise ValueError(f"eq_latents must have shape (N,C,3). Got {tuple(eq_latents.shape)}")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (N,3). Got {tuple(coords.shape)}")
    if eq_latents.shape[0] != coords.shape[0]:
        raise ValueError(
            f"eq_latents and coords must have matching first dim. "
            f"Got {eq_latents.shape[0]} and {coords.shape[0]}"
        )
    if getattr(model, "pose_head", None) is None:
        raise ValueError("model.pose_head is required for pairwise orientation synchronization analysis.")

    node_idx = _subsample_nodes(eq_latents.shape[0], max_nodes=max_nodes, seed=seed)
    eq_sub = eq_latents[node_idx]
    coords_sub = coords[node_idx]

    edges = _build_knn_edges(coords_sub, k=knn_k)
    rot_ij = _predict_pairwise_relative_rotations(
        model,
        eq_sub,
        edges,
        batch_size=pair_batch_size,
    )
    R_abs, sync_updates = synchronize_rotations_iterative(
        num_nodes=len(node_idx),
        edges=edges,
        rot_ij=rot_ij,
        num_iters=sync_iters,
        tol_deg=sync_tol_deg,
    )

    edge_err = _edge_residuals_deg(R_abs, edges, rot_ij)
    cycle_err = _cycle_errors_deg(
        num_nodes=len(node_idx),
        edges=edges,
        rot_ij=rot_ij,
        max_triplets=max_triplets,
    )
    smooth_err = _spatial_smoothness_deg(R_abs, coords_sub, k=smooth_k)

    metrics: Dict[str, Any] = {
        "num_nodes_used": int(len(node_idx)),
        "num_edges": int(len(edges)),
        "num_triplets": int(len(cycle_err)),
        "knn_k": int(knn_k),
        "smooth_k": int(smooth_k),
        "sync_iters_requested": int(sync_iters),
        "sync_iters_run": int(len(sync_updates)),
        "sync_last_update_deg": float(sync_updates[-1]) if len(sync_updates) else float("nan"),
    }
    metrics.update(_nan_stats(edge_err, "edge_residual_deg"))
    metrics.update(_nan_stats(cycle_err, "cycle_error_deg"))
    metrics.update(_nan_stats(smooth_err, "spatial_smoothness_deg"))

    return {
        "metrics": metrics,
        "node_indices": node_idx.astype(np.int64),
        "edges": edges.astype(np.int64),
        "pair_rotations": rot_ij.astype(np.float32),
        "absolute_rotations": R_abs.astype(np.float32),
        "edge_residual_deg": edge_err.astype(np.float32),
        "cycle_error_deg": cycle_err.astype(np.float32),
        "spatial_smoothness_deg": smooth_err.astype(np.float32),
        "sync_update_deg": sync_updates.astype(np.float32),
    }


__all__ = [
    "evaluate_pairwise_orientation_synchronization",
    "synchronize_rotations_iterative",
]
