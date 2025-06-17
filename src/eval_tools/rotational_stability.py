from __future__ import annotations

from typing import Callable, List, Tuple, Dict, Any, Optional
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from functools import partial
import sys, os
from tqdm.auto import tqdm
import glob
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import read_off_file, process_sample
from src.data_utils.data_load import pc_normalize
from src.autoencoder.eval_autoencoder import autoencoder_predict_latent, load_model_and_config
from src.SOAP.predict_soap_pca import soap_pca_predict_latent, fit_soap_pca
torch.set_float32_matmul_precision('high')
# -------------------------------------------------------------------------
# 0.  Helper utilities
# -------------------------------------------------------------------------



def _sample_rotations(n: int, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """Return `n` SO(3) rotation matrices. Index 0 is identity."""
    if n < 1:
        raise ValueError("n_rotations must be ≥ 1 (identity included).")
    rng = np.random.default_rng(rng)
    mats = list(R.random(n - 1, random_state=rng).as_matrix())
    mats.insert(0, np.eye(3))
    return mats


def _hungarian_relabel(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Map `pred` labels on to `ref` labels via maximal overlap."""
    ref_labels, ref_inv = np.unique(ref, return_inverse=True)
    pred_labels, pred_inv = np.unique(pred, return_inverse=True)

    contingency = np.zeros((ref_labels.size, pred_labels.size), dtype=int)
    np.add.at(contingency, (ref_inv, pred_inv), 1)

    cost = contingency.max() - contingency  # maximise overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {pred_labels[j]: ref_labels[i] for i, j in zip(row_ind, col_ind)}
    return np.vectorize(lambda lbl: mapping.get(lbl, lbl))(pred)


def _fraction_unchanged(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(ref == pred))


# -------------------------------------------------------------------------
# 1.  Centre‑selection helpers
# -------------------------------------------------------------------------

def _pick_centre_indices(
    points: np.ndarray,
    radius: float,
    n_max: Optional[int] = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick centre atoms as **nearest atoms to the nodes of a regular cubic grid**.

    The grid spacing equals ``radius`` (i.e. one *sphere radius*).  Every grid
    node is associated with its nearest atom; duplicates are removed so each
    atom appears at most once.  If ``n_max`` is given and the number of unique
    centres exceeds it, a reproducible random subset of size ``n_max`` is
    returned.
    """
    # Build grid covering the snapshot bounding box
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spacing = radius
    # Slight epsilon so the max boundary is included when it falls exactly on a multiple of spacing
    coords = [np.arange(mins[d], maxs[d] + 1e-6, spacing) for d in range(3)]
    grid = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)

    # Map each grid node to its nearest atom
    tree = KDTree(points)
    _, nearest_idx = tree.query(grid)
    centre_idx = np.unique(nearest_idx)

    # Optionally limit to at most n_max centres
    if n_max is not None and centre_idx.size > n_max:
        rng = np.random.default_rng(rng)
        centre_idx = rng.choice(centre_idx, size=n_max, replace=False)

    return centre_idx


def _filter_edge_centres(
    points: np.ndarray,
    centre_idx: np.ndarray,
    radius: float,
    factor: float = 1.5,
) -> np.ndarray:
    """Drop centre indices whose coordinates are closer than ``factor * radius`` to any face of the simulation box."""
    if centre_idx.size == 0:
        return centre_idx

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    margin = factor * radius

    centres = points[centre_idx]
    inside = np.all((centres >= mins + margin) & (centres <= maxs - margin), axis=1)
    return centre_idx[inside]


# -------------------------------------------------------------------------
# 2.  Sampling utilities
# -------------------------------------------------------------------------

def _extract_samples(
    points: np.ndarray,
    centre_idx: np.ndarray,
    radius: float,
    n_points: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of local-environment point clouds and mask of valid centres."""
    tree = KDTree(points)
    centres = points[centre_idx]
    samples: List[np.ndarray] = []
    valid_mask = np.ones(len(centres), dtype=bool)
    for i, c in enumerate(centres):
        struct, _, _ = process_sample(points, tree, c, radius, n_points)
        if struct is None:
            valid_mask[i] = False
            continue
        struct = pc_normalize(struct)
        samples.append(struct)
    return samples, valid_mask

# -------------------------------------------------------------------------
# 3.  Public API – rotational robustness
# -------------------------------------------------------------------------

def rotational_robustness(
    predict_fn: Callable[[np.ndarray], Tuple[Any, float, float, float]],
    points: np.ndarray,
    n_centres: Optional[int] = 200,
    sphere_radius: float = 6.0,
    n_points: int = 128,
    n_rotations: int = 24,
    metric: str = "ari",  # "ari" | "fraction"
    align: bool = True,
    rng_seed: int | None = 0,
    n_clusters: int = 8,
    edge_margin_factor: float = 1.5,
    batch_size: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Compute mean rotational-robustness score & per-rotation distribution.

    Parameters
    ----------
    predict_fn
        Function that maps a point cloud (shape ``(N, 3)``) **or** a batch of
        point clouds (shape ``(B, N, 3)``) to a representation suitable for
        comparison/clustering.  The return type can be either
        ``np.ndarray``/``torch.Tensor`` of shape ``(d,)`` or ``(B, d)`` or a
        1-D array of labels.  When *batched* predictions are requested the
        function **must** support batched input.

    batch_size
        If ``None`` (default) every sample is forwarded to ``predict_fn``
        individually exactly as before.  Otherwise samples are stacked into
        batches of the given size, dramatically speeding-up evaluation on
        accelerators (GPU).

    Other parameters are identical to the previous version.

    **Centre selection** – Atoms are chosen as the **nearest atoms to the
    nodes of a regular cubic grid** whose spacing equals ``sphere_radius``.  A
    maximum of ``n_centres`` atoms is kept (if not ``None``).

    Centres whose coordinates are nearer than ``edge_margin_factor *
    sphere_radius`` to any face of the simulation box are **ignored** to avoid
    incomplete neighbourhoods at the edges.
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Pick central atoms by grid & drop those near the box edges
    centre_idx = _pick_centre_indices(points, sphere_radius, n_max=n_centres, rng=rng)
    centre_idx = _filter_edge_centres(points, centre_idx, sphere_radius, factor=edge_margin_factor)
    if centre_idx.size == 0:
        raise ValueError(
            "All candidate centres were filtered out by edge criteria. "
            "Try reducing `edge_margin_factor`, increasing `n_centres`, or using a larger snapshot."
        )

    # 2. Reference samples & predictions (no rotation)
    ref_samples, valid_mask = _extract_samples(points, centre_idx, sphere_radius, n_points)
    if not valid_mask.all():
        centre_idx = centre_idx[valid_mask]  # keep only valid ones

    # ------------------------------------------------------------------
    # Helper — run prediction on a list of point clouds with optional batching
    # ------------------------------------------------------------------
    def _predict(samples: List[np.ndarray], desc: str) -> np.ndarray:
        """Run *predict_fn* on *samples* with optional batching."""

        if batch_size is None or batch_size <= 1:
            reps = []
            for sample in tqdm(samples, desc=desc, leave=False):
                rep = predict_fn(sample)
                reps.append(rep)
            reps = np.asarray(reps)
        else:
            reps_chunks: List[np.ndarray] = []
            for i in tqdm(range(0, len(samples), batch_size), desc=desc, leave=False):
                batch_samples = samples[i : i + batch_size]
                batch = np.stack(batch_samples, axis=0)  # (B, N, 3)
                batch_rep = predict_fn(batch)

                # Convert to numpy & ensure first dimension is batch-dim
                if isinstance(batch_rep, torch.Tensor):
                    batch_rep = batch_rep.cpu().numpy()
                batch_rep = np.asarray(batch_rep)
                reps_chunks.append(batch_rep)
            reps = np.concatenate(reps_chunks, axis=0)
        return reps

    ref_reps = _predict(ref_samples, desc="Predicting reference samples")

    # 3. Generate rotations
    R_mats = _sample_rotations(n_rotations, rng)
    scores: List[float] = []

    # precompute original centre coordinates for mapping
    original_centres = points[centre_idx]

    for rot in tqdm(R_mats[1:], desc="Processing rotations"):  # skip identity
        # rotate whole snapshot
        pos_rot = points @ rot.T

        # rotate centre coordinates and find nearest atoms
        tree_rot = KDTree(pos_rot)
        _, new_idx = tree_rot.query(original_centres @ rot.T)  # same order

        rot_samples, _ = _extract_samples(pos_rot, new_idx, sphere_radius, n_points)

        pred_reps = _predict(rot_samples, desc="Predicting rotated samples")

        if metric == "ari":
            if ref_reps.ndim == 1:  # already labels
                ref_labels = ref_reps
                pred_labels = pred_reps
            else:  # representations are vectors → cluster them first
                kmeans = KMeans(n_clusters=n_clusters, random_state=rng_seed, n_init="auto")
                ref_labels = kmeans.fit_predict(ref_reps)
                pred_labels = KMeans(n_clusters=n_clusters, random_state=rng_seed, n_init="auto").fit_predict(
                    pred_reps
                )
            score = adjusted_rand_score(ref_labels, pred_labels)
        elif metric == "fraction":
            if align:
                pred_labels = _hungarian_relabel(ref_reps, pred_reps)
            score = _fraction_unchanged(ref_reps, pred_reps)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

        scores.append(score)

    return float(np.mean(scores)), np.asarray(scores)


if __name__ == "__main__":
    checkpoint_path = "output/2025-06-16/16-39-19/PnAE_Folding2Step_CD_Repulsion_RotCon_80_l16_no_edges-epoch=39-val_loss=0.05.ckpt"
    file_paths = ["datasets/Al/inherent_configurations_off/175ps.off"]
    model, cfg, device = load_model_and_config(checkpoint_path, cuda_device=0)

    all_points = []
    for file_path in file_paths:
        points = read_off_file(file_path, verbose=True)
        all_points.append(points)
    all_points = np.concatenate(all_points, axis=0)

    rr_mean, rr_dist = rotational_robustness(
        predict_fn=partial(autoencoder_predict_latent, model=model, device=device),
        points=all_points,
        n_centres=None,
        sphere_radius=cfg.data.radius,
        n_points=cfg.data.num_points,
        n_rotations=2,
        metric="ari",
        align=False,
        rng_seed=42,
        n_clusters=4,
        batch_size=4096,
    )


    print(f"Rotational robustness AE = {rr_mean:.3f}")



    # 1.  Collect *all* snapshots once and fit PCA
    soap_desc, pca_model = fit_soap_pca(
        [all_points],
        species="Al",
        soap_params=dict(r_cut=7.4, n_max=8, l_max=6, sigma=0.2),
        n_components=16,          # PCA components
    )

    # 2.  Wrap the predictor exactly like autoencoder_predict_latent
    predict_fn = partial(
        soap_pca_predict_latent,
        soap=soap_desc,
        pca=pca_model,
        species="Al",
    )

    # 3.  Drop it into your rotational-robustness benchmark
    rr_mean, rr_dist = rotational_robustness(
        predict_fn=predict_fn,
        points=all_points,        # ← same variable as before
        n_centres=None,
        sphere_radius=cfg.data.radius,
        n_points=cfg.data.num_points,
        n_rotations=6,
        metric="ari",
        align=False,
        rng_seed=42,
        n_clusters=4,
        batch_size=4096,
    )

    print(f"Rotational robustness SOAP = {rr_mean:.3f}")