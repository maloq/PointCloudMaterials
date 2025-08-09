from __future__ import annotations

"""Rotational consistency metric."""

from typing import Any, List, Optional, Tuple
import sys
import numpy as np

from .base import MultiRunMetric
from ..predictors import Predictor

from tqdm.auto import tqdm

from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import sys,os
sys.path.append(os.getcwd())
from src.data_utils.prepare_data import process_sample
from src.data_utils.data_load import pc_normalize


def _tqdm(
    iterable=None,
    *,
    desc: str | None = None,
    total: int | None = None,
):
    """Consistent, in-place progress bar.

    - leave=False so finished bars don't stack
    - dynamic_ncols=True for correct width
    - file=sys.stdout and disable when stdout is not a TTY to avoid newline spam
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.1,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )

def _sample_rotations(n: int, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """Return *n* SO(3) rotation matrices.  Index 0 is identity."""
    if n < 1:
        raise ValueError("n_rotations must be ≥ 1 (identity included).")
    rng = np.random.default_rng(rng)
    mats = list(R.random(n - 1, random_state=rng).as_matrix())
    mats.insert(0, np.eye(3))
    return mats


def _hungarian_relabel(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Map *pred* labels on to *ref* labels via maximal overlap."""
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


# -----------------------------------------------------------------------------
# 1.  Centre‑selection helpers
# -----------------------------------------------------------------------------


def _pick_centre_indices(
    points: np.ndarray,
    radius: float,
    n_max: Optional[int] = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick centre atoms as **nearest atoms to the nodes of a regular cubic grid**."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spacing = radius
    coords = [np.arange(mins[d], maxs[d] + 1e-6, spacing) for d in range(3)]
    grid = (
        np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(points.dtype)
    )

    tree = KDTree(points)
    _, nearest_idx = tree.query(grid)
    centre_idx = np.unique(nearest_idx)

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
    """Drop centres that are closer than *factor·radius* to any box face."""
    if centre_idx.size == 0:
        return centre_idx

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    margin = factor * radius

    centres = points[centre_idx]
    inside = np.all((centres >= mins + margin) & (centres <= maxs - margin), axis=1)
    return centre_idx[inside]


# -----------------------------------------------------------------------------
# 2.  Local‑environment sampling
# -----------------------------------------------------------------------------


def _extract_samples(
    points: np.ndarray,
    centre_idx: np.ndarray,
    radius: float,
    n_points: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of local point clouds and mask of *valid* centres."""
    tree = KDTree(points)
    centres = points[centre_idx]
    samples: List[np.ndarray] = []
    valid_mask = np.ones(len(centres), dtype=bool)
    for i, c in enumerate(centres):
        struct, _, _ = process_sample(points, tree, c, radius, n_points)
        if struct is None:
            valid_mask[i] = False
            continue
        samples.append(pc_normalize(struct))
    return samples, valid_mask


# =============================================================================
# Rotational‑robustness core
# =============================================================================


def _predict_batch(
    predictor: Any,
    samples: List[np.ndarray],
    batch_size: Optional[int],
    desc: str,
) -> np.ndarray:
    """Utility: run *predictor* on *samples* with optional batching."""

    if batch_size is None or batch_size <= 1:
        reps = [predictor.predict(s) for s in _tqdm(samples, desc=desc)]
        reps = np.asarray(reps)
    else:
        reps_chunks: List[np.ndarray] = []
        for i in _tqdm(range(0, len(samples), batch_size), desc=desc):
            batch = np.stack(samples[i : i + batch_size], axis=0)  # (B, N, 3)
            reps_chunks.append(predictor.predict(batch))
        reps = np.concatenate(reps_chunks, axis=0)
    return reps



def rotational_robustness(
    predictor: Any,
    points: np.ndarray,
    *,
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
    """Compute **mean** rotational‑robustness score & distribution (per rotation)."""

    rng = np.random.default_rng(rng_seed)

    # 1. Pick central atoms on a grid & drop edge‑neighbours
    centre_idx_raw = _pick_centre_indices(points, sphere_radius, n_max=n_centres, rng=rng)
    centre_idx = _filter_edge_centres(points, centre_idx_raw, sphere_radius, factor=edge_margin_factor)
    # Fallback: if filtering removed everything, proceed without edge filtering
    if centre_idx.size == 0:
        centre_idx = centre_idx_raw
        if centre_idx.size == 0:
            raise ValueError(
                "No candidate centres found. Adjust 'n_centres', 'sphere_radius' or input snapshot size."
            )

    # 2. Reference samples / predictions (no rotation)
    ref_samples, valid_mask = _extract_samples(points, centre_idx, sphere_radius, n_points)
    if not valid_mask.all():
        centre_idx = centre_idx[valid_mask]

    ref_reps = _predict_batch(predictor, ref_samples, batch_size, desc="Predicting reference samples")

    # 3. Generate rotations & compute scores
    R_mats = _sample_rotations(n_rotations, rng)
    scores: List[float] = []

    original_centres = points[centre_idx]

    for rot in _tqdm(R_mats[1:], desc="Processing rotations"):  # skip identity
        pos_rot = points @ rot.T
        tree_rot = KDTree(pos_rot)
        _, new_idx = tree_rot.query(original_centres @ rot.T)
        rot_samples, _ = _extract_samples(pos_rot, new_idx, sphere_radius, n_points)
        pred_reps = _predict_batch(predictor, rot_samples, batch_size, desc="Predicting rotated samples")

        if metric == "ari":
            if ref_reps.ndim == 1:  # already labels
                ref_labels, pred_labels = ref_reps, pred_reps
            else:
                # Ensure we do not request more clusters than samples
                n_samples = ref_reps.shape[0]
                if n_samples < 1:
                    raise ValueError(
                        "No valid samples available for clustering in rotational_robustness."
                    )
                effective_n_clusters = max(1, min(n_clusters, n_samples))

                kmeans_ref = KMeans(
                    n_clusters=effective_n_clusters, random_state=rng_seed, n_init="auto"
                )
                ref_labels = kmeans_ref.fit_predict(ref_reps)
                pred_labels = KMeans(
                    n_clusters=effective_n_clusters, random_state=rng_seed, n_init="auto"
                ).fit_predict(pred_reps)
            score = adjusted_rand_score(ref_labels, pred_labels)
        elif metric == "fraction":
            if align:
                pred_reps = _hungarian_relabel(ref_reps, pred_reps)
            score = _fraction_unchanged(ref_reps, pred_reps)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

        scores.append(score)

    return float(np.mean(scores)), np.asarray(scores)


def rotational_robustness_joint(
    predictor: Any,
    points_batched: np.ndarray,
    *,
    n_centres: Optional[int] = 200,
    sphere_radius: float = 6.0,
    n_points: int = 128,
    n_rotations: int = 24,
    metric: str = "ari",
    align: bool = True,
    rng_seed: int | None = 0,
    n_clusters: int = 8,
    edge_margin_factor: float = 1.5,
    batch_size: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    if points_batched.ndim != 3 or points_batched.shape[-1] != 3:
        raise ValueError(f"Expected (B, N, 3), got {points_batched.shape!r}")
    rng = np.random.default_rng(rng_seed)

    B = points_batched.shape[0]
    all_ref_samples: List[np.ndarray] = []
    all_original_centres: List[np.ndarray] = []

    for b in range(B):
        pts = points_batched[b]
        centre_idx_raw = _pick_centre_indices(pts, sphere_radius, n_max=n_centres, rng=rng)
        centre_idx = _filter_edge_centres(pts, centre_idx_raw, sphere_radius, factor=edge_margin_factor)
        if centre_idx.size == 0:
            centre_idx = centre_idx_raw
            if centre_idx.size == 0:
                raise ValueError(
                    "No candidate centres found. Adjust 'n_centres', 'sphere_radius' or input snapshot size."
                )
        ref_samples, valid_mask = _extract_samples(pts, centre_idx, sphere_radius, n_points)
        if not valid_mask.all():
            centre_idx = centre_idx[valid_mask]
        all_ref_samples.extend(ref_samples)
        all_original_centres.append(pts[centre_idx])

    if len(all_ref_samples) == 0:
        raise ValueError("No valid samples available for clustering in rotational_robustness_joint.")

    ref_reps = _predict_batch(predictor, all_ref_samples, batch_size, desc="Predicting reference samples")

    R_mats = _sample_rotations(n_rotations, rng)
    scores: List[float] = []

    for rot in _tqdm(R_mats[1:], desc="Processing rotations"):
        all_rot_samples: List[np.ndarray] = []
        for b in range(B):
            pts = points_batched[b]
            original_centres = all_original_centres[b]
            pos_rot = pts @ rot.T
            tree_rot = KDTree(pos_rot)
            _, new_idx = tree_rot.query(original_centres @ rot.T)
            rot_samples, _ = _extract_samples(pos_rot, new_idx, sphere_radius, n_points)
            all_rot_samples.extend(rot_samples)

        pred_reps = _predict_batch(predictor, all_rot_samples, batch_size, desc="Predicting rotated samples")

        if metric == "ari":
            if ref_reps.ndim == 1:
                ref_labels, pred_labels = ref_reps, pred_reps
            else:
                n_samples = ref_reps.shape[0]
                if n_samples < 1:
                    raise ValueError("No valid samples available for clustering in rotational_robustness_joint.")
                effective_n_clusters = max(1, min(n_clusters, n_samples))
                kmeans_ref = KMeans(n_clusters=effective_n_clusters, random_state=rng_seed, n_init="auto")
                ref_labels = kmeans_ref.fit_predict(ref_reps)
                pred_labels = KMeans(
                    n_clusters=effective_n_clusters, random_state=rng_seed, n_init="auto"
                ).fit_predict(pred_reps)
            score = adjusted_rand_score(ref_labels, pred_labels)
        elif metric == "fraction":
            if align:
                pred_reps = _hungarian_relabel(ref_reps, pred_reps)
            score = _fraction_unchanged(ref_reps, pred_reps)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

        scores.append(score)

    return float(np.mean(scores)), np.asarray(scores)
class _Wrapper:
    """Adapter exposing a vectorised ``predict`` on top of ``predict_raw``."""

    def __init__(self, predictor: Predictor):
        self._pred = predictor

    def predict(self, pcs: np.ndarray) -> np.ndarray:
        if pcs.ndim == 3:  # (B, N, 3)
            return np.asarray([self._pred.predict_raw(s) for s in pcs])
        return self._pred.predict_raw(pcs)


class RotationalConsistencyMetric(MultiRunMetric):
    """Evaluate rotational robustness of a predictor."""

    def __init__(self, predictor: Predictor, **params: Any):
        super().__init__(name="rotational_consistency", predictor=predictor, **params)

    def run_once(self, _predictions, *, points: np.ndarray, **_: Any) -> float:  # type: ignore[override]
        wrapper = _Wrapper(self.predictor)

        # Accept shapes: (N, 3), (B, N, 3), or higher (e.g., SOAP: (num_batches, B, N, 3))
        if points.ndim == 2:
            score, _ = rotational_robustness(wrapper, points, **self.params)
            return float(score)

        # Collapse any leading batch dimensions into a single batch dimension
        if points.ndim >= 3 and points.shape[-1] == 3:
            b = int(np.prod(points.shape[:-2]))
            points_batched = points.reshape(b, points.shape[-2], points.shape[-1])
            score, _ = rotational_robustness_joint(wrapper, points_batched, **self.params)
            return float(score)

        raise ValueError(f"Unsupported points shape: {points.shape!r}")
