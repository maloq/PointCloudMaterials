from __future__ import annotations

from typing import Callable, List, Tuple, Dict, Any
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

import sys,os
sys.path.append(os.getcwd())
from src.data_utils.soap_parquet import select_points_from_parquet


def process_sample(points, tree, center, size, n_points):  # type: ignore
    """Minimal fallback: grab n_points nearest within `size`."""
    idx = tree.query_ball_point(center, size)
    if not idx:
        return None, 0, 0
    sample = points[idx]
    # pad / trim
    if len(sample) >= n_points:
        sample = sample[: n_points]
    else:
        extra = sample[np.random.choice(len(sample), n_points - len(sample), replace=True)]
        sample = np.vstack([sample, extra])
    return sample, 0, 0

# -------------------------------------------------------------------------
# 1.  Utilities – rotations & Hungarian relabelling
# -------------------------------------------------------------------------

def _sample_rotations(n: int, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """Return `n` SO(3) rotation matrices. Index 0 is identity."""
    if n < 1:
        raise ValueError("n_rotations must be ≥ 1 (identity included).")
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
# 2.  Sampling helpers – fixed centres across rotations
# -------------------------------------------------------------------------

def _pick_centre_indices(points: np.ndarray, n_centres: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministically pick `n_centres` unique atoms as centres."""
    if n_centres > len(points):
        raise ValueError("n_centres > number of atoms.")
    return rng.choice(len(points), size=n_centres, replace=False)


def _extract_samples(points: np.ndarray,
                     centre_idx: np.ndarray,
                     radius: float,
                     n_points: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of local‑environment point clouds and their centres."""
    tree = KDTree(points)
    centres = points[centre_idx]
    samples: List[np.ndarray] = []
    valid_mask = np.ones(len(centres), dtype=bool)
    for i, c in enumerate(centres):
        struct, _, _ = process_sample(points, tree, c, radius, n_points)
        if struct is None:
            valid_mask[i] = False
            continue
        samples.append(struct)
    return samples, valid_mask

# -------------------------------------------------------------------------
# 3.  Public API – rotational robustness
# -------------------------------------------------------------------------

def rotational_robustness(
    predict_fn: Callable[[np.ndarray], Tuple[Any, float, float, float]],
    positions: np.ndarray,
    *,
    n_centres: int = 200,
    sphere_radius: float = 6.0,
    n_points: int = 128,
    n_rotations: int = 24,
    metric: str = "ari",                      # "ari" | "fraction"
    align: bool = True,
    rng_seed: int | None = 0,
) -> Tuple[float, np.ndarray]:
    """Compute mean rotational‑robustness score & per‑rotation distro.

    Parameters
    ----------
    predict_fn
        Function called as `predict_fn(sample)` where *sample* is an
        ``(n_points, 3)`` numpy array. **Must** return a tuple
        ``(rep, x, y, z)``; only *rep* is used here.
    positions
        ``(N, 3)`` Cartesian coordinates of the snapshot.
    n_centres
        Number of atoms to treat as central environments.
    sphere_radius, n_points
        Geometry of each local sample (same for all rotations).
    n_rotations
        Total number of rotation matrices (identity included).
    metric
        "ari" = Adjusted Rand Index (recommended, label‑perm‑invariant).
        "fraction" = fraction of atoms whose label stays identical
                      *after* Hungarian relabelling.
    align
        If True and metric=="fraction", run Hungarian alignment first.
        Ignored for "ari".

    Returns
    -------
    mean_score : float
        Average stability over all rotations (higher = better).
    per_rotation : (n_rotations‑1,) float64
        Stability for each *non‑identity* rotation in generation order.
    """
    rng = np.random.default_rng(rng_seed)

    # 1. Pick fixed central atoms
    centre_idx = _pick_centre_indices(positions, n_centres, rng)

    # 2. Reference samples & predictions (no rotation)
    ref_samples, valid_mask = _extract_samples(positions, centre_idx, sphere_radius, n_points)
    if not valid_mask.all():
        centre_idx = centre_idx[valid_mask]  # keep only valid ones
    ref_labels = []
    for sample in ref_samples:
        rep, *_ = predict_fn(sample)
        ref_labels.append(rep)
    ref_labels = np.array(ref_labels)

    # 3. Generate rotations
    R_mats = _sample_rotations(n_rotations, rng)
    scores: List[float] = []

    # precompute original centre coordinates for mapping
    original_centres = positions[centre_idx]

    for rot in R_mats[1:]:                     # skip identity
        # rotate whole snapshot
        pos_rot = positions @ rot.T

        # rotate centre coordinates and find nearest atoms
        tree_rot = KDTree(pos_rot)
        _, new_idx = tree_rot.query(original_centres @ rot.T)  # same order

        rot_samples, _ = _extract_samples(pos_rot, new_idx, sphere_radius, n_points)

        pred_labels = []
        for sample in rot_samples:
            rep, *_ = predict_fn(sample)
            pred_labels.append(rep)
        pred_labels = np.array(pred_labels)

        if metric == "ari":
            score = adjusted_rand_score(ref_labels, pred_labels)
        elif metric == "fraction":
            if align:
                pred_labels = _hungarian_relabel(ref_labels, pred_labels)
            score = _fraction_unchanged(ref_labels, pred_labels)
        else:
            raise ValueError("Unsupported metric: {metric!r}")

        scores.append(score)

    return float(np.mean(scores)), np.asarray(scores)

# -------------------------------------------------------------------------
# 4.  Example entry point – quick smoke test
# -------------------------------------------------------------------------

class DummyPointNetEncoder(nn.Module):
    """A very simple PointNet-like encoder for demonstration."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)
        self.relu = nn.ReLU()
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input x is (B, 3, N)"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_dim)
        return x

class DummyAE(nn.Module):
    """Dummy Autoencoder model."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = DummyPointNetEncoder(latent_dim)
        # Decoder is not needed for this task.

    def forward(self, x: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """Returns (None, latent_vector) to mimic expected model output."""
        latent = self.encoder(x)
        return None, latent

def make_ae_predict_fn(model: nn.Module, kmeans: KMeans, device: torch.device):
    """
    Factory to create a prediction function using an AE model and a KMeans model.
    """
    model.eval()
    def AE_predict(sample: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Predicts a cluster label for a sample using an autoencoder and KMeans.
        """
        centre = sample.mean(axis=0)

        sample_tensor = torch.from_numpy(sample).float().to(device)  # (N, 3)
        sample_tensor = sample_tensor.unsqueeze(0)                   # (1, N, 3)
        sample_tensor = sample_tensor.transpose(1, 2)                # (1, 3, N)

        with torch.no_grad():
            model_outputs = model(sample_tensor)
            if isinstance(model_outputs, tuple):
                latent_vector = model_outputs[1]
            else:
                latent_vector = model_outputs

        # The latent vector is on the specified device, KMeans expects CPU numpy array.
        latent_np = latent_vector.cpu().numpy()
        label = kmeans.predict(latent_np)[0]
        return int(label), centre[0], centre[1], centre[2]
    return AE_predict


if __name__ == "__main__":
    # --- setup ---
    N_atoms = 1000
    N_points = 64
    xyz = np.random.rand(N_atoms, 3) * 20  # 20×20×20 box
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder_model = DummyAE().to(device)

    # --- Pre-computation for AE_predict: train KMeans on reference latents ---
    # 1. Get reference samples (this is also done inside rotational_robustness)
    rng_for_sampling = np.random.default_rng(42) # Use same seed for consistency
    centre_idx_for_kmeans = _pick_centre_indices(xyz, 150, rng_for_sampling)
    ref_samples_for_kmeans, _ = _extract_samples(xyz, centre_idx_for_kmeans, 4.0, N_points)

    # 2. Get their latent vectors
    latents_list = []
    with torch.no_grad():
        for sample in ref_samples_for_kmeans:
            sample_tensor = torch.from_numpy(sample).float().to(device).unsqueeze(0).transpose(1, 2)
            _, latent = autoencoder_model(sample_tensor)
            latents_list.append(latent.cpu().numpy())
    all_latents = np.concatenate(latents_list, axis=0)

    # 3. Train KMeans on latent vectors
    kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto').fit(all_latents)

    # 4. Create the prediction function
    ae_predictor = make_ae_predict_fn(autoencoder_model, kmeans, device)

    # --- Run rotational robustness ---
    rr_mean, rr_dist = rotational_robustness(
        predict_fn=ae_predictor,
        positions=xyz,
        n_centres=150,
        sphere_radius=4.0,
        n_points=N_points,
        n_rotations=12,
        metric="ari",
        align=False,
        rng_seed=42,
    )
    print(f"Rotational robustness = {rr_mean:.3f}")