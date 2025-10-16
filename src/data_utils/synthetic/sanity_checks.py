"""Sanity checks and diagnostics for synthetic dataset quality."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .grains import compute_phase_autocorrelation
from .phases import PhaseLibrary, chamfer_distance, earth_mover_distance
from .scene import Scene


def moran_i(centers: np.ndarray, phases: Iterable[str]) -> float:
    return compute_phase_autocorrelation(centers, phases)


def ripley_k(centers: np.ndarray, radii: Sequence[float], volume: float) -> Dict[float, float]:
    centers = np.asarray(centers)
    tree = cKDTree(centers)
    density = centers.shape[0] / volume
    stats: Dict[float, float] = {}
    for r in radii:
        neighbors = tree.query_ball_point(centers, r)
        total_pairs = sum(len(idx_list) - 1 for idx_list in neighbors)
        stats[r] = total_pairs / (density * centers.shape[0])
    return stats


def motif_separability(
    library: PhaseLibrary,
    phases: Sequence[str],
    sample_size: int = 400,
) -> Dict[str, float]:
    prototypes = [library.get(name) for name in phases]
    distances = []
    emd_scores = []
    for i in range(len(prototypes)):
        for j in range(i + 1, len(prototypes)):
            a = prototypes[i].canonical_points()
            b = prototypes[j].canonical_points()
            if a.shape[0] > sample_size:
                a = a[np.random.choice(a.shape[0], size=sample_size, replace=False)]
            if b.shape[0] > sample_size:
                b = b[np.random.choice(b.shape[0], size=sample_size, replace=False)]
            distances.append(chamfer_distance(a, b))
            if a.shape[0] == b.shape[0]:
                emd_scores.append(earth_mover_distance(a, b))
    return {
        "min_chamfer": min(distances) if distances else 0.0,
        "min_emd": min(emd_scores) if emd_scores else 0.0,
    }


def check_equivariance(scene: Scene, rotations: Sequence[np.ndarray]) -> Dict[str, Dict[str, float]]:
    centers = scene.centers
    points = scene.points
    orientations = scene.orientations
    results: Dict[str, Dict[str, float]] = {}
    identity = np.eye(3)
    for idx, R in enumerate(rotations):
        rotated_centers = centers @ R.T
        centered_points = points - centers[:, None, :]
        rotated_points = np.einsum("ij,nkj->nki", R, centered_points) + rotated_centers[:, None, :]
        rotated_orientations = np.einsum("ij,njk->nik", R, orientations)
        radial_orig = np.linalg.norm(centered_points, axis=2)
        radial_rot = np.linalg.norm(rotated_points - rotated_centers[:, None, :], axis=2)
        radial_error = float(np.max(np.abs(radial_orig - radial_rot)))
        ortho_error = float(
            np.max(
                np.abs(
                    np.matmul(rotated_orientations, np.transpose(rotated_orientations, (0, 2, 1)))
                    - identity
                )
            )
        )
        det_error = float(np.max(np.abs(np.linalg.det(rotated_orientations) - 1)))
        results[f"rot_{idx}"] = {
            "radial_invariance_error": radial_error,
            "orthogonality_error": ortho_error,
            "determinant_error": det_error,
        }
    return results


def boundary_band_fraction(scene: Scene, threshold: float) -> float:
    boundary = scene.meta["boundary_distance"]
    return float((boundary <= threshold).mean())


def phase_cardinality(scene: Scene, phase_mix: Dict[str, float], tol: float = 0.05) -> Dict[str, float | bool]:
    counts = np.bincount(scene.phase_labels, minlength=len(scene.phase_ids))
    total = counts.sum()
    observed = counts / total
    expected = np.zeros_like(observed, dtype=float)
    for phase, weight in phase_mix.items():
        idx = scene.phase_ids.get(phase)
        if idx is not None:
            expected[idx] = weight
    max_dev = float(np.max(np.abs(observed - expected)))
    return {"max_deviation": max_dev, "within_tolerance": bool(max_dev <= tol)}
