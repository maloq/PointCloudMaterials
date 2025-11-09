"""
Metrics for evaluating Shape-Pose Disentanglement model.
Includes embedding quality, rotation equivariance, and reconstruction consistency metrics.
"""

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.transform import Rotation

from src.loss.reconstruction_loss import sinkhorn_distance


def random_rotation_matrix():
    """Generate random 3D rotation using quaternions"""
    return Rotation.random().as_matrix()


def rotation_geodesic_distance_np(R1: np.ndarray, R2: np.ndarray, eps: float = 1e-6) -> float:
    """Geodesic distance between two rotation matrices in degrees."""
    R_error = R1 @ R2.T
    trace = np.trace(R_error)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1 + eps, 1 - eps))
    return np.rad2deg(angle_rad)


def compute_embedding_quality_metrics(Z_inv: np.ndarray, motif_labels: np.ndarray, include_expensive: bool = False) -> dict:
    """
    Compute embedding quality metrics.

    Args:
        Z_inv: (N, d) embeddings
        motif_labels: (N,) ground truth motif labels
        include_expensive: If True, compute expensive metrics (classification, silhouette)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Classification accuracy (train simple classifier) - EXPENSIVE, test only
    if include_expensive and len(np.unique(motif_labels)) > 1 and len(Z_inv) >= 10:

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, Z_inv, motif_labels, cv=min(5, len(Z_inv)))
        metrics['classification_accuracy'] = float(scores.mean())


    # Silhouette score (if labels available) - EXPENSIVE, test only
    if include_expensive and len(np.unique(motif_labels)) > 1 and len(Z_inv) >= 2:
        metrics['silhouette_score'] = float(silhouette_score(Z_inv, motif_labels))


    # Intra vs inter class distances
    intra_distances = []
    inter_distances = []

    for motif in np.unique(motif_labels):
        mask = motif_labels == motif
        intra_mask = motif_labels != motif

        Z_motif = Z_inv[mask]
        Z_other = Z_inv[intra_mask]

        if len(Z_motif) > 1:
            # Intra-class distances
            intra = pdist(Z_motif, metric='euclidean')
            intra_distances.extend(intra)

        if len(Z_other) > 0 and len(Z_motif) > 0:
            # Inter-class distances
            inter = cdist(Z_motif, Z_other, metric='euclidean')
            inter_distances.extend(inter.flatten())

    if intra_distances:
        metrics['intra_distance_mean'] = float(np.mean(intra_distances))
    if inter_distances:
        metrics['inter_distance_mean'] = float(np.mean(inter_distances))
    if intra_distances and inter_distances:
        metrics['separation_ratio'] = float(np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8))

    # Embedding norm statistics (detect collapse)
    norms = np.linalg.norm(Z_inv, axis=1)
    metrics['embedding_norm_mean'] = float(norms.mean())
    metrics['embedding_norm_std'] = float(norms.std())

    # Pairwise distance statistics (detect if embeddings too similar)
    if len(Z_inv) > 1:
        all_distances = pdist(Z_inv, metric='euclidean')
        metrics['pairwise_distance_mean'] = float(all_distances.mean())
        metrics['pairwise_distance_std'] = float(all_distances.std())
        metrics['pairwise_distance_min'] = float(all_distances.min())

    return metrics


def compute_canonical_consistency_metrics(canonicals: np.ndarray, invariant_embeddings: np.ndarray,
                                          phase_labels: np.ndarray) -> dict:
    """
    Test if different rotations of the same structure produce the same canonical pose.
    For samples from the same phase, canonical poses should be similar.

    Args:
        canonicals: (N, num_points, 3) canonical point clouds
        invariant_embeddings: (N, d) invariant embeddings
        phase_labels: (N,) phase labels

    Returns:
        Dictionary of consistency metrics per phase
    """
    metrics = {}

    # For each phase, compute variance in canonical poses
    for phase in np.unique(phase_labels):
        phase_mask = phase_labels == phase
        phase_canonicals = canonicals[phase_mask]
        phase_z_inv = invariant_embeddings[phase_mask]

        if len(phase_canonicals) < 2:
            continue

        # Measure variance in canonical poses using pairwise distances
        canonical_distances = []
        z_inv_distances = []

        n_samples = len(phase_canonicals)
        # Sample pairs to avoid quadratic complexity
        max_pairs = min(100, n_samples * (n_samples - 1) // 2)
        pairs_sampled = 0

        for i in range(n_samples):
            if pairs_sampled >= max_pairs:
                break
            for j in range(i + 1, n_samples):
                if pairs_sampled >= max_pairs:
                    break

                # Simple L2 distance between point clouds
                cd = np.mean(np.linalg.norm(phase_canonicals[i] - phase_canonicals[j], axis=1))
                canonical_distances.append(cd)

                # L2 distance between embeddings
                z_dist = np.linalg.norm(phase_z_inv[i] - phase_z_inv[j])
                z_inv_distances.append(z_dist)

                pairs_sampled += 1

        if canonical_distances:
            metrics[f'canonical_pose_variance_phase_{int(phase)}'] = float(np.mean(canonical_distances))
            metrics[f'canonical_pose_max_dist_phase_{int(phase)}'] = float(np.max(canonical_distances))
        if z_inv_distances:
            metrics[f'z_inv_variance_phase_{int(phase)}'] = float(np.mean(z_inv_distances))
            metrics[f'z_inv_max_dist_phase_{int(phase)}'] = float(np.max(z_inv_distances))

    return metrics


def compute_reconstruction_emd_per_phase(originals: np.ndarray, reconstructions: np.ndarray,
                                         phase_labels: np.ndarray) -> dict:
    """
    Compute reconstruction EMD per phase.

    Args:
        originals: (N, num_points, 3) original point clouds
        reconstructions: (N, num_points, 3) reconstructed point clouds
        phase_labels: (N,) phase labels

    Returns:
        Dictionary with EMD per phase
    """
    metrics = {}

    # Convert to torch for using sinkhorn distance
    originals_t = torch.from_numpy(originals).float()
    reconstructions_t = torch.from_numpy(reconstructions).float()
    phase_labels_t = torch.from_numpy(phase_labels).long()

    for phase in np.unique(phase_labels):
        phase_mask = phase_labels_t == phase
        phase_orig = originals_t[phase_mask]
        phase_recon = reconstructions_t[phase_mask]

        if len(phase_orig) == 0:
            continue

        emd, _ = sinkhorn_distance(phase_recon.contiguous(), phase_orig)
        metrics[f'emd_phase_{int(phase)}'] = float(emd.item())


    return metrics


def test_rotation_equivariance_sample(model, reference_pcs: dict, phase_labels: np.ndarray,
                                       n_test_rotations: int = 10, max_samples_per_phase: int = 5) -> dict:
    """
    Test rotation equivariance using reference point clouds.
    Test if: R_pred(R_test @ X) ≈ R_test @ R_pred(X)

    Args:
        model: The model to test
        reference_pcs: Dictionary mapping phase names to reference point clouds
        phase_labels: Phase labels from dataset
        n_test_rotations: Number of random rotations to test per phase
        max_samples_per_phase: Maximum samples to use per phase

    Returns:
        Dictionary of equivariance metrics
    """
    metrics = {'equivariance_errors': []}

    # Map phase labels to phase names
    phase_map = {0: 'crystal_fcc', 1: 'crystal_bcc', 2: 'amorphous_repeat',
                 3: 'amorphous_random', 4: 'amorphous_mixed'}

    model.eval()
    with torch.no_grad():
        for phase_id in np.unique(phase_labels):
            phase_name = phase_map.get(int(phase_id))
            if phase_name is None or phase_name not in reference_pcs:
                continue

            X_base = reference_pcs[phase_name]  # (N_points, 3)
            X_base_t = torch.from_numpy(X_base).float().to(model.device).unsqueeze(0)  # (1, N_points, 3)

            # Get prediction for original
            _, _, _, R_pred_orig = model(X_base_t)
            R_pred_orig_np = R_pred_orig[0].cpu().numpy()

            # Test with random rotations
            for _ in range(min(n_test_rotations, max_samples_per_phase)):
                R_test = random_rotation_matrix()

                # Apply known rotation to input
                X_rotated = (R_test @ X_base.T).T  # (N_points, 3)
                X_rotated_t = torch.from_numpy(X_rotated).float().to(model.device).unsqueeze(0)

                # Get prediction for rotated input
                _, _, _, R_pred_rot = model(X_rotated_t)
                R_pred_rot_np = R_pred_rot[0].cpu().numpy()

                # Key test: R_pred_rot should equal R_test @ R_pred_orig
                R_expected = R_test @ R_pred_orig_np

                # Geodesic distance
                angle_error_deg = rotation_geodesic_distance_np(R_pred_rot_np, R_expected)
                metrics['equivariance_errors'].append(angle_error_deg)



    if metrics['equivariance_errors']:
        metrics['equivariance_mean_deg'] = float(np.mean(metrics['equivariance_errors']))
        metrics['equivariance_std_deg'] = float(np.std(metrics['equivariance_errors']))
        metrics['equivariance_max_deg'] = float(np.max(metrics['equivariance_errors']))

    # Remove raw errors from logged metrics
    del metrics['equivariance_errors']

    return metrics


def test_reconstruction_consistency_sample(model, reference_pcs: dict, phase_labels: np.ndarray,
                                            n_rotations: int = 10, max_samples_per_phase: int = 3) -> dict:
    """
    Test reconstruction consistency across rotations.
    Reconstruction quality should be similar for all rotations (if system is rotation invariant).

    Args:
        model: The model to test
        reference_pcs: Dictionary mapping phase names to reference point clouds
        phase_labels: Phase labels from dataset
        n_rotations: Number of random rotations to test
        max_samples_per_phase: Maximum samples to use per phase

    Returns:
        Dictionary of reconstruction consistency metrics
    """
    metrics = {}
    reconstruction_errors_all = []

    # Map phase labels to phase names
    phase_map = {0: 'crystal_fcc', 1: 'crystal_bcc', 2: 'amorphous_repeat',
                 3: 'amorphous_random', 4: 'amorphous_mixed'}

    model.eval()
    with torch.no_grad():
        for phase_id in np.unique(phase_labels):
            phase_name = phase_map.get(int(phase_id))
            if phase_name is None or phase_name not in reference_pcs:
                continue

            X_base = reference_pcs[phase_name]  # (N_points, 3)
            reconstruction_errors = []

            for _ in range(min(n_rotations, max_samples_per_phase)):
                R_test = random_rotation_matrix()
                X_rotated = (R_test @ X_base.T).T
                X_rotated_t = torch.from_numpy(X_rotated).float().to(model.device).unsqueeze(0)

                _, recon, _, _ = model(X_rotated_t)

                # Compute reconstruction error
                emd, _ = sinkhorn_distance(recon.contiguous(), X_rotated_t)
                reconstruction_errors.append(float(emd.item()))


            if reconstruction_errors:
                reconstruction_errors_all.extend(reconstruction_errors)
                mean_err = np.mean(reconstruction_errors)
                std_err = np.std(reconstruction_errors)
                metrics[f'reconstruction_mean_phase_{int(phase_id)}'] = float(mean_err)
                metrics[f'reconstruction_std_phase_{int(phase_id)}'] = float(std_err)
                if mean_err > 0:
                    metrics[f'reconstruction_cv_phase_{int(phase_id)}'] = float(std_err / mean_err)

    if reconstruction_errors_all:
        metrics['reconstruction_mean_all'] = float(np.mean(reconstruction_errors_all))
        metrics['reconstruction_std_all'] = float(np.std(reconstruction_errors_all))
        mean_all = np.mean(reconstruction_errors_all)
        if mean_all > 0:
            metrics['reconstruction_cv_all'] = float(np.std(reconstruction_errors_all) / mean_all)

    return metrics
