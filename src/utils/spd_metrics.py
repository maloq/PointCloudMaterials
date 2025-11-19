"""
Metrics for evaluating Shape-Pose Disentanglement model.
Includes embedding quality, rotation equivariance, and reconstruction consistency metrics.
"""

import numpy as np
import torch
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.transform import Rotation

from src.loss.reconstruction_loss import sinkhorn_distance
from src.training_methods.spd.rot_heads import kabsch_rotation


def get_cubic_symmetry_matrices() -> np.ndarray:
    """
    Generate the 24 rotation matrices of the octahedral (cubic) symmetry group.
    Returns:
        (24, 3, 3) numpy array
    """
    # Basic 90 degree rotations
    s1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    s2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    s3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    s4 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    base_rots = [s1, s2, s3, s4]
    
    # Permutations of axes
    perms = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    ]
    
    # Generate all 24 by combining
    symmetries = []
    # This is a simplified construction. A robust way is to generate all signed permutations
    # with determinant +1.
    # Let's use a generator approach to be sure.
    
    # All signed permutations of identity
    import itertools
    for p in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            mat = np.zeros((3, 3))
            mat[0, p[0]] = signs[0]
            mat[1, p[1]] = signs[1]
            mat[2, p[2]] = signs[2]
            if np.linalg.det(mat) > 0:
                symmetries.append(mat)
                
    return np.array(symmetries)


def compute_global_aligned_rot_metric(pred_rots: np.ndarray, gt_rots: np.ndarray, 
                                     phase_labels: np.ndarray) -> dict:
    """
    Compute geodesic error after globally aligning the predicted frame to the GT frame per phase.
    This removes the penalty for the network learning a consistent but rotated canonical frame.
    
    Args:
        pred_rots: (N, 3, 3) predicted rotation matrices
        gt_rots: (N, 3, 3) ground truth rotation matrices
        phase_labels: (N,) phase labels
        
    Returns:
        Dictionary of aligned metrics per phase
    """
    metrics = {}
    
    # Convert to torch for kabsch helper
    pred_rots_t = torch.from_numpy(pred_rots).float()
    gt_rots_t = torch.from_numpy(gt_rots).float()
    
    for phase in np.unique(phase_labels):
        mask = phase_labels == phase
        if np.sum(mask) < 3:
            continue
            
        phase_pred = pred_rots_t[mask]
        phase_gt = gt_rots_t[mask]
        
        # We want to find R_align such that R_align @ R_pred ~ R_gt
        # This is equivalent to finding R_align that aligns the "frame vectors"
        # But a simpler way is to treat the rotation matrices as point clouds in R9? No.
        # Correct way: 
        # We want R_align s.t. R_align * R_pred_i \approx R_gt_i
        # => R_align \approx R_gt_i * R_pred_i^T
        # So we want the "average" rotation of (R_gt_i * R_pred_i^T)
        
        diffs = torch.bmm(phase_gt, phase_pred.transpose(1, 2)) # (N, 3, 3)
        
        # Average rotation can be found by SVD of the sum of matrices
        avg_diff = diffs.mean(dim=0) # (3, 3)
        
        # Project back to SO(3)
        U, _, Vh = torch.linalg.svd(avg_diff.unsqueeze(0))
        R_align = torch.bmm(U, Vh).squeeze(0)
        
        # Ensure det is +1
        if torch.det(R_align) < 0:
            # This is rare for average of rotations but possible
            U, S, Vh = torch.linalg.svd(avg_diff.unsqueeze(0))
            S_fix = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=avg_diff.device))
            R_align = U @ S_fix @ Vh
            R_align = R_align.squeeze(0)
            
        # Apply alignment
        # R_aligned = R_align @ R_pred
        phase_pred_aligned = torch.matmul(R_align.unsqueeze(0), phase_pred)
        
        # Compute geodesic error
        # trace(R_aligned^T @ R_gt)
        delta = torch.bmm(phase_pred_aligned.transpose(1, 2), phase_gt)
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        errors = torch.arccos(cos_theta) * (180.0 / np.pi)
        
        metrics[f'rot_aligned_error_phase_{int(phase)}'] = float(errors.mean())
        
    return metrics


def compute_symmetry_aware_rot_metric(pred_rots: np.ndarray, gt_rots: np.ndarray, 
                                     phase_labels: np.ndarray,
                                     symmetry_phases: list = None) -> dict:
    """
    Compute geodesic error allowing for symmetry operations.
    min_S dist(R_pred, R_gt @ S)
    
    Args:
        pred_rots: (N, 3, 3)
        gt_rots: (N, 3, 3)
        phase_labels: (N,)
        symmetry_phases: List of phase IDs (ints) to apply cubic symmetry to.
                         If None, applies to all.
    """
    metrics = {}
    symmetries = torch.from_numpy(get_cubic_symmetry_matrices()).float() # (24, 3, 3)
    
    pred_rots_t = torch.from_numpy(pred_rots).float()
    gt_rots_t = torch.from_numpy(gt_rots).float()
    
    for phase in np.unique(phase_labels):
        mask = phase_labels == phase
        if np.sum(mask) == 0:
            continue
            
        phase_pred = pred_rots_t[mask] # (B, 3, 3)
        phase_gt = gt_rots_t[mask]     # (B, 3, 3)
        
        if symmetry_phases is None or phase in symmetry_phases:
            # Apply all 24 symmetries to GT: R_gt_sym = R_gt @ S
            # Shape: (B, 24, 3, 3)
            phase_gt_expanded = phase_gt.unsqueeze(1) # (B, 1, 3, 3)
            symmetries_expanded = symmetries.unsqueeze(0) # (1, 24, 3, 3)
            
            # (B, 1, 3, 3) @ (1, 24, 3, 3) -> (B, 24, 3, 3)
            targets = torch.matmul(phase_gt_expanded, symmetries_expanded)
            
            # Compute distance to all 24 targets
            # R_pred: (B, 3, 3) -> (B, 1, 3, 3)
            preds = phase_pred.unsqueeze(1)
            
            # delta = preds^T @ targets
            delta = torch.matmul(preds.transpose(-1, -2), targets) # (B, 24, 3, 3)
            trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1) # (B, 24)
            cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            errors = torch.arccos(cos_theta) * (180.0 / np.pi) # (B, 24)
            
            min_errors, _ = errors.min(dim=1) # (B,)
            metrics[f'rot_sym_error_phase_{int(phase)}'] = float(min_errors.mean())
        else:
            # Standard geodesic
            delta = torch.bmm(phase_pred.transpose(1, 2), phase_gt)
            trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
            cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            errors = torch.arccos(cos_theta) * (180.0 / np.pi)
            metrics[f'rot_sym_error_phase_{int(phase)}'] = float(errors.mean())
            
    return metrics




def random_rotation_matrix():
    """Generate random 3D rotation using quaternions"""
    return Rotation.random().as_matrix()


def rotation_geodesic_distance_np(R1: np.ndarray, R2: np.ndarray, eps: float = 1e-6) -> float:
    """Geodesic distance between two rotation matrices in degrees."""
    R_error = R1 @ R2.T
    trace = np.trace(R_error)
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1 + eps, 1 - eps))
    return np.rad2deg(angle_rad)


def compute_cluster_metrics(latents: np.ndarray, labels: np.ndarray, stage: str) -> dict:
    """
    Compute clustering metrics (ARI, NMI, Silhouette).

    Args:
        latents: (N, d) latent embeddings
        labels: (N,) ground truth labels
        stage: Stage name (e.g., 'train', 'val', 'test')

    Returns:
        Dictionary of clustering metrics, or None if metrics cannot be computed
    """
    metrics = {}
    unique = np.unique(labels)
    if unique.size >= 2 and latents.shape[0] >= unique.size:
        try:
            assignments = KMeans(n_clusters=unique.size, n_init=10, random_state=0).fit_predict(latents)
            metrics["ARI"] = float(adjusted_rand_score(labels, assignments))
            metrics["NMI"] = float(normalized_mutual_info_score(labels, assignments))
        except Exception:
            pass
    if stage == "val" and latents.shape[0] >= 3:
        try:
            assignments_k3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(latents)
            if np.unique(assignments_k3).size > 1:
                metrics["Silhouette"] = float(silhouette_score(latents, assignments_k3))
        except Exception:
            pass
    return metrics or None


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
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, Z_inv, motif_labels, cv=min(5, len(Z_inv)))
        metrics['classification_accuracy'] = float(scores.mean())


    # Silhouette score (if labels available) - EXPENSIVE, test only
    if include_expensive and len(np.unique(motif_labels)) > 1 and len(Z_inv) >= 2:
        from sklearn.metrics import silhouette_score
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
