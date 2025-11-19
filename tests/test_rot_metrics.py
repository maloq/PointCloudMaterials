import torch
import numpy as np
import sys, os
sys.path.append(os.getcwd())

from src.training_methods.spd.spd_metrics import (
    compute_global_aligned_rot_metric,
    compute_symmetry_aware_rot_metric,
    get_cubic_symmetry_matrices,
    random_rotation_matrix
)

def test_global_alignment():
    print("\n--- Testing Global Alignment Metric ---")
    
    # Generate random GT rotations
    N = 100
    gt_rots = np.array([random_rotation_matrix() for _ in range(N)])
    
    # Create a random global offset
    R_offset = random_rotation_matrix()
    
    # Create predicted rotations: R_pred = R_offset @ R_gt
    # Note: The metric assumes R_align @ R_pred ~ R_gt
    # So if R_pred = R_offset^T @ R_gt, then R_offset @ R_pred = R_gt
    # Let's set R_pred = R_offset @ R_gt
    # Then we expect the metric to find R_align ~ R_offset^T and error ~ 0
    
    pred_rots = np.array([R_offset @ R for R in gt_rots])
    
    labels = np.zeros(N, dtype=int)
    
    metrics = compute_global_aligned_rot_metric(pred_rots, gt_rots, labels)
    print(f"Metrics with perfect global offset: {metrics}")
    
    assert metrics['rot_aligned_error_phase_0'] < 0.1, f"Error should be near zero for perfect global offset, got {metrics['rot_aligned_error_phase_0']}"
    
    # Test with noise
    noise_deg = 5.0
    pred_rots_noisy = []
    for R in pred_rots:
        # Add small random rotation
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(noise_deg)
        from scipy.spatial.transform import Rotation
        R_noise = Rotation.from_rotvec(axis * angle).as_matrix()
        pred_rots_noisy.append(R_noise @ R)
    pred_rots_noisy = np.array(pred_rots_noisy)
    
    metrics_noisy = compute_global_aligned_rot_metric(pred_rots_noisy, gt_rots, labels)
    print(f"Metrics with {noise_deg} deg noise: {metrics_noisy}")
    
    assert abs(metrics_noisy['rot_aligned_error_phase_0'] - noise_deg) < 1.0, "Error should be close to added noise"

def test_symmetry_aware():
    print("\n--- Testing Symmetry Aware Metric ---")
    
    symmetries = get_cubic_symmetry_matrices()
    print(f"Generated {len(symmetries)} symmetry matrices")
    
    # Test 1: Identity GT, Pred is a symmetry rotation
    gt_rot = np.eye(3)
    # Pick a symmetry that is NOT identity (e.g. 90 deg rotation)
    sym_rot = symmetries[1] 
    
    pred_rots = np.array([sym_rot])
    gt_rots = np.array([gt_rot])
    labels = np.array([0])
    
    # Standard metric would be high
    # Symmetry metric should be 0
    
    metrics = compute_symmetry_aware_rot_metric(pred_rots, gt_rots, labels, symmetry_phases=[0])
    print(f"Symmetry metric for exact symmetry: {metrics}")
    
    assert metrics['rot_sym_error_phase_0'] < 0.1, f"Error should be zero when prediction is a symmetry of GT, got {metrics['rot_sym_error_phase_0']}"
    
    # Test 2: Random GT, Pred = GT @ S
    N = 10
    gt_rots = np.array([random_rotation_matrix() for _ in range(N)])
    pred_rots = []
    for i in range(N):
        s = symmetries[np.random.randint(0, 24)]
        # R_pred = R_gt @ S_inv (or just S, since group is closed)
        # The metric checks min dist(R_pred, R_gt @ S)
        # So if R_pred = R_gt, dist is 0 (with S=I)
        # If R_pred = R_gt @ S_k, then dist(R_gt @ S_k, R_gt @ S_k) is 0
        pred_rots.append(gt_rots[i] @ s)
    pred_rots = np.array(pred_rots)
    labels = np.zeros(N, dtype=int)
    
    metrics = compute_symmetry_aware_rot_metric(pred_rots, gt_rots, labels, symmetry_phases=[0])
    print(f"Symmetry metric for random symmetries: {metrics}")
    
    assert metrics['rot_sym_error_phase_0'] < 1e-4

if __name__ == "__main__":
    test_global_alignment()
    test_symmetry_aware()
    print("\nAll tests passed!")
