"""
Sanity check: Verify NN distances in RAW atomic positions (Ångströms)
before normalization to identify if analysis artifact was from normalization.

FIXED: Uses cKDTree against full dataset instead of cdist on subsample only.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(".")


def compute_nn_dist_correct(
    query_positions: np.ndarray,
    all_positions: np.ndarray,
    k: int = 1
) -> np.ndarray:
    """Compute nearest neighbor distances correctly using KDTree.
    
    Args:
        query_positions: Points to find NN for (N, 3)
        all_positions: Full dataset to search in (M, 3)
        k: Number of neighbors to find (default 1)
        
    Returns:
        Array of NN distances for each query point
    """
    if len(all_positions) < 2:
        return np.array([])
    
    tree = cKDTree(all_positions)
    
    # If query and all are the same, we need k+1 to skip self
    same_set = (query_positions is all_positions or 
                (len(query_positions) == len(all_positions) and 
                 np.allclose(query_positions, all_positions)))
    
    if same_set:
        distances, _ = tree.query(query_positions, k=k+1)
        return distances[:, 1] if k == 1 else distances[:, 1:]
    else:
        distances, _ = tree.query(query_positions, k=k)
        return distances if k == 1 else distances


def compute_nn_stats(positions: np.ndarray, sample_size: int = 50000) -> dict:
    """Compute NN statistics efficiently for large datasets.
    
    Uses sampling for query points but searches against FULL dataset.
    """
    n = len(positions)
    if n < 2:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, 
                "p1": np.nan, "p5": np.nan, "p95": np.nan, "p99": np.nan}
    
    # Build tree on full dataset
    tree = cKDTree(positions)
    
    # Sample query points if dataset is large
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        query = positions[idx]
    else:
        query = positions
    
    # Find 2 nearest neighbors (self + actual NN)
    distances, _ = tree.query(query, k=2)
    nn_dists = distances[:, 1]  # Skip self (distance 0)
    
    return {
        "mean": float(np.mean(nn_dists)),
        "std": float(np.std(nn_dists)),
        "min": float(np.min(nn_dists)),
        "max": float(np.max(nn_dists)),
        "p1": float(np.percentile(nn_dists, 1)),
        "p5": float(np.percentile(nn_dists, 5)),
        "p95": float(np.percentile(nn_dists, 95)),
        "p99": float(np.percentile(nn_dists, 99)),
        "n_below_2A": int(np.sum(nn_dists < 2.0)),
        "n_below_1A": int(np.sum(nn_dists < 1.0)),
    }


def compute_global_nn_stats(
    positions: np.ndarray, 
    phase_ids: np.ndarray,
    sample_size: int = 50000
) -> dict:
    """Compute NN where neighbor can be in ANY phase (global)."""
    return compute_nn_stats(positions, sample_size)


def compute_within_phase_nn_stats(
    positions: np.ndarray,
    phase_ids: np.ndarray, 
    target_phase: int,
    sample_size: int = 10000
) -> dict:
    """Compute NN where neighbor must be in SAME phase."""
    mask = phase_ids == target_phase
    phase_positions = positions[mask]
    return compute_nn_stats(phase_positions, sample_size)


def main():
    # Load raw atomic data (before any normalization)
    data_path = Path("output/synthetic_data/polycrystalline_metal")
    
    if not data_path.exists():
        print(f"ERROR: {data_path} does not exist")
        print("Please run data generation first")
        return
    
    # Load phase mapping (source of truth)
    phase_mapping_file = data_path / "phase_mapping.json"
    if phase_mapping_file.exists():
        with open(phase_mapping_file) as f:
            pm = json.load(f)
        id_to_name = {int(k): v for k, v in pm.get("id_to_name", {}).items()}
        print("Phase Mapping (from phase_mapping.json):")
        for k, v in sorted(id_to_name.items()):
            print(f"  {k}: {v}")
    else:
        print("WARNING: phase_mapping.json not found")
        id_to_name = {}
    
    # Load full atomic data
    atoms_file = data_path / "atoms_full.npy"
    if not atoms_file.exists():
        atoms_file = data_path / "atoms.npy"
    
    print(f"\nLoading: {atoms_file}")
    atoms = np.load(atoms_file, allow_pickle=True)
    
    if atoms.dtype.names:  # Structured array
        positions = atoms['position']
        phase_ids = atoms['phase_id']
        alive = atoms['alive'] if 'alive' in atoms.dtype.names else np.ones(len(atoms), dtype=bool)
    else:
        print("ERROR: Expected structured array with 'position' and 'phase_id'")
        return
    
    # Filter to alive atoms only
    alive_mask = alive.astype(bool)
    positions = positions[alive_mask]
    phase_ids = phase_ids[alive_mask]
    
    print(f"Total alive atoms: {len(positions):,}")
    
    # ==========================================================================
    # GLOBAL NN ANALYSIS (neighbor can be any phase)
    # ==========================================================================
    print("\n" + "="*90)
    print("GLOBAL NN Distance Analysis (neighbor can be ANY phase)")
    print("="*90)
    
    global_stats = compute_global_nn_stats(positions, phase_ids)
    print(f"  Mean:   {global_stats['mean']:.4f} Å")
    print(f"  Std:    {global_stats['std']:.4f} Å")
    print(f"  Min:    {global_stats['min']:.4f} Å")
    print(f"  Max:    {global_stats['max']:.4f} Å")
    print(f"  P1:     {global_stats['p1']:.4f} Å")
    print(f"  P5:     {global_stats['p5']:.4f} Å")
    print(f"  P95:    {global_stats['p95']:.4f} Å")
    print(f"  P99:    {global_stats['p99']:.4f} Å")
    print(f"  Atoms with NN < 2.0Å: {global_stats['n_below_2A']:,}")
    print(f"  Atoms with NN < 1.0Å: {global_stats['n_below_1A']:,}")
    
    # ==========================================================================
    # PER-PHASE NN ANALYSIS (within-phase neighbors only)
    # ==========================================================================
    print("\n" + "="*90)
    print("WITHIN-PHASE NN Distance Analysis (neighbor in SAME phase only)")
    print("="*90)
    
    header = f"{'Phase':<35} {'Count':>10} {'NN Mean':>10} {'NN Std':>9} {'NN Min':>9} {'NN P1':>9} {'<2Å':>8}"
    print(header)
    print("-"*90)
    
    unique_phases = np.unique(phase_ids)
    phase_stats = {}
    
    for phase_id in sorted(unique_phases):
        phase_mask = phase_ids == phase_id
        phase_positions = positions[phase_mask]
        phase_name = id_to_name.get(phase_id, f"Phase {phase_id}")
        
        if len(phase_positions) < 2:
            print(f"{phase_name:<35} {len(phase_positions):>10} (too few atoms)")
            continue
        
        stats = compute_nn_stats(phase_positions, sample_size=10000)
        phase_stats[phase_id] = stats
        
        print(f"{phase_name:<35} {len(phase_positions):>10,} {stats['mean']:>10.4f} "
              f"{stats['std']:>9.4f} {stats['min']:>9.4f} {stats['p1']:>9.4f} "
              f"{stats['n_below_2A']:>8,}")
    
    # ==========================================================================
    # PHYSICS CHECKS
    # ==========================================================================
    print("\n" + "="*90)
    print("Physics Validation")
    print("="*90)
    
    # Check for unphysical overlaps
    if global_stats['min'] < 1.0:
        print(f"⚠️  CRITICAL: Global NN min = {global_stats['min']:.4f} Å is UNPHYSICAL")
        print("   Iron atoms cannot be closer than ~1.8Å (sum of covalent radii)")
    elif global_stats['min'] < 1.8:
        print(f"⚠️  WARNING: Global NN min = {global_stats['min']:.4f} Å is suspicious")
    else:
        print(f"✓  Global NN min = {global_stats['min']:.4f} Å is physically reasonable")
    
    # Check against expected NN distance
    expected_nn = 2.49  # From avg_nn_dist in config
    if abs(global_stats['mean'] - expected_nn) < 0.2:
        print(f"✓  Global NN mean = {global_stats['mean']:.4f} Å matches expected {expected_nn} Å")
    else:
        print(f"⚠️  Global NN mean = {global_stats['mean']:.4f} Å differs from expected {expected_nn} Å")
    
    # Expected NN distance from config
    print("\n" + "="*90)
    print("Expected Values (from avg_nn_dist=2.49 Å in config)")
    print("="*90)
    print("BCC iron: NN = a√3/2 where a = 2*avg_nn/√3 → NN ≈ avg_nn_dist = 2.49 Å")
    print("FCC iron: NN = a/√2 where a = avg_nn*√2 → NN ≈ avg_nn_dist = 2.49 Å")
    print("\nIf density_target differs, expect slight scaling:")
    print("  BCC target: 0.0849 atoms/Å³")
    print("  FCC target: 0.0830 atoms/Å³ (2% lower → ~0.7% larger NN)")


if __name__ == "__main__":
    main()
