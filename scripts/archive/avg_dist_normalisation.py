# src/eval_tools/avg_dist_normalisation.py
import os
import numpy as np
from tqdm import tqdm
import sys,os
sys.path.append(os.getcwd())
from src.data_utils.data_load import PointCloudDataset, pc_normalize
from src.data_utils.prepare_data import calculate_stride, compute_dimensions
from src.data_utils.data_load import _load_points


def mean_pairwise_distance(points: np.ndarray) -> float:
    """
    Average Euclidean distance over all unique point pairs in `points` (N×3).
    """
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)          # (N, N) matrix
    i, j = np.triu_indices(len(points), k=1)      # upper triangle, exclude diagonal
    return dist[i, j].mean()


def main() -> None:
    root = "datasets/Al/inherent_configurations_off"
    npy_files = sorted(f for f in os.listdir(root) if f.endswith(".npy"))
    if not npy_files:
        print(f"No .npy files found under {root}")
        return

    first_path = os.path.join(root, npy_files[0])
    points_full = _load_points(first_path)
    min_coords = points_full.min(axis=0)
    max_coords = points_full.max(axis=0)
    box_lengths = max_coords - min_coords
    volume = float(np.prod(box_lengths))  # Å^3
    n_atoms = int(points_full.shape[0])
    number_density = n_atoms / volume if volume > 0 else float('nan')  # atoms/Å^3

    print("Simulation box (from first snapshot):")
    print(f"  Min coords: {min_coords} Å")
    print(f"  Max coords: {max_coords} Å")
    print(f"  Box lengths: Lx={box_lengths[0]:.3f} Å, Ly={box_lengths[1]:.3f} Å, Lz={box_lengths[2]:.3f} Å")
    print(f"  Atoms: {n_atoms}")
    print(f"  Number density: {number_density:.6f} atoms/Å^3")

    # Parameters used for sampling/normalisation
    radius = 7.5
    overlap_fraction = 0.0  # adjust if you use overlapped regular grid

    # Geometric fraction of volume excluded near edges (centers must be >= radius from faces)
    inner_lengths = np.maximum(box_lengths - 2.0 * radius, 0.0)
    if np.any(box_lengths <= 0):
        frac_excluded = float('nan')
    else:
        vol_inner = float(np.prod(inner_lengths))
        frac_excluded = 1.0 - (vol_inner / volume) if volume > 0 else float('nan')
    print(f"  Edge-excluded volume: {frac_excluded * 100.0:.2f}% of box volume")

    # Grid-based estimate (if you were to use a regular grid)
    stride = calculate_stride(radius, overlap_fraction)
    dims_no_pad = compute_dimensions(min_coords, max_coords, stride)
    dims_pad    = compute_dimensions(min_coords + radius, max_coords - radius, stride)
    total_no_pad = int(np.prod(np.maximum(dims_no_pad, 0)))
    total_pad    = int(np.prod(np.maximum(dims_pad, 0)))
    if total_no_pad > 0:
        pct_pad = 100.0 * max(total_no_pad - total_pad, 0) / total_no_pad
        print(
            f"  Grid edge exclusion (padding): drop {total_no_pad - total_pad}/{total_no_pad} centers ({pct_pad:.2f}%)"
        )
        # If also dropping the outermost interior layer
        keep_i = dims_pad[0] - 2 if dims_pad[0] >= 3 else 0
        keep_j = dims_pad[1] - 2 if dims_pad[1] >= 3 else 0
        keep_k = dims_pad[2] - 2 if dims_pad[2] >= 3 else 0
        kept_interior = int(max(keep_i, 0) * max(keep_j, 0) * max(keep_k, 0))
        dropped_edges = max(total_pad - kept_interior, 0)
        if total_pad > 0:
            pct_edges = 100.0 * dropped_edges / total_pad
            print(
                f"  Grid edge-layer drop (drop_edge_samples=True): drop {dropped_edges}/{total_pad} centers ({pct_edges:.2f}%)"
            )

    # 10 000 random samples, 128 atoms each, radius 8 Å
    ds = PointCloudDataset(
        root=root,
        data_files=npy_files,
        sample_type="random",
        n_samples=10_000,
        num_points=80,
        radius=radius,
        pre_normalize=False,   # keep raw coordinates
        normalize=False,       # no automatic normalisation
        return_coords=False,
    )

    tot_before, tot_after = 0.0, 0.0
    for sample in tqdm(ds.samples, desc="Processing samples"):
        tot_before += mean_pairwise_distance(sample)
        tot_after  += mean_pairwise_distance(pc_normalize(sample.copy(), radius))

    avg_before = tot_before / len(ds.samples)
    avg_after  = tot_after  / len(ds.samples)

    print(f"Average pair-wise distance BEFORE normalisation : {avg_before:.4f} Å")
    print(f"Average pair-wise distance AFTER  normalisation : {avg_after:.4f} (unitless)")


if __name__ == "__main__":
    main()
