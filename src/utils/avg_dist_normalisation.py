# src/eval_tools/avg_dist_normalisation.py
import os
import numpy as np
from tqdm import tqdm
import sys,os
sys.path.append(os.getcwd())
from src.data_utils.data_load import PointCloudDataset, pc_normalize


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
    off_files = [f for f in os.listdir(root) if f.endswith(".off")]

    # 10 000 random samples, 128 atoms each, radius 8 Å
    ds = PointCloudDataset(
        root=root,
        data_files=off_files,
        sample_type="random",
        n_samples=10_000,
        num_points=80,
        radius=7.5,
        pre_normalize=False,   # keep raw coordinates
        normalize=False,       # no automatic normalisation
        return_coords=False,
    )

    tot_before, tot_after = 0.0, 0.0
    for sample in tqdm(ds.samples, desc="Processing samples"):
        tot_before += mean_pairwise_distance(sample)
        tot_after  += mean_pairwise_distance(pc_normalize(sample.copy()))

    avg_before = tot_before / len(ds.samples)
    avg_after  = tot_after  / len(ds.samples)

    print(f"Average pair-wise distance BEFORE normalisation : {avg_before:.4f} Å")
    print(f"Average pair-wise distance AFTER  normalisation : {avg_after:.4f} (unitless)")


if __name__ == "__main__":
    main()