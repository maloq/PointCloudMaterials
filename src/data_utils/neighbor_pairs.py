from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Sequence
from scipy.spatial import cKDTree


def _extract_coords_from_dataset(ds: Dataset) -> np.ndarray:
    """Extract an (M, 3) array of coordinates for a dataset or a Subset of PointCloudDataset.

    This avoids calling ds[i] (which would load point sets) by accessing the underlying
    PointCloudDataset.coords when available.
    """
    # torch.utils.data.Subset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = getattr(ds, "dataset")
        idxs = np.asarray(getattr(ds, "indices"), dtype=np.int64)
        if hasattr(base, "coords") and base.coords is not None:
            coords = np.asarray(base.coords)
            return np.asarray(coords)[idxs]
    # Fallback: dataset exposes coords directly
    if hasattr(ds, "coords") and getattr(ds, "coords") is not None:
        return np.asarray(getattr(ds, "coords"))
    # Last resort: iterate (costly)
    coords = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            _, c = item[0], item[1]
            coords.append(np.asarray(c))
        else:
            raise ValueError("Dataset does not provide coords for neighbor graph")
    return np.stack(coords, axis=0)


class NeighborPairDataset(Dataset):
    """Dataset of neighbor pairs from a base dataset providing (points, coords).

    Each item is a tuple: (points_i, points_j, distance_ij).
    The neighbor graph is built once in __init__ using a KD-tree with a given radius,
    optionally capping the number of neighbors per anchor.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        radius: float,
        max_neighbors: Optional[int] = None,
        directed: bool = True,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.radius = float(radius)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.directed = bool(directed)

        self._pairs, self._dists = self._build_pairs()

    def _build_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        coords = _extract_coords_from_dataset(self.base)  # (M, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords of shape (M, 3), got {coords.shape}")

        n = coords.shape[0]
        tree = cKDTree(coords)

        pairs_i = []
        pairs_j = []
        dists = []

        rng = np.random.default_rng()

        for i in range(n):
            nbrs: Sequence[int] = tree.query_ball_point(coords[i], r=self.radius)
            # Drop self
            nbrs = [j for j in nbrs if j != i]
            if len(nbrs) == 0:
                continue

            if self.max_neighbors is not None and len(nbrs) > self.max_neighbors:
                nbrs = rng.choice(nbrs, size=self.max_neighbors, replace=False).tolist()

            for j in nbrs:
                if not self.directed and j < i:
                    # Keep only (i, j) with i < j in undirected mode
                    continue
                pairs_i.append(i)
                pairs_j.append(j)
                dists.append(np.linalg.norm(coords[i] - coords[j]))

        if len(pairs_i) == 0:
            # No pairs found → create empty arrays with correct shape
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        pairs = np.stack([np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64)], axis=1)
        dists_arr = np.asarray(dists, dtype=np.float32)
        return pairs, dists_arr

    def __len__(self) -> int:
        return self._pairs.shape[0]

    def __getitem__(self, idx: int):
        i, j = self._pairs[idx]
        dij = self._dists[idx]
        item_i = self.base[i]
        item_j = self.base[j]

        # base dataset is expected to return (points, coords)
        if isinstance(item_i, (tuple, list)) and len(item_i) >= 1:
            pts_i = item_i[0]
        else:
            pts_i = item_i

        if isinstance(item_j, (tuple, list)) and len(item_j) >= 1:
            pts_j = item_j[0]
        else:
            pts_j = item_j

        # Ensure torch tensors
        if not isinstance(pts_i, torch.Tensor):
            pts_i = torch.tensor(pts_i, dtype=torch.float32)
        if not isinstance(pts_j, torch.Tensor):
            pts_j = torch.tensor(pts_j, dtype=torch.float32)

        dij_t = torch.tensor(dij, dtype=torch.float32)
        return pts_i, pts_j, dij_t

