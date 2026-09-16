"""Group observables with exactly the encoder's compact pooling weights."""

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.liquid_structure import ORDER_NAMES, bond_order
from src.models.encoders.mace_context import inner_weights


GROUP_NAMES = [f'mean_{n}' for n in ORDER_NAMES] + [f'std_{n}' for n in ORDER_NAMES]
# The other six group descriptors and both TDA readouts are evaluation-only.
TEACHER_COLUMNS = np.array([0, 1, 4, 5, 6, 8, 9, 12, 13, 14])


def group_observables(cloud, *, inner=5., outer=7., halo=18., coordination_radius=3.7):
    x = np.asarray(cloud, dtype=np.float64)
    radius = np.linalg.norm(x-x[0], axis=1)
    targets = np.flatnonzero(radius < outer)
    tree = cKDTree(x)
    nearest = tree.query(x[targets], k=13)[1]
    distance, second = tree.query(x[nearest].reshape(-1, 3), k=13)
    # All 12-neighbor queries must be complete inside the retained spherical halo.
    extent = np.linalg.norm(x[nearest].reshape(-1, 3)-x[0], axis=1) + distance[:, -1]
    if np.any(extent >= halo) or np.any(distance[:, 1] <= 0):
        raise ValueError(f'Incomplete or duplicate-atom group geometry: extent={extent.max()}, halo={halo}')
    vectors = (x[second[:, 1:]]-x[nearest].reshape(-1, 1, 3)).reshape(-1, 13, 12, 3)
    values = bond_order(vectors, coordination_radius)[0].astype(np.float64)
    weights = inner_weights(radius[targets], inner, outer)
    weights /= weights.sum()
    mean = weights @ values
    std = np.sqrt(weights @ np.square(values-mean))
    result = np.r_[mean, std]
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite group observables')
    return result
