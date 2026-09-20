"""Fixed local structural observation shared by training and inference.

Material coordinates use the training-calibrated nearest-160 reference scale.
The encoder sees only the local sphere: no computational halo or fixed-k cap.
"""
import numpy as np

from src.data.predictive_memory.targets import taper

REFERENCE_RADIUS = 9.192189
INNER_RADIUS = 6.
OUTER_RADIUS = 8.
EDGE_CUTOFF = 5.
POOL_SCALES = ((0., 3.), (3., 5.), (INNER_RADIUS, OUTER_RADIUS))
SUPPORT = dict(protocol='local_structure_v10', reference_radius=REFERENCE_RADIUS,
    inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS, edge_cutoff=EDGE_CUTOFF,
    atom_count='variable', halo=False)


def local_crop(positions, scale):
    """Return normalized local coordinates and source row indices, before packing."""
    x = np.asarray(positions, dtype=np.float32)*(REFERENCE_RADIUS/scale)
    rows = np.flatnonzero(np.linalg.norm(x, axis=-1) < OUTER_RADIUS)
    return x[rows], rows


def support_weights(positions):
    return taper(np.linalg.norm(positions, axis=-1), INNER_RADIUS, OUTER_RADIUS)
