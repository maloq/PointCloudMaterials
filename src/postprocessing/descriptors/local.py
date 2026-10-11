"""Local orientational descriptors used by birth-history analysis."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import sph_harm_y


def local_order6(patch):
    """The saved l=6 fields for a nearest80 patch from histories.common."""
    centers = patch[:13]
    near = np.argsort(cdist(centers, patch), axis=1)[:, 1:13]
    vectors = patch[near] - centers[:, None]
    radius = np.linalg.norm(vectors, axis=-1)
    theta = np.arccos(np.clip(vectors[..., 2] / radius, -1, 1))
    phi = np.arctan2(vectors[..., 1], vectors[..., 0])
    q = sph_harm_y(6, np.arange(-6, 7)[:, None, None], theta, phi).mean(-1).T
    norm = np.linalg.norm(q, axis=1)
    factor = np.sqrt(4 * np.pi / 13)
    unit = q / np.maximum(norm[:, None], 1e-14)
    coherence = (unit[0].conj() * unit[1:]).sum(1).real
    return float(factor * norm[0]), float(factor * np.linalg.norm(q.mean(0))), float(coherence.mean())
