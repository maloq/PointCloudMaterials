"""Train-only affine maps and density states; no encoder parameter updates."""

from dataclasses import dataclass
import warnings

import hdbscan
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from src.temporal_vamp.linear_vamp import LinearVAMP


@dataclass
class AffineMap:
    mean: np.ndarray
    matrix: np.ndarray
    offset: np.ndarray

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float64)-self.mean) @ self.matrix+self.offset

    def save(self, path):
        np.savez(path, mean=self.mean, matrix=self.matrix, offset=self.offset)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['mean'], data['matrix'], data['offset'])


def residualize(x, context):
    """Remove context means for fitting/scoring, never from deployed inputs."""
    result = np.asarray(x, dtype=np.float64).copy()
    for key in np.unique(context):
        ids = context == key
        result[ids] -= result[ids].mean(0)
    return result


def feature_scale(x):
    mean, scale = x.mean(0), x.std(0)
    # Constant coordinates have exactly zero centered contribution.
    scale[scale == 0] = 1.
    return mean, scale


def pca_map(x, dimension):
    mean, scale = feature_scale(x)
    pca = PCA(n_components=dimension, svd_solver='full').fit((x-mean)/scale)
    return AffineMap(mean, pca.components_.T/scale[:, None], np.zeros(dimension))


def temporal_map(current, previous, context, dimension, regularization):
    mean, scale = feature_scale(current)
    a = residualize((current-mean)/scale, context)
    b = residualize((previous-mean)/scale, context)
    fitted = LinearVAMP(regularization=regularization, eigenvalue_cutoff=1e-8).fit(a, b)
    if fitted.rank < dimension:
        raise ValueError(f'Temporal rank {fitted.rank} is smaller than requested {dimension}')
    # Equal-weight canonical coordinates, not singular-value-weighted kinetic maps.
    matrix = (fitted.whitening0_ @ fitted.left_singular_vectors_[:, :dimension])/scale[:, None]
    return AffineMap(mean, matrix, np.zeros(dimension)), fitted


def ridge_maps(x, y, alphas):
    """One float64 SVD shared by all ridge strengths and targets."""
    mean, scale = feature_scale(x)
    ym = y.mean(0)
    u, s, vt = np.linalg.svd((x-mean)/scale, full_matrices=False)
    uy = u.T @ (y-ym)
    return [AffineMap(mean, (vt.T @ ((s/(s*s+alpha))[:, None]*uy))/scale[:, None], ym)
            for alpha in alphas]


def neighborhood_metrics(z, target, context, k=8):
    """Within-context rank distortion and directed top-k reference recall."""
    result = []
    for key in np.unique(context):
        ids = np.flatnonzero(context == key)
        if len(ids) <= k:
            raise ValueError(f'Context {key} has only {len(ids)} rows, needs more than k={k}')
        dz, dy = cdist(z[ids], z[ids]), cdist(target[ids], target[ids])
        np.fill_diagonal(dz, np.inf); np.fill_diagonal(dy, np.inf)
        iz, iy = np.argsort(dz, axis=1)[:, :k], np.argsort(dy, axis=1)[:, :k]
        ranks = rankdata(dy, method='average', axis=1)
        # II uses rank of the nearest representation neighbor in reference space.
        imbalance = 2*np.mean(ranks[np.arange(len(ids)), iz[:, 0]])/len(ids)
        recall = np.mean([len(set(a) & set(b))/k for a, b in zip(iz, iy, strict=True)])
        result.append((key, imbalance, recall))
    return result


def fit_states(z, minimum_size, minimum_samples, method='eom'):
    return hdbscan.HDBSCAN(min_cluster_size=minimum_size, min_samples=minimum_samples,
        cluster_selection_method=method, prediction_data=True, core_dist_n_jobs=4,
        allow_single_cluster=False).fit(np.asarray(z, dtype=np.float64))


def state_membership(model, z, *, soft=True):
    """Keep HDBSCAN rejection; never argmax the soft vector to force a label."""
    z = np.asarray(z, dtype=np.float64)
    count = len(model.cluster_persistence_)
    if count == 0:
        return np.full(len(z), -1, dtype=np.int32), np.zeros(len(z)), np.zeros((len(z), 0))
    labels, strength = hdbscan.approximate_predict(model, z)
    membership = hdbscan.membership_vector(model, z) if soft else np.empty((len(z), 0))
    if soft and (membership.shape != (len(z), count) or not np.isfinite(membership).all()
                 or np.any(membership < -1e-8) or np.any(membership.sum(1) > 1+1e-6)):
        raise FloatingPointError(f'Invalid HDBSCAN membership vectors: {membership.shape}')
    return labels, strength, membership


def uncertainty(membership):
    n, k = membership.shape
    if k == 0:
        return dict(unassigned_mass=np.ones(n), ambiguity=np.zeros(n), margin=np.zeros(n))
    total = membership.sum(1)
    conditional = np.divide(membership, total[:, None], out=np.zeros_like(membership), where=total[:, None] > 0)
    entropy = -np.sum(conditional*np.log(np.maximum(conditional, 1e-300)), axis=1)
    ordered = np.sort(membership, axis=1)
    margin = ordered[:, -1]-(ordered[:, -2] if k > 1 else 0)
    return dict(unassigned_mass=1-total, ambiguity=entropy/np.log(k) if k > 1 else np.zeros(n), margin=margin)


def adjusted_pair_agreement(a, b):
    valid = (a >= 0) & (b >= 0)
    if not valid.any():
        return dict(assigned_pair_fraction=0., adjusted_agreement=np.nan, same_label=np.nan)
    a, b = a[valid], b[valid]
    labels = np.union1d(a, b)
    chance = sum(np.mean(a == k)*np.mean(b == k) for k in labels)
    same = np.mean(a == b)
    return dict(assigned_pair_fraction=float(valid.mean()),
        adjusted_agreement=float((same-chance)/(1-chance)) if chance < 1-1e-12 else np.nan,
        same_label=float(same))
