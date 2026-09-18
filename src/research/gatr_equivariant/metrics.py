"""Directional statistics, with explicit weak-vector masking and shuffle nulls."""
import numpy as np


def unit_vectors(values, threshold):
    v = np.asarray(values, dtype=np.float64)
    if v.shape[-1] != 3 or not np.isfinite(v).all() or np.any(np.asarray(threshold) < 0):
        raise ValueError('Expected finite 3-vectors and a nonnegative norm threshold')
    norm = np.linalg.norm(v, axis=-1)
    valid = norm > threshold
    unit = np.zeros_like(v)
    np.divide(v, norm[..., None], out=unit, where=valid[..., None])
    return unit, norm, valid


def direction_pair(a, b, threshold):
    u, _, va = unit_vectors(a, threshold)
    v, _, vb = unit_vectors(b, threshold)
    valid = va & vb
    cosine = np.clip(np.sum(u*v, axis=-1), -1, 1)
    angle = np.full(valid.shape, np.nan)
    angle[valid] = np.degrees(np.arccos(cosine[valid]))
    return cosine, angle, valid


def angular_metrics(a, b, threshold):
    cosine, angle, valid = direction_pair(a, b, threshold)
    fields = ('mean_angle_deg', 'median_angle_deg', 'p95_angle_deg', 'mean_axis_angle_deg',
              'jump60_fraction', 'flip90_fraction', 'p1', 'p2')
    result = dict(pairs=int(valid.size), valid_pairs=int(valid.sum()), coverage=float(valid.mean()))
    if not valid.any():
        return dict(result, **dict.fromkeys(fields))
    c, theta = cosine[valid], angle[valid]
    return dict(result, mean_angle_deg=float(theta.mean()), median_angle_deg=float(np.median(theta)),
        p95_angle_deg=float(np.quantile(theta, .95)),
        mean_axis_angle_deg=float(np.degrees(np.arccos(np.abs(c))).mean()),
        jump60_fraction=float(np.mean(theta > 60)), flip90_fraction=float(np.mean(theta > 90)),
        p1=float(c.mean()), p2=float(((3*c*c-1)/2).mean()))


def phase_shuffle_expectation(unit, labels, left, right):
    """Exact expectation of a uniform permutation within each PTM group.

    Inputs contain only valid directions. A permutation assigns distinct vectors
    to distinct positions, so same-group pairs require the finite-N correction.
    """
    labels = np.asarray(labels)
    unique, group = np.unique(labels, return_inverse=True)
    n = np.bincount(group)
    sums = np.stack([unit[group == i].sum(0) for i in range(len(unique))])
    second = np.stack([unit[group == i].T@unit[group == i] for i in range(len(unique))])
    p1 = np.empty((len(unique), len(unique)))
    c2 = np.empty_like(p1)
    for i in range(len(unique)):
        for j in range(len(unique)):
            if i == j:
                if n[i] == 1:
                    p1[i, j] = c2[i, j] = np.nan  # cannot occur for a distinct pair
                else:
                    p1[i, j] = (sums[i]@sums[i]-n[i])/(n[i]*(n[i]-1))
                    c2[i, j] = (np.sum(second[i]*second[i])-n[i])/(n[i]*(n[i]-1))
            else:
                p1[i, j] = (sums[i]@sums[j])/(n[i]*n[j])
                c2[i, j] = np.sum(second[i]*second[j])/(n[i]*n[j])
    return p1[group[left], group[right]], (3*c2[group[left], group[right]]-1)/2


def spatial_metrics(vectors, threshold, labels, positions, box, edges, overlap, *, phase_code=None):
    unit, _, valid = unit_vectors(vectors, threshold)
    indices = np.flatnonzero(valid)
    u, positions, labels = unit[valid], positions[valid], labels[valid]
    left, right = np.triu_indices(len(u), 1)
    delta = positions[left]-positions[right]
    delta -= box*np.round(delta/box)
    distance = np.linalg.norm(delta, axis=-1)
    p1 = np.clip(np.sum(u[left]*u[right], axis=-1), -1, 1)
    p2 = (3*p1*p1-1)/2
    null1, null2 = phase_shuffle_expectation(u, labels, left, right) if len(u) else (np.array([]), np.array([]))
    if not np.isfinite(null1).all() or not np.isfinite(null2).all():
        raise ValueError('Undefined shuffle expectation for an observed pair')
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        take = (distance >= lo) & (distance < hi)
        if phase_code is not None:
            take &= (labels[left] == phase_code) & (labels[right] == phase_code)
        if not take.any():
            continue
        rows.append(dict(distance_lo_A=lo, distance_hi_A=hi, pairs=int(take.sum()),
            distance_mean_A=float(distance[take].mean()), p1=float(p1[take].mean()), p2=float(p2[take].mean()),
            shuffle_p1=float(null1[take].mean()), shuffle_p2=float(null2[take].mean()),
            excess_p1=float((p1-null1)[take].mean()), excess_p2=float((p2-null2)[take].mean()),
            overlap80=float(overlap[indices[left[take]], indices[right[take]]].mean()),
            valid_centers=len(u), centers=len(valid)))
    return rows


def interval(values, draws):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Source interval requires finite values for every source')
    lo, hi = np.quantile(x[draws].mean(1), [.025, .975])
    return dict(mean=float(x.mean()), low=float(lo), high=float(hi))
