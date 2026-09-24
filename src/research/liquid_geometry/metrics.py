"""CPU metrics for held-out liquid representation geometry.

Every fitted transform uses training rows only. Distances are ordinary Euclidean
distances in the selected transformed space; physical target errors average over
both neighbors and target channels. Targets for neighbor evaluation must already
be standardized using training statistics.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


def _matrix(value, name: str, *, min_rows: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape (n >= {min_rows}, d >= 1); got {array.shape}"
        )
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values; got {array.dtype}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _vector(value, name: str, n: int, *, kind: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must have shape ({n},); got {array.shape}")
    if array.dtype.kind not in kind:
        raise TypeError(f"{name} has unsupported dtype {array.dtype}; expected {kind}")
    if array.dtype.kind in "iuf" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _train_eval(train, evaluation) -> tuple[np.ndarray, np.ndarray]:
    train = _matrix(train, "train", min_rows=2)
    evaluation = _matrix(evaluation, "eval")
    if train.shape[1] != evaluation.shape[1]:
        raise ValueError(
            f"train/eval feature counts differ: {train.shape[1]} != {evaluation.shape[1]}"
        )
    return train, evaluation


def _standardization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    constant = np.flatnonzero(scale == 0.0)
    # Match the explicit StandardScaler convention: constant training channels
    # remain centered but unscaled. Record them so this choice is reviewable.
    scale[constant] = 1.0
    return mean, scale, constant.tolist()


def participation(x) -> dict[str, float]:
    """Sample-covariance participation rank, trace and largest-eigenvalue share.

    For an exactly constant representation, all three fields are zero by
    convention. Centering and covariance accumulation use float64. Eigenvalues
    are calculated from singular values to avoid negative roundoff eigenvalues.
    """
    x = _matrix(x, "x", min_rows=2)
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular**2 / (len(x) - 1)
    trace = float(eigenvalues.sum())
    if not np.isfinite(trace):
        raise ValueError("x covariance overflowed float64")
    if trace == 0.0:
        return {"rank": 0.0, "trace": 0.0, "top_fraction": 0.0}
    # Normalizing first avoids squaring a potentially very large trace.
    fractions = eigenvalues / trace
    return {
        "rank": float(1.0 / np.dot(fractions, fractions)),
        "trace": trace,
        "top_fraction": float(fractions[0]),
    }


def fit_transform(train, eval, mode: str = "raw", ridge: float = 1e-3):
    """Return transformed train/eval arrays and JSON-serializable fit metadata.

    ``raw`` subtracts the training mean; ``standardized`` also divides by the
    training population standard deviation. ``whitened`` uses sample covariance
    C and W = (C + ridge * trace(C)/d * I)^(-1/2). This isotropic regularization
    preserves overall rescaling invariance without amplifying null directions
    infinitely. A wholly constant training representation cannot be whitened.
    """
    train, eval = _train_eval(train, eval)
    if mode not in {"raw", "standardized", "whitened"}:
        raise ValueError(f"Unknown distance transform mode {mode!r}")
    if not isinstance(ridge, Real) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError(f"ridge must be finite and positive; got {ridge!r}")
    mean = train.mean(axis=0)
    centered_train, centered_eval = train - mean, eval - mean
    metadata = {"mode": mode, "train_n": len(train), "center": mean.tolist()}
    if mode == "raw":
        return centered_train, centered_eval, metadata
    if mode == "standardized":
        mean, scale, constant = _standardization(train)
        metadata.update(scale=scale.tolist(), constant_channels=constant, ddof=0)
        return centered_train / scale, centered_eval / scale, metadata

    covariance = centered_train.T @ centered_train / (len(train) - 1)
    average_variance = float(np.trace(covariance) / train.shape[1])
    if not np.isfinite(average_variance) or average_variance <= 0:
        raise ValueError("whitening requires finite, nonzero training variance")
    ridge_variance = float(ridge * average_variance)
    regularized = covariance + np.eye(train.shape[1]) * ridge_variance
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("regularized training covariance is not positive definite")
    whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    metadata.update(
        ridge=float(ridge),
        ridge_variance=ridge_variance,
        covariance_ddof=1,
        whitening=whitening.tolist(),
    )
    return centered_train @ whitening, centered_eval @ whitening, metadata


def fit_physical_metric(train_x, train_y, eval_x, alpha: float = 1.0):
    """Distance embedding from multivariate ridge predictions of physical targets.

    Features and targets use training population standard deviations. Output
    coordinates are predicted standardized target values; evaluation targets are
    deliberately absent from the API. Ridge minimizes summed squared error plus
    ``alpha * ||coefficient||_F**2`` with an unpenalized intercept.
    """
    train_x, eval_x = _train_eval(train_x, eval_x)
    train_y = _matrix(train_y, "train_y", min_rows=2)
    if len(train_y) != len(train_x):
        raise ValueError(f"train_y rows {len(train_y)} != train_x rows {len(train_x)}")
    if not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and nonnegative; got {alpha!r}")
    train_scaled, eval_scaled, feature_metadata = fit_transform(
        train_x, eval_x, mode="standardized"
    )
    target_mean, target_scale, constant = _standardization(train_y)
    target_scaled = (train_y - target_mean) / target_scale
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit(train_scaled, target_scaled)
    metadata = {
        "mode": "physical_metric",
        "alpha": float(alpha),
        "features": feature_metadata,
        "target_center": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "constant_target_channels": constant,
        "target_ddof": 0,
        "coefficient": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
    return model.predict(train_scaled), model.predict(eval_scaled), metadata


def neighbor_metrics(ref_z, query_z, ref_y, query_y, ref_source, query_source, k: int = 15):
    """Per-query target MSE and mean distance among k other-source neighbors.

    Different-source exclusion is applied before neighbor search, including when
    queries and references overlap. No target statistics are fitted here. Each
    query must have at least k eligible reference rows; otherwise this fails.
    """
    ref_z = _matrix(ref_z, "ref_z")
    query_z = _matrix(query_z, "query_z")
    ref_y = _matrix(ref_y, "ref_y")
    query_y = _matrix(query_y, "query_y")
    if ref_z.shape[1] != query_z.shape[1]:
        raise ValueError("ref_z and query_z must have equal feature counts")
    if ref_y.shape[1] != query_y.shape[1]:
        raise ValueError("ref_y and query_y must have equal target counts")
    if len(ref_y) != len(ref_z) or len(query_y) != len(query_z):
        raise ValueError("Each target matrix must have the same rows as its embedding")
    ref_source = _vector(ref_source, "ref_source", len(ref_z), kind="iuSU")
    query_source = _vector(query_source, "query_source", len(query_z), kind="iuSU")
    if ref_source.dtype.kind in "iu" and query_source.dtype.kind in "SU" or (
        ref_source.dtype.kind in "SU" and query_source.dtype.kind in "iu"
    ):
        raise TypeError("ref_source and query_source must both use numeric or string IDs")
    if not isinstance(k, Integral) or isinstance(k, (bool, np.bool_)) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    mse = np.empty(len(query_z), dtype=np.float64)
    distance = np.empty(len(query_z), dtype=np.float64)
    indices = np.empty((len(query_z), int(k)), dtype=np.int64)
    for source in np.unique(query_source):
        query_ids = np.flatnonzero(query_source == source)
        allowed = np.flatnonzero(ref_source != source)
        if len(allowed) < k:
            raise ValueError(
                f"Query source {source!r} has {len(allowed)} different-source references; k={k}"
            )
        search = NearestNeighbors(n_neighbors=int(k), algorithm="brute", n_jobs=1)
        search.fit(ref_z[allowed])
        distances, local_indices = search.kneighbors(query_z[query_ids])
        indices[query_ids] = allowed[local_indices]
        target_difference = ref_y[indices[query_ids]] - query_y[query_ids, None, :]
        mse[query_ids] = np.mean(target_difference**2, axis=(1, 2))
        distance[query_ids] = distances.mean(axis=1)
    return {
        "neighbor_target_mse": mse,
        "neighbor_distance": distance,
        "neighbor_indices": indices,
    }


def conditional_rank(x, temperature) -> dict:
    """Participation statistics by exact temperature, with explicit singletons.

    Temperature bins are supplied by the caller; no implicit rounding or binning
    is performed. Singletons have null statistics and ``insufficient_rows``.
    """
    x = _matrix(x, "x")
    temperature = _vector(temperature, "temperature", len(x), kind="iuf")
    result = {}
    for value in np.unique(temperature):
        selected = x[temperature == value]
        n = len(selected)
        if n < 2:
            metrics = {"rank": None, "trace": None, "top_fraction": None}
            status = "insufficient_rows"
        else:
            metrics = participation(selected)
            status = "ok"
        result[value.item()] = {"n": n, "status": status, **metrics}
    return result


def lag_pairs(source, atom, frame, lag_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices paired by exact (source, atom, frame + lag) identity.

    Output order follows the past rows in the input. Missing exact future frames
    are omitted; there is no nearest-frame substitution. Duplicate keys fail
    because they make identity tracking ambiguous. Frames and atom IDs must use
    integer arrays, preserving exact identity without float conversion.
    """
    source = np.asarray(source)
    if source.ndim != 1:
        raise ValueError(f"source must be one-dimensional; got {source.shape}")
    source = _vector(source, "source", len(source), kind="iuSU")
    atom = _vector(atom, "atom", len(source), kind="iu")
    frame = _vector(frame, "frame", len(source), kind="iu")
    if not isinstance(lag_frames, Integral) or isinstance(lag_frames, (bool, np.bool_)) or lag_frames < 1:
        raise ValueError(f"lag_frames must be a positive integer; got {lag_frames!r}")
    keys = [(s.item(), int(a), int(f)) for s, a, f in zip(source, atom, frame)]
    lookup = {}
    for index, key in enumerate(keys):
        if key in lookup:
            raise ValueError(f"Duplicate (source, atom, frame) key {key!r} at rows {lookup[key]} and {index}")
        lookup[key] = index
    past, future = [], []
    for index, (s, a, f) in enumerate(keys):
        following = lookup.get((s, a, f + int(lag_frames)))
        if following is not None:
            past.append(index)
            future.append(following)
    return np.asarray(past, dtype=np.int64), np.asarray(future, dtype=np.int64)
