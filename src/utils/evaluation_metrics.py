"""Metrics used by the current supervised representation cache."""

import re

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def primary_kmeansplusplus_hungarian_key(
    metrics: dict[str, float], eval_k: int | None
) -> str | None:
    key = "ACC_KMEANS_PLUSPLUS_HUNGARIAN"
    if eval_k is not None:
        key += f"_K{eval_k}"
    return key if key in metrics else None


_ACC_K_SUFFIX_RE = re.compile(r"^(ACC_[A-Z0-9_]+)_K\d+((?:_(?:MEAN|STD|BEST|RUNS|MIN|MAX))?)$")


def stabilize_class_metric_keys(
    metrics: dict[str, float], *, hungarian_eval_k: int | None
) -> dict[str, float]:
    stable: dict[str, float] = {}
    for raw_name, raw_value in metrics.items():
        name = str(raw_name).upper()
        match = _ACC_K_SUFFIX_RE.match(name)
        if match is not None:
            name = f"{match.group(1)}{match.group(2)}"
        if name in stable:
            raise ValueError(
                "Metric name collision while removing class-count suffixes: "
                f"{raw_name!r} maps to {name!r}."
            )
        value = finite_float(raw_value)
        if value is None:
            raise ValueError(
                f"Metric {raw_name!r} has non-finite value {raw_value!r}; "
                "cannot log stable metrics."
            )
        stable[name] = value
    if hungarian_eval_k is not None:
        if "HUNGARIAN_EVAL_K" in stable:
            raise ValueError("Metric key collision: 'HUNGARIAN_EVAL_K' is already present.")
        stable["HUNGARIAN_EVAL_K"] = float(int(hungarian_eval_k))
    return stable


def random_rotation_matrix() -> np.ndarray:
    """Generate one random 3D rotation matrix."""
    return Rotation.random().as_matrix()


def _hungarian_cluster_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = np.asarray(labels)
    assignments = np.asarray(assignments)
    if labels.shape != assignments.shape or labels.size == 0:
        raise ValueError(
            "labels and assignments must have identical non-empty shape, "
            f"got labels={labels.shape}, assignments={assignments.shape}."
        )
    label_values, label_indices = np.unique(labels, return_inverse=True)
    cluster_values, cluster_indices = np.unique(assignments, return_inverse=True)
    contingency = np.zeros((label_values.size, cluster_values.size), dtype=np.int64)
    np.add.at(contingency, (label_indices, cluster_indices), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[rows, columns].sum() / labels.size)


def compute_cluster_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    stage: str,
    *,
    hungarian_eval_k: int | None,
    acc_eval_methods: list[str],
    acc_eval_runs: int,
    acc_eval_runs_by_method: dict[str, int],
    acc_random_seed: int,
) -> dict[str, float]:
    """Compute KMeans ARI/NMI and configured k-means++ Hungarian accuracy."""
    if latents.ndim != 2 or labels.ndim != 1 or latents.shape[0] != labels.shape[0]:
        raise ValueError(
            "Cluster metrics require latents (N, D) and labels (N,), "
            f"got latents={latents.shape}, labels={labels.shape}."
        )
    if stage not in {"train", "val", "test"}:
        raise ValueError(f"stage must be 'train', 'val', or 'test', got {stage!r}.")
    if any(method != "kmeans++" for method in acc_eval_methods):
        raise ValueError(
            "The repository config supports only the 'kmeans++' ACC evaluator, "
            f"got {acc_eval_methods}."
        )
    if acc_eval_runs < 1:
        raise ValueError(f"acc_eval_runs must be positive, got {acc_eval_runs}.")
    invalid_overrides = {
        method: runs
        for method, runs in acc_eval_runs_by_method.items()
        if method != "kmeans++" or runs < 1
    }
    if invalid_overrides:
        raise ValueError(f"Invalid ACC run overrides: {invalid_overrides}.")

    metrics: dict[str, float] = {}
    class_count = np.unique(labels).size
    if class_count >= 2 and latents.shape[0] >= class_count:
        assignments = KMeans(
            n_clusters=class_count,
            init="k-means++",
            n_init=10,
            random_state=0,
        ).fit_predict(latents)
        metrics["ARI"] = float(adjusted_rand_score(labels, assignments))
        metrics["NMI"] = float(normalized_mutual_info_score(labels, assignments))

    if stage not in {"val", "test"} or hungarian_eval_k is None:
        return metrics
    if hungarian_eval_k < 2:
        raise ValueError(f"hungarian_eval_k must be at least 2, got {hungarian_eval_k}.")
    if latents.shape[0] < hungarian_eval_k:
        raise ValueError(
            "Hungarian evaluation needs at least one sample per requested cluster: "
            f"samples={latents.shape[0]}, clusters={hungarian_eval_k}."
        )

    for method in acc_eval_methods:
        run_count = acc_eval_runs_by_method.get(method, acc_eval_runs)
        accuracies = np.empty(run_count, dtype=np.float64)
        for run_index in range(run_count):
            assignments = KMeans(
                n_clusters=hungarian_eval_k,
                init="k-means++",
                n_init=10,
                random_state=acc_random_seed + run_index,
            ).fit_predict(latents)
            accuracies[run_index] = _hungarian_cluster_accuracy(labels, assignments)

        prefix = f"ACC_KMEANS_PLUSPLUS_HUNGARIAN_K{hungarian_eval_k}"
        metrics[prefix] = float(accuracies.mean())
        if run_count > 1:
            metrics[f"{prefix}_MEAN"] = float(accuracies.mean())
            metrics[f"{prefix}_STD"] = float(accuracies.std())
            metrics[f"{prefix}_BEST"] = float(accuracies.max())
            metrics[f"{prefix}_RUNS"] = float(run_count)
    return metrics


def compute_embedding_quality_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    include_expensive: bool = False,
) -> dict[str, float]:
    """Compute class separation and embedding-scale metrics."""
    if latents.ndim != 2 or labels.ndim != 1 or latents.shape[0] != labels.shape[0]:
        raise ValueError(
            "Embedding metrics require latents (N, D) and labels (N,), "
            f"got latents={latents.shape}, labels={labels.shape}."
        )

    metrics: dict[str, float] = {}
    unique_labels = np.unique(labels)
    if include_expensive and unique_labels.size > 1 and latents.shape[0] >= 10:
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(
            classifier,
            latents,
            labels,
            cv=min(5, latents.shape[0]),
        )
        metrics["classification_accuracy"] = float(scores.mean())

    intra_distances: list[float] = []
    inter_distances: list[float] = []
    for label in unique_labels:
        class_latents = latents[labels == label]
        other_latents = latents[labels != label]
        if class_latents.shape[0] > 1:
            intra_distances.extend(pdist(class_latents, metric="euclidean"))
        if class_latents.shape[0] and other_latents.shape[0]:
            inter_distances.extend(cdist(class_latents, other_latents, metric="euclidean").ravel())

    if intra_distances:
        metrics["intra_distance_mean"] = float(np.mean(intra_distances))
    if inter_distances:
        metrics["inter_distance_mean"] = float(np.mean(inter_distances))
    norms = np.linalg.norm(latents, axis=1)
    metrics["embedding_norm_mean"] = float(norms.mean())
    metrics["embedding_norm_std"] = float(norms.std())
    return metrics
