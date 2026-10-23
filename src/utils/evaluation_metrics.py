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


def normalize_cluster_method_name(method) -> str:
    return str(method).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def ensure_kmeansplusplus(methods: list[str]) -> list[str]:
    """Keep requested method names/order and append the default only if absent."""
    methods = list(methods)
    aliases = {"kmeans++", "kmeansplusplus", "kmeanspp"}
    if not any(normalize_cluster_method_name(method) in aliases for method in methods):
        methods.append("kmeans++")
    return methods


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
        value = finite_float(raw_value)
        if value is None:
            raise ValueError(
                f"Metric {raw_name!r} has non-finite value {raw_value!r}; "
                "cannot log stable metrics."
            )
        stable[name] = value
    if hungarian_eval_k is not None:
        stable["HUNGARIAN_EVAL_K"] = float(int(hungarian_eval_k))
    return stable


def random_rotation_matrix() -> np.ndarray:
    """Generate one random 3D rotation matrix."""
    return Rotation.random().as_matrix()


def _hungarian_cluster_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = np.asarray(labels)
    assignments = np.asarray(assignments)
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
    aggregation_dtype=np.float64,
) -> dict[str, float]:
    """Compute clustering scores; preserve the caller's repeated-run precision."""
    metrics: dict[str, float] = {}
    class_count = np.unique(labels).size
    if class_count >= 2:
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
    for method in acc_eval_methods:
        run_count = acc_eval_runs_by_method.get(method, acc_eval_runs)
        # Single runs historically report the unrounded Hungarian fraction.
        accuracies = np.empty(run_count, dtype=aggregation_dtype if run_count > 1 else np.float64)
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
        if other_latents.shape[0]:
            inter_distances.extend(cdist(class_latents, other_latents, metric="euclidean").ravel())

    if intra_distances:
        metrics["intra_distance_mean"] = float(np.mean(intra_distances))
    if inter_distances:
        metrics["inter_distance_mean"] = float(np.mean(inter_distances))
    norms = np.linalg.norm(latents, axis=1)
    metrics["embedding_norm_mean"] = float(norms.mean())
    metrics["embedding_norm_std"] = float(norms.std())
    return metrics
