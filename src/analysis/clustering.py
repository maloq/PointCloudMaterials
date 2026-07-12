"""Clustering logic for post-training analysis.

Provides k-means and HDBSCAN clustering on invariant latent features,
with shared feature preparation and color-map construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from .cluster_figures import (
    _build_cluster_color_map,
)
from src.vis_tools.latent_analysis_vis import (
    FittedClusteringModel,
    _compute_internal_clustering_metrics,
    _l2_normalize_rows_strict,
    _normalize_clustering_method_name,
    _prepare_clustering_features,
    compute_hdbscan_labels,
    compute_kmeans_labels,
    compute_transfer_kmeans_labels,
    fit_clustering_model,
    predict_clustering_model,
    transform_clustering_features,
)


@dataclass(frozen=True)
class HDBSCANResult:
    labels: np.ndarray | None
    info: dict[str, Any] | None
    color_map: dict[int, str] | None


def _resolve_cluster_k_values(k_values: list[int], *, n_samples: int) -> list[int]:
    resolved = list(dict.fromkeys(int(k) for k in k_values))
    if not resolved:
        raise ValueError("Clustering requires at least one configured k value.")
    invalid = [k for k in resolved if k < 2 or k > int(n_samples)]
    if invalid:
        raise ValueError(
            "Clustering k values must satisfy 2 <= k <= number of samples. "
            f"invalid={invalid}, n_samples={int(n_samples)}, configured={resolved}."
        )
    return resolved


def _build_clustering_state(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
    fit_latents: np.ndarray | None = None,
    requested_k_values: list[int],
    cluster_method: str,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    prepared_features: np.ndarray | None = None,
    prep_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[int], Dict[int, np.ndarray], Dict[int, str]]:
    configured_k_values = _resolve_cluster_k_values(requested_k_values, n_samples=len(latents))
    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    cluster_info_by_k: Dict[int, dict[str, Any]] = {}
    feature_prep: dict[str, Any] | None = (
        dict(prep_info)
        if prep_info is not None
        else None
    )

    for k_value in configured_k_values:
        if fit_latents is None:
            labels_k, info_k = compute_kmeans_labels(
                latents,
                int(k_value),
                random_state=int(random_state),
                method=cluster_method,
                l2_normalize=l2_normalize,
                standardize=standardize,
                pca_variance=pca_variance,
                pca_max_components=pca_max_components,
                prepared_features=prepared_features,
                prep_info=feature_prep,
                return_info=True,
            )
        else:
            labels_k, info_k = compute_transfer_kmeans_labels(
                fit_latents,
                latents,
                int(k_value),
                random_state=int(random_state),
                method=cluster_method,
                l2_normalize=l2_normalize,
                standardize=standardize,
                pca_variance=pca_variance,
                pca_max_components=pca_max_components,
                return_info=True,
            )
        cluster_labels_by_k[int(k_value)] = labels_k
        cluster_methods_by_k[int(k_value)] = str(info_k["method"])
        cluster_info_by_k[int(k_value)] = dict(info_k)
        if feature_prep is None:
            feature_prep = {
                key: info_k[key]
                for key in (
                    "input_dim",
                    "output_dim",
                    "l2_normalize",
                    "standardize",
                    "pca_components",
                    "pca_explained_variance",
                )
                if key in info_k
            }

    primary_k = int(configured_k_values[0])
    requested_method_normalized = _normalize_clustering_method_name(cluster_method)
    metrics: dict[str, Any] = {
        "cluster_method_requested": str(cluster_method).lower(),
        "cluster_method_resolved": str(requested_method_normalized),
        "random_state": int(random_state),
        "k_values_requested": [int(k) for k in requested_k_values],
        "k_values_used": [int(k) for k in configured_k_values],
        "primary_k": int(primary_k),
        "labels_k_method": str(cluster_methods_by_k[int(primary_k)]),
        "labels_method_by_k": {
            int(k): str(cluster_methods_by_k[int(k)])
            for k in configured_k_values
        },
        "cluster_fit_info_by_k": {
            int(k): dict(cluster_info_by_k[int(k)])
            for k in configured_k_values
        },
    }
    if feature_prep:
        metrics["cluster_feature_prep"] = feature_prep

    if phases.size == len(latents):
        unique_phases = np.unique(phases)
        if unique_phases.size > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            gt_k = max(2, min(int(unique_phases.size), len(latents)))
            gt_labels = cluster_labels_by_k.get(int(gt_k))
            if gt_labels is None:
                if fit_latents is None:
                    gt_labels = compute_kmeans_labels(
                        latents,
                        int(gt_k),
                        random_state=int(random_state),
                        method=cluster_method,
                        l2_normalize=l2_normalize,
                        standardize=standardize,
                        pca_variance=pca_variance,
                        pca_max_components=pca_max_components,
                        prepared_features=prepared_features,
                        prep_info=feature_prep,
                    )
                else:
                    gt_labels = compute_transfer_kmeans_labels(
                        fit_latents,
                        latents,
                        int(gt_k),
                        random_state=int(random_state),
                        method=cluster_method,
                        l2_normalize=l2_normalize,
                        standardize=standardize,
                        pca_variance=pca_variance,
                        pca_max_components=pca_max_components,
                    )
            metrics["ari_with_gt"] = float(adjusted_rand_score(phases, gt_labels))
            metrics["nmi_with_gt"] = float(normalized_mutual_info_score(phases, gt_labels))

    return metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k


def _clustering_feature_prep_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: info[key]
        for key in (
            "input_dim",
            "output_dim",
            "l2_normalize",
            "standardize",
            "pca_components",
            "pca_explained_variance",
        )
        if key in info
    }


def clustering_fit_quality_rejection_reasons(
    metrics: dict[str, Any],
    *,
    primary_k: int,
    pca_explained_variance_min: float = 0.5,
    mean_cosine_min: float = 0.6,
    silhouette_cosine_min: float = 0.15,
) -> list[str]:
    """Return reasons an external clustering fit is too weak to transfer."""
    info_by_k = metrics.get("cluster_fit_info_by_k")
    if not isinstance(info_by_k, dict):
        return ["missing cluster_fit_info_by_k in clustering fit metrics"]

    k = int(primary_k)
    info = info_by_k.get(k)
    if not isinstance(info, dict):
        return [f"missing clustering fit info for primary_k={k}"]

    reasons: list[str] = []

    def _add_if_below(metric_name: str, minimum: float) -> None:
        raw_value = info.get(metric_name)
        if raw_value is None:
            return
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Clustering fit metric is not numeric. "
                f"metric={metric_name!r}, value={raw_value!r}, primary_k={k}."
            ) from exc
        if not np.isfinite(value):
            reasons.append(f"{metric_name} is not finite ({value!r})")
        elif value < float(minimum):
            reasons.append(f"{metric_name}={value:.4g} < {minimum:.4g}")

    _add_if_below("pca_explained_variance", pca_explained_variance_min)
    _add_if_below("mean_assigned_cosine_similarity", mean_cosine_min)
    _add_if_below("silhouette_cosine", silhouette_cosine_min)
    return reasons


def _build_clustering_metrics_summary(
    *,
    requested_k_values: list[int],
    configured_k_values: list[int],
    cluster_method: str,
    random_state: int,
    cluster_methods_by_k: Dict[int, str],
    cluster_info_by_k: Dict[int, dict[str, Any]],
    feature_prep: dict[str, Any] | None,
) -> dict[str, Any]:
    primary_k = int(configured_k_values[0])
    requested_method_normalized = _normalize_clustering_method_name(cluster_method)
    metrics: dict[str, Any] = {
        "cluster_method_requested": str(cluster_method).lower(),
        "cluster_method_resolved": str(requested_method_normalized),
        "random_state": int(random_state),
        "k_values_requested": [int(k) for k in requested_k_values],
        "k_values_used": [int(k) for k in configured_k_values],
        "primary_k": int(primary_k),
        "labels_k_method": str(cluster_methods_by_k[int(primary_k)]),
        "labels_method_by_k": {
            int(k): str(cluster_methods_by_k[int(k)])
            for k in configured_k_values
        },
        "cluster_fit_info_by_k": {
            int(k): dict(cluster_info_by_k[int(k)])
            for k in configured_k_values
        },
    }
    if feature_prep:
        metrics["cluster_feature_prep"] = feature_prep
    return metrics


def fit_reusable_clustering_models(
    fit_latents: np.ndarray,
    phases: np.ndarray,
    *,
    requested_k_values: list[int],
    cluster_method: str,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[
    dict[str, Any],
    list[int],
    Dict[int, np.ndarray],
    Dict[int, str],
    Dict[int, FittedClusteringModel],
]:
    configured_k_values = _resolve_cluster_k_values(
        requested_k_values,
        n_samples=len(fit_latents),
    )
    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    cluster_info_by_k: Dict[int, dict[str, Any]] = {}
    fitted_models_by_k: Dict[int, FittedClusteringModel] = {}
    feature_prep: dict[str, Any] | None = None

    for k_value in configured_k_values:
        fitted_model, labels_k, info_k = fit_clustering_model(
            fit_latents,
            int(k_value),
            random_state=int(random_state),
            method=cluster_method,
            l2_normalize=l2_normalize,
            standardize=standardize,
            pca_variance=pca_variance,
            pca_max_components=pca_max_components,
        )
        cluster_labels_by_k[int(k_value)] = labels_k
        cluster_methods_by_k[int(k_value)] = str(info_k["method"])
        cluster_info_by_k[int(k_value)] = dict(info_k)
        fitted_models_by_k[int(k_value)] = fitted_model
        if feature_prep is None:
            feature_prep = _clustering_feature_prep_from_info(info_k)

    metrics = _build_clustering_metrics_summary(
        requested_k_values=requested_k_values,
        configured_k_values=configured_k_values,
        cluster_method=cluster_method,
        random_state=random_state,
        cluster_methods_by_k=cluster_methods_by_k,
        cluster_info_by_k=cluster_info_by_k,
        feature_prep=feature_prep,
    )
    if phases.size == len(fit_latents):
        unique_phases = np.unique(phases)
        if unique_phases.size > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            gt_k = max(2, min(int(unique_phases.size), len(fit_latents)))
            gt_labels = cluster_labels_by_k.get(int(gt_k))
            if gt_labels is not None:
                metrics["ari_with_gt"] = float(adjusted_rand_score(phases, gt_labels))
                metrics["nmi_with_gt"] = float(normalized_mutual_info_score(phases, gt_labels))
    return (
        metrics,
        configured_k_values,
        cluster_labels_by_k,
        cluster_methods_by_k,
        fitted_models_by_k,
    )


def predict_clustering_state_from_models(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
    fitted_models_by_k: Dict[int, FittedClusteringModel],
    requested_k_values: list[int],
    cluster_method: str,
    random_state: int,
) -> tuple[dict[str, Any], list[int], Dict[int, np.ndarray], Dict[int, str]]:
    configured_k_values = _resolve_cluster_k_values(
        requested_k_values,
        n_samples=len(latents),
    )
    missing = [int(k) for k in configured_k_values if int(k) not in fitted_models_by_k]
    if missing:
        raise KeyError(
            "Reusable clustering model set is missing requested k values. "
            f"missing={missing}, available={sorted(int(k) for k in fitted_models_by_k)}."
        )

    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    cluster_info_by_k: Dict[int, dict[str, Any]] = {}
    feature_prep: dict[str, Any] | None = None

    for k_value in configured_k_values:
        fitted_model = fitted_models_by_k[int(k_value)]
        labels_k, target_features = predict_clustering_model(
            latents,
            fitted_model,
            return_features=True,
        )
        target_metrics = _compute_internal_clustering_metrics(
            target_features,
            labels_k,
            random_state=int(random_state),
        )
        fit_info = dict(fitted_model.fit_info)
        info_k = {
            **fit_info,
            **target_metrics,
            "target_sample_count": int(len(latents)),
            "reused_fitted_model": True,
        }
        cluster_labels_by_k[int(k_value)] = labels_k
        cluster_methods_by_k[int(k_value)] = str(fitted_model.method)
        cluster_info_by_k[int(k_value)] = info_k
        if feature_prep is None:
            feature_prep = _clustering_feature_prep_from_info(fit_info)

    metrics = _build_clustering_metrics_summary(
        requested_k_values=requested_k_values,
        configured_k_values=configured_k_values,
        cluster_method=cluster_method,
        random_state=random_state,
        cluster_methods_by_k=cluster_methods_by_k,
        cluster_info_by_k=cluster_info_by_k,
        feature_prep=feature_prep,
    )
    if phases.size == len(latents):
        unique_phases = np.unique(phases)
        if unique_phases.size > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            gt_k = max(2, min(int(unique_phases.size), len(latents)))
            gt_labels = cluster_labels_by_k.get(int(gt_k))
            if gt_labels is not None:
                metrics["ari_with_gt"] = float(adjusted_rand_score(phases, gt_labels))
                metrics["nmi_with_gt"] = float(normalized_mutual_info_score(phases, gt_labels))
    return metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k


def compute_clustering_assignment_margins(
    latents: np.ndarray,
    *,
    fitted_model: FittedClusteringModel,
    expected_labels: np.ndarray,
    chunk_size: int = 200_000,
) -> dict[str, Any]:
    latents_arr = np.asarray(latents, dtype=np.float32)
    if latents_arr.ndim != 2:
        raise ValueError(
            f"Cluster assignment margin latents must have shape (N, D), got {latents_arr.shape}."
        )
    expected = np.asarray(expected_labels, dtype=int)
    if expected.ndim != 1:
        raise ValueError(
            f"Cluster assignment margin labels must have shape (N,), got {expected.shape}."
        )
    if expected.shape[0] != latents_arr.shape[0]:
        raise ValueError(
            "Cluster assignment margin labels must match latent rows: "
            f"labels={expected.shape[0]}, latents={latents_arr.shape[0]}."
        )
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    n_samples = int(latents_arr.shape[0])
    n_clusters = int(fitted_model.n_clusters)
    if n_clusters < 2:
        raise ValueError(
            "Cluster assignment margins require at least two clusters, "
            f"got n_clusters={n_clusters}."
        )
    raw_to_canonical = np.arange(n_clusters, dtype=np.int64)
    seen_new: set[int] = set()
    for old_label, new_label in fitted_model.label_remap.items():
        old = int(old_label)
        new = int(new_label)
        if old < 0 or old >= n_clusters or new < 0 or new >= n_clusters:
            raise ValueError(
                "Fitted clustering label_remap contains an out-of-range label: "
                f"old={old}, new={new}, n_clusters={n_clusters}."
            )
        if new in seen_new:
            raise ValueError(
                "Fitted clustering label_remap maps multiple raw labels to the "
                f"same canonical label {new}."
            )
        seen_new.add(new)
        raw_to_canonical[old] = new
    if len(set(int(v) for v in raw_to_canonical.tolist())) != n_clusters:
        raise ValueError(
            "Fitted clustering label_remap does not define a one-to-one raw-to-canonical mapping. "
            f"raw_to_canonical={raw_to_canonical.tolist()}."
        )

    if str(fitted_model.method) == "kmeans":
        if fitted_model.sklearn_model is None:
            raise RuntimeError("Fitted k-means model is missing the sklearn model state.")
        centers = np.asarray(fitted_model.sklearn_model.cluster_centers_, dtype=np.float32)
        score_name = "negative_squared_euclidean_distance"
    elif str(fitted_model.method) == "spherical_kmeans":
        if fitted_model.spherical_centers is None:
            raise RuntimeError("Fitted spherical k-means model is missing the spherical centers.")
        centers = np.asarray(fitted_model.spherical_centers, dtype=np.float32)
        score_name = "cosine_similarity"
    else:
        raise ValueError(
            f"Unsupported fitted clustering method {fitted_model.method!r} for assignment margins."
        )
    if centers.ndim != 2 or centers.shape[0] != n_clusters:
        raise ValueError(
            "Fitted clustering centers have an invalid shape for assignment margins: "
            f"centers={tuple(centers.shape)}, n_clusters={n_clusters}."
        )

    assigned_score = np.empty(n_samples, dtype=np.float32)
    runner_up_score = np.empty(n_samples, dtype=np.float32)
    runner_up_cluster = np.empty(n_samples, dtype=np.int64)
    margin = np.empty(n_samples, dtype=np.float32)

    for start in range(0, n_samples, int(chunk_size)):
        stop = min(start + int(chunk_size), n_samples)
        features = transform_clustering_features(
            latents_arr[start:stop],
            transform=fitted_model.feature_transform,
        )
        if str(fitted_model.method) == "spherical_kmeans":
            features_for_scores = _l2_normalize_rows_strict(
                features,
                context="Spherical k-means assignment margin features",
            )
            raw_scores = features_for_scores @ centers.T
        else:
            feature_norm2 = np.sum(features * features, axis=1, keepdims=True)
            center_norm2 = np.sum(centers * centers, axis=1, keepdims=True).T
            raw_scores = -(feature_norm2 + center_norm2 - 2.0 * (features @ centers.T))
        canonical_scores = np.empty_like(raw_scores, dtype=np.float32)
        canonical_scores[:, raw_to_canonical] = raw_scores
        order = np.argsort(canonical_scores, axis=1)
        top1 = np.asarray(order[:, -1], dtype=np.int64)
        top2 = np.asarray(order[:, -2], dtype=np.int64)
        chunk_expected = expected[start:stop]
        if not np.array_equal(top1, chunk_expected):
            mismatch = np.flatnonzero(top1 != chunk_expected)
            first_local = int(mismatch[0])
            first_global = int(start + first_local)
            raise RuntimeError(
                "Cluster assignment margin labels do not match the fitted model prediction. "
                f"first_mismatch={first_global}, predicted={int(top1[first_local])}, "
                f"expected={int(chunk_expected[first_local])}, "
                f"mismatch_count_in_chunk={int(mismatch.size)}, chunk_start={start}, chunk_stop={stop}."
            )
        rows = np.arange(stop - start)
        assigned = np.asarray(canonical_scores[rows, top1], dtype=np.float32)
        runner_up = np.asarray(canonical_scores[rows, top2], dtype=np.float32)
        assigned_score[start:stop] = assigned
        runner_up_score[start:stop] = runner_up
        runner_up_cluster[start:stop] = top2
        margin[start:stop] = assigned - runner_up

    return {
        "score_name": str(score_name),
        "higher_is_better": True,
        "assigned_score": assigned_score,
        "runner_up_score": runner_up_score,
        "runner_up_cluster": runner_up_cluster,
        "margin": margin,
    }


def representative_features_from_clustering_model(
    latents: np.ndarray,
    *,
    fitted_model: FittedClusteringModel,
    expected_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    predicted_labels, features = predict_clustering_model(
        latents,
        fitted_model,
        return_features=True,
    )
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    expected = np.asarray(expected_labels, dtype=int)
    if predicted_labels.ndim != 1 or expected.ndim != 1:
        raise ValueError(
            "Representative feature labels must have shape (N,), "
            f"got predicted={predicted_labels.shape}, expected={expected.shape}."
        )
    if predicted_labels.shape != expected.shape:
        raise ValueError(
            "Representative feature validation found a label-shape mismatch: "
            f"predicted={tuple(predicted_labels.shape)}, expected={tuple(expected.shape)}."
        )
    if not np.array_equal(predicted_labels, expected):
        mismatch = np.flatnonzero(predicted_labels != expected)
        first = int(mismatch[0])
        raise RuntimeError(
            "Representative feature labels do not match the clustering labels used for outputs. "
            f"first_mismatch={first}, predicted={int(predicted_labels[first])}, "
            f"expected={int(expected[first])}, mismatch_count={int(mismatch.size)}."
        )

    selection_features = np.asarray(features, dtype=np.float32)
    selection_space = "transformed_clustering_features"
    if str(fitted_model.method) == "spherical_kmeans":
        norms = np.linalg.norm(selection_features, axis=1, keepdims=True)
        zero_mask = np.asarray(norms <= 1.0e-8).reshape(-1)
        if np.any(zero_mask):
            first_zero = int(np.flatnonzero(zero_mask)[0])
            raise ValueError(
                "Cannot build spherical-kmeans representative features from zero-norm rows. "
                f"first_zero_row={first_zero}, shape={tuple(selection_features.shape)}."
            )
        selection_features = selection_features / norms
        selection_space = "unit_transformed_clustering_features"

    return selection_features.astype(np.float32, copy=False), {
        "selection_space": str(selection_space),
        "clustering_method": str(fitted_model.method),
        "feature_dim": int(selection_features.shape[1]),
    }


def build_clustering_method_comparison(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
    fit_latents: np.ndarray | None = None,
    requested_k_values: list[int],
    primary_method: str,
    compare_methods: list[str] | None,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    prepared_features: np.ndarray | None = None,
    prep_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[int, np.ndarray]] | None]:
    raw_methods = [str(primary_method)]
    if compare_methods:
        raw_methods.extend(str(method_name) for method_name in compare_methods)

    resolved_methods: list[tuple[str, str]] = []
    seen_methods: set[str] = set()
    for requested_method in raw_methods:
        normalized_method = _normalize_clustering_method_name(requested_method)
        if normalized_method in seen_methods:
            continue
        seen_methods.add(normalized_method)
        resolved_methods.append((str(requested_method), str(normalized_method)))

    if len(resolved_methods) < 2:
        return None, None

    methods_summary: dict[str, Any] = {}
    labels_by_method: dict[str, dict[int, np.ndarray]] = {}
    configured_k_values_ref: list[int] | None = None
    primary_k_ref: int | None = None

    for requested_method, resolved_method in resolved_methods:
        method_metrics, configured_k_values, cluster_labels_by_k, _ = _build_clustering_state(
            latents,
            phases,
            fit_latents=fit_latents,
            requested_k_values=requested_k_values,
            cluster_method=requested_method,
            random_state=random_state,
            l2_normalize=l2_normalize,
            standardize=standardize,
            pca_variance=pca_variance,
            pca_max_components=pca_max_components,
            prepared_features=prepared_features,
            prep_info=prep_info,
        )
        if configured_k_values_ref is None:
            configured_k_values_ref = [int(v) for v in configured_k_values]
            primary_k_ref = int(method_metrics["primary_k"])
        elif configured_k_values_ref != [int(v) for v in configured_k_values]:
            raise ValueError(
                "Clustering comparison methods resolved to different k-value grids, "
                f"got reference={configured_k_values_ref}, current={configured_k_values}, "
                f"method={resolved_method!r}."
            )

        methods_summary[resolved_method] = {
            "requested_method": str(requested_method),
            **method_metrics,
        }
        labels_by_method[resolved_method] = {
            int(k_value): np.asarray(labels, dtype=int)
            for k_value, labels in cluster_labels_by_k.items()
        }

    if configured_k_values_ref is None or primary_k_ref is None:
        raise RuntimeError("Clustering comparison did not collect any method outputs.")

    pairwise_by_k: dict[int, dict[str, Any]] = {}
    method_keys = [resolved_method for _, resolved_method in resolved_methods]
    for k_value in configured_k_values_ref:
        pairwise_metrics: dict[str, Any] = {}
        for left_idx, left_method in enumerate(method_keys):
            for right_method in method_keys[left_idx + 1 :]:
                labels_left = labels_by_method[left_method][int(k_value)]
                labels_right = labels_by_method[right_method][int(k_value)]
                pairwise_key = f"{left_method}__vs__{right_method}"
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

                pairwise_metrics[pairwise_key] = {
                    "ari": float(adjusted_rand_score(labels_left, labels_right)),
                    "nmi": float(normalized_mutual_info_score(labels_left, labels_right)),
                }
        pairwise_by_k[int(k_value)] = pairwise_metrics

    return (
        {
            "methods_requested": [requested for requested, _ in resolved_methods],
            "methods_resolved": [resolved for _, resolved in resolved_methods],
            "k_values_used": [int(v) for v in configured_k_values_ref],
            "primary_k": int(primary_k_ref),
            "methods": methods_summary,
            "pairwise_by_k": pairwise_by_k,
        },
        labels_by_method,
    )


def prepare_clustering_features(
    latents: np.ndarray,
    *,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Thin wrapper around _prepare_clustering_features for public use."""
    return _prepare_clustering_features(
        latents,
        random_state=random_state,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )


def _run_optional_hdbscan_analysis(
    latents: np.ndarray,
    *,
    coords_count: int,
    settings: Any,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    prepared_features: np.ndarray | None,
    prep_info: dict[str, Any] | None,
    cluster_color_assignment: dict[int, int | str] | None,
    step: Callable[[str], None],
) -> HDBSCANResult:
    if not settings.enabled:
        return HDBSCANResult(labels=None, info=None, color_map=None)

    step("Running HDBSCAN clustering (sampled fit)")
    hdbscan_labels, hdbscan_info = compute_hdbscan_labels(
        latents,
        sample_fraction=settings.fit_fraction,
        max_fit_samples=settings.max_fit_samples,
        random_state=random_state,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
        target_clusters_min=settings.target_k_min,
        target_clusters_max=settings.target_k_max,
        min_cluster_size_candidates=settings.min_cluster_size_candidates,
        min_samples=settings.min_samples,
        min_samples_candidates=settings.min_samples_candidates,
        cluster_selection_epsilon=settings.cluster_selection_epsilon,
        cluster_selection_method=settings.cluster_selection_method,
        refit_full_data=settings.refit_full_data,
        prepared_features=prepared_features,
        prep_info=prep_info,
        return_info=True,
    )
    if hdbscan_info["status"] != "ok":
        raise RuntimeError(
            "HDBSCAN did not produce a valid fitted model. "
            f"status={hdbscan_info['status']!r}, info={hdbscan_info}."
        )
    if hdbscan_labels.size != int(coords_count):
        raise ValueError(
            "HDBSCAN labels must match the coordinate count. "
            f"labels={hdbscan_labels.shape}, coords_count={int(coords_count)}."
        )

    valid_hdbscan = hdbscan_labels[hdbscan_labels >= 0]
    if valid_hdbscan.size > 0:
        hdbscan_color_map = {
            int(cluster_id): str(color)
            for cluster_id, color in _build_cluster_color_map(
                hdbscan_labels,
                cluster_color_assignment=cluster_color_assignment,
            ).items()
        }
    else:
        hdbscan_color_map = {}
    if np.any(hdbscan_labels < 0):
        hdbscan_color_map[-1] = "lightgray"
    return HDBSCANResult(
        labels=np.asarray(hdbscan_labels, dtype=int),
        info=hdbscan_info,
        color_map=hdbscan_color_map,
    )
