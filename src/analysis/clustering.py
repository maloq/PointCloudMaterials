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
    _normalize_clustering_method_name,
    _prepare_clustering_features,
    compute_hdbscan_labels,
    compute_kmeans_labels,
)


@dataclass(frozen=True)
class HDBSCANResult:
    labels: np.ndarray | None
    info: dict[str, Any] | None
    color_map: dict[int, str] | None


def _resolve_cluster_k_values(k_values: list[int], *, n_samples: int) -> list[int]:
    resolved = [
        max(2, min(int(k), int(n_samples)))
        for k in k_values
        if int(k) >= 2
    ]
    resolved = list(dict.fromkeys(resolved))
    if resolved:
        return resolved
    return [max(2, min(int(n_samples), 3))]


def _build_clustering_state(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
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
        cluster_labels_by_k[int(k_value)] = labels_k
        cluster_methods_by_k[int(k_value)] = str(info_k.get("method", "kmeans"))
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
            metrics["ari_with_gt"] = float(adjusted_rand_score(phases, gt_labels))
            metrics["nmi_with_gt"] = float(normalized_mutual_info_score(phases, gt_labels))

    return metrics, configured_k_values, cluster_labels_by_k, cluster_methods_by_k


def build_clustering_method_comparison(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
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
    try:
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
        n_hdb_clusters_full = int(hdbscan_info.get("n_clusters_full", -1))
        if (
            settings.cluster_selection_method != "auto"
            and n_hdb_clusters_full >= 0
            and n_hdb_clusters_full < settings.target_k_min
        ):
            print(
                "[analysis][hdbscan] cluster count below target "
                f"({n_hdb_clusters_full} < {settings.target_k_min}); "
                "retrying with cluster_selection_method='auto'."
            )
            hdbscan_labels_retry, hdbscan_info_retry = compute_hdbscan_labels(
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
                cluster_selection_method="auto",
                refit_full_data=settings.refit_full_data,
                prepared_features=prepared_features,
                prep_info=prep_info,
                return_info=True,
            )
            retry_clusters = int(hdbscan_info_retry.get("n_clusters_full", -1))
            retry_noise = float(hdbscan_info_retry.get("noise_fraction_full", 1.0))
            base_noise = float(hdbscan_info.get("noise_fraction_full", 1.0))
            if (
                retry_clusters > n_hdb_clusters_full
                or (retry_clusters == n_hdb_clusters_full and retry_noise < base_noise)
            ):
                hdbscan_labels = hdbscan_labels_retry
                hdbscan_info = hdbscan_info_retry
                print(
                    "[analysis][hdbscan] using retry result: "
                    f"clusters={retry_clusters}, noise={retry_noise:.4f}."
                )
        if hdbscan_labels.size != int(coords_count):
            print(
                "Warning: HDBSCAN labels do not match coordinate count; "
                "skipping HDBSCAN MD outputs."
            )
            return HDBSCANResult(labels=None, info=hdbscan_info, color_map=None)

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
    except ImportError:
        print("HDBSCAN package not installed; skipping HDBSCAN analysis.")
        return HDBSCANResult(labels=None, info=None, color_map=None)
