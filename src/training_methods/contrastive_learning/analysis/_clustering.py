"""Clustering logic for post-training analysis.

Provides k-means and HDBSCAN clustering on invariant latent features,
with shared feature preparation and color-map construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from src.training_methods.contrastive_learning.cluster_figure_utils import (
    _build_cluster_color_map,
)
from src.vis_tools.latent_analysis_vis import (
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
    metrics: dict[str, Any] = {
        "cluster_method_requested": str(cluster_method).lower(),
        "random_state": int(random_state),
        "k_values_requested": [int(k) for k in requested_k_values],
        "k_values_used": [int(k) for k in configured_k_values],
        "primary_k": int(primary_k),
        "labels_k_method": str(cluster_methods_by_k[int(primary_k)]),
        "labels_method_by_k": {
            int(k): str(cluster_methods_by_k[int(k)])
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
