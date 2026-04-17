from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot


def _log_saved_figure(path: Path | str) -> None:
    print(f"[analysis][savefig] {Path(path).resolve()}")


def _tab10_colors(n_colors: int) -> np.ndarray:
    if int(n_colors) <= 0:
        return np.empty((0, 4), dtype=np.float32)
    base = plt.cm.tab10(np.linspace(0, 1, 10)).astype(np.float32)
    n = int(n_colors)
    if n <= base.shape[0]:
        return base[:n]
    extras = plt.cm.tab20(np.linspace(0, 1, 20)).astype(np.float32)
    colors = [base[i] for i in range(base.shape[0])]
    for extra in extras:
        if len(colors) >= n:
            break
        if not any(np.allclose(extra, c, atol=1e-6) for c in colors):
            colors.append(extra)
    out = np.asarray(colors, dtype=np.float32)
    if out.shape[0] < n:
        raise RuntimeError(
            f"Failed to create {n} colors for tab10 palette; got {out.shape[0]}."
        )
    return out[:n]


def save_latent_tsne(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    max_samples: int | None = None,
    class_names: Dict[int, str] | None = None,
    random_state: int = 42,
) -> None:
    """Save t-SNE plots: one with ground truth phases, one with clustering results."""
    if inv_latents.size == 0 or len(inv_latents) < 2:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents = inv_latents
    has_phases = phases.size == len(latents)
    gt_labels = phases if has_phases else None

    if max_samples is not None and len(latents) > max_samples:
        idx = np.random.default_rng(int(random_state)).choice(
            len(latents),
            size=max_samples,
            replace=False,
        )
        latents = latents[idx]
        if gt_labels is not None:
            gt_labels = gt_labels[idx]

    perplexity = min(50, max(5, len(latents) // 100))
    tsne_coords = compute_tsne(
        latents,
        random_state=int(random_state),
        perplexity=perplexity,
        n_iter=1500,
    )

    if gt_labels is not None:
        save_tsne_plot(
            tsne_coords,
            gt_labels,
            out_file=str(out_dir / "latent_tsne_ground_truth.png"),
            title=f"Latent space t-SNE (n={len(latents)}, ground truth phases)",
            legend_title="phase",
            class_names=class_names,
        )

    n_clusters = len(np.unique(gt_labels)) if gt_labels is not None else 4
    n_clusters = max(2, min(n_clusters, len(latents) // 2))

    kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
    cluster_labels = kmeans.fit_predict(latents).astype(int, copy=False)
    cluster_labels, _ = _canonicalize_cluster_labels(cluster_labels, latents)

    save_tsne_plot(
        tsne_coords,
        cluster_labels,
        out_file=str(out_dir / "latent_tsne_clusters.png"),
        title=f"Latent space t-SNE (KMeans k={n_clusters})",
        legend_title="cluster",
    )


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _l2_normalize_rows_strict(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    context: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"{context} must be a 2D array, got shape={tuple(arr.shape)}."
        )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero_mask = np.asarray(norms <= float(eps)).reshape(-1)
    if np.any(zero_mask):
        zero_count = int(np.count_nonzero(zero_mask))
        raise ValueError(
            f"{context} contains {zero_count} zero-norm row(s), which cannot be "
            "projected onto the unit sphere for spherical k-means."
        )
    return arr / norms


def _normalize_clustering_method_name(method: str) -> str:
    raw_method = str(method).strip()
    if raw_method == "":
        raise ValueError("Clustering method must be a non-empty string.")
    key = raw_method.lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in {"auto", "kmeans", "euclideankmeans", "regularkmeans", "standardkmeans"}:
        return "kmeans"
    if key in {"spherical", "sphericalkmeans"}:
        return "spherical_kmeans"
    raise ValueError(
        "Unsupported clustering method "
        f"{method!r}. Supported values: auto, kmeans, spherical_kmeans."
    )


@dataclass(frozen=True)
class ClusteringFeatureTransform:
    input_dim: int
    output_dim: int
    l2_normalize: bool
    standardize: bool
    scaler: Any | None
    pca: Any | None
    pca_components: int
    pca_explained_variance: float


@dataclass(frozen=True)
class FittedClusteringModel:
    method: str
    requested_method: str
    n_clusters: int
    random_state: int
    feature_transform: ClusteringFeatureTransform
    label_remap: dict[int, int]
    fit_info: dict[str, Any]
    sklearn_model: Any | None = None
    spherical_centers: np.ndarray | None = None


def _compute_internal_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
    max_samples: int = 3000,
) -> Dict[str, Any]:
    feats = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    if feats.ndim != 2:
        raise ValueError(
            "Internal clustering metric features must be 2D, "
            f"got shape={tuple(feats.shape)}."
        )
    if feats.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            "Internal clustering metrics require matching feature/label lengths, "
            f"got features.shape[0]={feats.shape[0]} and labels.shape[0]={labels_arr.shape[0]}."
        )

    cluster_ids, cluster_counts = np.unique(labels_arr, return_counts=True)
    metrics: Dict[str, Any] = {
        "cluster_counts": {
            int(cluster_id): int(count)
            for cluster_id, count in zip(cluster_ids, cluster_counts)
        },
        "n_clusters_observed": int(cluster_ids.size),
        "smallest_cluster_size": int(cluster_counts.min()),
        "largest_cluster_size": int(cluster_counts.max()),
        "cluster_validation_sample_size": int(min(int(max_samples), feats.shape[0])),
    }
    fractions = cluster_counts.astype(np.float64) / float(cluster_counts.sum())
    entropy = -np.sum(fractions * np.log(np.clip(fractions, 1.0e-12, None)))
    metrics["cluster_size_entropy"] = float(entropy)
    metrics["cluster_size_entropy_normalized"] = float(
        entropy / np.log(float(cluster_ids.size))
    ) if int(cluster_ids.size) > 1 else 0.0
    metrics["largest_cluster_fraction"] = float(fractions.max())
    metrics["smallest_cluster_fraction"] = float(fractions.min())

    if cluster_ids.size < 2 or cluster_ids.size >= feats.shape[0]:
        metrics["internal_validation_skipped"] = (
            "Requires 2 <= number of clusters < number of samples."
        )
        return metrics

    sample_size = min(int(max_samples), feats.shape[0])
    if sample_size < feats.shape[0]:
        sample_idx = np.random.default_rng(int(random_state)).choice(
            feats.shape[0],
            size=sample_size,
            replace=False,
        )
        feats_eval = feats[sample_idx]
        labels_eval = labels_arr[sample_idx]
    else:
        feats_eval = feats
        labels_eval = labels_arr

    sampled_cluster_ids = np.unique(labels_eval)
    if sampled_cluster_ids.size < 2 or sampled_cluster_ids.size >= feats_eval.shape[0]:
        metrics["internal_validation_skipped"] = (
            "Sampled subset does not contain a valid multi-cluster partition."
        )
        return metrics

    unit_feats_eval = _l2_normalize_rows_strict(
        feats_eval,
        context="Internal clustering metric features",
    )
    metrics["silhouette_euclidean"] = float(
        silhouette_score(feats_eval, labels_eval, metric="euclidean")
    )
    metrics["silhouette_cosine"] = float(
        silhouette_score(unit_feats_eval, labels_eval, metric="cosine")
    )
    metrics["calinski_harabasz"] = float(
        calinski_harabasz_score(feats_eval, labels_eval)
    )
    metrics["davies_bouldin"] = float(
        davies_bouldin_score(feats_eval, labels_eval)
    )
    return metrics


def fit_clustering_feature_transform(
    latents: np.ndarray,
    *,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, ClusteringFeatureTransform, Dict[str, Any]]:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    info: Dict[str, Any] = {
        "input_dim": int(x.shape[1]),
        "l2_normalize": bool(l2_normalize),
        "standardize": bool(standardize),
    }

    if l2_normalize:
        x = _l2_normalize_rows(x)

    scaler = None
    if standardize and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    pca_model = None
    keep_components = int(x.shape[1])
    explained_variance = 1.0
    use_pca = (
        pca_variance is not None
        and float(pca_variance) > 0.0
        and x.shape[1] > 2
        and x.shape[0] > 3
    )
    if use_pca:
        n_max = min(int(pca_max_components), x.shape[1], x.shape[0] - 1)
        if n_max >= 2:
            pca_model = PCA(n_components=n_max, random_state=random_state)
            x_proj = pca_model.fit_transform(x)
            if float(pca_variance) >= 1.0:
                keep_components = n_max
            else:
                csum = np.cumsum(pca_model.explained_variance_ratio_)
                keep_components = int(np.searchsorted(csum, float(pca_variance)) + 1)
                keep_components = max(2, min(keep_components, n_max))
            x = x_proj[:, :keep_components]
            explained_variance = float(
                np.sum(pca_model.explained_variance_ratio_[:keep_components])
            )

    info["pca_components"] = int(keep_components)
    info["pca_explained_variance"] = float(explained_variance)
    info["output_dim"] = int(x.shape[1])

    transform = ClusteringFeatureTransform(
        input_dim=int(info["input_dim"]),
        output_dim=int(info["output_dim"]),
        l2_normalize=bool(l2_normalize),
        standardize=bool(standardize),
        scaler=scaler,
        pca=pca_model,
        pca_components=int(keep_components),
        pca_explained_variance=float(explained_variance),
    )
    return x.astype(np.float32, copy=False), transform, info


def transform_clustering_features(
    latents: np.ndarray,
    *,
    transform: ClusteringFeatureTransform,
) -> np.ndarray:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if int(x.shape[1]) != int(transform.input_dim):
        raise ValueError(
            "Clustering feature transform input dimension mismatch. "
            f"Expected {int(transform.input_dim)}, got {int(x.shape[1])}."
        )

    if bool(transform.l2_normalize):
        x = _l2_normalize_rows(x)
    if transform.scaler is not None:
        x = transform.scaler.transform(x)
    if transform.pca is not None:
        x = transform.pca.transform(x)[:, : int(transform.pca_components)]
    return np.asarray(x, dtype=np.float32)


def _remap_cluster_labels(
    labels: np.ndarray,
    *,
    label_remap: dict[int, int],
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    remapped = labels_arr.copy()
    for old_label, new_label in label_remap.items():
        remapped[labels_arr == int(old_label)] = int(new_label)
    return remapped.astype(int, copy=False)


def _fit_spherical_kmeans_model(
    features: np.ndarray,
    n_clusters: int,
    *,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    feats_unit = _l2_normalize_rows_strict(
        features,
        context="Spherical k-means input features",
    )
    n_samples = int(feats_unit.shape[0])
    if n_clusters < 2 or n_clusters > n_samples:
        raise ValueError(
            "spherical k-means requires 2 <= n_clusters <= n_samples, "
            f"got n_clusters={n_clusters}, n_samples={n_samples}."
        )

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_objective = -np.inf
    best_info: Dict[str, Any] | None = None

    for init_idx in range(int(n_init)):
        init_seed = int(random_state) + int(init_idx)
        centers_init, center_indices = kmeans_plusplus(
            feats_unit,
            n_clusters=int(n_clusters),
            random_state=int(init_seed),
        )
        centers = _l2_normalize_rows_strict(
            centers_init,
            context=f"Spherical k-means initial centers (init {init_idx})",
        )
        prev_labels: np.ndarray | None = None
        empty_cluster_reassignments = 0
        final_iteration = 0
        converged = False

        for iteration in range(1, int(max_iter) + 1):
            similarities = feats_unit @ centers.T
            labels = np.argmax(similarities, axis=1).astype(int, copy=False)
            assigned_similarity = similarities[np.arange(n_samples), labels]

            new_centers = np.zeros_like(centers)
            empty_clusters: list[int] = []
            for cluster_id in range(int(n_clusters)):
                mask = labels == int(cluster_id)
                if not np.any(mask):
                    empty_clusters.append(int(cluster_id))
                    continue
                centroid = np.mean(feats_unit[mask], axis=0, dtype=np.float64)
                centroid_norm = float(np.linalg.norm(centroid))
                if centroid_norm <= 1.0e-12:
                    raise ValueError(
                        "Spherical k-means produced a zero-norm centroid for "
                        f"cluster_id={cluster_id}, iteration={iteration}, init_idx={init_idx}."
                    )
                new_centers[cluster_id] = (
                    centroid / centroid_norm
                ).astype(np.float32, copy=False)

            if empty_clusters:
                candidate_order = np.argsort(assigned_similarity, kind="stable")
                used_candidate_indices: set[int] = set()
                candidate_cursor = 0
                for cluster_id in empty_clusters:
                    while (
                        candidate_cursor < candidate_order.size
                        and int(candidate_order[candidate_cursor]) in used_candidate_indices
                    ):
                        candidate_cursor += 1
                    if candidate_cursor >= candidate_order.size:
                        raise RuntimeError(
                            "Failed to reseed an empty spherical k-means cluster. "
                            f"iteration={iteration}, init_idx={init_idx}, "
                            f"empty_clusters={empty_clusters}."
                        )
                    sample_idx = int(candidate_order[candidate_cursor])
                    used_candidate_indices.add(sample_idx)
                    new_centers[cluster_id] = feats_unit[sample_idx]
                    empty_cluster_reassignments += 1
                    candidate_cursor += 1

            center_alignment = np.sum(centers * new_centers, axis=1)
            center_shift = float(
                np.max(1.0 - np.clip(center_alignment, -1.0, 1.0))
            )
            centers = new_centers
            final_iteration = int(iteration)
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                converged = True
                break
            if center_shift <= float(tol):
                converged = True
                break
            prev_labels = labels.copy()

        final_similarities = feats_unit @ centers.T
        final_labels = np.argmax(final_similarities, axis=1).astype(int, copy=False)
        final_assigned_similarity = final_similarities[np.arange(n_samples), final_labels]
        objective = float(np.sum(final_assigned_similarity, dtype=np.float64))
        spherical_inertia = float(
            np.sum(1.0 - final_assigned_similarity, dtype=np.float64)
        )

        if objective > best_objective:
            best_objective = objective
            best_labels = final_labels.copy()
            best_centers = centers.astype(np.float32, copy=True)
            best_info = {
                "requested_method": "spherical_kmeans",
                "method": "spherical_kmeans",
                "fallback_used": False,
                "model_score_name": "spherical_inertia",
                "model_score": float(spherical_inertia),
                "mean_assigned_cosine_similarity": float(
                    np.mean(final_assigned_similarity, dtype=np.float64)
                ),
                "sum_assigned_cosine_similarity": float(objective),
                "n_init": int(n_init),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "best_init_index": int(init_idx),
                "best_init_seed": int(init_seed),
                "iterations_run": int(final_iteration),
                "converged": bool(converged),
                "empty_cluster_reassignments": int(empty_cluster_reassignments),
                "initial_center_indices": [int(v) for v in np.asarray(center_indices, dtype=int)],
                "random_state": int(random_state),
            }

    if best_labels is None or best_centers is None or best_info is None:
        raise RuntimeError("Spherical k-means failed to produce a valid clustering result.")
    return best_labels, best_centers, best_info, feats_unit


def _run_spherical_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1.0e-4,
) -> tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    labels, _, fit_info, feats_unit = _fit_spherical_kmeans_model(
        features,
        n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    return labels, fit_info, feats_unit


def _prepare_clustering_features(
    latents: np.ndarray,
    *,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    x, _, info = fit_clustering_feature_transform(
        latents,
        random_state=random_state,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )
    return x, info


def _resolve_clustering_features(
    latents: np.ndarray,
    *,
    prepared_features: np.ndarray | None,
    prep_info: Dict[str, Any] | None,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if prepared_features is None:
        return _prepare_clustering_features(
            latents,
            random_state=random_state,
            l2_normalize=l2_normalize,
            standardize=standardize,
            pca_variance=pca_variance,
            pca_max_components=pca_max_components,
        )

    latents_arr = np.asarray(latents, dtype=np.float32)
    if latents_arr.ndim != 2:
        latents_arr = np.reshape(latents_arr, (latents_arr.shape[0], -1))
    features = np.asarray(prepared_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            "prepared_features must be a 2D array of shape (N, D), "
            f"got {features.shape}."
        )
    if features.shape[0] != latents_arr.shape[0]:
        raise ValueError(
            "prepared_features row count must match the number of latent samples, "
            f"got features.shape[0]={features.shape[0]}, latents.shape[0]={latents_arr.shape[0]}."
        )

    resolved_info = (
        dict(prep_info)
        if prep_info is not None
        else {
            "input_dim": int(latents_arr.shape[1]),
            "output_dim": int(features.shape[1]),
            "l2_normalize": bool(l2_normalize),
            "standardize": bool(standardize),
            "pca_components": int(features.shape[1]),
            "pca_explained_variance": 1.0,
        }
    )
    return features.astype(np.float32, copy=False), resolved_info


def fit_clustering_model(
    latents: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
    method: str = "auto",
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_variance: float | None = 0.98,
    pca_max_components: int = 32,
) -> tuple[FittedClusteringModel, np.ndarray, Dict[str, Any]]:
    if latents.size == 0 or len(latents) < 2:
        raise ValueError("Cannot fit a clustering model on fewer than 2 samples.")

    n_clusters = max(2, min(int(n_clusters), len(latents)))
    fit_features, feature_transform, prep_info = fit_clustering_feature_transform(
        latents,
        random_state=int(random_state),
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )
    resolved_method = _normalize_clustering_method_name(method)

    sklearn_model = None
    spherical_centers = None
    if resolved_method == "kmeans":
        sklearn_model = KMeans(
            n_clusters=int(n_clusters),
            random_state=int(random_state),
            n_init=20,
        )
        raw_fit_labels = sklearn_model.fit_predict(fit_features).astype(int, copy=False)
        canonical_features = fit_features
        fit_info: Dict[str, Any] = {
            "method": "kmeans",
            "requested_method": str(method),
            "fallback_used": False,
            "model_score_name": "inertia",
            "model_score": float(sklearn_model.inertia_),
            "random_state": int(random_state),
        }
    elif resolved_method == "spherical_kmeans":
        raw_fit_labels, spherical_centers, fit_info, canonical_features = _fit_spherical_kmeans_model(
            fit_features,
            int(n_clusters),
            random_state=int(random_state),
        )
    else:
        raise AssertionError(f"Unhandled resolved clustering method {resolved_method!r}.")

    fit_labels, canonical_info = _canonicalize_cluster_labels(
        raw_fit_labels,
        canonical_features,
    )
    fit_internal_metrics = _compute_internal_clustering_metrics(
        fit_features,
        fit_labels,
        random_state=int(random_state),
    )
    fit_summary = {
        **prep_info,
        **fit_info,
        **canonical_info,
        **fit_internal_metrics,
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
        "fit_sample_count": int(len(latents)),
    }
    fitted_model = FittedClusteringModel(
        method=str(resolved_method),
        requested_method=str(method),
        n_clusters=int(n_clusters),
        random_state=int(random_state),
        feature_transform=feature_transform,
        label_remap={
            int(old): int(new)
            for old, new in canonical_info.get("cluster_label_remap", {}).items()
        },
        fit_info=dict(fit_summary),
        sklearn_model=sklearn_model,
        spherical_centers=(
            None if spherical_centers is None else np.asarray(spherical_centers, dtype=np.float32)
        ),
    )
    return fitted_model, fit_labels.astype(int, copy=False), fit_summary


def predict_clustering_model(
    latents: np.ndarray,
    fitted_model: FittedClusteringModel,
    *,
    return_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    features = transform_clustering_features(
        latents,
        transform=fitted_model.feature_transform,
    )
    if fitted_model.method == "kmeans":
        if fitted_model.sklearn_model is None:
            raise RuntimeError("Fitted k-means model is missing the sklearn model state.")
        raw_labels = fitted_model.sklearn_model.predict(features).astype(int, copy=False)
    elif fitted_model.method == "spherical_kmeans":
        if fitted_model.spherical_centers is None:
            raise RuntimeError(
                "Fitted spherical k-means model is missing the spherical centers."
            )
        features_unit = _l2_normalize_rows_strict(
            features,
            context="Spherical k-means prediction features",
        )
        raw_labels = np.argmax(
            features_unit @ fitted_model.spherical_centers.T,
            axis=1,
        ).astype(int, copy=False)
    else:
        raise ValueError(
            f"Unsupported fitted clustering method {fitted_model.method!r}."
        )

    labels = _remap_cluster_labels(
        raw_labels,
        label_remap=fitted_model.label_remap,
    )
    if return_features:
        return labels, features
    return labels


def compute_transfer_kmeans_labels(
    fit_latents: np.ndarray,
    target_latents: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
    method: str = "auto",
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_variance: float | None = 0.98,
    pca_max_components: int = 32,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    fitted_model, fit_labels, fit_info = fit_clustering_model(
        fit_latents,
        n_clusters,
        random_state=int(random_state),
        method=method,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )
    target_labels, target_features = predict_clustering_model(
        target_latents,
        fitted_model,
        return_features=True,
    )
    target_metrics = _compute_internal_clustering_metrics(
        target_features,
        target_labels,
        random_state=int(random_state),
    )
    info = {
        **{
            key: value
            for key, value in fit_info.items()
            if key
            not in {
                "cluster_counts",
                "n_clusters_observed",
                "smallest_cluster_size",
                "largest_cluster_size",
                "cluster_validation_sample_size",
                "cluster_size_entropy",
                "cluster_size_entropy_normalized",
                "largest_cluster_fraction",
                "smallest_cluster_fraction",
                "silhouette_euclidean",
                "silhouette_cosine",
                "calinski_harabasz",
                "davies_bouldin",
                "internal_validation_skipped",
            }
        },
        **target_metrics,
        "fit_sample_count": int(len(fit_latents)),
        "target_sample_count": int(len(target_latents)),
        "transfer_fit_enabled": True,
        "fit_cluster_counts": {
            int(k): int(v)
            for k, v in zip(*np.unique(fit_labels, return_counts=True))
        },
    }
    if return_info:
        return target_labels.astype(int, copy=False), info
    return target_labels.astype(int, copy=False)


def _canonicalize_cluster_labels(
    labels: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    feats = np.asarray(features, dtype=np.float32)
    valid_ids = sorted(int(v) for v in np.unique(labels_arr) if int(v) >= 0)
    if not valid_ids:
        return labels_arr.copy(), {
            "cluster_label_canonicalization": "none",
            "cluster_label_remap": {},
        }

    centroids_by_label = {
        int(cluster_id): tuple(
            float(v)
            for v in np.mean(feats[labels_arr == cluster_id], axis=0, dtype=np.float64).tolist()
        )
        for cluster_id in valid_ids
    }
    ordered_ids = sorted(
        valid_ids,
        key=lambda cluster_id: centroids_by_label[int(cluster_id)] + (int(cluster_id),),
    )
    remap = {int(old): int(new) for new, old in enumerate(ordered_ids)}
    if all(int(old) == int(new) for old, new in remap.items()):
        return labels_arr.copy(), {
            "cluster_label_canonicalization": "feature_centroid_lexicographic",
            "cluster_label_remap": remap,
        }

    canonical = labels_arr.copy()
    for old_label, new_label in remap.items():
        canonical[labels_arr == int(old_label)] = int(new_label)
    return canonical.astype(int, copy=False), {
        "cluster_label_canonicalization": "feature_centroid_lexicographic",
        "cluster_label_remap": remap,
    }


def _cluster_with_method_selection(
    features: np.ndarray,
    n_clusters: int,
    *,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    features_arr = np.asarray(features, dtype=np.float32)
    resolved_method = _normalize_clustering_method_name(method)

    if resolved_method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(features_arr).astype(int, copy=False)
        canonical_features = features_arr
        fit_info: Dict[str, Any] = {
            "method": "kmeans",
            "requested_method": str(method),
            "fallback_used": False,
            "model_score_name": "inertia",
            "model_score": float(model.inertia_),
            "random_state": int(random_state),
        }
    elif resolved_method == "spherical_kmeans":
        labels, fit_info, canonical_features = _run_spherical_kmeans(
            features_arr,
            int(n_clusters),
            random_state=int(random_state),
        )
    else:
        raise AssertionError(f"Unhandled resolved clustering method {resolved_method!r}.")

    labels_canonical, canonical_info = _canonicalize_cluster_labels(
        labels,
        canonical_features,
    )
    internal_metrics = _compute_internal_clustering_metrics(
        features_arr,
        labels_canonical,
        random_state=int(random_state),
    )
    return labels_canonical, {
        **fit_info,
        **canonical_info,
        **internal_metrics,
    }


def compute_kmeans_labels(
    latents: np.ndarray,
    n_clusters: int,
    *,
    max_samples: int | None = None,
    random_state: int = 42,
    method: str = "auto",
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_variance: float | None = 0.98,
    pca_max_components: int = 32,
    prepared_features: np.ndarray | None = None,
    prep_info: Dict[str, Any] | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    if latents.size == 0 or len(latents) < 2:
        empty = np.empty((0,), dtype=int)
        if return_info:
            return empty, {"method": "none"}
        return empty
    n_clusters = max(2, min(int(n_clusters), len(latents)))
    features, prep_info_resolved = _resolve_clustering_features(
        latents,
        prepared_features=prepared_features,
        prep_info=prep_info,
        random_state=random_state,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )

    if max_samples is not None and len(features) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(features), size=max_samples, replace=False)
        features_fit = features[idx]
    else:
        features_fit = features

    labels_fit, fit_info = _cluster_with_method_selection(
        features_fit,
        n_clusters,
        method=method,
        random_state=random_state,
    )
    if len(features_fit) == len(features):
        labels = labels_fit
    else:
        # Refit on full preprocessed features with selected method for consistent labels.
        selected_method = str(fit_info.get("method", "kmeans"))
        labels, fit_info = _cluster_with_method_selection(
            features,
            n_clusters,
            method=selected_method,
            random_state=random_state,
        )

    info = {
        **prep_info_resolved,
        **fit_info,
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
    }
    if return_info:
        return labels, info
    return labels


def compute_hdbscan_labels(
    latents: np.ndarray,
    *,
    sample_fraction: float = 0.25,
    max_fit_samples: int | None = 50000,
    random_state: int = 42,
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_variance: float | None = 0.98,
    pca_max_components: int = 32,
    target_clusters_min: int = 5,
    target_clusters_max: int = 6,
    min_cluster_size_candidates: list[int] | None = None,
    min_samples: int | None = None,
    min_samples_candidates: list[int] | None = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "leaf",
    refit_full_data: bool = True,
    prepared_features: np.ndarray | None = None,
    prep_info: Dict[str, Any] | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    if latents.size == 0 or len(latents) < 2:
        empty = np.empty((0,), dtype=int)
        if return_info:
            return empty, {"method": "hdbscan", "status": "empty"}
        return empty

    if min_samples is not None and min_samples_candidates is not None:
        raise ValueError(
            "Specify only one of min_samples or min_samples_candidates."
        )
    if not np.isfinite(sample_fraction) or float(sample_fraction) <= 0.0:
        raise ValueError(
            f"sample_fraction must be finite and > 0, got {sample_fraction}."
        )
    if float(sample_fraction) > 1.0:
        raise ValueError(
            f"sample_fraction must be <= 1.0, got {sample_fraction}."
        )
    if max_fit_samples is not None and int(max_fit_samples) < 2:
        raise ValueError(
            f"max_fit_samples must be >= 2 when provided, got {max_fit_samples}."
        )
    if min_samples is not None and int(min_samples) < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}.")
    if not np.isfinite(cluster_selection_epsilon) or float(cluster_selection_epsilon) < 0.0:
        raise ValueError(
            "cluster_selection_epsilon must be finite and >= 0, got "
            f"{cluster_selection_epsilon}."
        )

    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requested but 'hdbscan' is not installed."
        ) from exc

    features, prep_info_resolved = _resolve_clustering_features(
        latents,
        prepared_features=prepared_features,
        prep_info=prep_info,
        random_state=random_state,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )
    n_total = len(features)
    fit_size = max(2, int(np.ceil(float(sample_fraction) * n_total)))
    if max_fit_samples is not None:
        fit_size = min(fit_size, max(2, int(max_fit_samples)))
    fit_size = min(fit_size, n_total)

    rng = np.random.default_rng(random_state)
    if fit_size < n_total:
        fit_idx = rng.choice(n_total, size=fit_size, replace=False)
    else:
        fit_idx = np.arange(n_total)
    fit_features = features[fit_idx]

    if min_cluster_size_candidates is None or len(min_cluster_size_candidates) == 0:
        ratio_candidates = [0.0010, 0.0015, 0.0025, 0.0040, 0.0060, 0.0090, 0.0130, 0.0200, 0.0300, 0.0500]
        abs_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64]
        min_cluster_size_candidates = sorted(
            {
                max(2, min(fit_size, int(round(fit_size * ratio))))
                for ratio in ratio_candidates
            }
            | {
                max(2, min(fit_size, int(v)))
                for v in abs_candidates
                if int(v) <= fit_size
            }
        )
    else:
        min_cluster_size_candidates = sorted(
            {
                max(2, min(fit_size, int(val)))
                for val in min_cluster_size_candidates
                if int(val) >= 2
            }
        )
    if not min_cluster_size_candidates:
        min_cluster_size_candidates = [max(2, min(fit_size, fit_size // 40))]

    requested_selection_method = str(cluster_selection_method).strip().lower()
    if requested_selection_method == "auto":
        selection_methods = ["leaf", "eom"]
    elif requested_selection_method in {"leaf", "eom"}:
        selection_methods = [requested_selection_method]
    else:
        raise ValueError(
            "cluster_selection_method must be one of ['leaf', 'eom', 'auto'], "
            f"got {cluster_selection_method!r}."
        )

    target_low = max(2, int(target_clusters_min))
    target_high = max(target_low, int(target_clusters_max))
    target_mid = 0.5 * (target_low + target_high)

    best: Dict[str, Any] | None = None
    best_clusterer = None
    best_labels_fit: np.ndarray | None = None
    best_selection_method: str | None = None
    best_score = None
    fit_failures: list[tuple[int, int, str, Exception]] = []
    trials_evaluated = 0

    def _resolve_min_samples_candidates(mcs: int) -> list[int]:
        mcs = int(mcs)
        if mcs < 1:
            raise ValueError(f"Internal error: min_cluster_size must be >= 1, got {mcs}.")
        if min_samples is not None:
            return [max(1, min(mcs, int(min_samples)))]
        if min_samples_candidates is not None and len(min_samples_candidates) > 0:
            resolved = sorted(
                {
                    max(1, min(mcs, int(v)))
                    for v in min_samples_candidates
                    if int(v) >= 1
                }
            )
            if resolved:
                return resolved
        auto_candidates = {
            1,
            2,
            3,
            int(round(mcs * 0.05)),
            int(round(mcs * 0.10)),
            int(round(mcs * 0.20)),
            int(round(np.sqrt(float(mcs)))),
            int(round(np.log2(float(mcs) + 1.0))),
        }
        resolved_auto = sorted(
            {
                max(1, min(mcs, int(v)))
                for v in auto_candidates
                if int(v) >= 1
            }
        )
        if not resolved_auto:
            return [1]
        return resolved_auto

    for mcs in min_cluster_size_candidates:
        min_samples_grid = _resolve_min_samples_candidates(int(mcs))
        for method_name in selection_methods:
            for ms in min_samples_grid:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=int(mcs),
                        min_samples=int(ms),
                        cluster_selection_method=str(method_name),
                        cluster_selection_epsilon=float(cluster_selection_epsilon),
                        prediction_data=True,
                    )
                    labels_fit = clusterer.fit_predict(fit_features).astype(int)
                except Exception as exc:
                    fit_failures.append((int(mcs), int(ms), str(method_name), exc))
                    continue
                trials_evaluated += 1

                valid = labels_fit[labels_fit >= 0]
                n_clusters = int(len(np.unique(valid)))
                noise_frac = float(np.mean(labels_fit < 0))
                if valid.size > 0:
                    _, valid_counts = np.unique(valid, return_counts=True)
                    dominant_frac = float(np.max(valid_counts) / max(1, valid.size))
                    cluster_balance_std = float(
                        np.std(valid_counts.astype(np.float64) / max(1.0, float(valid.size)))
                    )
                else:
                    dominant_frac = 1.0
                    cluster_balance_std = 1.0

                under_target = max(0, target_low - n_clusters)
                over_target = max(0, n_clusters - target_high)
                cluster_penalty = float(2.0 * under_target + over_target)
                if n_clusters <= 1:
                    cluster_penalty += 4.0
                score = (
                    cluster_penalty,
                    float(abs(n_clusters - target_mid)),
                    float(noise_frac),
                    float(dominant_frac),
                    float(cluster_balance_std),
                    float(-n_clusters),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = {
                        "min_cluster_size": int(mcs),
                        "min_samples": int(ms),
                        "n_clusters_fit": int(n_clusters),
                        "noise_fraction_fit": float(noise_frac),
                        "dominant_cluster_fraction_fit": float(dominant_frac),
                        "cluster_balance_std_fit": float(cluster_balance_std),
                    }
                    best_clusterer = clusterer
                    best_labels_fit = labels_fit
                    best_selection_method = str(method_name)

    if fit_failures:
        failed_mcs, failed_ms, failed_method, failed_exc = fit_failures[0]
        warnings.warn(
            f"HDBSCAN fitting failed for {len(fit_failures)} candidate setting(s); "
            f"first failure min_cluster_size={failed_mcs}, min_samples={failed_ms}, "
            f"selection_method={failed_method}. "
            f"Error: {failed_exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    if (
        best is None
        or best_clusterer is None
        or best_labels_fit is None
        or best_selection_method is None
    ):
        fallback = np.full((n_total,), -1, dtype=int)
        info = {
            **prep_info_resolved,
            "method": "hdbscan",
            "status": "failed",
            "fit_samples": int(fit_size),
            "total_samples": int(n_total),
            "target_clusters_min": int(target_low),
            "target_clusters_max": int(target_high),
            "cluster_selection_method_requested": requested_selection_method,
            "trials_evaluated": int(trials_evaluated),
            "trials_failed": int(len(fit_failures)),
        }
        if return_info:
            return fallback, info
        return fallback

    if refit_full_data:
        try:
            clusterer_full = hdbscan.HDBSCAN(
                min_cluster_size=int(best["min_cluster_size"]),
                min_samples=int(best["min_samples"]),
                cluster_selection_method=str(best_selection_method),
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                prediction_data=False,
            )
            labels_full = clusterer_full.fit_predict(features).astype(int)
        except Exception as full_exc:
            warnings.warn(
                "HDBSCAN full-data refit failed; falling back to sampled-fit labels/predictions. "
                f"Error: {full_exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if fit_size == n_total:
                labels_full = best_labels_fit
            else:
                try:
                    labels_full, _ = hdbscan.approximate_predict(best_clusterer, features)
                    labels_full = np.asarray(labels_full, dtype=int)
                except Exception as approx_exc:
                    warnings.warn(
                        "hdbscan.approximate_predict failed after refit failure; assigning "
                        "non-fit samples to noise. "
                        f"Error: {approx_exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    labels_full = np.full((n_total,), -1, dtype=int)
                    labels_full[fit_idx] = best_labels_fit
    elif fit_size == n_total:
        labels_full = best_labels_fit
    else:
        try:
            labels_full, _ = hdbscan.approximate_predict(best_clusterer, features)
            labels_full = np.asarray(labels_full, dtype=int)
        except Exception as exc:
            warnings.warn(
                f"hdbscan.approximate_predict failed; trying full refit. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                clusterer_full = hdbscan.HDBSCAN(
                    min_cluster_size=int(best["min_cluster_size"]),
                    min_samples=int(best["min_samples"]),
                    cluster_selection_method=str(best_selection_method),
                    cluster_selection_epsilon=float(cluster_selection_epsilon),
                    prediction_data=False,
                )
                labels_full = clusterer_full.fit_predict(features).astype(int)
            except Exception as full_exc:
                warnings.warn(
                    "HDBSCAN full refit failed; using fit-sample labels and assigning "
                    f"non-fit samples to noise. Error: {full_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                labels_full = np.full((n_total,), -1, dtype=int)
                labels_full[fit_idx] = best_labels_fit

    valid_full = labels_full[labels_full >= 0]
    info = {
        **prep_info_resolved,
        **best,
        "method": "hdbscan",
        "status": "ok",
        "fit_samples": int(fit_size),
        "total_samples": int(n_total),
        "fit_fraction": float(fit_size / max(1, n_total)),
        "n_clusters_full": int(len(np.unique(valid_full))),
        "noise_fraction_full": float(np.mean(labels_full < 0)),
        "target_clusters_min": int(target_low),
        "target_clusters_max": int(target_high),
        "sample_fraction_requested": float(sample_fraction),
        "cluster_selection_method_requested": requested_selection_method,
        "cluster_selection_method": str(best_selection_method),
        "refit_full_data": bool(refit_full_data),
        "trials_evaluated": int(trials_evaluated),
        "trials_failed": int(len(fit_failures)),
        "min_cluster_size_candidates_evaluated": [int(v) for v in min_cluster_size_candidates],
    }
    if return_info:
        return labels_full, info
    return labels_full


def _set_equal_axes_3d(ax, coords: np.ndarray) -> None:
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def save_tsne_with_labels(
    latents: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    out_name: str = "latent_tsne_clusters.png",
    max_samples: int | None = None,
    title: str | None = None,
    legend_title: str = "cluster",
) -> None:
    if latents.size == 0 or len(latents) < 2:
        return
    if labels.size != len(latents):
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents_sub = latents
    labels_sub = labels
    if max_samples is not None and len(latents) > max_samples:
        idx = np.random.default_rng(0).choice(len(latents), size=max_samples, replace=False)
        latents_sub = latents[idx]
        labels_sub = labels[idx]

    perplexity = min(50, max(5, len(latents_sub) // 100))
    tsne_coords = compute_tsne(latents_sub, perplexity=perplexity, n_iter=1500)

    save_tsne_plot(
        tsne_coords,
        labels_sub,
        out_file=str(out_dir / out_name),
        title=title or f"Latent space t-SNE (n={len(latents_sub)})",
        legend_title=legend_title,
    )


def save_tsne_plot_with_coords(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    out_name: str,
    title: str | None,
    legend_title: str = "cluster",
    cluster_color_map: dict[int, str] | None = None,
    paper_out_name: str | None = None,
    paper_title: str | None = None,
    paper_label_prefix: str | None = None,
) -> None:
    if tsne_coords.size == 0 or labels.size != len(tsne_coords):
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tsne_plot(
        tsne_coords,
        labels,
        out_file=str(out_dir / out_name),
        title=title,
        legend_title=legend_title,
        cluster_color_map=cluster_color_map,
    )
    if paper_out_name:
        save_tsne_plot(
            tsne_coords,
            labels,
            out_file=str(out_dir / paper_out_name),
            title=paper_title,
            legend_title=legend_title,
            cluster_color_map=cluster_color_map,
            paper_style=True,
            label_prefix=paper_label_prefix,
        )


def save_tsne_continuous_plot(
    tsne_coords: np.ndarray,
    values: np.ndarray,
    out_file: Path,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if tsne_coords.size == 0 or values.size != len(tsne_coords):
        return
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(values)
    finite_mask = np.isfinite(values)

    plt.figure(figsize=(7, 6), dpi=150)
    if finite_mask.any():
        sc = plt.scatter(
            tsne_coords[finite_mask, 0],
            tsne_coords[finite_mask, 1],
            s=6,
            c=values[finite_mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            linewidths=0,
        )
        plt.colorbar(sc, shrink=0.8)
    if (~finite_mask).any():
        plt.scatter(
            tsne_coords[~finite_mask, 0],
            tsne_coords[~finite_mask, 1],
            s=6,
            c="lightgray",
            alpha=0.4,
            linewidths=0,
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def save_cluster_orientation_histograms(
    angles_deg: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: Path,
    *,
    max_clusters: int = 9,
    bins: int = 30,
) -> None:
    if angles_deg.size == 0 or cluster_labels.size != len(angles_deg):
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles_deg = np.asarray(angles_deg)
    cluster_labels = np.asarray(cluster_labels)
    finite_mask = np.isfinite(angles_deg)
    if not finite_mask.any():
        return

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    selected = unique_labels[order][:max_clusters]

    n_plots = len(selected)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=150)
    axes = np.array(axes).reshape(-1)

    for ax_idx, cluster in enumerate(selected):
        ax = axes[ax_idx]
        mask = (cluster_labels == cluster) & finite_mask
        ax.hist(angles_deg[mask], bins=bins, color="#3498db", alpha=0.75)
        ax.set_title(f"Cluster {int(cluster)} (n={mask.sum()})")
        ax.set_xlabel("Alignment angle (deg)")
        ax.set_ylabel("Count")

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    orientation_out = out_dir / "cluster_orientation_histograms.png"
    fig.savefig(orientation_out)
    plt.close(fig)
    _log_saved_figure(orientation_out)


def save_cluster_symmetry_boxplots(
    n_eff: np.ndarray,
    n_modes: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: Path,
    *,
    max_clusters: int = 10,
) -> None:
    if n_eff.size == 0 or n_modes.size == 0 or cluster_labels.size != len(n_eff):
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_labels = np.asarray(cluster_labels)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    selected = unique_labels[order][:max_clusters]

    eff_data = []
    modes_data = []
    labels = []
    for cluster in selected:
        mask = cluster_labels == cluster
        eff_vals = n_eff[mask]
        mode_vals = n_modes[mask]
        eff_vals = eff_vals[np.isfinite(eff_vals)]
        mode_vals = mode_vals[np.isfinite(mode_vals)]
        if eff_vals.size == 0 and mode_vals.size == 0:
            continue
        eff_data.append(eff_vals)
        modes_data.append(mode_vals)
        labels.append(str(int(cluster)))

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    axes[0].boxplot(eff_data, labels=labels, showfliers=False)
    axes[0].set_title("Per-cluster N_eff")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("N_eff")

    axes[1].boxplot(modes_data, labels=labels, showfliers=False)
    axes[1].set_title("Per-cluster symmetry modes")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("n_modes")

    plt.tight_layout()
    symmetry_out = out_dir / "cluster_symmetry_boxplots.png"
    fig.savefig(symmetry_out)
    plt.close(fig)
    _log_saved_figure(symmetry_out)


def save_local_structure_assignments(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: Path,
    *,
    prefix: str = "local_structure",
) -> Dict[str, str]:
    if coords.size == 0 or cluster_labels.size != len(coords):
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{prefix}_coords_clusters.npz"
    np.savez_compressed(npz_path, coords=coords, clusters=cluster_labels)

    csv_path = out_dir / f"{prefix}_coords_clusters.csv"
    data = np.column_stack([np.arange(len(coords)), coords, cluster_labels])
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="sample_idx,x,y,z,cluster",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%d"],
    )

    return {"npz": str(npz_path), "csv": str(csv_path)}


def save_md_space_clusters_plot(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    out_file: Path,
    *,
    cluster_color_map: dict[int, str] | None = None,
    max_points: int | None = None,
    title: str | None = None,
) -> None:
    if coords.size == 0 or len(coords) < 2:
        return
    if cluster_labels.size != len(coords):
        return

    coords_plot = coords
    labels_plot = cluster_labels
    if max_points is not None and len(coords) > max_points:
        idx = np.random.default_rng(0).choice(len(coords), size=max_points, replace=False)
        coords_plot = coords[idx]
        labels_plot = cluster_labels[idx]

    unique_labels = np.unique(labels_plot)
    colors = _tab10_colors(len(unique_labels))

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for i, label in enumerate(unique_labels):
        mask = labels_plot == label
        label_int = int(label)
        color = (
            str(cluster_color_map[label_int])
            if cluster_color_map is not None and label_int in cluster_color_map
            else colors[i]
        )
        ax.scatter(
            coords_plot[mask, 0],
            coords_plot[mask, 1],
            coords_plot[mask, 2],
            c=[color],
            s=6,
            alpha=0.6,
            depthshade=True,
            label=str(label_int),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        title
        or f"MD local-structure clusters (n={len(coords_plot)}, k={len(unique_labels)})"
    )
    _set_equal_axes_3d(ax, coords_plot)
    if len(unique_labels) <= 15:
        ax.legend(title="cluster", fontsize=7, markerscale=1.5)

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    _log_saved_figure(out_file)


def save_pca_visualization(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    max_samples: int | None = None,
    class_names: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    """Generate PCA visualizations and statistics for latent space analysis."""
    if inv_latents.size == 0 or len(inv_latents) < 2:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents = inv_latents
    has_phases = phases.size == len(latents)
    gt_labels = phases if has_phases else None

    if max_samples is not None and len(latents) > max_samples:
        idx = np.random.default_rng(0).choice(len(latents), size=max_samples, replace=False)
        latents = latents[idx]
        if gt_labels is not None:
            gt_labels = gt_labels[idx]

    n_components = min(latents.shape[1], 50)
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(latents)

    pca_stats = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components_95_var": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    if gt_labels is not None:
        unique_labels = np.unique(gt_labels)
        colors = _tab10_colors(len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = gt_labels == label
            label_text = class_names.get(int(label), f"Phase {int(label)}") if class_names else f"Phase {int(label)}"
            axes[0].scatter(
                pca_coords[mask, 0],
                pca_coords[mask, 1],
                c=[colors[i]],
                s=8,
                alpha=0.6,
                label=label_text,
            )
        if len(unique_labels) <= 15:
            axes[0].legend(fontsize=7, markerscale=1.5)
    else:
        axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1], s=8, alpha=0.6, c="#3498db")

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title(f"PCA Projection (n={len(latents)})")

    components = np.arange(1, n_components + 1)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)

    axes[1].bar(components, pca.explained_variance_ratio_, alpha=0.7, label="Individual")
    axes[1].plot(components, cumulative_var, "r-o", markersize=3, label="Cumulative")
    axes[1].axhline(y=0.95, color="g", linestyle="--", alpha=0.7, label="95% threshold")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title("PCA Explained Variance")
    axes[1].legend()
    axes[1].set_xlim(0.5, min(20, n_components) + 0.5)

    plt.tight_layout()
    pca_out = out_dir / "latent_pca_analysis.png"
    fig.savefig(pca_out)
    plt.close(fig)
    _log_saved_figure(pca_out)

    if pca_coords.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        if gt_labels is not None:
            for i, label in enumerate(unique_labels):
                mask = gt_labels == label
                label_text = class_names.get(int(label), f"Phase {int(label)}") if class_names else f"Phase {int(label)}"
                ax.scatter(
                    pca_coords[mask, 0],
                    pca_coords[mask, 1],
                    pca_coords[mask, 2],
                    c=[colors[i]],
                    s=6,
                    alpha=0.5,
                    label=label_text,
                )
            if len(unique_labels) <= 15:
                ax.legend(fontsize=7, markerscale=1.5)
        else:
            ax.scatter(
                pca_coords[:, 0],
                pca_coords[:, 1],
                pca_coords[:, 2],
                s=6,
                alpha=0.5,
                c="#3498db",
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax.set_title("3D PCA Projection")

        plt.tight_layout()
        pca_3d_out = out_dir / "latent_pca_3d.png"
        fig.savefig(pca_3d_out)
        plt.close(fig)
        _log_saved_figure(pca_3d_out)

    return pca_stats


def save_latent_statistics(
    inv_latents: np.ndarray,
    eq_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    class_names: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    """Compute and visualize detailed latent space statistics."""
    if inv_latents.size == 0:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {}
    has_phases = phases.size == len(inv_latents)
    has_eq_latents = eq_latents.size > 0

    stats["inv_latent"] = {
        "mean": float(np.mean(inv_latents)),
        "std": float(np.std(inv_latents)),
        "min": float(np.min(inv_latents)),
        "max": float(np.max(inv_latents)),
        "dim": int(inv_latents.shape[1]) if inv_latents.ndim > 1 else 1,
        "n_samples": int(len(inv_latents)),
    }

    dim_means = np.mean(inv_latents, axis=0)
    dim_stds = np.std(inv_latents, axis=0)
    stats["inv_latent"]["dim_mean_range"] = [float(dim_means.min()), float(dim_means.max())]
    stats["inv_latent"]["dim_std_range"] = [float(dim_stds.min()), float(dim_stds.max())]

    norms = np.linalg.norm(inv_latents, axis=1)
    stats["inv_latent"]["norm_mean"] = float(np.mean(norms))
    stats["inv_latent"]["norm_std"] = float(np.std(norms))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

    if has_phases:
        unique_labels = np.unique(phases)
        colors = _tab10_colors(len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = phases == label
            label_text = class_names.get(int(label), f"Phase {int(label)}") if class_names else f"Phase {int(label)}"
            axes[0, 0].hist(norms[mask], bins=50, alpha=0.5, label=label_text, color=colors[i])
        if len(unique_labels) <= 10:
            axes[0, 0].legend(fontsize=7)
    else:
        axes[0, 0].hist(norms, bins=50, alpha=0.7, color="#3498db")
    axes[0, 0].set_xlabel("Latent Norm ||z||")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Invariant Latent Norm Distribution")

    dims = np.arange(len(dim_means))
    axes[0, 1].fill_between(dims, dim_means - dim_stds, dim_means + dim_stds, alpha=0.3, color="#3498db")
    axes[0, 1].plot(dims, dim_means, color="#2980b9", linewidth=1)
    axes[0, 1].set_xlabel("Latent Dimension")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_title("Per-Dimension Statistics (mean +/- std)")

    dim_activity = np.abs(dim_means) + dim_stds
    sorted_idx = np.argsort(dim_activity)[::-1]
    top_dims = min(20, len(dim_activity))
    axes[0, 2].bar(range(top_dims), dim_activity[sorted_idx[:top_dims]], color="#27ae60", alpha=0.7)
    axes[0, 2].set_xlabel("Dimension Rank")
    axes[0, 2].set_ylabel("Activity (|mean| + std)")
    axes[0, 2].set_title(f"Top {top_dims} Active Dimensions")

    if inv_latents.shape[1] > 1:
        corr_matrix = np.corrcoef(inv_latents.T)
        n_show = min(20, corr_matrix.shape[0])
        sns.heatmap(
            corr_matrix[:n_show, :n_show],
            ax=axes[1, 0],
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.5},
        )
        axes[1, 0].set_title(f"Dimension Correlation (first {n_show} dims)")
    else:
        axes[1, 0].text(0.5, 0.5, "Single dimension", ha="center", va="center")
        axes[1, 0].set_title("Correlation Matrix")

    if has_phases and len(unique_labels) > 1:
        class_means = []
        class_stds = []
        for label in unique_labels:
            mask = phases == label
            class_means.append(np.mean(norms[mask]))
            class_stds.append(np.std(norms[mask]))

        x_pos = np.arange(len(unique_labels))
        axes[1, 1].bar(
            x_pos,
            class_means,
            yerr=class_stds,
            capsize=3,
            color=colors[: len(unique_labels)],
            alpha=0.7,
        )
        axes[1, 1].set_xlabel("Phase")
        axes[1, 1].set_ylabel("Mean Latent Norm")
        axes[1, 1].set_title("Per-Class Latent Norm")
        axes[1, 1].set_xticks(x_pos)

        labels_list = []
        for l in unique_labels:
            labels_list.append(class_names.get(int(l), str(int(l))) if class_names else str(int(l)))

        axes[1, 1].set_xticklabels(labels_list, rotation=45, ha="right", fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No class labels available", ha="center", va="center")
        axes[1, 1].set_title("Per-Class Statistics")

    if has_eq_latents and eq_latents.shape[0] == inv_latents.shape[0]:
        eq_norms = np.linalg.norm(eq_latents.reshape(eq_latents.shape[0], -1), axis=1)
        axes[1, 2].scatter(norms, eq_norms, alpha=0.3, s=5, c="#9b59b6")
        axes[1, 2].set_xlabel("Invariant Latent Norm")
        axes[1, 2].set_ylabel("Equivariant Latent Norm")
        axes[1, 2].set_title("Inv. vs Eq. Latent Norms")

        correlation = np.corrcoef(norms, eq_norms)[0, 1]
        axes[1, 2].text(
            0.05,
            0.95,
            f"r = {correlation:.3f}",
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        stats["inv_eq_norm_correlation"] = float(correlation)
    else:
        axes[1, 2].text(0.5, 0.5, "No equivariant latents", ha="center", va="center")
        axes[1, 2].set_title("Inv. vs Eq. Comparison")

    plt.tight_layout()
    latent_stats_out = out_dir / "latent_statistics.png"
    fig.savefig(latent_stats_out)
    plt.close(fig)
    _log_saved_figure(latent_stats_out)

    return stats


def save_equivariance_plot(eq_errors: np.ndarray, out_file: Path) -> None:
    if eq_errors.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
    ax.hist(eq_errors, bins=30, color="#2980b9", alpha=0.8)
    ax.set_title("Equivariant latent relative error (seeded)")
    ax.set_xlabel("||z_R - Rz|| / ||Rz||")
    ax.set_ylabel("count")

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    _log_saved_figure(out_file)


__all__ = [
    "ClusteringFeatureTransform",
    "FittedClusteringModel",
    "fit_clustering_feature_transform",
    "transform_clustering_features",
    "fit_clustering_model",
    "predict_clustering_model",
    "compute_transfer_kmeans_labels",
    "save_latent_tsne",
    "compute_kmeans_labels",
    "compute_hdbscan_labels",
    "save_tsne_with_labels",
    "save_tsne_plot_with_coords",
    "save_local_structure_assignments",
    "save_md_space_clusters_plot",
    "save_pca_visualization",
    "save_latent_statistics",
    "save_equivariance_plot",
]
