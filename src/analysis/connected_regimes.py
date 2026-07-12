"""Continuous analysis for adjacent hard-cluster regimes.

Hard clustering is useful for naming regimes, but two labels may occupy adjacent
parts of one connected manifold.  This module separates two questions:

* are the regimes locally connected (cross-label k-nearest-neighbor contact)?
* are their latent distributions different along a continuous coordinate?

All geometry is computed in the original invariant latent space.  UMAP is used
nowhere in the metrics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .output_layout import log_saved_figure, write_json


@dataclass(frozen=True)
class ConnectedRegimeSettings:
    enabled: bool
    explicit_pairs: tuple[tuple[int, int], ...]
    auto_detect: bool
    auto_max_pairs: int
    max_samples_per_cluster: int
    neighbor_k: int
    min_cross_neighbor_fraction: float
    pca_components: int
    histogram_bins: int
    random_state: int
    interactive_3d: bool = False
    representative_steps: int = 30


def _normalize_explicit_pairs(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair_index, pair_value in enumerate(list(value)):
        pair_items = list(pair_value)
        if len(pair_items) != 2:
            raise ValueError(
                "clustering.connected_regimes.pairs entries must contain exactly two "
                f"zero-based cluster IDs. Entry {pair_index} is {pair_value!r}."
            )
        cluster_a, cluster_b = (int(pair_items[0]), int(pair_items[1]))
        if cluster_a < 0 or cluster_b < 0 or cluster_a == cluster_b:
            raise ValueError(
                "Connected-regime pairs require two distinct non-negative cluster IDs, "
                f"got {(cluster_a, cluster_b)}."
            )
        normalized = tuple(sorted((cluster_a, cluster_b)))
        if normalized not in seen:
            seen.add(normalized)
            pairs.append(normalized)
    return tuple(pairs)


def resolve_connected_regime_settings(
    analysis_cfg: DictConfig,
    *,
    default_random_state: int,
) -> ConnectedRegimeSettings:
    prefix = "clustering.connected_regimes"
    enabled = bool(OmegaConf.select(analysis_cfg, f"{prefix}.enabled", default=False))
    settings = ConnectedRegimeSettings(
        enabled=enabled,
        explicit_pairs=_normalize_explicit_pairs(
            OmegaConf.select(analysis_cfg, f"{prefix}.pairs", default=None)
        ),
        auto_detect=bool(
            OmegaConf.select(analysis_cfg, f"{prefix}.auto_detect", default=True)
        ),
        auto_max_pairs=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.auto_max_pairs", default=4)
        ),
        max_samples_per_cluster=int(
            OmegaConf.select(
                analysis_cfg,
                f"{prefix}.max_samples_per_cluster",
                default=1500,
            )
        ),
        neighbor_k=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.neighbor_k", default=15)
        ),
        min_cross_neighbor_fraction=float(
            OmegaConf.select(
                analysis_cfg,
                f"{prefix}.min_cross_neighbor_fraction",
                default=0.002,
            )
        ),
        pca_components=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.pca_components", default=8)
        ),
        histogram_bins=int(
            OmegaConf.select(analysis_cfg, f"{prefix}.histogram_bins", default=40)
        ),
        random_state=int(
            OmegaConf.select(
                analysis_cfg,
                f"{prefix}.random_state",
                default=int(default_random_state),
            )
        ),
        interactive_3d=bool(
            OmegaConf.select(analysis_cfg, f"{prefix}.interactive_3d", default=False)
        ),
        representative_steps=int(
            OmegaConf.select(
                analysis_cfg, f"{prefix}.representative_steps", default=30
            )
        ),
    )
    if not settings.enabled:
        return settings
    if not settings.explicit_pairs and not settings.auto_detect:
        raise ValueError(
            "Connected-regime analysis is enabled, but pairs is empty and "
            "auto_detect=false. Provide clustering.connected_regimes.pairs or enable "
            "automatic adjacent-pair discovery."
        )
    if settings.auto_max_pairs < 1:
        raise ValueError("clustering.connected_regimes.auto_max_pairs must be >= 1.")
    if settings.max_samples_per_cluster < 32:
        raise ValueError(
            "clustering.connected_regimes.max_samples_per_cluster must be >= 32."
        )
    if settings.neighbor_k < 2:
        raise ValueError("clustering.connected_regimes.neighbor_k must be >= 2.")
    if not 0.0 <= settings.min_cross_neighbor_fraction <= 1.0:
        raise ValueError(
            "clustering.connected_regimes.min_cross_neighbor_fraction must be in [0, 1]."
        )
    if settings.pca_components < 2:
        raise ValueError("clustering.connected_regimes.pca_components must be >= 2.")
    if settings.histogram_bins < 10:
        raise ValueError("clustering.connected_regimes.histogram_bins must be >= 10.")
    if settings.interactive_3d and settings.representative_steps < 3:
        raise ValueError(
            "clustering.connected_regimes.representative_steps must be >= 3."
        )
    return settings


def _stratified_sample_indices(
    labels: np.ndarray,
    cluster_ids: Sequence[int],
    *,
    max_samples_per_cluster: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    sampled: list[np.ndarray] = []
    for cluster_id in cluster_ids:
        indices = np.flatnonzero(labels == int(cluster_id))
        if indices.size < 2:
            raise ValueError(
                f"Cluster {cluster_id} has only {indices.size} samples; at least two are required."
            )
        if indices.size > int(max_samples_per_cluster):
            indices = np.sort(
                rng.choice(indices, size=int(max_samples_per_cluster), replace=False)
            )
        sampled.append(indices.astype(np.int64, copy=False))
    return np.concatenate(sampled)


def _compute_neighbor_contact(
    latents: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    cluster_ids: Sequence[int],
    *,
    neighbor_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled_latents = np.asarray(latents[sample_indices], dtype=np.float32)
    sampled_labels = np.asarray(labels[sample_indices], dtype=np.int64)
    norms = np.linalg.norm(sampled_latents, axis=1)
    if np.any(norms <= 1.0e-12):
        bad_count = int(np.sum(norms <= 1.0e-12))
        raise ValueError(
            "Connected-regime cosine neighbors received zero-norm latent rows: "
            f"count={bad_count}, sampled_rows={sampled_latents.shape[0]}."
        )
    neighbor_features = sampled_latents / norms[:, None]
    resolved_k = min(int(neighbor_k), int(neighbor_features.shape[0]) - 1)
    if resolved_k < 1:
        raise ValueError("Connected-regime neighbor analysis needs at least two samples.")
    model = NearestNeighbors(
        n_neighbors=resolved_k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(neighbor_features)
    raw_neighbors = model.kneighbors(neighbor_features, return_distance=False)
    row_ids = np.arange(raw_neighbors.shape[0], dtype=np.int64)
    neighbors = np.empty((raw_neighbors.shape[0], resolved_k), dtype=np.int64)
    for row_id in row_ids:
        without_self = raw_neighbors[row_id][raw_neighbors[row_id] != row_id]
        if without_self.size < resolved_k:
            raise RuntimeError(
                "Nearest-neighbor query did not return enough non-self neighbors. "
                f"row={int(row_id)}, requested={resolved_k}, returned={without_self.size}."
            )
        neighbors[row_id] = without_self[:resolved_k]
    neighbor_labels = sampled_labels[neighbors]

    cluster_ids_list = [int(v) for v in cluster_ids]
    directed_contact = np.zeros((len(cluster_ids_list), len(cluster_ids_list)), dtype=np.float64)
    boundary_fraction = np.zeros_like(directed_contact)
    for source_index, source_cluster in enumerate(cluster_ids_list):
        source_rows = sampled_labels == source_cluster
        for target_index, target_cluster in enumerate(cluster_ids_list):
            if source_index == target_index:
                continue
            target_neighbors = neighbor_labels[source_rows] == target_cluster
            directed_contact[source_index, target_index] = float(target_neighbors.mean())
            boundary_fraction[source_index, target_index] = float(
                np.any(target_neighbors, axis=1).mean()
            )
    adjacency = 0.5 * (directed_contact + directed_contact.T)
    return sampled_labels, neighbor_labels, adjacency, boundary_fraction


def _pooled_standard_deviation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    var_a = float(np.var(values_a, ddof=1))
    var_b = float(np.var(values_b, ddof=1))
    numerator = (values_a.size - 1) * var_a + (values_b.size - 1) * var_b
    denominator = values_a.size + values_b.size - 2
    return float(np.sqrt(max(numerator / denominator, 0.0)))


def _histogram_overlap(values_a: np.ndarray, values_b: np.ndarray, *, bins: int) -> float:
    lower = float(min(values_a.min(), values_b.min()))
    upper = float(max(values_a.max(), values_b.max()))
    if not upper > lower:
        return 1.0
    edges = np.linspace(lower, upper, int(bins) + 1)
    hist_a, _ = np.histogram(values_a, bins=edges, density=True)
    hist_b, _ = np.histogram(values_b, bins=edges, density=True)
    return float(np.sum(np.minimum(hist_a, hist_b) * np.diff(edges)))


def _fit_order_parameter(
    pair_latents: np.ndarray,
    pair_labels: np.ndarray,
    cluster_a: int,
    cluster_b: int,
    *,
    pca_components: int,
    random_state: int,
    histogram_bins: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    component_count = min(
        int(pca_components),
        int(pair_latents.shape[1]),
        int(pair_latents.shape[0]) - 1,
    )
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=int(random_state),
    )
    pca_scores = pca.fit_transform(pair_latents)
    mask_a = pair_labels == int(cluster_a)
    mask_b = pair_labels == int(cluster_b)

    effects: list[float] = []
    for component_index in range(component_count):
        values_a = pca_scores[mask_a, component_index]
        values_b = pca_scores[mask_b, component_index]
        pooled_std = _pooled_standard_deviation(values_a, values_b)
        mean_delta = abs(float(values_b.mean() - values_a.mean()))
        effects.append(mean_delta / max(pooled_std, 1.0e-12))
    selected_component = int(np.argmax(np.asarray(effects)))
    raw_score = pca_scores[:, selected_component].astype(np.float64, copy=True)
    if float(raw_score[mask_b].mean()) < float(raw_score[mask_a].mean()):
        raw_score *= -1.0
        pca_scores[:, selected_component] *= -1.0

    mean_a = float(raw_score[mask_a].mean())
    mean_b = float(raw_score[mask_b].mean())
    mean_delta = mean_b - mean_a
    if mean_delta <= 1.0e-12:
        raise RuntimeError(
            "Could not orient a continuous order parameter with distinct regime means. "
            f"clusters=({cluster_a}, {cluster_b}), selected_component={selected_component + 1}, "
            f"mean_delta={mean_delta}."
        )
    order_parameter = (raw_score - mean_a) / mean_delta
    order_a = order_parameter[mask_a]
    order_b = order_parameter[mask_b]
    pooled_order_std = _pooled_standard_deviation(order_a, order_b)
    cohen_d = 1.0 / max(pooled_order_std, 1.0e-12)

    centroid_a = pair_latents[mask_a].mean(axis=0)
    centroid_b = pair_latents[mask_b].mean(axis=0)
    within_radius_sq = 0.5 * (
        float(np.mean(np.sum((pair_latents[mask_a] - centroid_a) ** 2, axis=1)))
        + float(np.mean(np.sum((pair_latents[mask_b] - centroid_b) ** 2, axis=1)))
    )
    wasserstein_order = float(wasserstein_distance(order_a, order_b))
    metrics = {
        "selected_pca_component": int(selected_component + 1),
        "selected_pca_explained_variance_ratio": float(
            pca.explained_variance_ratio_[selected_component]
        ),
        "pca_explained_variance_ratio": [
            float(v) for v in pca.explained_variance_ratio_.tolist()
        ],
        "cohen_d": float(cohen_d),
        "wasserstein_order_parameter": wasserstein_order,
        "wasserstein_over_pooled_std": float(
            wasserstein_order / max(pooled_order_std, 1.0e-12)
        ),
        "energy_distance_order_parameter": float(energy_distance(order_a, order_b)),
        "distribution_overlap": _histogram_overlap(
            order_a,
            order_b,
            bins=int(histogram_bins),
        ),
        "centroid_distance": float(np.linalg.norm(centroid_b - centroid_a)),
        "centroid_distance_over_within_radius": float(
            np.linalg.norm(centroid_b - centroid_a)
            / max(np.sqrt(within_radius_sq), 1.0e-12)
        ),
        "order_parameter_mean": {
            str(cluster_a): float(order_a.mean()),
            str(cluster_b): float(order_b.mean()),
        },
        "order_parameter_std": {
            str(cluster_a): float(order_a.std()),
            str(cluster_b): float(order_b.std()),
        },
    }
    orthogonal_candidates = [index for index in range(component_count) if index != selected_component]
    orthogonal_component = orthogonal_candidates[0] if orthogonal_candidates else selected_component
    orthogonal_score = pca_scores[:, orthogonal_component].astype(np.float64, copy=False)
    return metrics, order_parameter, orthogonal_score, pca_scores


def _save_pair_csv(
    path: Path,
    *,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    order_parameter: np.ndarray,
    orthogonal_score: np.ndarray,
    local_other_fraction: np.ndarray,
    source_indices: np.ndarray | None,
    source_names: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "cluster_id",
                "cluster_label",
                "order_parameter",
                "orthogonal_coordinate",
                "local_other_cluster_fraction",
                "source_name",
            ]
        )
        for row_index, sample_index in enumerate(sample_indices):
            source_name = ""
            if source_indices is not None:
                source_index = int(source_indices[int(sample_index)])
                if source_index >= 0:
                    source_name = str(source_names[source_index])
            cluster_id = int(labels[row_index])
            writer.writerow(
                [
                    int(sample_index),
                    cluster_id,
                    f"C{cluster_id + 1}",
                    float(order_parameter[row_index]),
                    float(orthogonal_score[row_index]),
                    float(local_other_fraction[row_index]),
                    source_name,
                ]
            )


def _save_pair_figure(
    path: Path,
    *,
    cluster_a: int,
    cluster_b: int,
    labels: np.ndarray,
    order_parameter: np.ndarray,
    orthogonal_score: np.ndarray,
    local_other_fraction: np.ndarray,
    metrics: dict[str, Any],
    color_map: dict[int, str],
    histogram_bins: int,
) -> None:
    import matplotlib.pyplot as plt

    mask_a = labels == int(cluster_a)
    mask_b = labels == int(cluster_b)
    colors = np.asarray([color_map[int(label)] for label in labels], dtype=object)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), dpi=180)

    axes[0, 0].scatter(
        order_parameter,
        orthogonal_score,
        c=colors,
        s=8,
        alpha=0.50,
        linewidths=0,
    )
    axes[0, 0].set_title("Hard labels on one connected coordinate")
    axes[0, 0].set_xlabel(f"C{cluster_a + 1} → C{cluster_b + 1} order parameter")
    axes[0, 0].set_ylabel("orthogonal PCA coordinate")

    scatter = axes[0, 1].scatter(
        order_parameter,
        orthogonal_score,
        c=order_parameter,
        cmap="viridis",
        s=8,
        alpha=0.60,
        linewidths=0,
    )
    axes[0, 1].set_title("Same samples colored continuously")
    axes[0, 1].set_xlabel("order parameter")
    axes[0, 1].set_ylabel("orthogonal PCA coordinate")
    fig.colorbar(scatter, ax=axes[0, 1], label="continuous order parameter")

    axes[1, 0].hist(
        order_parameter[mask_a],
        bins=int(histogram_bins),
        density=True,
        alpha=0.52,
        color=color_map[int(cluster_a)],
        label=f"C{cluster_a + 1}",
    )
    axes[1, 0].hist(
        order_parameter[mask_b],
        bins=int(histogram_bins),
        density=True,
        alpha=0.52,
        color=color_map[int(cluster_b)],
        label=f"C{cluster_b + 1}",
    )
    axes[1, 0].set_title(
        "Different distributions without requiring a gap\n"
        f"Cohen d={metrics['cohen_d']:.2f}, overlap={metrics['distribution_overlap']:.2f}"
    )
    axes[1, 0].set_xlabel("order parameter")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend()

    lower, upper = np.quantile(order_parameter, [0.01, 0.99])
    if not upper > lower:
        lower, upper = float(order_parameter.min()), float(order_parameter.max())
    bin_edges = np.linspace(float(lower), float(upper), 21)
    bin_ids = np.digitize(order_parameter, bin_edges[1:-1], right=False)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_cross_neighbor_fraction = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    for bin_index in range(centers.size):
        selected = bin_ids == bin_index
        counts[bin_index] = int(selected.sum())
        if counts[bin_index] > 0:
            mean_cross_neighbor_fraction[bin_index] = float(
                local_other_fraction[selected].mean()
            )
    valid = np.isfinite(mean_cross_neighbor_fraction)
    axes[1, 1].plot(
        centers[valid],
        mean_cross_neighbor_fraction[valid],
        marker="o",
        color="#5e548e",
        linewidth=2.0,
    )
    axes[1, 1].fill_between(
        centers[valid],
        0.0,
        mean_cross_neighbor_fraction[valid],
        color="#9f86c0",
        alpha=0.25,
    )
    axes[1, 1].set_ylim(
        0.0,
        max(0.05, float(np.nanmax(mean_cross_neighbor_fraction)) * 1.15),
    )
    axes[1, 1].set_xlabel("order parameter")
    axes[1, 1].set_ylabel("fraction of neighbors from the other regime")
    axes[1, 1].set_title(
        "Local cross-regime mixing (transition zone)\n"
        f"kNN contact={metrics['symmetric_cross_neighbor_fraction']:.3f}"
    )

    fig.suptitle(
        f"Connected regimes C{cluster_a + 1} and C{cluster_b + 1}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log_saved_figure(path)


def _save_overview_figure(
    path: Path,
    *,
    cluster_ids: Sequence[int],
    adjacency: np.ndarray,
    pair_metrics: Sequence[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    difference = np.full(adjacency.shape, np.nan, dtype=np.float64)
    id_to_position = {int(cluster_id): index for index, cluster_id in enumerate(cluster_ids)}
    for metrics in pair_metrics:
        cluster_a, cluster_b = metrics["cluster_ids"]
        row, column = id_to_position[int(cluster_a)], id_to_position[int(cluster_b)]
        value = float(metrics["wasserstein_over_pooled_std"])
        difference[row, column] = value
        difference[column, row] = value

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=180)
    labels = [f"C{int(cluster_id) + 1}" for cluster_id in cluster_ids]
    for axis, matrix, title, cmap in (
        (axes[0], adjacency, "kNN cross-contact (connectedness)", "magma"),
        (
            axes[1],
            difference,
            "Order-coordinate Wasserstein / pooled std",
            "viridis",
        ),
    ):
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, cmap=cmap, vmin=0.0)
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if len(labels) <= 12:
            for row in range(len(labels)):
                for column in range(len(labels)):
                    value = matrix[row, column]
                    if np.isfinite(value) and row != column:
                        axis.text(
                            column,
                            row,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if value > np.nanmax(matrix) * 0.55 else "black",
                        )
    fig.suptitle("Connected but distinguishable cluster regimes", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log_saved_figure(path)


def _build_source_index(
    sample_count: int,
    frame_groups: Sequence[tuple[str, np.ndarray]] | None,
) -> tuple[np.ndarray | None, list[str]]:
    if not frame_groups:
        return None, []
    source_index = np.full(int(sample_count), -1, dtype=np.int32)
    source_names: list[str] = []
    for source_id, (source_name, indices_value) in enumerate(frame_groups):
        indices = np.asarray(indices_value, dtype=np.int64).reshape(-1)
        if indices.size > 0 and (indices.min() < 0 or indices.max() >= int(sample_count)):
            raise IndexError(
                "Connected-regime frame group contains sample indices outside the latent cache: "
                f"source={source_name!r}, min={int(indices.min())}, max={int(indices.max())}, "
                f"sample_count={sample_count}."
            )
        source_index[indices] = int(source_id)
        source_names.append(str(source_name))
    return source_index, source_names


def run_connected_regime_analysis(
    *,
    latents: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    settings: ConnectedRegimeSettings,
    cluster_color_map: dict[int, str],
    frame_groups: Sequence[tuple[str, np.ndarray]] | None = None,
    dataset: Any | None = None,
    representatives_out_dir: Path | None = None,
    representative_point_scale: float = 1.0,
    representative_target_points: int = 64,
    representative_selection_features: np.ndarray | None = None,
    step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not settings.enabled:
        return {}
    latents_array = np.asarray(latents, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if latents_array.ndim != 2 or latents_array.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "Connected-regime analysis expects latents (N, D) aligned with labels (N,), "
            f"got latents={latents_array.shape}, labels={labels_array.shape}."
        )
    if not np.isfinite(latents_array).all():
        raise ValueError("Connected-regime analysis received non-finite latent values.")
    if settings.interactive_3d and dataset is None:
        raise TypeError(
            "Connected-regime interactive_3d=true requires the analysis dataset so real "
            "local structures can be loaded."
        )
    if settings.interactive_3d and representatives_out_dir is None:
        raise ValueError(
            "Connected-regime interactive_3d=true requires representatives_out_dir."
        )
    cluster_ids = sorted(int(v) for v in np.unique(labels_array) if int(v) >= 0)
    if len(cluster_ids) < 2:
        raise ValueError(
            "Connected-regime analysis requires at least two non-noise clusters, "
            f"got {cluster_ids}."
        )
    missing_color_ids = [cluster_id for cluster_id in cluster_ids if cluster_id not in cluster_color_map]
    if missing_color_ids:
        raise KeyError(
            "Connected-regime color map is missing cluster IDs "
            f"{missing_color_ids}; available={sorted(cluster_color_map)}."
        )
    available_ids = set(cluster_ids)
    missing_explicit = [
        pair
        for pair in settings.explicit_pairs
        if pair[0] not in available_ids or pair[1] not in available_ids
    ]
    if missing_explicit:
        raise ValueError(
            "Explicit connected-regime pairs reference unavailable cluster IDs. "
            f"missing_pairs={missing_explicit}, available={cluster_ids}."
        )

    if step is not None:
        step("Analyzing connected cluster regimes in original latent space")
    sample_indices = _stratified_sample_indices(
        labels_array,
        cluster_ids,
        max_samples_per_cluster=settings.max_samples_per_cluster,
        random_state=settings.random_state,
    )
    sampled_labels, neighbor_labels, adjacency, boundary_fraction = _compute_neighbor_contact(
        latents_array,
        labels_array,
        sample_indices,
        cluster_ids,
        neighbor_k=settings.neighbor_k,
    )
    id_to_position = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    selected_pairs = list(settings.explicit_pairs)
    if settings.auto_detect:
        candidates: list[tuple[float, tuple[int, int]]] = []
        for first_position, cluster_a in enumerate(cluster_ids):
            for second_position in range(first_position + 1, len(cluster_ids)):
                cluster_b = cluster_ids[second_position]
                score = float(adjacency[first_position, second_position])
                if score >= settings.min_cross_neighbor_fraction:
                    candidates.append((score, (cluster_a, cluster_b)))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _score, pair in candidates[: settings.auto_max_pairs]:
            if pair not in selected_pairs:
                selected_pairs.append(pair)
    connected_cluster_ids = {
        int(cluster_id) for pair in selected_pairs for cluster_id in pair
    }
    within_cluster_ids = [
        cluster_id for cluster_id in cluster_ids if cluster_id not in connected_cluster_ids
    ]

    output_root = Path(out_dir) / "connected_regimes"
    output_root.mkdir(parents=True, exist_ok=True)
    source_index, source_names = _build_source_index(
        latents_array.shape[0],
        frame_groups,
    )
    sampled_position_by_index = {
        int(sample_index): int(position)
        for position, sample_index in enumerate(sample_indices)
    }
    pair_metrics: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []
    for cluster_a, cluster_b in selected_pairs:
        pair_global_mask = np.isin(labels_array[sample_indices], [cluster_a, cluster_b])
        pair_indices = sample_indices[pair_global_mask]
        pair_labels = labels_array[pair_indices]
        pair_latents = latents_array[pair_indices]
        coordinate_metrics, order_parameter, orthogonal_score, _pca_scores = (
            _fit_order_parameter(
                pair_latents,
                pair_labels,
                cluster_a,
                cluster_b,
                pca_components=settings.pca_components,
                random_state=settings.random_state,
                histogram_bins=settings.histogram_bins,
            )
        )
        pair_positions = np.asarray(
            [sampled_position_by_index[int(index)] for index in pair_indices],
            dtype=np.int64,
        )
        local_other_fraction = np.empty(pair_indices.shape[0], dtype=np.float64)
        for row_index, (sampled_position, label) in enumerate(
            zip(pair_positions, pair_labels, strict=True)
        ):
            other_label = cluster_b if int(label) == cluster_a else cluster_a
            local_other_fraction[row_index] = float(
                np.mean(neighbor_labels[int(sampled_position)] == int(other_label))
            )
        position_a = id_to_position[cluster_a]
        position_b = id_to_position[cluster_b]
        pair_name = f"C{cluster_a + 1}_C{cluster_b + 1}"
        figure_path = output_root / f"{pair_name}_connected_regime.png"
        csv_path = output_root / f"{pair_name}_order_parameter.csv"
        metrics = {
            "pair_name": pair_name,
            "cluster_ids": [int(cluster_a), int(cluster_b)],
            "cluster_labels": [f"C{cluster_a + 1}", f"C{cluster_b + 1}"],
            "sample_count": int(pair_indices.size),
            "sample_count_by_cluster": {
                str(cluster_a): int(np.sum(pair_labels == cluster_a)),
                str(cluster_b): int(np.sum(pair_labels == cluster_b)),
            },
            "symmetric_cross_neighbor_fraction": float(adjacency[position_a, position_b]),
            "boundary_sample_fraction": {
                str(cluster_a): float(boundary_fraction[position_a, position_b]),
                str(cluster_b): float(boundary_fraction[position_b, position_a]),
            },
            "mean_local_other_cluster_fraction": {
                str(cluster_a): float(local_other_fraction[pair_labels == cluster_a].mean()),
                str(cluster_b): float(local_other_fraction[pair_labels == cluster_b].mean()),
            },
            **coordinate_metrics,
            "artifacts": {
                "figure": str(figure_path.relative_to(out_dir)),
                "order_parameter_csv": str(csv_path.relative_to(out_dir)),
            },
        }
        _save_pair_csv(
            csv_path,
            sample_indices=pair_indices,
            labels=pair_labels,
            order_parameter=order_parameter,
            orthogonal_score=orthogonal_score,
            local_other_fraction=local_other_fraction,
            source_indices=source_index,
            source_names=source_names,
        )
        _save_pair_figure(
            figure_path,
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            labels=pair_labels,
            order_parameter=order_parameter,
            orthogonal_score=orthogonal_score,
            local_other_fraction=local_other_fraction,
            metrics=metrics,
            color_map=cluster_color_map,
            histogram_bins=settings.histogram_bins,
        )
        if settings.interactive_3d:
            from .representative_transitions import render_connected_pair_transition_3d

            pair_source_names = None
            if source_index is not None:
                pair_source_names = [
                    (
                        ""
                        if int(source_index[int(sample_index)]) < 0
                        else str(source_names[int(source_index[int(sample_index)])])
                    )
                    for sample_index in pair_indices
                ]
            transition_path = (
                Path(representatives_out_dir)
                / f"11_{pair_name}_connected_transition_3d.html"
            )
            transition_summary = render_connected_pair_transition_3d(
                dataset=dataset,
                pair_latents=pair_latents,
                pair_labels=pair_labels,
                pair_sample_indices=pair_indices,
                order_parameter=order_parameter,
                local_other_fraction=local_other_fraction,
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                cluster_color_map=cluster_color_map,
                out_file=transition_path,
                steps=settings.representative_steps,
                point_scale=float(representative_point_scale),
                target_points=int(representative_target_points),
                source_names=pair_source_names,
            )
            transition_summaries.append(transition_summary)
            metrics["artifacts"]["interactive_3d_representatives"] = str(
                transition_path.relative_to(out_dir)
            )
            metrics["interactive_3d_representatives"] = {
                key: value
                for key, value in transition_summary.items()
                if key != "out_file"
            }
        pair_metrics.append(metrics)

    within_cluster_summaries: list[dict[str, Any]] = []
    if settings.interactive_3d:
        from .representative_transitions import render_within_cluster_transition_3d

        for cluster_id in within_cluster_ids:
            cluster_mask = sampled_labels == int(cluster_id)
            cluster_indices = sample_indices[cluster_mask]
            cluster_latents = latents_array[cluster_indices]
            cluster_source_names = None
            if source_index is not None:
                cluster_source_names = [
                    (
                        ""
                        if int(source_index[int(sample_index)]) < 0
                        else str(source_names[int(source_index[int(sample_index)])])
                    )
                    for sample_index in cluster_indices
                ]
            transition_path = (
                Path(representatives_out_dir)
                / f"11_C{cluster_id + 1}_within_cluster_transition_3d.html"
            )
            transition_summary = render_within_cluster_transition_3d(
                dataset=dataset,
                cluster_latents=cluster_latents,
                cluster_sample_indices=cluster_indices,
                cluster_id=int(cluster_id),
                cluster_color_map=cluster_color_map,
                out_file=transition_path,
                steps=settings.representative_steps,
                point_scale=float(representative_point_scale),
                target_points=int(representative_target_points),
                source_names=cluster_source_names,
            )
            transition_summaries.append(transition_summary)
            within_cluster_summaries.append(
                {
                    **{
                        key: value
                        for key, value in transition_summary.items()
                        if key != "out_file"
                    },
                    "artifact": str(transition_path.relative_to(out_dir)),
                }
            )

    overview_path = output_root / "connected_regime_overview.png"
    _save_overview_figure(
        overview_path,
        cluster_ids=cluster_ids,
        adjacency=adjacency,
        pair_metrics=pair_metrics,
    )
    pair_csv_path = output_root / "connected_regime_pair_metrics.csv"
    with pair_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "pair_name",
            "cluster_a",
            "cluster_b",
            "sample_count",
            "cross_neighbor_fraction",
            "cohen_d",
            "wasserstein_order_parameter",
            "wasserstein_over_pooled_std",
            "energy_distance_order_parameter",
            "distribution_overlap",
            "centroid_distance_over_within_radius",
            "selected_pca_component",
            "selected_pca_explained_variance_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in pair_metrics:
            writer.writerow(
                {
                    "pair_name": metrics["pair_name"],
                    "cluster_a": metrics["cluster_ids"][0],
                    "cluster_b": metrics["cluster_ids"][1],
                    "sample_count": metrics["sample_count"],
                    "cross_neighbor_fraction": metrics["symmetric_cross_neighbor_fraction"],
                    "cohen_d": metrics["cohen_d"],
                    "wasserstein_order_parameter": metrics[
                        "wasserstein_order_parameter"
                    ],
                    "wasserstein_over_pooled_std": metrics[
                        "wasserstein_over_pooled_std"
                    ],
                    "energy_distance_order_parameter": metrics[
                        "energy_distance_order_parameter"
                    ],
                    "distribution_overlap": metrics["distribution_overlap"],
                    "centroid_distance_over_within_radius": metrics[
                        "centroid_distance_over_within_radius"
                    ],
                    "selected_pca_component": metrics["selected_pca_component"],
                    "selected_pca_explained_variance_ratio": metrics[
                        "selected_pca_explained_variance_ratio"
                    ],
                }
            )

    cluster_gallery_path = None
    cluster_gallery_summary = None
    if settings.interactive_3d:
        from .representative_transitions import render_cluster_representatives_3d

        source_names_by_sample = None
        if source_index is not None:
            source_names_by_sample = [
                (
                    ""
                    if int(source_index[sample_index]) < 0
                    else str(source_names[int(source_index[sample_index])])
                )
                for sample_index in range(len(labels_array))
            ]
        cluster_gallery_path = (
            Path(representatives_out_dir) / "12_cluster_representatives_3d.html"
        )
        cluster_gallery_summary = render_cluster_representatives_3d(
            dataset=dataset,
            latents=latents_array,
            cluster_labels=labels_array,
            cluster_color_map=cluster_color_map,
            out_file=cluster_gallery_path,
            point_scale=float(representative_point_scale),
            target_points=int(representative_target_points),
            selection_features=representative_selection_features,
            source_names_by_sample=source_names_by_sample,
        )

    transition_index_path = None
    if transition_summaries:
        from .representative_transitions import write_transition_representatives_index

        transition_index_path = (
            Path(representatives_out_dir) / "11_connected_regime_transitions_3d.html"
        )
        write_transition_representatives_index(
            transition_index_path,
            transition_summaries,
            cluster_gallery_path=cluster_gallery_path,
        )

    summary = {
        "enabled": True,
        "primary_cluster_ids": cluster_ids,
        "sample_count_total": int(labels_array.size),
        "sample_count_neighbor_analysis": int(sample_indices.size),
        "neighbor_k": int(settings.neighbor_k),
        "interactive_3d": {
            "enabled": bool(settings.interactive_3d),
            "representative_steps": int(settings.representative_steps),
            "connected_pair_direction": (
                "cluster A centroid to cluster B centroid in original latent space"
            ),
            "within_cluster_direction": (
                "dominant PCA direction inside clusters outside every connected pair"
            ),
            "real_structures_only": True,
            "within_cluster_transitions": within_cluster_summaries,
            "cluster_gallery": (
                None
                if cluster_gallery_summary is None
                else {
                    key: value
                    for key, value in cluster_gallery_summary.items()
                    if key != "out_file"
                }
            ),
        },
        "selection": {
            "explicit_pairs": [list(pair) for pair in settings.explicit_pairs],
            "auto_detect": bool(settings.auto_detect),
            "auto_max_pairs": int(settings.auto_max_pairs),
            "min_cross_neighbor_fraction": float(settings.min_cross_neighbor_fraction),
            "selected_pairs": [list(pair) for pair in selected_pairs],
            "within_cluster_ids": [int(value) for value in within_cluster_ids],
        },
        "pairs": pair_metrics,
        "artifacts": {
            "overview_figure": str(overview_path.relative_to(out_dir)),
            "pair_metrics_csv": str(pair_csv_path.relative_to(out_dir)),
            "metrics_json": "connected_regimes/connected_regime_metrics.json",
            **(
                {}
                if transition_index_path is None
                else {
                    "interactive_3d_index": str(
                        transition_index_path.relative_to(out_dir)
                    )
                }
            ),
            **(
                {}
                if cluster_gallery_path is None
                else {
                    "interactive_3d_cluster_gallery": str(
                        cluster_gallery_path.relative_to(out_dir)
                    )
                }
            ),
        },
    }
    metrics_path = output_root / "connected_regime_metrics.json"
    write_json(metrics_path, summary)
    readme_path = output_root / "README.md"
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write("# Connected-regime analysis\n\n")
        handle.write(
            "Cross-neighbor contact measures whether hard cluster labels touch locally. "
            "The PCA-selected order parameter measures how their distributions differ "
            "without requiring an empty boundary. All metrics use original invariant "
            "latents, not UMAP coordinates.\n\n"
        )
        if transition_index_path is not None:
            handle.write(
                "Interactive 3D real-structure paths are indexed at "
                f"`{transition_index_path.relative_to(out_dir)}`. Connected pairs follow "
                "their cluster-centroid direction; clusters outside every selected pair "
                "follow their dominant internal PCA direction. Each page has a manual "
                "slider only and never interpolates atom coordinates.\n\n"
            )
        for metrics in pair_metrics:
            handle.write(
                f"- **{metrics['pair_name']}**: contact="
                f"{metrics['symmetric_cross_neighbor_fraction']:.4f}, "
                f"Cohen d={metrics['cohen_d']:.3f}, "
                f"overlap={metrics['distribution_overlap']:.3f}, "
                "standardized Wasserstein="
                f"{metrics['wasserstein_over_pooled_std']:.3f}.\n"
            )
    summary["artifacts"]["readme"] = str(readme_path.relative_to(out_dir))
    write_json(metrics_path, summary)
    return summary
