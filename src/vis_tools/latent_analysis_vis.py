from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot


def save_latent_tsne(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    max_samples: int | None = None,
    class_names: Dict[int, str] | None = None,
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
        idx = np.random.default_rng(0).choice(len(latents), size=max_samples, replace=False)
        latents = latents[idx]
        if gt_labels is not None:
            gt_labels = gt_labels[idx]

    perplexity = min(50, max(5, len(latents) // 100))
    tsne_coords = compute_tsne(latents, perplexity=perplexity, n_iter=1500)

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

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latents)

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


def _prepare_clustering_features(
    latents: np.ndarray,
    *,
    random_state: int,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
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

    if standardize and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    use_pca = (
        pca_variance is not None
        and float(pca_variance) > 0.0
        and x.shape[1] > 2
        and x.shape[0] > 3
    )
    if use_pca:
        n_max = min(int(pca_max_components), x.shape[1], x.shape[0] - 1)
        if n_max >= 2:
            pca = PCA(n_components=n_max, random_state=random_state)
            x_proj = pca.fit_transform(x)
            if float(pca_variance) >= 1.0:
                keep = n_max
            else:
                csum = np.cumsum(pca.explained_variance_ratio_)
                keep = int(np.searchsorted(csum, float(pca_variance)) + 1)
                keep = max(2, min(keep, n_max))
            x = x_proj[:, :keep]
            info["pca_components"] = int(keep)
            info["pca_explained_variance"] = float(
                np.sum(pca.explained_variance_ratio_[:keep])
            )
        else:
            info["pca_components"] = int(x.shape[1])
            info["pca_explained_variance"] = 1.0
    else:
        info["pca_components"] = int(x.shape[1])
        info["pca_explained_variance"] = 1.0

    info["output_dim"] = int(x.shape[1])
    return x.astype(np.float32, copy=False), info


def _safe_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
    max_samples: int,
) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("-inf")

    x_eval = features
    y_eval = labels
    if len(features) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(features), size=max_samples, replace=False)
        x_eval = features[idx]
        y_eval = labels[idx]
        if len(np.unique(y_eval)) < 2:
            return float("-inf")

    try:
        return float(silhouette_score(x_eval, y_eval))
    except ValueError:
        return float("-inf")


def _fit_labels_single_method(
    features: np.ndarray,
    n_clusters: int,
    *,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    method = str(method).lower()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(features)
        return labels.astype(int), {
            "method": "kmeans",
            "model_score_name": "inertia",
            "model_score": float(model.inertia_),
        }

    if method in {"gmm", "gmm_diag", "gmm_full"}:
        covariance = "diag" if method == "gmm_diag" else "full"
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance,
            random_state=random_state,
            n_init=3,
            reg_covar=1e-5,
            max_iter=300,
        )
        model.fit(features)
        labels = model.predict(features)
        return labels.astype(int), {
            "method": f"gmm_{covariance}",
            "model_score_name": "bic",
            "model_score": float(model.bic(features)),
        }

    raise ValueError(f"Unsupported clustering method: {method}")


def _resolve_method_candidates(method: str, num_samples: int) -> list[str]:
    method = str(method).lower()
    if method in {"kmeans", "gmm", "gmm_diag", "gmm_full"}:
        return [method]
    if method != "auto":
        return ["kmeans"]

    if num_samples > 25000:
        return ["kmeans"]
    if num_samples > 12000:
        return ["kmeans", "gmm_diag"]
    return ["kmeans", "gmm_diag", "gmm_full"]


def _cluster_with_method_selection(
    features: np.ndarray,
    n_clusters: int,
    *,
    method: str,
    random_state: int,
    silhouette_max_samples: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    candidates = _resolve_method_candidates(method, len(features))
    best_labels: np.ndarray | None = None
    best_info: Dict[str, Any] | None = None

    for candidate in candidates:
        try:
            labels, info = _fit_labels_single_method(
                features,
                n_clusters,
                method=candidate,
                random_state=random_state,
            )
        except Exception:
            continue

        sil = _safe_silhouette(
            features,
            labels,
            random_state=random_state,
            max_samples=silhouette_max_samples,
        )
        info = dict(info)
        info["silhouette"] = sil

        if best_info is None or sil > float(best_info.get("silhouette", float("-inf"))):
            best_info = info
            best_labels = labels

    if best_labels is None or best_info is None:
        # Guaranteed fallback.
        best_labels, best_info = _fit_labels_single_method(
            features, n_clusters, method="kmeans", random_state=random_state
        )
        best_info["silhouette"] = _safe_silhouette(
            features,
            best_labels,
            random_state=random_state,
            max_samples=silhouette_max_samples,
        )

    best_info["requested_method"] = str(method).lower()
    return best_labels, best_info


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
    silhouette_max_samples: int = 5000,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    if latents.size == 0 or len(latents) < 2:
        empty = np.empty((0,), dtype=int)
        if return_info:
            return empty, {"method": "none"}
        return empty
    n_clusters = max(2, min(int(n_clusters), len(latents)))
    features, prep_info = _prepare_clustering_features(
        latents,
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
        silhouette_max_samples=silhouette_max_samples,
    )
    if len(features_fit) == len(features):
        labels = labels_fit
    else:
        # Refit on full preprocessed features with selected method for consistent labels.
        selected_method = str(fit_info.get("method", "kmeans"))
        if selected_method.startswith("gmm_"):
            selected_method = selected_method
        labels, fit_info = _cluster_with_method_selection(
            features,
            n_clusters,
            method=selected_method,
            random_state=random_state,
            silhouette_max_samples=silhouette_max_samples,
        )

    info = {
        **prep_info,
        **fit_info,
        "n_clusters": int(n_clusters),
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
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "leaf",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    if latents.size == 0 or len(latents) < 2:
        empty = np.empty((0,), dtype=int)
        if return_info:
            return empty, {"method": "hdbscan", "status": "empty"}
        return empty

    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN clustering requested but 'hdbscan' is not installed."
        ) from exc

    features, prep_info = _prepare_clustering_features(
        latents,
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

    if min_cluster_size_candidates is None:
        ratios = [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040]
        min_cluster_size_candidates = sorted(
            {
                max(5, min(fit_size, int(round(fit_size * ratio))))
                for ratio in ratios
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
        min_cluster_size_candidates = [max(5, min(fit_size, fit_size // 30))]

    selection_method = str(cluster_selection_method).strip().lower()
    if selection_method not in {"eom", "leaf"}:
        selection_method = "leaf"

    target_low = max(2, int(target_clusters_min))
    target_high = max(target_low, int(target_clusters_max))
    target_mid = 0.5 * (target_low + target_high)

    best: Dict[str, Any] | None = None
    best_clusterer = None
    best_labels_fit: np.ndarray | None = None
    best_score = None

    for mcs in min_cluster_size_candidates:
        ms = int(min_samples) if min_samples is not None else max(4, min(64, int(round(mcs * 0.35))))
        ms = max(1, min(ms, mcs))
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
                cluster_selection_method=selection_method,
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                prediction_data=True,
            )
            labels_fit = clusterer.fit_predict(fit_features).astype(int)
        except Exception:
            continue

        valid = labels_fit[labels_fit >= 0]
        n_clusters = int(len(np.unique(valid)))
        noise_frac = float(np.mean(labels_fit < 0))

        in_target = target_low <= n_clusters <= target_high
        distance_to_target = 0.0 if in_target else abs(n_clusters - target_mid)
        score = (
            float(distance_to_target),
            float(noise_frac),
            float(abs(n_clusters - target_mid)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = {
                "min_cluster_size": int(mcs),
                "min_samples": int(ms),
                "n_clusters_fit": int(n_clusters),
                "noise_fraction_fit": float(noise_frac),
            }
            best_clusterer = clusterer
            best_labels_fit = labels_fit

    if best is None or best_clusterer is None or best_labels_fit is None:
        fallback = np.full((n_total,), -1, dtype=int)
        info = {
            **prep_info,
            "method": "hdbscan",
            "status": "failed",
            "fit_samples": int(fit_size),
            "total_samples": int(n_total),
        }
        if return_info:
            return fallback, info
        return fallback

    if fit_size == n_total:
        labels_full = best_labels_fit
    else:
        try:
            labels_full, _ = hdbscan.approximate_predict(best_clusterer, features)
            labels_full = np.asarray(labels_full, dtype=int)
        except Exception:
            try:
                clusterer_full = hdbscan.HDBSCAN(
                    min_cluster_size=int(best["min_cluster_size"]),
                    min_samples=int(best["min_samples"]),
                    cluster_selection_method=selection_method,
                    cluster_selection_epsilon=float(cluster_selection_epsilon),
                    prediction_data=False,
                )
                labels_full = clusterer_full.fit_predict(features).astype(int)
            except Exception:
                labels_full = np.full((n_total,), -1, dtype=int)
                labels_full[fit_idx] = best_labels_fit

    valid_full = labels_full[labels_full >= 0]
    info = {
        **prep_info,
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
        "cluster_selection_method": selection_method,
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
    title: str,
    legend_title: str = "cluster",
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
    fig.savefig(out_dir / "cluster_orientation_histograms.png")
    plt.close(fig)


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
    fig.savefig(out_dir / "cluster_symmetry_boxplots.png")
    plt.close(fig)


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
    max_points: int | None = None,
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
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for i, label in enumerate(unique_labels):
        mask = labels_plot == label
        ax.scatter(
            coords_plot[mask, 0],
            coords_plot[mask, 1],
            coords_plot[mask, 2],
            c=[colors[i]],
            s=6,
            alpha=0.6,
            depthshade=True,
            label=str(int(label)),
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"MD local-structure clusters (n={len(coords_plot)}, k={len(unique_labels)})"
    )
    _set_equal_axes_3d(ax, coords_plot)
    if len(unique_labels) <= 15:
        ax.legend(title="cluster", fontsize=7, markerscale=1.5)

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


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
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
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
    fig.savefig(out_dir / "latent_pca_analysis.png")
    plt.close(fig)

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
        fig.savefig(out_dir / "latent_pca_3d.png")
        plt.close(fig)

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
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
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
    fig.savefig(out_dir / "latent_statistics.png")
    plt.close(fig)

    return stats


def save_clustering_analysis(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    max_samples: int | None = None,
    class_names: Dict[int, str] | None = None,
    cluster_method: str = "auto",
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_variance: float | None = 0.98,
    pca_max_components: int = 32,
    silhouette_max_samples: int = 5000,
    silhouette_tolerance: float = 0.03,
    k_min: int = 2,
    k_max: int = 12,
) -> Dict[str, Any]:
    """Perform clustering analysis on latent space and compute quality metrics."""
    if inv_latents.size == 0 or len(inv_latents) < 10:
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

    features, prep_info = _prepare_clustering_features(
        latents,
        random_state=42,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_variance=pca_variance,
        pca_max_components=pca_max_components,
    )

    metrics: Dict[str, Any] = {
        "cluster_feature_prep": prep_info,
        "cluster_method_requested": str(cluster_method).lower(),
    }

    upper = min(int(k_max), max(2, len(features) // 20))
    lower = max(2, int(k_min))
    if upper < lower:
        upper = lower
    k_range = list(range(lower, upper + 1))
    if len(k_range) < 1:
        return metrics

    silhouette_vals: list[float] = []
    inertias: list[float] = []
    selected_methods: dict[int, str] = {}
    selected_model_scores: dict[int, float] = {}

    for k in k_range:
        labels_k, info_k = _cluster_with_method_selection(
            features,
            k,
            method=cluster_method,
            random_state=42,
            silhouette_max_samples=silhouette_max_samples,
        )
        sil_k = float(
            info_k.get(
                "silhouette",
                _safe_silhouette(
                    features,
                    labels_k,
                    random_state=42,
                    max_samples=silhouette_max_samples,
                ),
            )
        )
        silhouette_vals.append(sil_k)
        selected_methods[int(k)] = str(info_k.get("method", "kmeans"))
        model_score = info_k.get("model_score", np.nan)
        selected_model_scores[int(k)] = float(model_score) if np.isfinite(model_score) else float("nan")

        # Keep a consistent elbow metric independent of selected method.
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(features)
        inertias.append(float(km.inertia_))

    raw_best_idx = int(np.nanargmax(silhouette_vals))
    raw_best_sil = float(silhouette_vals[raw_best_idx])

    tol = max(0.0, float(silhouette_tolerance))
    candidate_idx = [
        idx for idx, sil in enumerate(silhouette_vals) if np.isfinite(sil) and sil >= (raw_best_sil - tol)
    ]
    if candidate_idx:
        best_idx = int(min(candidate_idx))
    else:
        best_idx = raw_best_idx

    best_k = int(k_range[best_idx])
    best_method = selected_methods.get(best_k, "kmeans")
    best_sil = float(silhouette_vals[best_idx])

    metrics["silhouette_scores"] = {int(k): float(s) for k, s in zip(k_range, silhouette_vals)}
    metrics["selected_method_by_k"] = {int(k): selected_methods[int(k)] for k in k_range}
    metrics["selected_model_score_by_k"] = {
        int(k): float(selected_model_scores[int(k)]) for k in k_range
    }
    metrics["best_k_silhouette"] = best_k
    metrics["best_method"] = best_method
    metrics["best_silhouette_score"] = best_sil
    metrics["best_k_silhouette_raw"] = int(k_range[raw_best_idx])
    metrics["best_silhouette_score_raw"] = float(raw_best_sil)
    metrics["silhouette_tolerance"] = float(tol)

    if gt_labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        n_true_clusters = len(np.unique(gt_labels))
        pred_labels, _ = _cluster_with_method_selection(
            features,
            n_true_clusters,
            method=cluster_method,
            random_state=42,
            silhouette_max_samples=silhouette_max_samples,
        )

        ari = adjusted_rand_score(gt_labels, pred_labels)
        nmi = normalized_mutual_info_score(gt_labels, pred_labels)
        gt_silhouette = _safe_silhouette(
            features,
            gt_labels,
            random_state=42,
            max_samples=silhouette_max_samples,
        )

        metrics["ari_with_gt"] = float(ari)
        metrics["nmi_with_gt"] = float(nmi)
        metrics["gt_silhouette_score"] = float(gt_silhouette)

    if gt_labels is not None:
        unique_labels = np.unique(gt_labels)
        if len(unique_labels) > 1:
            intra_dists = []
            inter_dists = []
            class_centroids = {}

            for label in unique_labels:
                mask = gt_labels == label
                class_points = features[mask]
                centroid = np.mean(class_points, axis=0)
                class_centroids[label] = centroid

                dists_to_centroid = np.linalg.norm(class_points - centroid, axis=1)
                intra_dists.extend(dists_to_centroid.tolist())

            centroids_arr = np.array(list(class_centroids.values()))
            for i in range(len(centroids_arr)):
                for j in range(i + 1, len(centroids_arr)):
                    inter_dists.append(np.linalg.norm(centroids_arr[i] - centroids_arr[j]))

            metrics["mean_intra_class_distance"] = float(np.mean(intra_dists))
            metrics["mean_inter_class_distance"] = float(np.mean(inter_dists))
            metrics["class_separation_ratio"] = float(
                np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    axes[0].plot(k_range, silhouette_vals, "b-o", markersize=6)
    axes[0].axvline(
        x=metrics["best_k_silhouette"],
        color="r",
        linestyle="--",
        label=f"Best k={metrics['best_k_silhouette']} ({best_method})",
    )
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette Score vs. k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, inertias, "g-o", markersize=6)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Inertia (Within-cluster SS)")
    axes[1].set_title("Elbow Plot")
    axes[1].grid(True, alpha=0.3)

    if gt_labels is not None and len(unique_labels) > 1:
        separation_data = []
        labels_list = []
        for label in unique_labels:
            mask = gt_labels == label
            class_points = features[mask]
            centroid = class_centroids[label]
            dists = np.linalg.norm(class_points - centroid, axis=1)
            separation_data.append(dists)

            label_text = class_names.get(int(label), f"Phase {int(label)}") if class_names else f"Phase {int(label)}"
            labels_list.append(label_text)

        bp = axes[2].boxplot(separation_data, labels=labels_list, patch_artist=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[2].set_xlabel("Phase")
        axes[2].set_ylabel("Distance to Class Centroid")
        axes[2].set_title("Intra-Class Distance Distribution")
        axes[2].tick_params(axis="x", rotation=45)
    else:
        axes[2].text(0.5, 0.5, "No class labels available", ha="center", va="center")
        axes[2].set_title("Class Separation")

    plt.tight_layout()
    fig.savefig(out_dir / "clustering_analysis.png")
    plt.close(fig)

    return metrics


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


__all__ = [
    "save_latent_tsne",
    "compute_kmeans_labels",
    "compute_hdbscan_labels",
    "save_tsne_with_labels",
    "save_tsne_plot_with_coords",
    "save_local_structure_assignments",
    "save_md_space_clusters_plot",
    "save_pca_visualization",
    "save_latent_statistics",
    "save_clustering_analysis",
    "save_equivariance_plot",
]
