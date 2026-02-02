import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.utils.spd_metrics import random_rotation_matrix
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot


def load_barlow_model(
    checkpoint_path: str, cuda_device: int = 0, cfg: DictConfig | None = None
) -> Tuple[BarlowTwinsModule, DictConfig, str]:
    """Restore the Barlow Twins module together with its Hydra cfg and device string."""
    if cfg is None:
        config_dir, config_name = resolve_config_path(checkpoint_path)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[2]
        absolute_config_dir = project_root / config_dir
        relative_config_dir = os.path.relpath(absolute_config_dir, current_dir)
        with initialize(version_base=None, config_path=relative_config_dir):
            cfg = compose(config_name=config_name)

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    model: BarlowTwinsModule = load_model_from_checkpoint(
        checkpoint_path, cfg, device=device, module=BarlowTwinsModule
    )
    model.to(device).eval()
    return model, cfg, device


def build_datamodule(cfg: DictConfig):
    """Instantiate and setup the matching datamodule."""
    if getattr(cfg, "data", None) is None:
        raise ValueError("Config missing data section")
    if getattr(cfg.data, "kind", None) == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    dm.setup(stage="test")
    return dm


def _extract_pc_and_phase(batch: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        pc = batch["points"]
        phase = batch.get("class_id", None)
    elif isinstance(batch, (tuple, list)):
        pc = batch[0]
        phase = batch[1] if len(batch) > 1 else None
    else:
        pc = batch
        phase = None
    if phase is not None and not torch.is_tensor(phase):
        phase = torch.as_tensor(phase)
    return pc, phase


def gather_inference_batches(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int | None = 4,
) -> Dict[str, np.ndarray]:
    """Collect inputs and latents from batches."""
    inv_latents, eq_latents, phases = [], [], []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pc, phase = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)

            z, _, eq_z = model(pc)
            if z is not None:
                inv_latents.append(z.detach().cpu())
            if eq_z is not None:
                eq_latents.append(eq_z.detach().cpu())
            if phase is not None:
                phases.append(phase.detach().view(-1).cpu())

    def _cat(tensors):
        return torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,))

    return {
        "inv_latents": _cat(inv_latents),
        "eq_latents": _cat(eq_latents),
        "phases": _cat(phases),
    }


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

    metrics: Dict[str, Any] = {}

    k_range = range(2, min(11, len(latents) // 10))
    silhouette_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latents)
        silhouette_scores.append(silhouette_score(latents, cluster_labels))
        inertias.append(kmeans.inertia_)

    metrics["silhouette_scores"] = {int(k): float(s) for k, s in zip(k_range, silhouette_scores)}
    metrics["best_k_silhouette"] = int(list(k_range)[np.argmax(silhouette_scores)])
    metrics["best_silhouette_score"] = float(max(silhouette_scores))

    if gt_labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        n_true_clusters = len(np.unique(gt_labels))
        kmeans_gt = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
        pred_labels = kmeans_gt.fit_predict(latents)

        ari = adjusted_rand_score(gt_labels, pred_labels)
        nmi = normalized_mutual_info_score(gt_labels, pred_labels)
        gt_silhouette = silhouette_score(latents, gt_labels)

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
                class_points = latents[mask]
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

    axes[0].plot(list(k_range), silhouette_scores, "b-o", markersize=6)
    axes[0].axvline(
        x=metrics["best_k_silhouette"],
        color="r",
        linestyle="--",
        label=f"Best k={metrics['best_k_silhouette']}",
    )
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Silhouette Score vs. k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), inertias, "g-o", markersize=6)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Inertia (Within-cluster SS)")
    axes[1].set_title("Elbow Plot")
    axes[1].grid(True, alpha=0.3)

    if gt_labels is not None and len(unique_labels) > 1:
        separation_data = []
        labels_list = []
        for label in unique_labels:
            mask = gt_labels == label
            class_points = latents[mask]
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


def save_umap_visualization(
    inv_latents: np.ndarray,
    phases: np.ndarray,
    out_dir: Path,
    max_samples: int | None = None,
    class_names: Dict[int, str] | None = None,
) -> None:
    """Generate UMAP visualization as an alternative to t-SNE."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed, skipping UMAP visualization. Install with: pip install umap-learn")
        return

    if inv_latents.size == 0 or len(inv_latents) < 10:
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

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    umap_coords = reducer.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    if gt_labels is not None:
        unique_labels = np.unique(gt_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = gt_labels == label
            label_text = class_names.get(int(label), f"Phase {int(label)}") if class_names else f"Phase {int(label)}"
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[colors[i]],
                s=8,
                alpha=0.6,
                label=label_text,
            )
        if len(unique_labels) <= 15:
            ax.legend(fontsize=8, markerscale=1.5)
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=8, alpha=0.6, c="#3498db")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP Projection (n={len(latents)})")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "latent_umap.png")
    plt.close(fig)


def evaluate_latent_equivariance(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: int = 2,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate equivariance in encoder outputs (if equivariant latents exist)."""
    eq_errors = []

    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            pc, _ = _extract_pc_and_phase(batch)
            pc = pc.to(device)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)
            batch_size = pc.shape[0]

            rots = torch.stack(
                [
                    torch.tensor(random_rotation_matrix(), device=device, dtype=pc.dtype)
                    for _ in range(batch_size)
                ]
            )
            pc_rot = torch.einsum("bij,bnj->bni", rots, pc)

            _, _, eq_z = model(pc)
            _, _, eq_z_rot = model(pc_rot)

            if eq_z is None or eq_z_rot is None:
                continue

            expected_eq = torch.einsum("bij,bcj->bci", rots, eq_z)
            rel = torch.linalg.norm(eq_z_rot - expected_eq, dim=-1) / torch.linalg.norm(
                expected_eq, dim=-1
            ).clamp_min(1e-6)
            eq_errors.extend(rel.mean(dim=1).detach().cpu().numpy().tolist())

    eq_arr = np.asarray(eq_errors)
    metrics = {
        "eq_latent_rel_error_mean": float(eq_arr.mean()) if eq_arr.size else float("nan"),
        "eq_latent_rel_error_median": float(np.median(eq_arr)) if eq_arr.size else float("nan"),
        "num_samples": int(eq_arr.size),
    }
    return metrics, eq_arr


def save_equivariance_plot(eq_errors: np.ndarray, out_file: Path) -> None:
    if eq_errors.size == 0:
        return

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
    ax.hist(eq_errors, bins=30, color="#2980b9", alpha=0.8)
    ax.set_title("Equivariant latent relative error")
    ax.set_xlabel("||z_R - Rz|| / ||Rz||")
    ax.set_ylabel("count")

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    max_batches_latent: int | None = None,
    max_samples_visualization: int | None = None,
    use_train_data: bool = True,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for Barlow Twins."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, cfg, device = load_barlow_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)
    dm = build_datamodule(cfg)

    if use_train_data:
        dm.setup(stage="fit")
        dl = dm.train_dataloader()
        print("Using TRAINING dataset for latent analysis")
    else:
        dl = dm.test_dataloader()
        print("Using TEST dataset for latent analysis")

    class_names = None
    if hasattr(dm, "train_dataset"):
        ds = getattr(dm, "train_dataset", None)
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        if hasattr(ds, "class_names"):
            class_names = ds.class_names
            print(f"Loaded class names: {class_names}")

    if class_names is None and hasattr(dl, "dataset"):
        ds = dl.dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        if hasattr(ds, "class_names"):
            class_names = ds.class_names
            print(f"Loaded class names from DL: {class_names}")

    if max_batches_latent is None:
        print("Gathering inference batches (ALL batches)...")
    else:
        print(f"Gathering inference batches (up to {max_batches_latent} batches)...")
    cache = gather_inference_batches(model, dl, device, max_batches=max_batches_latent)

    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")

    all_metrics: Dict[str, Any] = {}

    print("Computing t-SNE visualization...")
    save_latent_tsne(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=max_samples_visualization,
        class_names=class_names,
    )

    print("Computing PCA analysis...")
    pca_stats = save_pca_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=max_samples_visualization,
        class_names=class_names,
    )
    all_metrics["pca"] = pca_stats

    print("Computing latent statistics...")
    latent_stats = save_latent_statistics(
        cache["inv_latents"],
        cache["eq_latents"],
        cache["phases"],
        out_dir,
        class_names=class_names,
    )
    all_metrics["latent_stats"] = latent_stats

    print("Computing clustering analysis...")
    clustering_metrics = save_clustering_analysis(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=max_samples_visualization,
        class_names=class_names,
    )
    all_metrics["clustering"] = clustering_metrics

    print("Computing UMAP visualization...")
    save_umap_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=max_samples_visualization,
        class_names=class_names,
    )

    print("Evaluating equivariance (encoder latents)...")
    eq_metrics, eq_err = evaluate_latent_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    all_metrics["equivariance"] = eq_metrics

    import json

    metrics_path = out_dir / "analysis_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(all_metrics, handle, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total samples analyzed: {n_samples}")

    if "pca" in all_metrics and all_metrics["pca"]:
        print(f"PCA: {all_metrics['pca'].get('n_components_95_var', 'N/A')} components for 95% variance")

    if "clustering" in all_metrics and all_metrics["clustering"]:
        print(f"Best k (silhouette): {all_metrics['clustering'].get('best_k_silhouette', 'N/A')}")
        print(
            f"Best silhouette score: {all_metrics['clustering'].get('best_silhouette_score', 'N/A'):.4f}"
            if isinstance(all_metrics["clustering"].get("best_silhouette_score"), float)
            else f"Best silhouette score: {all_metrics['clustering'].get('best_silhouette_score', 'N/A')}"
        )
        if "ari_with_gt" in all_metrics["clustering"]:
            print(f"ARI with ground truth: {all_metrics['clustering']['ari_with_gt']:.4f}")
            print(f"NMI with ground truth: {all_metrics['clustering']['nmi_with_gt']:.4f}")
        if "class_separation_ratio" in all_metrics["clustering"]:
            print(
                f"Class separation ratio: {all_metrics['clustering']['class_separation_ratio']:.4f}"
            )

    if "equivariance" in all_metrics:
        eq = all_metrics["equivariance"]
        print(
            f"Equivariant latent error (mean): {eq.get('eq_latent_rel_error_mean', 'N/A'):.4f}"
            if isinstance(eq.get("eq_latent_rel_error_mean"), float)
            else f"Equivariant latent error (mean): {eq.get('eq_latent_rel_error_mean', 'N/A')}"
        )
        print(
            f"Equivariant latent error (median): {eq.get('eq_latent_rel_error_median', 'N/A'):.4f}"
            if isinstance(eq.get("eq_latent_rel_error_median"), float)
            else f"Equivariant latent error (median): {eq.get('eq_latent_rel_error_median', 'N/A')}"
        )

    print("=" * 60)
    print(f"\nSaved all analyses to {out_dir}")
    print("Generated files:")
    print("  - latent_tsne_ground_truth.png: t-SNE with ground truth labels")
    print("  - latent_tsne_clusters.png: t-SNE with KMeans clusters")
    print("  - latent_pca_analysis.png: PCA projection and variance")
    print("  - latent_pca_3d.png: 3D PCA projection")
    print("  - latent_statistics.png: Comprehensive latent statistics")
    print("  - clustering_analysis.png: Clustering quality metrics")
    print("  - latent_umap.png: UMAP projection (if umap-learn installed)")
    print("  - equivariance.png: Equivariant latent error distribution")
    print("  - analysis_metrics.json: All numerical metrics")

    return all_metrics
