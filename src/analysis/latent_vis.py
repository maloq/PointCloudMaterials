"""Latent-space visualization: PCA, t-SNE, latent statistics, equivariance."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .utils import (
    _sample_indices,
    evaluate_latent_equivariance,
)
from src.vis_tools.latent_analysis_vis import (
    save_equivariance_plot,
    save_latent_statistics,
    save_pca_visualization,
    save_tsne_plot_with_coords,
)
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def run_pca_and_latent_stats(
    cache: dict[str, np.ndarray],
    out_dir: Path,
    *,
    class_names: Dict[int, str] | None,
    step: Callable[[str], None],
) -> dict[str, Any]:
    """Compute PCA visualization and latent statistics."""
    metrics: dict[str, Any] = {}

    step("Computing PCA analysis")
    pca_stats = save_pca_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=None,
        class_names=class_names,
    )
    metrics["pca"] = pca_stats

    step("Computing latent statistics")
    latent_stats = save_latent_statistics(
        cache["inv_latents"],
        cache["eq_latents"],
        cache["phases"],
        out_dir,
        class_names=class_names,
    )
    metrics["latent_stats"] = latent_stats

    return metrics


def run_tsne_visualizations(
    cache: dict[str, np.ndarray],
    out_dir: Path,
    *,
    analysis_cfg: DictConfig,
    cluster_labels_by_k: dict[int, np.ndarray],
    cluster_methods_by_k: dict[int, str],
    configured_k_values: list[int],
    primary_k: int,
    shared_cluster_color_maps_by_k: dict[int, dict[int, str]],
    class_names: Dict[int, str] | None,
    is_synthetic: bool,
    clustering_random_state: int,
    tsne_max_samples: int | None,
    tsne_n_iter: int,
    cluster_method: str,
    step: Callable[[str], None],
) -> None:
    """Compute t-SNE and save ground-truth + cluster plots."""
    if not bool(OmegaConf.select(analysis_cfg, "tsne.enabled", default=True)):
        step("Skipping t-SNE visualization")
        return

    step("Computing t-SNE visualization (clusters)")
    tsne_idx = _sample_indices(
        len(cache["inv_latents"]),
        tsne_max_samples,
    )
    tsne_latents = cache["inv_latents"][tsne_idx]
    tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
    tsne_coords = compute_tsne(
        tsne_latents,
        random_state=clustering_random_state,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
    )
    if is_synthetic and cache["phases"].size == len(cache["inv_latents"]):
        save_tsne_plot(
            tsne_coords,
            cache["phases"][tsne_idx],
            out_file=str(out_dir / "latent_tsne_ground_truth.png"),
            title=f"Latent space t-SNE (n={len(tsne_latents)}, ground truth phases)",
            legend_title="phase",
            class_names=class_names,
        )

    for idx_k, k_val in enumerate(configured_k_values):
        labels_k = cluster_labels_by_k[int(k_val)]
        method_k = cluster_methods_by_k.get(int(k_val), cluster_method)
        out_name = "latent_tsne_clusters.png" if idx_k == 0 else f"latent_tsne_clusters_k{k_val}.png"
        save_tsne_plot_with_coords(
            tsne_coords,
            labels_k[tsne_idx],
            out_dir,
            out_name=out_name,
            title=f"Latent space t-SNE ({method_k}, k={k_val})",
            cluster_color_map=shared_cluster_color_maps_by_k.get(int(k_val)),
            paper_out_name=(
                f"latent_tsne_clusters_paper_k{k_val}.svg"
                if (
                    not is_synthetic
                    and bool(OmegaConf.select(analysis_cfg, "tsne.paper_enabled", default=True))
                    and int(k_val) == int(primary_k)
                )
                else None
            ),
            paper_title=None,
            paper_label_prefix="C",
        )


def run_equivariance_evaluation(
    model: Any,
    dl: Any,
    device: str,
    out_dir: Path,
    *,
    analysis_cfg: DictConfig,
    step: Callable[[str], None],
    temporal_sequence_mode: str = "static_anchor",
    temporal_static_frame_index: int | None = 0,
) -> dict[str, Any]:
    """Evaluate equivariance and save plot. Returns metrics dict (may be empty)."""
    step("Evaluating equivariance")
    if not bool(OmegaConf.select(analysis_cfg, "equivariance.enabled", default=True)):
        return {}
    eq_max_batches = int(OmegaConf.select(analysis_cfg, "equivariance.max_batches", default=2))
    eq_metrics, eq_err = evaluate_latent_equivariance(
        model,
        dl,
        device,
        max_batches=int(eq_max_batches),
        temporal_sequence_mode=temporal_sequence_mode,
        temporal_static_frame_index=temporal_static_frame_index,
    )
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    return {"equivariance": eq_metrics}


def print_analysis_summary(
    all_metrics: dict[str, Any],
    *,
    n_samples: int,
    out_dir: Path,
    elapsed: float,
) -> None:
    """Print the final analysis summary to stdout."""
    print(f"\n{'=' * 60}\nANALYSIS SUMMARY\n{'=' * 60}")
    print(f"Total samples: {n_samples}, runtime: {elapsed:.1f}s, output: {out_dir}")
    if "pca" in all_metrics and all_metrics["pca"]:
        print(f"PCA: {all_metrics['pca'].get('n_components_95_var', 'N/A')} components for 95% variance")
    if "clustering" in all_metrics and all_metrics["clustering"]:
        cl = all_metrics["clustering"]
        print(f"Clustering: k_values={cl.get('k_values_used')}, primary_k={cl.get('primary_k')}")
        if "ari_with_gt" in cl:
            print(f"ARI={_fmt_metric(cl['ari_with_gt'])}, NMI={_fmt_metric(cl['nmi_with_gt'])}")
    if "equivariance" in all_metrics:
        eq = all_metrics["equivariance"]
        print(f"Equivariance: mean={_fmt_metric(eq.get('eq_latent_rel_error_mean', 'N/A'))}, "
              f"median={_fmt_metric(eq.get('eq_latent_rel_error_median', 'N/A'))}")
    if "real_md_qualitative" in all_metrics:
        real_md_summary = all_metrics["real_md_qualitative"]
        print(
            "Real-MD qualitative analysis: "
            f"{real_md_summary.get('summary_markdown', real_md_summary.get('root_dir', 'N/A'))}"
        )
