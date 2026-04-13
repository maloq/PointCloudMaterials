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
from src.vis_tools.tsne_vis import (
    compute_tsne,
    compute_umap,
    save_embedding_comparison_plot,
    save_tsne_plot,
)


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
    comparison_labels_by_method: dict[str, dict[int, np.ndarray]] | None,
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
) -> dict[str, Any]:
    """Compute latent 2D projections and optional clustering-method comparison plots."""
    if not bool(OmegaConf.select(analysis_cfg, "tsne.enabled", default=True)):
        step("Skipping t-SNE visualization")
        return {}

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
    projection_metrics: dict[str, Any] = {
        "sample_count": int(len(tsne_latents)),
        "sample_indices_count": int(len(tsne_idx)),
        "tsne": {
            "perplexity": int(tsne_perplexity),
            "n_iter": int(tsne_n_iter),
            "random_state": int(clustering_random_state),
            "sample_count": int(len(tsne_latents)),
        },
    }
    if is_synthetic and cache["phases"].size == len(cache["inv_latents"]):
        save_tsne_plot(
            tsne_coords,
            cache["phases"][tsne_idx],
            out_file=str(out_dir / "latent_tsne_ground_truth.png"),
            title=f"Latent space t-SNE (n={len(tsne_latents)}, ground truth phases)",
            legend_title="phase",
            class_names=class_names,
        )

    umap_comparison_enabled = bool(
        OmegaConf.select(analysis_cfg, "tsne.umap_comparison_enabled", default=False)
    )
    umap_coords: np.ndarray | None = None
    if umap_comparison_enabled:
        step("Computing UMAP visualization (for t-SNE comparison)")
        umap_neighbors = int(
            OmegaConf.select(analysis_cfg, "tsne.umap_neighbors", default=30)
        )
        umap_min_dist = float(
            OmegaConf.select(analysis_cfg, "tsne.umap_min_dist", default=0.15)
        )
        umap_metric = str(
            OmegaConf.select(analysis_cfg, "tsne.umap_metric", default="euclidean")
        )
        umap_backend = str(
            OmegaConf.select(analysis_cfg, "tsne.umap_backend", default="auto")
        )
        umap_coords, umap_info = compute_umap(
            tsne_latents,
            random_state=clustering_random_state,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            backend=umap_backend,
            return_info=True,
        )
        projection_metrics["umap"] = {
            **umap_info,
            "sample_count": int(len(tsne_latents)),
        }
        if is_synthetic and cache["phases"].size == len(cache["inv_latents"]):
            save_tsne_plot(
                umap_coords,
                cache["phases"][tsne_idx],
                out_file=str(out_dir / "latent_umap_ground_truth.png"),
                title=f"Latent space UMAP (n={len(tsne_latents)}, ground truth phases)",
                legend_title="phase",
                class_names=class_names,
            )

    comparison_method_names: list[str] = []
    if comparison_labels_by_method:
        comparison_method_names = [str(method_name) for method_name in comparison_labels_by_method.keys()]
    projection_metrics["comparison_methods"] = comparison_method_names

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
        if umap_coords is not None:
            umap_out_name = (
                "latent_umap_clusters.png"
                if idx_k == 0
                else f"latent_umap_clusters_k{k_val}.png"
            )
            save_tsne_plot_with_coords(
                umap_coords,
                labels_k[tsne_idx],
                out_dir,
                out_name=umap_out_name,
                title=f"Latent space UMAP ({method_k}, k={k_val})",
                cluster_color_map=shared_cluster_color_maps_by_k.get(int(k_val)),
            )

        if comparison_labels_by_method:
            primary_method_name = str(method_k)
            primary_method_key = next(
                (
                    method_name
                    for method_name in comparison_labels_by_method.keys()
                    if str(method_name) == str(primary_method_name)
                ),
                None,
            )
            if primary_method_key is None:
                raise KeyError(
                    "Primary clustering method is missing from comparison_labels_by_method. "
                    f"primary_method={primary_method_name!r}, "
                    f"available={list(comparison_labels_by_method.keys())}."
                )
            left_labels = np.asarray(
                comparison_labels_by_method[primary_method_key][int(k_val)][tsne_idx],
                dtype=int,
            )
            for comparison_method_name, labels_by_k in comparison_labels_by_method.items():
                if str(comparison_method_name) == str(primary_method_key):
                    continue
                if int(k_val) not in labels_by_k:
                    raise KeyError(
                        "Comparison clustering method is missing labels for the requested k. "
                        f"method={comparison_method_name!r}, k={int(k_val)}, "
                        f"available={sorted(labels_by_k.keys())}."
                    )
                right_labels = np.asarray(labels_by_k[int(k_val)][tsne_idx], dtype=int)
                tsne_method_out_name = (
                    f"latent_tsne_cluster_method_comparison_{primary_method_key}_vs_{comparison_method_name}.png"
                    if idx_k == 0
                    else (
                        "latent_tsne_cluster_method_comparison_"
                        f"{primary_method_key}_vs_{comparison_method_name}_k{k_val}.png"
                    )
                )
                save_embedding_comparison_plot(
                    tsne_coords,
                    tsne_coords,
                    left_labels,
                    right_labels,
                    out_file=str(out_dir / tsne_method_out_name),
                    left_title=f"t-SNE ({primary_method_key}, k={k_val})",
                    right_title=f"t-SNE ({comparison_method_name}, k={k_val})",
                    overall_title=f"t-SNE clustering-method comparison (k={k_val})",
                    legend_title="cluster",
                    cluster_color_map=shared_cluster_color_maps_by_k.get(int(k_val)),
                    label_prefix="C",
                )
                if umap_coords is not None:
                    umap_method_out_name = (
                        f"latent_umap_cluster_method_comparison_{primary_method_key}_vs_{comparison_method_name}.png"
                        if idx_k == 0
                        else (
                            "latent_umap_cluster_method_comparison_"
                            f"{primary_method_key}_vs_{comparison_method_name}_k{k_val}.png"
                        )
                    )
                    save_embedding_comparison_plot(
                        umap_coords,
                        umap_coords,
                        left_labels,
                        right_labels,
                        out_file=str(out_dir / umap_method_out_name),
                        left_title=f"UMAP ({primary_method_key}, k={k_val})",
                        right_title=f"UMAP ({comparison_method_name}, k={k_val})",
                        overall_title=f"UMAP clustering-method comparison (k={k_val})",
                        legend_title="cluster",
                        cluster_color_map=shared_cluster_color_maps_by_k.get(int(k_val)),
                        label_prefix="C",
                    )

    return projection_metrics


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
    if "clustering_comparison" in all_metrics and all_metrics["clustering_comparison"]:
        comparison = all_metrics["clustering_comparison"]
        primary_k = int(comparison.get("primary_k", -1))
        pairwise_by_k = comparison.get("pairwise_by_k", {})
        pairwise_for_primary = (
            pairwise_by_k.get(primary_k)
            if isinstance(pairwise_by_k, dict)
            else None
        )
        if pairwise_for_primary is None and isinstance(pairwise_by_k, dict):
            pairwise_for_primary = pairwise_by_k.get(str(primary_k))
        if isinstance(pairwise_for_primary, dict) and pairwise_for_primary:
            for pair_name, pair_metrics in pairwise_for_primary.items():
                print(
                    "Clustering comparison "
                    f"(k={primary_k}, {pair_name}): "
                    f"ARI={_fmt_metric(pair_metrics.get('ari', 'N/A'))}, "
                    f"NMI={_fmt_metric(pair_metrics.get('nmi', 'N/A'))}"
                )
    if "latent_projection_visualizations" in all_metrics:
        projection_summary = all_metrics["latent_projection_visualizations"]
        method_text = projection_summary.get("comparison_methods") or []
        if "umap" in projection_summary and method_text:
            print(
                "Latent 2D projections: "
                f"t-SNE + UMAP with clustering-method comparison on {projection_summary.get('sample_count', 'N/A')} samples "
                f"for methods {method_text}"
            )
        elif "umap" in projection_summary:
            print(
                "Latent 2D projections: "
                f"t-SNE + UMAP on {projection_summary.get('sample_count', 'N/A')} samples"
            )
        else:
            print(
                "Latent 2D projections: "
                f"t-SNE on {projection_summary.get('sample_count', 'N/A')} samples"
            )
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
