import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, ListConfig

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.analysis_utils import (
    _default_cluster_count,
    _sample_indices,
    build_real_coords_dataloader,
    cap_cluster_labels,
    evaluate_latent_equivariance,
    gather_inference_batches,
    segment_grains_with_pose_head,
)
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.vis_tools.md_cluster_plot import (
    save_interactive_md_plot,
)
from src.vis_tools.latent_analysis_vis import (
    compute_hdbscan_labels,
    compute_kmeans_labels,
    save_clustering_analysis,
    save_equivariance_plot,
    save_latent_statistics,
    save_latent_tsne,
    save_local_structure_assignments,
    save_md_space_clusters_plot,
    save_pca_visualization,
    save_tsne_plot_with_coords,
)
from src.vis_tools.tsne_vis import compute_tsne


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


def run_post_training_analysis(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
    cfg: DictConfig | None = None,
    max_batches_latent: int | None = None,
    max_samples_visualization: int | None = None,
    data_files_override: list[str] | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for Barlow Twins."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    step_idx = [0]

    def _step(msg: str) -> None:
        step_idx[0] += 1
        elapsed = time.perf_counter() - t0
        print(f"[analysis][step {step_idx[0]}][{elapsed:7.1f}s] {msg}")

    _step("Loading model")
    model, cfg, device = load_barlow_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)

    def _resolve_analysis_files() -> list[str] | None:
        if getattr(cfg, "data", None) is None:
            return None
        if getattr(cfg.data, "kind", None) != "real":
            return None
        if data_files_override:
            return data_files_override
        if hasattr(cfg.data, "analysis_data_files"):
            files = cfg.data.analysis_data_files
            if isinstance(files, ListConfig):
                return list(files)
            if isinstance(files, str):
                return [files]
            if isinstance(files, list):
                return files
        if hasattr(cfg.data, "analysis_data_file"):
            file = cfg.data.analysis_data_file
            if isinstance(file, str):
                return [file]
        data_files = cfg.data.data_files
        if isinstance(data_files, ListConfig):
            data_files = list(data_files)
        if isinstance(data_files, str):
            data_files = [data_files]
        if not data_files:
            return None
        analysis_single = bool(getattr(cfg.data, "analysis_single_timestep", True))
        if not analysis_single:
            return data_files
        mid_idx = len(data_files) // 2
        return [data_files[mid_idx]]

    analysis_files = _resolve_analysis_files()
    if analysis_files is not None and getattr(cfg, "data", None) is not None:
        cfg.data.data_files = analysis_files
        print(f"Analysis data_files: {analysis_files}")

    # Prefer single-process data loading for analysis to avoid semaphore/shm issues
    # in restricted runtime environments. Can be overridden via config.
    analysis_num_workers = int(getattr(cfg, "analysis_num_workers", 0))
    if hasattr(cfg, "num_workers"):
        cfg.num_workers = analysis_num_workers
    if getattr(cfg, "data", None) is not None and hasattr(cfg.data, "num_workers"):
        cfg.data.num_workers = analysis_num_workers
    print(f"Analysis dataloader workers: {analysis_num_workers}")

    tsne_max_samples = int(getattr(cfg, "analysis_tsne_max_samples", 8000))
    if max_samples_visualization is not None:
        tsne_max_samples = min(tsne_max_samples, max_samples_visualization)
    clustering_max_samples = int(getattr(cfg, "analysis_clustering_max_samples", 20000))
    cluster_method = str(getattr(cfg, "analysis_cluster_method", "auto")).lower()
    cluster_l2_normalize = bool(getattr(cfg, "analysis_cluster_l2_normalize", True))
    cluster_standardize = bool(getattr(cfg, "analysis_cluster_standardize", True))
    cluster_pca_var = float(getattr(cfg, "analysis_cluster_pca_var", 0.98))
    cluster_pca_max_components = int(
        getattr(cfg, "analysis_cluster_pca_max_components", 32)
    )
    cluster_silhouette_max_samples = int(
        getattr(cfg, "analysis_cluster_silhouette_max_samples", 5000)
    )
    cluster_silhouette_tolerance = float(
        getattr(cfg, "analysis_cluster_silhouette_tolerance", 0.03)
    )
    cluster_k_min = int(getattr(cfg, "analysis_cluster_k_min", 2))
    cluster_k_max = int(getattr(cfg, "analysis_cluster_k_max", 12))
    md_use_all_points = bool(getattr(cfg, "analysis_md_use_all_points", True))
    hdbscan_enabled = bool(getattr(cfg, "analysis_hdbscan_enabled", True))
    hdbscan_fit_fraction = float(getattr(cfg, "analysis_hdbscan_fit_fraction", 0.25))
    hdbscan_max_fit_samples = int(getattr(cfg, "analysis_hdbscan_max_fit_samples", 50000))
    hdbscan_target_k_min = int(getattr(cfg, "analysis_hdbscan_target_k_min", 5))
    hdbscan_target_k_max = int(getattr(cfg, "analysis_hdbscan_target_k_max", 6))
    hdbscan_min_samples = getattr(cfg, "analysis_hdbscan_min_samples", None)
    if hdbscan_min_samples is not None:
        hdbscan_min_samples = int(hdbscan_min_samples)
    hdbscan_cluster_selection_epsilon = float(
        getattr(cfg, "analysis_hdbscan_cluster_selection_epsilon", 0.0)
    )
    hdbscan_cluster_selection_method = str(
        getattr(cfg, "analysis_hdbscan_cluster_selection_method", "leaf")
    ).lower()
    hdbscan_min_cluster_size_candidates_cfg = getattr(
        cfg, "analysis_hdbscan_min_cluster_size_candidates", None
    )
    hdbscan_min_cluster_size_candidates = None
    if isinstance(hdbscan_min_cluster_size_candidates_cfg, ListConfig):
        hdbscan_min_cluster_size_candidates = [
            int(v) for v in hdbscan_min_cluster_size_candidates_cfg
        ]
    elif isinstance(hdbscan_min_cluster_size_candidates_cfg, list):
        hdbscan_min_cluster_size_candidates = [int(v) for v in hdbscan_min_cluster_size_candidates_cfg]
    grain_enabled = bool(getattr(cfg, "analysis_grain_enabled", True))
    grain_knn_k = int(getattr(cfg, "analysis_grain_knn_k", 12))
    grain_edge_weight_threshold = float(
        getattr(cfg, "analysis_grain_edge_weight_threshold", 0.35)
    )
    grain_orientation_tau_deg = float(
        getattr(cfg, "analysis_grain_orientation_tau_deg", 18.0)
    )
    grain_alpha_scale_quantile = float(
        getattr(cfg, "analysis_grain_alpha_scale_quantile", 0.75)
    )
    grain_align_n_iters = int(getattr(cfg, "analysis_grain_align_n_iters", 5))
    grain_align_min_cluster_size = int(
        getattr(cfg, "analysis_grain_align_min_cluster_size", 3)
    )
    grain_align_normalize_channels = bool(
        getattr(cfg, "analysis_grain_align_normalize_channels", True)
    )
    grain_min_size = int(getattr(cfg, "analysis_grain_min_size", 1))
    grain_interactive_max_points = getattr(
        cfg, "analysis_grain_interactive_max_points", 120000
    )
    if grain_interactive_max_points is not None:
        grain_interactive_max_points = int(grain_interactive_max_points)
        if grain_interactive_max_points <= 0:
            grain_interactive_max_points = None
    grain_interactive_max_grains = int(
        getattr(cfg, "analysis_grain_interactive_max_grains", 80)
    )
    if grain_interactive_max_grains <= 0:
        grain_interactive_max_grains = 0
    grain_tsne_max_grains = int(getattr(cfg, "analysis_grain_tsne_max_grains", 60))
    if grain_tsne_max_grains <= 0:
        grain_tsne_max_grains = 0

    # Older checkpoint-local Hydra configs can carry analysis_extra_k_plots=false.
    # Keep k+1/k+2 analyses always enabled for MD diagnostics.
    extra_k_plots_cfg = bool(getattr(cfg, "analysis_extra_k_plots", True))
    extra_k_plots = True
    if not extra_k_plots_cfg:
        print(
            "[analysis] Overriding analysis_extra_k_plots=false from checkpoint config "
            "-> enabled (k+1,k+2)."
        )
    progress_every_batches = int(getattr(cfg, "analysis_progress_every_batches", 25))
    print(f"t-SNE sample cap: {tsne_max_samples}")
    print(f"Clustering metrics cap: {clustering_max_samples}")
    _step("Building datamodule")
    dm = build_datamodule(cfg)
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    dm.setup(stage="fit")
    print("Using ALL dataset splits (train + test) for latent analysis")

    if is_synthetic:
        train_dataset = getattr(dm, "train_dataset", None)
        test_dataset = getattr(dm, "test_dataset", None)
        if train_dataset is None or test_dataset is None:
            raise ValueError("Synthetic datamodule is missing train/test datasets.")
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        dl = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        dl = build_real_coords_dataloader(cfg, dm, use_train_data=True, use_full_dataset=True)
        print(
            "Real data detected: using full dataset for local-structure clustering visualization"
        )

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
        cfg_max_batches = getattr(cfg, "analysis_max_batches_latent", None)
        if cfg_max_batches is not None:
            max_batches_latent = int(cfg_max_batches)
            if max_batches_latent <= 0:
                max_batches_latent = None
    max_samples_total = getattr(cfg, "analysis_max_samples_total", None)
    if max_samples_total is None and not is_synthetic:
        max_samples_total = 20000
    if not is_synthetic and md_use_all_points:
        max_samples_total = None
    if max_samples_total is not None:
        max_samples_total = int(max_samples_total)
        if max_samples_total <= 0:
            max_samples_total = None

    _step("Collecting inference batches")
    if max_batches_latent is None:
        print("Gathering inference batches (ALL batches)...")
    else:
        print(f"Gathering inference batches (up to {max_batches_latent} batches)...")
    if max_samples_total is not None:
        print(f"Collecting up to {max_samples_total} samples for analysis")
    seed_base = getattr(cfg, "analysis_seed_base", 123)
    cache = gather_inference_batches(
        model,
        dl,
        device,
        max_batches=max_batches_latent,
        max_samples_total=max_samples_total,
        collect_coords=not is_synthetic,
        seed_base=seed_base,
        progress_every_batches=progress_every_batches,
        verbose=True,
    )

    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")
    has_phases = cache["phases"].size == n_samples

    all_metrics: Dict[str, Any] = {}

    if is_synthetic:
        _step("Computing t-SNE visualization")
        save_latent_tsne(
            cache["inv_latents"],
            cache["phases"],
            out_dir,
            max_samples=tsne_max_samples,
            class_names=class_names,
        )

    _step("Computing PCA analysis")
    pca_stats = save_pca_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=None,
        class_names=class_names,
    )
    all_metrics["pca"] = pca_stats

    _step("Computing latent statistics")
    latent_stats = save_latent_statistics(
        cache["inv_latents"],
        cache["eq_latents"],
        cache["phases"],
        out_dir,
        class_names=class_names,
    )
    all_metrics["latent_stats"] = latent_stats

    _step("Computing clustering analysis")
    clustering_metrics = save_clustering_analysis(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=clustering_max_samples,
        class_names=class_names,
        cluster_method=cluster_method,
        l2_normalize=cluster_l2_normalize,
        standardize=cluster_standardize,
        pca_variance=cluster_pca_var,
        pca_max_components=cluster_pca_max_components,
        silhouette_max_samples=cluster_silhouette_max_samples,
        silhouette_tolerance=cluster_silhouette_tolerance,
        k_min=cluster_k_min,
        k_max=cluster_k_max,
    )
    all_metrics["clustering"] = clustering_metrics

    if not is_synthetic:
        coords = cache.get("coords", np.empty((0, 3), dtype=np.float32))
        if coords.shape[0] != len(cache["inv_latents"]):
            print(
                "Warning: coordinate count does not match latent count; "
                "skipping spatial clustering visualization."
            )
            coords = np.empty((0, 3), dtype=np.float32)

        best_k = clustering_metrics.get("best_k_silhouette") if clustering_metrics else None
        if not isinstance(best_k, int) or best_k <= 1:
            best_k = _default_cluster_count(len(cache["inv_latents"]))
        selected_method_for_best_k = (
            str(clustering_metrics.get("best_method", cluster_method)).lower()
            if clustering_metrics
            else cluster_method
        )
        cluster_labels, cluster_fit_info = compute_kmeans_labels(
            cache["inv_latents"],
            best_k,
            method=selected_method_for_best_k,
            l2_normalize=cluster_l2_normalize,
            standardize=cluster_standardize,
            pca_variance=cluster_pca_var,
            pca_max_components=cluster_pca_max_components,
            silhouette_max_samples=cluster_silhouette_max_samples,
            return_info=True,
        )
        cluster_label_method = str(cluster_fit_info.get("method", selected_method_for_best_k))
        if "clustering" in all_metrics and isinstance(all_metrics["clustering"], dict):
            all_metrics["clustering"]["labels_k_method"] = cluster_label_method

        _step("Computing t-SNE visualization (clusters)")
        tsne_n_iter = int(getattr(cfg, "analysis_tsne_n_iter", 1000))
        tsne_idx = _sample_indices(len(cache["inv_latents"]), tsne_max_samples)
        tsne_latents = cache["inv_latents"][tsne_idx]
        tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
        tsne_coords = compute_tsne(tsne_latents, perplexity=tsne_perplexity, n_iter=tsne_n_iter)

        save_tsne_plot_with_coords(
            tsne_coords,
            cluster_labels[tsne_idx],
            out_dir,
            out_name="latent_tsne_clusters.png",
            title=f"Latent space t-SNE ({cluster_label_method}, k={best_k})",
        )

        if extra_k_plots:
            k_candidates = [int(best_k) + 1, int(best_k) + 2]
            unique_k: list[int] = []
            for k_val in k_candidates:
                k_val = max(2, min(k_val, len(cache["inv_latents"])))
                if k_val not in unique_k:
                    unique_k.append(k_val)

            for k_val in unique_k:
                if k_val == int(best_k):
                    continue
                labels_k, labels_k_info = compute_kmeans_labels(
                    cache["inv_latents"],
                    k_val,
                    method=cluster_method,
                    l2_normalize=cluster_l2_normalize,
                    standardize=cluster_standardize,
                    pca_variance=cluster_pca_var,
                    pca_max_components=cluster_pca_max_components,
                    silhouette_max_samples=cluster_silhouette_max_samples,
                    return_info=True,
                )
                method_k = str(labels_k_info.get("method", cluster_method))
                save_tsne_plot_with_coords(
                    tsne_coords,
                    labels_k[tsne_idx],
                    out_dir,
                    out_name=f"latent_tsne_clusters_k{k_val}.png",
                    title=f"Latent space t-SNE ({method_k}, k={k_val})",
                )

        if coords.size and cluster_labels.size:
            _step("Saving coordinate-space clustering outputs")
            interactive_max_points = getattr(cfg, "analysis_interactive_max_points", None)
            if interactive_max_points is not None:
                interactive_max_points = int(interactive_max_points)
                if interactive_max_points <= 0:
                    interactive_max_points = None
            if md_use_all_points:
                interactive_max_points = None
            coord_files = save_local_structure_assignments(
                coords,
                cluster_labels,
                out_dir,
            )
            if coord_files:
                save_md_space_clusters_plot(
                    coords,
                    cluster_labels,
                    out_dir / "md_space_clusters.png",
                    max_points=interactive_max_points,
                )
                interactive_path = None
                interactive_paths: Dict[int, str] = {}
                try:
                    interactive_path = out_dir / "md_space_clusters.html"
                    save_interactive_md_plot(
                        coords,
                        cluster_labels,
                        interactive_path,
                        palette="Set3",
                        max_points=interactive_max_points,
                        marker_size=3.0,
                        marker_line_width=0.0,
                        aspect_mode="cube",
                    )
                    interactive_paths[int(best_k)] = str(interactive_path)

                    if extra_k_plots:
                        k_candidates = [int(best_k) + 1, int(best_k) + 2]

                        unique_k: list[int] = []
                        for k_val in k_candidates:
                            k_val = max(2, min(k_val, len(cache["inv_latents"])))
                            if k_val not in unique_k:
                                unique_k.append(k_val)

                        for k_val in unique_k:
                            if k_val == int(best_k):
                                continue
                            labels_k, _ = compute_kmeans_labels(
                                cache["inv_latents"],
                                k_val,
                                method=cluster_method,
                                l2_normalize=cluster_l2_normalize,
                                standardize=cluster_standardize,
                                pca_variance=cluster_pca_var,
                                pca_max_components=cluster_pca_max_components,
                                silhouette_max_samples=cluster_silhouette_max_samples,
                                return_info=True,
                            )
                            out_path = out_dir / f"md_space_clusters_k{k_val}.html"
                            save_interactive_md_plot(
                                coords,
                                labels_k,
                                out_path,
                                palette="Set3",
                                max_points=interactive_max_points,
                                marker_size=3.0,
                                marker_line_width=0.0,
                                aspect_mode="cube",
                            )
                            interactive_paths[int(k_val)] = str(out_path)
                except ImportError:
                    interactive_path = None
                    print("Plotly not installed; skipping interactive MD plot.")

                hdbscan_info: Dict[str, Any] | None = None
                hdbscan_path = None
                hdbscan_coord_files: Dict[str, str] = {}
                if hdbscan_enabled:
                    _step("Running HDBSCAN clustering (sampled fit)")
                    try:
                        hdbscan_labels, hdbscan_info = compute_hdbscan_labels(
                            cache["inv_latents"],
                            sample_fraction=hdbscan_fit_fraction,
                            max_fit_samples=hdbscan_max_fit_samples,
                            random_state=42,
                            l2_normalize=cluster_l2_normalize,
                            standardize=cluster_standardize,
                            pca_variance=cluster_pca_var,
                            pca_max_components=cluster_pca_max_components,
                            target_clusters_min=hdbscan_target_k_min,
                            target_clusters_max=hdbscan_target_k_max,
                            min_cluster_size_candidates=hdbscan_min_cluster_size_candidates,
                            min_samples=hdbscan_min_samples,
                            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                            cluster_selection_method=hdbscan_cluster_selection_method,
                            return_info=True,
                        )
                        if hdbscan_labels.size == len(coords):
                            hdbscan_coord_files = save_local_structure_assignments(
                                coords,
                                hdbscan_labels,
                                out_dir,
                                prefix="local_structure_hdbscan",
                            )
                            try:
                                hdbscan_path = out_dir / "md_space_clusters_hdbscan.html"
                                n_hdb_clusters = int(
                                    len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
                                )
                                save_interactive_md_plot(
                                    coords,
                                    hdbscan_labels,
                                    hdbscan_path,
                                    palette="Set3",
                                    max_points=interactive_max_points,
                                    marker_size=3.0,
                                    marker_line_width=0.0,
                                    title=(
                                        "MD local-structure clusters "
                                        f"(HDBSCAN, n={len(coords)}, k={n_hdb_clusters})"
                                    ),
                                    label_prefix="HDBSCAN",
                                    aspect_mode="cube",
                                )
                            except ImportError:
                                hdbscan_path = None
                                print("Plotly not installed; skipping HDBSCAN interactive MD plot.")
                        else:
                            print(
                                "Warning: HDBSCAN labels do not match coordinate count; "
                                "skipping HDBSCAN MD outputs."
                            )
                    except ImportError:
                        print("HDBSCAN package not installed; skipping HDBSCAN analysis.")
                    except Exception as err:
                        print(f"HDBSCAN analysis failed: {err}")

                grain_info: Dict[str, Any] | None = None
                grain_path = None
                grain_coord_files: Dict[str, str] = {}
                grain_tsne_path = None
                if grain_enabled:
                    _step("Running grain segmentation")
                    try:
                        grain_result = segment_grains_with_pose_head(
                            model,
                            cache["inv_latents"],
                            cache["eq_latents"],
                            cluster_labels,
                            coords,
                            knn_k=grain_knn_k,
                            edge_weight_threshold=grain_edge_weight_threshold,
                            orientation_tau_deg=grain_orientation_tau_deg,
                            alpha_scale_quantile=grain_alpha_scale_quantile,
                            align_n_iters=grain_align_n_iters,
                            align_min_cluster_size=grain_align_min_cluster_size,
                            align_normalize_channels=grain_align_normalize_channels,
                            min_grain_size=grain_min_size,
                        )
                        grain_labels = np.asarray(
                            grain_result.get("grain_labels", np.empty((0,), dtype=int)),
                            dtype=int,
                        )
                        grain_info = grain_result.get("metrics", None)
                        if grain_labels.size == len(coords):
                            grain_labels_tsne = grain_labels
                            if grain_tsne_max_grains > 0:
                                n_unique_grains = int(len(np.unique(grain_labels[grain_labels >= 0])))
                                if n_unique_grains > grain_tsne_max_grains:
                                    grain_labels_tsne = cap_cluster_labels(
                                        grain_labels,
                                        max_clusters=grain_tsne_max_grains,
                                        other_label=-1,
                                    )
                            n_grains_tsne = int(
                                len(np.unique(grain_labels_tsne[grain_labels_tsne >= 0]))
                            )
                            save_tsne_plot_with_coords(
                                tsne_coords,
                                grain_labels_tsne[tsne_idx],
                                out_dir,
                                out_name="latent_tsne_grains.png",
                                title=(
                                    "Latent space t-SNE "
                                    f"(grains, shown={n_grains_tsne})"
                                ),
                                legend_title="grain",
                            )
                            grain_tsne_path = str(out_dir / "latent_tsne_grains.png")

                            grain_coord_files = save_local_structure_assignments(
                                coords,
                                grain_labels,
                                out_dir,
                                prefix="local_structure_grains",
                            )
                            if grain_coord_files:
                                save_md_space_clusters_plot(
                                    coords,
                                    grain_labels,
                                    out_dir / "md_space_grains.png",
                                    max_points=interactive_max_points,
                                )
                                try:
                                    grain_path = out_dir / "md_space_grains.html"
                                    grain_labels_plot = grain_labels
                                    if grain_interactive_max_grains > 0:
                                        n_unique_grains = int(
                                            len(np.unique(grain_labels[grain_labels >= 0]))
                                        )
                                        if n_unique_grains > grain_interactive_max_grains:
                                            grain_labels_plot = cap_cluster_labels(
                                                grain_labels,
                                                max_clusters=grain_interactive_max_grains,
                                                other_label=-1,
                                            )
                                            print(
                                                "Reducing interactive grain labels: "
                                                f"{n_unique_grains} -> {grain_interactive_max_grains} "
                                                "(smaller grains collapsed to -1)"
                                            )
                                    n_grains = int(
                                        len(np.unique(grain_labels_plot[grain_labels_plot >= 0]))
                                    )
                                    save_interactive_md_plot(
                                        coords,
                                        grain_labels_plot,
                                        grain_path,
                                        palette="Set3",
                                        max_points=grain_interactive_max_points,
                                        marker_size=2.6,
                                        marker_line_width=0.0,
                                        title=(
                                            "MD grain segmentation "
                                            f"(n={len(coords)}, shown_grains={n_grains})"
                                        ),
                                        label_prefix="Grain",
                                        aspect_mode="cube",
                                    )
                                except ImportError:
                                    grain_path = None
                                    print("Plotly not installed; skipping interactive MD grain plot.")
                        else:
                            print(
                                "Warning: grain labels do not match coordinate count; "
                                "skipping grain MD outputs."
                            )
                    except ImportError:
                        print("Scipy not installed; skipping grain segmentation.")
                    except Exception as err:
                        print(f"Grain segmentation failed: {err}")

                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                all_metrics["real_md"] = {
                    "n_clusters": int(len(unique_labels)),
                    "cluster_counts": {int(k): int(v) for k, v in zip(unique_labels, counts)},
                    "coords_files": coord_files,
                }
                if interactive_path is not None:
                    all_metrics["real_md"]["interactive_html"] = str(interactive_path)
                if interactive_paths:
                    all_metrics["real_md"]["interactive_htmls"] = interactive_paths
                if hdbscan_info is not None:
                    all_metrics["real_md"]["hdbscan"] = hdbscan_info
                    if hdbscan_coord_files:
                        all_metrics["real_md"]["hdbscan_coords_files"] = hdbscan_coord_files
                    if hdbscan_path is not None:
                        all_metrics["real_md"]["hdbscan_interactive_html"] = str(hdbscan_path)
                if grain_info is not None:
                    all_metrics["real_md"]["grains"] = grain_info
                    all_metrics["real_md"]["grains"]["interactive_max_points"] = (
                        None if grain_interactive_max_points is None else int(grain_interactive_max_points)
                    )
                    all_metrics["real_md"]["grains"]["interactive_max_grains"] = int(
                        grain_interactive_max_grains
                    )
                    all_metrics["real_md"]["grains"]["tsne_max_grains"] = int(
                        grain_tsne_max_grains
                    )
                    if grain_coord_files:
                        all_metrics["real_md"]["grains_coords_files"] = grain_coord_files
                    if grain_path is not None:
                        all_metrics["real_md"]["grains_interactive_html"] = str(grain_path)
                    if grain_tsne_path is not None:
                        all_metrics["real_md"]["grains_tsne_png"] = grain_tsne_path
    _step("Evaluating equivariance")
    eq_metrics, eq_err = evaluate_latent_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    all_metrics["equivariance"] = eq_metrics

    _step("Writing metrics")
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
            f"Equivariant latent error (seeded mean): {eq.get('eq_latent_rel_error_mean', 'N/A'):.4f}"
            if isinstance(eq.get("eq_latent_rel_error_mean"), float)
            else f"Equivariant latent error (seeded mean): {eq.get('eq_latent_rel_error_mean', 'N/A')}"
        )
        print(
            f"Equivariant latent error (seeded median): {eq.get('eq_latent_rel_error_median', 'N/A'):.4f}"
            if isinstance(eq.get("eq_latent_rel_error_median"), float)
            else f"Equivariant latent error (seeded median): {eq.get('eq_latent_rel_error_median', 'N/A')}"
        )
        if "eq_latent_rel_error_unseeded" in eq:
            print(
                f"Equivariant latent error (unseeded mean): {eq.get('eq_latent_rel_error_unseeded', 'N/A'):.4f}"
                if isinstance(eq.get("eq_latent_rel_error_unseeded"), float)
                else f"Equivariant latent error (unseeded mean): {eq.get('eq_latent_rel_error_unseeded', 'N/A')}"
            )
        if "eq_latent_rel_error_unseeded_median" in eq:
            print(
                f"Equivariant latent error (unseeded median): {eq.get('eq_latent_rel_error_unseeded_median', 'N/A'):.4f}"
                if isinstance(eq.get("eq_latent_rel_error_unseeded_median"), float)
                else f"Equivariant latent error (unseeded median): {eq.get('eq_latent_rel_error_unseeded_median', 'N/A')}"
            )
        if "eq_latent_nondeterminism_contribution" in eq:
            print(
                f"Non-determinism contribution (unseeded - seeded): {eq.get('eq_latent_nondeterminism_contribution', 'N/A'):.4f}"
                if isinstance(eq.get("eq_latent_nondeterminism_contribution"), float)
                else f"Non-determinism contribution (unseeded - seeded): {eq.get('eq_latent_nondeterminism_contribution', 'N/A')}"
            )

    print("=" * 60)
    print(f"\nSaved all analyses to {out_dir}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    print("Generated files:")
    if has_phases:
        print("  - latent_tsne_ground_truth.png: t-SNE with ground truth labels")
    print("  - latent_tsne_clusters.png: t-SNE with clustering labels")
    print("  - latent_pca_analysis.png: PCA projection and variance")
    print("  - latent_pca_3d.png: 3D PCA projection")
    print("  - latent_statistics.png: Comprehensive latent statistics")
    print("  - clustering_analysis.png: Clustering quality metrics")
    print("  - equivariance.png: Equivariant latent error distribution")
    print("  - analysis_metrics.json: All numerical metrics")
    if not is_synthetic and "real_md" in all_metrics:
        print("  - local_structure_coords_clusters.csv: local-structure centers with cluster IDs")
        print("  - local_structure_coords_clusters.npz: local-structure centers + cluster IDs")
        print("  - md_space_clusters.png: 3D MD space clusters")
        print("  - md_space_clusters.html: interactive 3D MD space clusters")
        if extra_k_plots:
            print("  - md_space_clusters_k*.html: interactive 3D MD plots for k+1,k+2")
            print("  - latent_tsne_clusters_k*.png: t-SNE plots for k+1,k+2")
        if hdbscan_enabled:
            print("  - local_structure_hdbscan_coords_clusters.csv: MD centers with HDBSCAN labels")
            print("  - local_structure_hdbscan_coords_clusters.npz: MD centers + HDBSCAN labels")
            print("  - md_space_clusters_hdbscan.html: interactive 3D MD HDBSCAN clusters")
        if "grains_coords_files" in all_metrics["real_md"]:
            print("  - local_structure_grains_coords_clusters.csv: MD centers with grain IDs")
            print("  - local_structure_grains_coords_clusters.npz: MD centers + grain IDs")
            print("  - md_space_grains.png: 3D MD grain segmentation")
            if "grains_tsne_png" in all_metrics["real_md"]:
                print("  - latent_tsne_grains.png: t-SNE colored by grain IDs")
            if "grains_interactive_html" in all_metrics["real_md"]:
                print("  - md_space_grains.html: interactive 3D MD grain segmentation")

    return all_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-training analysis for contrastive (Barlow Twins) checkpoints.",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write analysis outputs (default: <ckpt_dir>/analysis).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--max_batches_latent",
        type=int,
        default=None,
        help="Max batches to use for latent analysis (default: all).",
    )
    parser.add_argument(
        "--max_samples_visualization",
        type=int,
        default=None,
        help="Max samples for t-SNE (default: analysis_tsne_max_samples or 8000).",
    )
    parser.add_argument(
        "--data_file",
        action="append",
        default=None,
        help="Override real data files (repeat for multiple). Example: --data_file 175ps.off",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "analysis")
    else:
        output_dir = os.path.expanduser(output_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)

    run_post_training_analysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        cuda_device=int(args.cuda_device),
        cfg=None,
        max_batches_latent=args.max_batches_latent,
        max_samples_visualization=args.max_samples_visualization,
        data_files_override=args.data_file,
    )


if __name__ == "__main__":
    main()
