import argparse
import json
import os
import sys
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
    cluster_orientations_by_cluster,
    compute_cluster_prototypes_and_alignments,
    compute_cluster_prototypes_and_alignments_with_rotation_head,
    compute_self_symmetry_metrics,
    evaluate_pose_head_relative_rotation,
    evaluate_latent_equivariance,
    gather_inference_batches,
)
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.training_methods.contrastive_learning.orientation_sync_analysis import (
    evaluate_pairwise_orientation_synchronization,
)
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path
from src.vis_tools.md_cluster_plot import (
    save_interactive_md_continuous_plot,
    save_interactive_md_plot,
    save_interactive_md_two_layer_plot,
)
from src.vis_tools.latent_analysis_vis import (
    compute_kmeans_labels,
    save_cluster_orientation_histograms,
    save_cluster_symmetry_boxplots,
    save_clustering_analysis,
    save_equivariance_plot,
    save_latent_statistics,
    save_latent_tsne,
    save_local_structure_assignments,
    save_md_space_clusters_plot,
    save_pca_visualization,
    save_tsne_continuous_plot,
    save_tsne_plot_with_coords,
)
from src.vis_tools.tsne_vis import compute_tsne


def _rotation_angles_deg(rot_mats: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    if rot_mats.size == 0:
        return np.empty((0,), dtype=np.float32)
    traces = np.trace(rot_mats, axis1=1, axis2=2)
    cos_theta = (traces - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return np.degrees(np.arccos(cos_theta)).astype(np.float32)


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
    use_train_data: bool = True,
    data_files_override: list[str] | None = None,
    analysis_orientation_reference: str | None = None,
    analysis_pose_eval_max_batches: int | None = None,
    analysis_pose_eval_seed_base: int | None = None,
) -> Dict[str, Any]:
    """Generate qualitative and quantitative diagnostics for Barlow Twins."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, cfg, device = load_barlow_model(checkpoint_path, cuda_device=cuda_device, cfg=cfg)
    orientation_reference_override = (
        str(analysis_orientation_reference).strip().lower()
        if analysis_orientation_reference is not None
        else None
    )
    pose_eval_max_batches_override = (
        int(analysis_pose_eval_max_batches)
        if analysis_pose_eval_max_batches is not None
        else None
    )
    pose_eval_seed_base_override = (
        int(analysis_pose_eval_seed_base)
        if analysis_pose_eval_seed_base is not None
        else None
    )

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

    tsne_max_samples = 20000
    if max_samples_visualization is not None:
        tsne_max_samples = min(tsne_max_samples, max_samples_visualization)
    clustering_max_samples = int(getattr(cfg, "analysis_clustering_max_samples", 50000))
    print(f"t-SNE sample cap: {tsne_max_samples}")
    print(f"Clustering metrics cap: {clustering_max_samples}")
    dm = build_datamodule(cfg)
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    if use_train_data:
        dm.setup(stage="fit")
        print("Using TRAINING dataset for latent analysis")
    else:
        print("Using TEST dataset for latent analysis")

    if is_synthetic:
        dl = dm.train_dataloader() if use_train_data else dm.test_dataloader()
    else:
        dl = build_real_coords_dataloader(cfg, dm, use_train_data, use_full_dataset=True)
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
        print("Gathering inference batches (ALL batches)...")
    else:
        print(f"Gathering inference batches (up to {max_batches_latent} batches)...")
    seed_base = getattr(cfg, "analysis_seed_base", 123)
    cache = gather_inference_batches(
        model,
        dl,
        device,
        max_batches=max_batches_latent,
        collect_coords=not is_synthetic,
        seed_base=seed_base,
    )

    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")
    has_phases = cache["phases"].size == n_samples

    all_metrics: Dict[str, Any] = {}

    if is_synthetic:
        print("Computing t-SNE visualization...")
        save_latent_tsne(
            cache["inv_latents"],
            cache["phases"],
            out_dir,
            max_samples=tsne_max_samples,
            class_names=class_names,
        )

    print("Computing PCA analysis...")
    pca_stats = save_pca_visualization(
        cache["inv_latents"],
        cache["phases"],
        out_dir,
        max_samples=None,
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
        max_samples=clustering_max_samples,
        class_names=class_names,
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

        cluster_labels = compute_kmeans_labels(cache["inv_latents"], best_k)

        orientation_outputs = None
        symmetry_outputs = None
        angles_deg = None
        orientation_grain_labels = None
        orientation_method = None
        orientation_sync_outputs = None

        if (
            cache["eq_latents"].size
            and cache["eq_latents"].shape[0] == len(cache["inv_latents"])
            and cluster_labels.size == len(cache["inv_latents"])
        ):
            print("Computing orientation analysis (rotation-head alignments)...")
            orient_iters = int(getattr(cfg, "analysis_orientation_iters", 5))
            orient_min_cluster = int(getattr(cfg, "analysis_orientation_min_cluster", 3))
            orient_channel_weighting = None
            orient_head_batch_size = int(
                getattr(cfg, "analysis_orientation_head_batch_size", 4096)
            )
            sym_num_probes = int(getattr(cfg, "analysis_symmetry_num_probes", 64))
            sym_beta = float(getattr(cfg, "analysis_symmetry_beta", 10.0))
            sym_delta = float(getattr(cfg, "analysis_symmetry_delta", 0.05))
            sym_mode_thresh = float(getattr(cfg, "analysis_symmetry_mode_threshold_deg", 15.0))
            sym_max_samples = getattr(cfg, "analysis_symmetry_max_samples", None)
            if sym_max_samples is not None:
                sym_max_samples = int(sym_max_samples)
            grain_thresh = float(
                getattr(cfg, "analysis_orientation_grain_threshold_deg", 15.0)
            )
            grain_max = int(getattr(cfg, "analysis_orientation_grain_max", 20))
            orient_ref_mode = (
                orientation_reference_override
                if orientation_reference_override is not None
                else str(getattr(cfg, "analysis_orientation_reference", "cluster")).strip().lower()
            )
            if orient_ref_mode not in {"cluster", "global"}:
                print(
                    f"Unknown analysis_orientation_reference='{orient_ref_mode}', "
                    "falling back to 'cluster'."
                )
                orient_ref_mode = "cluster"
            if orient_ref_mode == "global":
                orient_ref_labels = np.zeros_like(cluster_labels)
                print("Orientation reference: single global prototype.")
            else:
                orient_ref_labels = cluster_labels
                print("Orientation reference: per-cluster prototypes.")

            if getattr(model, "pose_head", None) is not None:
                try:
                    orientation_outputs = (
                        compute_cluster_prototypes_and_alignments_with_rotation_head(
                            model,
                            cache["eq_latents"],
                            orient_ref_labels,
                            n_iters=orient_iters,
                            min_cluster_size=orient_min_cluster,
                            normalize_channels=False,
                            batch_size=orient_head_batch_size,
                        )
                    )
                    orientation_method = "rotation_head"
                except ValueError as exc:
                    print(
                        f"Rotation-head alignment failed ({exc}); "
                        "falling back to legacy Kabsch alignment."
                    )
                    orient_channel_weighting = str(
                        getattr(cfg, "analysis_orientation_channel_weighting", "norm")
                    )
                    orientation_outputs = compute_cluster_prototypes_and_alignments(
                        cache["eq_latents"],
                        orient_ref_labels,
                        n_iters=orient_iters,
                        min_cluster_size=orient_min_cluster,
                        normalize_channels=False,
                        channel_weighting=orient_channel_weighting,
                    )
                    orientation_method = "kabsch_fallback"
            else:
                print("Rotation head not found in checkpoint; using legacy Kabsch alignment.")
                orient_channel_weighting = str(
                    getattr(cfg, "analysis_orientation_channel_weighting", "norm")
                )
                orientation_outputs = compute_cluster_prototypes_and_alignments(
                    cache["eq_latents"],
                    orient_ref_labels,
                    n_iters=orient_iters,
                    min_cluster_size=orient_min_cluster,
                    normalize_channels=False,
                    channel_weighting=orient_channel_weighting,
                )
                orientation_method = "kabsch_fallback"
            symmetry_outputs = compute_self_symmetry_metrics(
                cache["eq_latents"],
                num_probes=sym_num_probes,
                beta=sym_beta,
                delta=sym_delta,
                mode_threshold_deg=sym_mode_thresh,
                max_samples=sym_max_samples,
            )

            angles_deg = _rotation_angles_deg(orientation_outputs["R_align"])
            orientation_grain_labels = cluster_orientations_by_cluster(
                orientation_outputs["R_align"],
                orient_ref_labels,
                threshold_deg=grain_thresh,
            )
            orientation_grain_labels = cap_cluster_labels(
                orientation_grain_labels,
                max_clusters=grain_max,
                other_label=-1,
            )

            orientation_npz = out_dir / "orientation_analysis.npz"
            np.savez_compressed(
                orientation_npz,
                cluster_labels=cluster_labels,
                R_align=orientation_outputs["R_align"],
                alignment_residuals=orientation_outputs["residuals"],
                proto_eq=orientation_outputs["proto_eq"],
                alignment_angles_deg=angles_deg,
                orientation_grain_labels=orientation_grain_labels,
                orientation_reference_labels=orient_ref_labels,
                symmetry_e_min=symmetry_outputs["e_min"],
                symmetry_e_p10=symmetry_outputs["e_p10"],
                symmetry_e_median=symmetry_outputs["e_median"],
                symmetry_entropy=symmetry_outputs["entropy"],
                symmetry_n_eff=symmetry_outputs["n_eff"],
                symmetry_n_modes=symmetry_outputs["n_modes"],
                symmetry_mode_width_deg=symmetry_outputs["mode_width_deg"],
                symmetry_computed_mask=symmetry_outputs["computed_mask"],
            )

            orientation_csv = out_dir / "orientation_analysis.csv"
            sample_idx = np.arange(len(cluster_labels))
            data = np.column_stack(
                [
                    sample_idx,
                    cluster_labels,
                    orientation_grain_labels,
                    angles_deg,
                    orientation_outputs["residuals"],
                    symmetry_outputs["e_min"],
                    symmetry_outputs["e_p10"],
                    symmetry_outputs["e_median"],
                    symmetry_outputs["entropy"],
                    symmetry_outputs["n_eff"],
                    symmetry_outputs["n_modes"],
                    symmetry_outputs["mode_width_deg"],
                    symmetry_outputs["computed_mask"].astype(int),
                ]
            )
            header = (
                "sample_idx,cluster,grain_label,align_angle_deg,align_residual,"
                "sym_e_min,sym_e_p10,sym_e_median,sym_entropy,sym_n_eff,"
                "sym_n_modes,sym_mode_width_deg,sym_computed"
            )
            np.savetxt(
                orientation_csv,
                data,
                delimiter=",",
                header=header,
                comments="",
                fmt=[
                    "%d",
                    "%d",
                    "%d",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%.6f",
                    "%d",
                ],
            )

            cluster_summary: Dict[int, Dict[str, float]] = {}
            unique_labels = np.unique(cluster_labels)
            for label in unique_labels:
                mask = cluster_labels == label
                summary = {"n_samples": int(mask.sum())}
                if angles_deg is not None:
                    vals = angles_deg[mask]
                    vals = vals[np.isfinite(vals)]
                    summary["median_align_angle_deg"] = (
                        float(np.median(vals)) if vals.size else float("nan")
                    )
                res_vals = orientation_outputs["residuals"][mask]
                res_vals = res_vals[np.isfinite(res_vals)]
                summary["median_align_residual"] = (
                    float(np.median(res_vals)) if res_vals.size else float("nan")
                )
                sym_eff = symmetry_outputs["n_eff"][mask]
                sym_eff = sym_eff[np.isfinite(sym_eff)]
                summary["median_sym_n_eff"] = (
                    float(np.median(sym_eff)) if sym_eff.size else float("nan")
                )
                sym_modes = symmetry_outputs["n_modes"][mask]
                sym_modes = sym_modes[np.isfinite(sym_modes)]
                summary["median_sym_n_modes"] = (
                    float(np.median(sym_modes)) if sym_modes.size else float("nan")
                )
                sym_width = symmetry_outputs["mode_width_deg"][mask]
                sym_width = sym_width[np.isfinite(sym_width)]
                summary["median_sym_mode_width_deg"] = (
                    float(np.median(sym_width)) if sym_width.size else float("nan")
                )
                cluster_summary[int(label)] = summary

            cluster_summary_csv = out_dir / "orientation_cluster_medians.csv"
            if cluster_summary:
                rows = []
                for label, summary in cluster_summary.items():
                    rows.append(
                        [
                            label,
                            summary.get("n_samples", float("nan")),
                            summary.get("median_align_angle_deg", float("nan")),
                            summary.get("median_align_residual", float("nan")),
                            summary.get("median_sym_n_eff", float("nan")),
                            summary.get("median_sym_n_modes", float("nan")),
                            summary.get("median_sym_mode_width_deg", float("nan")),
                        ]
                    )
                rows = np.asarray(rows, dtype=np.float64)
                np.savetxt(
                    cluster_summary_csv,
                    rows,
                    delimiter=",",
                    header=(
                        "cluster,n_samples,median_align_angle_deg,median_align_residual,"
                        "median_sym_n_eff,median_sym_n_modes,median_sym_mode_width_deg"
                    ),
                    comments="",
                    fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
                )

            all_metrics["orientation"] = {
                "num_samples": int(len(cluster_labels)),
                "num_clusters": int(len(unique_labels)),
                "alignment_angle_median_deg": float(np.nanmedian(angles_deg))
                if angles_deg is not None and angles_deg.size
                else float("nan"),
                "alignment_residual_median": float(np.nanmedian(orientation_outputs["residuals"]))
                if orientation_outputs["residuals"].size
                else float("nan"),
                "symmetry_n_eff_median": float(np.nanmedian(symmetry_outputs["n_eff"]))
                if symmetry_outputs["n_eff"].size
                else float("nan"),
                "symmetry_n_modes_median": float(np.nanmedian(symmetry_outputs["n_modes"]))
                if symmetry_outputs["n_modes"].size
                else float("nan"),
                "symmetry_mode_width_median_deg": float(
                    np.nanmedian(symmetry_outputs["mode_width_deg"])
                )
                if symmetry_outputs["mode_width_deg"].size
                else float("nan"),
                "symmetry_num_probes": int(sym_num_probes),
                "symmetry_beta": float(sym_beta),
                "symmetry_delta": float(sym_delta),
                "symmetry_mode_threshold_deg": float(sym_mode_thresh),
                "orientation_grain_threshold_deg": float(grain_thresh),
                "orientation_grain_max": int(grain_max),
                "orientation_method": orientation_method,
                "orientation_reference": orient_ref_mode,
                "orientation_reference_num_prototypes": int(len(np.unique(orient_ref_labels))),
                "symmetry_samples_computed": int(symmetry_outputs["computed_mask"].sum()),
                "cluster_medians": cluster_summary,
                "outputs": {
                    "npz": str(orientation_npz),
                    "csv": str(orientation_csv),
                    "cluster_csv": str(cluster_summary_csv),
                },
            }
            if orientation_method == "rotation_head":
                all_metrics["orientation"]["orientation_head_batch_size"] = int(
                    orient_head_batch_size
                )
            if orient_channel_weighting is not None:
                all_metrics["orientation"]["orientation_channel_weighting"] = (
                    orient_channel_weighting
                )

            save_cluster_orientation_histograms(
                angles_deg,
                cluster_labels,
                out_dir,
            )
            save_cluster_symmetry_boxplots(
                symmetry_outputs["n_eff"],
                symmetry_outputs["n_modes"],
                cluster_labels,
                out_dir,
            )

        if (
            bool(getattr(cfg, "analysis_orientation_sync_enabled", True))
            and cache["eq_latents"].size
            and cache["eq_latents"].shape[0] == len(cache["inv_latents"])
            and coords.size
            and len(coords) == len(cache["inv_latents"])
            and getattr(model, "pose_head", None) is not None
        ):
            print("Computing pairwise orientation synchronization analysis...")
            sync_knn_k = int(getattr(cfg, "analysis_orientation_sync_knn_k", 12))
            sync_max_nodes = getattr(cfg, "analysis_orientation_sync_max_nodes", 6000)
            sync_max_nodes = None if sync_max_nodes is None else int(sync_max_nodes)
            sync_iters = int(getattr(cfg, "analysis_orientation_sync_iters", 30))
            sync_tol_deg = float(getattr(cfg, "analysis_orientation_sync_tol_deg", 1e-3))
            sync_smooth_k = int(getattr(cfg, "analysis_orientation_sync_smooth_k", 6))
            sync_max_triplets = int(
                getattr(cfg, "analysis_orientation_sync_max_triplets", 20000)
            )
            sync_pair_batch = int(
                getattr(cfg, "analysis_orientation_sync_pair_batch_size", 4096)
            )
            sync_seed = int(getattr(cfg, "analysis_orientation_sync_seed", 0))

            try:
                orientation_sync_outputs = evaluate_pairwise_orientation_synchronization(
                    model,
                    cache["eq_latents"],
                    coords,
                    knn_k=sync_knn_k,
                    max_nodes=sync_max_nodes,
                    sync_iters=sync_iters,
                    sync_tol_deg=sync_tol_deg,
                    smooth_k=sync_smooth_k,
                    max_triplets=sync_max_triplets,
                    pair_batch_size=sync_pair_batch,
                    seed=sync_seed,
                )

                sync_npz = out_dir / "orientation_sync_analysis.npz"
                np.savez_compressed(
                    sync_npz,
                    node_indices=orientation_sync_outputs["node_indices"],
                    edges=orientation_sync_outputs["edges"],
                    pair_rotations=orientation_sync_outputs["pair_rotations"],
                    absolute_rotations=orientation_sync_outputs["absolute_rotations"],
                    edge_residual_deg=orientation_sync_outputs["edge_residual_deg"],
                    cycle_error_deg=orientation_sync_outputs["cycle_error_deg"],
                    spatial_smoothness_deg=orientation_sync_outputs["spatial_smoothness_deg"],
                    sync_update_deg=orientation_sync_outputs["sync_update_deg"],
                )

                sync_metrics_path = out_dir / "orientation_sync_metrics.json"
                with sync_metrics_path.open("w") as handle:
                    json.dump(orientation_sync_outputs["metrics"], handle, indent=2)

                all_metrics["orientation_sync"] = {
                    **orientation_sync_outputs["metrics"],
                    "outputs": {
                        "npz": str(sync_npz),
                        "json": str(sync_metrics_path),
                    },
                }
            except ValueError as exc:
                print(f"Skipping orientation synchronization analysis: {exc}")

        print("Computing t-SNE visualization (clusters)...")
        tsne_idx = _sample_indices(len(cache["inv_latents"]), tsne_max_samples)
        tsne_latents = cache["inv_latents"][tsne_idx]
        tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
        tsne_coords = compute_tsne(tsne_latents, perplexity=tsne_perplexity, n_iter=1500)

        save_tsne_plot_with_coords(
            tsne_coords,
            cluster_labels[tsne_idx],
            out_dir,
            out_name="latent_tsne_clusters.png",
            title=f"Latent space t-SNE (KMeans k={best_k})",
        )

        if symmetry_outputs is not None:
            save_tsne_continuous_plot(
                tsne_coords,
                symmetry_outputs["n_eff"][tsne_idx],
                out_dir / "latent_tsne_symmetry_neff.png",
                title="Latent space t-SNE (colored by N_eff)",
            )

        k_candidates = [int(best_k)]
        if int(best_k) > 3:
            k_candidates.append(int(best_k) - 1)
        k_candidates.extend([int(best_k) + 1, int(best_k) + 2])
        unique_k: list[int] = []
        for k_val in k_candidates:
            k_val = max(2, min(k_val, len(cache["inv_latents"])))
            if k_val not in unique_k:
                unique_k.append(k_val)

        for k_val in unique_k:
            if k_val == int(best_k):
                continue
            labels_k = compute_kmeans_labels(cache["inv_latents"], k_val)
            save_tsne_plot_with_coords(
                tsne_coords,
                labels_k[tsne_idx],
                out_dir,
                out_name=f"latent_tsne_clusters_k{k_val}.png",
                title=f"Latent space t-SNE (KMeans k={k_val})",
            )

        if coords.size and cluster_labels.size:
            print("Saving local-structure coordinate assignments...")
            coord_files = save_local_structure_assignments(
                coords,
                cluster_labels,
                out_dir,
            )
            if coord_files:
                print("Saving MD space clustering plot...")
                save_md_space_clusters_plot(
                    coords,
                    cluster_labels,
                    out_dir / "md_space_clusters.png",
                    max_points=None,
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
                        max_points=None,
                        marker_size=3.0,
                        marker_line_width=0.0,
                    )
                    interactive_paths[int(best_k)] = str(interactive_path)

                    k_candidates = [int(best_k)]
                    if int(best_k) > 3:
                        k_candidates.append(int(best_k) - 1)
                    k_candidates.extend([int(best_k) + 1, int(best_k) + 2])

                    unique_k: list[int] = []
                    for k_val in k_candidates:
                        k_val = max(2, min(k_val, len(cache["inv_latents"])))
                        if k_val not in unique_k:
                            unique_k.append(k_val)

                    for k_val in unique_k:
                        if k_val == int(best_k):
                            continue
                        labels_k = compute_kmeans_labels(cache["inv_latents"], k_val)
                        out_path = out_dir / f"md_space_clusters_k{k_val}.html"
                        save_interactive_md_plot(
                            coords,
                            labels_k,
                            out_path,
                            palette="Set3",
                            max_points=None,
                            marker_size=3.0,
                            marker_line_width=0.0,
                        )
                        interactive_paths[int(k_val)] = str(out_path)
                except ImportError:
                    interactive_path = None
                    print("Plotly not installed; skipping interactive MD plot.")

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
                if (
                    orientation_outputs is not None
                    and len(coords) == len(cache["inv_latents"])
                ):
                    try:
                        if angles_deg is not None and len(angles_deg) == len(coords):
                            angle_path = out_dir / "md_space_orientation_angles.html"
                            save_interactive_md_continuous_plot(
                                coords,
                                angles_deg,
                                angle_path,
                                max_points=None,
                                marker_size=3.5,
                                colorscale="Turbo",
                                title="MD orientation (rotation-head alignment angle)",
                                value_label="align_deg",
                            )
                            all_metrics["real_md"]["orientation_angles_html"] = str(angle_path)

                        if symmetry_outputs is not None and len(symmetry_outputs["n_eff"]) == len(coords):
                            symmetry_path = out_dir / "md_space_orientation_symmetry.html"
                            save_interactive_md_continuous_plot(
                                coords,
                                symmetry_outputs["n_eff"],
                                symmetry_path,
                                max_points=None,
                                marker_size=3.5,
                                colorscale="Viridis",
                                title="MD orientation ambiguity (N_eff)",
                                value_label="N_eff",
                            )
                            all_metrics["real_md"]["orientation_symmetry_html"] = str(symmetry_path)

                        if (
                            symmetry_outputs is not None
                            and len(symmetry_outputs["n_eff"]) == len(coords)
                            and angles_deg is not None
                            and len(angles_deg) == len(coords)
                        ):
                            two_layer_path = out_dir / "md_space_orientation_two_layer.html"
                            save_interactive_md_two_layer_plot(
                                coords,
                                cluster_labels,
                                symmetry_outputs["n_eff"],
                                two_layer_path,
                                palette="Set3",
                                max_points=None,
                                base_marker_size=2.2,
                                base_opacity=0.3,
                                overlay_marker_size=4.0,
                                overlay_opacity=0.9,
                                overlay_sizes=angles_deg,
                                overlay_size_range=(3.0, 7.0),
                                colorscale="Viridis",
                                title="MD clusters + ambiguity (color) + head align angle (size)",
                                value_label="N_eff",
                            )
                            all_metrics["real_md"]["orientation_two_layer_html"] = str(
                                two_layer_path
                            )

                        if orientation_grain_labels is not None and len(orientation_grain_labels) == len(coords):
                            grain_path = out_dir / "md_space_orientation_grains.html"
                            save_interactive_md_plot(
                                coords,
                                orientation_grain_labels,
                                grain_path,
                                palette="Set3",
                                max_points=None,
                                marker_size=3.0,
                                marker_line_width=0.0,
                                title="MD orientation grains (rotation-head alignment)",
                                label_prefix="Grain",
                                hover_values=angles_deg,
                                hover_label="align_deg",
                            )
                            all_metrics["real_md"]["orientation_grains_html"] = str(grain_path)
                    except ImportError:
                        print("Plotly not installed; skipping orientation interactive plots.")
                if (
                    orientation_sync_outputs is not None
                    and len(orientation_sync_outputs["node_indices"]) > 0
                    and len(orientation_sync_outputs["edges"]) > 0
                    and len(orientation_sync_outputs["edge_residual_deg"])
                    == len(orientation_sync_outputs["edges"])
                ):
                    try:
                        node_idx = orientation_sync_outputs["node_indices"].astype(np.int64)
                        edges = orientation_sync_outputs["edges"].astype(np.int64)
                        edge_residual = orientation_sync_outputs["edge_residual_deg"].astype(
                            np.float32
                        )
                        node_median_edge_residual = np.full(
                            len(node_idx), np.nan, dtype=np.float32
                        )
                        per_node_values: list[list[float]] = [[] for _ in range(len(node_idx))]
                        for (i_node, j_node), val in zip(edges, edge_residual):
                            if np.isfinite(val):
                                per_node_values[int(i_node)].append(float(val))
                                per_node_values[int(j_node)].append(float(val))
                        for i_node, vals in enumerate(per_node_values):
                            if vals:
                                node_median_edge_residual[i_node] = float(np.median(vals))

                        sync_residual_path = (
                            out_dir / "md_space_orientation_sync_edge_residual.html"
                        )
                        save_interactive_md_continuous_plot(
                            coords[node_idx],
                            node_median_edge_residual,
                            sync_residual_path,
                            max_points=None,
                            marker_size=3.5,
                            colorscale="Turbo",
                            title="MD orientation sync quality (median edge residual)",
                            value_label="edge_residual_deg",
                        )
                        all_metrics["real_md"]["orientation_sync_edge_residual_html"] = str(
                            sync_residual_path
                        )
                    except ImportError:
                        print(
                            "Plotly not installed; skipping orientation synchronization interactive plots."
                        )

    print("Evaluating equivariance (encoder latents)...")
    eq_metrics, eq_err = evaluate_latent_equivariance(model, dl, device, max_batches=2)
    save_equivariance_plot(eq_err, out_dir / "equivariance.png")
    all_metrics["equivariance"] = eq_metrics

    if getattr(model, "pose_head", None) is not None:
        pose_eval_batches = (
            int(pose_eval_max_batches_override)
            if pose_eval_max_batches_override is not None
            else int(getattr(cfg, "analysis_pose_eval_max_batches", 2))
        )
        pose_eval_seed_base = (
            int(pose_eval_seed_base_override)
            if pose_eval_seed_base_override is not None
            else int(getattr(cfg, "analysis_pose_eval_seed_base", 7331))
        )
        print("Evaluating pose-head relative rotation (x vs R·x)...")
        pose_metrics, _ = evaluate_pose_head_relative_rotation(
            model,
            dl,
            device,
            max_batches=pose_eval_batches,
            seed_base=pose_eval_seed_base,
        )
        all_metrics["pose_head_relative"] = pose_metrics

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
    if "pose_head_relative" in all_metrics:
        pose_rel = all_metrics["pose_head_relative"]
        print(
            f"Pose-head relative angle (mean deg): {pose_rel.get('pose_head_rel_angle_mean_deg', 'N/A'):.4f}"
            if isinstance(pose_rel.get("pose_head_rel_angle_mean_deg"), float)
            else f"Pose-head relative angle (mean deg): {pose_rel.get('pose_head_rel_angle_mean_deg', 'N/A')}"
        )
        if "pose_head_rel_angle_sym_mean_deg" in pose_rel:
            print(
                f"Pose-head symmetry-min angle (mean deg): {pose_rel.get('pose_head_rel_angle_sym_mean_deg', 'N/A'):.4f}"
                if isinstance(pose_rel.get("pose_head_rel_angle_sym_mean_deg"), float)
                else f"Pose-head symmetry-min angle (mean deg): {pose_rel.get('pose_head_rel_angle_sym_mean_deg', 'N/A')}"
            )
    if "orientation_sync" in all_metrics:
        sync = all_metrics["orientation_sync"]
        print(
            f"Orientation sync edge residual (median deg): {sync.get('edge_residual_deg_median', 'N/A'):.4f}"
            if isinstance(sync.get("edge_residual_deg_median"), float)
            else f"Orientation sync edge residual (median deg): {sync.get('edge_residual_deg_median', 'N/A')}"
        )
        print(
            f"Orientation sync cycle error (median deg): {sync.get('cycle_error_deg_median', 'N/A'):.4f}"
            if isinstance(sync.get("cycle_error_deg_median"), float)
            else f"Orientation sync cycle error (median deg): {sync.get('cycle_error_deg_median', 'N/A')}"
        )
        print(
            f"Orientation sync spatial smoothness (median deg): {sync.get('spatial_smoothness_deg_median', 'N/A'):.4f}"
            if isinstance(sync.get("spatial_smoothness_deg_median"), float)
            else f"Orientation sync spatial smoothness (median deg): {sync.get('spatial_smoothness_deg_median', 'N/A')}"
        )

    print("=" * 60)
    print(f"\nSaved all analyses to {out_dir}")
    print("Generated files:")
    if has_phases:
        print("  - latent_tsne_ground_truth.png: t-SNE with ground truth labels")
    print("  - latent_tsne_clusters.png: t-SNE with KMeans clusters")
    print("  - latent_pca_analysis.png: PCA projection and variance")
    print("  - latent_pca_3d.png: 3D PCA projection")
    print("  - latent_statistics.png: Comprehensive latent statistics")
    print("  - clustering_analysis.png: Clustering quality metrics")
    print("  - equivariance.png: Equivariant latent error distribution")
    print("  - analysis_metrics.json: All numerical metrics")
    if "orientation" in all_metrics:
        print("  - orientation_analysis.npz: Alignment rotations + symmetry metrics")
        print("  - orientation_analysis.csv: Per-sample orientation + symmetry metrics")
        print("  - orientation_cluster_medians.csv: Cluster-wise orientation medians")
        print("  - cluster_orientation_histograms.png: Per-cluster alignment angles")
        print("  - cluster_symmetry_boxplots.png: Per-cluster symmetry summaries")
        print("  - latent_tsne_symmetry_neff.png: t-SNE colored by N_eff")
    if "orientation_sync" in all_metrics:
        print("  - orientation_sync_analysis.npz: Pairwise rotations + synchronized SO(3)")
        print("  - orientation_sync_metrics.json: Pairwise-to-absolute quality metrics")
    if not is_synthetic and "real_md" in all_metrics:
        print("  - local_structure_coords_clusters.csv: local-structure centers with cluster IDs")
        print("  - local_structure_coords_clusters.npz: local-structure centers + cluster IDs")
        print("  - md_space_clusters.png: 3D MD space clusters")
        print("  - md_space_clusters.html: interactive 3D MD space clusters")
        print("  - md_space_clusters_k*.html: interactive 3D MD plots for k±1,k+2")
        print("  - latent_tsne_clusters_k*.png: t-SNE plots for k±1,k+2")
        if "orientation_grains_html" in all_metrics.get("real_md", {}):
            print("  - md_space_orientation_grains.html: interactive orientation grains")
        if "orientation_angles_html" in all_metrics.get("real_md", {}):
            print("  - md_space_orientation_angles.html: interactive align-angle coloring")
        if "orientation_symmetry_html" in all_metrics.get("real_md", {}):
            print("  - md_space_orientation_symmetry.html: interactive N_eff coloring")
        if "orientation_two_layer_html" in all_metrics.get("real_md", {}):
            print("  - md_space_orientation_two_layer.html: clusters + ambiguity overlay")
        if "orientation_sync_edge_residual_html" in all_metrics.get("real_md", {}):
            print("  - md_space_orientation_sync_edge_residual.html: sync residual quality map")

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
        help="Max samples for t-SNE (default: 20000).",
    )
    parser.add_argument(
        "--use_train_data",
        action="store_true",
        help="Use training data instead of test data.",
    )
    parser.add_argument(
        "--data_file",
        action="append",
        default=None,
        help="Override real data files (repeat for multiple). Example: --data_file 175ps.off",
    )
    parser.add_argument(
        "--orientation_reference",
        type=str,
        choices=["cluster", "global"],
        default=None,
        help="Orientation reference prototype mode: per-cluster or single global average z_eq.",
    )
    parser.add_argument(
        "--pose_eval_max_batches",
        type=int,
        default=None,
        help="Max batches for pose-head relative-rotation diagnostics.",
    )
    parser.add_argument(
        "--pose_eval_seed_base",
        type=int,
        default=None,
        help="Seed base for pose-head relative-rotation diagnostics.",
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
        use_train_data=bool(args.use_train_data),
        data_files_override=args.data_file,
        analysis_orientation_reference=args.orientation_reference,
        analysis_pose_eval_max_batches=args.pose_eval_max_batches,
        analysis_pose_eval_seed_base=args.pose_eval_seed_base,
    )


if __name__ == "__main__":
    main()
