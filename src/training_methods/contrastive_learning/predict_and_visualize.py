import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.data_utils.data_module import RealPointCloudDataModule, SyntheticPointCloudDataModule
from src.training_methods.contrastive_learning.analysis_utils import (
    _sample_indices,
    build_real_coords_dataloader,
    cap_cluster_labels,
    evaluate_latent_equivariance,
    gather_inference_batches,
    segment_grains_with_pose_head,
)
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.training_methods.contrastive_learning.supervised_cache import (
    _infer_test_cluster_eval_k,
)
from src.utils.spd_metrics import (
    compute_cluster_metrics,
    compute_embedding_quality_metrics,
)
from src.utils.spd_utils import apply_rotation
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


def _as_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in list(value)]


def _as_list_of_int(value: Any, *, field_name: str = "value") -> list[int] | None:
    if value is None:
        return None
    values = list(value)
    if not values:
        raise ValueError(f"Invalid {field_name}: expected a non-empty list")
    return [int(v) for v in values]


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _extract_class_names(dataset: Any) -> Dict[int, str] | None:
    dataset = _unwrap_dataset(dataset)
    class_names_raw = getattr(dataset, "class_names", None)
    if class_names_raw is None:
        return None
    class_names = dict(class_names_raw)
    return {int(k): str(v) for k, v in class_names.items()}


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _as_int_mapping(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = dict(value)
    if not hasattr(value, "items"):
        return {}

    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            runs = int(raw)
        except Exception:
            continue
        if runs > 0:
            out[str(key)] = runs
    return out


def _extract_pc_and_class_id(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        if "points" not in batch:
            raise KeyError("Batch dict is missing required key 'points'")
        return batch["points"], batch.get("class_id")

    if torch.is_tensor(batch):
        return batch, None

    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        points = batch[0]
        class_id = None
        if len(batch) > 1:
            candidate = batch[1]
            if torch.is_tensor(candidate):
                if candidate.dtype in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                    torch.long,
                ) and candidate.dim() <= 1:
                    class_id = candidate
            elif isinstance(candidate, (int, np.integer)):
                class_id = torch.as_tensor(candidate, dtype=torch.long)
        return points, class_id

    raise TypeError(f"Unsupported batch type for extraction: {type(batch)!r}")


def _random_rotation_matrices(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    rand = torch.randn(
        (batch_size, 3, 3),
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    q, r = torch.linalg.qr(rand)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d.unsqueeze(-2)
    det = torch.det(q)
    neg_mask = det < 0
    if torch.any(neg_mask):
        q[neg_mask, :, 0] *= -1.0
    return q.to(device=device, dtype=dtype)


@contextmanager
def _temporary_disable_dataset_aug(dataloader: torch.utils.data.DataLoader):
    changes: list[tuple[Any, str, float]] = []
    ds = getattr(dataloader, "dataset", None)
    while ds is not None:
        for attr in ("rotation_scale", "noise_scale", "jitter_scale", "scaling_range"):
            if hasattr(ds, attr):
                prev = float(getattr(ds, attr))
                changes.append((ds, attr, prev))
                if prev != 0.0:
                    setattr(ds, attr, 0.0)
        ds = getattr(ds, "dataset", None)
    try:
        yield
    finally:
        for target, attr, prev in changes:
            setattr(target, attr, float(prev))


def _collect_test_latents_labels(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    *,
    max_samples_total: int | None = None,
    apply_random_rotations: bool = False,
    rotation_seed_base: int | None = None,
    progress_every_batches: int = 25,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    labels_available = True
    collected = 0
    max_samples = None if max_samples_total is None else max(1, int(max_samples_total))
    every = max(1, int(progress_every_batches))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            pc, class_id = _extract_pc_and_class_id(batch)
            if not torch.is_tensor(pc):
                pc = torch.as_tensor(pc)
            pc = pc.to(device=device, dtype=model.dtype, non_blocking=True)
            if apply_random_rotations:
                seed = None
                if rotation_seed_base is not None:
                    seed = int(rotation_seed_base) + int(batch_idx)
                rots = _random_rotation_matrices(
                    int(pc.shape[0]),
                    pc.device,
                    pc.dtype,
                    seed=seed,
                )
                pc = apply_rotation(pc, rots)
            if hasattr(model, "_prepare_model_input"):
                pc = model._prepare_model_input(pc)

            z, _, _ = model(pc)
            z_cpu = z.detach().to(dtype=torch.float32).cpu()
            batch_size = int(z_cpu.shape[0])
            if class_id is None:
                labels_available = False
                class_id_cpu = torch.zeros((batch_size,), dtype=torch.long)
            else:
                if not torch.is_tensor(class_id):
                    class_id = torch.as_tensor(class_id)
                class_id = class_id.detach().view(-1)
                if class_id.numel() == 1 and batch_size > 1:
                    class_id = class_id.expand(batch_size)
                class_id_cpu = class_id.to(dtype=torch.long).cpu()

            take = min(int(z_cpu.shape[0]), int(class_id_cpu.shape[0]))
            if max_samples is not None:
                take = min(take, max_samples - collected)
            if take <= 0:
                break

            latents.append(z_cpu[:take])
            labels.append(class_id_cpu[:take])

            collected += int(take)
            if verbose and (batch_idx + 1) % every == 0:
                print(
                    "[analysis][test] "
                    f"batch={batch_idx + 1} samples={collected}"
                    + (f"/{max_samples}" if max_samples is not None else "")
                )
            if max_samples is not None and collected >= max_samples:
                break

    lat_arr = torch.cat(latents, dim=0).numpy() if latents else np.empty((0,), dtype=np.float32)
    if labels_available:
        label_arr = torch.cat(labels, dim=0).numpy() if labels else np.empty((0,), dtype=np.int64)
    else:
        label_arr = np.zeros((int(lat_arr.shape[0]),), dtype=np.int64)
    return lat_arr, label_arr, labels_available


def _resolve_test_metric_settings(cfg: DictConfig) -> Dict[str, Any]:
    test_cluster_eval_k = int(getattr(cfg, "test_cluster_eval_k", _infer_test_cluster_eval_k(cfg)))
    test_cluster_acc_methods = _as_list_of_str(
        getattr(cfg, "test_cluster_acc_methods", ["kmeans", "kmeans++", "gmm_full"])
    ) or ["kmeans", "kmeans++", "gmm_full"]
    test_cluster_acc_runs = int(getattr(cfg, "test_cluster_acc_runs", 1))
    test_cluster_acc_runs_by_method = _as_int_mapping(getattr(cfg, "test_cluster_acc_runs_by_method", {}))
    cluster_acc_seed = int(getattr(cfg, "cluster_acc_seed", 0))

    test_max_samples = _positive_int_or_none(getattr(cfg, "analysis_test_max_samples", None))
    if test_max_samples is None:
        test_max_samples = _positive_int_or_none(getattr(cfg, "max_test_samples", None))

    rotation_runs = int(getattr(cfg, "analysis_test_rotation_runs", 5))
    rotation_seed = int(getattr(cfg, "analysis_test_rotation_seed", 12345))

    return {
        "hungarian_eval_k": test_cluster_eval_k,
        "acc_eval_methods": test_cluster_acc_methods,
        "acc_eval_runs": test_cluster_acc_runs,
        "acc_eval_runs_by_method": test_cluster_acc_runs_by_method,
        "acc_random_seed": cluster_acc_seed,
        "max_samples": test_max_samples,
        "rotation_runs": rotation_runs,
        "rotation_seed": rotation_seed,
    }


def _to_finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    return val if np.isfinite(val) else None


def _primary_accuracy_key(cluster_metrics: dict[str, Any], eval_k: int | None) -> str | None:
    preferred = f"ACC_KMEANS_HUNGARIAN_K{eval_k}"
    if preferred in cluster_metrics:
        return preferred
    for key in sorted(cluster_metrics.keys()):
        if (
            key.startswith("ACC_")
            and not key.endswith("_MEAN")
            and not key.endswith("_STD")
            and not key.endswith("_BEST")
            and not key.endswith("_RUNS")
        ):
            return key
    return None


def _aggregate_metric(values: list[float | None]) -> dict[str, Any]:
    valid = [float(v) for v in values if _to_finite_float(v) is not None]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None, "n_valid": 0}
    arr = np.asarray(valid, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_valid": int(arr.size),
    }


def _evaluate_test_run(
    model: BarlowTwinsModule,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    settings: dict[str, Any],
    *,
    progress_every_batches: int,
    apply_random_rotations: bool,
    rotation_seed_base: int | None = None,
    include_embedding_metrics: bool = False,
) -> dict[str, Any]:
    latents, labels, labels_available = _collect_test_latents_labels(
        model,
        dataloader,
        device,
        max_samples_total=settings["max_samples"],
        apply_random_rotations=apply_random_rotations,
        rotation_seed_base=rotation_seed_base,
        progress_every_batches=progress_every_batches,
        verbose=True,
    )
    cluster_metrics = {}
    if labels_available:
        cluster_metrics = (
            compute_cluster_metrics(
                latents,
                labels,
                stage="test",
                hungarian_eval_k=settings["hungarian_eval_k"],
                acc_eval_methods=settings["acc_eval_methods"],
                acc_eval_runs=settings["acc_eval_runs"],
                acc_eval_runs_by_method=settings["acc_eval_runs_by_method"],
                acc_random_seed=settings["acc_random_seed"],
            )
            or {}
        )
    embedding_metrics = {}
    if include_embedding_metrics:
        embedding_labels = labels if labels_available else np.zeros((int(latents.shape[0]),), dtype=np.int64)
        embedding_metrics = (
            compute_embedding_quality_metrics(
                latents,
                embedding_labels,
                include_expensive=True,
            )
            or {}
        )

    acc_key = _primary_accuracy_key(cluster_metrics, settings["hungarian_eval_k"])
    result = {
        "num_samples": int(latents.shape[0]),
        "labels_available": bool(labels_available),
        "accuracy": _to_finite_float(cluster_metrics.get(acc_key)),
        "nmi": _to_finite_float(cluster_metrics.get("NMI")),
        "ari": _to_finite_float(cluster_metrics.get("ARI")),
        "cluster_metrics": cluster_metrics,
    }
    if not labels_available:
        result["label_metrics_skipped_reason"] = "Batch does not provide class_id labels."
    result["accuracy_key"] = acc_key
    if embedding_metrics:
        result["embedding_metrics"] = embedding_metrics
    return result


def _run_test_phase_metrics(
    model: BarlowTwinsModule,
    dm: Any,
    cfg: DictConfig,
    device: str,
    *,
    progress_every_batches: int = 25,
) -> Dict[str, Any]:
    settings = _resolve_test_metric_settings(cfg)
    test_dl = dm.test_dataloader()
    with _temporary_disable_dataset_aug(test_dl):
        canonical_result = _evaluate_test_run(
            model,
            test_dl,
            device,
            settings,
            progress_every_batches=progress_every_batches,
            apply_random_rotations=False,
            include_embedding_metrics=True,
        )
        multi_rotation_runs = []
        for run_idx in range(settings["rotation_runs"]):
            run_result = _evaluate_test_run(
                model,
                test_dl,
                device,
                settings,
                progress_every_batches=progress_every_batches,
                apply_random_rotations=True,
                rotation_seed_base=settings["rotation_seed"] + run_idx * 10000,
            )
            run_result["rotation_run"] = int(run_idx + 1)
            multi_rotation_runs.append(run_result)

    return {
        "settings": settings,
        "canonical": canonical_result,
        "multi_rotation": {
            "num_rotations": int(settings["rotation_runs"]),
            "accuracy": _aggregate_metric([run.get("accuracy") for run in multi_rotation_runs]),
            "nmi": _aggregate_metric([run.get("nmi") for run in multi_rotation_runs]),
            "ari": _aggregate_metric([run.get("ari") for run in multi_rotation_runs]),
            "runs": multi_rotation_runs,
        },
    }


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
    test_rotation_runs: int | None = None,
    test_max_samples: int | None = None,
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

    if test_rotation_runs is not None:
        cfg.analysis_test_rotation_runs = int(test_rotation_runs)
    if test_max_samples is not None:
        cfg.analysis_test_max_samples = int(test_max_samples)

    def _resolve_analysis_files() -> list[str] | None:
        if cfg.data.kind != "real":
            return None
        if data_files_override:
            return data_files_override
        files = _as_list_of_str(cfg.data.analysis_data_files)
        if files:
            return files
        if cfg.data.analysis_data_file:
            return [cfg.data.analysis_data_file]
        data_files = _as_list_of_str(cfg.data.data_files)
        analysis_single = bool(cfg.data.analysis_single_timestep)
        if not analysis_single:
            return data_files
        mid_idx = len(data_files) // 2
        return [data_files[mid_idx]]

    analysis_files = _resolve_analysis_files()
    if analysis_files is not None:
        cfg.data.data_files = analysis_files
        print(f"Analysis data_files: {analysis_files}")

    # Prefer single-process data loading for analysis to avoid semaphore/shm issues
    # in restricted runtime environments. Can be overridden via config.
    analysis_num_workers = max(0, int(getattr(cfg, "analysis_num_workers", 0)))
    cfg.num_workers = int(analysis_num_workers)
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
    cluster_k_values_cfg = _as_list_of_int(
        getattr(cfg, "analysis_cluster_k_values", None),
        field_name="analysis_cluster_k_values",
    )
    cluster_k_values = cluster_k_values_cfg or [3, 4, 5, 6]
    cluster_k_values = [int(k) for k in cluster_k_values if int(k) >= 2]
    cluster_k_values = list(dict.fromkeys(cluster_k_values))
    if not cluster_k_values:
        cluster_k_values = [3, 4, 5, 6]
    md_use_all_points = bool(getattr(cfg, "analysis_md_use_all_points", True))
    hdbscan_enabled = bool(getattr(cfg, "analysis_hdbscan_enabled", True))
    hdbscan_fit_fraction = float(getattr(cfg, "analysis_hdbscan_fit_fraction", 0.25))
    hdbscan_max_fit_samples = int(getattr(cfg, "analysis_hdbscan_max_fit_samples", 50000))
    hdbscan_target_k_min = int(getattr(cfg, "analysis_hdbscan_target_k_min", 5))
    hdbscan_target_k_max = int(getattr(cfg, "analysis_hdbscan_target_k_max", 6))
    hdbscan_min_samples = cfg.analysis_hdbscan_min_samples
    if hdbscan_min_samples is not None:
        hdbscan_min_samples = int(hdbscan_min_samples)
    hdbscan_cluster_selection_epsilon = float(
        getattr(cfg, "analysis_hdbscan_cluster_selection_epsilon", 0.0)
    )
    hdbscan_cluster_selection_method = str(
        getattr(cfg, "analysis_hdbscan_cluster_selection_method", "leaf")
    ).lower()
    hdbscan_min_cluster_size_candidates = (
        [int(v) for v in cfg.analysis_hdbscan_min_cluster_size_candidates]
        if cfg.analysis_hdbscan_min_cluster_size_candidates is not None
        else None
    )
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
    grain_interactive_max_points = _positive_int_or_none(
        getattr(cfg, "analysis_grain_interactive_max_points", 120000)
    )
    grain_interactive_max_grains = max(0, int(getattr(cfg, "analysis_grain_interactive_max_grains", 80)))
    grain_tsne_max_grains = max(0, int(getattr(cfg, "analysis_grain_tsne_max_grains", 60)))
    progress_every_batches = int(getattr(cfg, "analysis_progress_every_batches", 25))
    print(f"t-SNE sample cap: {tsne_max_samples}")
    print(f"Clustering metrics cap: {clustering_max_samples}")
    print(f"Clustering k values (configured): {cluster_k_values}")
    _step("Building datamodule")
    dm = build_datamodule(cfg)
    is_synthetic = getattr(cfg.data, "kind", None) == "synthetic"

    dm.setup(stage="fit")
    all_metrics: Dict[str, Any] = {}

    _step("Running test-phase metrics")
    test_phase_metrics = _run_test_phase_metrics(
        model,
        dm,
        cfg,
        device,
        progress_every_batches=progress_every_batches,
    )
    all_metrics["test_phase"] = test_phase_metrics

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
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        dl = build_real_coords_dataloader(cfg, dm, use_train_data=True, use_full_dataset=True)
        print(
            "Real data detected: using full dataset for local-structure clustering visualization"
        )

    class_names = _extract_class_names(dm.train_dataset)
    print(f"Loaded class names: {class_names}")

    if max_batches_latent is None:
        max_batches_latent = _positive_int_or_none(getattr(cfg, "analysis_max_batches_latent", None))
    max_samples_total = getattr(cfg, "analysis_max_samples_total", None)
    if max_samples_total is None and not is_synthetic:
        max_samples_total = 20000
    if not is_synthetic and md_use_all_points:
        max_samples_total = None
    max_samples_total = _positive_int_or_none(max_samples_total)

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
        collect_coords=True,
        seed_base=seed_base,
        progress_every_batches=progress_every_batches,
        verbose=True,
    )

    n_samples = len(cache["inv_latents"])
    print(f"Collected {n_samples} samples for analysis")
    has_phases = cache["phases"].size == n_samples
    gt_instance_ids = cache.get("instance_ids", np.empty((0,), dtype=int))
    has_gt_grains = gt_instance_ids.size == n_samples and np.any(gt_instance_ids >= 0)

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
        k_values=cluster_k_values,
    )
    all_metrics["clustering"] = clustering_metrics

    coords = cache["coords"]

    md_metrics_key = "synthetic_md" if is_synthetic else "real_md"
    num_latents = len(cache["inv_latents"])
    configured_k_values = _as_list_of_int(
        clustering_metrics.get("k_values_evaluated", None),
        field_name="clustering.k_values_evaluated",
    ) or []
    if not configured_k_values:
        configured_k_values = [int(k) for k in cluster_k_values]
    configured_k_values = [max(2, min(int(k), num_latents)) for k in configured_k_values]
    configured_k_values = list(dict.fromkeys(configured_k_values))
    if not configured_k_values:
        configured_k_values = [max(2, min(num_latents, 3))]

    selected_method_by_k_cfg: Dict[int, str] = {
        int(key): str(value).lower()
        for key, value in clustering_metrics.get("selected_method_by_k", {}).items()
    }

    def _compute_labels_for_k(k_value: int, *, method_name: str):
        return compute_kmeans_labels(
            cache["inv_latents"],
            k_value,
            method=method_name,
            l2_normalize=cluster_l2_normalize,
            standardize=cluster_standardize,
            pca_variance=cluster_pca_var,
            pca_max_components=cluster_pca_max_components,
            return_info=True,
        )

    cluster_labels_by_k: Dict[int, np.ndarray] = {}
    cluster_methods_by_k: Dict[int, str] = {}
    for k_val in configured_k_values:
        method_for_k = selected_method_by_k_cfg.get(int(k_val), cluster_method)
        labels_k, fit_info_k = _compute_labels_for_k(
            int(k_val),
            method_name=method_for_k,
        )
        cluster_labels_by_k[int(k_val)] = labels_k
        cluster_methods_by_k[int(k_val)] = str(fit_info_k.get("method", method_for_k))

    primary_k = int(configured_k_values[0])
    cluster_labels = cluster_labels_by_k[primary_k]
    cluster_label_method = cluster_methods_by_k.get(primary_k, cluster_method)
    all_metrics["clustering"]["labels_k_method"] = cluster_label_method
    all_metrics["clustering"]["labels_method_by_k"] = {
        int(k): cluster_methods_by_k[int(k)] for k in configured_k_values
    }
    all_metrics["clustering"]["primary_k"] = int(primary_k)
    all_metrics["clustering"]["k_values_used"] = [int(k) for k in configured_k_values]

    _step("Computing t-SNE visualization (clusters)")
    tsne_n_iter = int(getattr(cfg, "analysis_tsne_n_iter", 1000))
    tsne_idx = _sample_indices(len(cache["inv_latents"]), tsne_max_samples)
    tsne_latents = cache["inv_latents"][tsne_idx]
    tsne_perplexity = min(50, max(5, len(tsne_latents) // 100))
    tsne_coords = compute_tsne(tsne_latents, perplexity=tsne_perplexity, n_iter=tsne_n_iter)

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
        )

        if idx_k == 0:
            _step("Saving coordinate-space clustering outputs")
            interactive_max_points = _positive_int_or_none(
                getattr(cfg, "analysis_interactive_max_points", None)
            )
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
                    for idx_k, k_val in enumerate(configured_k_values):
                        labels_k = cluster_labels_by_k[int(k_val)]
                        out_path = (
                            out_dir / "md_space_clusters.html"
                            if idx_k == 0
                            else out_dir / f"md_space_clusters_k{k_val}.html"
                        )
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
                        if idx_k == 0:
                            interactive_path = out_path
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
                            gt_grain_labels=gt_instance_ids if has_gt_grains else None,
                        )
                        grain_labels = np.asarray(
                            grain_result.get("grain_labels", np.empty((0,), dtype=int)),
                            dtype=int,
                        )
                        grain_info = grain_result.get("metrics", None)
                        # Print GT comparison if available
                        if grain_info and "gt_ari" in grain_info:
                            print(
                                f"  Grain GT comparison: "
                                f"ARI={grain_info['gt_ari']:.4f}  "
                                f"NMI={grain_info['gt_nmi']:.4f}  "
                                f"mean_IoU={grain_info['gt_mean_iou']:.4f}  "
                                f"weighted_IoU={grain_info['gt_weighted_iou']:.4f}  "
                                f"(pred={grain_info['num_grains']} grains, "
                                f"gt={grain_info['gt_num_grains']} grains)"
                            )
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

                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                all_metrics[md_metrics_key] = {
                    "primary_k": int(primary_k),
                    "k_values_used": [int(k) for k in configured_k_values],
                    "n_clusters": int(len(unique_labels)),
                    "cluster_counts": {int(k): int(v) for k, v in zip(unique_labels, counts)},
                    "coords_files": coord_files,
                }
                if interactive_path is not None:
                    all_metrics[md_metrics_key]["interactive_html"] = str(interactive_path)
                if interactive_paths:
                    all_metrics[md_metrics_key]["interactive_htmls"] = interactive_paths
                if hdbscan_info is not None:
                    all_metrics[md_metrics_key]["hdbscan"] = hdbscan_info
                    if hdbscan_coord_files:
                        all_metrics[md_metrics_key]["hdbscan_coords_files"] = hdbscan_coord_files
                    if hdbscan_path is not None:
                        all_metrics[md_metrics_key]["hdbscan_interactive_html"] = str(hdbscan_path)
                if grain_info is not None:
                    all_metrics[md_metrics_key]["grains"] = grain_info
                    all_metrics[md_metrics_key]["grains"]["interactive_max_points"] = (
                        None if grain_interactive_max_points is None else int(grain_interactive_max_points)
                    )
                    all_metrics[md_metrics_key]["grains"]["interactive_max_grains"] = int(
                        grain_interactive_max_grains
                    )
                    all_metrics[md_metrics_key]["grains"]["tsne_max_grains"] = int(
                        grain_tsne_max_grains
                    )
                    if grain_coord_files:
                        all_metrics[md_metrics_key]["grains_coords_files"] = grain_coord_files
                    if grain_path is not None:
                        all_metrics[md_metrics_key]["grains_interactive_html"] = str(grain_path)
                    if grain_tsne_path is not None:
                        all_metrics[md_metrics_key]["grains_tsne_png"] = grain_tsne_path
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

    if "test_phase" in all_metrics:
        test_phase = all_metrics["test_phase"]
        canonical = test_phase.get("canonical", {})
        multi_rotation = test_phase.get("multi_rotation", {})
        print(
            "Test canonical: "
            f"ACC={_fmt_metric(canonical.get('accuracy', 'N/A'))}, "
            f"NMI={_fmt_metric(canonical.get('nmi', 'N/A'))}, "
            f"ARI={_fmt_metric(canonical.get('ari', 'N/A'))}"
        )
        print(
            "Test multi-rotation (mean): "
            f"ACC={_fmt_metric(multi_rotation.get('accuracy', {}).get('mean', 'N/A'))}, "
            f"NMI={_fmt_metric(multi_rotation.get('nmi', {}).get('mean', 'N/A'))}, "
            f"ARI={_fmt_metric(multi_rotation.get('ari', {}).get('mean', 'N/A'))}"
        )
        print(f"Test multi-rotation runs: {multi_rotation.get('num_rotations', 'N/A')}")

    if "pca" in all_metrics and all_metrics["pca"]:
        print(f"PCA: {all_metrics['pca'].get('n_components_95_var', 'N/A')} components for 95% variance")

    if "clustering" in all_metrics and all_metrics["clustering"]:
        k_values_summary = all_metrics["clustering"].get(
            "k_values_used",
            all_metrics["clustering"].get("k_values_evaluated", "N/A"),
        )
        print(f"Cluster k values: {k_values_summary}")
        print(f"Primary k: {all_metrics['clustering'].get('primary_k', 'N/A')}")
        if "labels_k_method" in all_metrics["clustering"]:
            print(f"Primary k method: {all_metrics['clustering'].get('labels_k_method')}")
        if "ari_with_gt" in all_metrics["clustering"]:
            print(f"ARI with ground truth: {_fmt_metric(all_metrics['clustering']['ari_with_gt'])}")
            print(f"NMI with ground truth: {_fmt_metric(all_metrics['clustering']['nmi_with_gt'])}")

    if "equivariance" in all_metrics:
        eq = all_metrics["equivariance"]
        print(
            "Equivariant latent error (seeded mean): "
            f"{_fmt_metric(eq.get('eq_latent_rel_error_mean', 'N/A'))}"
        )
        print(
            "Equivariant latent error (seeded median): "
            f"{_fmt_metric(eq.get('eq_latent_rel_error_median', 'N/A'))}"
        )
        if "eq_latent_rel_error_unseeded" in eq:
            print(
                "Equivariant latent error (unseeded mean): "
                f"{_fmt_metric(eq.get('eq_latent_rel_error_unseeded', 'N/A'))}"
            )
        if "eq_latent_rel_error_unseeded_median" in eq:
            print(
                "Equivariant latent error (unseeded median): "
                f"{_fmt_metric(eq.get('eq_latent_rel_error_unseeded_median', 'N/A'))}"
            )
        if "eq_latent_nondeterminism_contribution" in eq:
            print(
                "Non-determinism contribution (unseeded - seeded): "
                f"{_fmt_metric(eq.get('eq_latent_nondeterminism_contribution', 'N/A'))}"
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
    if md_metrics_key in all_metrics:
        print("  - local_structure_coords_clusters.csv: local-structure centers with cluster IDs")
        print("  - local_structure_coords_clusters.npz: local-structure centers + cluster IDs")
        print("  - md_space_clusters.png: 3D MD space clusters")
        print("  - md_space_clusters.html: interactive 3D MD space clusters")
        if len(configured_k_values) > 1:
            print("  - md_space_clusters_k*.html: interactive 3D MD plots for configured k values")
            print("  - latent_tsne_clusters_k*.png: t-SNE plots for configured k values")
        if hdbscan_enabled:
            print("  - local_structure_hdbscan_coords_clusters.csv: MD centers with HDBSCAN labels")
            print("  - local_structure_hdbscan_coords_clusters.npz: MD centers + HDBSCAN labels")
            print("  - md_space_clusters_hdbscan.html: interactive 3D MD HDBSCAN clusters")
        if "grains_coords_files" in all_metrics[md_metrics_key]:
            print("  - local_structure_grains_coords_clusters.csv: MD centers with grain IDs")
            print("  - local_structure_grains_coords_clusters.npz: MD centers + grain IDs")
            print("  - md_space_grains.png: 3D MD grain segmentation")
            if "grains_tsne_png" in all_metrics[md_metrics_key]:
                print("  - latent_tsne_grains.png: t-SNE colored by grain IDs")
            if "grains_interactive_html" in all_metrics[md_metrics_key]:
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
    parser.add_argument(
        "--test_rotation_runs",
        type=int,
        default=None,
        help="Override number of random-rotation test runs (default: analysis_test_rotation_runs or 5).",
    )
    parser.add_argument(
        "--test_max_samples",
        type=int,
        default=None,
        help="Override max samples for test-phase metrics (default: analysis_test_max_samples or max_test_samples).",
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
        test_rotation_runs=args.test_rotation_runs,
        test_max_samples=args.test_max_samples,
    )


if __name__ == "__main__":
    main()
