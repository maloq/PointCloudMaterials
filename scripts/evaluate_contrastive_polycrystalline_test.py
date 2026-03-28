#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.append(os.getcwd())

from src.training_methods.contrastive_learning.eval_checkpoint import (
    _build_eval_trainer,
    _normalize_metrics,
    _print_metrics,
    build_datamodule,
    load_barlow_model,
)
from src.training_methods.contrastive_learning.supervised_cache import (
    _collect_rotated_split_supervised_features,
    _collect_split_supervised_features,
    _resolve_hungarian_eval_k,
)
from src.utils.evaluation_metrics import _hungarian_cluster_accuracy


torch.set_float32_matmul_precision("high")


DEFAULT_DATA_CONFIG = "configs/data/data_synth_polycrystalline_balanced_geometries.yaml"
DEFAULT_OUTPUT_DIR = "output/eval_contrastive_polycrystalline_balanced_geometries"
DEFAULT_CHECKPOINT_SPECS = [
    (
        "multi",
        "output/2026-03-11/02-16-07/VICREG_l512_N160_M80_RI_MAE_Invariant-epoch=49.ckpt",
    ),
    (
        "Aluminum",
        "output/2026-03-10/19-24-39/VICREG_l512_N160_M80_RI_MAE_Invariant-epoch=59.ckpt",
    ),
]


def _status_print(message: str) -> None:
    print(message, flush=True)


def _resolve_file(path: str, *, what: str) -> Path:
    resolved = Path(os.path.expanduser(path))
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{what} not found: {resolved}")
    return resolved


def _resolve_dir(path: str) -> Path:
    resolved = Path(os.path.expanduser(path))
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def _parse_checkpoint_spec(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --checkpoint value {raw!r}. Expected LABEL=PATH."
        )
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Checkpoint label is empty in {raw!r}.")
    if not path:
        raise ValueError(f"Checkpoint path is empty in {raw!r}.")
    return label, path


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    slug = slug.strip("._-")
    if not slug:
        raise ValueError(f"Could not derive output folder name from label {label!r}.")
    return slug


def _load_checkpoint_cfg(checkpoint_path: Path) -> DictConfig:
    cfg_path = checkpoint_path.parent / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint config not found: {cfg_path}. Expected Hydra output next to checkpoint."
        )
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Expected DictConfig in {cfg_path}, got {type(cfg)!r}."
        )
    return cfg


def _load_data_cfg(data_config_path: Path) -> DictConfig:
    data_cfg = OmegaConf.load(data_config_path)
    if not isinstance(data_cfg, DictConfig):
        raise TypeError(
            f"Expected DictConfig in {data_config_path}, got {type(data_cfg)!r}."
        )
    return data_cfg


def _clone_resolved_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _build_eval_cfg(
    checkpoint_cfg: DictConfig,
    data_cfg: DictConfig,
    *,
    split_seed: int,
    batch_size_override: int | None,
    num_workers_override: int | None,
    test_so3_rotation_runs: int,
    kmeans_seed_base: int,
) -> DictConfig:
    cfg = _clone_resolved_cfg(checkpoint_cfg)
    cfg.data = _clone_resolved_cfg(data_cfg)

    cfg.data.split_seed = int(split_seed)
    cfg.max_samples = 0
    cfg.max_supervised_samples = None
    cfg.max_test_samples = None
    cfg.enable_supervised_metrics = True
    cfg.enable_test_so3_metrics = True
    cfg.test_so3_rotation_runs = int(test_so3_rotation_runs)
    cfg.cluster_acc_seed = int(kmeans_seed_base)
    cfg.test_cluster_acc_methods = ["kmeans++"]
    cfg.test_cluster_acc_runs = 1
    cfg.test_cluster_acc_runs_by_method = {}
    cfg.enable_embedding_metrics = False
    cfg.enable_svm_accuracy = False
    cfg.enable_train_split_svm_test_metric = False

    if not hasattr(cfg, "augmentation") or cfg.augmentation is None:
        cfg.augmentation = OmegaConf.create({})
    cfg.augmentation.rotation_scale = 0.0
    cfg.augmentation.noise_scale = 0.0
    cfg.augmentation.jitter_scale = 0.0
    cfg.augmentation.scaling_range = 0.0
    cfg.augmentation.track_augmentation = False

    if batch_size_override is not None:
        if int(batch_size_override) < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {batch_size_override}."
            )
        cfg.batch_size = int(batch_size_override)

    if num_workers_override is not None:
        if int(num_workers_override) < 0:
            raise ValueError(
                f"num_workers must be >= 0, got {num_workers_override}."
            )
        cfg.num_workers = int(num_workers_override)

    return cfg


def _coerce_metric(metrics: dict[str, Any], key: str) -> float:
    if key not in metrics:
        raise KeyError(
            f"Required metric {key!r} is missing. Available keys: {sorted(metrics.keys())}."
        )
    value = metrics[key]
    if value is None:
        raise ValueError(f"Metric {key!r} is null.")
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Metric {key!r} must be scalar, got shape {tuple(value.shape)}."
            )
        value = value.detach().cpu().item()
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Metric {key!r} must be numeric, got {value!r}."
        ) from exc
    if not torch.isfinite(torch.tensor(out)).item():
        raise ValueError(f"Metric {key!r} is not finite: {out}.")
    return out


def _require_test_metrics(report: dict[str, Any]) -> dict[str, Any]:
    stages = report.get("stages")
    if not isinstance(stages, dict):
        raise TypeError(
            f"Evaluation report has invalid 'stages': expected dict, got {type(stages)!r}."
        )
    test_metrics = stages.get("test")
    if not isinstance(test_metrics, list) or len(test_metrics) != 1:
        raise ValueError(
            "Expected exactly one test dataloader metrics entry, "
            f"got {test_metrics!r}."
        )
    metrics = test_metrics[0]
    if not isinstance(metrics, dict):
        raise TypeError(
            f"Expected test metrics dict, got {type(metrics)!r}."
        )
    return metrics


def _require_feature_bundle(
    features: np.ndarray | None,
    labels: np.ndarray | None,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if features is None or labels is None:
        raise RuntimeError(f"{context}: missing features or labels.")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    y = y.reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"{context}: expected 2D features, got shape {tuple(x.shape)}.")
    if y.ndim != 1:
        raise ValueError(f"{context}: expected 1D labels, got shape {tuple(y.shape)}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{context}: feature/label count mismatch: "
            f"features.shape={tuple(x.shape)}, labels.shape={tuple(y.shape)}."
        )
    if x.shape[0] == 0:
        raise ValueError(f"{context}: empty feature array.")
    if not np.isfinite(x).all():
        raise ValueError(f"{context}: non-finite values detected in features.")
    if np.issubdtype(y.dtype, np.floating) and not np.isfinite(y).all():
        raise ValueError(f"{context}: non-finite values detected in labels.")
    try:
        y = y.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context}: labels are not integer-like: dtype={y.dtype!r}.") from exc
    return x, y


def _run_single_kmeans_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    seed: int,
    context: str,
) -> dict[str, float]:
    if int(k) < 2:
        raise ValueError(f"{context}: k must be >= 2, got {k}.")
    if int(features.shape[0]) < int(k):
        raise ValueError(
            f"{context}: insufficient samples for k={k}: "
            f"samples={int(features.shape[0])}."
        )
    assignments = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=10,
        random_state=int(seed),
    ).fit_predict(features)
    return {
        "acc": float(_hungarian_cluster_accuracy(labels, assignments)),
        "nmi": float(normalized_mutual_info_score(labels, assignments)),
        "ari": float(adjusted_rand_score(labels, assignments)),
    }


def _summarize_metric_runs(values: list[float], *, context: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{context}: expected non-empty 1D run values, got shape {tuple(arr.shape)}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{context}: non-finite metric values encountered: {arr}.")
    return {
        "mean": float(arr.mean()),
        "error": float(arr.std()),
        "runs": [float(v) for v in arr.tolist()],
    }


def _compute_canonical_seed_stats(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    label: str,
    k: int,
    kmeans_seed_base: int,
    kmeans_seed_runs: int,
) -> dict[str, Any]:
    acc_runs: list[float] = []
    nmi_runs: list[float] = []
    ari_runs: list[float] = []
    for run_idx in range(int(kmeans_seed_runs)):
        _status_print(
            f"[kmeans] {label}: canonical seed {run_idx + 1}/{int(kmeans_seed_runs)}"
        )
        seed = int(kmeans_seed_base) + int(run_idx)
        metrics = _run_single_kmeans_metrics(
            features,
            labels,
            k=int(k),
            seed=seed,
            context=f"canonical seed run {run_idx + 1}",
        )
        acc_runs.append(metrics["acc"])
        nmi_runs.append(metrics["nmi"])
        ari_runs.append(metrics["ari"])
    return {
        "acc": _summarize_metric_runs(acc_runs, context="canonical acc"),
        "nmi": _summarize_metric_runs(nmi_runs, context="canonical nmi"),
        "ari": _summarize_metric_runs(ari_runs, context="canonical ari"),
    }


def _collect_rotated_feature_runs(
    model,
    *,
    label: str,
    rotation_runs: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    bundles: list[tuple[np.ndarray, np.ndarray]] = []
    rotation_seed_base = int(getattr(model, "test_so3_rotation_seed", 12345))
    for rotation_run_idx in range(int(rotation_runs)):
        _status_print(
            f"[features] {label}: collecting rotated features "
            f"{rotation_run_idx + 1}/{int(rotation_runs)}"
        )
        features, labels = _collect_rotated_split_supervised_features(
            model,
            "test",
            max_samples=None,
            rotation_seed_base=rotation_seed_base,
            rotation_run_idx=rotation_run_idx,
        )
        bundles.append(
            _require_feature_bundle(
                features,
                labels,
                context=f"rotated feature collection run {rotation_run_idx + 1}",
            )
        )
    if not bundles:
        raise RuntimeError("No rotated feature runs were collected.")
    ref_labels = bundles[0][1]
    for run_idx, (_, labels) in enumerate(bundles[1:], start=2):
        if labels.shape != ref_labels.shape or not np.array_equal(labels, ref_labels):
            raise ValueError(
                "Rotated feature runs produced inconsistent label order. "
                f"Reference shape={tuple(ref_labels.shape)}, "
                f"run#{run_idx} shape={tuple(labels.shape)}."
            )
    return bundles


def _compute_rotated_seed_stats(
    rotated_feature_runs: list[tuple[np.ndarray, np.ndarray]],
    *,
    label: str,
    k: int,
    kmeans_seed_base: int,
    kmeans_seed_runs: int,
) -> dict[str, Any]:
    if not rotated_feature_runs:
        raise ValueError("rotated_feature_runs must be non-empty.")

    acc_seed_means: list[float] = []
    nmi_seed_means: list[float] = []
    ari_seed_means: list[float] = []
    for seed_run_idx in range(int(kmeans_seed_runs)):
        _status_print(
            f"[kmeans] {label}: rotated seed {seed_run_idx + 1}/{int(kmeans_seed_runs)}"
        )
        seed = int(kmeans_seed_base) + int(seed_run_idx)
        acc_per_rotation: list[float] = []
        nmi_per_rotation: list[float] = []
        ari_per_rotation: list[float] = []
        for rotation_run_idx, (features, labels) in enumerate(rotated_feature_runs):
            metrics = _run_single_kmeans_metrics(
                features,
                labels,
                k=int(k),
                seed=seed,
                context=(
                    f"rotated seed run {seed_run_idx + 1}, "
                    f"rotation run {rotation_run_idx + 1}"
                ),
            )
            acc_per_rotation.append(metrics["acc"])
            nmi_per_rotation.append(metrics["nmi"])
            ari_per_rotation.append(metrics["ari"])
        acc_seed_means.append(float(np.mean(np.asarray(acc_per_rotation, dtype=np.float64))))
        nmi_seed_means.append(float(np.mean(np.asarray(nmi_per_rotation, dtype=np.float64))))
        ari_seed_means.append(float(np.mean(np.asarray(ari_per_rotation, dtype=np.float64))))

    return {
        "acc": _summarize_metric_runs(acc_seed_means, context="rotated acc"),
        "nmi": _summarize_metric_runs(nmi_seed_means, context="rotated nmi"),
        "ari": _summarize_metric_runs(ari_seed_means, context="rotated ari"),
    }


def _format_mean_plus_error(summary: dict[str, Any]) -> str:
    mean = float(summary["mean"])
    error = float(summary["error"])
    if not np.isfinite(mean) or not np.isfinite(error):
        raise ValueError(f"Cannot format non-finite summary values: {summary}.")
    return f"{mean:.3f} +- {error:.3f}"


def _build_summary_row(
    *,
    label: str,
    checkpoint_path: Path,
    data_config_path: Path,
    canonical_stats: dict[str, Any],
    rotated_stats: dict[str, Any],
    kmeans_seed_runs: int,
    test_so3_rotation_runs: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "data_config": str(data_config_path),
        "kmeans_seed_runs": int(kmeans_seed_runs),
        "test_so3_rotation_runs": int(test_so3_rotation_runs),
        "acc_not_rotated": _format_mean_plus_error(canonical_stats["acc"]),
        "nmi_not_rotated": _format_mean_plus_error(canonical_stats["nmi"]),
        "ari_not_rotated": _format_mean_plus_error(canonical_stats["ari"]),
        "acc_rotated": _format_mean_plus_error(rotated_stats["acc"]),
        "nmi_rotated": _format_mean_plus_error(rotated_stats["nmi"]),
        "ari_rotated": _format_mean_plus_error(rotated_stats["ari"]),
    }


def _run_test_evaluation(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    cuda_device: int,
    cfg: DictConfig,
    disable_progress_bar: bool,
) -> tuple[dict[str, Any], Any]:
    model, resolved_cfg, device = load_barlow_model(
        checkpoint_path=str(checkpoint_path),
        cuda_device=int(cuda_device),
        cfg=cfg,
    )
    dm = build_datamodule(resolved_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_eval_trainer(
        resolved_cfg,
        device=device,
        cuda_device=int(cuda_device),
        output_dir=output_dir,
        disable_progress_bar=disable_progress_bar,
    )
    raw_test = trainer.test(model=model, datamodule=dm, verbose=not disable_progress_bar)
    test_metrics = _normalize_metrics(raw_test, stage="test")
    report: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": device,
        "batch_size": int(resolved_cfg.batch_size),
        "num_workers": int(resolved_cfg.num_workers),
        "stages": {
            "test": test_metrics,
        },
    }
    _print_metrics("test", test_metrics)
    report_path = output_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[eval] Saved metrics to {report_path}")
    return report, model


def _write_summary_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "label",
        "checkpoint_path",
        "data_config",
        "kmeans_seed_runs",
        "test_so3_rotation_runs",
        "acc_not_rotated",
        "nmi_not_rotated",
        "ari_not_rotated",
        "acc_rotated",
        "nmi_rotated",
        "ari_rotated",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"[summary] {row['label']}: "
        f"ACC={row['acc_not_rotated']}, "
        f"NMI={row['nmi_not_rotated']}, "
        f"ARI={row['ari_not_rotated']}, "
        f"ACC_rot={row['acc_rotated']}, "
        f"NMI_rot={row['nmi_rotated']}, "
        f"ARI_rot={row['ari_rotated']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate contrastive checkpoints on the held-out synthetic "
            "polycrystalline test split and export canonical/rotated ACC, NMI, ARI."
        )
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=DEFAULT_DATA_CONFIG,
        help=(
            "Synthetic data config to use for evaluation "
            f"(default: {DEFAULT_DATA_CONFIG})."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint spec in LABEL=PATH form. Repeat for multiple checkpoints. "
            "If omitted, the script uses the requested multi and Aluminum checkpoints."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory. The script writes per-checkpoint metrics.json "
            "and a top-level summary.csv here."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index to use when CUDA is available (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional test batch size override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used by the synthetic train/val split (default: 42).",
    )
    parser.add_argument(
        "--test-so3-rotation-runs",
        type=int,
        default=5,
        help="Number of random SO(3) test reruns for rotated metrics (default: 5).",
    )
    parser.add_argument(
        "--kmeans-seed-runs",
        type=int,
        default=10,
        help="Number of K-means reruns used to estimate clustering seed error (default: 10).",
    )
    parser.add_argument(
        "--kmeans-seed-base",
        type=int,
        default=0,
        help="Base K-means random seed; reruns use consecutive seeds from this base (default: 0).",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable Lightning progress display.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.split_seed) < 0:
        raise ValueError(f"split_seed must be >= 0, got {args.split_seed}.")
    if int(args.test_so3_rotation_runs) < 1:
        raise ValueError(
            "test_so3_rotation_runs must be >= 1, "
            f"got {args.test_so3_rotation_runs}."
        )
    if int(args.kmeans_seed_runs) < 1:
        raise ValueError(
            f"kmeans_seed_runs must be >= 1, got {args.kmeans_seed_runs}."
        )

    data_config_path = _resolve_file(args.data_config, what="Data config")
    data_cfg = _load_data_cfg(data_config_path)

    checkpoint_specs = (
        [_parse_checkpoint_spec(raw) for raw in args.checkpoint]
        if args.checkpoint
        else list(DEFAULT_CHECKPOINT_SPECS)
    )

    output_dir = _resolve_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for label, checkpoint_raw in checkpoint_specs:
        _status_print(f"[checkpoint] Starting {label}")
        checkpoint_path = _resolve_file(checkpoint_raw, what=f"Checkpoint for {label}")
        checkpoint_cfg = _load_checkpoint_cfg(checkpoint_path)
        eval_cfg = _build_eval_cfg(
            checkpoint_cfg,
            data_cfg,
            split_seed=int(args.split_seed),
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            test_so3_rotation_runs=int(args.test_so3_rotation_runs),
            kmeans_seed_base=int(args.kmeans_seed_base),
        )

        checkpoint_output_dir = output_dir / _slugify_label(label)
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config=eval_cfg,
            f=checkpoint_output_dir / "resolved_eval_config.yaml",
        )

        report, model = _run_test_evaluation(
            checkpoint_path=checkpoint_path,
            output_dir=checkpoint_output_dir,
            cuda_device=int(args.cuda_device),
            cfg=eval_cfg,
            disable_progress_bar=bool(args.disable_progress_bar),
        )

        metrics = _require_test_metrics(report)
        _status_print(f"[features] {label}: collecting canonical test features")
        canonical_features, canonical_labels = _collect_split_supervised_features(
            model,
            "test",
            max_samples=None,
        )
        canonical_features, canonical_labels = _require_feature_bundle(
            canonical_features,
            canonical_labels,
            context=f"canonical features for {label}",
        )
        k_eval = _resolve_hungarian_eval_k(model, "test", canonical_labels)
        if k_eval is None:
            raise ValueError(
                f"Could not infer test class count for checkpoint {checkpoint_path}."
            )
        canonical_stats = _compute_canonical_seed_stats(
            canonical_features,
            canonical_labels,
            label=label,
            k=int(k_eval),
            kmeans_seed_base=int(args.kmeans_seed_base),
            kmeans_seed_runs=int(args.kmeans_seed_runs),
        )

        rotated_feature_runs = _collect_rotated_feature_runs(
            model,
            label=label,
            rotation_runs=int(args.test_so3_rotation_runs),
        )
        rotated_stats = _compute_rotated_seed_stats(
            rotated_feature_runs,
            label=label,
            k=int(k_eval),
            kmeans_seed_base=int(args.kmeans_seed_base),
            kmeans_seed_runs=int(args.kmeans_seed_runs),
        )

        seed_stats = {
            "label": label,
            "checkpoint_path": str(checkpoint_path),
            "data_config": str(data_config_path),
            "kmeans_seed_runs": int(args.kmeans_seed_runs),
            "kmeans_seed_base": int(args.kmeans_seed_base),
            "test_so3_rotation_runs": int(args.test_so3_rotation_runs),
            "test_metrics_from_lightning": {
                "canonical_acc": _coerce_metric(
                    metrics, "test/class/acc_kmeans_plusplus_hungarian_canonical"
                ),
                "canonical_nmi": _coerce_metric(metrics, "test/class/nmi"),
                "canonical_ari": _coerce_metric(metrics, "test/class/ari"),
                "rotated_acc": _coerce_metric(
                    metrics, "test/class/acc_kmeans_plusplus_hungarian_rotated"
                ),
                "rotated_nmi": _coerce_metric(metrics, "test/class/nmi_rotated"),
                "rotated_ari": _coerce_metric(metrics, "test/class/ari_rotated"),
            },
            "kmeans_seed_summary": {
                "canonical": canonical_stats,
                "rotated": rotated_stats,
            },
        }
        with (checkpoint_output_dir / "kmeans_seed_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(seed_stats, handle, indent=2)

        row = _build_summary_row(
            label=label,
            checkpoint_path=checkpoint_path,
            data_config_path=data_config_path,
            canonical_stats=canonical_stats,
            rotated_stats=rotated_stats,
            kmeans_seed_runs=int(args.kmeans_seed_runs),
            test_so3_rotation_runs=int(args.test_so3_rotation_runs),
        )
        summary_rows.append(row)
        _print_row(row)

    csv_path = output_dir / "summary.csv"
    _write_summary_csv(summary_rows, csv_path)
    print(f"[summary] Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
