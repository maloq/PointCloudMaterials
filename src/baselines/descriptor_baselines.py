from __future__ import annotations

import json
import math
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.special import sph_harm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True)
class CenterShell:
    center_idx: int
    cutoff: float
    shell_indices: np.ndarray
    shell_distances: np.ndarray


@dataclass(frozen=True)
class EvalSettings:
    max_supervised_samples: int
    max_test_samples: int
    val_cluster_eval_k: int | None
    cluster_acc_seed: int
    val_cluster_acc_methods: list[str]
    val_cluster_acc_runs: int
    val_cluster_acc_runs_by_method: dict[str, int]
    test_cluster_acc_methods: list[str]
    test_cluster_acc_runs: int
    test_cluster_acc_runs_by_method: dict[str, int]
    enable_test_so3_metrics: bool
    test_so3_rotation_runs: int
    test_so3_rotation_seed: int


def _as_string_list(value, default: list[str]) -> list[str]:
    if value is None:
        return list(default)

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif hasattr(value, "__iter__") and not isinstance(value, (bytes, dict)):
        values = list(value)
    else:
        values = [value]

    out = [str(v).strip() for v in values if str(v).strip()]
    return out or list(default)


def _as_int_mapping(value, default: dict[str, int]) -> dict[str, int]:
    if value is None:
        source = default.items()
    elif hasattr(value, "items"):
        source = value.items()
    else:
        source = default.items()

    out: dict[str, int] = {}
    for key, raw in source:
        k = str(key).strip()
        if not k:
            continue
        try:
            runs = int(raw)
        except (TypeError, ValueError):
            warnings.warn(
                f"Ignoring non-integer run count for key {k!r}: {raw!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if runs > 0:
            out[k] = runs
    return out


def _parse_optional_eval_k(value, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be null or integer >= 2, got bool {value!r}")
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be null or integer >= 2, got {value!r}") from exc
    if k < 2:
        raise ValueError(f"{field_name} must be null or integer >= 2, got {value!r}")
    return k


def _normalize_method_name(method) -> str:
    return str(method).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _is_kmeans_plus_plus_method(method) -> bool:
    return _normalize_method_name(method) in {"kmeans++", "kmeansplusplus", "kmeanspp"}


def _ensure_kmeans_plus_plus_method(methods: list[str]) -> list[str]:
    out = list(methods)
    if not any(_is_kmeans_plus_plus_method(method) for method in out):
        out.append("kmeans++")
    return out


def _to_finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _primary_kmeansplusplus_hungarian_key(
    cluster_metrics: dict[str, float],
    eval_k: int | None,
) -> str | None:
    preferred_keys: list[str] = ["ACC_KMEANS_PLUSPLUS_HUNGARIAN"]
    if eval_k is not None:
        preferred_keys.insert(0, f"ACC_KMEANS_PLUSPLUS_HUNGARIAN_K{eval_k}")
    for preferred in preferred_keys:
        if preferred in cluster_metrics:
            return preferred
    for key in sorted(cluster_metrics.keys()):
        upper = str(key).upper()
        if not upper.startswith("ACC_"):
            continue
        if "KMEANS_PLUSPLUS" not in upper or "HUNGARIAN" not in upper:
            continue
        if upper.endswith(("_MEAN", "_STD", "_BEST", "_RUNS", "_MIN", "_MAX")):
            continue
        return key
    return None


_ACC_K_SUFFIX_RE = re.compile(r"^(ACC_[A-Z0-9_]+)_K\d+((?:_(?:MEAN|STD|BEST|RUNS|MIN|MAX))?)$")


def _stabilize_class_metric_keys(
    metrics: dict[str, float],
    *,
    hungarian_eval_k: int | None,
) -> dict[str, float]:
    stable: dict[str, float] = {}
    for raw_name, raw_value in metrics.items():
        upper_name = str(raw_name).upper()
        match = _ACC_K_SUFFIX_RE.match(upper_name)
        if match is not None:
            stable_name = f"{match.group(1)}{match.group(2)}"
        else:
            stable_name = upper_name
        if stable_name in stable:
            raise ValueError(
                "Metric name collision while removing class-count suffixes: "
                f"{raw_name!r} collides with another metric at {stable_name!r}."
            )
        val = _to_finite_float(raw_value)
        if val is None:
            raise ValueError(
                f"Metric {raw_name!r} has non-finite value {raw_value!r}; cannot log stable metrics."
            )
        stable[stable_name] = val

    if hungarian_eval_k is not None:
        if "HUNGARIAN_EVAL_K" in stable:
            raise ValueError("Metric key collision: 'HUNGARIAN_EVAL_K' is already present.")
        stable["HUNGARIAN_EVAL_K"] = float(int(hungarian_eval_k))
    return stable


def _normalize_acc_eval_methods(acc_eval_methods) -> list[str]:
    if acc_eval_methods is None:
        raw_methods = ["kmeans++"]
    elif isinstance(acc_eval_methods, str):
        raw_methods = [acc_eval_methods]
    else:
        raw_methods = list(acc_eval_methods)

    methods: list[str] = []
    seen: set[str] = set()
    for raw in raw_methods:
        key = _normalize_method_name(raw)
        method = None
        if key in {"kmeans", "kmeansrandom"}:
            continue
        if key in {"kmeans++", "kmeansplusplus", "kmeanspp"}:
            method = "kmeans++"
        if method is None or method in seen:
            continue
        seen.add(method)
        methods.append(method)
    return methods


def _normalize_acc_eval_runs_by_method(acc_eval_runs_by_method) -> dict[str, int]:
    if acc_eval_runs_by_method is None:
        return {}
    if not hasattr(acc_eval_runs_by_method, "items"):
        return {}

    runs_by_method: dict[str, int] = {}
    for raw_method, raw_runs in acc_eval_runs_by_method.items():
        methods = _normalize_acc_eval_methods([raw_method])
        if not methods:
            continue
        try:
            runs = max(1, int(raw_runs))
        except (TypeError, ValueError):
            continue
        runs_by_method[methods[0]] = runs
    return runs_by_method


def _compute_cluster_assignments_for_acc(
    latents: np.ndarray,
    n_clusters: int,
    method: str,
    *,
    seed: int,
) -> np.ndarray:
    if method == "kmeans++":
        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=seed)
        return model.fit_predict(latents)
    raise ValueError(f"Unsupported ACC evaluator method: {method}")


def _acc_metric_prefix(method: str, k_eval: int) -> str:
    if method == "kmeans++":
        return f"ACC_KMEANS_PLUSPLUS_HUNGARIAN_K{k_eval}"
    raise ValueError(f"Unsupported ACC evaluator method: {method}")


def _hungarian_cluster_accuracy(labels: np.ndarray, assignments: np.ndarray) -> float:
    labels = np.asarray(labels)
    assignments = np.asarray(assignments)
    if labels.shape != assignments.shape or labels.size == 0:
        raise ValueError("labels and assignments must have identical non-empty shape")

    label_vals, label_inv = np.unique(labels, return_inverse=True)
    cluster_vals, cluster_inv = np.unique(assignments, return_inverse=True)
    contingency = np.zeros((label_vals.size, cluster_vals.size), dtype=np.int64)
    np.add.at(contingency, (label_inv, cluster_inv), 1)
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    correct = contingency[row_ind, col_ind].sum()
    return float(correct / labels.size)


def compute_cluster_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    stage: str,
    *,
    hungarian_eval_k: int | None = None,
    acc_eval_methods=None,
    acc_eval_runs: int = 1,
    acc_eval_runs_by_method=None,
    acc_random_seed: int = 0,
) -> dict[str, float] | None:
    metrics: dict[str, float] = {}
    unique = np.unique(labels)
    if unique.size >= 2 and latents.shape[0] >= unique.size:
        assignments = KMeans(n_clusters=unique.size, n_init=10, random_state=0).fit_predict(latents)
        metrics["ARI"] = float(adjusted_rand_score(labels, assignments))
        metrics["NMI"] = float(normalized_mutual_info_score(labels, assignments))

    stage_l = str(stage).lower()
    k_eval = int(hungarian_eval_k) if hungarian_eval_k is not None else None
    if stage_l in {"val", "test"} and k_eval is not None and k_eval >= 2 and latents.shape[0] >= k_eval:
        methods = _normalize_acc_eval_methods(acc_eval_methods)
        runs = max(1, int(acc_eval_runs))
        runs_by_method = _normalize_acc_eval_runs_by_method(acc_eval_runs_by_method)
        base_seed = int(acc_random_seed)
        for method in methods:
            method_runs = runs_by_method.get(method, runs)
            acc_values: list[float] = []
            for run_idx in range(method_runs):
                pred_labels = _compute_cluster_assignments_for_acc(
                    latents,
                    k_eval,
                    method,
                    seed=base_seed + run_idx,
                )
                acc_values.append(_hungarian_cluster_accuracy(labels, pred_labels))
            if not acc_values:
                continue
            prefix = _acc_metric_prefix(method, k_eval)
            if len(acc_values) == 1:
                metrics[prefix] = float(acc_values[0])
            else:
                acc_arr = np.asarray(acc_values, dtype=np.float32)
                metrics[prefix] = float(acc_arr.mean())
                metrics[f"{prefix}_MEAN"] = float(acc_arr.mean())
                metrics[f"{prefix}_STD"] = float(acc_arr.std())
                metrics[f"{prefix}_BEST"] = float(acc_arr.max())
                metrics[f"{prefix}_RUNS"] = float(acc_arr.size)

    return metrics or None


def load_eval_settings_from_cfg(cfg) -> EvalSettings:
    val_methods = _ensure_kmeans_plus_plus_method(
        _as_string_list(getattr(cfg, "val_cluster_acc_methods", None), default=["kmeans++"])
    )
    test_methods = _ensure_kmeans_plus_plus_method(
        _as_string_list(getattr(cfg, "test_cluster_acc_methods", None), default=["kmeans++"])
    )
    rotation_runs = int(
        getattr(
            cfg,
            "test_so3_rotation_runs",
            getattr(cfg, "analysis_test_rotation_runs", 5),
        )
    )
    if rotation_runs < 1:
        raise ValueError(f"test_so3_rotation_runs must be >= 1, got {rotation_runs}.")

    return EvalSettings(
        max_supervised_samples=int(getattr(cfg, "max_supervised_samples", 8192)),
        max_test_samples=int(getattr(cfg, "max_test_samples", 1000)),
        val_cluster_eval_k=_parse_optional_eval_k(
            getattr(cfg, "val_cluster_eval_k", None),
            field_name="val_cluster_eval_k",
        ),
        cluster_acc_seed=int(getattr(cfg, "cluster_acc_seed", 0)),
        val_cluster_acc_methods=val_methods,
        val_cluster_acc_runs=max(1, int(getattr(cfg, "val_cluster_acc_runs", 1))),
        val_cluster_acc_runs_by_method=_as_int_mapping(
            getattr(cfg, "val_cluster_acc_runs_by_method", None),
            default={},
        ),
        test_cluster_acc_methods=test_methods,
        test_cluster_acc_runs=max(1, int(getattr(cfg, "test_cluster_acc_runs", 1))),
        test_cluster_acc_runs_by_method=_as_int_mapping(
            getattr(cfg, "test_cluster_acc_runs_by_method", None),
            default={},
        ),
        enable_test_so3_metrics=bool(getattr(cfg, "enable_test_so3_metrics", True)),
        test_so3_rotation_runs=rotation_runs,
        test_so3_rotation_seed=int(
            getattr(
                cfg,
                "test_so3_rotation_seed",
                getattr(cfg, "analysis_test_rotation_seed", 12345),
            )
        ),
    )


def _unwrap_dataset(dataset):
    current = dataset
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        inner = getattr(current, "dataset", None)
        if inner is None or inner is current:
            return current
        current = inner
    return current


def dataset_class_count(dataset) -> int | None:
    base = _unwrap_dataset(dataset)
    if base is None:
        return None

    class_names = getattr(base, "class_names", None)
    if class_names is not None:
        if hasattr(class_names, "items"):
            count = len(dict(class_names))
            return count if count > 1 else None
        if hasattr(class_names, "__len__") and not isinstance(class_names, (str, bytes)):
            count = len(class_names)
            return count if count > 1 else None

    num_classes = getattr(base, "num_classes", None)
    if callable(num_classes):
        value = num_classes()
    else:
        value = num_classes
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dataset reports invalid num_classes={value!r}; expected integer-like value."
        ) from exc
    return count if count > 1 else None


def collect_split_point_clouds_and_labels(
    dataset,
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset is None:
        raise ValueError("dataset must not be None")
    total = int(len(dataset))
    if total <= 0:
        raise ValueError("dataset is empty; cannot collect point clouds")

    limit = total if max_samples is None or int(max_samples) <= 0 else min(total, int(max_samples))
    points: list[np.ndarray] = []
    labels: list[int] = []
    for idx in range(limit):
        sample = dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError(
                "Expected dataset samples to be dictionaries with 'points' and 'class_id', "
                f"got type={type(sample)} at index={idx}."
            )
        sample_points = sample.get("points")
        sample_label = sample.get("class_id")
        if sample_points is None or sample_label is None:
            raise KeyError(
                "Dataset sample is missing required keys 'points'/'class_id': "
                f"available_keys={sorted(sample.keys())}, index={idx}."
            )

        if torch.is_tensor(sample_points):
            points_np = sample_points.detach().cpu().numpy()
        else:
            points_np = np.asarray(sample_points)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError(
                f"Expected points shape (N, 3), got {tuple(points_np.shape)} at index={idx}."
            )
        points.append(np.asarray(points_np, dtype=np.float64))

        if torch.is_tensor(sample_label):
            label_val = int(sample_label.detach().cpu().item())
        else:
            label_val = int(sample_label)
        labels.append(label_val)

    if not points:
        raise RuntimeError("Collected zero samples from dataset; cannot evaluate baseline.")
    return np.stack(points, axis=0), np.asarray(labels, dtype=np.int64)


def _format_label_histogram(labels: np.ndarray, *, max_entries: int = 10) -> str:
    arr = np.asarray(labels).reshape(-1)
    if arr.size == 0:
        return "[]"
    unique, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    parts: list[str] = []
    for idx in order[:max_entries]:
        parts.append(f"{int(unique[idx])}:{int(counts[idx])}")
    if unique.size > max_entries:
        parts.append(f"...(+{int(unique.size - max_entries)} classes)")
    return "[" + ", ".join(parts) + "]"


def _resolve_hungarian_eval_k(
    *,
    stage: str,
    labels: np.ndarray,
    dataset_k: int | None,
    configured_val_k: int | None,
) -> int | None:
    stage_l = str(stage).lower()
    if stage_l not in {"val", "test"}:
        return None

    observed_k = int(np.unique(labels).size) if labels.size > 0 else None
    inferred_k = dataset_k if dataset_k is not None else observed_k
    if dataset_k is not None and observed_k is not None and int(dataset_k) != int(observed_k):
        raise ValueError(
            "Hungarian ACC class-count mismatch: "
            f"observed {observed_k} unique labels in the collected {stage_l} split, "
            f"but dataset reports {dataset_k} classes. "
            "Increase the split sample limit or inspect the split/label balance. "
            f"label_histogram={_format_label_histogram(labels)}."
        )
    if (
        stage_l == "val"
        and configured_val_k is not None
        and inferred_k is not None
        and int(configured_val_k) != int(inferred_k)
    ):
        raise ValueError(
            "Hungarian ACC k must match the class count. "
            f"Configured val_cluster_eval_k={int(configured_val_k)}, "
            f"inferred classes={int(inferred_k)}."
        )
    return inferred_k


def _random_rotation_matrices_numpy(batch_size: int, *, seed: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rand = torch.randn(batch_size, 3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(rand)
    signs = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * signs.unsqueeze(-1)
    det = torch.det(q)
    neg = det < 0
    if bool(neg.any()):
        q[neg, :, 0] *= -1
    return q.cpu().numpy()


def rotate_point_cloud_batch(points: np.ndarray, *, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(
            f"Expected batched point clouds with shape (B, N, 3), got {tuple(pts.shape)}."
        )
    rotations = _random_rotation_matrices_numpy(int(pts.shape[0]), seed=int(seed))
    return np.matmul(pts, rotations)


def compute_supervised_stage_metrics_from_features(
    *,
    stage: str,
    features: np.ndarray,
    labels: np.ndarray,
    dataset_k: int | None,
    settings: EvalSettings,
    rotated_feature_fn=None,
) -> dict[str, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(x.shape)} for stage={stage!r}.")
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {tuple(y.shape)} for stage={stage!r}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "features/labels sample-count mismatch: "
            f"stage={stage!r}, features.shape={tuple(x.shape)}, labels.shape={tuple(y.shape)}."
        )
    if x.shape[0] <= 0:
        raise ValueError(f"No samples available for stage={stage!r}.")
    if not np.isfinite(x).all():
        bad = int((~np.isfinite(x)).sum())
        raise ValueError(
            "Descriptor features contain non-finite values: "
            f"stage={stage!r}, shape={tuple(x.shape)}, nonfinite_values={bad}/{x.size}."
        )

    stage_l = str(stage).lower()
    if stage_l == "val":
        acc_methods = settings.val_cluster_acc_methods
        acc_runs = settings.val_cluster_acc_runs
        acc_runs_by_method = settings.val_cluster_acc_runs_by_method
    elif stage_l == "test":
        acc_methods = settings.test_cluster_acc_methods
        acc_runs = settings.test_cluster_acc_runs
        acc_runs_by_method = settings.test_cluster_acc_runs_by_method
    else:
        acc_methods = []
        acc_runs = 1
        acc_runs_by_method = {}

    hungarian_eval_k = _resolve_hungarian_eval_k(
        stage=stage_l,
        labels=y,
        dataset_k=dataset_k,
        configured_val_k=settings.val_cluster_eval_k,
    )
    if stage_l in {"val", "test"} and hungarian_eval_k is not None and x.shape[0] < int(hungarian_eval_k):
        raise RuntimeError(
            "Insufficient samples for Hungarian ACC evaluation: "
            f"stage={stage_l!r}, samples={int(x.shape[0])}, hungarian_eval_k={int(hungarian_eval_k)}."
        )

    metrics = compute_cluster_metrics(
        x,
        y,
        stage_l,
        hungarian_eval_k=hungarian_eval_k,
        acc_eval_methods=acc_methods,
        acc_eval_runs=max(1, int(acc_runs)),
        acc_eval_runs_by_method=acc_runs_by_method,
        acc_random_seed=int(settings.cluster_acc_seed),
    ) or {}

    if stage_l == "test" and hungarian_eval_k is not None:
        canonical_acc_key = _primary_kmeansplusplus_hungarian_key(metrics, int(hungarian_eval_k))
        if canonical_acc_key is None:
            raise RuntimeError(
                "Canonical test metrics are missing kmeans++ Hungarian ACC. "
                f"Available keys: {sorted(metrics.keys())}."
            )
        canonical_acc = _to_finite_float(metrics.get(canonical_acc_key))
        if canonical_acc is None:
            raise RuntimeError(
                f"Canonical test metric {canonical_acc_key!r} is non-finite: {metrics.get(canonical_acc_key)!r}."
            )
        metrics["ACC_KMEANS_PLUSPLUS_HUNGARIAN_CANONICAL"] = canonical_acc
        canonical_nmi = _to_finite_float(metrics.get("NMI"))
        canonical_ari = _to_finite_float(metrics.get("ARI"))

        if settings.enable_test_so3_metrics:
            if rotated_feature_fn is None:
                raise ValueError(
                    "rotated_feature_fn must be provided for test metrics when enable_test_so3_metrics=true."
                )

            run_acc_values: list[float] = []
            run_nmi_values: list[float] = []
            run_ari_values: list[float] = []
            for run_idx in range(settings.test_so3_rotation_runs):
                rotated_features, rotated_labels = rotated_feature_fn(run_idx)
                run_x = np.asarray(rotated_features, dtype=np.float32)
                run_y = np.asarray(rotated_labels, dtype=np.int64).reshape(-1)
                if run_x.shape[0] != run_y.shape[0]:
                    raise ValueError(
                        "Rotated features/labels mismatch: "
                        f"run={run_idx}, features.shape={tuple(run_x.shape)}, labels.shape={tuple(run_y.shape)}."
                    )
                run_metrics = compute_cluster_metrics(
                    run_x,
                    run_y,
                    stage="test",
                    hungarian_eval_k=int(hungarian_eval_k),
                    acc_eval_methods=["kmeans++"],
                    acc_eval_runs=1,
                    acc_eval_runs_by_method={},
                    acc_random_seed=int(settings.test_so3_rotation_seed) + run_idx,
                ) or {}
                run_acc_key = _primary_kmeansplusplus_hungarian_key(run_metrics, int(hungarian_eval_k))
                if run_acc_key is None:
                    raise RuntimeError(
                        "Rotated test metrics are missing kmeans++ Hungarian ACC. "
                        f"run={run_idx}, available_keys={sorted(run_metrics.keys())}."
                    )
                run_acc = _to_finite_float(run_metrics.get(run_acc_key))
                run_nmi = _to_finite_float(run_metrics.get("NMI"))
                run_ari = _to_finite_float(run_metrics.get("ARI"))
                if run_acc is None or run_nmi is None or run_ari is None:
                    raise RuntimeError(
                        "Rotated test metrics must contain finite ACC/NMI/ARI values. "
                        f"run={run_idx}, metrics={run_metrics}."
                    )
                run_acc_values.append(run_acc)
                run_nmi_values.append(run_nmi)
                run_ari_values.append(run_ari)

            acc_arr = np.asarray(run_acc_values, dtype=np.float64)
            nmi_arr = np.asarray(run_nmi_values, dtype=np.float64)
            ari_arr = np.asarray(run_ari_values, dtype=np.float64)
            rotated_mean = float(acc_arr.mean())
            rotated_nmi_mean = float(nmi_arr.mean())
            rotated_ari_mean = float(ari_arr.mean())
            metrics.update(
                {
                    "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED": rotated_mean,
                    "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_STD": float(acc_arr.std()),
                    "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_MIN": float(acc_arr.min()),
                    "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_MAX": float(acc_arr.max()),
                    "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_RUNS": float(acc_arr.size),
                    "NMI_ROTATED": rotated_nmi_mean,
                    "NMI_ROTATED_STD": float(nmi_arr.std()),
                    "NMI_ROTATED_MIN": float(nmi_arr.min()),
                    "NMI_ROTATED_MAX": float(nmi_arr.max()),
                    "NMI_ROTATED_RUNS": float(nmi_arr.size),
                    "ARI_ROTATED": rotated_ari_mean,
                    "ARI_ROTATED_STD": float(ari_arr.std()),
                    "ARI_ROTATED_MIN": float(ari_arr.min()),
                    "ARI_ROTATED_MAX": float(ari_arr.max()),
                    "ARI_ROTATED_RUNS": float(ari_arr.size),
                    "SO3_ACCURACY_KMEANS_PLUSPLUS_HUNGARIAN": rotated_mean,
                    "SO3_VS_CANONICAL_ACC_DELTA": rotated_mean - float(canonical_acc),
                    "SO3_ROTATION_RUNS": float(acc_arr.size),
                }
            )
            if canonical_nmi is not None:
                metrics["SO3_VS_CANONICAL_NMI_DELTA"] = rotated_nmi_mean - float(canonical_nmi)
            if canonical_ari is not None:
                metrics["SO3_VS_CANONICAL_ARI_DELTA"] = rotated_ari_mean - float(canonical_ari)
            if abs(float(canonical_acc)) > 1e-12:
                metrics["SO3_VS_CANONICAL_ACC_RATIO"] = rotated_mean / float(canonical_acc)

    return _stabilize_class_metric_keys(
        metrics,
        hungarian_eval_k=hungarian_eval_k if stage_l in {"val", "test"} else None,
    )


def _prefix_stage_class_metrics(stage: str, metrics: dict[str, float]) -> dict[str, float]:
    prefixed: dict[str, float] = {}
    for name, value in metrics.items():
        prefixed[f"{stage}/class/{name.lower()}"] = float(value)
    return prefixed


def infer_center_shell(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
    shell_min_neighbors: int,
    shell_max_neighbors: int,
) -> CenterShell:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {tuple(pts.shape)}.")
    if pts.shape[0] < shell_min_neighbors + 2:
        raise ValueError(
            "Point cloud does not contain enough atoms to infer a center shell: "
            f"num_points={int(pts.shape[0])}, shell_min_neighbors={shell_min_neighbors}."
        )

    norms = np.linalg.norm(pts, axis=1)
    center_idx = int(np.argmin(norms))
    center_norm = float(norms[center_idx])
    if center_norm > float(center_atom_tolerance):
        raise ValueError(
            "Point cloud is expected to be centered on an atom at the origin, but the nearest atom is too far away: "
            f"center_idx={center_idx}, center_distance={center_norm:.6e}, "
            f"center_atom_tolerance={float(center_atom_tolerance):.6e}."
        )

    center = pts[center_idx]
    distances = np.linalg.norm(pts - center[None, :], axis=1)
    keep = np.arange(pts.shape[0]) != center_idx
    neighbor_indices = np.flatnonzero(keep)
    neighbor_distances = distances[keep]
    order = np.argsort(neighbor_distances)
    ordered_indices = neighbor_indices[order]
    ordered_distances = neighbor_distances[order]
    if ordered_distances.size < shell_min_neighbors + 1:
        raise ValueError(
            "Need at least shell_min_neighbors + 1 neighbors to infer a shell cutoff, "
            f"got {int(ordered_distances.size)}."
        )

    max_rank = min(int(shell_max_neighbors), int(ordered_distances.size - 1))
    if max_rank < shell_min_neighbors:
        raise ValueError(
            "shell_max_neighbors must allow at least one cutoff gap candidate: "
            f"shell_min_neighbors={shell_min_neighbors}, shell_max_neighbors={shell_max_neighbors}, "
            f"available_neighbors={int(ordered_distances.size)}."
        )
    candidate_distances = ordered_distances[: max_rank + 1]
    gaps = np.diff(candidate_distances)
    start_idx = int(shell_min_neighbors - 1)
    if start_idx >= gaps.size:
        raise RuntimeError(
            "Internal shell-inference error: no candidate gap remains after applying shell_min_neighbors. "
            f"candidate_distances.shape={tuple(candidate_distances.shape)}, start_idx={start_idx}."
        )
    search_gaps = gaps[start_idx:]
    gap_tol = max(1e-10, 1e-8 * float(candidate_distances[-1]))
    if float(np.max(search_gaps)) <= gap_tol:
        shell_size = int(candidate_distances.size)
        cutoff = float(candidate_distances[-1] + gap_tol)
    else:
        gap_idx = int(start_idx + np.argmax(search_gaps))
        shell_size = int(gap_idx + 1)
        cutoff = float(0.5 * (candidate_distances[gap_idx] + candidate_distances[gap_idx + 1]))
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError(
            f"Inferred an invalid shell cutoff={cutoff!r} from candidate distances {candidate_distances.tolist()}."
        )
    return CenterShell(
        center_idx=center_idx,
        cutoff=cutoff,
        shell_indices=ordered_indices[:shell_size].copy(),
        shell_distances=ordered_distances[:shell_size].copy(),
    )


class DescriptorBaseline(ABC):
    requires_fit: bool = False

    @abstractmethod
    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, point_clouds: np.ndarray) -> None:
        return None

    def fit_transform(self, point_clouds: np.ndarray) -> np.ndarray:
        self.fit(point_clouds)
        return self.transform(point_clouds)

    def metadata(self) -> dict[str, Any]:
        return {}


class SteinhardtDescriptorBaseline(DescriptorBaseline):
    def __init__(
        self,
        *,
        l_values: Sequence[int],
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        append_shell_size: bool,
    ) -> None:
        self.l_values = [int(v) for v in l_values]
        if not self.l_values:
            raise ValueError("Steinhardt baseline requires at least one l value.")
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.append_shell_size = bool(append_shell_size)

    @staticmethod
    def _ql(vectors: np.ndarray, l: int) -> float:
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(
                f"Steinhardt q_{l} received zero-length neighbor vectors; cannot evaluate spherical harmonics."
            )
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        phi = np.arccos(np.clip(z / norms, -1.0, 1.0))
        qlm = []
        for m in range(-l, l + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                qlm.append(np.mean(sph_harm(m, l, theta, phi)))
        qlm_arr = np.asarray(qlm, dtype=np.complex128)
        prefactor = 4.0 * np.pi / float(2 * l + 1)
        return float(np.sqrt(prefactor * np.sum(np.abs(qlm_arr) ** 2)).real)

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        rows: list[np.ndarray] = []
        for sample_idx, points in enumerate(pcs):
            shell = infer_center_shell(
                points,
                center_atom_tolerance=self.center_atom_tolerance,
                shell_min_neighbors=self.shell_min_neighbors,
                shell_max_neighbors=self.shell_max_neighbors,
            )
            center = points[shell.center_idx]
            vectors = points[shell.shell_indices] - center[None, :]
            values = [self._ql(vectors, l) for l in self.l_values]
            if self.append_shell_size:
                values.append(float(shell.shell_indices.size))
            rows.append(np.asarray(values, dtype=np.float32))
            if not np.isfinite(rows[-1]).all():
                raise ValueError(
                    "Steinhardt baseline produced non-finite features: "
                    f"sample_idx={sample_idx}, values={rows[-1].tolist()}."
                )
        return np.vstack(rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "l_values": list(self.l_values),
            "append_shell_size": self.append_shell_size,
            "shell_min_neighbors": self.shell_min_neighbors,
            "shell_max_neighbors": self.shell_max_neighbors,
        }


class SOAPDescriptorBaseline(DescriptorBaseline):
    requires_fit = True

    def __init__(
        self,
        *,
        species: str,
        point_scale: float,
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        r_cut: float | None,
        r_cut_multiplier: float,
        r_cut_min: float,
        n_max: int,
        l_max: int,
        sigma: float,
        pca_components: int | None,
        fit_max_pointclouds: int,
        n_jobs: int,
    ) -> None:
        self.species = str(species)
        self.point_scale = float(point_scale)
        if self.point_scale <= 0.0:
            raise ValueError(f"SOAP point_scale must be > 0, got {self.point_scale}.")
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.requested_r_cut = None if r_cut is None else float(r_cut)
        self.r_cut_multiplier = float(r_cut_multiplier)
        self.r_cut_min = float(r_cut_min)
        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.sigma = float(sigma)
        self.pca_components = None if pca_components is None else int(pca_components)
        self.fit_max_pointclouds = int(fit_max_pointclouds)
        self.n_jobs = int(n_jobs)
        self.soap: Any | None = None
        self.pca: PCA | None = None
        self.effective_r_cut: float | None = None

    def _scale_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * self.point_scale

    @staticmethod
    def _build_soap_descriptor(
        *,
        species: str,
        r_cut: float,
        n_max: int,
        l_max: int,
        sigma: float,
    ):
        try:
            from src.training_methods.SOAP.predict_soap_pca import build_soap
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SOAP baseline requires the optional dependencies 'dscribe' and 'ase'. "
                "Install them before running descriptor.name=soap."
            ) from exc
        return build_soap(
            species=[species],
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=False,
            compression_mode="off",
            dtype="float64",
        )

    def _infer_r_cut(self, point_clouds: np.ndarray) -> float:
        if self.requested_r_cut is not None:
            if self.requested_r_cut <= 0.0:
                raise ValueError(f"SOAP r_cut must be > 0, got {self.requested_r_cut}.")
            return float(self.requested_r_cut)

        if self.fit_max_pointclouds <= 0:
            raise ValueError(
                "SOAP baseline requires fit_max_pointclouds > 0 when r_cut is not explicitly configured."
            )
        shell_cutoffs: list[float] = []
        max_clouds = min(int(point_clouds.shape[0]), int(self.fit_max_pointclouds))
        for idx in range(max_clouds):
            scaled_points = self._scale_points(point_clouds[idx])
            shell = infer_center_shell(
                scaled_points,
                center_atom_tolerance=self.center_atom_tolerance,
                shell_min_neighbors=self.shell_min_neighbors,
                shell_max_neighbors=self.shell_max_neighbors,
            )
            shell_cutoffs.append(float(shell.cutoff))
        if not shell_cutoffs:
            raise RuntimeError("Failed to infer any shell cutoff while estimating SOAP r_cut.")
        r_cut = max(float(np.median(shell_cutoffs)) * self.r_cut_multiplier, self.r_cut_min)
        if not np.isfinite(r_cut) or r_cut <= 0.0:
            raise ValueError(
                f"Estimated an invalid SOAP r_cut={r_cut!r} from shell cutoffs {shell_cutoffs[:10]!r}."
            )
        return float(r_cut)

    def _center_soap_vector(self, points: np.ndarray) -> np.ndarray:
        if self.soap is None:
            raise RuntimeError("SOAP baseline has not been fitted yet.")
        try:
            from ase import Atoms
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SOAP baseline requires the optional dependency 'ase'. "
                "Install it before running descriptor.name=soap."
            ) from exc
        scaled_points = self._scale_points(points)
        shell = infer_center_shell(
            scaled_points,
            center_atom_tolerance=self.center_atom_tolerance,
            shell_min_neighbors=self.shell_min_neighbors,
            shell_max_neighbors=self.shell_max_neighbors,
        )
        atoms = Atoms(
            symbols=[self.species] * int(scaled_points.shape[0]),
            positions=np.asarray(scaled_points, dtype=np.float64),
        )
        raw = self.soap.create(atoms, centers=[int(shell.center_idx)], n_jobs=self.n_jobs)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                "Expected SOAP.create(..., centers=[center_idx]) to return shape (1, F), "
                f"got {tuple(arr.shape)}."
            )
        return arr[0]

    def fit(self, point_clouds: np.ndarray) -> None:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if pcs.shape[0] <= 0:
            raise ValueError("SOAP baseline received zero training point clouds.")
        r_cut = self._infer_r_cut(pcs)
        if r_cut <= 1.0:
            raise ValueError(
                "SOAP baseline requires r_cut > 1.0 for DScribe's Gaussian basis, but got "
                f"r_cut={r_cut:.6f}. Increase descriptor.soap.point_scale or set descriptor.soap.r_cut explicitly."
            )
        self.effective_r_cut = float(r_cut)
        self.soap = self._build_soap_descriptor(
            species=self.species,
            r_cut=float(r_cut),
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
        )

        max_clouds = int(pcs.shape[0]) if self.fit_max_pointclouds <= 0 else min(int(pcs.shape[0]), self.fit_max_pointclouds)
        rows = [self._center_soap_vector(pcs[idx]) for idx in range(max_clouds)]
        if not rows:
            raise RuntimeError("SOAP baseline could not build any training descriptor rows.")
        X = np.vstack(rows)
        requested_components = X.shape[1] if self.pca_components is None else int(self.pca_components)
        n_components = min(int(requested_components), int(X.shape[0]), int(X.shape[1]))
        if n_components <= 0:
            raise ValueError(
                "SOAP PCA resolved to zero components: "
                f"requested={requested_components}, training_shape={tuple(X.shape)}."
            )
        self.pca = PCA(n_components=n_components, whiten=False, random_state=0)
        self.pca.fit(X)

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if self.soap is None or self.pca is None:
            raise RuntimeError("SOAP baseline must be fitted before calling transform().")
        rows = [self._center_soap_vector(points) for points in pcs]
        X = np.vstack(rows)
        return np.asarray(self.pca.transform(X), dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "point_scale": self.point_scale,
            "effective_r_cut": self.effective_r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
            "pca_components": None if self.pca is None else int(self.pca.n_components_),
            "fit_max_pointclouds": self.fit_max_pointclouds,
        }


class CNADescriptorBaseline(DescriptorBaseline):
    requires_fit = True

    def __init__(
        self,
        *,
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        max_signatures: int,
        append_shell_size: bool,
        fit_max_pointclouds: int,
    ) -> None:
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.max_signatures = int(max_signatures)
        self.append_shell_size = bool(append_shell_size)
        self.fit_max_pointclouds = int(fit_max_pointclouds)
        self.signature_vocab: list[str] = []

    @staticmethod
    def _longest_chain_length(nodes: Sequence[int], adjacency: dict[int, set[int]]) -> int:
        if not nodes:
            return 0
        node_set = {int(v) for v in nodes}
        seen: set[int] = set()
        best = 0

        # Use the maximum geodesic distance within each connected component of the
        # common-neighbor bond graph. For CNA motifs this preserves the intended
        # chain/ring ordering while keeping runtime bounded on noisy dense graphs.
        for start in sorted(node_set):
            if start in seen:
                continue
            component: list[int] = []
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in node_set or neighbor in seen:
                        continue
                    seen.add(neighbor)
                    stack.append(neighbor)

            if len(component) <= 1:
                continue

            component_set = set(component)
            for source in component:
                distances = {source: 0}
                queue = [source]
                for current in queue:
                    cur_dist = distances[current]
                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in component_set or neighbor in distances:
                            continue
                        distances[neighbor] = cur_dist + 1
                        queue.append(neighbor)
                best = max(best, max(distances.values(), default=0))

        return int(best)

    def _signature_counts(self, points: np.ndarray) -> tuple[Counter, int]:
        shell = infer_center_shell(
            points,
            center_atom_tolerance=self.center_atom_tolerance,
            shell_min_neighbors=self.shell_min_neighbors,
            shell_max_neighbors=self.shell_max_neighbors,
        )
        local_indices = np.concatenate(
            [
                np.asarray([int(shell.center_idx)], dtype=np.int64),
                np.asarray(shell.shell_indices, dtype=np.int64),
            ]
        )
        local_points = np.asarray(points[local_indices], dtype=np.float64)
        pairwise = np.linalg.norm(
            local_points[:, None, :] - local_points[None, :, :],
            axis=-1,
        )
        within_cutoff = pairwise <= float(shell.cutoff)
        adjacency: dict[int, set[int]] = {}
        for idx in range(local_points.shape[0]):
            neighbor_idx = set(np.flatnonzero(within_cutoff[idx]).tolist())
            neighbor_idx.discard(idx)
            adjacency[int(idx)] = neighbor_idx

        center_idx = 0
        shell_neighbor_set = set(range(1, int(local_points.shape[0])))
        counts: Counter = Counter()
        for neighbor_idx in sorted(shell_neighbor_set):
            common = sorted(adjacency[center_idx].intersection(adjacency[neighbor_idx]))
            n_common = len(common)
            common_set = set(common)
            subgraph: dict[int, set[int]] = {
                node: adjacency[node].intersection(common_set)
                for node in common
            }
            n_bonds = int(sum(len(neigh) for neigh in subgraph.values()) // 2)
            longest_chain = self._longest_chain_length(common, subgraph)
            signature = f"{n_common}-{n_bonds}-{longest_chain}"
            counts[signature] += 1

        if sum(counts.values()) != len(shell_neighbor_set):
            raise RuntimeError(
                "CNA signature counting mismatch: "
                f"counted_bonds={sum(counts.values())}, shell_size={len(shell_neighbor_set)}."
            )
        return counts, len(shell_neighbor_set)

    def fit(self, point_clouds: np.ndarray) -> None:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        max_clouds = int(pcs.shape[0]) if self.fit_max_pointclouds <= 0 else min(int(pcs.shape[0]), self.fit_max_pointclouds)
        global_counts: Counter = Counter()
        for idx in range(max_clouds):
            counts, _ = self._signature_counts(pcs[idx])
            global_counts.update(counts)
        if not global_counts:
            raise RuntimeError("CNA baseline did not observe any signatures during fit().")
        most_common = global_counts.most_common(self.max_signatures)
        self.signature_vocab = [signature for signature, _ in most_common]

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if not self.signature_vocab:
            raise RuntimeError("CNA baseline must be fitted before calling transform().")

        rows: list[np.ndarray] = []
        for sample_idx, points in enumerate(pcs):
            counts, shell_size = self._signature_counts(points)
            total = max(1, sum(counts.values()))
            values = np.zeros(len(self.signature_vocab) + 1 + int(self.append_shell_size), dtype=np.float32)
            for sig_idx, signature in enumerate(self.signature_vocab):
                values[sig_idx] = float(counts.get(signature, 0)) / float(total)
            other_count = total - sum(int(counts.get(signature, 0)) for signature in self.signature_vocab)
            values[len(self.signature_vocab)] = float(other_count) / float(total)
            if self.append_shell_size:
                values[-1] = float(shell_size)
            if not np.isfinite(values).all():
                raise ValueError(
                    "CNA baseline produced non-finite features: "
                    f"sample_idx={sample_idx}, values={values.tolist()}."
                )
            rows.append(values)
        return np.vstack(rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "signature_vocab": list(self.signature_vocab),
            "max_signatures": self.max_signatures,
            "append_shell_size": self.append_shell_size,
            "fit_max_pointclouds": self.fit_max_pointclouds,
        }


def build_descriptor_from_cfg(cfg) -> DescriptorBaseline:
    descriptor_cfg = getattr(cfg, "descriptor", None)
    if descriptor_cfg is None:
        raise ValueError("Config must define a 'descriptor' section.")
    descriptor_name = str(getattr(descriptor_cfg, "name", "")).strip().lower()
    center_atom_tolerance = float(getattr(descriptor_cfg, "center_atom_tolerance", 1e-6))
    shell_min_neighbors = int(getattr(descriptor_cfg, "shell_min_neighbors", 8))
    shell_max_neighbors = int(getattr(descriptor_cfg, "shell_max_neighbors", 24))
    if shell_min_neighbors < 2:
        raise ValueError(f"descriptor.shell_min_neighbors must be >= 2, got {shell_min_neighbors}.")
    if shell_max_neighbors <= shell_min_neighbors:
        raise ValueError(
            "descriptor.shell_max_neighbors must exceed shell_min_neighbors, "
            f"got {shell_max_neighbors} <= {shell_min_neighbors}."
        )

    if descriptor_name == "steinhardt":
        steinhardt_cfg = getattr(descriptor_cfg, "steinhardt", None)
        return SteinhardtDescriptorBaseline(
            l_values=list(getattr(steinhardt_cfg, "l_values", [4, 6, 8, 10, 12])),
            center_atom_tolerance=center_atom_tolerance,
            shell_min_neighbors=shell_min_neighbors,
            shell_max_neighbors=shell_max_neighbors,
            append_shell_size=bool(getattr(steinhardt_cfg, "append_shell_size", True)),
        )

    if descriptor_name == "soap":
        soap_cfg = getattr(descriptor_cfg, "soap", None)
        point_scale = getattr(soap_cfg, "point_scale", None)
        if point_scale is None:
            data_cfg = getattr(cfg, "data", None)
            normalize = bool(getattr(data_cfg, "normalize", False)) if data_cfg is not None else False
            pre_normalize = bool(getattr(data_cfg, "pre_normalize", False)) if data_cfg is not None else False
            radius = getattr(data_cfg, "radius", None) if data_cfg is not None else None
            if normalize and pre_normalize and radius is not None:
                point_scale = float(radius)
            else:
                point_scale = 1.0
        return SOAPDescriptorBaseline(
            species=str(getattr(soap_cfg, "species", "Al")),
            point_scale=float(point_scale),
            center_atom_tolerance=center_atom_tolerance,
            shell_min_neighbors=shell_min_neighbors,
            shell_max_neighbors=shell_max_neighbors,
            r_cut=getattr(soap_cfg, "r_cut", None),
            r_cut_multiplier=float(getattr(soap_cfg, "r_cut_multiplier", 1.25)),
            r_cut_min=float(getattr(soap_cfg, "r_cut_min", 0.35)),
            n_max=int(getattr(soap_cfg, "n_max", 8)),
            l_max=int(getattr(soap_cfg, "l_max", 6)),
            sigma=float(getattr(soap_cfg, "sigma", 0.3)),
            pca_components=getattr(soap_cfg, "pca_components", 32),
            fit_max_pointclouds=int(getattr(soap_cfg, "fit_max_pointclouds", 4000)),
            n_jobs=int(getattr(soap_cfg, "n_jobs", 1)),
        )

    if descriptor_name == "cna":
        cna_cfg = getattr(descriptor_cfg, "cna", None)
        return CNADescriptorBaseline(
            center_atom_tolerance=center_atom_tolerance,
            shell_min_neighbors=shell_min_neighbors,
            shell_max_neighbors=shell_max_neighbors,
            max_signatures=int(getattr(cna_cfg, "max_signatures", 12)),
            append_shell_size=bool(getattr(cna_cfg, "append_shell_size", True)),
            fit_max_pointclouds=int(getattr(cna_cfg, "fit_max_pointclouds", 4000)),
        )

    raise ValueError(
        f"Unsupported descriptor baseline {descriptor_name!r}. Expected one of ['steinhardt', 'soap', 'cna']."
    )


def run_descriptor_baseline(cfg, *, output_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data_utils.data_module import PointCloudDataModule
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Running descriptor baselines requires the training/data dependencies, including "
            "'pytorch_lightning'. Install the project runtime environment before executing the baseline runner."
        ) from exc

    dm = PointCloudDataModule(cfg)
    dm.setup("fit")
    dm_impl = getattr(dm, "impl", dm)
    train_dataset = getattr(dm, "train_dataset", None)
    if train_dataset is None:
        train_dataset = getattr(dm_impl, "train_dataset", None)
    val_dataset = getattr(dm, "val_dataset", None)
    if val_dataset is None:
        val_dataset = getattr(dm_impl, "val_dataset", None)
    test_dataset = getattr(dm, "test_dataset", None)
    if test_dataset is None:
        test_dataset = getattr(dm_impl, "test_dataset", None)
    if train_dataset is None or val_dataset is None or test_dataset is None:
        raise RuntimeError(
            "PointCloudDataModule did not expose train/val/test datasets after setup('fit')."
        )

    settings = load_eval_settings_from_cfg(cfg)
    descriptor = build_descriptor_from_cfg(cfg)

    fit_train_points = None
    if descriptor.requires_fit:
        fit_max = None
        if hasattr(descriptor, "fit_max_pointclouds"):
            fit_max = int(getattr(descriptor, "fit_max_pointclouds"))
        fit_train_points, _ = collect_split_point_clouds_and_labels(train_dataset, max_samples=fit_max)
        descriptor.fit(fit_train_points)

    val_points, val_labels = collect_split_point_clouds_and_labels(
        val_dataset,
        max_samples=settings.max_supervised_samples,
    )
    test_points, test_labels = collect_split_point_clouds_and_labels(
        test_dataset,
        max_samples=settings.max_test_samples,
    )

    val_features = descriptor.transform(val_points)
    test_features = descriptor.transform(test_points)
    val_dataset_k = dataset_class_count(val_dataset)
    test_dataset_k = dataset_class_count(test_dataset)

    def rotated_test_feature_fn(run_idx: int) -> tuple[np.ndarray, np.ndarray]:
        seed = int(settings.test_so3_rotation_seed) + int(run_idx) * 100000
        rotated_points = rotate_point_cloud_batch(test_points, seed=seed)
        return descriptor.transform(rotated_points), test_labels

    val_metrics = compute_supervised_stage_metrics_from_features(
        stage="val",
        features=val_features,
        labels=val_labels,
        dataset_k=val_dataset_k,
        settings=settings,
        rotated_feature_fn=None,
    )
    test_metrics = compute_supervised_stage_metrics_from_features(
        stage="test",
        features=test_features,
        labels=test_labels,
        dataset_k=test_dataset_k,
        settings=settings,
        rotated_feature_fn=rotated_test_feature_fn,
    )

    final_metrics: dict[str, float] = {}
    final_metrics.update(_prefix_stage_class_metrics("val", val_metrics))
    final_metrics.update(_prefix_stage_class_metrics("test", test_metrics))

    summary = {
        "descriptor_name": str(getattr(cfg.descriptor, "name", "")),
        "descriptor_metadata": descriptor.metadata(),
        "eval_settings": {
            "max_supervised_samples": settings.max_supervised_samples,
            "max_test_samples": settings.max_test_samples,
            "val_cluster_eval_k": settings.val_cluster_eval_k,
            "cluster_acc_seed": settings.cluster_acc_seed,
            "val_cluster_acc_methods": list(settings.val_cluster_acc_methods),
            "test_cluster_acc_methods": list(settings.test_cluster_acc_methods),
            "test_so3_rotation_runs": settings.test_so3_rotation_runs,
            "test_so3_rotation_seed": settings.test_so3_rotation_seed,
        },
        "split_sizes": {
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)),
            "test": int(len(test_dataset)),
            "fit_train_samples": None if fit_train_points is None else int(fit_train_points.shape[0]),
            "val_eval_samples": int(val_points.shape[0]),
            "test_eval_samples": int(test_points.shape[0]),
        },
        "class_counts": {
            "train": dataset_class_count(train_dataset),
            "val": val_dataset_k,
            "test": test_dataset_k,
        },
    }

    final_metrics_path = output_dir / "final_metrics.json"
    final_metrics_path.write_text(json.dumps(final_metrics, indent=2, sort_keys=True))
    metadata_path = output_dir / "descriptor_baseline_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return final_metrics, summary
