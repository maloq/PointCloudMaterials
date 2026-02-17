import warnings

import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader

from src.utils.spd_metrics import compute_cluster_metrics, compute_embedding_quality_metrics
from src.utils.spd_utils import cached_sample_count


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
                f"Ignoring non-integer run count for key '{k}': {raw!r}",
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
    if not any(_is_kmeans_plus_plus_method(m) for m in out):
        out.append("kmeans++")
    return out


def _flatten_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _prepare_features_and_labels(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = _flatten_features(features)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] == 0:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    n = min(int(x.shape[0]), int(y.shape[0]))
    if n <= 0:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x = x[:n]
    y = y[:n]
    valid = np.isfinite(x).all(axis=1)
    if np.issubdtype(y.dtype, np.floating):
        valid &= np.isfinite(y)
    x = x[valid]
    y = y[valid]
    return x, y


def _compute_linear_svm_accuracy(features: np.ndarray, labels: np.ndarray) -> float | None:
    x, y = _prepare_features_and_labels(features, labels)
    if x.shape[0] <= 1:
        return None

    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        return None

    min_class = int(counts.min())
    if min_class < 2:
        return None
    cv = max(2, min(5, min_class))

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(dual="auto", random_state=42, max_iter=20000),
    )
    try:
        scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy", n_jobs=1)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"Linear SVM cross-val failed: {exc}")
        return None
    if scores.size == 0:
        return None
    return float(np.mean(scores))


def _compute_linear_svm_train_to_test_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    svm_c: float = 0.018,
) -> float | None:
    x_train, y_train = _prepare_features_and_labels(train_features, train_labels)
    x_test, y_test = _prepare_features_and_labels(test_features, test_labels)
    if x_train.shape[0] <= 1 or x_test.shape[0] <= 0:
        return None

    unique_train = np.unique(y_train)
    if unique_train.size < 2:
        return None

    unique_test = np.unique(y_test)
    if unique_test.size < 1:
        return None

    # SVM evaluation protocol aligned with the official RI-MAE code.
    clf = SVC(C=float(svm_c), kernel="linear")
    try:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"Linear SVM train-to-test eval failed: {exc}")
        return None
    if pred.shape[0] == 0:
        return None
    return float(np.mean(pred == y_test))


def _build_supervised_eval_loader(module, split: str) -> DataLoader | None:
    trainer = getattr(module, "trainer", None)
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        return None

    dataset = getattr(datamodule, f"{split}_dataset", None)
    if dataset is None and hasattr(datamodule, "impl"):
        dataset = getattr(datamodule.impl, f"{split}_dataset", None)

    if dataset is None:
        dataloader_fn = getattr(datamodule, f"{split}_dataloader", None)
        if callable(dataloader_fn):
            try:
                return dataloader_fn()
            except (TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(f"Dataloader construction failed for split '{split}': {exc}")
                return None
        return None

    batch_size = int(getattr(module.hparams, "batch_size", 64))
    num_workers = int(getattr(module.hparams, "num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, num_workers),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _collect_split_supervised_features(
    module,
    split: str,
    *,
    max_samples: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    extractor = getattr(module, "_extract_supervised_features_from_batch", None)
    if not callable(extractor):
        return None, None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            # Avoid duplicated full-dataset passes on every rank.
            return None, None

    loader = _build_supervised_eval_loader(module, split)
    if loader is None:
        return None, None

    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    collected = 0
    limit = None if max_samples is None or int(max_samples) <= 0 else int(max_samples)

    was_training = bool(module.training)
    module.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                features, labels = extractor(batch)
                if features is None or labels is None:
                    continue

                if not torch.is_tensor(features):
                    features = torch.as_tensor(features)
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels)

                features = features.detach().to(torch.float32)
                if features.dim() == 1:
                    features = features.unsqueeze(-1)
                elif features.dim() > 2:
                    features = features.reshape(features.shape[0], -1)
                labels = labels.detach().view(-1).to(torch.long)

                take = min(int(features.shape[0]), int(labels.shape[0]))
                if limit is not None:
                    take = min(take, int(limit - collected))
                if take <= 0:
                    break

                feature_parts.append(features[:take].cpu())
                label_parts.append(labels[:take].cpu())
                collected += int(take)
                if limit is not None and collected >= limit:
                    break
    finally:
        if was_training:
            module.train()

    if not feature_parts or not label_parts:
        return None, None
    features_np = torch.cat(feature_parts, dim=0).numpy()
    labels_np = torch.cat(label_parts, dim=0).numpy()
    return features_np, labels_np


def _compute_train_to_test_svm_accuracy(
    module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float | None:
    if not bool(getattr(module, "enable_train_split_svm_test_metric", True)):
        return None

    max_train_samples = int(getattr(module, "train_split_svm_max_samples", 0) or 0)
    if max_train_samples <= 0:
        max_train_samples = int(getattr(module, "max_supervised_samples", 8192) or 8192)

    train_features, train_labels = _collect_split_supervised_features(
        module,
        "train",
        max_samples=max_train_samples,
    )
    if train_features is None or train_labels is None:
        return None

    svm_c = float(getattr(module, "train_split_svm_c", 0.018))
    return _compute_linear_svm_train_to_test_accuracy(
        train_features,
        train_labels,
        test_features,
        test_labels,
        svm_c=svm_c,
    )


def _infer_stage_acc_settings(cfg, stage: str) -> tuple[list[str], int, dict[str, int]]:
    stage_l = str(stage).lower()
    if stage_l == "val":
        methods = _as_string_list(
            getattr(cfg, "val_cluster_acc_methods", None),
            default=["kmeans", "kmeans++"],
        )
        methods = _ensure_kmeans_plus_plus_method(methods)
        runs = int(getattr(cfg, "val_cluster_acc_runs", 1))
        runs_by_method = _as_int_mapping(
            getattr(cfg, "val_cluster_acc_runs_by_method", None),
            default={},
        )
        return methods, max(1, runs), runs_by_method

    if stage_l == "test":
        methods = _as_string_list(
            getattr(cfg, "test_cluster_acc_methods", None),
            default=["kmeans", "kmeans++", "gmm_full"],
        )
        methods = _ensure_kmeans_plus_plus_method(methods)
        runs = int(getattr(cfg, "test_cluster_acc_runs", 1))
        runs_by_method = _as_int_mapping(
            getattr(cfg, "test_cluster_acc_runs_by_method", None),
            default={"kmeans": 20},
        )
        return methods, max(1, runs), runs_by_method

    return [], 1, {}


def _infer_label_cluster_count(labels: np.ndarray) -> int | None:
    if labels is None:
        return None
    arr = np.asarray(labels).reshape(-1)
    if arr.size == 0:
        return None
    k = int(np.unique(arr).size)
    return k if k > 1 else None


def _get_stage_dataset(module, stage: str):
    trainer = getattr(module, "trainer", None)
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        return None

    split_name = f"{stage}_dataset"
    dataset = getattr(datamodule, split_name, None)
    if dataset is None and hasattr(datamodule, "impl"):
        dataset = getattr(datamodule.impl, split_name, None)
    while dataset is not None and hasattr(dataset, "dataset"):
        inner = getattr(dataset, "dataset")
        if inner is dataset:
            break
        dataset = inner
    return dataset


def _infer_dataset_class_count(module, stage: str) -> int | None:
    dataset = _get_stage_dataset(module, stage)
    if dataset is None:
        return None

    class_names = getattr(dataset, "class_names", None)
    if class_names is not None:
        if hasattr(class_names, "items"):
            k = len(dict(class_names))
            return k if k > 1 else None
        if hasattr(class_names, "__len__") and not isinstance(class_names, (str, bytes)):
            k = len(class_names)
            return k if k > 1 else None

    num_classes = getattr(dataset, "num_classes", None)
    if callable(num_classes):
        value = num_classes()
    else:
        value = num_classes
    if value is not None:
        try:
            k = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {stage}_dataset.num_classes={value!r}; expected integer-like value."
            ) from None
        return k if k > 1 else None
    return None


def _resolve_hungarian_eval_k(module, stage: str, labels: np.ndarray) -> int | None:
    stage_l = str(stage).lower()
    if stage_l not in {"val", "test"}:
        return None

    observed_k = _infer_label_cluster_count(labels)
    dataset_k = _infer_dataset_class_count(module, stage_l)

    if dataset_k is not None:
        if observed_k is not None and observed_k != dataset_k:
            raise ValueError(
                "Hungarian ACC class-count mismatch: "
                f"observed {observed_k} unique labels in cached {stage_l} embeddings, "
                f"but dataset reports {dataset_k} classes. "
                "This typically means sampling limits dropped classes; increase "
                f"{'max_test_samples' if stage_l == 'test' else 'max_supervised_samples'} "
                "or disable that limit."
            )
        inferred_k = dataset_k
    else:
        inferred_k = observed_k

    configured = getattr(module, "val_cluster_eval_k", None) if stage_l == "val" else None
    if configured is not None and inferred_k is not None and int(configured) != int(inferred_k):
        raise ValueError(
            "Hungarian ACC k must match the class count. "
            f"Configured val_cluster_eval_k={int(configured)}, inferred classes={int(inferred_k)} "
            f"for stage='{stage_l}'. Set val_cluster_eval_k=null (recommended) or to {int(inferred_k)}."
        )
    return inferred_k


def init_supervised_cache(module, cfg) -> None:
    module._supervised_cache = {
        "train": {"latents": [], "encoder_features": [], "class_id": []},
        "val": {"latents": [], "encoder_features": [], "class_id": []},
        "test": {"latents": [], "encoder_features": [], "class_id": []},
    }
    module.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, "max_supervised_samples") else 8192
    module.max_test_samples = cfg.max_test_samples if hasattr(cfg, "max_test_samples") else 1000
    module.val_cluster_eval_k = _parse_optional_eval_k(
        getattr(cfg, "val_cluster_eval_k", None),
        field_name="val_cluster_eval_k",
    )
    module.cluster_acc_seed = int(getattr(cfg, "cluster_acc_seed", 0))
    module.enable_train_split_svm_test_metric = bool(
        getattr(cfg, "enable_train_split_svm_test_metric", True)
    )
    module.train_split_svm_max_samples = int(
        getattr(cfg, "train_split_svm_max_samples", 0) or 0
    )
    module.train_split_svm_c = float(getattr(cfg, "train_split_svm_c", 0.018))
    (
        module.val_cluster_acc_methods,
        module.val_cluster_acc_runs,
        module.val_cluster_acc_runs_by_method,
    ) = _infer_stage_acc_settings(cfg, "val")
    (
        module.test_cluster_acc_methods,
        module.test_cluster_acc_runs,
        module.test_cluster_acc_runs_by_method,
    ) = _infer_stage_acc_settings(cfg, "test")


def reset_supervised_cache(module, stage: str) -> None:
    cache = module._supervised_cache.get(stage)
    if cache is None:
        return
    for key in cache:
        cache[key].clear()


def cache_limit_for_stage(module, stage: str):
    if stage == "test":
        return module.max_test_samples
    if stage in {"train", "val"}:
        return module.max_supervised_samples
    return None


def cache_supervised_batch(
    module,
    stage: str,
    z_inv: torch.Tensor,
    meta: dict,
    encoder_features: torch.Tensor | None = None,
) -> None:
    cache = module._supervised_cache.get(stage)
    if cache is None:
        return

    limit = cache_limit_for_stage(module, stage)
    remaining = None
    if limit is not None and limit > 0:
        cached = cached_sample_count(cache)
        remaining = int(limit - cached)
        if remaining <= 0:
            return

    if z_inv is None:
        return

    batch_size = int(z_inv.shape[0])
    effective_batch = batch_size if remaining is None else min(batch_size, remaining)
    if effective_batch <= 0:
        return

    class_id = meta.get("class_id")
    if class_id is None:
        return
    if not torch.is_tensor(class_id):
        class_id = torch.as_tensor(class_id)
    class_id = class_id.detach().view(-1)
    effective_batch = min(effective_batch, class_id.shape[0])
    if encoder_features is not None:
        if not torch.is_tensor(encoder_features):
            encoder_features = torch.as_tensor(encoder_features)
        enc = encoder_features.detach().to(torch.float32)
        if enc.dim() == 1:
            enc = enc.unsqueeze(-1)
        elif enc.dim() > 2:
            enc = enc.reshape(enc.shape[0], -1)
        effective_batch = min(effective_batch, enc.shape[0])
    else:
        enc = None
    if effective_batch <= 0:
        return

    cache["latents"].append(z_inv[:effective_batch].detach().to(torch.float32).cpu())
    if enc is not None:
        cache["encoder_features"].append(enc[:effective_batch].cpu())
    cache["class_id"].append(class_id[:effective_batch].cpu())


def log_supervised_metrics(module, stage: str) -> None:
    cache = module._supervised_cache.get(stage)
    if cache is None:
        return

    if not cache["latents"] or not cache["class_id"]:
        for key in cache:
            cache[key].clear()
        return

    latents = torch.cat(cache["latents"], dim=0).numpy()
    labels = torch.cat(cache["class_id"], dim=0).numpy()
    encoder_features = None
    if cache.get("encoder_features"):
        encoder_features = torch.cat(cache["encoder_features"], dim=0).numpy()
    latents, labels, encoder_features = _gather_latents_labels_ddp(latents, labels, encoder_features)

    stage_l = str(stage).lower()
    if stage_l == "val":
        acc_methods = _ensure_kmeans_plus_plus_method(
            list(getattr(module, "val_cluster_acc_methods", []))
        )
        acc_runs = int(getattr(module, "val_cluster_acc_runs", 1))
        acc_runs_by_method = getattr(module, "val_cluster_acc_runs_by_method", {})
    elif stage_l == "test":
        acc_methods = _ensure_kmeans_plus_plus_method(
            list(getattr(module, "test_cluster_acc_methods", ["kmeans", "kmeans++", "gmm_full"]))
        )
        acc_runs = int(getattr(module, "test_cluster_acc_runs", 1))
        acc_runs_by_method = getattr(module, "test_cluster_acc_runs_by_method", {"kmeans": 20})
    else:
        acc_methods = []
        acc_runs = 1
        acc_runs_by_method = {}

    hungarian_eval_k = _resolve_hungarian_eval_k(module, stage_l, labels)
    metrics = compute_cluster_metrics(
        latents,
        labels,
        stage,
        hungarian_eval_k=hungarian_eval_k,
        acc_eval_methods=acc_methods,
        acc_eval_runs=max(1, acc_runs),
        acc_eval_runs_by_method=acc_runs_by_method,
        acc_random_seed=int(getattr(module, "cluster_acc_seed", 0)),
    ) or {}
    if stage_l in {"val", "test"} and encoder_features is not None:
        svm_acc = _compute_linear_svm_accuracy(encoder_features, labels)
        if svm_acc is not None:
            metrics["ENCODER_LINEAR_SVM_ACCURACY"] = svm_acc
    if stage_l == "test":
        svm_eval_features = encoder_features if encoder_features is not None else latents
        train_to_test_svm = _compute_train_to_test_svm_accuracy(
            module,
            svm_eval_features,
            labels,
        )
        if train_to_test_svm is not None:
            metrics["ENCODER_LINEAR_SVM_TRAIN_TO_TEST_ACCURACY"] = train_to_test_svm
    if metrics:
        # class/* metrics: clustering/classification quality against class_id labels.
        for name, value in metrics.items():
            module._log_metric(
                stage,
                f"class/{name.lower()}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    emb_metrics = compute_embedding_quality_metrics(latents, labels, include_expensive=(stage == "test"))
    for name, value in emb_metrics.items():
        module._log_metric(
            stage,
            f"embedding/{name}",
            value,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    for key in cache:
        cache[key].clear()


def _gather_latents_labels_ddp(
    latents: np.ndarray,
    labels: np.ndarray,
    encoder_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Gather cached latent/label arrays across DDP ranks for global metrics."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return latents, labels, encoder_features

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return latents, labels, encoder_features

    payload = {
        "latents": np.asarray(latents, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    if encoder_features is not None:
        payload["encoder_features"] = np.asarray(encoder_features, dtype=np.float32)
    gathered: list[dict[str, np.ndarray] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, payload)

    latents_parts: list[np.ndarray] = []
    encoder_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for item in gathered:
        if not item:
            continue
        part_lat = np.asarray(item.get("latents", []), dtype=np.float32)
        part_lab = np.asarray(item.get("labels", []), dtype=np.int64)
        part_enc = np.asarray(item.get("encoder_features", []), dtype=np.float32)
        if part_lat.ndim != 2 or part_lab.ndim != 1 or part_lat.shape[0] == 0:
            continue
        n = min(part_lat.shape[0], part_lab.shape[0])
        if part_enc.ndim == 2 and part_enc.shape[0] > 0:
            n = min(n, part_enc.shape[0])
        if n <= 0:
            continue
        latents_parts.append(part_lat[:n])
        labels_parts.append(part_lab[:n])
        if part_enc.ndim == 2 and part_enc.shape[0] >= n:
            encoder_parts.append(part_enc[:n])

    if not latents_parts:
        return latents, labels, encoder_features

    gathered_latents = np.concatenate(latents_parts, axis=0)
    gathered_labels = np.concatenate(labels_parts, axis=0)
    if len(encoder_parts) == len(latents_parts):
        gathered_enc = np.concatenate(encoder_parts, axis=0)
    else:
        gathered_enc = None
    return gathered_latents, gathered_labels, gathered_enc
