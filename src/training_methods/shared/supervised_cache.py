"""Epoch cache ownership and logging for supervised evaluation."""

import warnings

import torch

from src.utils.evaluation_metrics import ensure_kmeansplusplus
from .optimizers import cached_sample_count
from .supervised_features import cache_limit_for_stage, _gather_latents_labels_ddp
from .supervised_metrics import (
    _validate_cached_supervised_arrays,
    evaluate_supervised_features,
)


def _as_string_list(value, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_int_mapping(value, default: dict[str, int]) -> dict[str, int]:
    if value is None:
        return dict(default)
    return {str(k): int(v) for k, v in value.items()}


def _parse_optional_eval_k(value, *, field_name: str) -> int | None:
    return int(value) if value is not None else None


def _infer_stage_acc_settings(cfg, stage: str) -> tuple[list[str], int, dict[str, int]]:
    stage_l = str(stage).lower()
    if stage_l not in {"val", "test"}:
        return [], 1, {}
    methods = _as_string_list(
        getattr(cfg, f"{stage_l}_cluster_acc_methods", None), default=["kmeans++"]
    )
    runs = int(getattr(cfg, f"{stage_l}_cluster_acc_runs", 1))
    runs_by_method = _as_int_mapping(
        getattr(cfg, f"{stage_l}_cluster_acc_runs_by_method", None), default={}
    )
    return ensure_kmeansplusplus(methods), max(1, runs), runs_by_method


def init_supervised_cache(module, cfg) -> None:
    module.enable_supervised_metrics = bool(
        getattr(cfg, "enable_supervised_metrics", True)
    )
    module.enable_embedding_metrics = bool(
        getattr(cfg, "enable_embedding_metrics", False)
    )
    module.enable_probe_metrics = bool(getattr(cfg, "enable_probe_metrics", True))
    module.representation_source = (
        str(getattr(cfg, "representation_source", "encoder")).strip().lower()
    )
    valid_representation_sources = {"encoder", "vicreg_projector"}
    if module.representation_source not in valid_representation_sources:
        raise ValueError(
            "representation_source must be one of"
            f" {sorted(valid_representation_sources)}, got"
            f" {module.representation_source!r}."
        )
    valid_metric_stages = {"train", "val", "test"}
    for field_name in (
        "supervised_metric_stages",
        "embedding_metric_stages",
        "probe_metric_stages",
    ):
        raw_stages = getattr(cfg, field_name, ("train", "val", "test"))
        if isinstance(raw_stages, str):
            raw_stages = [raw_stages]
        resolved_stages = frozenset(str(stage).strip().lower() for stage in raw_stages)
        invalid_stages = sorted(resolved_stages.difference(valid_metric_stages))
        if invalid_stages:
            raise ValueError(
                f"{field_name} contains unsupported stages {invalid_stages}. "
                f"Expected a subset of {sorted(valid_metric_stages)}."
            )
        setattr(module, field_name, resolved_stages)
    module._supervised_cache = {
        "train": {"latents": [], "encoder_features": [], "class_id": []},
        "val": {"latents": [], "encoder_features": [], "class_id": []},
        "test": {"latents": [], "encoder_features": [], "class_id": []},
    }
    module.max_supervised_samples = (
        cfg.max_supervised_samples if hasattr(cfg, "max_supervised_samples") else 8192
    )
    module.max_test_samples = (
        cfg.max_test_samples if hasattr(cfg, "max_test_samples") else 1000
    )
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
    module.enable_svm_accuracy = bool(getattr(cfg, "enable_svm_accuracy", True))
    module.probe_k = _parse_optional_eval_k(
        getattr(cfg, "probe_k", None),
        field_name="probe_k",
    )
    module.probe_max_samples = int(getattr(cfg, "probe_max_samples", 8192) or 0)
    module.probe_pca_max_components = int(getattr(cfg, "probe_pca_max_components", 64))
    module.probe_silhouette_samples = int(
        getattr(cfg, "probe_silhouette_samples", 5000)
    )
    module.probe_kmeans_seed_count = int(getattr(cfg, "probe_kmeans_seed_count", 3))
    module.probe_kmeans_n_init = int(getattr(cfg, "probe_kmeans_n_init", 5))
    module.probe_seed = int(
        getattr(cfg, "probe_seed", getattr(cfg, "cluster_acc_seed", 0))
    )
    module.probe_eps = float(getattr(cfg, "probe_eps", 1e-12))
    module.probe_best_k_min = int(getattr(cfg, "probe_best_k_min", 2))
    module.probe_best_k_max = _parse_optional_eval_k(
        getattr(cfg, "probe_best_k_max", None),
        field_name="probe_best_k_max",
    )
    if module.probe_k is not None and int(module.probe_k) < 2:
        raise ValueError(f"probe_k must be >= 2, got {module.probe_k}.")
    if module.probe_max_samples < 0:
        raise ValueError(
            f"probe_max_samples must be >= 0, got {module.probe_max_samples}."
        )
    if module.probe_pca_max_components < 1:
        raise ValueError(
            "probe_pca_max_components must be >= 1, got"
            f" {module.probe_pca_max_components}."
        )
    if module.probe_silhouette_samples < 2:
        raise ValueError(
            "probe_silhouette_samples must be >= 2, got"
            f" {module.probe_silhouette_samples}."
        )
    if module.probe_kmeans_seed_count < 2:
        raise ValueError(
            "probe_kmeans_seed_count must be >= 2, got"
            f" {module.probe_kmeans_seed_count}."
        )
    if module.probe_kmeans_n_init < 1:
        raise ValueError(
            f"probe_kmeans_n_init must be >= 1, got {module.probe_kmeans_n_init}."
        )
    if module.probe_eps <= 0.0:
        raise ValueError(f"probe_eps must be > 0, got {module.probe_eps}.")
    if module.probe_best_k_min < 2:
        raise ValueError(
            f"probe_best_k_min must be >= 2, got {module.probe_best_k_min}."
        )
    if (
        module.probe_best_k_max is not None
        and module.probe_best_k_max < module.probe_best_k_min
    ):
        raise ValueError(
            "probe_best_k_max must be >= probe_best_k_min when set. Got"
            f" min={module.probe_best_k_min}, max={module.probe_best_k_max}."
        )
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
    module.enable_test_so3_metrics = bool(getattr(cfg, "enable_test_so3_metrics", True))
    module.test_so3_rotation_runs = int(
        getattr(
            cfg,
            "test_so3_rotation_runs",
            getattr(cfg, "analysis_test_rotation_runs", 5),
        )
    )
    if module.test_so3_rotation_runs < 1:
        raise ValueError(
            f"test_so3_rotation_runs must be >= 1, got {module.test_so3_rotation_runs}."
        )
    module.test_so3_rotation_seed = int(
        getattr(
            cfg,
            "test_so3_rotation_seed",
            getattr(cfg, "analysis_test_rotation_seed", 12345),
        )
    )


def reset_supervised_cache(module, stage: str) -> None:
    cache = module._supervised_cache.get(stage)
    if cache is None:
        return
    for key in cache:
        cache[key].clear()


def cache_supervised_batch(
    module,
    stage: str,
    z_inv_contrastive: torch.Tensor,
    meta: dict,
    encoder_features: torch.Tensor | None = None,
) -> None:
    wants_supervised_metrics = bool(getattr(module, "enable_supervised_metrics", True))
    wants_probe_metrics = bool(getattr(module, "enable_probe_metrics", True))
    if not (wants_supervised_metrics or wants_probe_metrics):
        return

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

    if z_inv_contrastive is None:
        return

    batch_size = int(z_inv_contrastive.shape[0])
    effective_batch = batch_size if remaining is None else min(batch_size, remaining)
    if effective_batch <= 0:
        return

    class_id = meta.get("class_id")
    if class_id is None and not wants_probe_metrics:
        return
    if class_id is not None:
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.detach().view(-1)
        if class_id.shape[0] != batch_size:
            raise ValueError(
                f"Supervised cache label row mismatch: stage={stage!r}, "
                f"latents={batch_size}, labels={class_id.shape[0]}."
            )
    if encoder_features is not None:
        if not torch.is_tensor(encoder_features):
            encoder_features = torch.as_tensor(encoder_features)
        enc = encoder_features.detach().to(torch.float32)
        if enc.dim() == 1:
            enc = enc.unsqueeze(-1)
        elif enc.dim() > 2:
            enc = enc.reshape(enc.shape[0], -1)
        if enc.shape[0] != batch_size:
            raise ValueError(
                f"Supervised cache encoder row mismatch: stage={stage!r}, "
                f"latents={batch_size}, encoder_features={enc.shape[0]}."
            )
    else:
        enc = None
    lat_chunk = z_inv_contrastive[:effective_batch].detach().to(torch.float32)
    if not bool(torch.isfinite(lat_chunk).all()):
        nonfinite = int((~torch.isfinite(lat_chunk)).sum().item())
        raise RuntimeError(
            "Non-finite z_inv_contrastive latents detected while caching"
            f" supervised metrics: stage='{stage}',"
            f" batch_rows={effective_batch},"
            f" latent_shape={tuple(lat_chunk.shape)},"
            f" nonfinite_values={nonfinite}/{lat_chunk.numel()}."
        )
    if enc is not None:
        enc_chunk = enc[:effective_batch]
        if not bool(torch.isfinite(enc_chunk).all()):
            nonfinite = int((~torch.isfinite(enc_chunk)).sum().item())
            raise RuntimeError(
                "Non-finite encoder features detected while caching"
                f" supervised metrics: stage='{stage}',"
                f" batch_rows={effective_batch},"
                f" feature_shape={tuple(enc_chunk.shape)},"
                f" nonfinite_values={nonfinite}/{enc_chunk.numel()}."
            )
    cache["latents"].append(lat_chunk.cpu())
    if enc is not None:
        cache["encoder_features"].append(enc_chunk.cpu())
    if class_id is not None:
        cache["class_id"].append(class_id[:effective_batch].cpu())


def log_supervised_metrics(module, stage: str) -> None:
    stage_l = str(stage).lower()
    enabled = {
        group
        for group, setting in (
            ("class", "supervised"),
            ("probe", "probe"),
            ("embedding", "embedding"),
        )
        if getattr(module, f"enable_{setting}_metrics")
        and stage_l in getattr(module, f"{setting}_metric_stages")
    }
    if not enabled:
        reset_supervised_cache(module, stage)
        return

    cache = module._supervised_cache.get(stage)
    if cache is None:
        return

    if not cache["latents"]:
        for key in cache:
            cache[key].clear()
        return

    latents = torch.cat(cache["latents"], dim=0).numpy()
    labels = torch.cat(cache["class_id"], dim=0).numpy() if cache["class_id"] else None
    encoder_features = None
    if cache.get("encoder_features"):
        encoder_features = torch.cat(cache["encoder_features"], dim=0).numpy()
    latents, labels, encoder_features = _gather_latents_labels_ddp(
        latents, labels, encoder_features
    )
    latents, labels, encoder_features = _validate_cached_supervised_arrays(
        stage,
        latents,
        labels,
        encoder_features,
    )

    trainer = getattr(module, "trainer", None)
    if stage_l in {"val", "test"} and bool(getattr(trainer, "sanity_checking", False)):
        warnings.warn(
            "Skipping supervised metric logging during Lightning sanity check"
            " because sanity validation uses only a subset of batches and may"
            " not cover all classes.",
            RuntimeWarning,
            stacklevel=2,
        )
        for key in cache:
            cache[key].clear()
        return

    groups = evaluate_supervised_features(
        module, stage, latents, labels, encoder_features, enabled=enabled
    )
    probe_categories = {
        "pca95": "manifold",
        "effective_rank": "manifold",
        "silhouette": "clustering",
        "stability": "clustering",
        "balance": "clustering",
        "best_k": "clustering",
    }
    for group, metrics in groups.items():
        for name, value in metrics.items():
            extra = (
                {"metric_name": f"{probe_categories[name]}/{stage}_{name}"}
                if group == "probe"
                else {}
            )
            module._log_metric(
                stage,
                f"{group}/{name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                **extra,
            )
    reset_supervised_cache(module, stage)
