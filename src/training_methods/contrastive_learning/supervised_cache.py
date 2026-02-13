import numpy as np
import torch

from src.utils.spd_metrics import compute_cluster_metrics, compute_embedding_quality_metrics
from src.utils.spd_utils import cached_sample_count


def _infer_test_cluster_eval_k(cfg):
    configured = getattr(cfg, "test_cluster_eval_k", None)
    if configured is not None:
        k = int(configured)
        return k if k > 1 else None

    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return None
    dataset_type = str(getattr(data_cfg, "dataset_type", "")).lower()
    if dataset_type not in {"modelnet_objects", "modelnet_objects_balanced_topk"}:
        return None
    if dataset_type == "modelnet_objects_balanced_topk":
        top_k = getattr(data_cfg, "top_k_classes", None)
        if top_k is not None:
            k = int(top_k)
            return k if k > 1 else None

    classes = getattr(data_cfg, "classes", None)
    if classes is None:
        return 40
    if isinstance(classes, (list, tuple)):
        k = len(classes)
        return k if k > 1 else None
    if hasattr(classes, "__len__") and not isinstance(classes, (str, bytes, dict)):
        k = len(classes)
        return k if k > 1 else None
    return None


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
        except Exception:
            continue
        if runs > 0:
            out[k] = runs
    return out


def _infer_stage_acc_settings(cfg, stage: str) -> tuple[list[str], int, dict[str, int]]:
    stage_l = str(stage).lower()
    if stage_l == "val":
        methods = _as_string_list(
            getattr(cfg, "val_cluster_acc_methods", None),
            default=["kmeans++"],
        )
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
        runs = int(getattr(cfg, "test_cluster_acc_runs", 1))
        runs_by_method = _as_int_mapping(
            getattr(cfg, "test_cluster_acc_runs_by_method", None),
            default={"kmeans": 20},
        )
        return methods, max(1, runs), runs_by_method

    return [], 1, {}


def init_supervised_cache(module, cfg) -> None:
    module._supervised_cache = {
        "train": {"latents": [], "class_id": []},
        "val": {"latents": [], "class_id": []},
        "test": {"latents": [], "class_id": []},
    }
    module.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, "max_supervised_samples") else 8192
    module.max_test_samples = cfg.max_test_samples if hasattr(cfg, "max_test_samples") else 1000
    module.test_cluster_eval_k = _infer_test_cluster_eval_k(cfg)
    module.cluster_acc_seed = int(getattr(cfg, "cluster_acc_seed", 0))
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


def cache_supervised_batch(module, stage: str, z: torch.Tensor, meta: dict) -> None:
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

    if z is None:
        return

    batch_size = int(z.shape[0])
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
    if effective_batch <= 0:
        return

    cache["latents"].append(z[:effective_batch].detach().to(torch.float32).cpu())
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
    latents, labels = _gather_latents_labels_ddp(latents, labels)

    stage_l = str(stage).lower()
    if stage_l == "val":
        acc_methods = getattr(module, "val_cluster_acc_methods", ["kmeans++"])
        acc_runs = int(getattr(module, "val_cluster_acc_runs", 1))
        acc_runs_by_method = getattr(module, "val_cluster_acc_runs_by_method", {})
    elif stage_l == "test":
        acc_methods = getattr(module, "test_cluster_acc_methods", ["kmeans", "kmeans++", "gmm_full"])
        acc_runs = int(getattr(module, "test_cluster_acc_runs", 1))
        acc_runs_by_method = getattr(module, "test_cluster_acc_runs_by_method", {"kmeans": 20})
    else:
        acc_methods = []
        acc_runs = 1
        acc_runs_by_method = {}

    metrics = compute_cluster_metrics(
        latents,
        labels,
        stage,
        hungarian_eval_k=getattr(module, "test_cluster_eval_k", None),
        acc_eval_methods=acc_methods,
        acc_eval_runs=max(1, acc_runs),
        acc_eval_runs_by_method=acc_runs_by_method,
        acc_random_seed=int(getattr(module, "cluster_acc_seed", 0)),
    )
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

    try:
        emb_metrics = compute_embedding_quality_metrics(latents, labels, include_expensive=(stage == "test"))
        # embedding/* metrics: geometry/quality scores of latent embedding with class labels.
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
    except Exception as exc:
        print(f"Error computing embedding quality metrics: {exc}")

    for key in cache:
        cache[key].clear()


def _gather_latents_labels_ddp(
    latents: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather cached latent/label arrays across DDP ranks for global metrics."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return latents, labels

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return latents, labels

    payload = {
        "latents": np.asarray(latents, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    gathered: list[dict[str, np.ndarray] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, payload)

    latents_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for item in gathered:
        if not item:
            continue
        part_lat = np.asarray(item.get("latents", []), dtype=np.float32)
        part_lab = np.asarray(item.get("labels", []), dtype=np.int64)
        if part_lat.ndim != 2 or part_lab.ndim != 1 or part_lat.shape[0] == 0:
            continue
        n = min(part_lat.shape[0], part_lab.shape[0])
        if n <= 0:
            continue
        latents_parts.append(part_lat[:n])
        labels_parts.append(part_lab[:n])

    if not latents_parts:
        return latents, labels
    return np.concatenate(latents_parts, axis=0), np.concatenate(labels_parts, axis=0)
