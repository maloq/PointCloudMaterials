import torch

from src.utils.spd_metrics import compute_cluster_metrics, compute_embedding_quality_metrics
from src.utils.spd_utils import cached_sample_count


def init_supervised_cache(module, cfg) -> None:
    module._supervised_cache = {
        "train": {"latents": [], "class_id": []},
        "val": {"latents": [], "class_id": []},
        "test": {"latents": [], "class_id": []},
    }
    module.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, "max_supervised_samples") else 8192
    module.max_test_samples = cfg.max_test_samples if hasattr(cfg, "max_test_samples") else 1000


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

    metrics = compute_cluster_metrics(latents, labels, stage)
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
