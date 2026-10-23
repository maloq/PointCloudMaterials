"""Collect evaluation embeddings, preserving split order and model state."""

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader


def _random_rotation_matrices(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rand = torch.randn(batch_size, 3, 3, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(rand)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-1)
    det = torch.det(q)
    neg = det < 0
    if bool(neg.any()):
        q[neg, :, 0] *= -1
    return q.to(device=device, dtype=dtype)


def _build_supervised_eval_loader(module, split: str) -> DataLoader | None:
    trainer = getattr(module, "trainer", None)
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        return None

    dataset = getattr(datamodule, f"{split}_dataset", None)

    if dataset is None:
        dataloader_fn = getattr(datamodule, f"{split}_dataloader", None)
        if callable(dataloader_fn):
            try:
                return dataloader_fn()
            except (TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(
                    f"Dataloader construction failed for split '{split}': {exc}"
                )
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
    extractor = module._extract_supervised_features_from_batch

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

                features = torch.as_tensor(features).detach().to(torch.float32)
                features = features.reshape(features.shape[0], -1)
                labels = torch.as_tensor(labels).detach().view(-1).to(torch.long)

                if features.shape[0] != labels.shape[0]:
                    raise ValueError(
                        f"Supervised extractor row mismatch: split={split!r}, "
                        f"features={tuple(features.shape)}, labels={tuple(labels.shape)}."
                    )
                take = int(features.shape[0])
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


def _extract_rotated_supervised_features_from_batch(
    module,
    batch,
    *,
    seed: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    pc_raw, meta = module._unpack_batch(batch)
    class_id = meta.get("class_id")
    if class_id is None:
        return None, None

    if pc_raw.dim() != 3 or int(pc_raw.shape[-1]) != 3:
        raise ValueError(
            "Expected point clouds with shape (B, N, 3) for rotated test"
            f" metrics, got shape={tuple(pc_raw.shape)}."
        )

    pc_raw = pc_raw.to(device=module.device, dtype=module.dtype, non_blocking=True)
    rots = _random_rotation_matrices(
        int(pc_raw.shape[0]),
        device=pc_raw.device,
        dtype=pc_raw.dtype,
        seed=int(seed),
    )
    # Row-vector convention: each point is rotated as x @ R.
    pc_rot = torch.matmul(pc_raw, rots)
    # Both maintained SSL forwards prepare/crop their input themselves.
    out = module(pc_rot)
    z_inv_contrastive = out[0]
    z_inv_model = out[1]
    features = z_inv_contrastive if z_inv_contrastive is not None else z_inv_model
    return features.detach().to(torch.float32), class_id


def _collect_rotated_split_supervised_features(
    module,
    split: str,
    *,
    max_samples: int | None,
    rotation_seed_base: int,
    rotation_run_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            raise RuntimeError(
                "Rotated test metrics currently require single-device"
                " evaluation. Set test_single_device=true (recommended) or"
                " disable distributed test."
            )

    loader = _build_supervised_eval_loader(module, split)
    if loader is None:
        raise RuntimeError(
            f"Could not build dataloader for split='{split}' while computing"
            " rotated test metrics."
        )

    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    collected = 0
    limit = None if max_samples is None or int(max_samples) <= 0 else int(max_samples)

    was_training = bool(module.training)
    module.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                seed = (
                    int(rotation_seed_base)
                    + int(rotation_run_idx) * 100000
                    + int(batch_idx)
                )
                features, labels = _extract_rotated_supervised_features_from_batch(
                    module,
                    batch,
                    seed=seed,
                )
                if features is None or labels is None:
                    raise RuntimeError(
                        "Test batch does not provide class_id labels; "
                        "cannot compute rotated class-accuracy metrics."
                    )

                features = (
                    features.detach().to(torch.float32).reshape(features.shape[0], -1)
                )
                labels = torch.as_tensor(labels).detach().view(-1).to(torch.long)

                if features.shape[0] != labels.shape[0]:
                    raise ValueError(
                        f"Rotated supervised extractor row mismatch: split={split!r}, "
                        f"features={tuple(features.shape)}, labels={tuple(labels.shape)}."
                    )
                take = int(features.shape[0])
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
        raise RuntimeError(
            f"Collected zero samples for rotated split='{split}' supervised metrics."
        )
    features_np = torch.cat(feature_parts, dim=0).numpy()
    labels_np = torch.cat(label_parts, dim=0).numpy()
    return features_np, labels_np


def _get_stage_dataset(module, stage: str):
    trainer = getattr(module, "trainer", None)
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        return None

    split_name = f"{stage}_dataset"
    dataset = getattr(datamodule, split_name, None)
    while dataset is not None and hasattr(dataset, "dataset"):
        inner = getattr(dataset, "dataset")
        if inner is dataset:
            break
        dataset = inner
    return dataset


def cache_limit_for_stage(module, stage: str):
    if stage == "test":
        return module.max_test_samples
    if stage in {"train", "val"}:
        return module.max_supervised_samples
    return None


def _gather_latents_labels_ddp(
    latents: np.ndarray,
    labels: np.ndarray | None,
    encoder_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Gather cached latent/label arrays across DDP ranks for global metrics."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return latents, labels, encoder_features

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return latents, labels, encoder_features

    payload = {
        "latents": np.asarray(latents, dtype=np.float32),
        "labels": (None if labels is None else np.asarray(labels, dtype=np.int64)),
    }
    if encoder_features is not None:
        payload["encoder_features"] = np.asarray(encoder_features, dtype=np.float32)
    gathered: list[dict[str, np.ndarray] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, payload)

    latents_parts: list[np.ndarray] = []
    encoder_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    labels_expected = labels is not None
    for rank_idx, item in enumerate(gathered):
        if not item:
            continue
        part_lat = np.asarray(item.get("latents", []), dtype=np.float32)
        if part_lat.ndim != 2 or part_lat.shape[0] == 0:
            continue

        raw_labels = item.get("labels", None)
        if labels_expected:
            if raw_labels is None:
                raise RuntimeError(
                    "DDP metric cache label availability mismatch: local rank"
                    f" has class_id labels but gathered rank {rank_idx} does"
                    " not."
                )
            part_lab = np.asarray(raw_labels, dtype=np.int64)
            if part_lab.ndim != 1 or part_lab.shape[0] != part_lat.shape[0]:
                raise RuntimeError(
                    "DDP metric cache gathered mismatched latent/label rows:"
                    f" rank={rank_idx}, latents={part_lat.shape},"
                    f" labels={part_lab.shape}."
                )
            labels_parts.append(part_lab)
        elif raw_labels is not None and np.asarray(raw_labels).size > 0:
            raise RuntimeError(
                "DDP metric cache label availability mismatch: local rank has"
                f" no class_id labels but gathered rank {rank_idx} does."
            )

        raw_encoder = item.get("encoder_features", None)
        if raw_encoder is not None:
            part_enc = np.asarray(raw_encoder, dtype=np.float32)
            if part_enc.size > 0:
                if part_enc.ndim != 2 or part_enc.shape[0] != part_lat.shape[0]:
                    raise RuntimeError(
                        "DDP metric cache gathered mismatched latent/encoder"
                        f" rows: rank={rank_idx}, latents={part_lat.shape},"
                        f" encoder_features={part_enc.shape}."
                    )
                encoder_parts.append(part_enc)
        latents_parts.append(part_lat)

    if not latents_parts:
        return latents, labels, encoder_features

    gathered_latents = np.concatenate(latents_parts, axis=0)
    gathered_labels = np.concatenate(labels_parts, axis=0) if labels_expected else None
    if len(encoder_parts) == len(latents_parts):
        gathered_enc = np.concatenate(encoder_parts, axis=0)
    else:
        gathered_enc = None
    return gathered_latents, gathered_labels, gathered_enc
