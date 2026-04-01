"""
Shared training utilities.
Contains optimizer/scheduler setup, tensor operations, rotation utilities,
batch processing, and configuration helpers.
"""

import warnings

import torch
from typing import Tuple, Dict, Any, Optional
from bisect import bisect_right


class _SequentialLRNoEpoch(torch.optim.lr_scheduler.SequentialLR):
    """SequentialLR variant that avoids deprecated scheduler.step(epoch=...)."""

    def step(self):  # type: ignore[override]
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step()
        self._last_lr = scheduler.get_last_lr()


def get_optimizers_and_scheduler(hparams, parameters):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=hparams.learning_rate,
        weight_decay=hparams.decay_rate
    )

    if hparams.enable_swa:
        epochs_before_swa = hparams.swa_epoch_start + 1
    else:
        epochs_before_swa = hparams.epochs

    scheduler_name = hparams.scheduler_name
    step_interval_scheduler = False
    if scheduler_name == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100, 
            gamma=hparams.scheduler_gamma
        )
    elif scheduler_name == 'OneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hparams.learning_rate, 
            total_steps=epochs_before_swa
        )
        step_interval_scheduler = True
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20, 
            T_mult=3
        )
    elif scheduler_name == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_before_swa,
            eta_min=getattr(hparams, 'scheduler_min_lr', 1e-6)
        )
    elif scheduler_name == 'chained':
        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hparams.learning_rate, 
            total_steps=int(epochs_before_swa/2)
        )
        scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hparams.learning_rate, 
            total_steps=int(epochs_before_swa/2)
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
    
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

    if getattr(hparams, "scheduler_gamma", None) is not None and scheduler_name != "Step":
        warnings.warn(
            f"Ignoring configured field scheduler_gamma: scheduler_name={scheduler_name!r} does not use gamma.",
            stacklevel=2,
        )
    if getattr(hparams, "scheduler_min_lr", None) is not None and scheduler_name != "Cosine":
        warnings.warn(
            f"Ignoring configured field scheduler_min_lr: scheduler_name={scheduler_name!r} does not use eta_min.",
            stacklevel=2,
        )

    # Optional epoch-level linear warmup for non-step schedulers.
    warmup_enabled = bool(getattr(hparams, "warmup_enabled", False))
    warmup_epochs = int(getattr(hparams, "warmup_epochs", 0) or 0)
    warmup_start_factor = float(getattr(hparams, "warmup_start_factor", 0.1))
    if not warmup_enabled:
        if getattr(hparams, "warmup_epochs", None) is not None:
            warnings.warn(
                "Ignoring configured field warmup_epochs: warmup_enabled=false disables warmup.",
                stacklevel=2,
            )
        if getattr(hparams, "warmup_start_factor", None) is not None:
            warnings.warn(
                "Ignoring configured field warmup_start_factor: warmup_enabled=false disables warmup.",
                stacklevel=2,
            )
    elif warmup_epochs <= 0:
        if getattr(hparams, "warmup_start_factor", None) is not None:
            warnings.warn(
                f"Ignoring configured field warmup_start_factor: warmup_epochs={warmup_epochs} disables warmup.",
                stacklevel=2,
            )
    elif step_interval_scheduler:
        warnings.warn(
            f"Ignoring configured warmup fields: scheduler_name={scheduler_name!r} already schedules learning rate per step.",
            stacklevel=2,
        )
    if warmup_enabled and warmup_epochs > 0 and not step_interval_scheduler:
        warmup_epochs = min(warmup_epochs, max(1, int(epochs_before_swa)))
        warmup_start_factor = min(max(warmup_start_factor, 1e-6), 1.0)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = _SequentialLRNoEpoch(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs],
        )

    if step_interval_scheduler:
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    return [optimizer], [scheduler] 
    
# ============================================================================
# Tensor Conversion Utilities
# ============================================================================

def to_float32(*tensors):
    """Convert multiple tensors to float32 at once."""
    return tuple(t.to(torch.float32) for t in tensors)


# ============================================================================
# Rotation Utilities
# ============================================================================

def apply_rotation(points: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrices to batched point clouds.

    Args:
        points: (B, N, 3) point clouds
        rot: (B, 3, 3) rotation matrices

    Returns:
        Rotated point clouds (B, N, 3)
    """
    return (rot @ points.transpose(1, 2)).transpose(1, 2).contiguous()


def identity_rotation(batch_size: int, device, dtype) -> torch.Tensor:
    """
    Create identity rotation matrices for a batch.

    Args:
        batch_size: Number of rotation matrices to create
        device: Device to create tensors on
        dtype: Data type of tensors

    Returns:
        Identity rotation matrices (B, 3, 3)
    """
    eye = torch.eye(3, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch_size, -1, -1)


def order_points_for_kabsch(points: torch.Tensor) -> torch.Tensor:
    """
    Provide a deterministic ordering of points (by radial distance to centroid)
    so that Kabsch correspondences become permutation agnostic.
    """
    if points.dim() != 3:
        raise ValueError(f"Point tensor must be 3D (B, N, 3). Got shape {tuple(points.shape)}")
    if points.size(-1) != 3:
        if points.size(1) == 3:
            points = points.transpose(1, 2)
        else:
            raise ValueError(f"Point tensor must have 3 coordinates. Got shape {tuple(points.shape)}")

    pts_bn3 = points
    pts_for_order = pts_bn3.detach() if pts_bn3.requires_grad else pts_bn3
    centered = pts_for_order - pts_for_order.mean(dim=1, keepdim=True)
    radii = centered.norm(dim=-1)
    order = torch.argsort(radii, dim=1)
    ordered = torch.gather(pts_bn3, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    return ordered


def rotation_geodesic(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean geodesic angle (radians) between predicted and ground-truth rotations.

    Args:
        pred: (B, 3, 3) predicted rotation matrices
        target: (B, 3, 3) ground truth rotation matrices
        eps: Small value for numerical stability

    Returns:
        Mean geodesic distance in radians
    """
    if pred.shape != target.shape:
        raise ValueError(f"Rotation shapes must match (got {pred.shape} vs {target.shape})")
    delta = pred.transpose(-1, -2) @ target
    trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(cos_theta).mean()


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def unpack_batch(batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Unpack batch data into point clouds and labels.

    Args:
        batch: Batch data (dict with "points", "class_id", "instance_id", "rotation")

    Returns:
        Tuple of (point_clouds, meta_dict)
    """
    if isinstance(batch, dict):
        pc = batch["points"]
        meta = {
            "class_id": batch.get("class_id"),
            "instance_id": batch.get("instance_id"),
            "rotation": batch.get("rotation"),
        }
        return pc, meta
    
    if not isinstance(batch, (tuple, list)):
        return batch, {}

    # Fallback for legacy tuple format
    return batch[0], {}


# ============================================================================
# Cache Utilities
# ============================================================================

def cached_sample_count(cache: dict) -> int:
    """
    Count the number of samples currently cached.

    Args:
        cache: Cache dictionary with 'latents' key containing list of tensors

    Returns:
        Total number of cached samples
    """
    latents = cache.get("latents") if cache is not None else None
    if not latents:
        return 0
    return sum(t.shape[0] for t in latents)


# ============================================================================
# Sinkhorn Blur Schedule
# ============================================================================

def init_sinkhorn_blur_schedule(cfg) -> Dict[str, Any]:
    """
    Initialize sinkhorn blur schedule from configuration.

    Args:
        cfg: Configuration object with sinkhorn_blur_schedule section

    Returns:
        Dictionary with schedule parameters
    """
    schedule_cfg = cfg.sinkhorn_blur_schedule
    start = float(schedule_cfg.start)
    end = float(schedule_cfg.end)
    start_epoch = int(schedule_cfg.start_epoch)
    duration = int(schedule_cfg.duration_epochs)
    enabled = bool(schedule_cfg.enable)
    return {
        "enabled": enabled,
        "start": start,
        "end": end,
        "start_epoch": max(0, start_epoch),
        "duration_epochs": duration,
    }


def get_current_sinkhorn_blur(schedule: Optional[Dict[str, Any]], current_epoch: int) -> float:
    """
    Compute current sinkhorn blur value based on schedule and epoch.

    Args:
        schedule: Sinkhorn blur schedule dictionary (from init_sinkhorn_blur_schedule)
        current_epoch: Current training epoch

    Returns:
        Current sinkhorn blur value
    """
    if not schedule:
        return 0.02

    start = float(schedule["start"])
    if not schedule.get("enabled", False):
        return start

    duration = int(schedule["duration_epochs"])
    if duration <= 1:
        return float(schedule["end"])

    epoch = max(0, int(current_epoch))
    start_epoch = int(schedule["start_epoch"])
    elapsed = max(0, epoch - start_epoch)
    max_elapsed = duration - 1
    if elapsed >= max_elapsed:
        return float(schedule["end"])

    alpha = elapsed / max_elapsed
    return float(start + alpha * (schedule["end"] - start))
