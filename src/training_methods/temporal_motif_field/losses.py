from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def entropy_from_probs(probs: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    probs_f = probs.to(dtype=torch.float32).clamp_min(float(eps))
    return -(probs_f * probs_f.log()).sum(dim=dim)


def usage_balance_loss(probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    usage = probs.to(dtype=torch.float32).mean(dim=0).clamp_min(float(eps))
    entropy = -(usage * usage.log()).sum()
    max_entropy = math.log(float(probs.shape[-1]))
    return usage.new_tensor(max_entropy) - entropy


def sample_entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    return entropy_from_probs(probs).mean()


def kl_from_logits_to_probs(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    pred_log_probs = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    target = target_probs.to(dtype=torch.float32).clamp_min(1e-8)
    return F.kl_div(pred_log_probs, target, reduction="batchmean")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred_f = pred.to(dtype=torch.float32)
    target_f = target.to(dtype=torch.float32)
    loss = (pred_f - target_f).square().mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask_f = mask.to(device=loss.device, dtype=torch.float32).reshape(-1)
    denom = mask_f.sum().clamp_min(1.0)
    return (loss * mask_f).sum() / denom


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    value_f = value.to(dtype=torch.float32).reshape(-1)
    if mask is None:
        return value_f.mean()
    mask_f = mask.to(device=value_f.device, dtype=torch.float32).reshape(-1)
    denom = mask_f.sum().clamp_min(1.0)
    return (value_f * mask_f).sum() / denom


def binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    logits_f = logits.to(dtype=torch.float32).reshape(-1)
    targets_f = targets.to(device=logits.device, dtype=torch.float32).reshape(-1)
    bce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
    if float(focal_gamma) <= 0.0:
        return bce.mean()
    probs = torch.sigmoid(logits_f)
    pt = torch.where(targets_f > 0.5, probs, 1.0 - probs).clamp_min(1e-6)
    return ((1.0 - pt) ** float(focal_gamma) * bce).mean()


def prototype_separation_loss(
    bridge_prototypes: torch.Tensor,
    stable_prototypes: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if bridge_prototypes.ndim != 2 or stable_prototypes.ndim != 2:
        raise ValueError(
            "prototype_separation_loss expects 2D prototype matrices, "
            f"got bridge={tuple(bridge_prototypes.shape)}, stable={tuple(stable_prototypes.shape)}."
        )
    distances = torch.cdist(
        bridge_prototypes.to(dtype=torch.float32),
        stable_prototypes.to(dtype=torch.float32),
        p=2,
    )
    nearest = distances.min(dim=1).values
    return torch.relu(torch.as_tensor(float(margin), device=nearest.device) - nearest).mean()


def top1_accuracy_from_logits(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    target = target_probs.argmax(dim=-1)
    return (pred == target).to(dtype=torch.float32).mean()


def prototype_usage_metrics(probs: torch.Tensor, *, dead_threshold: float = 1e-4) -> dict[str, torch.Tensor]:
    probs_f = probs.to(dtype=torch.float32)
    usage = probs_f.mean(dim=0)
    usage_entropy = entropy_from_probs(usage.unsqueeze(0)).squeeze(0)
    active = usage > float(dead_threshold)
    max_prob_mean = probs_f.max(dim=-1).values.mean()
    return {
        "usage_entropy": usage_entropy,
        "active_count": usage.new_tensor(float(active.sum().item())),
        "dead_fraction": usage.new_tensor(float((~active).float().mean().item())),
        "max_prob_mean": max_prob_mean,
    }
