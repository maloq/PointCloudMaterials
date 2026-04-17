from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def score_bridge_candidates(
    cache: Mapping[str, Any] | None,
    lags: Sequence[int],
    pred_errors: torch.Tensor,
    stable_ids: torch.Tensor,
    stable_confidence: torch.Tensor,
    hazard_probs: torch.Tensor,
) -> torch.Tensor:
    if pred_errors.ndim != 2:
        raise ValueError(f"pred_errors must have shape (B, L), got {tuple(pred_errors.shape)}.")
    if stable_ids.ndim != 2:
        raise ValueError(f"stable_ids must have shape (B, 1 + L), got {tuple(stable_ids.shape)}.")
    if stable_confidence.ndim != 1:
        raise ValueError(
            f"stable_confidence must have shape (B,), got {tuple(stable_confidence.shape)}."
        )
    if hazard_probs.ndim != 2:
        raise ValueError(f"hazard_probs must have shape (B, L), got {tuple(hazard_probs.shape)}.")

    num_lags = len([int(lag) for lag in lags])
    if int(pred_errors.shape[1]) != num_lags:
        raise ValueError(
            "pred_errors second dimension must match len(lags). "
            f"Got pred_errors.shape[1]={int(pred_errors.shape[1])}, len(lags)={num_lags}."
        )
    if int(hazard_probs.shape[1]) != num_lags:
        raise ValueError(
            "hazard_probs second dimension must match len(lags). "
            f"Got hazard_probs.shape[1]={int(hazard_probs.shape[1])}, len(lags)={num_lags}."
        )
    if int(stable_ids.shape[1]) != num_lags + 1:
        raise ValueError(
            "stable_ids second dimension must contain anchor plus one target per lag. "
            f"Got stable_ids.shape[1]={int(stable_ids.shape[1])}, len(lags)={num_lags}."
        )

    weights = {}
    if cache is not None:
        raw_weights = cache.get("score_weights", None)
        if raw_weights is not None:
            weights = dict(raw_weights)

    stable_change_weight = float(weights.get("stable_change", 1.0))
    prediction_error_weight = float(weights.get("prediction_error", 0.5))
    hazard_weight = float(weights.get("hazard", 0.5))
    entropy_weight = float(weights.get("entropy", 0.25))

    anchor_ids = stable_ids[:, 0]
    target_ids = stable_ids[:, 1:]
    stable_change = (target_ids != anchor_ids.unsqueeze(1)).any(dim=1).to(dtype=torch.float32)
    prediction_error = pred_errors.to(dtype=torch.float32).mean(dim=1)
    hazard_score = hazard_probs.to(dtype=torch.float32).mean(dim=1)
    entropy_score = 1.0 - stable_confidence.to(dtype=torch.float32).clamp(0.0, 1.0)

    return (
        stable_change_weight * stable_change
        + prediction_error_weight * prediction_error
        + hazard_weight * hazard_score
        + entropy_weight * entropy_score
    )


def select_bridge_candidate_keys(
    sample_keys: Sequence[tuple[str, Any, Any]],
    scores: torch.Tensor,
    *,
    candidate_fraction: float,
    max_candidates: int | None = None,
) -> set[tuple[str, Any, Any]]:
    if not sample_keys:
        return set()
    if scores.ndim != 1:
        raise ValueError(f"scores must have shape (N,), got {tuple(scores.shape)}.")
    if len(sample_keys) != int(scores.shape[0]):
        raise ValueError(
            "sample_keys and scores must have the same length. "
            f"Got len(sample_keys)={len(sample_keys)}, scores.shape[0]={int(scores.shape[0])}."
        )

    fraction = float(candidate_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"candidate_fraction must be in (0, 1], got {fraction}.")

    top_k = max(1, int(math.ceil(fraction * len(sample_keys))))
    if max_candidates is not None:
        top_k = min(top_k, int(max_candidates))
    top_k = min(top_k, len(sample_keys))
    top_indices = torch.topk(scores.to(dtype=torch.float32), k=top_k, largest=True).indices.tolist()
    return {sample_keys[int(index)] for index in top_indices}
