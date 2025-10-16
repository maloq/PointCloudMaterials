"""Evaluation utilities for synthetic dataset experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from .orientation import so3_geodesic_angle


def clustering_scores(pred_labels: Sequence[int], true_phases: Sequence[int]) -> Dict[str, float]:
    pred = np.asarray(pred_labels)
    truth = np.asarray(true_phases)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and truth label vectors must match in shape.")
    return {
        "ARI": float(adjusted_rand_score(truth, pred)),
        "NMI": float(normalized_mutual_info_score(truth, pred)),
        "AMI": float(adjusted_mutual_info_score(truth, pred)),
        "FowlkesMallows": float(fowlkes_mallows_score(truth, pred)),
    }


def rotation_errors(
    pred_R: np.ndarray,
    true_R: np.ndarray,
    *,
    degrees: bool = True,
) -> Dict[str, float]:
    pred = np.asarray(pred_R)
    truth = np.asarray(true_R)
    if pred.shape != truth.shape:
        raise ValueError("pred_R and true_R must have identical shape.")
    angles = np.array([
        so3_geodesic_angle(pred[i], truth[i], degrees=degrees)
        for i in range(pred.shape[0])
    ])
    within_5 = float((angles <= (5 if degrees else np.deg2rad(5))).mean())
    return {
        "mean_angle": float(angles.mean()),
        "median_angle": float(np.median(angles)),
        "p_within_5deg": within_5,
    }


def phase_iou(pred_phase_grid: np.ndarray, true_phase_grid: np.ndarray) -> Dict[int, float]:
    pred = np.asarray(pred_phase_grid)
    truth = np.asarray(true_phase_grid)
    if pred.shape != truth.shape:
        raise ValueError("Phase grids must have matching shape.")
    classes = np.unique(np.concatenate([pred.ravel(), truth.ravel()]))
    scores: Dict[int, float] = {}
    for c in classes:
        pred_mask = pred == c
        truth_mask = truth == c
        intersection = np.logical_and(pred_mask, truth_mask).sum()
        union = np.logical_or(pred_mask, truth_mask).sum()
        scores[int(c)] = float(intersection / union) if union > 0 else 1.0
    return scores


def robustness_curve(results: List[Dict[str, float]], noise_levels: Sequence[float]) -> pd.DataFrame:
    if len(results) != len(noise_levels):
        raise ValueError("Results and noise_levels must align in length.")
    rows = []
    for metrics, level in zip(results, noise_levels):
        row = dict(metrics)
        row["noise_level"] = level
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_stratified_metrics(
    metric_values: Sequence[float],
    boundary_distances: Sequence[float],
    bins: Sequence[float],
) -> Dict[str, float]:
    metric_arr = np.asarray(metric_values)
    boundary_arr = np.asarray(boundary_distances)
    bins = np.asarray(bins)
    if metric_arr.shape != boundary_arr.shape:
        raise ValueError("metric_values and boundary_distances must align.")
    digitized = np.digitize(boundary_arr, bins, right=True)
    scores: Dict[str, float] = {}
    for idx in range(len(bins)):
        mask = digitized == idx
        if not mask.any():
            continue
        key = f"bin_{idx}"
        scores[key] = float(metric_arr[mask].mean())
    return scores
