"""Color palette and contrast utilities for cluster visualization."""

from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors
import numpy as np
from typing import Any


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


_FIXED_CLUSTER_BASE_PALETTE = [
    "#b61f31",
    "#2a7fde",
    "#f26a00",
    "#ff2f92",
    "#5a2f8a",
    "#3f9142",
    "#b36a08",
    "#1557a6",
    "#c94a36",
    "#0c7c68",
    "#a53d8f",
    "#628d2f",
]


def _srgb_to_linear_unit(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float64)
    threshold = 0.04045
    return np.where(
        arr <= threshold,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )


def _relative_luminance(color: Any) -> float:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    linear = _srgb_to_linear_unit(rgb)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def _contrast_ratio(color_a: Any, color_b: Any) -> float:
    lum_a = _relative_luminance(color_a)
    lum_b = _relative_luminance(color_b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return float((lighter + 0.05) / (darker + 0.05))


def _adjust_color_lightness(color: Any, factor: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    h, l, s = colorsys.rgb_to_hls(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    l_new = float(np.clip(l * float(factor), 0.0, 1.0))
    rgb_new = colorsys.hls_to_rgb(h, l_new, s)
    return mcolors.to_hex(rgb_new)


def _cluster_label_color(color: Any, *, darken_factor: float = 0.58) -> str:
    factor = float(darken_factor)
    return _adjust_color_lightness(color, factor)


def _ensure_text_contrast(
    color: Any,
    *,
    background: Any = "#ffffff",
    min_ratio: float = 4.8,
    fallback: str = "#303030",
) -> str:
    color_hex = mcolors.to_hex(mcolors.to_rgb(color))
    if _contrast_ratio(color_hex, background) >= float(min_ratio):
        return color_hex
    for factor in (0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.44, 0.36):
        candidate = _adjust_color_lightness(color_hex, factor)
        if _contrast_ratio(candidate, background) >= float(min_ratio):
            return candidate
    fallback_hex = mcolors.to_hex(mcolors.to_rgb(fallback))
    return fallback_hex


def _darken_rgb(rgb: np.ndarray, factor: float) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return np.clip(arr * float(factor), 0.0, 1.0).astype(np.float32, copy=False)


def _boost_saturation(colors: np.ndarray, factor: float) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.float32)
    if abs(float(factor) - 1.0) < 1e-6:
        return arr.copy()
    # Rec.709 luminance weights preserve perceived brightness while boosting chroma.
    lum = (
        0.2126 * arr[:, 0:1]
        + 0.7152 * arr[:, 1:2]
        + 0.0722 * arr[:, 2:3]
    )
    out = lum + (arr - lum) * float(factor)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _cluster_palette(n_colors: int) -> list[str]:
    if n_colors <= 0:
        return []
    palette = list(_FIXED_CLUSTER_BASE_PALETTE[: int(n_colors)])
    if len(palette) >= int(n_colors):
        return palette[: int(n_colors)]

    golden_ratio = 0.6180339887498949
    hue = 0.03
    extra_idx = 0
    while len(palette) < int(n_colors):
        hue = (hue + golden_ratio) % 1.0
        saturation = 0.90 if extra_idx % 2 == 0 else 0.78
        value = 0.90 if extra_idx % 3 else 0.82
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = mcolors.to_hex(rgb)
        if color not in palette:
            palette.append(color)
        extra_idx += 1
    if len(palette) < n_colors:
        raise RuntimeError(
            f"Failed to build enough distinct colors for {n_colors} clusters; got {len(palette)}."
        )
    return palette


def _build_cluster_color_map(
    cluster_labels: np.ndarray,
    *,
    cluster_color_assignment: dict[int, Any] | None = None,
) -> dict[int, str]:
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    valid_ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if not valid_ids:
        raise ValueError("Cannot build cluster color map: no non-negative cluster IDs were found.")
    palette = _cluster_palette(len(valid_ids))
    color_map = {cid: palette[i] for i, cid in enumerate(valid_ids)}
    if cluster_color_assignment is None:
        return color_map

    for cluster_id in valid_ids:
        if cluster_id not in cluster_color_assignment:
            continue
        raw_value = cluster_color_assignment[cluster_id]
        if isinstance(raw_value, (int, np.integer)):
            color_map[cluster_id] = palette[int(raw_value) % len(palette)]
        elif isinstance(raw_value, str):
            text = raw_value.strip()
            if text.lstrip("+-").isdigit():
                color_map[cluster_id] = palette[int(text) % len(palette)]
            else:
                color_map[cluster_id] = mcolors.to_hex(mcolors.to_rgb(text))

    return color_map


def _compute_center_to_edge_colors(
    points: np.ndarray,
    base_color: str,
    *,
    center_lighten: float = 0.85,
    edge_darken: float = 0.10,
    radius_percentile: float = 95.0,
    gamma: float = 0.85,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    base_rgb = np.asarray(mcolors.to_rgb(str(base_color)), dtype=np.float32)

    if pts.shape[0] == 1:
        return base_rgb.reshape(1, 3)

    centroid = np.mean(pts, axis=0, dtype=np.float64)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)

    radius_scale = float(np.percentile(dists, float(radius_percentile)))
    if radius_scale <= 1e-12:
        t = np.zeros_like(dists, dtype=np.float32)
    else:
        t = np.clip(dists / radius_scale, 0.0, 1.0).astype(np.float32, copy=False)
    t = np.power(t, float(gamma)).astype(np.float32, copy=False)

    center_rgb = base_rgb + float(center_lighten) * (1.0 - base_rgb)
    edge_rgb = base_rgb * (1.0 - float(edge_darken))
    colors = center_rgb[None, :] * (1.0 - t[:, None]) + edge_rgb[None, :] * t[:, None]
    return np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)
