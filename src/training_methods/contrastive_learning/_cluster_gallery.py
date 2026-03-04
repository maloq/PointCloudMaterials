from __future__ import annotations

"""Image normalization and horizontal gallery stitching."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _normalize_image_rgba(image: np.ndarray, *, path: Path) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise ValueError(
            f"Expected image with shape (H, W, 3/4) for {path}, got {arr.shape}."
        )
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32, copy=False)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape[2] == 3:
        alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=2)
    return arr


def _save_horizontal_image_gallery(
    image_paths: list[Path],
    out_file: Path,
    *,
    spacing_px: int = 20,
    outer_padding_px: int = 18,
    background_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("image_paths must be a non-empty list.")
    spacing_px = max(0, int(spacing_px))
    outer_padding_px = max(0, int(outer_padding_px))
    bg = np.asarray(background_rgba, dtype=np.float32)
    if bg.shape != (4,):
        raise ValueError(
            f"background_rgba must contain exactly 4 values, got shape {bg.shape}."
        )
    bg = np.clip(bg, 0.0, 1.0)

    images: list[np.ndarray] = []
    widths: list[int] = []
    max_height = 0
    for path_like in image_paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Cannot build gallery: image is missing: {path}")
        image = _normalize_image_rgba(plt.imread(path), path=path)
        images.append(image)
        widths.append(int(image.shape[1]))
        max_height = max(max_height, int(image.shape[0]))

    total_width = (
        2 * outer_padding_px
        + int(sum(widths))
        + spacing_px * max(0, len(images) - 1)
    )
    total_height = max_height + 2 * outer_padding_px
    gallery = np.tile(
        bg[None, None, :],
        (int(total_height), int(total_width), 1),
    ).astype(np.float32, copy=False)

    x_cursor = int(outer_padding_px)
    placements: list[dict[str, int]] = []
    for image, width, path_like in zip(images, widths, image_paths):
        path = Path(path_like)
        height = int(image.shape[0])
        y_offset = int(outer_padding_px + (max_height - height) // 2)
        gallery[y_offset : y_offset + height, x_cursor : x_cursor + width, :] = image
        placements.append(
            {
                "file": str(path),
                "x": int(x_cursor),
                "y": int(y_offset),
                "width": int(width),
                "height": int(height),
            }
        )
        x_cursor += int(width) + int(spacing_px)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_file, gallery)
    return {
        "out_file": str(out_file),
        "num_images": int(len(images)),
        "image_paths": [str(Path(p)) for p in image_paths],
        "canvas_width": int(total_width),
        "canvas_height": int(total_height),
        "spacing_px": int(spacing_px),
        "outer_padding_px": int(outer_padding_px),
        "placements": placements,
    }
