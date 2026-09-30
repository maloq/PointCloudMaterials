"""Blender Cycles raytracing for MD cluster snapshots."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np

from .cluster_colors import _boost_saturation
from .cluster_geometry import _estimate_ball_radius_world, _sample_indices_stratified
from .output_layout import log_saved_figure as _log_saved_figure


def _run_blender_render_batch(
    blender_exec: str,
    blender_script: Path,
    payload: dict,
    out_files: list[Path],
    *,
    timeout_seconds: int,
    tmp_prefix: str = "blender_render_",
) -> None:
    """Run a Blender scene module and verify every requested output."""
    if not out_files:
        raise ValueError("A Blender render batch must contain at least one output file.")
    output_summary = ", ".join(str(path) for path in out_files)
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp_dir:
        tmp_root = Path(tmp_dir)
        payload_path = tmp_root / "payload.json"
        payload_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

        cmd = [
            blender_exec, "-b", "--factory-startup",
            "-P", str(blender_script),
            "--", "--payload_json", str(payload_path),
        ]
        try:
            proc = subprocess.run(
                cmd, check=False, capture_output=True, text=True,
                timeout=int(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "Blender render batch timed out after "
                f"{timeout_seconds}s for outputs [{output_summary}]."
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                "Blender render batch failed "
                f"(exit {proc.returncode}) for outputs [{output_summary}].\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )
        if "Traceback (most recent call last):" in proc.stderr:
            raise RuntimeError(
                "Blender script raised an exception for outputs "
                f"[{output_summary}].\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )

    for out_file in out_files:
        if not out_file.exists():
            candidates = sorted(out_file.parent.glob(f"{out_file.stem}*{out_file.suffix}"))
            raise FileNotFoundError(
                f"Blender render batch succeeded but output is missing: {out_file}, "
                f"candidates={[str(p) for p in candidates]}."
            )


def _resolve_blender_executable(blender_executable: str) -> str:
    exe = str(blender_executable).strip()
    if exe == "":
        raise ValueError("blender_executable must be a non-empty string.")
    if "/" in exe or "\\" in exe:
        path = Path(exe).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Blender executable does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Blender executable path is not a file: {path}")
        return str(path)
    resolved = shutil.which(exe)
    if resolved is None:
        raise FileNotFoundError(
            "Blender executable was not found in PATH. "
            f"Tried '{exe}'. Install Blender or provide an absolute path."
        )
    return str(resolved)


def _save_md_cluster_snapshots_raytrace_blender(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    render_jobs: list[dict[str, Any]],
    *,
    max_points: int | None = None,
    image_width: int = 1200,
    image_height: int = 1200,
    projection: str = "perspective",
    perspective_fov_deg: float = 34.0,
    camera_distance_factor: float = 2.8,
    sphere_radius_fraction: float = 0.0105,
    blender_executable: str = "blender",
    cycles_samples: int = 32,
    use_denoise: bool = True,
    use_gpu: bool = False,
    timeout_seconds: int = 1200,
    wireframe_enabled: bool = True,
    wireframe_width_fraction: float = 0.0017,
) -> list[dict[str, Any]]:
    """Render every requested view of one MD snapshot in one Blender process.

    This is an additive renderer used alongside the existing matplotlib
    outputs. It requires a Blender executable.
    """
    if not render_jobs:
        raise ValueError("Raytracing requires at least one render job.")
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"Raytracing coordinates must have shape (N, 3), got {coords_arr.shape}.")
    if labels.ndim != 1 or labels.shape[0] != coords_arr.shape[0]:
        raise ValueError(
            "Raytracing cluster labels must have shape (N,) matching coordinates. "
            f"labels={labels.shape}, coordinates={coords_arr.shape}."
        )
    projection_norm = str(projection).strip().lower()
    if projection_norm not in {"perspective", "persp", "orthographic", "ortho"}:
        raise ValueError(
            "Raytracing projection must be one of "
            f"['perspective', 'persp', 'orthographic', 'ortho'], got {projection!r}."
        )

    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError(
            "Raytrace image dimensions must be positive, "
            f"got width={image_width}, height={image_height}."
        )
    if int(cycles_samples) <= 0:
        raise ValueError(f"cycles_samples must be positive, got {cycles_samples}.")

    labeled_mask = labels >= 0
    if not np.any(labeled_mask):
        raise ValueError("Raytracing requires at least one non-noise cluster point.")
    coords_labeled = coords_arr[labeled_mask]
    labels_labeled = labels[labeled_mask]
    sample_indices = _sample_indices_stratified(labels_labeled, max_points, random_seed=0)
    coords_plot = coords_labeled[sample_indices]
    labels_plot = labels_labeled[sample_indices]
    unique_labels = sorted(int(v) for v in np.unique(labels_plot) if int(v) >= 0)
    missing_colors = [cluster_id for cluster_id in unique_labels if cluster_id not in color_map]
    if missing_colors:
        raise KeyError(f"Blender raytrace is missing colors for clusters {missing_colors}.")

    bbox_min = np.min(coords_arr, axis=0)
    bbox_max = np.max(coords_arr, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(bbox_diag) or bbox_diag <= 0.0:
        raise ValueError(
            "Blender raytracing requires coordinates with a nonzero finite bounding box, "
            f"got bbox_min={bbox_min.tolist()}, bbox_max={bbox_max.tolist()}."
        )
    # Keep physical sizing anchored to the full labeled cloud so downsampling
    # and subset views do not inflate sphere diameter.
    radius_ref_points = coords_arr[labels >= 0]
    auto_radius_world = _estimate_ball_radius_world(
        radius_ref_points,
        sample_limit=1024,
        random_seed=0,
    )
    # The configured fraction scales the data-driven radius relative to its
    # reference value in the checked-in analysis configs.
    if not np.isfinite(sphere_radius_fraction) or float(sphere_radius_fraction) <= 0.0:
        raise ValueError(
            "sphere_radius_fraction must be finite and positive, "
            f"got {sphere_radius_fraction}."
        )
    user_radius_scale = float(sphere_radius_fraction) / 0.0105
    sphere_radius_world = float(auto_radius_world * user_radius_scale)
    wireframe_width_world = float(wireframe_width_fraction) * bbox_diag

    clusters_payload: list[dict[str, Any]] = []
    for cluster_id in unique_labels:
        cmask = labels_plot == cluster_id
        pts = coords_plot[cmask]
        if pts.shape[0] == 0:
            continue
        color_rgb = np.asarray(
            mcolors.to_rgb(str(color_map[cluster_id])),
            dtype=np.float32,
        )
        color_rgb = _boost_saturation(color_rgb[None, :], 1.08)[0]
        clusters_payload.append(
            {
                "cluster_id": int(cluster_id),
                "color": [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]), 1.0],
                "points": np.round(pts.astype(np.float64), 6).tolist(),
            }
        )
    normalized_jobs: list[dict[str, Any]] = []
    render_metadata: list[dict[str, Any]] = []
    out_files: list[Path] = []
    for job_index, job in enumerate(render_jobs):
        missing_keys = [
            key
            for key in ("out_file", "title", "view_elev", "view_azim")
            if key not in job
        ]
        if missing_keys:
            raise KeyError(
                f"Raytrace render job {job_index} is missing required keys {missing_keys}."
            )
        out_file = Path(job["out_file"])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        visible_value = job.get("visible_cluster_ids")
        visible_ids = (
            None
            if visible_value is None
            else sorted(set(int(cluster_id) for cluster_id in visible_value))
        )
        if visible_ids is None:
            visible_mask = labeled_mask
            rendered_mask = np.ones(labels_plot.shape[0], dtype=bool)
        else:
            visible_arr = np.asarray(visible_ids, dtype=int)
            visible_mask = labeled_mask & np.isin(labels, visible_arr)
            rendered_mask = np.isin(labels_plot, visible_arr)
        if not np.any(visible_mask):
            raise ValueError(
                f"Raytrace render job {job_index} ({out_file}) has no visible points; "
                f"visible_cluster_ids={visible_ids}."
            )
        if not np.any(rendered_mask):
            raise ValueError(
                f"Raytrace sampling removed every visible point for job {job_index} "
                f"({out_file}); visible_cluster_ids={visible_ids}, max_points={max_points}."
            )
        clusters_rendered = sorted(
            int(cluster_id) for cluster_id in np.unique(labels_plot[rendered_mask])
        )
        normalized_jobs.append(
            {
                "out_file": str(out_file),
                "title": str(job["title"]),
                "visible_cluster_ids": visible_ids,
                "view_elev": float(job["view_elev"]),
                "view_azim": float(job["view_azim"]),
            }
        )
        render_metadata.append(
            {
                "out_file": str(out_file),
                "num_points_total": int(coords_arr.shape[0]),
                "num_points_visible": int(np.count_nonzero(visible_mask)),
                "num_points_rendered": int(np.count_nonzero(rendered_mask)),
                "clusters_rendered": clusters_rendered,
                "view_elev": float(job["view_elev"]),
                "view_azim": float(job["view_azim"]),
            }
        )
        out_files.append(out_file)
    if len(set(out_files)) != len(out_files):
        raise ValueError(
            "Raytrace render jobs must use distinct output files, "
            f"got {[str(path) for path in out_files]}."
        )

    blender_exec = _resolve_blender_executable(blender_executable)

    payload = {
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "clusters": clusters_payload,
        "render": {
            "image_width": int(image_width),
            "image_height": int(image_height),
            "projection": str(projection_norm),
            "perspective_fov_deg": float(perspective_fov_deg),
            "camera_distance_factor": float(camera_distance_factor),
            "sphere_radius_world": float(sphere_radius_world),
            "cycles_samples": int(cycles_samples),
            "use_denoise": bool(use_denoise),
            "use_gpu": bool(use_gpu),
            "wireframe_enabled": bool(wireframe_enabled),
            "wireframe_width_world": float(wireframe_width_world),
            "wireframe_color": [0.12, 0.12, 0.12, 1.0],
            "background_color": [1.0, 1.0, 1.0, 1.0],
            "background_strength": 1.0,
        },
        "renders": normalized_jobs,
    }

    _run_blender_render_batch(
        blender_exec, Path(__file__).with_name("blender_scene.py"), payload, out_files,
        timeout_seconds=int(timeout_seconds),
        tmp_prefix="md_raytrace_blender_",
    )
    for out_file in out_files:
        _log_saved_figure(out_file)

    for metadata in render_metadata:
        metadata.update(
            {
                "projection": str(projection_norm),
                "image_size": (int(image_width), int(image_height)),
                "cycles_samples": int(cycles_samples),
                "use_denoise": bool(use_denoise),
                "use_gpu": bool(use_gpu),
                "sphere_radius_reference_points": int(radius_ref_points.shape[0]),
                "sphere_radius_world_auto": float(auto_radius_world),
                "sphere_radius_user_scale": float(user_radius_scale),
                "sphere_radius_world": float(sphere_radius_world),
                "color_saturation_boost": 1.0,
                "color_contrast_boost": 1.0,
                "blender_executable": str(blender_exec),
                "render_mode": "raytrace_blender_batch",
                "batch_size": int(len(render_jobs)),
            }
        )
    return render_metadata
