"""Blender Cycles raytracing for MD cluster snapshots."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np

from src.training_methods.contrastive_learning._cluster_colors import _boost_saturation
from src.training_methods.contrastive_learning._cluster_geometry import _estimate_ball_radius_world


def _run_blender_render(
    blender_exec: str,
    blender_script: str,
    payload: dict,
    out_file: Path,
    *,
    timeout_seconds: int,
    tmp_prefix: str = "blender_render_",
) -> None:
    """Run an inline Blender script with a JSON payload and verify output."""
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp_dir:
        tmp_root = Path(tmp_dir)
        payload_path = tmp_root / "payload.json"
        script_path = tmp_root / "render.py"
        payload_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        script_path.write_text(blender_script, encoding="utf-8")

        cmd = [
            blender_exec, "-b", "--factory-startup",
            "-P", str(script_path),
            "--", "--payload_json", str(payload_path),
        ]
        try:
            proc = subprocess.run(
                cmd, check=False, capture_output=True, text=True,
                timeout=int(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Blender render timed out after {timeout_seconds}s for {out_file}."
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"Blender render failed (exit {proc.returncode}) for {out_file}.\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )
        if "Traceback (most recent call last):" in proc.stderr:
            raise RuntimeError(
                f"Blender script raised an exception for {out_file}.\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )

    if not out_file.exists():
        candidates = sorted(out_file.parent.glob(f"{out_file.stem}*{out_file.suffix}"))
        raise FileNotFoundError(
            f"Blender render succeeded but output missing: {out_file}, "
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


def _save_md_cluster_snapshot_raytrace_blender(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: dict[int, str],
    out_file: Path,
    *,
    title: str,
    visible_cluster_ids: list[int] | None = None,
    max_points: int | None = None,
    view_elev: float = 24.0,
    view_azim: float = 35.0,
    image_width: int = 1600,
    image_height: int = 1600,
    projection: str = "perspective",
    perspective_fov_deg: float = 34.0,
    camera_distance_factor: float = 2.8,
    sphere_radius_fraction: float = 0.0105,
    blender_executable: str = "blender",
    cycles_samples: int = 64,
    use_denoise: bool = True,
    use_gpu: bool = False,
    timeout_seconds: int = 1200,
    wireframe_enabled: bool = True,
    wireframe_width_fraction: float = 0.0017,
) -> dict[str, Any]:
    """Render a raytraced MD snapshot with Blender Cycles.

    This is an additive renderer used alongside the existing matplotlib
    outputs. It requires a Blender executable.
    """
    coords_arr = np.asarray(coords, dtype=np.float32)[:, :3]
    labels = np.asarray(cluster_labels, dtype=int).reshape(-1)
    projection_norm = str(projection).strip().lower()
    if projection_norm not in {"perspective", "persp", "orthographic", "ortho"}:
        projection_norm = "perspective"

    mask = labels >= 0
    if visible_cluster_ids is not None:
        visible = np.asarray(sorted(set(int(v) for v in visible_cluster_ids)), dtype=int)
        mask &= np.isin(labels, visible)
    if not np.any(mask):
        raise ValueError("No points remained after applying cluster visibility filters.")

    coords_use = coords_arr[mask]
    labels_use = labels[mask]
    coords_plot = coords_use
    labels_plot = labels_use
    unique_labels = sorted(int(v) for v in np.unique(labels_plot) if int(v) >= 0)

    bbox_min = np.min(coords_arr, axis=0)
    bbox_max = np.max(coords_arr, axis=0)
    bbox_diag = max(1e-8, float(np.linalg.norm(bbox_max - bbox_min)))
    # Keep physical sizing anchored to the full labeled cloud so downsampling
    # and subset views do not inflate sphere diameter.
    radius_ref_points = coords_arr[labels >= 0]
    auto_radius_world = _estimate_ball_radius_world(
        radius_ref_points,
        sample_limit=1024,
        random_seed=0,
    )
    # Keep backward compatibility with existing config value while making size
    # physically data-driven by default.
    user_radius_scale = max(1e-6, float(sphere_radius_fraction) / 0.0105)
    sphere_radius_world = max(1e-9, float(auto_radius_world * user_radius_scale))
    wireframe_width_world = float(wireframe_width_fraction) * bbox_diag

    clusters_payload: list[dict[str, Any]] = []
    for cluster_id in unique_labels:
        cmask = labels_plot == cluster_id
        pts = coords_plot[cmask]
        if pts.shape[0] == 0:
            continue
        color_rgb = np.asarray(
            mcolors.to_rgb(str(color_map.get(cluster_id, "#777777"))),
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
    blender_exec = _resolve_blender_executable(blender_executable)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": str(title),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "clusters": clusters_payload,
        "render": {
            "image_width": int(image_width),
            "image_height": int(image_height),
            "projection": str(projection_norm),
            "view_elev": float(view_elev),
            "view_azim": float(view_azim),
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
        "out_file": str(out_file),
    }

    blender_script = textwrap.dedent(
        """
        import argparse
        import json
        import math
        import shutil
        import sys
        from pathlib import Path

        import bpy
        from mathutils import Vector


        def _parse_args():
            argv = sys.argv
            if "--" not in argv:
                raise RuntimeError("Blender script expected '--' argument separator.")
            argv = argv[argv.index("--") + 1 :]
            p = argparse.ArgumentParser(description="Raytrace MD clusters via Blender.")
            p.add_argument("--payload_json", type=str, required=True)
            return p.parse_args(argv)


        def _principled_input(node, names):
            for name in names:
                if name in node.inputs:
                    return node.inputs[name]
            return None

        def _build_material(name, rgba):
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            out.location = (300, 0)
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

            base_col = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
            _principled_input(bsdf, ["Base Color"]).default_value = base_col

            sss_inp = _principled_input(bsdf, ["Subsurface", "Subsurface Weight"])
            if sss_inp is not None:
                sss_inp.default_value = 0.05
            sss_col = _principled_input(bsdf, ["Subsurface Color"])
            if sss_col is not None:
                sss_col.default_value = base_col
            sss_rad = _principled_input(bsdf, ["Subsurface Radius"])
            if sss_rad is not None:
                sss_rad.default_value = (1.0, 0.35, 0.22)

            rough = _principled_input(bsdf, ["Roughness"])
            if rough is not None:
                rough.default_value = 0.34
            spec = _principled_input(bsdf, ["Specular", "Specular IOR Level"])
            if spec is not None:
                spec.default_value = 0.32
            return mat


        def _add_area_light(name, center, location, energy, size):
            light_data = bpy.data.lights.new(name=name, type="AREA")
            light_data.energy = float(energy)
            light_data.size = float(size)
            obj = bpy.data.objects.new(name, light_data)
            bpy.context.scene.collection.objects.link(obj)
            obj.location = Vector(location)
            direction = center - obj.location
            obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            return obj


        def _add_wireframe_box(bbox_min, bbox_max, width_world, color_rgba):
            x0, y0, z0 = bbox_min
            x1, y1, z1 = bbox_max
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            curve = bpy.data.curves.new("MDBoxEdges", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = max(float(width_world), 1e-6)
            curve.bevel_resolution = 1
            for a_idx, b_idx in edges:
                spline = curve.splines.new("POLY")
                spline.points.add(1)
                xa, ya, za = corners[a_idx]
                xb, yb, zb = corners[b_idx]
                spline.points[0].co = (xa, ya, za, 1.0)
                spline.points[1].co = (xb, yb, zb, 1.0)
            obj = bpy.data.objects.new("MDBoxWire", curve)
            bpy.context.scene.collection.objects.link(obj)
            mat = _build_material("MDBoxWireMat", color_rgba)
            obj.data.materials.append(mat)
            return obj


        def main():
            args = _parse_args()
            payload_path = Path(args.payload_json)
            if not payload_path.exists():
                raise FileNotFoundError(f"Payload JSON is missing: {payload_path}")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            cfg = payload["render"]

            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.render.resolution_x = int(cfg["image_width"])
            scene.render.resolution_y = int(cfg["image_height"])
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "PNG"
            scene.render.film_transparent = False
            out_path = Path(str(payload["out_file"])).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(out_path)
            scene.cycles.samples = int(cfg["cycles_samples"])
            if hasattr(scene.cycles, "use_adaptive_sampling"):
                scene.cycles.use_adaptive_sampling = True
            if hasattr(scene.cycles, "use_denoising"):
                scene.cycles.use_denoising = bool(cfg["use_denoise"])
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
            scene.view_settings.exposure = 0.0
            scene.view_settings.gamma = 1.0

            if bool(cfg.get("use_gpu", False)):
                prefs = bpy.context.preferences
                addon = prefs.addons.get("cycles", None)
                if addon is None:
                    raise RuntimeError("Cycles addon is unavailable; cannot enable GPU raytracing.")
                cprefs = addon.preferences
                cprefs.get_devices()
                gpu_count = 0
                for dev in cprefs.devices:
                    dev.use = True
                    if dev.type != "CPU":
                        gpu_count += 1
                if gpu_count <= 0:
                    raise RuntimeError(
                        "use_gpu=True was requested but no GPU Cycles device is available."
                    )
                scene.cycles.device = "GPU"
            else:
                scene.cycles.device = "CPU"

            world = bpy.data.worlds.new("MDWorld")
            scene.world = world
            world.use_nodes = True
            bg = world.node_tree.nodes.get("Background", None)
            if bg is None:
                raise RuntimeError("Blender world background node is missing.")
            bg.inputs[0].default_value = tuple(float(v) for v in cfg["background_color"])
            bg.inputs[1].default_value = float(cfg.get("background_strength", 1.0))

            bbox_min = Vector(tuple(float(v) for v in payload["bbox_min"]))
            bbox_max = Vector(tuple(float(v) for v in payload["bbox_max"]))
            center = 0.5 * (bbox_min + bbox_max)
            extent = bbox_max - bbox_min
            span = max(float(extent.x), float(extent.y), float(extent.z))
            if not math.isfinite(span) or span <= 1e-8:
                span = 1.0

            camera_data = bpy.data.cameras.new("MDCamera")
            camera_obj = bpy.data.objects.new("MDCamera", camera_data)
            scene.collection.objects.link(camera_obj)
            scene.camera = camera_obj

            elev = math.radians(float(cfg["view_elev"]))
            azim = math.radians(float(cfg["view_azim"]))
            cam_dir = Vector(
                (
                    math.cos(elev) * math.cos(azim),
                    math.cos(elev) * math.sin(azim),
                    math.sin(elev),
                )
            )
            cam_dist = float(cfg["camera_distance_factor"]) * span
            camera_obj.location = center + cam_dir * cam_dist
            camera_obj.rotation_euler = (center - camera_obj.location).to_track_quat("-Z", "Y").to_euler()
            proj = str(cfg["projection"]).lower()
            if proj in {"perspective", "persp"}:
                camera_data.type = "PERSP"
                camera_data.lens_unit = "FOV"
                camera_data.angle = math.radians(float(cfg["perspective_fov_deg"]))
            elif proj in {"orthographic", "ortho"}:
                camera_data.type = "ORTHO"
                camera_data.ortho_scale = 2.25 * span
            else:
                raise ValueError(f"Unsupported projection mode for Blender render: {proj!r}.")

            light_size = 0.65 * span
            _add_area_light(
                "KeyLight",
                center,
                center + Vector((1.9 * span, -1.7 * span, 2.1 * span)),
                energy=600.0,
                size=light_size,
            )
            _add_area_light(
                "FillLight",
                center,
                center + Vector((-2.2 * span, 1.6 * span, 0.9 * span)),
                energy=170.0,
                size=0.9 * light_size,
            )
            _add_area_light(
                "RimLight",
                center,
                center + Vector((-0.4 * span, -2.0 * span, 1.8 * span)),
                energy=280.0,
                size=0.8 * light_size,
            )

            pointcloud_add_supported = False
            if hasattr(bpy.data, "pointclouds"):
                probe_pc = bpy.data.pointclouds.new("MDPointCloudProbe")
                pointcloud_add_supported = hasattr(probe_pc.points, "add")
                bpy.data.pointclouds.remove(probe_pc)

            if pointcloud_add_supported:
                for cluster in payload["clusters"]:
                    pts = cluster["points"]
                    if len(pts) == 0:
                        continue
                    cid = int(cluster["cluster_id"])
                    pc_data = bpy.data.pointclouds.new(f"Cluster_{cid:02d}_Points")
                    pc_data.points.add(len(pts))
                    co_flat = [float(v) for p in pts for v in p]
                    rad_flat = [float(cfg["sphere_radius_world"])] * len(pts)
                    pc_data.points.foreach_set("co", co_flat)
                    pc_data.points.foreach_set("radius", rad_flat)

                    obj = bpy.data.objects.new(f"Cluster_{cid:02d}", pc_data)
                    scene.collection.objects.link(obj)
                    mat = _build_material(f"Cluster_{cid:02d}_Mat", cluster["color"])
                    obj.data.materials.append(mat)
            else:
                # Blender 5.x removed PointCloud.points.add(). Use fast
                # vertex-instancing fallback that works in Cycles.
                print(
                    "INFO: PointCloud points.add API unavailable; "
                    "using mesh-vertex instancing fallback."
                )
                inst_offset = Vector((max(50.0 * span, 10.0), 0.0, 0.0))
                sphere_radius = float(cfg["sphere_radius_world"])
                for cluster in payload["clusters"]:
                    pts = cluster["points"]
                    if len(pts) == 0:
                        continue
                    cid = int(cluster["cluster_id"])
                    mesh = bpy.data.meshes.new(f"Cluster_{cid:02d}_Verts")
                    shifted_pts = [
                        (
                            float(p[0]) - float(inst_offset.x),
                            float(p[1]) - float(inst_offset.y),
                            float(p[2]) - float(inst_offset.z),
                        )
                        for p in pts
                    ]
                    mesh.from_pydata(shifted_pts, [], [])
                    mesh.update()

                    instancer = bpy.data.objects.new(f"Cluster_{cid:02d}_Instancer", mesh)
                    scene.collection.objects.link(instancer)
                    instancer.instance_type = "VERTS"
                    instancer.show_instancer_for_render = False
                    instancer.show_instancer_for_viewport = False

                    bpy.ops.mesh.primitive_ico_sphere_add(
                        subdivisions=2,
                        radius=sphere_radius,
                        location=(float(inst_offset.x), float(inst_offset.y), float(inst_offset.z)),
                    )
                    sphere_obj = bpy.context.active_object
                    if sphere_obj is None:
                        raise RuntimeError(
                            f"Failed to create template sphere for cluster {cid}."
                        )
                    sphere_obj.name = f"Cluster_{cid:02d}_TemplateSphere"
                    sphere_obj.parent = instancer
                    sphere_obj.matrix_parent_inverse = instancer.matrix_world.inverted()
                    bpy.ops.object.shade_smooth()

                    mat = _build_material(f"Cluster_{cid:02d}_Mat", cluster["color"])
                    sphere_obj.data.materials.append(mat)

            if bool(cfg.get("wireframe_enabled", True)) and float(cfg.get("wireframe_width_world", 0.0)) > 0.0:
                _add_wireframe_box(
                    bbox_min,
                    bbox_max,
                    float(cfg["wireframe_width_world"]),
                    cfg["wireframe_color"],
                )

            result = bpy.ops.render.render(write_still=True)
            if "FINISHED" not in set(result):
                raise RuntimeError(
                    "Blender render operator did not finish successfully: "
                    f"result={result}."
                )

            # Blender may resolve output path with frame tokens depending on
            # internal render settings/version. Ensure requested output exists.
            expected = out_path
            if not expected.exists():
                resolved = Path(
                    bpy.path.abspath(scene.render.frame_path(frame=scene.frame_current))
                )
                if resolved.exists():
                    shutil.copy2(resolved, expected)
                else:
                    candidates = sorted(
                        expected.parent.glob(f"{expected.stem}*{expected.suffix}")
                    )
                    if len(candidates) == 1 and candidates[0].exists():
                        shutil.copy2(candidates[0], expected)
                    else:
                        raise FileNotFoundError(
                            "Blender render finished but output is missing. "
                            f"expected={expected}, resolved={resolved}, "
                            f"candidates={[str(p) for p in candidates]}."
                        )


        if __name__ == "__main__":
            main()
        """
    )

    _run_blender_render(
        blender_exec, blender_script, payload, out_file,
        timeout_seconds=int(timeout_seconds),
        tmp_prefix="md_raytrace_blender_",
    )

    return {
        "out_file": str(out_file),
        "num_points_total": int(coords_arr.shape[0]),
        "num_points_visible": int(coords_use.shape[0]),
        "num_points_rendered": int(coords_plot.shape[0]),
        "clusters_rendered": unique_labels,
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "projection": str(projection_norm),
        "image_size": (int(image_width), int(image_height)),
        "cycles_samples": int(cycles_samples),
        "sphere_radius_reference_points": int(radius_ref_points.shape[0]),
        "sphere_radius_world_auto": float(auto_radius_world),
        "sphere_radius_user_scale": float(user_radius_scale),
        "sphere_radius_world": float(sphere_radius_world),
        "color_saturation_boost": 1.0,
        "color_contrast_boost": 1.0,
        "blender_executable": str(blender_exec),
        "render_mode": "raytrace_blender",
    }
