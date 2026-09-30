"""MD cluster scene executed by Blender's Python, using a JSON render payload."""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _principled_input(node, names):
    return next((node.inputs[name] for name in names if name in node.inputs), None)


def _build_material(name, rgba):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    base_color = (*map(float, rgba[:3]), 1.0)
    _principled_input(bsdf, ["Base Color"]).default_value = base_color
    for names, value in (
        (["Subsurface", "Subsurface Weight"], 0.05),
        (["Subsurface Color"], base_color),
        (["Subsurface Radius"], (1.0, 0.35, 0.22)),
        (["Roughness"], 0.34),
        (["Specular", "Specular IOR Level"], 0.32),
    ):
        socket = _principled_input(bsdf, names)
        if socket is not None:
            socket.default_value = value
    return mat


def _link_object(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _add_area_light(name, center, location, energy, size):
    light = bpy.data.lights.new(name=name, type="AREA")
    light.energy = float(energy)
    light.size = float(size)
    obj = _link_object(name, light)
    obj.location = Vector(location)
    obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()


def _add_wireframe_box(bbox_min, bbox_max, width_world, color_rgba):
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
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
    for a, b in edges:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        for point, corner in zip(spline.points, (corners[a], corners[b])):
            point.co = (*corner, 1.0)
    obj = _link_object("MDBoxWire", curve)
    obj.data.materials.append(_build_material("MDBoxWireMat", color_rgba))


def _position_camera(camera, center, span, bbox_corners, cfg, render_spec):
    elev = math.radians(float(render_spec["view_elev"]))
    azim = math.radians(float(render_spec["view_azim"]))
    direction = Vector((
        math.cos(elev) * math.cos(azim),
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
    ))
    camera.location = center + direction * float(cfg["camera_distance_factor"]) * span
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    corner_distance = max(float((corner - camera.location).length) for corner in bbox_corners)
    camera.data.clip_end = max(1000.0, 1.1 * corner_distance)


def _configure_gpu(scene):
    addon = bpy.context.preferences.addons.get("cycles", None)
    if addon is None:
        raise RuntimeError("Cycles addon is unavailable; cannot enable GPU raytracing.")
    prefs = addon.preferences
    gpu_backend = None
    backend_errors = []
    candidates = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "OPENCL"]
    for backend in candidates:
        try:
            prefs.compute_device_type = backend
        except Exception as exc:
            backend_errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            continue
        prefs.get_devices()
        if any(getattr(device, "type", "CPU") != "CPU" for device in prefs.devices):
            gpu_backend = backend
            break
    if gpu_backend is None:
        prefs.get_devices()
    gpu_count = 0
    for device in prefs.devices:
        is_gpu = str(getattr(device, "type", "CPU")) != "CPU"
        device.use = is_gpu
        gpu_count += int(is_gpu)
    if gpu_count <= 0:
        raise RuntimeError(
            "use_gpu=True was requested but no GPU Cycles device is available. "
            f"Tried backends={candidates}, backend_errors={backend_errors}."
        )
    scene.cycles.device = "GPU"
    print(f"INFO: Cycles GPU mode enabled (backend={gpu_backend}, devices={gpu_count}).")


def _configure_scene(cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(cfg["image_width"])
    scene.render.resolution_y = int(cfg["image_height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = int(cfg["cycles_samples"])
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    if not hasattr(scene.cycles, "use_denoising"):
        raise RuntimeError("This Blender Cycles version does not expose use_denoising.")
    scene.cycles.use_denoising = bool(cfg["use_denoise"])
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if bool(cfg.get("use_gpu", False)):
        _configure_gpu(scene)
    else:
        scene.cycles.device = "CPU"

    world = bpy.data.worlds.new("MDWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background", None)
    if background is None:
        raise RuntimeError("Blender world background node is missing.")
    background.inputs[0].default_value = tuple(float(v) for v in cfg["background_color"])
    background.inputs[1].default_value = float(cfg.get("background_strength", 1.0))
    return scene


def _build_clusters(clusters, radius, span):
    objects = {}
    pointcloud_supported = False
    if hasattr(bpy.data, "pointclouds"):
        probe = bpy.data.pointclouds.new("MDPointCloudProbe")
        pointcloud_supported = hasattr(probe.points, "add")
        bpy.data.pointclouds.remove(probe)
    if not pointcloud_supported:
        print("INFO: PointCloud points.add API unavailable; using mesh-vertex instancing fallback.")
    offset = Vector((max(50.0 * span, 10.0), 0.0, 0.0))
    for cluster in clusters:
        points = cluster["points"]
        if not points:
            continue
        cid = int(cluster["cluster_id"])
        name = f"Cluster_{cid:02d}"
        if pointcloud_supported:
            data = bpy.data.pointclouds.new(f"{name}_Points")
            data.points.add(len(points))
            data.points.foreach_set("co", [float(v) for point in points for v in point])
            data.points.foreach_set("radius", [float(radius)] * len(points))
            material_object = _link_object(name, data)
            objects[cid] = [material_object]
        else:
            # Blender 5.x removed PointCloud.points.add(); use vertex instancing.
            mesh = bpy.data.meshes.new(f"{name}_Verts")
            shifted = [tuple(float(point[i]) - float(offset[i]) for i in range(3)) for point in points]
            mesh.from_pydata(shifted, [], [])
            mesh.update()
            instancer = _link_object(f"{name}_Instancer", mesh)
            instancer.instance_type = "VERTS"
            instancer.show_instancer_for_render = False
            instancer.show_instancer_for_viewport = False
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2, radius=float(radius), location=tuple(float(v) for v in offset),
            )
            material_object = bpy.context.active_object
            if material_object is None:
                raise RuntimeError(f"Failed to create template sphere for cluster {cid}.")
            material_object.name = f"{name}_TemplateSphere"
            material_object.parent = instancer
            material_object.matrix_parent_inverse = instancer.matrix_world.inverted()
            bpy.ops.object.shade_smooth()
            objects[cid] = [instancer, material_object]
        material_object.data.materials.append(_build_material(f"{name}_Mat", cluster["color"]))
    return objects


def _ensure_render_output(scene, out_path, render_index):
    if out_path.exists():
        return
    resolved = Path(bpy.path.abspath(scene.render.frame_path(frame=scene.frame_current)))
    if resolved.exists():
        shutil.copy2(resolved, out_path)
        return
    candidates = sorted(out_path.parent.glob(f"{out_path.stem}*{out_path.suffix}"))
    if len(candidates) == 1 and candidates[0].exists():
        shutil.copy2(candidates[0], out_path)
        return
    raise FileNotFoundError(
        "Blender render finished but output is missing. "
        f"render_index={render_index}, expected={out_path}, resolved={resolved}, "
        f"candidates={[str(p) for p in candidates]}."
    )


def render(payload):
    cfg = payload["render"]
    scene = _configure_scene(cfg)
    bbox_min = Vector(tuple(float(v) for v in payload["bbox_min"]))
    bbox_max = Vector(tuple(float(v) for v in payload["bbox_max"]))
    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min
    span = max(float(extent.x), float(extent.y), float(extent.z))
    if not math.isfinite(span) or span <= 1e-8:
        span = 1.0

    camera_data = bpy.data.cameras.new("MDCamera")
    camera = _link_object("MDCamera", camera_data)
    scene.camera = camera
    projection = str(cfg["projection"]).lower()
    if projection in {"perspective", "persp"}:
        camera_data.type = "PERSP"
        camera_data.lens_unit = "FOV"
        camera_data.angle = math.radians(float(cfg["perspective_fov_deg"]))
    elif projection in {"orthographic", "ortho"}:
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = 2.25 * span
    else:
        raise ValueError(f"Unsupported projection mode for Blender render: {projection!r}.")
    corners = [
        Vector((x, y, z))
        for x in (bbox_min.x, bbox_max.x)
        for y in (bbox_min.y, bbox_max.y)
        for z in (bbox_min.z, bbox_max.z)
    ]
    light_size = 0.65 * span
    for name, offset, energy, size in (
        ("KeyLight", (1.9 * span, -1.7 * span, 2.1 * span), 600.0, light_size),
        ("FillLight", (-2.2 * span, 1.6 * span, 0.9 * span), 170.0, 0.9 * light_size),
        ("RimLight", (-0.4 * span, -2.0 * span, 1.8 * span), 280.0, 0.8 * light_size),
    ):
        _add_area_light(name, center, center + Vector(offset), energy, size)
    clusters = _build_clusters(payload["clusters"], cfg["sphere_radius_world"], span)
    if bool(cfg.get("wireframe_enabled", True)) and float(cfg.get("wireframe_width_world", 0.0)) > 0.0:
        _add_wireframe_box(bbox_min, bbox_max, float(cfg["wireframe_width_world"]), cfg["wireframe_color"])

    for render_index, spec in enumerate(payload["renders"]):
        visible = spec.get("visible_cluster_ids")
        visible_ids = None if visible is None else set(int(cid) for cid in visible)
        for cid, objects in clusters.items():
            for obj in objects:
                obj.hide_render = visible_ids is not None and cid not in visible_ids
        _position_camera(camera, center, span, corners, cfg, spec)
        out_path = Path(str(spec["out_file"])).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(out_path)
        result = bpy.ops.render.render(write_still=True)
        if "FINISHED" not in set(result):
            raise RuntimeError(
                "Blender render operator did not finish successfully for "
                f"render {render_index} ({out_path}): result={result}."
            )
        _ensure_render_output(scene, out_path, render_index)


def main():
    if "--" not in sys.argv:
        raise RuntimeError("Blender script expected '--' argument separator.")
    parser = argparse.ArgumentParser(description="Raytrace MD clusters via Blender.")
    parser.add_argument("--payload_json", type=str, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    payload_path = Path(args.payload_json)
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload JSON is missing: {payload_path}")
    render(json.loads(payload_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
