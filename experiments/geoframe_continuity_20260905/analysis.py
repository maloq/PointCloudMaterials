"""Controlled continuity interventions on the September 5 GFv2 checkpoints."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_utils.spatiotemporal_views import periodic_tree
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.models.encoders.ri_mae_encoder import (
    _farthest_point_sample, _index_points, _knn_point, RIMAEBackbone,
)
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_paths(cfg, output):
    manifest = json.loads(Path(cfg["views_manifest"]).read_text())
    rng = np.random.default_rng(cfg["seed"])
    endpoints, records, identities = [], [], []
    a, b = cfg["anchor_frame"], cfg["anchor_frame"] + cfg["lag_frames"]
    for source in manifest["sources"]:
        trajectory = TemporalLAMMPSBinaryTrajectory.load(source["path"])
        count = cfg["centers_Ta"] if source["material"] == "Ta" else cfg["centers_per_Al_Mg_branch"]
        # This is a held-out central-ID pool from the existing cache protocol.
        centers = rng.choice(np.arange(0, trajectory.atom_count, 5), count, replace=False)
        length0 = trajectory.box_high[a] - trajectory.box_low[a]
        p0, tree0 = periodic_tree(trajectory.positions[a], length0)
        _, indices = tree0.query(p0[centers], k=cfg["candidate_atoms"], workers=1)
        length1 = trajectory.box_high[b] - trajectory.box_low[b]
        p1, tree1 = periodic_tree(trajectory.positions[b], length1)
        _, nearest1 = tree1.query(p1[centers], k=80, workers=1)
        coverage = np.asarray([np.isin(ids, candidates).all() for ids, candidates in zip(nearest1, indices)])
        if not coverage.all():
            raise RuntimeError(f"Candidate pool misses endpoint neighbors: {source['path']}, centers={centers[~coverage]}")
        x0 = p0[indices].astype(np.float64) - p0[centers, None]
        x0 -= length0 * np.round(x0 / length0)
        x1 = p1[indices].astype(np.float64) - p1[centers, None]
        x1 -= length1 * np.round(x1 / length1)
        if np.max(np.abs(x1 - x0)) > 0.25 * min(length0.min(), length1.min()):
            raise RuntimeError(f"Ambiguous local periodic interpolation: {source['path']}")
        if not np.array_equal(indices[:, 0], centers):
            raise RuntimeError(f"Tracked center is not first nearest atom: {source['path']}")
        endpoints.extend(np.stack([x0, x1], axis=1) / source["radius"])
        identities.extend(trajectory.atom_ids[indices])
        for i, center in enumerate(centers):
            records.append(dict(material=source["material"], snapshot=source["snapshot"],
                                source=source["path"], source_dtype=source["storage_dtype"],
                                source_manifest_sha256=source["manifest_sha256"],
                                center_atom_id=int(trajectory.atom_ids[center]), radius_A=source["radius"],
                                frame0=a, frame1=b, time0_ps=a * .1, time1_ps=b * .1,
                                endpoint_outer_retention=float(np.isin(nearest1[i], indices[i, :80]).mean())))
        print(f"Prepared {source['material']}/{source['snapshot']}: {count} paths", flush=True)
    values = np.asarray(endpoints, dtype=np.float32)
    np.savez_compressed(output / "inputs.npz", endpoints=values, atom_ids=np.asarray(identities))
    write_json(output / "paths.json", records)
    return values, records


def load_model(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["hyper_parameters"])
    cfg.compile_encoder = False
    model = VICRegModule(cfg)
    model.load_state_dict({k.replace("encoder._orig_mod.", "encoder."): v
                           for k, v in payload["state_dict"].items()}, strict=True)
    model = model.cuda().eval().requires_grad_(False)
    group = model.encoder.token_encoder.group_divider
    assert group.deterministic_fps and group.sorting_mode == "none"
    assert model.encoder.token_encoder.frame_builder == "triad"
    assert not model.encoder.token_encoder.use_frame_gating
    return model


def groups(points):
    centered = points - points.mean(1, keepdim=True)
    ci = _farthest_point_sample(centered, 24, deterministic=True)
    centers = _index_points(centered, ci)
    gi = _knn_point(16, centered, centers)
    neighborhood = _index_points(centered, gi) - centers.unsqueeze(2)
    return neighborhood, centers, ci, gi


def gather_groups(points, ci, gi):
    centered = points - points.mean(1, keepdim=True)
    centers = _index_points(centered, ci)
    return _index_points(centered, gi) - centers.unsqueeze(2), centers


def transport_frames(reference, current, frames):
    # Row-vector Kabsch: reference @ Q approximates current; F_current=Q.T @ F_reference.
    # Each patch is expressed relative to its tracked group center, not its mean.
    cross = reference.double().transpose(-1, -2) @ current.double()
    u, _, vh = torch.linalg.svd(cross)
    sign = torch.ones_like(u[..., 0, :])
    sign[..., 2] = torch.linalg.det(u @ vh)
    q = (u * sign.unsqueeze(-2)) @ vh
    return (q.transpose(-1, -2) @ frames.double()).float()


@contextmanager
def replace_frames(token_encoder, frames):
    original = token_encoder._frames_shape_and_chirality
    def overridden(self, neighborhood, centers):
        _, confidence, shape, chirality = original(neighborhood, centers)
        return frames, confidence, shape, chirality
    token_encoder._frames_shape_and_chirality = MethodType(overridden, token_encoder)
    try:
        yield
    finally:
        del token_encoder._frames_shape_and_chirality


def encode_groups(model, neighborhood, centers, frame_override=None):
    token_encoder = model.encoder.token_encoder
    if frame_override is None:
        tokens = token_encoder.encode_grouped_features(neighborhood, centers)
    else:
        with replace_frames(token_encoder, frame_override):
            tokens = token_encoder.encode_grouped_features(neighborhood, centers)
    features = model.encoder._pool_tokens(tokens)
    return torch.stack([features, model.vicreg.project_features(features)], dim=1)


def signatures(neighborhood):
    # Axis selection uses the exact repository implementation's arithmetic.
    patch = neighborhood.flatten(0, 1)
    rows = torch.arange(len(patch), device=patch.device)
    primary = patch.square().sum(-1).argmax(1)
    axis = RIMAEBackbone._normalize_frame_vectors(patch[rows, primary], eps=1e-6)
    axis = RIMAEBackbone._apply_axis_sign_convention(patch, axis)
    residual = patch - (patch * axis[:, None]).sum(-1, keepdim=True) * axis[:, None]
    secondary = residual.square().sum(-1).argmax(1)
    return torch.stack([primary, secondary], dim=-1).reshape(neighborhood.shape[0], 24, 2)


@torch.inference_mode()
def run_paths(model, endpoints, cfg, output, name):
    device = torch.device("cuda")
    x = torch.from_numpy(endpoints).to(device)
    n, steps = len(x), cfg["path_steps"]
    alpha = torch.linspace(0, 1, steps, device=device)
    base, base_c, ci, gi = groups(x[:, 0, :80])
    base_frames = model.encoder.token_encoder._frames_shape_and_chirality(base, base_c)[0]
    reconstructed = encode_groups(model, base, base_c)
    direct = model.encoder.forward_features(x[:, 0, :80])
    max_error = float((reconstructed[:, 0] - direct).abs().max())
    if max_error > 2e-6:
        raise RuntimeError(f"Grouped reconstruction disagrees with encoder: max error={max_error}")
    # Rigid motion control for the transported-frame intervention.
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    rotation[:, -1] *= torch.linalg.det(rotation)
    rotated_frames = transport_frames(base, base @ rotation, base_frames)
    transport_error = float((rotated_frames - rotation.T @ base_frames).abs().max())
    rotated = encode_groups(model, base @ rotation, base_c @ rotation, rotated_frames)
    rotation_error = float((rotated - reconstructed).abs().max())
    if transport_error > 2e-5 or rotation_error > 2e-4:
        raise RuntimeError(f"Transport rotation control failed: frame={transport_error}, embedding={rotation_error}")
    repeated = encode_groups(model, base, base_c)
    repeat_error = float((repeated - reconstructed).abs().max())
    if repeat_error != 0:
        raise RuntimeError(f"Repeat inference changed: {repeat_error}")
    modes = ["normal", "fixed_outer", "fixed_groups", "transported_frames", "fixed_frames"]
    arrays = {}
    batch = cfg["batch_size"]
    for mode in modes:
        embeddings, frame_values, axis_values, group_values = [], [], [], []
        for start in range(0, n * steps, batch):
            linear = torch.arange(start, min(start + batch, n * steps), device=device)
            rows, t = linear // steps, linear % steps
            current = x[rows, 0] + alpha[t, None, None] * (x[rows, 1] - x[rows, 0])
            if mode == "normal":
                oi = current.square().sum(-1).topk(80, largest=False, sorted=True).indices
                points = _index_points(current, oi)
                ng, centers, current_ci, current_gi = groups(points)
            elif mode == "fixed_outer":
                ng, centers, current_ci, current_gi = groups(current[:, :80])
            else:
                ng, centers = gather_groups(current[:, :80], ci[rows], gi[rows])
            frames = None
            if mode == "fixed_frames":
                frames = base_frames[rows]
            elif mode == "transported_frames":
                frames = transport_frames(base[rows], ng, base_frames[rows])
            values = encode_groups(model, ng, centers, frames)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Non-finite embeddings in {name}/{mode}")
            embeddings.append(values.cpu().numpy())
            if mode == "fixed_groups":
                frame_values.append(model.encoder.token_encoder._frames_shape_and_chirality(ng, centers)[0].cpu().numpy())
                axis_values.append(signatures(ng).cpu().numpy())
            if mode == "fixed_outer":
                group_values.append(torch.cat([current_ci, current_gi.flatten(1)], 1).cpu().numpy())
        arrays[mode] = np.concatenate(embeddings).reshape(n, steps, 2, 128)
        if frame_values:
            arrays["static_frames"] = np.concatenate(frame_values).reshape(n, steps, 24, 3, 3)
            arrays["axis_indices"] = np.concatenate(axis_values).reshape(n, steps, 24, 2)
        if group_values:
            arrays["group_indices"] = np.concatenate(group_values).reshape(n, steps, -1)
        print(f"{name}: completed {mode} ({n * steps} clouds)", flush=True)
    arrays["alpha"] = alpha.cpu().numpy()
    arrays["base_ci"] = ci.cpu().numpy()
    arrays["base_gi"] = gi.cpu().numpy()
    np.savez_compressed(output / f"{name}_paths.npz", **arrays)
    # All interventions must begin from the same grouping and frames.
    # Normal outer sorting may resolve exact distance ties differently, so it is reported separately.
    origin_errors = {mode: float(np.abs(arrays[mode][:, 0] - arrays["fixed_groups"][:, 0]).max()) for mode in modes}
    if max(origin_errors[mode] for mode in modes if mode != "normal") > 2e-4:
        raise RuntimeError(f"Intervention origins disagree: {origin_errors}")
    controls = dict(grouped_forward_max_abs_error=max_error, repeat_max_abs_error=repeat_error,
                    transported_rotation_frame_error=transport_error,
                    transported_rotation_embedding_error=rotation_error, origin_errors=origin_errors)
    write_json(output / f"{name}_controls.json", controls)
    return arrays


@torch.inference_mode()
def refine_jumps(model, endpoints, records, arrays, cfg, output, name):
    # Select one largest encoder step per path under fixed grouping, then locate its frame switch.
    x = torch.from_numpy(endpoints[:, :, :80]).cuda()
    ci = torch.from_numpy(arrays["base_ci"]).cuda()
    gi = torch.from_numpy(arrays["base_gi"]).cuda()
    step_drift = np.linalg.norm(np.diff(arrays["fixed_groups"][:, :, 0].astype(np.float64), axis=1), axis=-1)
    angles = arrays["static_frames"]
    relative = np.swapaxes(angles[:, :-1], -1, -2) @ angles[:, 1:]
    max_angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)).max(-1)
    has_jump = max_angle.max(1) > np.deg2rad(10)
    if not has_jump.all():
        raise RuntimeError(f"No >10 degree fixed-group frame switch on paths {np.flatnonzero(~has_jump).tolist()}; revise explicit refinement selection.")
    j = np.argmax(np.where(max_angle > np.deg2rad(10), step_drift, -1), axis=1)
    left = torch.from_numpy(arrays["alpha"][j].copy()).cuda()
    right = torch.from_numpy(arrays["alpha"][j + 1].copy()).cuda()
    radius = torch.tensor([r["radius_A"] for r in records], device="cuda")
    def at(alpha):
        points = x[:, 0] + alpha[:, None, None] * (x[:, 1] - x[:, 0])
        ng, centers = gather_groups(points, ci, gi)
        frame = model.encoder.token_encoder._frames_shape_and_chirality(ng, centers)[0]
        return points, ng, centers, frame
    trace = []
    for iteration in range(cfg["bisection_steps"] + 1):
        pl, nl, cl, fl = at(left)
        pr, nr, cr, fr = at(right)
        el = encode_groups(model, nl, cl)
        er = encode_groups(model, nr, cr)
        held = encode_groups(model, nr, cr, fl)
        transported = encode_groups(model, nr, cr, transport_frames(nl, nr, fl))
        trace.append(dict(iteration=iteration, left=left.cpu().numpy(), right=right.cpu().numpy(),
                          input_rms_A=((pl - pr).square().sum(-1).mean(-1).sqrt() * radius).cpu().numpy(),
                          native_jump=torch.linalg.vector_norm(er - el, dim=-1).cpu().numpy(),
                          held_jump=torch.linalg.vector_norm(held - el, dim=-1).cpu().numpy(),
                          transported_jump=torch.linalg.vector_norm(transported - el, dim=-1).cpu().numpy(),
                          angle_deg=(torch.acos((((fl.transpose(-1, -2) @ fr).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).amax(-1) * (180 / np.pi)).cpu().numpy()))
        mid = (left + right) / 2
        _, nm, _, _ = at(mid)
        same = (signatures(nm) == signatures(nl)).flatten(1).all(1)
        left = torch.where(same, mid, left)
        right = torch.where(same, right, mid)
    results = {key: np.stack([row[key] for row in trace]) for key in trace[0]}
    results["selected_step"] = j
    np.savez_compressed(output / f"{name}_refinement.npz", **results)
    print(f"{name}: refined {len(x)} frame-switch brackets", flush=True)


def main():
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text())
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    write_json(output / "status.json", dict(state="running"))
    try:
        endpoints, records = prepare_paths(cfg, output)
        for name, checkpoint in cfg["checkpoints"].items():
            model = load_model(ROOT / checkpoint)
            arrays = run_paths(model, endpoints, cfg, output, name)
            refine_jumps(model, endpoints, records, arrays, cfg, output, name)
            del model
            torch.cuda.empty_cache()
        paths = [Path(__file__), config_path,
                 ROOT / "src/models/encoders/geo_frame_transformer_v2.py",
                 ROOT / "src/models/encoders/ri_mae_encoder.py"]
        paths.extend(ROOT / value for value in cfg["checkpoints"].values())
        write_json(output / "provenance.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        write_json(output / "status.json", dict(state="complete", elapsed_seconds=time.monotonic() - started,
                                                path_count=len(records), interpolated_steps=cfg["path_steps"]))
    except BaseException as error:
        write_json(output / "status.json", dict(state="failed", error=repr(error), elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == "__main__":
    main()
