import itertools
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from src.training_methods.contrastive_learning.pose.head import PoseRotationHead
from src.training_methods.spd.rot_heads import sixd_to_so3
from src.utils.spd_utils import apply_rotation


def cfg_get(obj, name: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _perm_parity(perm: tuple[int, int, int]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return -1 if (inv % 2) else 1


def cubic_symmetry_mats() -> torch.Tensor:
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        parity = _perm_parity(perm)
        for signs in itertools.product([-1, 1], repeat=3):
            det = parity * (signs[0] * signs[1] * signs[2])
            if det <= 0:
                continue
            mat = torch.zeros(3, 3, dtype=torch.float32)
            mat[0, perm[0]] = signs[0]
            mat[1, perm[1]] = signs[1]
            mat[2, perm[2]] = signs[2]
            mats.append(mat)
    if not mats:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)
    return torch.stack(mats, dim=0)


def ensure_identity(mats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mats.numel() == 0:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)
    identity = torch.eye(3, dtype=mats.dtype).unsqueeze(0)
    diffs = (mats - identity).abs().max(dim=-1).values.max(dim=-1).values
    if (diffs <= eps).any():
        return mats
    return torch.cat([identity, mats], dim=0)


def build_symmetry_mats(symmetry_type: str) -> torch.Tensor:
    sym_type = str(symmetry_type).lower()
    if sym_type == "none":
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)
    if sym_type == "cubic24":
        return cubic_symmetry_mats()
    raise ValueError(f"Unknown symmetry.type '{sym_type}'. Expected none, cubic24.")


def random_rotation_matrices(batch_size: int, *, device, dtype) -> torch.Tensor:
    rot_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    rand = torch.randn(batch_size, 3, 3, device=device, dtype=rot_dtype)
    q, r = torch.linalg.qr(rand)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-1)
    det = torch.det(q)
    neg = det < 0
    if neg.any():
        q[neg, :, 0] *= -1
    return q.to(dtype=dtype)


def sample_axis_angle_rotation(module, batch_size: int, *, max_rad: float, device, dtype) -> torch.Tensor:
    if max_rad <= 0:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    axis = module.barlow._random_unit_vectors(batch_size, device=device, dtype=dtype)
    angle = (torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0) * float(max_rad)
    return module.barlow._axis_angle_to_matrix(axis, angle)


def rotation_geodesic_angles(pred: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    delta = pred.transpose(-1, -2) @ target
    trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = 0.5 * (trace - 1.0)
    cos_clamped = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.arccos(cos_clamped)
    clamped = cos_clamped != cos_theta
    return angle, clamped


def compute_unambiguous_cap_rad(module) -> float:
    mats = module.symmetry_mats.detach().to(device="cpu", dtype=torch.float32)
    if mats.shape[0] <= 1:
        return math.pi

    eye = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    diffs = (mats - eye).abs().amax(dim=(-2, -1))
    non_identity = mats[diffs > 1e-6]
    if non_identity.numel() == 0:
        return math.pi

    eye_n = eye.expand(non_identity.shape[0], -1, -1)
    angles, _ = rotation_geodesic_angles(eye_n, non_identity, eps=float(module.symmetry_angle_eps))
    min_sym_angle = float(angles.min().item())
    margin = math.radians(max(0.0, float(module.pose_unambiguous_margin_deg)))
    return max(0.0, 0.5 * min_sym_angle - margin)


def pose_rotation_cap_rad(module, *, mode: str) -> float:
    if mode == "none":
        return 0.0
    if mode == "full":
        max_rad = math.pi
    else:
        max_deg = float(module.pose_view_rotation_deg)
        max_rad = math.radians(max(0.0, max_deg))
    if module.pose_unambiguous_rotation:
        max_rad = min(max_rad, float(module.pose_unambiguous_cap_rad))
    return max(0.0, float(max_rad))


def apply_pose_curriculum(module, max_rad: float) -> float:
    if not module.pose_curriculum_enabled or max_rad <= 0.0:
        return max_rad
    if module.pose_curriculum_epochs <= 1:
        return max_rad

    start_rad = math.radians(max(0.0, float(module.pose_curriculum_start_deg)))
    start_rad = min(start_rad, max_rad)
    start_epoch = int(module.pose_start_epoch)
    denom = max(1, int(module.pose_curriculum_epochs) - 1)
    progress = (int(module.current_epoch) - start_epoch) / float(denom)
    progress = max(0.0, min(1.0, progress))
    return start_rad + (max_rad - start_rad) * progress


def sample_pose_relative_rotation(module, batch_size: int, *, device, dtype) -> torch.Tensor:
    mode = module.pose_view_rotation_mode
    if mode == "none":
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    if mode == "full" and not module.pose_unambiguous_rotation and not module.pose_curriculum_enabled:
        return random_rotation_matrices(batch_size, device=device, dtype=dtype)

    max_rad = pose_rotation_cap_rad(module, mode=mode)
    max_rad = apply_pose_curriculum(module, max_rad)

    if mode == "full" and not module.pose_unambiguous_rotation and max_rad >= (math.pi - 1e-6):
        return random_rotation_matrices(batch_size, device=device, dtype=dtype)

    return sample_axis_angle_rotation(module, batch_size, max_rad=max_rad, device=device, dtype=dtype)


def prepare_eq_latent(eq_z: torch.Tensor | None) -> torch.Tensor | None:
    if eq_z is None:
        return None
    if eq_z.dim() == 3 and eq_z.shape[-1] == 3:
        return eq_z
    if eq_z.dim() == 4:
        if eq_z.shape[-1] == 3:
            return eq_z.mean(dim=-2)
        if eq_z.shape[2] == 3:
            return eq_z.mean(dim=-1)
    return None


def pose_should_run(module) -> bool:
    has_rot_objective = bool(module.pose_head is not None and module.pose_rot_loss_weight > 0)
    has_eq_objective = bool(module.pose_eq_loss_weight > 0)
    return bool(
        module.pose_enabled
        and module.pose_weight > 0
        and (has_rot_objective or has_eq_objective)
        and int(module.current_epoch) >= module.pose_start_epoch
    )


def prepare_pose_equivariant(module, eq_z: torch.Tensor) -> torch.Tensor:
    eq = eq_z
    if module.pose_eq_center:
        eq = eq - eq.mean(dim=1, keepdim=True)
    if module.pose_eq_l2_normalize:
        eq = F.normalize(eq, dim=-1, eps=1e-6)
    return eq


def channel_gram(eq: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bci,bdi->bcd", eq, eq)


def pose_regressor_features(module, eq_a: torch.Tensor, eq_b: torch.Tensor) -> torch.Tensor:
    if module.pose_regressor_input == "gram":
        gram_a = channel_gram(eq_a)
        gram_b = channel_gram(eq_b)
        if module.pose_cov_normalize and eq_a.shape[1] > 0:
            scale = float(eq_a.shape[1])
            gram_a = gram_a / scale
            gram_b = gram_b / scale
        return torch.cat([gram_a.reshape(gram_a.shape[0], -1), gram_b.reshape(gram_b.shape[0], -1)], dim=-1)

    cov = torch.einsum("bci,bcj->bij", eq_b, eq_a)
    if module.pose_cov_normalize and eq_a.shape[1] > 0:
        cov = cov / float(eq_a.shape[1])
    return cov


def so3_log_map(rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (0.5 * (trace - 1.0)).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.arccos(cos_theta)

    skew = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    coeff = theta / (2.0 * sin_theta.clamp_min(eps))
    small = theta.abs() < 1e-4
    coeff_small = 0.5 + (theta * theta) / 12.0
    coeff = torch.where(small, coeff_small, coeff)
    return coeff.unsqueeze(-1) * skew


def pose_identifiability(module, logvar: torch.Tensor) -> torch.Tensor:
    if module.pose_identifiability_mode == "inv_mean_var":
        c = 1.0 / (1.0 + torch.exp(logvar).mean(dim=-1))
    else:
        c = torch.exp(-logvar.mean(dim=-1))
    return c.clamp(min=1e-6, max=1.0)


def seeded_encoder_output(module, pc: torch.Tensor, seed: int):
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if pc.is_cuda else None
    torch.manual_seed(seed)
    if pc.is_cuda:
        torch.cuda.manual_seed_all(seed)
    try:
        enc_out = module.encoder(module._prepare_encoder_input(pc))
        return module._split_encoder_output(enc_out)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def autocast_disabled_context(tensor: torch.Tensor):
    device_type = tensor.device.type
    if device_type == "cuda" and torch.is_autocast_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def init_pose_components(module, cfg, latent_dim: int) -> None:
    pose_cfg = getattr(cfg, "pose", None)
    module.pose_enabled = bool(getattr(cfg, "pose_enabled", cfg_get(pose_cfg, "enabled", False)))
    module.pose_weight = float(getattr(cfg, "pose_weight", cfg_get(pose_cfg, "weight", 0.0)))
    module.pose_start_epoch = int(getattr(cfg, "pose_start_epoch", cfg_get(pose_cfg, "start_epoch", 0)))
    module.pose_jitter_std = float(getattr(cfg, "pose_jitter_std", cfg_get(pose_cfg, "jitter_std", 0.0)))
    module.pose_seeded_forward = bool(
        getattr(cfg, "pose_seeded_forward", cfg_get(pose_cfg, "seeded_forward", True))
    )
    module.pose_head_hidden = int(getattr(cfg, "pose_head_hidden", cfg_get(pose_cfg, "head_hidden", 128)))
    module.pose_view_rotation_mode = str(
        getattr(cfg, "pose_view_rotation_mode", cfg_get(pose_cfg, "view_rotation_mode", "full"))
    ).lower()
    module.pose_view_rotation_deg = float(
        getattr(cfg, "pose_view_rotation_deg", cfg_get(pose_cfg, "view_rotation_deg", 0.0))
    )
    module.pose_unambiguous_rotation = bool(
        getattr(cfg, "pose_unambiguous_rotation", cfg_get(pose_cfg, "unambiguous_rotation", True))
    )
    module.pose_unambiguous_margin_deg = float(
        getattr(cfg, "pose_unambiguous_margin_deg", cfg_get(pose_cfg, "unambiguous_margin_deg", 1.0))
    )
    module.pose_curriculum_enabled = bool(
        getattr(cfg, "pose_curriculum_enabled", cfg_get(pose_cfg, "curriculum_enabled", False))
    )
    module.pose_curriculum_epochs = int(
        getattr(cfg, "pose_curriculum_epochs", cfg_get(pose_cfg, "curriculum_epochs", 120))
    )
    module.pose_curriculum_start_deg = float(
        getattr(cfg, "pose_curriculum_start_deg", cfg_get(pose_cfg, "curriculum_start_deg", 8.0))
    )
    module.pose_eq_center = bool(getattr(cfg, "pose_eq_center", cfg_get(pose_cfg, "eq_center", True)))
    module.pose_eq_l2_normalize = bool(
        getattr(cfg, "pose_eq_l2_normalize", cfg_get(pose_cfg, "eq_l2_normalize", True))
    )
    module.pose_cov_normalize = bool(getattr(cfg, "pose_cov_normalize", cfg_get(pose_cfg, "cov_normalize", True)))

    legacy_head_weight = max(
        0.0,
        float(getattr(cfg, "pose_head_loss_weight", cfg_get(pose_cfg, "head_loss_weight", 1.0))),
    )
    legacy_procrustes_weight = max(
        0.0,
        float(getattr(cfg, "pose_procrustes_loss_weight", cfg_get(pose_cfg, "procrustes_loss_weight", 0.0))),
    )
    module.pose_rot_loss_weight = max(
        0.0,
        float(getattr(cfg, "pose_rot_loss_weight", cfg_get(pose_cfg, "rot_loss_weight", legacy_head_weight))),
    )
    module.pose_eq_loss_weight = max(
        0.0,
        float(getattr(cfg, "pose_eq_loss_weight", cfg_get(pose_cfg, "eq_loss_weight", legacy_procrustes_weight))),
    )
    module.pose_eq_gate_by_identifiability = bool(
        getattr(cfg, "pose_eq_gate_by_identifiability", cfg_get(pose_cfg, "eq_gate_by_identifiability", True))
    )
    module.pose_known_rotation_prob = float(
        getattr(cfg, "pose_known_rotation_prob", cfg_get(pose_cfg, "known_rotation_prob", 1.0))
    )
    module.pose_known_rotation_prob = max(0.0, min(1.0, module.pose_known_rotation_prob))
    module.pose_regressor_input = str(
        getattr(cfg, "pose_regressor_input", cfg_get(pose_cfg, "regressor_input", "cross_cov"))
    ).lower()
    if module.pose_regressor_input not in {"gram", "cross_cov"}:
        raise ValueError(
            "pose_regressor_input must be one of {'gram','cross_cov'}, "
            f"got '{module.pose_regressor_input}'."
        )
    module.pose_identifiability_mode = str(
        getattr(cfg, "pose_identifiability_mode", cfg_get(pose_cfg, "identifiability_mode", "exp_mean_logvar"))
    ).lower()
    if module.pose_identifiability_mode not in {"exp_mean_logvar", "inv_mean_var"}:
        raise ValueError(
            "pose_identifiability_mode must be one of {'exp_mean_logvar','inv_mean_var'}, "
            f"got '{module.pose_identifiability_mode}'."
        )
    module.pose_logvar_min = float(getattr(cfg, "pose_logvar_min", cfg_get(pose_cfg, "logvar_min", -10.0)))
    module.pose_logvar_max = float(getattr(cfg, "pose_logvar_max", cfg_get(pose_cfg, "logvar_max", 10.0)))
    if module.pose_logvar_min > module.pose_logvar_max:
        module.pose_logvar_min, module.pose_logvar_max = module.pose_logvar_max, module.pose_logvar_min
    module.pose_log_map_eps = float(getattr(cfg, "pose_log_map_eps", cfg_get(pose_cfg, "log_map_eps", 1e-6)))
    module.pose_histogram_every_n_steps = int(
        getattr(cfg, "pose_histogram_every_n_steps", cfg_get(pose_cfg, "histogram_every_n_steps", 50))
    )
    module.pose_delta_max_deg = float(getattr(cfg, "pose_delta_max_deg", cfg_get(pose_cfg, "delta_max_deg", 25.0)))
    module.pose_delta_max_rad = math.radians(max(0.0, module.pose_delta_max_deg))

    # Backward-compatible aliases retained for downstream scripts that still reference old names.
    module.pose_head_loss_weight = module.pose_rot_loss_weight
    module.pose_procrustes_loss_weight = module.pose_eq_loss_weight

    sym_cfg = getattr(cfg, "symmetry", None)
    if sym_cfg is None and pose_cfg is not None:
        sym_cfg = cfg_get(pose_cfg, "symmetry", None)
    module.symmetry_type = str(cfg_get(sym_cfg, "type", "cubic24")).lower()
    module.symmetry_beta = float(cfg_get(sym_cfg, "beta", 30.0))
    module.symmetry_angle_eps = float(cfg_get(sym_cfg, "angle_eps", 1e-6))
    module.register_buffer("symmetry_mats", build_symmetry_mats(module.symmetry_type))
    module.pose_unambiguous_cap_rad = compute_unambiguous_cap_rad(module)

    module.pose_head = (
        PoseRotationHead(
            hidden=module.pose_head_hidden,
            delta_max_rad=module.pose_delta_max_rad,
            generic_in_dim=int(2 * latent_dim * latent_dim),
        )
        if module.pose_enabled and module.pose_rot_loss_weight > 0
        else None
    )


def compute_pose_loss(module, pc: torch.Tensor, batch_idx: int, stage: str):
    if not pose_should_run(module):
        return None, {}
    if pc.dim() != 3 or pc.shape[-1] != 3:
        return None, {}
    if module.training and module.pose_known_rotation_prob < 1.0:
        keep = bool(torch.rand((), device=pc.device) <= module.pose_known_rotation_prob)
        if not keep:
            zero = torch.zeros((), device=pc.device, dtype=torch.float32, requires_grad=True)
            return zero, {"pose_known_rotation_applied": zero.detach()}

    # Pose supervision uses two augmented views linked by a known relative rotation.
    xa_base = module._prepare_model_input(pc)
    if module.pose_jitter_std > 0:
        jitter = torch.randn_like(xa_base) * float(module.pose_jitter_std)
        xa_base = xa_base + jitter

    bsz = xa_base.shape[0]
    device = xa_base.device
    dtype = xa_base.dtype

    if module.pose_view_rotation_mode == "none":
        rot_anchor = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
    else:
        rot_anchor = random_rotation_matrices(bsz, device=device, dtype=dtype)

    rot_rel = sample_pose_relative_rotation(module, bsz, device=device, dtype=dtype)
    rot_a = rot_anchor
    rot_b = rot_rel @ rot_anchor
    xa = apply_rotation(xa_base, rot_a)
    xb = apply_rotation(xa_base, rot_b)
    rot = rot_rel

    seed = int(module.global_step) + int(batch_idx) * 100003
    if module.pose_seeded_forward:
        _, eq_a = seeded_encoder_output(module, xa, seed)
        _, eq_b = seeded_encoder_output(module, xb, seed)
    else:
        eq_a = module._split_encoder_output(module.encoder(module._prepare_encoder_input(xa)))[1]
        eq_b = module._split_encoder_output(module.encoder(module._prepare_encoder_input(xb)))[1]

    eq_a = prepare_eq_latent(eq_a)
    eq_b = prepare_eq_latent(eq_b)
    if eq_a is None or eq_b is None:
        return None, {}

    eq_a = prepare_pose_equivariant(module, eq_a)
    eq_b = prepare_pose_equivariant(module, eq_b)
    with autocast_disabled_context(eq_a):
        rot_f = rot.to(dtype=torch.float32)
        eq_a_f = eq_a.to(dtype=torch.float32)
        eq_b_f = eq_b.to(dtype=torch.float32)

        eq_target = torch.einsum("bij,bcj->bci", rot_f, eq_a_f)
        eq_err_per = (eq_b_f - eq_target).pow(2).mean(dim=(1, 2))

        rot_nll_per = None
        rot_angle = None
        logvar = None
        ident = None
        ambiguity = None
        eq_gate = torch.ones_like(eq_err_per)
        rot_loss = None
        head_weight_mean = None
        head_weight_std = None
        base_angle = None
        delta_angle = None

        if module.pose_head is not None and module.pose_rot_loss_weight > 0:
            rot_base = None
            delta_omega = None
            if module.pose_regressor_input == "cross_cov":
                rot_hat, logvar, head_aux = module.pose_head.forward_pair_with_uncertainty(eq_a_f, eq_b_f)
                weights = head_aux.get("weights")
                rot_base = head_aux.get("rot_base")
                delta_omega = head_aux.get("delta_omega")
                if weights is not None:
                    head_weight_mean = weights.mean()
                    head_weight_std = weights.std(unbiased=False)
            else:
                pose_features = pose_regressor_features(module, eq_a_f, eq_b_f)
                r6, logvar = module.pose_head.forward_with_uncertainty(pose_features)
                rot_hat = sixd_to_so3(r6, eps=1e-6)
            logvar = logvar.clamp(min=float(module.pose_logvar_min), max=float(module.pose_logvar_max))

            residual = so3_log_map(rot_hat.transpose(-1, -2) @ rot_f, eps=float(module.pose_log_map_eps))
            rot_nll_per = (residual.pow(2) * torch.exp(-logvar) + logvar).sum(dim=-1)
            rot_angle = residual.norm(dim=-1)
            rot_loss = rot_nll_per.mean()
            ident = pose_identifiability(module, logvar)
            ambiguity = torch.exp(logvar).mean(dim=-1)
            if rot_base is not None:
                base_residual = so3_log_map(rot_base.transpose(-1, -2) @ rot_f, eps=float(module.pose_log_map_eps))
                base_angle = base_residual.norm(dim=-1)
            if delta_omega is not None:
                delta_angle = torch.linalg.vector_norm(delta_omega, dim=-1)
            if module.pose_eq_gate_by_identifiability:
                eq_gate = ident.detach()

        eq_loss = None
        if module.pose_eq_loss_weight > 0:
            eq_loss = (eq_gate * eq_err_per).mean()

        weighted_terms: list[torch.Tensor] = []
        if rot_loss is not None and module.pose_rot_loss_weight > 0:
            weighted_terms.append(rot_loss * float(module.pose_rot_loss_weight))
        if eq_loss is not None and module.pose_eq_loss_weight > 0:
            weighted_terms.append(eq_loss * float(module.pose_eq_loss_weight))
        if not weighted_terms:
            return None, {}

        pose_loss = torch.stack(weighted_terms, dim=0).sum(dim=0)

    if not torch.isfinite(pose_loss).item():
        # pose_nonfinite: indicator that pose loss produced NaN/Inf on this step.
        metrics = {"pose_nonfinite": xa.new_tensor(1.0)}
        pose_loss = torch.nan_to_num(pose_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return pose_loss, metrics

    rad_to_deg = 180.0 / math.pi
    metrics = {
        "pose_known_rotation_applied": xa.new_tensor(1.0, dtype=torch.float32),
    }
    if rot_nll_per is not None:
        metrics["pose_rot_nll"] = rot_nll_per.mean().to(dtype=torch.float32)
    if rot_angle is not None:
        metrics["pose_rot_angle_mean_deg"] = (rot_angle * rad_to_deg).mean().to(dtype=torch.float32)
    if logvar is not None:
        metrics["pose_logvar_mean"] = logvar.mean().to(dtype=torch.float32)
        metrics["pose_logvar_std"] = logvar.std(unbiased=False).to(dtype=torch.float32)
    if ident is not None:
        metrics["pose_identifiability_mean"] = ident.mean().to(dtype=torch.float32)
    if head_weight_mean is not None:
        metrics["pose_head_weight_mean"] = head_weight_mean.to(dtype=torch.float32)
    if head_weight_std is not None:
        metrics["pose_head_weight_std"] = head_weight_std.to(dtype=torch.float32)
    if base_angle is not None:
        metrics["pose_base_angle_mean_deg"] = (base_angle * rad_to_deg).mean().to(dtype=torch.float32)
    if delta_angle is not None:
        metrics["pose_delta_angle_mean_deg"] = (delta_angle * rad_to_deg).mean().to(dtype=torch.float32)
    if module.pose_eq_loss_weight > 0:
        metrics["pose_eq_mse"] = eq_err_per.mean().to(dtype=torch.float32)
        metrics["pose_eq_gate_mean"] = eq_gate.mean().to(dtype=torch.float32)
        metrics["pose_eq_mse_weighted"] = (eq_gate * eq_err_per).mean().to(dtype=torch.float32)

    module._log_pose_ambiguity_histogram(
        stage,
        ambiguity,
        identifiability=ident,
        logvar=logvar,
        batch_idx=batch_idx,
    )

    target_cap_deg = math.degrees(
        apply_pose_curriculum(module, pose_rotation_cap_rad(module, mode=module.pose_view_rotation_mode))
    )
    metrics = {
        **metrics,
        "pose_target_max_deg": xa.new_tensor(float(target_cap_deg), dtype=torch.float32),
    }

    with torch.no_grad():
        rot_f = rot.to(dtype=torch.float32)
        eye = torch.eye(3, device=rot_f.device, dtype=rot_f.dtype).unsqueeze(0).expand_as(rot_f)
        target_ang, _ = rotation_geodesic_angles(eye, rot_f, eps=float(module.symmetry_angle_eps))
        metrics["pose_target_angle_mean_deg"] = (target_ang * (180.0 / math.pi)).mean()
    return pose_loss, metrics
