import math
import torch
import torch.nn as nn

from src.training_methods.contrastive_learning.invariant_utils import TensorProductInvariantHead
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        lambda_: float,
        embed_dim: int,
        start_epoch: int,
        jitter_std: float,
        jitter_mode: str,
        jitter_scale: float,
        drop_ratio: float,
        view_points: int | None,
        neighbor_view: bool,
        neighbor_view_mode: str,
        neighbor_k: int,
        neighbor_max_relative_distance: float,
        view_crop_mode: str,
        drop_apply_to_both: bool,
        rotation_mode: str,
        rotation_deg: float,
        strain_std: float,
        strain_volume_preserve: bool,
        occlusion_mode: str,
        occlusion_view: str,
        occlusion_slab_frac: float,
        occlusion_cone_deg: float,
        occlusion_prob: float,
        input_dim,
        invariant_mode: str = "norms",
        invariant_max_factor: float = 4.0,
        invariant_groups: int = 0,
        invariant_use_third_order: bool = True,
        invariant_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.lambda_ = float(lambda_)
        self.embed_dim = int(embed_dim)
        self.start_epoch = max(0, int(start_epoch))
        self.jitter_std = float(jitter_std)
        self.jitter_mode = str(jitter_mode).lower()
        self.jitter_scale = float(jitter_scale)
        self.drop_ratio = float(drop_ratio)
        self.view_points = int(view_points) if view_points is not None else None
        self.neighbor_view = bool(neighbor_view)
        self.neighbor_view_mode = str(neighbor_view_mode).lower()
        self.neighbor_k = int(neighbor_k)
        self.neighbor_max_relative_distance = max(0.0, float(neighbor_max_relative_distance))
        self.view_crop_mode = str(view_crop_mode).lower()
        self.drop_apply_to_both = bool(drop_apply_to_both)
        self.rotation_mode = str(rotation_mode).lower()
        self.rotation_deg = float(rotation_deg)
        self.strain_std = float(strain_std)
        self.strain_volume_preserve = bool(strain_volume_preserve)
        self.occlusion_mode = str(occlusion_mode).lower()
        self.occlusion_view = str(occlusion_view).lower()
        self.occlusion_slab_frac = float(occlusion_slab_frac)
        self.occlusion_cone_deg = float(occlusion_cone_deg)
        self.occlusion_prob = float(occlusion_prob)
        if not (0.0 <= self.occlusion_prob <= 1.0):
            raise ValueError(
                f"barlow_occlusion_prob must be in [0, 1], got {self.occlusion_prob}"
            )

        self.invariant_head = None
        projector_input_dim = int(input_dim) if input_dim is not None else None
        if projector_input_dim is not None:
            self.invariant_head = TensorProductInvariantHead(
                channels=projector_input_dim,
                mode=invariant_mode,
                max_factor=float(invariant_max_factor),
                groups=int(invariant_groups),
                include_third_order=bool(invariant_use_third_order),
                eps=float(invariant_eps),
            )
            projector_input_dim = int(self.invariant_head.output_dim)

        self.projector = None
        needs_projector = self.enabled and self.weight > 0
        if projector_input_dim is None:
            if needs_projector:
                raise ValueError("Barlow Twins requires latent_size to set projector input dim")
        elif needs_projector:
            self.projector = nn.Sequential(
                nn.Linear(projector_input_dim, self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            )

    @classmethod
    def from_config(cls, cfg, *, input_dim, invariant_mode_override: str | None = None):
        data_cfg = getattr(cfg, "data", None)
        view_points = getattr(cfg, "barlow_view_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "model_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "num_points", None)

        jitter_mode = str(getattr(cfg, "barlow_jitter_mode", "absolute")).lower()
        jitter_scale_cfg = getattr(cfg, "barlow_jitter_scale", None)
        jitter_scale = cls._resolve_jitter_scale(cfg, jitter_mode=jitter_mode, jitter_scale=jitter_scale_cfg)
        invariant_mode = (
            str(invariant_mode_override).lower()
            if invariant_mode_override is not None
            else str(getattr(cfg, "barlow_invariant_mode", "norms")).lower()
        )

        return cls(
            enabled=bool(getattr(cfg, "barlow_enabled", False)),
            weight=float(getattr(cfg, "barlow_weight", 0.0)),
            lambda_=float(getattr(cfg, "barlow_lambda", 5e-3)),
            embed_dim=int(getattr(cfg, "barlow_embed_dim", 8192)),
            start_epoch=int(getattr(cfg, "barlow_start_epoch", 0)),
            jitter_std=float(getattr(cfg, "barlow_jitter_std", 0.01)),
            jitter_mode=jitter_mode,
            jitter_scale=jitter_scale,
            drop_ratio=float(getattr(cfg, "barlow_drop_ratio", 0.2)),
            view_points=int(view_points) if view_points is not None else None,
            neighbor_view=bool(getattr(cfg, "barlow_neighbor_view", False)),
            neighbor_view_mode=str(getattr(cfg, "barlow_neighbor_view_mode", "both")),
            neighbor_k=int(getattr(cfg, "barlow_neighbor_k", 8)),
            neighbor_max_relative_distance=float(
                getattr(cfg, "barlow_neighbor_max_relative_distance", 0.0)
            ),
            view_crop_mode=str(getattr(cfg, "barlow_view_crop_mode", "random")),
            drop_apply_to_both=bool(getattr(cfg, "barlow_drop_apply_to_both", True)),
            rotation_mode=str(getattr(cfg, "barlow_rotation_mode", "none")),
            rotation_deg=float(getattr(cfg, "barlow_rotation_deg", 0.0)),
            strain_std=float(getattr(cfg, "barlow_strain_std", 0.0)),
            strain_volume_preserve=bool(getattr(cfg, "barlow_strain_volume_preserve", True)),
            occlusion_mode=str(getattr(cfg, "barlow_occlusion_mode", "none")),
            occlusion_view=str(getattr(cfg, "barlow_occlusion_view", "second")),
            occlusion_slab_frac=float(getattr(cfg, "barlow_occlusion_slab_frac", 0.4)),
            occlusion_cone_deg=float(getattr(cfg, "barlow_occlusion_cone_deg", 20.0)),
            occlusion_prob=float(getattr(cfg, "barlow_occlusion_prob", 1.0)),
            input_dim=input_dim,
            invariant_mode=invariant_mode,
            invariant_max_factor=float(getattr(cfg, "barlow_invariant_max_factor", 4.0)),
            invariant_groups=int(getattr(cfg, "barlow_invariant_groups", 0)),
            invariant_use_third_order=bool(getattr(cfg, "barlow_invariant_use_third_order", True)),
            invariant_eps=float(getattr(cfg, "barlow_invariant_eps", 1e-6)),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0
            and self.projector is not None
            and int(current_epoch) >= self.start_epoch
        )

    def compute_loss(
        self,
        *,
        pc: torch.Tensor,
        encoder,
        prepare_input,
        split_output,
        current_epoch: int,
        views: dict[str, torch.Tensor] | None = None,
        invariant_transform=None,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}
        if views is None:
            sampled = self.sample_view_pair(pc)
            y_a = sampled["y_a"]
            y_b = sampled["y_b"]
        else:
            y_a = views["y_a"]
            y_b = views["y_b"]

        enc_a = encoder(prepare_input(y_a))
        inv_a, eq_a = split_output(enc_a)
        if invariant_transform is None:
            inv_a = self._invariant(inv_a, eq_a)
        else:
            inv_a = invariant_transform(inv_a, eq_a)

        enc_b = encoder(prepare_input(y_b))
        inv_b, eq_b = split_output(enc_b)
        if invariant_transform is None:
            inv_b = self._invariant(inv_b, eq_b)
        else:
            inv_b = invariant_transform(inv_b, eq_b)

        if inv_a is None or inv_b is None:
            return None, {}

        proj_dtype = next(self.projector.parameters()).dtype
        z_a = self.projector(inv_a.to(dtype=proj_dtype))
        z_b = self.projector(inv_b.to(dtype=proj_dtype))
        loss, metrics = self._loss(z_a, z_b)
        if not torch.isfinite(loss).item():
            metrics["barlow_nonfinite"] = pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def sample_view_pair(self, pc: torch.Tensor) -> dict[str, torch.Tensor]:
        use_neighbor_a, use_neighbor_b = self._resolve_neighbor_flags(device=pc.device)
        apply_occlusion_a, apply_occlusion_b = self._resolve_pair_occlusion_flags(
            use_neighbor_a=use_neighbor_a,
            use_neighbor_b=use_neighbor_b,
            device=pc.device,
        )
        y_a, meta_a = self._augment(
            pc,
            use_neighbor=use_neighbor_a,
            apply_occlusion=apply_occlusion_a,
            return_metadata=True,
        )
        y_b, meta_b = self._augment(
            pc,
            use_neighbor=use_neighbor_b,
            apply_occlusion=apply_occlusion_b,
            return_metadata=True,
        )
        return {
            "y_a": y_a,
            "y_b": y_b,
            "rot_a": meta_a["rotation"],
            "rot_b": meta_b["rotation"],
        }

    def _resolve_neighbor_flags(self, *, device) -> tuple[bool, bool]:
        if not self.neighbor_view:
            return False, False
        mode = self.neighbor_view_mode
        if mode == "none":
            return False, False
        if mode == "first":
            return True, False
        if mode == "second":
            return False, True
        if mode == "random":
            use_a = bool((torch.rand((), device=device) < 0.5).item())
            use_b = bool((torch.rand((), device=device) < 0.5).item())
            if not (use_a or use_b):
                if bool((torch.rand((), device=device) < 0.5).item()):
                    use_a = True
                else:
                    use_b = True
            return use_a, use_b
        return True, True

    def _augment(
        self,
        pc: torch.Tensor,
        *,
        use_neighbor: bool,
        apply_occlusion: bool,
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = pc
        meta: dict[str, torch.Tensor] = {}
        if use_neighbor:
            x = shift_to_neighbor(
                x,
                neighbor_k=self.neighbor_k,
                max_relative_distance=self.neighbor_max_relative_distance,
            )
        if self.view_points is not None:
            x = crop_to_num_points(x, self.view_points, mode=self.view_crop_mode)
        if return_metadata:
            x, rot = self._apply_rotation(x, return_rotation=True)
            meta["rotation"] = rot
        else:
            x = self._apply_rotation(x)
        x = self._apply_strain(x)
        if apply_occlusion:
            mode = self._resolve_occlusion_mode(device=x.device)
            if mode == "slab":
                x = self._occlude_slab(x)
            elif mode == "cone":
                x = self._occlude_cone(x)
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * (self.jitter_std * self.jitter_scale)
        apply_drop = self.drop_ratio > 0 and (self.drop_apply_to_both or (not use_neighbor))
        if apply_drop:
            bsz, num_points, _ = x.shape
            keep = (torch.rand(bsz, num_points, device=x.device) > self.drop_ratio)
            keep[:, 0] = True
            w = keep.float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            idx = torch.multinomial(w, num_samples=num_points, replacement=True)
            x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

        if return_metadata:
            if "rotation" not in meta:
                eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
                meta["rotation"] = eye.expand(x.shape[0], -1, -1)
            return x, meta
        return x

    @staticmethod
    def _resolve_jitter_scale(cfg, *, jitter_mode: str, jitter_scale):
        if jitter_mode != "physical":
            return 1.0
        base_scale = None
        if jitter_scale is not None:
            base_scale = float(jitter_scale)
        else:
            data_cfg = getattr(cfg, "data", None)
            if data_cfg is not None:
                base_scale = getattr(data_cfg, "avg_nn_dist", None)
                if base_scale is None:
                    global_cfg = getattr(data_cfg, "global", None)
                    if global_cfg is not None:
                        base_scale = getattr(global_cfg, "avg_nn_dist", None)
        if base_scale is None or base_scale <= 0:
            return 1.0
        phys_to_model = 1.0
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is not None:
            normalize = getattr(data_cfg, "normalize", False)
            radius = getattr(data_cfg, "radius", None)
            if normalize and radius:
                normalization_scale = getattr(data_cfg, "normalization_scale", 1.0)
                phys_to_model = float(normalization_scale) / float(radius)
        return float(base_scale) * phys_to_model

    def _should_occlude(self, use_neighbor: bool) -> bool:
        if self.occlusion_mode == "none":
            return False
        if self.occlusion_view == "both":
            return True
        if self.occlusion_view == "first":
            return not use_neighbor
        return use_neighbor

    def _sample_pair_occlusion(self, *, device) -> bool:
        if self.occlusion_mode == "none":
            return False
        if self.occlusion_prob <= 0.0:
            return False
        if self.occlusion_prob >= 1.0:
            return True
        return bool((torch.rand((), device=device) < self.occlusion_prob).item())

    def _resolve_pair_occlusion_flags(
        self,
        *,
        use_neighbor_a: bool,
        use_neighbor_b: bool,
        device,
    ) -> tuple[bool, bool]:
        if not self._sample_pair_occlusion(device=device):
            return False, False
        return self._should_occlude(use_neighbor_a), self._should_occlude(use_neighbor_b)

    def _resolve_occlusion_mode(self, *, device) -> str:
        if self.occlusion_mode != "mixed":
            return self.occlusion_mode
        if torch.rand((), device=device) < 0.5:
            return "slab"
        return "cone"

    def _apply_rotation(
        self,
        x: torch.Tensor,
        *,
        return_rotation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mode = self.rotation_mode
        if mode == "none":
            if return_rotation:
                eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
                return x, eye.expand(x.shape[0], -1, -1)
            return x
        bsz = x.shape[0]
        device = x.device
        dtype = x.dtype
        if mode == "full":
            R = self._random_rotation_matrices(bsz, device=device, dtype=dtype)
        else:
            max_deg = float(self.rotation_deg)
            if max_deg <= 0:
                if return_rotation:
                    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
                    return x, eye.expand(bsz, -1, -1)
                return x
            max_rad = math.radians(max_deg)
            axis = self._random_unit_vectors(bsz, device=device, dtype=dtype)
            angle = (torch.rand(bsz, device=device, dtype=dtype) * 2.0 - 1.0) * max_rad
            R = self._axis_angle_to_matrix(axis, angle)
        out = torch.matmul(x, R)
        if return_rotation:
            # `_apply_rotation` uses row-vector convention (x @ R). Convert to
            # left-multiplication convention so downstream pose code stays consistent.
            return out, R.transpose(-1, -2)
        return out

    def _apply_strain(self, x: torch.Tensor) -> torch.Tensor:
        if self.strain_std <= 0:
            return x
        bsz = x.shape[0]
        device = x.device
        dtype = x.dtype
        strain_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        rand = torch.randn(bsz, 3, 3, device=device, dtype=strain_dtype)
        strain = 0.5 * (rand + rand.transpose(-1, -2))
        if self.strain_volume_preserve:
            trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
            strain = strain - (trace / 3.0).unsqueeze(-1) * torch.eye(3, device=device, dtype=strain_dtype)
        strain = strain * float(self.strain_std)
        transform = torch.eye(3, device=device, dtype=strain_dtype).unsqueeze(0) + strain
        out = torch.matmul(x.to(dtype=strain_dtype), transform)
        return out.to(dtype=dtype)

    def _occlude_slab(self, x: torch.Tensor) -> torch.Tensor:
        frac = float(self.occlusion_slab_frac)
        if frac <= 0:
            return x
        frac = min(frac, 1.0)
        bsz = x.shape[0]
        nvec = self._random_unit_vectors(bsz, device=x.device, dtype=x.dtype)
        proj = (x * nvec.unsqueeze(1)).sum(dim=-1)
        proj_min = proj.min(dim=1).values
        proj_max = proj.max(dim=1).values
        span = (proj_max - proj_min).clamp_min(1e-6)
        half_width = 0.5 * frac * span
        center = 0.5 * (proj_min + proj_max)
        drop_mask = (proj - center.unsqueeze(-1)).abs() <= half_width.unsqueeze(-1)
        keep_mask = ~drop_mask
        return self._resample_masked(x, keep_mask)

    def _occlude_cone(self, x: torch.Tensor) -> torch.Tensor:
        angle_deg = float(self.occlusion_cone_deg)
        if angle_deg <= 0 or angle_deg >= 180:
            return x
        bsz = x.shape[0]
        axis = self._random_unit_vectors(bsz, device=x.device, dtype=x.dtype)
        r = x.norm(dim=-1)
        eps = 1e-8
        cos_theta = math.cos(math.radians(angle_deg))
        cos_val = (x * axis.unsqueeze(1)).sum(dim=-1) / (r.clamp_min(eps))
        drop_mask = (cos_val >= cos_theta) & (r > eps)
        keep_mask = ~drop_mask
        return self._resample_masked(x, keep_mask)

    @staticmethod
    def _resample_masked(x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        if keep_mask is None:
            return x
        bsz, num_points, _ = x.shape
        if keep_mask.shape[:2] != (bsz, num_points):
            return x
        keep_any = keep_mask.any(dim=1)
        if not keep_any.all():
            keep_mask = keep_mask.clone()
            keep_mask[~keep_any, 0] = True
        if keep_mask.all():
            return x
        weights = keep_mask.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        idx = torch.multinomial(weights, num_samples=num_points, replacement=True)
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

    @staticmethod
    def _random_unit_vectors(batch_size: int, *, device, dtype) -> torch.Tensor:
        v = torch.randn(batch_size, 3, device=device, dtype=dtype)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _random_rotation_matrices(batch_size: int, *, device, dtype) -> torch.Tensor:
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

    @staticmethod
    def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        dtype = axis.dtype
        device = axis.device
        rot_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        axis = axis.to(dtype=rot_dtype)
        angle = angle.to(dtype=rot_dtype)
        x, y, z = axis.unbind(dim=-1)
        zeros = torch.zeros_like(x)
        K = torch.stack(
            [
                torch.stack([zeros, -z, y], dim=-1),
                torch.stack([z, zeros, -x], dim=-1),
                torch.stack([-y, x, zeros], dim=-1),
            ],
            dim=-2,
        )
        sin = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
        cos = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
        eye = torch.eye(3, device=device, dtype=rot_dtype).unsqueeze(0)
        R = eye + sin * K + (1.0 - cos) * (K @ K)
        return R.to(dtype=dtype)

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError("Input must be a square matrix")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _gather_all(self, z: torch.Tensor) -> torch.Tensor:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return z
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return z

        local_size = torch.tensor([z.shape[0]], device=z.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(sizes, local_size)
        sizes = [int(size.item()) for size in sizes]
        max_size = max(sizes)
        if max_size == 0:
            return z

        if z.shape[0] != max_size:
            pad = z.new_zeros((max_size - z.shape[0], *z.shape[1:]))
            z_padded = torch.cat([z, pad], dim=0)
        else:
            z_padded = z
        z_padded = z_padded.contiguous()

        gathered = [z_padded.new_zeros((max_size, *z.shape[1:])) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, z_padded)

        rank = torch.distributed.get_rank()
        gathered[rank] = z_padded

        trimmed = [chunk[:size] for chunk, size in zip(gathered, sizes)]
        return torch.cat(trimmed, dim=0)

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_a.dtype != torch.float32:
            z_a = z_a.float()
        if z_b.dtype != torch.float32:
            z_b = z_b.float()
        z_a = self._gather_all(z_a)
        z_b = self._gather_all(z_b)
        n, d = z_a.shape
        if n < 2:
            zero = z_a.new_tensor(0.0)
            return zero, {
                "barlow_diag_mean": zero,
                "barlow_offdiag_mean": zero,
            }

        z_a_mean = z_a.mean(0)
        z_b_mean = z_b.mean(0)
        z_a_std = z_a.std(0, unbiased=False).clamp_min(1e-4)
        z_b_std = z_b.std(0, unbiased=False).clamp_min(1e-4)
        z_a_norm = (z_a - z_a_mean) / z_a_std
        z_b_norm = (z_b - z_b_mean) / z_b_std

        c = (z_a_norm.T @ z_b_norm) / n
        c_diff = (c - torch.eye(d, device=c.device, dtype=c.dtype)).pow(2)
        diag_vals = torch.diagonal(c_diff)
        off_vals = self._off_diagonal(c_diff)

        diag_sum = diag_vals.sum()
        off_sum = off_vals.sum()
        loss = diag_sum + self.lambda_ * off_sum

        off_mean = off_vals.mean() if off_vals.numel() > 0 else diag_sum.new_tensor(0.0)
        metrics = {
            "barlow_diag_mean": diag_vals.mean().detach(),
            "barlow_offdiag_mean": off_mean.detach(),
        }
        return loss, metrics

    def _invariant(self, inv_z, eq_z):
        if self.invariant_head is None:
            if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
                import warnings
                warnings.warn(
                    "No invariant_head: reinterpreting inv_z (3D, last_dim=3) as eq_z "
                    f"and reducing via norm. Shape: {tuple(inv_z.shape)}. "
                    "Set invariant_mode explicitly if this is unintended.",
                )
                eq_z = inv_z
                inv_z = None
            if eq_z is not None:
                return eq_z.norm(dim=-1)
            if inv_z is None:
                raise ValueError(
                    "Both inv_z and eq_z are None after invariant processing. "
                    "Encoder must return at least one non-None latent."
                )
            return inv_z
        if eq_z is None and inv_z is not None and inv_z.dim() == 2:
            if inv_z.shape[1] == int(self.invariant_head.output_dim):
                return inv_z
        return self.invariant_head(inv_z, eq_z)
