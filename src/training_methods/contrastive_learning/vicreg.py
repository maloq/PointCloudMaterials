import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor


class VICRegLoss(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        sim_coeff: float,
        std_coeff: float,
        cov_coeff: float,
        embed_dim: int,
        start_epoch: int,
        jitter_std: float,
        jitter_mode: str,
        jitter_scale: float,
        drop_ratio: float,
        view_points: int | None,
        neighbor_view: bool,
        neighbor_k: int,
        rotation_mode: str,
        rotation_deg: float,
        strain_std: float,
        strain_volume_preserve: bool,
        occlusion_mode: str,
        occlusion_view: str,
        occlusion_slab_frac: float,
        occlusion_cone_deg: float,
        std_eps: float,
        std_target: float,
        input_dim,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.sim_coeff = float(sim_coeff)
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)
        self.embed_dim = int(embed_dim)
        self.start_epoch = max(0, int(start_epoch))
        self.jitter_std = float(jitter_std)
        self.jitter_mode = str(jitter_mode).lower()
        self.jitter_scale = float(jitter_scale)
        self.drop_ratio = float(drop_ratio)
        self.view_points = int(view_points) if view_points is not None else None
        self.neighbor_view = bool(neighbor_view)
        self.neighbor_k = int(neighbor_k)
        self.rotation_mode = str(rotation_mode).lower()
        self.rotation_deg = float(rotation_deg)
        self.strain_std = float(strain_std)
        self.strain_volume_preserve = bool(strain_volume_preserve)
        self.occlusion_mode = str(occlusion_mode).lower()
        self.occlusion_view = str(occlusion_view).lower()
        self.occlusion_slab_frac = float(occlusion_slab_frac)
        self.occlusion_cone_deg = float(occlusion_cone_deg)
        self.std_eps = float(std_eps)
        self.std_target = float(std_target)

        self.projector = None
        if input_dim is None:
            if self.enabled and self.weight > 0:
                raise ValueError("VICReg requires latent_size to set projector input dim")
        else:
            self.projector = nn.Sequential(
                nn.Linear(int(input_dim), self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            )

    @classmethod
    def from_config(cls, cfg, *, input_dim):
        data_cfg = getattr(cfg, "data", None)
        view_points = getattr(cfg, "vicreg_view_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "model_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "num_points", None)

        jitter_mode = str(getattr(cfg, "vicreg_jitter_mode", "absolute")).lower()
        jitter_scale_cfg = getattr(cfg, "vicreg_jitter_scale", None)
        jitter_scale = cls._resolve_jitter_scale(cfg, jitter_mode=jitter_mode, jitter_scale=jitter_scale_cfg)

        return cls(
            enabled=bool(getattr(cfg, "vicreg_enabled", False)),
            weight=float(getattr(cfg, "vicreg_weight", 0.0)),
            sim_coeff=float(getattr(cfg, "vicreg_sim_coeff", 25.0)),
            std_coeff=float(getattr(cfg, "vicreg_std_coeff", 25.0)),
            cov_coeff=float(getattr(cfg, "vicreg_cov_coeff", 1.0)),
            embed_dim=int(getattr(cfg, "vicreg_embed_dim", 8192)),
            start_epoch=int(getattr(cfg, "vicreg_start_epoch", 0)),
            jitter_std=float(getattr(cfg, "vicreg_jitter_std", 0.01)),
            jitter_mode=jitter_mode,
            jitter_scale=jitter_scale,
            drop_ratio=float(getattr(cfg, "vicreg_drop_ratio", 0.2)),
            view_points=int(view_points) if view_points is not None else None,
            neighbor_view=bool(getattr(cfg, "vicreg_neighbor_view", False)),
            neighbor_k=int(getattr(cfg, "vicreg_neighbor_k", 8)),
            rotation_mode=str(getattr(cfg, "vicreg_rotation_mode", "none")),
            rotation_deg=float(getattr(cfg, "vicreg_rotation_deg", 0.0)),
            strain_std=float(getattr(cfg, "vicreg_strain_std", 0.0)),
            strain_volume_preserve=bool(getattr(cfg, "vicreg_strain_volume_preserve", True)),
            occlusion_mode=str(getattr(cfg, "vicreg_occlusion_mode", "none")),
            occlusion_view=str(getattr(cfg, "vicreg_occlusion_view", "second")),
            occlusion_slab_frac=float(getattr(cfg, "vicreg_occlusion_slab_frac", 0.4)),
            occlusion_cone_deg=float(getattr(cfg, "vicreg_occlusion_cone_deg", 20.0)),
            std_eps=float(getattr(cfg, "vicreg_std_eps", 1e-4)),
            std_target=float(getattr(cfg, "vicreg_std_target", 1.0)),
            input_dim=input_dim,
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
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}
        y_a = self._augment(pc, use_neighbor=False)
        y_b = self._augment(pc, use_neighbor=self.neighbor_view)

        enc_a = encoder(prepare_input(y_a))
        inv_a, eq_a = split_output(enc_a)
        inv_a = self._invariant(inv_a, eq_a)

        enc_b = encoder(prepare_input(y_b))
        inv_b, eq_b = split_output(enc_b)
        inv_b = self._invariant(inv_b, eq_b)

        if inv_a is None or inv_b is None:
            return None, {}

        proj_dtype = next(self.projector.parameters()).dtype
        z_a = self.projector(inv_a.to(dtype=proj_dtype))
        z_b = self.projector(inv_b.to(dtype=proj_dtype))
        loss, metrics = self._loss(z_a, z_b)
        if not torch.isfinite(loss).item():
            metrics["vicreg_nonfinite"] = pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def _augment(self, pc: torch.Tensor, *, use_neighbor: bool) -> torch.Tensor:
        x = pc
        if use_neighbor:
            x = shift_to_neighbor(x, neighbor_k=self.neighbor_k)
        if self.view_points is not None:
            x = crop_to_num_points(x, self.view_points)
        x = self._apply_rotation(x)
        x = self._apply_strain(x)
        if self._should_occlude(use_neighbor):
            mode = self._resolve_occlusion_mode(device=x.device)
            if mode == "slab":
                x = self._occlude_slab(x)
            elif mode == "cone":
                x = self._occlude_cone(x)
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * (self.jitter_std * self.jitter_scale)
        if self.drop_ratio > 0 and not use_neighbor:
            bsz, num_points, _ = x.shape
            keep = (torch.rand(bsz, num_points, device=x.device) > self.drop_ratio)
            keep[:, 0] = True
            w = keep.float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            idx = torch.multinomial(w, num_samples=num_points, replacement=True)
            x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

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

    def _resolve_occlusion_mode(self, *, device) -> str:
        if self.occlusion_mode != "mixed":
            return self.occlusion_mode
        if torch.rand((), device=device) < 0.5:
            return "slab"
        return "cone"

    def _apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.rotation_mode
        if mode == "none":
            return x
        bsz = x.shape[0]
        device = x.device
        dtype = x.dtype
        if mode == "full":
            R = self._random_rotation_matrices(bsz, device=device, dtype=dtype)
        else:
            max_deg = float(self.rotation_deg)
            if max_deg <= 0:
                return x
            max_rad = math.radians(max_deg)
            axis = self._random_unit_vectors(bsz, device=device, dtype=dtype)
            angle = (torch.rand(bsz, device=device, dtype=dtype) * 2.0 - 1.0) * max_rad
            R = self._axis_angle_to_matrix(axis, angle)
        return torch.matmul(x, R)

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
        mask = (proj - center.unsqueeze(-1)).abs() <= half_width.unsqueeze(-1)
        return self._resample_masked(x, mask)

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
        mask = cos_val >= cos_theta
        mask = mask | (r <= eps)
        return self._resample_masked(x, mask)

    @staticmethod
    def _resample_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return x
        bsz, num_points, _ = x.shape
        if mask.shape[:2] != (bsz, num_points):
            return x
        keep_any = mask.any(dim=1)
        if not keep_any.all():
            mask = mask.clone()
            mask[~keep_any, 0] = True
        if mask.all():
            return x
        weights = mask.float()
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

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        var = z.var(dim=0, unbiased=False)
        std = torch.sqrt(var + self.std_eps)
        return torch.mean(F.relu(self.std_target - std))

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        if n < 2:
            return z.new_tensor(0.0)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (n - 1)
        off = self._off_diagonal(cov)
        return (off.pow(2).sum() / d)

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if z_a.dtype != torch.float32:
            z_a = z_a.float()
        if z_b.dtype != torch.float32:
            z_b = z_b.float()
        z_a = self._gather_all(z_a)
        z_b = self._gather_all(z_b)
        n, _ = z_a.shape
        if n < 2:
            zero = z_a.new_tensor(0.0)
            return zero, {"vicreg_sim": zero, "vicreg_std": zero, "vicreg_cov": zero}

        sim_loss = F.mse_loss(z_a, z_b)
        std_loss = 0.5 * (self._variance_loss(z_a) + self._variance_loss(z_b))
        cov_loss = 0.5 * (self._covariance_loss(z_a) + self._covariance_loss(z_b))
        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        metrics = {"vicreg_sim": sim_loss, "vicreg_std": std_loss, "vicreg_cov": cov_loss}
        return loss, metrics

    @staticmethod
    def _invariant(inv_z, eq_z):
        if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
            eq_z = inv_z
            inv_z = None
        if eq_z is not None:
            return eq_z.norm(dim=-1)
        return inv_z
