import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_methods.contrastive_learning.config_warnings import (
    warn_common_view_sampler_ignored_fields,
    warn_disabled_radial_fields,
    warn_fixed_invariant_fields,
)
from src.training_methods.contrastive_learning.invariant_utils import NormInvariantHead
from src.utils.pointcloud_ops import crop_to_num_points, shift_to_neighbor


_VIEW_POINTS_UNSET = object()
_SUPPORTED_OBJECTIVES = {"vicreg", "visreg"}


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
        neighbor_view_mode: str,
        neighbor_k: int,
        neighbor_max_relative_distance: float,
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
        std_eps: float,
        std_target: float,
        input_dim,
        objective: str = "vicreg",
        visreg_lambda: float = 0.9,
        visreg_num_projections: int = 4096,
        visreg_scale_coeff: float = 1.0,
        visreg_shape_coeff: float = 1.0,
        visreg_center_coeff: float = 1.0,
        visreg_std_eps: float = 1e-6,
        radial_enabled: bool = False,
        radial_beta1: float = 1.0,
        radial_beta2: float = 0.1,
        radial_m: int | None = None,
        radial_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.objective = str(objective).strip().lower()
        if self.objective not in _SUPPORTED_OBJECTIVES:
            raise ValueError(
                "vicreg_objective must be one of "
                f"{sorted(_SUPPORTED_OBJECTIVES)}, got {objective!r}."
            )
        self.metric_prefix = self.objective
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
        self.neighbor_view_mode = str(neighbor_view_mode).lower()
        self.neighbor_k = int(neighbor_k)
        self.neighbor_max_relative_distance = max(0.0, float(neighbor_max_relative_distance))
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

        self.std_eps = float(std_eps)
        self.std_target = float(std_target)
        self.radial_enabled = bool(radial_enabled)
        self.radial_beta1 = float(radial_beta1)
        self.radial_beta2 = float(radial_beta2)
        self.radial_m = int(radial_m) if radial_m is not None and int(radial_m) > 0 else None
        self.radial_eps = max(float(radial_eps), 1e-12)
        if self.objective == "visreg" and self.radial_enabled:
            raise ValueError(
                "vicreg_radial_enabled is incompatible with vicreg_objective='visreg'. "
                "VISReg replaces covariance with center/scale/shape regularization; "
                "disable radial regularization for a faithful VISReg objective."
            )

        self.visreg_lambda = float(visreg_lambda)
        self.visreg_num_projections = int(visreg_num_projections)
        self.visreg_scale_coeff = float(visreg_scale_coeff)
        self.visreg_shape_coeff = float(visreg_shape_coeff)
        self.visreg_center_coeff = float(visreg_center_coeff)
        self.visreg_std_eps = float(visreg_std_eps)
        if self.objective == "visreg":
            if not (0.0 <= self.visreg_lambda <= 1.0):
                raise ValueError(
                    "visreg_lambda must be in [0, 1] for "
                    "(1-lambda) * prediction + lambda * regularization, "
                    f"got {self.visreg_lambda}."
                )
            if self.visreg_num_projections <= 0:
                raise ValueError(
                    "visreg_num_projections must be > 0, "
                    f"got {self.visreg_num_projections}."
                )
            if self.visreg_std_eps <= 0.0:
                raise ValueError(
                    "visreg_std_eps must be > 0 to avoid division by zero "
                    f"during collapsed embeddings, got {self.visreg_std_eps}."
                )
            for name, value in (
                ("visreg_scale_coeff", self.visreg_scale_coeff),
                ("visreg_shape_coeff", self.visreg_shape_coeff),
                ("visreg_center_coeff", self.visreg_center_coeff),
            ):
                if value < 0.0:
                    raise ValueError(f"{name} must be >= 0 for VISReg, got {value}.")
        self._visreg_cached_target_n = -1
        self._visreg_cached_target = None

        self.invariant_head = None
        projector_input_dim = int(input_dim) if input_dim is not None else None
        if projector_input_dim is not None:
            self.invariant_head = NormInvariantHead(
                channels=projector_input_dim,
                eps=1e-6,
            )
            projector_input_dim = int(self.invariant_head.output_dim)

        self.projector = None
        needs_projector = self.enabled and self.weight > 0
        if projector_input_dim is not None and needs_projector:
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
        radial_m = getattr(cfg, "vicreg_radial_m", None)
        if radial_m is not None:
            radial_m = int(radial_m)
            if radial_m <= 0:
                radial_m = None
        raw_objective = getattr(cfg, "vicreg_objective", None)
        if raw_objective is None:
            raw_objective = getattr(cfg, "contrastive_objective", None)
        if raw_objective is None:
            model_type = str(getattr(cfg, "model_type", "")).strip().lower()
            raw_objective = "visreg" if model_type == "visreg" else "vicreg"
        objective = str(raw_objective).strip().lower()
        enabled = bool(getattr(cfg, "vicreg_enabled", False))
        weight = float(getattr(cfg, "vicreg_weight", 0.0))
        sim_coeff = float(getattr(cfg, "vicreg_sim_coeff", 25.0))
        std_coeff = float(getattr(cfg, "vicreg_std_coeff", 25.0))
        cov_coeff = float(getattr(cfg, "vicreg_cov_coeff", 1.0))
        embed_dim = int(getattr(cfg, "vicreg_embed_dim", 8192))
        start_epoch = int(getattr(cfg, "vicreg_start_epoch", 0))
        jitter_std = float(getattr(cfg, "vicreg_jitter_std", 0.01))
        drop_ratio = float(getattr(cfg, "vicreg_drop_ratio", 0.2))
        resolved_view_points = int(view_points) if view_points is not None else None
        neighbor_view = bool(getattr(cfg, "vicreg_neighbor_view", False))
        neighbor_view_mode = str(getattr(cfg, "vicreg_neighbor_view_mode", "both"))
        neighbor_k = int(getattr(cfg, "vicreg_neighbor_k", 8))
        neighbor_max_relative_distance = float(
            getattr(cfg, "vicreg_neighbor_max_relative_distance", 0.0)
        )
        drop_apply_to_both = bool(getattr(cfg, "vicreg_drop_apply_to_both", True))
        rotation_mode = str(getattr(cfg, "vicreg_rotation_mode", "none"))
        rotation_deg = float(getattr(cfg, "vicreg_rotation_deg", 0.0))
        strain_std = float(getattr(cfg, "vicreg_strain_std", 0.0))
        strain_volume_preserve = bool(getattr(cfg, "vicreg_strain_volume_preserve", True))
        occlusion_mode = str(getattr(cfg, "vicreg_occlusion_mode", "none"))
        occlusion_view = str(getattr(cfg, "vicreg_occlusion_view", "second"))
        occlusion_slab_frac = float(getattr(cfg, "vicreg_occlusion_slab_frac", 0.4))
        occlusion_cone_deg = float(getattr(cfg, "vicreg_occlusion_cone_deg", 20.0))
        occlusion_prob = float(getattr(cfg, "vicreg_occlusion_prob", 1.0))
        std_eps = float(getattr(cfg, "vicreg_std_eps", 1e-4))
        std_target = float(getattr(cfg, "vicreg_std_target", 1.0))
        radial_enabled = bool(getattr(cfg, "vicreg_radial_enabled", False))
        radial_beta1 = float(getattr(cfg, "vicreg_radial_beta1", 1.0))
        radial_beta2 = float(getattr(cfg, "vicreg_radial_beta2", 0.1))
        radial_eps = float(getattr(cfg, "vicreg_radial_eps", 1e-8))
        visreg_lambda = float(getattr(cfg, "visreg_lambda", 0.9))
        visreg_num_projections = int(getattr(cfg, "visreg_num_projections", 4096))
        visreg_scale_coeff = float(getattr(cfg, "visreg_scale_coeff", 1.0))
        visreg_shape_coeff = float(getattr(cfg, "visreg_shape_coeff", 1.0))
        visreg_center_coeff = float(getattr(cfg, "visreg_center_coeff", 1.0))
        visreg_std_eps = float(getattr(cfg, "visreg_std_eps", 1e-6))

        warn_common_view_sampler_ignored_fields(
            cfg,
            prefix="vicreg",
            jitter_std=jitter_std,
            jitter_mode=jitter_mode,
            neighbor_view=neighbor_view,
            neighbor_view_mode=neighbor_view_mode,
            drop_ratio=drop_ratio,
            rotation_mode=rotation_mode,
            strain_std=strain_std,
            occlusion_mode=occlusion_mode,
        )
        warn_fixed_invariant_fields(
            cfg,
            prefix="vicreg",
        )
        warn_disabled_radial_fields(
            cfg,
            prefix="vicreg",
            radial_enabled=radial_enabled,
        )

        return cls(
            enabled=enabled,
            weight=weight,
            sim_coeff=sim_coeff,
            std_coeff=std_coeff,
            cov_coeff=cov_coeff,
            embed_dim=embed_dim,
            start_epoch=start_epoch,
            jitter_std=jitter_std,
            jitter_mode=jitter_mode,
            jitter_scale=jitter_scale,
            drop_ratio=drop_ratio,
            view_points=resolved_view_points,
            neighbor_view=neighbor_view,
            neighbor_view_mode=neighbor_view_mode,
            neighbor_k=neighbor_k,
            neighbor_max_relative_distance=neighbor_max_relative_distance,
            drop_apply_to_both=drop_apply_to_both,
            rotation_mode=rotation_mode,
            rotation_deg=rotation_deg,
            strain_std=strain_std,
            strain_volume_preserve=strain_volume_preserve,
            occlusion_mode=occlusion_mode,
            occlusion_view=occlusion_view,
            occlusion_slab_frac=occlusion_slab_frac,
            occlusion_cone_deg=occlusion_cone_deg,
            occlusion_prob=occlusion_prob,
            std_eps=std_eps,
            std_target=std_target,
            input_dim=input_dim,
            objective=objective,
            visreg_lambda=visreg_lambda,
            visreg_num_projections=visreg_num_projections,
            visreg_scale_coeff=visreg_scale_coeff,
            visreg_shape_coeff=visreg_shape_coeff,
            visreg_center_coeff=visreg_center_coeff,
            visreg_std_eps=visreg_std_eps,
            radial_enabled=radial_enabled,
            radial_beta1=radial_beta1,
            radial_beta2=radial_beta2,
            radial_m=radial_m,
            radial_eps=radial_eps,
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0
            and self.projector is not None
            and int(current_epoch) >= self.start_epoch
        )

    def forward(self, features: torch.Tensor, *, profile_projector: bool = False) -> torch.Tensor:
        projector_dtype = next(self.projector.parameters()).dtype
        return self.projector(features.to(dtype=projector_dtype))

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
            use_neighbor_a, use_neighbor_b = self._resolve_neighbor_flags(device=pc.device)
            apply_occlusion_a, apply_occlusion_b = self._resolve_pair_occlusion_flags(
                use_neighbor_a=use_neighbor_a,
                use_neighbor_b=use_neighbor_b,
                device=pc.device,
            )
            y_a = self._augment(pc, use_neighbor=use_neighbor_a, apply_occlusion=apply_occlusion_a)
            y_b = self._augment(pc, use_neighbor=use_neighbor_b, apply_occlusion=apply_occlusion_b)
        else:
            y_a = views["y_a"]
            y_b = views["y_b"]

        # Fuse the two encoder forwards into a single concatenated pass. This
        # halves kernel-launch overhead for the patch encoder + transformer
        # relative to the old two-separate-forwards path. The per-view
        # invariant transform is applied after splitting the result. See
        # hot-path fusion note in the SSL module (#1).
        batch_size = int(y_a.shape[0])
        fused_input = torch.cat([y_a, y_b], dim=0)
        enc_fused = encoder(prepare_input(fused_input))
        inv_fused, eq_fused = split_output(enc_fused)
        if invariant_transform is None:
            inv_fused = self._invariant(inv_fused, eq_fused)
        else:
            inv_fused = invariant_transform(inv_fused, eq_fused)

        if inv_fused is None:
            return None, {}
        inv_a, inv_b = inv_fused.chunk(2, dim=0)

        return self.compute_loss_from_features(
            z_a_feat=inv_a,
            z_b_feat=inv_b,
            current_epoch=current_epoch,
        )

    def compute_loss_from_features(
        self,
        *,
        z_a_feat: torch.Tensor | None,
        z_b_feat: torch.Tensor | None,
        current_epoch: int,
    ):
        """Compute VICReg on already-encoded invariant features.

        Used by the SSL step when the encoder forward is shared with SwAV or
        another head (#1 encoder fusion). Skipping the encoder call here lets
        the caller run one forward for both losses instead of two.
        """
        if not self.should_run(current_epoch=current_epoch):
            return None, {}
        if z_a_feat is None or z_b_feat is None:
            return None, {}

        proj_dtype = next(self.projector.parameters()).dtype
        z_a = self.projector(z_a_feat.to(dtype=proj_dtype))
        z_b = self.projector(z_b_feat.to(dtype=proj_dtype))
        loss, metrics = self._loss(z_a, z_b)
        # Tensor-only nonfinite accounting (no `.item()` sync, see #3).
        nonfinite = (~torch.isfinite(loss)).to(dtype=loss.dtype)
        metrics[f"{self.metric_prefix}_nonfinite"] = nonfinite
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

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
        view_points=_VIEW_POINTS_UNSET,
    ) -> torch.Tensor:
        x = pc
        if use_neighbor:
            x = shift_to_neighbor(
                x,
                neighbor_k=self.neighbor_k,
                max_relative_distance=self.neighbor_max_relative_distance,
            )
        return self.apply_view_postprocessing(
            x,
            use_neighbor=use_neighbor,
            apply_occlusion=apply_occlusion,
            view_points=view_points,
        )

    @staticmethod
    def _expand_batch_mask(value: bool | torch.Tensor, *, batch_size: int, device) -> torch.Tensor:
        if isinstance(value, bool):
            return torch.full((batch_size,), value, dtype=torch.bool, device=device)
        mask = value.to(device=device)
        if mask.dim() == 0:
            return torch.full((batch_size,), bool(mask.item()), dtype=torch.bool, device=device)
        mask = mask.reshape(-1)
        if mask.dtype != torch.bool:
            mask = mask != 0
        return mask

    def _apply_masked_occlusion(self, x: torch.Tensor, *, apply_mask: torch.Tensor) -> torch.Tensor:
        # Short-circuit on the cheap Python attribute first so the `.item()`
        # sync below is only paid when occlusion is actually enabled (#3
        # hot-path cleanup).
        if self.occlusion_mode == "none":
            return x
        if not bool(apply_mask.any().item()):
            return x

        mode = self._resolve_occlusion_mode(device=x.device)
        if mode == "slab":
            occluded = self._occlude_slab(x)
        elif mode == "cone":
            occluded = self._occlude_cone(x)
        else:
            return x
        return torch.where(apply_mask.view(-1, 1, 1), occluded, x)

    def _apply_masked_drop(self, x: torch.Tensor, *, apply_mask: torch.Tensor) -> torch.Tensor:
        if self.drop_ratio <= 0:
            return x
        if not bool(apply_mask.any().item()):
            return x

        bsz, num_points, _ = x.shape
        keep = (torch.rand(bsz, num_points, device=x.device) > self.drop_ratio)
        keep[:, 0] = True
        weights = keep.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        idx = torch.multinomial(weights, num_samples=num_points, replacement=True)
        dropped = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return torch.where(apply_mask.view(-1, 1, 1), dropped, x)

    def apply_view_postprocessing(
        self,
        x: torch.Tensor,
        *,
        use_neighbor: bool | torch.Tensor,
        apply_occlusion: bool | torch.Tensor | None = None,
        view_points=_VIEW_POINTS_UNSET,
    ) -> torch.Tensor:
        bsz = int(x.shape[0])
        use_neighbor_mask = self._expand_batch_mask(
            use_neighbor, batch_size=bsz, device=x.device
        )
        if apply_occlusion is None:
            apply_occlusion_mask = self._expand_batch_mask(
                self._should_occlude(False), batch_size=bsz, device=x.device
            )
            if self.occlusion_view == "first":
                apply_occlusion_mask = ~use_neighbor_mask
            elif self.occlusion_view == "second":
                apply_occlusion_mask = use_neighbor_mask
            elif self.occlusion_view == "both":
                apply_occlusion_mask = torch.ones((bsz,), dtype=torch.bool, device=x.device)
        else:
            apply_occlusion_mask = self._expand_batch_mask(
                apply_occlusion, batch_size=bsz, device=x.device
            )

        target_view_points = self.view_points if view_points is _VIEW_POINTS_UNSET else view_points
        if target_view_points is not None:
            x = crop_to_num_points(x, int(target_view_points))
        x = self._apply_rotation(x)
        x = self._apply_strain(x)
        x = self._apply_masked_occlusion(x, apply_mask=apply_occlusion_mask)
        if self.jitter_std > 0:
            x = x + torch.randn_like(x) * (self.jitter_std * self.jitter_scale)

        drop_mask = torch.zeros((bsz,), dtype=torch.bool, device=x.device)
        if self.drop_ratio > 0:
            if self.drop_apply_to_both:
                drop_mask.fill_(True)
            else:
                drop_mask = ~use_neighbor_mask
        x = self._apply_masked_drop(x, apply_mask=drop_mask)
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

    def _radial_gaussianization_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, d = z.shape
        r = torch.linalg.norm(z, dim=1)

        ce = 0.5 * r.pow(2) - (float(d) - 1.0) * torch.log(r + self.radial_eps)
        ce = self.radial_beta1 * ce.mean()

        if self.radial_beta2 <= 0:
            ent = z.new_tensor(0.0)
        else:
            m = self.radial_m
            if m is None:
                m = max(1, int(math.sqrt(n)))
            m = min(max(1, int(m)), n - 1)
            r_sorted = torch.sort(r).values
            diffs = r_sorted[m:] - r_sorted[:-m]
            ent_est = torch.log(((n + 1.0) / float(m)) * (diffs + self.radial_eps))
            ent = self.radial_beta2 * ent_est.mean()

        return ce - ent, ce, ent

    def _visreg_gaussian_quantiles(self, n: int, *, device, dtype) -> torch.Tensor:
        if n <= 0:
            raise ValueError(f"VISReg requires a non-empty batch, got batch size {n}.")
        if self._visreg_cached_target_n != int(n) or self._visreg_cached_target is None:
            q = torch.arange(1, int(n) + 1, device=device, dtype=torch.float32)
            q = q / float(int(n) + 1)
            self._visreg_cached_target = torch.erfinv(2.0 * q - 1.0).mul_(math.sqrt(2.0))
            self._visreg_cached_target_n = int(n)
        return self._visreg_cached_target.to(device=device, dtype=dtype)

    def _visreg_regularization_loss(self, z_views: torch.Tensor) -> tuple[torch.Tensor, dict]:
        num_views, n, d = z_views.shape
        if num_views <= 0:
            raise ValueError("VISReg requires at least one view.")
        if n <= 0:
            raise ValueError("VISReg requires a non-empty batch.")

        mu = z_views.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z_views - mu
        std = z_centered.norm(dim=1).div(math.sqrt(float(n))) + self.visreg_std_eps
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        slices = F.normalize(
            torch.randn(
                d,
                self.visreg_num_projections,
                device=z_views.device,
                dtype=z_views.dtype,
            ),
            dim=0,
        )
        p_sorted = torch.matmul(z_norm, slices).sort(dim=1).values
        target = self._visreg_gaussian_quantiles(
            n,
            device=z_views.device,
            dtype=z_views.dtype,
        ).view(1, n, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        reg_loss = (
            self.visreg_scale_coeff * scale_loss
            + self.visreg_shape_coeff * shape_loss
            + self.visreg_center_coeff * center_loss
        )
        metrics = {
            "visreg_reg": reg_loss,
            "visreg_scale": scale_loss,
            "visreg_shape": shape_loss,
            "visreg_center": center_loss,
        }
        return reg_loss, metrics

    def _visreg_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z_views = torch.stack((z_a, z_b), dim=0)
        global_mean = z_views.mean(dim=0, keepdim=True)
        pred_loss = (z_views - global_mean).pow(2).mean()
        reg_loss, metrics = self._visreg_regularization_loss(z_views)
        loss = (1.0 - self.visreg_lambda) * pred_loss + self.visreg_lambda * reg_loss
        metrics = {
            "visreg_pred": pred_loss,
            **metrics,
            "visreg_lambda": z_a.new_tensor(self.visreg_lambda),
        }
        return loss, metrics

    def _vicreg_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        n, _ = z_a.shape
        if n < 2:
            zero = z_a.new_tensor(0.0)
            metrics = {"vicreg_sim": zero, "vicreg_std": zero, "vicreg_cov": zero}
            if self.radial_enabled:
                metrics["vicreg_radial"] = zero
            return zero, metrics

        sim_loss = F.mse_loss(z_a, z_b)
        std_loss = 0.5 * (self._variance_loss(z_a) + self._variance_loss(z_b))
        cov_loss = 0.5 * (self._covariance_loss(z_a) + self._covariance_loss(z_b))
        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        metrics = {"vicreg_sim": sim_loss, "vicreg_std": std_loss, "vicreg_cov": cov_loss}

        if self.radial_enabled:
            radial_a, _, _ = self._radial_gaussianization_loss(z_a)
            radial_b, _, _ = self._radial_gaussianization_loss(z_b)
            radial_loss = 0.5 * (radial_a + radial_b)
            loss = loss + radial_loss
            metrics["vicreg_radial"] = radial_loss

        return loss, metrics

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if z_a.dtype != torch.float32:
            z_a = z_a.float()
        if z_b.dtype != torch.float32:
            z_b = z_b.float()
        z_a = self._gather_all(z_a)
        z_b = self._gather_all(z_b)
        if z_a.shape != z_b.shape:
            raise ValueError(
                "Contrastive view embeddings must have identical shapes before "
                f"{self.objective.upper()} loss; got z_a={tuple(z_a.shape)}, "
                f"z_b={tuple(z_b.shape)}."
            )
        if self.objective == "visreg":
            return self._visreg_loss(z_a, z_b)
        return self._vicreg_loss(z_a, z_b)

    def _invariant(self, inv_z, eq_z):
        if self.invariant_head is None:
            if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
                import warnings
                warnings.warn(
                    "No invariant_head: reinterpreting inv_z (3D, last_dim=3) as eq_z "
                    f"and reducing via norm. Shape: {tuple(inv_z.shape)}. "
                    "Contrastive training is norms-only, so equivariant tensors are reduced channel-wise.",
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
