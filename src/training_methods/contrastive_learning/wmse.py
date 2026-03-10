import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss


class _Whitening1d(nn.Module):
    """Batch whitening for 2D embeddings shaped as (batch, dim)."""

    def __init__(self, num_features: int, *, eps: float = 0.0) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

        if self.num_features <= 0:
            raise ValueError(
                f"num_features must be > 0 for whitening, got {self.num_features}"
            )
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError(
                f"whitening eps must satisfy 0 <= eps <= 1, got {self.eps}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor for whitening, got {type(x)}")
        if x.dim() != 2:
            raise ValueError(
                f"Whitening expects a 2D tensor shaped (batch, dim), got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Whitening feature dim mismatch: got dim={x.shape[1]}, "
                f"expected num_features={self.num_features}."
            )
        if x.shape[0] < 2:
            raise ValueError(
                f"Whitening requires at least 2 samples, got batch_size={x.shape[0]}"
            )

        work = x if x.dtype == torch.float32 else x.float()
        centered = work - work.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / float(work.shape[0] - 1)
        eye = torch.eye(self.num_features, device=work.device, dtype=work.dtype)
        cov_shrunk = (1.0 - self.eps) * cov + self.eps * eye

        try:
            chol = torch.linalg.cholesky(cov_shrunk)
        except RuntimeError as exc:
            raise RuntimeError(
                "W-MSE whitening failed during Cholesky decomposition. "
                f"batch_size={work.shape[0]}, num_features={self.num_features}, eps={self.eps}. "
                "Increase wmse_whitening_eps, reduce wmse_embed_dim, or increase the "
                "effective whitening batch size."
            ) from exc

        inv_sqrt = torch.linalg.solve_triangular(chol, eye, upper=False)
        return centered @ inv_sqrt.T


class WMSELoss(nn.Module):
    """Whitening MSE (W-MSE) objective for invariant point-cloud latents."""

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
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
        whitening_eps: float = 0.0,
        whitening_iters: int = 1,
        whitening_size: int = 128,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.embed_dim = int(embed_dim)
        self.start_epoch = max(0, int(start_epoch))
        self.whitening_eps = float(whitening_eps)
        self.whitening_iters = int(whitening_iters)
        self.whitening_size = int(whitening_size)
        self.normalize_embeddings = bool(normalize_embeddings)

        if self.weight < 0:
            raise ValueError(f"wmse_weight must be >= 0, got {self.weight}")
        if self.embed_dim <= 0:
            raise ValueError(f"wmse_embed_dim must be > 0, got {self.embed_dim}")
        if self.whitening_iters <= 0:
            raise ValueError(
                f"wmse_whitening_iters must be > 0, got {self.whitening_iters}"
            )
        if self.whitening_size < 2:
            raise ValueError(
                f"wmse_whitening_size must be >= 2, got {self.whitening_size}"
            )

        self._view_sampler = BarlowTwinsLoss(
            enabled=False,
            weight=0.0,
            lambda_=0.0,
            embed_dim=max(1, self.embed_dim),
            start_epoch=0,
            jitter_std=float(jitter_std),
            jitter_mode=str(jitter_mode),
            jitter_scale=float(jitter_scale),
            drop_ratio=float(drop_ratio),
            view_points=int(view_points) if view_points is not None else None,
            neighbor_view=bool(neighbor_view),
            neighbor_view_mode=str(neighbor_view_mode),
            neighbor_k=int(neighbor_k),
            neighbor_max_relative_distance=float(neighbor_max_relative_distance),
            view_crop_mode=str(view_crop_mode),
            drop_apply_to_both=bool(drop_apply_to_both),
            rotation_mode=str(rotation_mode),
            rotation_deg=float(rotation_deg),
            strain_std=float(strain_std),
            strain_volume_preserve=bool(strain_volume_preserve),
            occlusion_mode=str(occlusion_mode),
            occlusion_view=str(occlusion_view),
            occlusion_slab_frac=float(occlusion_slab_frac),
            occlusion_cone_deg=float(occlusion_cone_deg),
            occlusion_prob=float(occlusion_prob),
            input_dim=input_dim,
            invariant_mode=str(invariant_mode),
            invariant_max_factor=float(invariant_max_factor),
            invariant_groups=int(invariant_groups),
            invariant_use_third_order=bool(invariant_use_third_order),
            invariant_eps=float(invariant_eps),
        )
        self.invariant_head = self._view_sampler.invariant_head

        projector_input_dim = int(input_dim) if input_dim is not None else None
        if self.invariant_head is not None:
            projector_input_dim = int(self.invariant_head.output_dim)

        self.projector = None
        needs_projector = self.enabled and self.weight > 0
        if projector_input_dim is None:
            if needs_projector:
                raise ValueError(
                    "W-MSE requires latent_size/encoder latent dim to build projector input"
                )
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

        self.whitening = _Whitening1d(self.embed_dim, eps=self.whitening_eps)

    @staticmethod
    def _cfg_value(cfg, primary: str, fallback: tuple[str, ...] = (), default=None):
        value = getattr(cfg, primary, None)
        if value is not None:
            return value

        resolved: list[tuple[str, object]] = []
        for field in fallback:
            candidate = getattr(cfg, field, None)
            if candidate is not None:
                resolved.append((field, candidate))

        if not resolved:
            return default

        reference_field, reference_value = resolved[0]
        mismatches = [
            f"{field}={value!r}"
            for field, value in resolved[1:]
            if value != reference_value
        ]
        if mismatches:
            choices = ", ".join(
                [f"{reference_field}={reference_value!r}", *mismatches]
            )
            raise ValueError(
                f"Ambiguous fallback values for {primary}: {choices}. "
                f"Set {primary} explicitly."
            )
        return reference_value

    @classmethod
    def from_config(cls, cfg, *, input_dim, invariant_mode_override: str | None = None):
        data_cfg = getattr(cfg, "data", None)
        view_points = cls._cfg_value(
            cfg,
            "wmse_view_points",
            fallback=("barlow_view_points", "vicreg_view_points"),
            default=None,
        )
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "model_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "num_points", None)

        jitter_mode = str(
            cls._cfg_value(
                cfg,
                "wmse_jitter_mode",
                fallback=("barlow_jitter_mode", "vicreg_jitter_mode"),
                default="absolute",
            )
        ).lower()
        jitter_scale_cfg = cls._cfg_value(
            cfg,
            "wmse_jitter_scale",
            fallback=("barlow_jitter_scale", "vicreg_jitter_scale"),
            default=None,
        )
        jitter_scale = BarlowTwinsLoss._resolve_jitter_scale(
            cfg,
            jitter_mode=jitter_mode,
            jitter_scale=jitter_scale_cfg,
        )

        invariant_mode = (
            str(invariant_mode_override).lower()
            if invariant_mode_override is not None
            else str(
                cls._cfg_value(
                    cfg,
                    "wmse_invariant_mode",
                    fallback=("barlow_invariant_mode", "vicreg_invariant_mode"),
                    default="norms",
                )
            ).lower()
        )

        return cls(
            enabled=bool(cls._cfg_value(cfg, "wmse_enabled", default=False)),
            weight=float(cls._cfg_value(cfg, "wmse_weight", default=0.0)),
            embed_dim=int(cls._cfg_value(cfg, "wmse_embed_dim", default=256)),
            start_epoch=int(cls._cfg_value(cfg, "wmse_start_epoch", default=0)),
            jitter_std=float(
                cls._cfg_value(
                    cfg,
                    "wmse_jitter_std",
                    fallback=("barlow_jitter_std", "vicreg_jitter_std"),
                    default=0.01,
                )
            ),
            jitter_mode=jitter_mode,
            jitter_scale=float(jitter_scale),
            drop_ratio=float(
                cls._cfg_value(
                    cfg,
                    "wmse_drop_ratio",
                    fallback=("barlow_drop_ratio", "vicreg_drop_ratio"),
                    default=0.2,
                )
            ),
            view_points=int(view_points) if view_points is not None else None,
            neighbor_view=bool(
                cls._cfg_value(
                    cfg,
                    "wmse_neighbor_view",
                    fallback=("barlow_neighbor_view", "vicreg_neighbor_view"),
                    default=False,
                )
            ),
            neighbor_view_mode=str(
                cls._cfg_value(
                    cfg,
                    "wmse_neighbor_view_mode",
                    fallback=("barlow_neighbor_view_mode", "vicreg_neighbor_view_mode"),
                    default="both",
                )
            ),
            neighbor_k=int(
                cls._cfg_value(
                    cfg,
                    "wmse_neighbor_k",
                    fallback=("barlow_neighbor_k", "vicreg_neighbor_k"),
                    default=8,
                )
            ),
            neighbor_max_relative_distance=float(
                cls._cfg_value(
                    cfg,
                    "wmse_neighbor_max_relative_distance",
                    fallback=(
                        "barlow_neighbor_max_relative_distance",
                        "vicreg_neighbor_max_relative_distance",
                    ),
                    default=0.0,
                )
            ),
            view_crop_mode=str(
                cls._cfg_value(
                    cfg,
                    "wmse_view_crop_mode",
                    fallback=("barlow_view_crop_mode", "vicreg_view_crop_mode"),
                    default="nearest_origin",
                )
            ),
            drop_apply_to_both=bool(
                cls._cfg_value(
                    cfg,
                    "wmse_drop_apply_to_both",
                    fallback=("barlow_drop_apply_to_both", "vicreg_drop_apply_to_both"),
                    default=True,
                )
            ),
            rotation_mode=str(
                cls._cfg_value(
                    cfg,
                    "wmse_rotation_mode",
                    fallback=("barlow_rotation_mode", "vicreg_rotation_mode"),
                    default="none",
                )
            ),
            rotation_deg=float(
                cls._cfg_value(
                    cfg,
                    "wmse_rotation_deg",
                    fallback=("barlow_rotation_deg", "vicreg_rotation_deg"),
                    default=0.0,
                )
            ),
            strain_std=float(
                cls._cfg_value(
                    cfg,
                    "wmse_strain_std",
                    fallback=("barlow_strain_std", "vicreg_strain_std"),
                    default=0.0,
                )
            ),
            strain_volume_preserve=bool(
                cls._cfg_value(
                    cfg,
                    "wmse_strain_volume_preserve",
                    fallback=(
                        "barlow_strain_volume_preserve",
                        "vicreg_strain_volume_preserve",
                    ),
                    default=True,
                )
            ),
            occlusion_mode=str(
                cls._cfg_value(
                    cfg,
                    "wmse_occlusion_mode",
                    fallback=("barlow_occlusion_mode", "vicreg_occlusion_mode"),
                    default="none",
                )
            ),
            occlusion_view=str(
                cls._cfg_value(
                    cfg,
                    "wmse_occlusion_view",
                    fallback=("barlow_occlusion_view", "vicreg_occlusion_view"),
                    default="second",
                )
            ),
            occlusion_slab_frac=float(
                cls._cfg_value(
                    cfg,
                    "wmse_occlusion_slab_frac",
                    fallback=("barlow_occlusion_slab_frac", "vicreg_occlusion_slab_frac"),
                    default=0.4,
                )
            ),
            occlusion_cone_deg=float(
                cls._cfg_value(
                    cfg,
                    "wmse_occlusion_cone_deg",
                    fallback=("barlow_occlusion_cone_deg", "vicreg_occlusion_cone_deg"),
                    default=20.0,
                )
            ),
            occlusion_prob=float(
                cls._cfg_value(
                    cfg,
                    "wmse_occlusion_prob",
                    fallback=("barlow_occlusion_prob", "vicreg_occlusion_prob"),
                    default=1.0,
                )
            ),
            input_dim=input_dim,
            invariant_mode=invariant_mode,
            invariant_max_factor=float(
                cls._cfg_value(
                    cfg,
                    "wmse_invariant_max_factor",
                    fallback=("barlow_invariant_max_factor", "vicreg_invariant_max_factor"),
                    default=4.0,
                )
            ),
            invariant_groups=int(
                cls._cfg_value(
                    cfg,
                    "wmse_invariant_groups",
                    fallback=("barlow_invariant_groups", "vicreg_invariant_groups"),
                    default=0,
                )
            ),
            invariant_use_third_order=bool(
                cls._cfg_value(
                    cfg,
                    "wmse_invariant_use_third_order",
                    fallback=(
                        "barlow_invariant_use_third_order",
                        "vicreg_invariant_use_third_order",
                    ),
                    default=True,
                )
            ),
            invariant_eps=float(
                cls._cfg_value(
                    cfg,
                    "wmse_invariant_eps",
                    fallback=("barlow_invariant_eps", "vicreg_invariant_eps"),
                    default=1e-6,
                )
            ),
            whitening_eps=float(
                cls._cfg_value(
                    cfg,
                    "wmse_whitening_eps",
                    fallback=("wmse_w_eps",),
                    default=0.0,
                )
            ),
            whitening_iters=int(
                cls._cfg_value(
                    cfg,
                    "wmse_whitening_iters",
                    fallback=("wmse_w_iter",),
                    default=1,
                )
            ),
            whitening_size=int(
                cls._cfg_value(
                    cfg,
                    "wmse_whitening_size",
                    fallback=("wmse_w_size",),
                    default=128,
                )
            ),
            normalize_embeddings=bool(
                cls._cfg_value(
                    cfg,
                    "wmse_normalize_embeddings",
                    fallback=("wmse_norm",),
                    default=True,
                )
            ),
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
        invariant_transform=None,
    ):
        if not self.should_run(current_epoch=current_epoch):
            return None, {}

        views = self._view_sampler.sample_view_pair(pc)
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
        h_a = self.projector(inv_a.to(dtype=proj_dtype))
        h_b = self.projector(inv_b.to(dtype=proj_dtype))
        loss, metrics = self._loss(h_a, h_b)
        if not torch.isfinite(loss).item():
            metrics["wmse_nonfinite"] = pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss, metrics

    def _distributed_randperm(self, batch_size: int, *, device) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return torch.randperm(batch_size, device=device)

        perm = torch.empty(batch_size, device=device, dtype=torch.long)
        if torch.distributed.get_rank() == 0:
            perm.copy_(torch.randperm(batch_size, device=device))
        torch.distributed.broadcast(perm, src=0)
        return perm

    def _whitening_chunks(self, batch_size: int, *, device) -> list[torch.Tensor]:
        if batch_size < 2:
            raise ValueError(
                f"W-MSE whitening requires at least 2 samples, got batch_size={batch_size}"
            )

        effective_size = min(self.whitening_size, batch_size)
        perm = self._distributed_randperm(batch_size, device=device)
        chunks = list(torch.split(perm, effective_size))
        if not chunks:
            raise RuntimeError(
                f"Failed to split whitening permutation for batch_size={batch_size}, "
                f"effective_size={effective_size}."
            )

        if chunks[-1].numel() == 1:
            if len(chunks) == 1:
                return [perm]
            chunks[-2] = torch.cat((chunks[-2], chunks[-1]), dim=0)
            chunks.pop()

        if any(chunk.numel() < 2 for chunk in chunks):
            chunk_sizes = [int(chunk.numel()) for chunk in chunks]
            raise RuntimeError(
                "W-MSE whitening produced an invalid chunk with fewer than 2 samples. "
                f"batch_size={batch_size}, whitening_size={self.whitening_size}, "
                f"chunk_sizes={chunk_sizes}."
            )
        return chunks

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

    def _pair_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if self.normalize_embeddings:
            x0 = F.normalize(x0, dim=-1, eps=1e-12)
            x1 = F.normalize(x1, dim=-1, eps=1e-12)
            return 2.0 - 2.0 * (x0 * x1).sum(dim=-1).mean()
        return F.mse_loss(x0, x1)

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_a.dtype != torch.float32:
            z_a = z_a.float()
        if z_b.dtype != torch.float32:
            z_b = z_b.float()
        z_a = self._gather_all(z_a)
        z_b = self._gather_all(z_b)
        n, _ = z_a.shape
        if n < 2:
            zero = z_a.new_tensor(0.0)
            return zero, {
                "wmse_raw_mse": zero,
                "wmse_cosine": zero,
                "wmse_whitening_chunks": zero,
                "wmse_effective_whitening_size": zero,
            }

        total_loss = z_a.new_tensor(0.0)
        total_mse = z_a.new_tensor(0.0)
        total_cosine = z_a.new_tensor(0.0)
        last_chunk_count = 0
        last_effective_size = 0

        for _ in range(self.whitening_iters):
            chunks = self._whitening_chunks(n, device=z_a.device)
            white_a = torch.empty_like(z_a, dtype=torch.float32)
            white_b = torch.empty_like(z_b, dtype=torch.float32)

            for chunk in chunks:
                white_a[chunk] = self.whitening(z_a[chunk])
                white_b[chunk] = self.whitening(z_b[chunk])

            total_loss = total_loss + self._pair_loss(white_a, white_b)
            total_mse = total_mse + F.mse_loss(white_a, white_b)
            total_cosine = total_cosine + F.cosine_similarity(
                F.normalize(white_a, dim=-1, eps=1e-12),
                F.normalize(white_b, dim=-1, eps=1e-12),
                dim=-1,
            ).mean()
            last_chunk_count = len(chunks)
            last_effective_size = max(int(chunk.numel()) for chunk in chunks)

        denom = float(self.whitening_iters)
        loss = total_loss / denom
        metrics = {
            "wmse_raw_mse": total_mse / denom,
            "wmse_cosine": total_cosine / denom,
            "wmse_whitening_chunks": z_a.new_tensor(float(last_chunk_count)),
            "wmse_effective_whitening_size": z_a.new_tensor(float(last_effective_size)),
        }
        return loss, metrics

    def _invariant(self, inv_z, eq_z):
        return self._view_sampler._invariant(inv_z, eq_z)
