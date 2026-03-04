import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss


class PointContrastLoss(nn.Module):
    """PointContrast-style instance discrimination on augmented point-cloud views.

    This implementation follows the PointContrast training philosophy (positive
    pairs from two transformed views + in-batch negatives) while operating on
    the existing encoder invariants available in this repository.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        temperature: float,
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
        queue_size: int = 0,
        symmetric: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.enabled = bool(enabled)
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.embed_dim = int(embed_dim)
        self.start_epoch = max(0, int(start_epoch))
        self.queue_size = int(queue_size)
        self.symmetric = bool(symmetric)
        self.normalize_embeddings = bool(normalize_embeddings)

        if self.weight < 0:
            raise ValueError(f"pointcontrast_weight must be >= 0, got {self.weight}")
        if self.temperature <= 0:
            raise ValueError(
                f"pointcontrast_temperature must be > 0, got {self.temperature}"
            )
        if self.embed_dim <= 0:
            raise ValueError(f"pointcontrast_embed_dim must be > 0, got {self.embed_dim}")
        if self.queue_size < 0:
            raise ValueError(f"pointcontrast_queue_size must be >= 0, got {self.queue_size}")

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
                    "PointContrast requires latent_size/encoder latent dim to build projector input"
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

        self.register_buffer(
            "_queue",
            torch.zeros((max(self.queue_size, 0), self.embed_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_queue_ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_queue_filled", torch.zeros(1, dtype=torch.long), persistent=False)

    @staticmethod
    def _cfg_value(cfg, primary: str, fallback: tuple[str, ...] = (), default=None):
        value = getattr(cfg, primary, None)
        if value is not None:
            return value
        for field in fallback:
            value = getattr(cfg, field, None)
            if value is not None:
                return value
        return default

    @classmethod
    def from_config(cls, cfg, *, input_dim, invariant_mode_override: str | None = None):
        data_cfg = getattr(cfg, "data", None)
        view_points = cls._cfg_value(
            cfg,
            "pointcontrast_view_points",
            fallback=("barlow_view_points",),
            default=None,
        )
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "model_points", None)
        if view_points is None and data_cfg is not None:
            view_points = getattr(data_cfg, "num_points", None)

        jitter_mode = str(
            cls._cfg_value(
                cfg,
                "pointcontrast_jitter_mode",
                fallback=("barlow_jitter_mode",),
                default="absolute",
            )
        ).lower()
        jitter_scale_cfg = cls._cfg_value(
            cfg,
            "pointcontrast_jitter_scale",
            fallback=("barlow_jitter_scale",),
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
                    "pointcontrast_invariant_mode",
                    fallback=("barlow_invariant_mode",),
                    default="norms",
                )
            ).lower()
        )

        return cls(
            enabled=bool(cls._cfg_value(cfg, "pointcontrast_enabled", default=False)),
            weight=float(cls._cfg_value(cfg, "pointcontrast_weight", default=0.0)),
            temperature=float(
                cls._cfg_value(cfg, "pointcontrast_temperature", default=0.07)
            ),
            embed_dim=int(cls._cfg_value(cfg, "pointcontrast_embed_dim", default=256)),
            start_epoch=int(cls._cfg_value(cfg, "pointcontrast_start_epoch", default=0)),
            jitter_std=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_jitter_std",
                    fallback=("barlow_jitter_std",),
                    default=0.01,
                )
            ),
            jitter_mode=jitter_mode,
            jitter_scale=float(jitter_scale),
            drop_ratio=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_drop_ratio",
                    fallback=("barlow_drop_ratio",),
                    default=0.2,
                )
            ),
            view_points=int(view_points) if view_points is not None else None,
            neighbor_view=bool(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_neighbor_view",
                    fallback=("barlow_neighbor_view",),
                    default=False,
                )
            ),
            neighbor_view_mode=str(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_neighbor_view_mode",
                    fallback=("barlow_neighbor_view_mode",),
                    default="both",
                )
            ),
            neighbor_k=int(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_neighbor_k",
                    fallback=("barlow_neighbor_k",),
                    default=8,
                )
            ),
            neighbor_max_relative_distance=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_neighbor_max_relative_distance",
                    fallback=("barlow_neighbor_max_relative_distance",),
                    default=0.0,
                )
            ),
            view_crop_mode=str(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_view_crop_mode",
                    fallback=("barlow_view_crop_mode",),
                    default="random",
                )
            ),
            drop_apply_to_both=bool(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_drop_apply_to_both",
                    fallback=("barlow_drop_apply_to_both",),
                    default=True,
                )
            ),
            rotation_mode=str(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_rotation_mode",
                    fallback=("barlow_rotation_mode",),
                    default="none",
                )
            ),
            rotation_deg=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_rotation_deg",
                    fallback=("barlow_rotation_deg",),
                    default=0.0,
                )
            ),
            strain_std=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_strain_std",
                    fallback=("barlow_strain_std",),
                    default=0.0,
                )
            ),
            strain_volume_preserve=bool(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_strain_volume_preserve",
                    fallback=("barlow_strain_volume_preserve",),
                    default=True,
                )
            ),
            occlusion_mode=str(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_occlusion_mode",
                    fallback=("barlow_occlusion_mode",),
                    default="none",
                )
            ),
            occlusion_view=str(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_occlusion_view",
                    fallback=("barlow_occlusion_view",),
                    default="second",
                )
            ),
            occlusion_slab_frac=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_occlusion_slab_frac",
                    fallback=("barlow_occlusion_slab_frac",),
                    default=0.4,
                )
            ),
            occlusion_cone_deg=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_occlusion_cone_deg",
                    fallback=("barlow_occlusion_cone_deg",),
                    default=20.0,
                )
            ),
            occlusion_prob=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_occlusion_prob",
                    fallback=("barlow_occlusion_prob",),
                    default=1.0,
                )
            ),
            input_dim=input_dim,
            invariant_mode=invariant_mode,
            invariant_max_factor=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_invariant_max_factor",
                    fallback=("barlow_invariant_max_factor",),
                    default=4.0,
                )
            ),
            invariant_groups=int(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_invariant_groups",
                    fallback=("barlow_invariant_groups",),
                    default=0,
                )
            ),
            invariant_use_third_order=bool(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_invariant_use_third_order",
                    fallback=("barlow_invariant_use_third_order",),
                    default=True,
                )
            ),
            invariant_eps=float(
                cls._cfg_value(
                    cfg,
                    "pointcontrast_invariant_eps",
                    fallback=("barlow_invariant_eps",),
                    default=1e-6,
                )
            ),
            queue_size=int(cls._cfg_value(cfg, "pointcontrast_queue_size", default=0)),
            symmetric=bool(cls._cfg_value(cfg, "pointcontrast_symmetric", default=True)),
            normalize_embeddings=bool(
                cls._cfg_value(cfg, "pointcontrast_normalize_embeddings", default=True)
            ),
        )

    def should_run(self, *, current_epoch: int) -> bool:
        return bool(
            self.enabled
            and self.weight > 0
            and self.projector is not None
            and int(current_epoch) >= self.start_epoch
        )

    def _invariant(self, inv_z, eq_z):
        return self._view_sampler._invariant(inv_z, eq_z)

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
            sampled = self._view_sampler.sample_view_pair(pc)
            y_a = sampled["y_a"]
            y_b = sampled["y_b"]
        else:
            y_a = views["y_a"]
            y_b = views["y_b"]

        enc_a = encoder(prepare_input(y_a))
        inv_a, eq_a = split_output(enc_a)
        if invariant_transform is None:
            inv_a = self._view_sampler._invariant(inv_a, eq_a)
        else:
            inv_a = invariant_transform(inv_a, eq_a)

        enc_b = encoder(prepare_input(y_b))
        inv_b, eq_b = split_output(enc_b)
        if invariant_transform is None:
            inv_b = self._view_sampler._invariant(inv_b, eq_b)
        else:
            inv_b = invariant_transform(inv_b, eq_b)

        if inv_a is None or inv_b is None:
            return None, {}

        proj_dtype = next(self.projector.parameters()).dtype
        z_a = self.projector(inv_a.to(dtype=proj_dtype))
        z_b = self.projector(inv_b.to(dtype=proj_dtype))

        loss, metrics = self._loss(z_a, z_b)
        if not torch.isfinite(loss).item():
            metrics["pointcontrast_nonfinite"] = pc.new_tensor(1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.queue_size > 0 and self.training:
            with torch.no_grad():
                queue_feats = torch.cat([z_a.detach(), z_b.detach()], dim=0).to(torch.float32)
                if self.normalize_embeddings:
                    queue_feats = F.normalize(queue_feats, dim=1)
                queue_feats = self._gather_all(queue_feats)
                self._enqueue(queue_feats)
            metrics["pointcontrast_queue_fill"] = z_a.new_tensor(
                self._queue_fill_fraction(), dtype=torch.float32
            )

        return loss, metrics

    def _loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z_a.dim() != 2 or z_b.dim() != 2:
            raise ValueError(
                "PointContrast expects projector outputs of shape (B, D), got "
                f"z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}"
            )
        if z_a.shape != z_b.shape:
            raise ValueError(
                "PointContrast view embeddings must have the same shape, got "
                f"z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}"
            )

        local_batch, dim = z_a.shape
        if local_batch <= 0:
            raise ValueError("PointContrast received an empty local batch")
        if dim != self.embed_dim:
            raise ValueError(
                "PointContrast projector output dim mismatch: "
                f"expected {self.embed_dim}, got {dim}"
            )

        z_a = z_a.to(torch.float32)
        z_b = z_b.to(torch.float32)
        if self.normalize_embeddings:
            z_a = F.normalize(z_a, dim=1)
            z_b = F.normalize(z_b, dim=1)

        z_a_all, sizes_a = self._gather_all(z_a, return_sizes=True)
        z_b_all, sizes_b = self._gather_all(z_b, return_sizes=True)
        if sizes_a != sizes_b:
            raise RuntimeError(
                "Distributed gather size mismatch between two PointContrast views: "
                f"sizes_a={sizes_a}, sizes_b={sizes_b}."
            )

        world_batch = int(z_a_all.shape[0])
        if world_batch <= 0:
            raise RuntimeError("PointContrast gathered zero global samples")

        rank = self._distributed_rank()
        if rank >= len(sizes_a):
            raise RuntimeError(
                "Distributed rank out of bounds for gathered size metadata: "
                f"rank={rank}, sizes={sizes_a}."
            )
        rank_offset = int(sum(sizes_a[:rank]))

        targets = torch.arange(local_batch, device=z_a.device, dtype=torch.long) + rank_offset
        if int(targets.max().item()) >= world_batch:
            raise RuntimeError(
                "PointContrast positive index is out of range: "
                f"max_target={int(targets.max().item())}, world_batch={world_batch}, "
                f"sizes={sizes_a}, rank={rank}."
            )

        logits_ab = z_a @ z_b_all.T
        logits_ba = z_b @ z_a_all.T

        queue_view = self._queue_view(device=z_a.device, dtype=z_a.dtype)
        if queue_view is not None:
            logits_ab = torch.cat([logits_ab, z_a @ queue_view.T], dim=1)
            logits_ba = torch.cat([logits_ba, z_b @ queue_view.T], dim=1)

        logits_ab = logits_ab / self.temperature
        logits_ba = logits_ba / self.temperature

        loss_ab = F.cross_entropy(logits_ab, targets)
        if self.symmetric:
            loss_ba = F.cross_entropy(logits_ba, targets)
            loss = 0.5 * (loss_ab + loss_ba)
        else:
            loss_ba = logits_ab.new_tensor(0.0)
            loss = loss_ab

        with torch.no_grad():
            row_idx = torch.arange(local_batch, device=z_a.device)
            pos_logit = logits_ab[row_idx, targets]
            best_logit = logits_ab.max(dim=1).values
            logit_margin = pos_logit - best_logit

            pos_sim = (z_a * z_b).sum(dim=1)
            neg_current = z_a @ z_b_all.T
            neg_mask = torch.ones_like(neg_current, dtype=torch.bool)
            neg_mask[row_idx, targets] = False
            neg_vals = neg_current[neg_mask]
            if queue_view is not None:
                queue_vals = (z_a @ queue_view.T).reshape(-1)
                neg_vals = torch.cat([neg_vals, queue_vals], dim=0)
            neg_mean = neg_vals.mean() if neg_vals.numel() > 0 else z_a.new_tensor(0.0)

            metrics = {
                "pointcontrast_ce_ab": loss_ab.detach(),
                "pointcontrast_pos_sim": pos_sim.mean().detach(),
                "pointcontrast_neg_sim": neg_mean.detach(),
                "pointcontrast_logit_margin": logit_margin.mean().detach(),
            }
            if self.symmetric:
                metrics["pointcontrast_ce_ba"] = loss_ba.detach()

        return loss, metrics

    @staticmethod
    def _distributed_rank() -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return 0
        return int(torch.distributed.get_rank())

    def _gather_all(
        self,
        z: torch.Tensor,
        *,
        return_sizes: bool = False,
    ):
        if z.dim() < 1:
            raise ValueError(f"Expected tensor with batch dimension, got shape={tuple(z.shape)}")

        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            sizes = [int(z.shape[0])]
            return (z, sizes) if return_sizes else z

        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            sizes = [int(z.shape[0])]
            return (z, sizes) if return_sizes else z

        local_size = torch.tensor([z.shape[0]], device=z.device, dtype=torch.long)
        sizes_t = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(sizes_t, local_size)
        sizes = [int(v.item()) for v in sizes_t]
        max_size = max(sizes)

        if max_size == 0:
            gathered_z = z.new_zeros((0, *z.shape[1:]))
            return (gathered_z, sizes) if return_sizes else gathered_z

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

        chunks = [chunk[:size] for chunk, size in zip(gathered, sizes) if size > 0]
        gathered_z = torch.cat(chunks, dim=0) if chunks else z.new_zeros((0, *z.shape[1:]))
        return (gathered_z, sizes) if return_sizes else gathered_z

    def _queue_view(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.queue_size <= 0:
            return None
        filled = int(self._queue_filled.item())
        if filled <= 0:
            return None
        queue = self._queue[:filled]
        if queue.device != device or queue.dtype != dtype:
            queue = queue.to(device=device, dtype=dtype)
        return queue

    @torch.no_grad()
    def _enqueue(self, feats: torch.Tensor) -> None:
        if self.queue_size <= 0:
            return
        if feats.dim() != 2:
            raise ValueError(f"Expected queue features with shape (B,D), got {tuple(feats.shape)}")
        if feats.shape[1] != self.embed_dim:
            raise ValueError(
                "Queue feature dimension mismatch: "
                f"expected {self.embed_dim}, got {feats.shape[1]}"
            )
        if feats.shape[0] <= 0:
            return

        feats = feats.detach().to(device=self._queue.device, dtype=self._queue.dtype)
        incoming = int(feats.shape[0])
        qsize = int(self.queue_size)

        if incoming >= qsize:
            self._queue.copy_(feats[-qsize:])
            self._queue_ptr.zero_()
            self._queue_filled.fill_(qsize)
            return

        ptr = int(self._queue_ptr.item())
        end = ptr + incoming

        if end <= qsize:
            self._queue[ptr:end] = feats
        else:
            first = qsize - ptr
            self._queue[ptr:] = feats[:first]
            self._queue[: end - qsize] = feats[first:]

        self._queue_ptr.fill_(end % qsize)
        filled = min(qsize, int(self._queue_filled.item()) + incoming)
        self._queue_filled.fill_(filled)

    def _queue_fill_fraction(self) -> float:
        if self.queue_size <= 0:
            return 0.0
        return float(int(self._queue_filled.item())) / float(self.queue_size)
