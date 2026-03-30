import os
import sys

import pytorch_lightning as pl
import torch

sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.pointcontrast import PointContrastLoss
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.contrastive_learning.wmse import WMSELoss
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import cached_sample_count, get_optimizers_and_scheduler


def resolve_latent_dim(cfg):
    if hasattr(cfg, "latent_size"):
        return int(cfg.latent_size)
    if hasattr(cfg, "encoder") and hasattr(cfg.encoder, "kwargs"):
        latent_size = cfg.encoder.kwargs.get("latent_size", None)
        if latent_size is not None:
            return int(latent_size)
    return None


class TemporalSSLModule(pl.LightningModule):
    """Temporal self-supervised training on local-structure frame sequences."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        data_cfg = getattr(cfg, "data", None)
        if data_cfg is None:
            raise ValueError("TemporalSSLModule requires a data configuration.")
        data_kind = str(getattr(data_cfg, "kind", "")).strip().lower()
        if data_kind != "temporal_lammps":
            raise ValueError(
                "TemporalSSLModule requires data.kind='temporal_lammps', "
                f"got {data_kind!r}."
            )

        self.sequence_length = int(getattr(data_cfg, "sequence_length", 0))
        if self.sequence_length < 3:
            raise ValueError(
                "TemporalSSLModule requires data.sequence_length >= 3 so previous and next frames exist, "
                f"got {self.sequence_length}."
            )

        center_frame_index_cfg = getattr(cfg, "temporal_center_frame_index", None)
        if center_frame_index_cfg is None:
            if self.sequence_length % 2 == 0:
                raise ValueError(
                    "TemporalSSLModule needs an unambiguous center frame. "
                    f"data.sequence_length={self.sequence_length} is even, so set temporal_center_frame_index "
                    "explicitly to a value in [1, sequence_length - 2]."
                )
            self.center_frame_index = self.sequence_length // 2
        else:
            self.center_frame_index = int(center_frame_index_cfg)
        if self.center_frame_index <= 0 or self.center_frame_index >= self.sequence_length - 1:
            raise ValueError(
                "temporal_center_frame_index must leave both a previous and next frame available. "
                f"Got temporal_center_frame_index={self.center_frame_index}, "
                f"sequence_length={self.sequence_length}."
            )
        self.use_temporal_vicreg_views = bool(getattr(cfg, "temporal_vicreg_use_adjacent_views", True))

        if bool(getattr(cfg, "vicreg_use_ri_mae_backbone", False)):
            raise ValueError(
                "vicreg_use_ri_mae_backbone is deprecated. "
                "Use encoder.name='RI_MAE_Invariant' with vicreg_invariant_mode='passthrough'."
            )

        enc_cfg = getattr(cfg, "encoder", None)
        enc_name = str(getattr(enc_cfg, "name", "")).strip()
        if enc_name == "RI_MAE_Invariant":
            vic_mode = str(getattr(cfg, "vicreg_invariant_mode", "norms")).lower()
            if vic_mode not in {"passthrough", "norms"}:
                raise ValueError(
                    "RI_MAE_Invariant returns invariant 2D features directly. "
                    f"Unsupported vicreg_invariant_mode='{vic_mode}'. "
                    "Use 'passthrough' (recommended) or 'norms'."
                )

        self.encoder, _ = build_model(cfg)
        latent_dim = resolve_latent_dim(cfg)

        self.sample_points = int(getattr(data_cfg, "num_points", 0))
        model_points = getattr(data_cfg, "model_points", None)
        if model_points is None:
            model_points = getattr(cfg, "model_points", None)
        if model_points is not None:
            model_points = int(model_points)
            if model_points <= 0:
                model_points = None
        self.model_points = model_points

        if self.model_points is not None and self.sample_points and self.model_points > self.sample_points:
            raise ValueError(
                f"model_points ({self.model_points}) cannot exceed data.num_points ({self.sample_points})"
            )

        self.norms_only_latent = bool(getattr(cfg, "contrastive_norms_only_latent", False))
        invariant_mode_override = "norms" if self.norms_only_latent else None
        self.barlow = BarlowTwinsLoss.from_config(
            cfg,
            input_dim=latent_dim,
            invariant_mode_override=invariant_mode_override,
        )
        vicreg_enabled = bool(getattr(cfg, "vicreg_enabled", False))
        self.vicreg = (
            VICRegLoss.from_config(
                cfg,
                input_dim=latent_dim,
                invariant_mode_override=invariant_mode_override,
            )
            if vicreg_enabled
            else None
        )
        self.wmse = WMSELoss.from_config(
            cfg,
            input_dim=latent_dim,
            invariant_mode_override=invariant_mode_override,
        )
        self.pointcontrast = PointContrastLoss.from_config(
            cfg,
            input_dim=latent_dim,
            invariant_mode_override=invariant_mode_override,
        )
        self._active_invariant_losses = self._resolve_active_invariant_losses()
        self._shared_invariant_spec = self._resolve_shared_invariant_spec()

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._warned_synthetic_eq_latent = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))

    @property
    def barlow_projector(self):
        return self.barlow.projector

    @property
    def vicreg_projector(self):
        return self.vicreg.projector if self.vicreg is not None else None

    @property
    def wmse_projector(self):
        return self.wmse.projector if self.wmse is not None else None

    @property
    def pointcontrast_projector(self):
        return self.pointcontrast.projector if self.pointcontrast is not None else None

    def _resolve_active_invariant_losses(self) -> dict[str, object]:
        losses: dict[str, object] = {}
        if getattr(self.barlow, "invariant_head", None) is not None and self.barlow.enabled and self.barlow.weight > 0:
            losses["barlow"] = self.barlow
        if (
            self.vicreg is not None
            and getattr(self.vicreg, "invariant_head", None) is not None
            and self.vicreg.enabled
            and self.vicreg.weight > 0
        ):
            losses["vicreg"] = self.vicreg
        if (
            self.wmse is not None
            and getattr(self.wmse, "invariant_head", None) is not None
            and self.wmse.enabled
            and self.wmse.weight > 0
        ):
            losses["wmse"] = self.wmse
        if (
            self.pointcontrast is not None
            and getattr(self.pointcontrast, "invariant_head", None) is not None
            and self.pointcontrast.enabled
            and self.pointcontrast.weight > 0
        ):
            losses["pointcontrast"] = self.pointcontrast
        if losses:
            return losses

        if getattr(self.barlow, "invariant_head", None) is not None:
            return {"barlow": self.barlow}
        if self.vicreg is not None and getattr(self.vicreg, "invariant_head", None) is not None:
            return {"vicreg": self.vicreg}
        if self.wmse is not None and getattr(self.wmse, "invariant_head", None) is not None:
            return {"wmse": self.wmse}
        if self.pointcontrast is not None and getattr(self.pointcontrast, "invariant_head", None) is not None:
            return {"pointcontrast": self.pointcontrast}
        return {}

    @staticmethod
    def _invariant_spec(loss_obj: object) -> dict[str, int | str] | None:
        head = getattr(loss_obj, "invariant_head", None)
        if head is None:
            return None
        return {
            "mode": str(getattr(head, "mode", "norms")).lower(),
            "channels": int(getattr(head, "channels", 0)),
            "output_dim": int(getattr(head, "output_dim", 0)),
            "num_groups": int(getattr(head, "num_groups", 0)),
            "num_second_order": int(getattr(head, "num_second_order", 0)),
            "num_third_order": int(getattr(head, "num_third_order", 0)),
        }

    def _resolve_shared_invariant_spec(self) -> dict[str, int | str] | None:
        if not self._active_invariant_losses:
            return None

        named_specs: dict[str, dict[str, int | str]] = {}
        for name, loss_obj in self._active_invariant_losses.items():
            spec = self._invariant_spec(loss_obj)
            if spec is None:
                continue
            named_specs[name] = spec

        if not named_specs:
            return None

        names = list(named_specs.keys())
        ref_name = names[0]
        ref_spec = named_specs[ref_name]
        for name in names[1:]:
            if named_specs[name] != ref_spec:
                raise ValueError(
                    "Active contrastive objectives must use matching invariant specs, but got: "
                    f"{ref_name}={ref_spec}, {name}={named_specs[name]}. "
                    "Align *invariant_* settings across objectives (or disable one objective)."
                )
        return ref_spec

    def _shared_invariant(self, z_inv_model, eq_z):
        if eq_z is not None:
            z_inv_model = None
        if "barlow" in self._active_invariant_losses:
            return self.barlow._invariant(z_inv_model, eq_z)
        if "vicreg" in self._active_invariant_losses and self.vicreg is not None:
            return self.vicreg._invariant(z_inv_model, eq_z)
        if "wmse" in self._active_invariant_losses and self.wmse is not None:
            return self.wmse._invariant(z_inv_model, eq_z)
        if "pointcontrast" in self._active_invariant_losses and self.pointcontrast is not None:
            return self.pointcontrast._invariant(z_inv_model, eq_z)
        return self.barlow._invariant(z_inv_model, eq_z)

    def _prepare_encoder_input(self, pc: torch.Tensor) -> torch.Tensor:
        if getattr(self.encoder, "expects_channel_first", False):
            return pc.permute(0, 2, 1).contiguous()
        return pc

    def _status_print(self, message: str) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.print(message)
            return
        print(message)

    @staticmethod
    def _channel_basis(channels: int, *, device, dtype) -> torch.Tensor:
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        idx = torch.arange(channels, device=device, dtype=torch.float32)
        theta = (2.0 * torch.pi * idx) / float(channels)
        z = (2.0 * (idx + 0.5) / float(channels)) - 1.0
        z = z.clamp(min=-0.999999, max=0.999999)
        radial = torch.sqrt((1.0 - z * z).clamp_min(0.0))
        basis = torch.stack(
            (
                radial * torch.cos(theta),
                radial * torch.sin(theta),
                z,
            ),
            dim=-1,
        )
        return basis.to(dtype=dtype)

    def _maybe_synthesize_eq_latent(
        self,
        z_inv_model: torch.Tensor | None,
        eq_z: torch.Tensor | None,
        *,
        source: str,
    ) -> torch.Tensor | None:
        if eq_z is not None:
            return eq_z
        if z_inv_model is None:
            return None
        if not torch.is_tensor(z_inv_model) or z_inv_model.dim() != 2:
            raise ValueError(
                f"{source}: expected z_inv_model to be a 2D tensor when eq_z is missing, "
                f"got type={type(z_inv_model)} with shape={getattr(z_inv_model, 'shape', None)}."
            )

        spec = self._shared_invariant_spec
        if spec is None:
            return None

        inv_dim = int(z_inv_model.shape[1])
        channels = int(spec["channels"])
        output_dim = int(spec["output_dim"])
        mode = str(spec["mode"])

        if inv_dim == output_dim:
            return None
        if channels <= 0:
            raise ValueError(
                f"{source}: invariant_head has invalid channels={channels} while inv_dim={inv_dim} "
                f"and output_dim={output_dim}."
            )
        if inv_dim != channels:
            raise ValueError(
                f"{source}: cannot synthesize eq_z because inv_dim={inv_dim} does not match "
                f"invariant_head.channels={channels}; output_dim={output_dim}, mode='{mode}'."
            )

        basis = self._channel_basis(channels, device=z_inv_model.device, dtype=z_inv_model.dtype)
        synthetic_eq = z_inv_model.unsqueeze(-1) * basis.unsqueeze(0)

        if synthetic_eq.shape != (z_inv_model.shape[0], channels, 3):
            raise RuntimeError(
                f"{source}: synthesized eq_z has unexpected shape {tuple(synthetic_eq.shape)}; "
                f"expected {(z_inv_model.shape[0], channels, 3)}."
            )

        if not self._warned_synthetic_eq_latent:
            self._status_print(
                "[temporal-ssl] Encoder did not return eq_z; synthesized pseudo-equivariant latent "
                f"from z_inv_model for invariant mode='{mode}' "
                f"(inv_dim={inv_dim}, output_dim={output_dim})."
            )
            self._warned_synthetic_eq_latent = True
        return synthetic_eq

    def _split_encoder_output(self, enc_out):
        if isinstance(enc_out, (tuple, list)):
            if not enc_out:
                raise ValueError("Encoder returned empty output")
            z_inv_model = enc_out[0]
            eq_z = None
            for candidate in enc_out[1:]:
                if not (torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3):
                    continue
                if torch.is_tensor(z_inv_model) and z_inv_model.dim() == 2:
                    if candidate.shape[1] == z_inv_model.shape[1]:
                        eq_z = candidate
                        break
                    if candidate.shape[1] == 3:
                        continue
                if candidate.shape[1] != 3:
                    eq_z = candidate
                    break
            eq_z = self._maybe_synthesize_eq_latent(
                z_inv_model,
                eq_z,
                source="split_encoder_output(tuple)",
            )
            return z_inv_model, eq_z
        z_inv_model = enc_out
        eq_z = self._maybe_synthesize_eq_latent(
            z_inv_model,
            None,
            source="split_encoder_output(tensor)",
        )
        return z_inv_model, eq_z

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _center_frame(self, sequence_points: torch.Tensor) -> torch.Tensor:
        return sequence_points[:, self.center_frame_index]

    def _validate_temporal_points(self, sequence_points: torch.Tensor) -> None:
        if not torch.is_tensor(sequence_points):
            raise TypeError(f"Temporal batch points must be a torch.Tensor, got {type(sequence_points)}")
        if sequence_points.dim() != 4 or sequence_points.shape[-1] != 3:
            raise ValueError(
                "TemporalSSLModule expects batch['points'] with shape (B, T, N, 3), "
                f"got {tuple(sequence_points.shape)}."
            )
        if int(sequence_points.shape[1]) != self.sequence_length:
            raise ValueError(
                "Temporal sequence length mismatch between config and batch. "
                f"Configured data.sequence_length={self.sequence_length}, "
                f"batch_sequence_length={int(sequence_points.shape[1])}."
            )

    def _temporal_meta_from_batch(self, batch: dict) -> dict:
        return {
            "class_id": batch.get("class_id"),
            "instance_id": batch.get("instance_id"),
            "rotation": batch.get("rotation"),
            "center_atom_id": batch.get("center_atom_id"),
            "anchor_frame_index": batch.get("anchor_frame_index"),
            "anchor_timestep": batch.get("anchor_timestep"),
            "frame_indices": batch.get("frame_indices"),
            "timesteps": batch.get("timesteps"),
            "source_path": batch.get("source_path"),
        }

    def _unpack_batch(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(
                f"TemporalSSLModule expects dict batches from TemporalLAMMPSDumpDataset, got {type(batch)}."
            )
        if "points" not in batch:
            raise KeyError("Temporal batch is missing required key 'points'.")
        sequence_points = batch["points"]
        self._validate_temporal_points(sequence_points)
        return self._center_frame(sequence_points), self._temporal_meta_from_batch(batch)

    def _unpack_temporal_batch(self, batch):
        center_pc, meta = self._unpack_batch(batch)
        sequence_points = batch["points"]
        prev_pc = sequence_points[:, self.center_frame_index - 1]
        next_pc = sequence_points[:, self.center_frame_index + 1]
        return center_pc, prev_pc, next_pc, meta

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
        pc_raw, meta = self._unpack_batch(batch)
        class_id = meta.get("class_id")
        if class_id is None:
            return None, None

        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)
        z_inv_model, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        features = z_inv_contrastive if z_inv_contrastive is not None else z_inv_model
        if features is None:
            return None, None
        return features.detach().to(torch.float32), class_id

    def _contrastive_invariant_latent(self, z_inv_model, eq_z):
        return self._shared_invariant(z_inv_model, eq_z)

    def _contrastive_invariant_from_eq_latent(
        self,
        eq_z,
        z_inv_model=None,
        *,
        stage: str | None = None,
    ):
        stage_name = stage if stage is not None else "unknown"
        eq_ready = self._maybe_synthesize_eq_latent(
            z_inv_model,
            eq_z,
            source=f"contrastive_invariant_from_eq_latent(stage={stage_name})",
        )
        if eq_ready is not None:
            return self._shared_invariant(None, eq_ready)
        if z_inv_model is not None and not self._warned_cache_eq_fallback:
            self._status_print(
                f"[temporal-ssl/cache] eq_z is missing at stage='{stage_name}'. "
                "Falling back to encoder invariant output (z_inv_model) for cached z_inv_contrastive."
            )
            self._warned_cache_eq_fallback = True
        return self._shared_invariant(z_inv_model, None)

    def _prepare_explicit_view(self, pc: torch.Tensor, *, target_points: int | None, crop_mode: str) -> torch.Tensor:
        return crop_to_num_points(pc, target_points, mode=crop_mode)

    def _build_vicreg_temporal_views(self, prev_pc: torch.Tensor, next_pc: torch.Tensor):
        if self.vicreg is None:
            return None
        return {
            "y_a": self._prepare_explicit_view(
                prev_pc,
                target_points=self.vicreg.view_points,
                crop_mode=self.vicreg.view_crop_mode,
            ),
            "y_b": self._prepare_explicit_view(
                next_pc,
                target_points=self.vicreg.view_points,
                crop_mode=self.vicreg.view_crop_mode,
            ),
        }

    def forward(self, pc: torch.Tensor):
        if torch.is_tensor(pc) and pc.dim() == 4:
            self._validate_temporal_points(pc)
            pc = self._center_frame(pc)
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        z_inv_model, eq_z = self._split_encoder_output(enc_out)
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        return z_inv_contrastive, z_inv_model, eq_z

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, prev_pc, next_pc, meta = self._unpack_temporal_batch(batch)
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        prev_pc = prev_pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        next_pc = next_pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)

        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )

        losses = {}

        barlow_loss, barlow_metrics = self.barlow.compute_loss(
            pc=pc_raw,
            encoder=self.encoder,
            prepare_input=self._prepare_encoder_input,
            split_output=self._split_encoder_output,
            current_epoch=int(self.current_epoch),
            invariant_transform=self._shared_invariant,
        )
        if barlow_loss is not None:
            losses["barlow"] = barlow_loss
        for name, value in barlow_metrics.items():
            self._log_metric(stage, name, value)

        if self.vicreg is not None:
            vicreg_views = (
                self._build_vicreg_temporal_views(prev_pc, next_pc)
                if self.use_temporal_vicreg_views
                else None
            )
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                views=vicreg_views,
                invariant_transform=self._shared_invariant,
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value)

        if self.wmse is not None:
            wmse_loss, wmse_metrics = self.wmse.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                invariant_transform=self._shared_invariant,
            )
            if wmse_loss is not None:
                losses["wmse"] = wmse_loss
            for name, value in wmse_metrics.items():
                self._log_metric(stage, name, value)

        if self.pointcontrast is not None:
            pointcontrast_loss, pointcontrast_metrics = self.pointcontrast.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                invariant_transform=self._shared_invariant,
            )
            if pointcontrast_loss is not None:
                losses["pointcontrast"] = pointcontrast_loss
            for name, value in pointcontrast_metrics.items():
                self._log_metric(stage, name, value)

        total_loss = None
        if "barlow" in losses:
            total_loss = self.barlow.weight * losses["barlow"]
        if "vicreg" in losses and self.vicreg is not None:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "wmse" in losses and self.wmse is not None:
            wmse_total = self.wmse.weight * losses["wmse"]
            total_loss = wmse_total if total_loss is None else total_loss + wmse_total
        if "pointcontrast" in losses and self.pointcontrast is not None:
            pointcontrast_total = self.pointcontrast.weight * losses["pointcontrast"]
            total_loss = pointcontrast_total if total_loss is None else total_loss + pointcontrast_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if not torch.isfinite(total_loss).item():
            self._consecutive_nan_steps += 1
            self._log_metric(stage, "loss_nonfinite", 1.0, on_step=True, on_epoch=False)
            if self._consecutive_nan_steps >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Training produced {self._consecutive_nan_steps} consecutive "
                    f"non-finite losses. Halting to prevent silent divergence."
                )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._consecutive_nan_steps = 0

        metrics_to_log = {"loss": total_loss}
        if "barlow" in losses:
            metrics_to_log["barlow"] = losses["barlow"]
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "wmse" in losses:
            metrics_to_log["wmse"] = losses["wmse"]
        if "pointcontrast" in losses:
            metrics_to_log["pointcontrast"] = losses["pointcontrast"]

        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys))

        if should_cache_stage:
            limit = self._cache_limit_for_stage(stage)
            cache = self._supervised_cache.get(stage)
            already_cached = cached_sample_count(cache) if cache is not None else 0
            if limit is None or already_cached < limit:
                with torch.no_grad():
                    z_inv_model, eq_z = self._split_encoder_output(
                        self.encoder(self._prepare_encoder_input(pc))
                    )
                    z_inv_contrastive = self._contrastive_invariant_from_eq_latent(
                        eq_z, z_inv_model=z_inv_model, stage=stage
                    )
                if z_inv_contrastive is not None:
                    self._cache_supervised_batch(
                        stage,
                        z_inv_contrastive,
                        meta,
                        encoder_features=z_inv_contrastive,
                    )

        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, **kwargs) -> None:
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    def _handle_epoch_boundary(self, stage: str, is_start: bool):
        if is_start:
            self._reset_supervised_cache(stage)
        else:
            self._log_supervised_metrics(stage)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._handle_epoch_boundary("train", True)

    def on_train_epoch_end(self) -> None:
        self._handle_epoch_boundary("train", False)
        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._handle_epoch_boundary("val", True)

    def on_validation_epoch_end(self) -> None:
        self._handle_epoch_boundary("val", False)
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._handle_epoch_boundary("test", True)

    def on_test_epoch_end(self) -> None:
        self._handle_epoch_boundary("test", False)
        super().on_test_epoch_end()

    def _reset_supervised_cache(self, stage: str) -> None:
        reset_supervised_cache(self, stage)

    def _cache_limit_for_stage(self, stage: str):
        return cache_limit_for_stage(self, stage)

    def _cache_supervised_batch(
        self,
        stage: str,
        z_inv_contrastive: torch.Tensor,
        meta: dict,
        encoder_features: torch.Tensor | None = None,
    ) -> None:
        cache_supervised_batch(
            self,
            stage,
            z_inv_contrastive,
            meta,
            encoder_features=encoder_features,
        )

    def _log_supervised_metrics(self, stage: str) -> None:
        log_supervised_metrics(self, stage)


__all__ = ["TemporalSSLModule"]
