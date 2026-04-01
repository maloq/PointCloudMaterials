import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.pointcontrast import PointContrastLoss
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.contrastive_learning.wmse import WMSELoss
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import get_optimizers_and_scheduler, cached_sample_count


def resolve_latent_dim(cfg):
    if hasattr(cfg, "latent_size"):
        return int(cfg.latent_size)
    if hasattr(cfg, "encoder") and hasattr(cfg.encoder, "kwargs"):
        latent_size = cfg.encoder.kwargs.get("latent_size", None)
        if latent_size is not None:
            return int(latent_size)
    return None


class BarlowTwinsModule(pl.LightningModule):
    """
    Self-supervised Barlow Twins / VICReg / W-MSE / PointContrast training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        if bool(getattr(cfg, "vicreg_use_ri_mae_backbone", False)):
            raise ValueError(
                "vicreg_use_ri_mae_backbone is deprecated. "
                "Use encoder.name='RI_MAE_Invariant'. Contrastive training now uses a fixed norms-only invariant path."
            )

        # Build encoder (decoder is not used for contrastive training)
        self.encoder, _ = build_model(cfg)
        latent_dim = resolve_latent_dim(cfg)

        data_cfg = getattr(cfg, "data", None)
        self.sample_points = int(getattr(data_cfg, "num_points", 0)) if data_cfg is not None else 0
        model_points = getattr(data_cfg, "model_points", None) if data_cfg is not None else None
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

        self.barlow = BarlowTwinsLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        vicreg_enabled = bool(getattr(cfg, "vicreg_enabled", False))
        self.vicreg = (
            VICRegLoss.from_config(
                cfg,
                input_dim=latent_dim,
            )
            if vicreg_enabled
            else None
        )
        self.wmse = WMSELoss.from_config(
            cfg,
            input_dim=latent_dim,
        )
        self.pointcontrast = PointContrastLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
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

    def _shared_invariant(self, z_inv_model, eq_z):
        # Contrastive training always prefers norms(eq_z) when eq_z exists and
        # otherwise falls back to the encoder invariant branch.
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

    def _split_encoder_output(self, enc_out):
        if isinstance(enc_out, (tuple, list)):
            if not enc_out:
                raise ValueError("Encoder returned empty output")
            z_inv_model = enc_out[0]
            eq_z = None
            for candidate in enc_out[1:]:
                if not (torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3):
                    continue
                # Accept canonical equivariant outputs (B, C, 3) where C matches
                # invariant width; reject auxiliary transform matrices (B, 3, 3).
                if torch.is_tensor(z_inv_model) and z_inv_model.dim() == 2:
                    if candidate.shape[1] == z_inv_model.shape[1]:
                        eq_z = candidate
                        break
                    if candidate.shape[1] == 3:
                        continue
                # Fallback for encoders that only expose equivariant latents.
                if candidate.shape[1] != 3:
                    eq_z = candidate
                    break
            return z_inv_model, eq_z
        return enc_out, None

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    @staticmethod
    def _unpack_batch(batch):
        """Unpack batch dict into points and metadata."""
        if isinstance(batch, dict):
            pc = batch["points"]
            meta = {
                "class_id": batch.get("class_id"),
                "instance_id": batch.get("instance_id"),
                "rotation": batch.get("rotation"),
            }
            return pc, meta
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        return batch[0], {}

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
        """Extract deterministic encoder features + labels for split-level SVM metrics."""
        pc_raw, meta = self._unpack_batch(batch)
        class_id = meta.get("class_id")
        if class_id is None:
            return None, None

        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)
        z_inv_model, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        # Keep supervised diagnostics aligned with contrastive objectives: use
        # the invariant latent consumed by losses (prefer eq_z -> invariant).
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
        if eq_z is not None:
            return self._shared_invariant(None, eq_z)
        if z_inv_model is not None and not self._warned_cache_eq_fallback:
            self._status_print(
                f"[contrastive/cache] eq_z is missing at stage='{stage_name}'. "
                "Falling back to encoder invariant output (z_inv_model) for cached z_inv_contrastive."
            )
            self._warned_cache_eq_fallback = True
        return self._shared_invariant(z_inv_model, None)

    def forward(self, pc: torch.Tensor):
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        z_inv_model, eq_z = self._split_encoder_output(enc_out)
        z_inv_contrastive = self._contrastive_invariant_latent(z_inv_model, eq_z)
        # Forward returns both invariant branches explicitly:
        # (z_inv_contrastive, z_inv_model, eq_z).
        return z_inv_contrastive, z_inv_model, eq_z

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, meta = self._unpack_batch(batch)
        batch_size = int(pc_raw.shape[0])
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)

        # Determine whether to cache embeddings for supervised diagnostics.
        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )

        losses = {}

        # Barlow Twins loss (self-supervised)
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
            self._log_metric(stage, name, value, batch_size=batch_size)

        # VICReg loss (self-supervised, optionally with Radial-VICReg regularization).
        if self.vicreg is not None:
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
                invariant_transform=self._shared_invariant,
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value, batch_size=batch_size)

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
                self._log_metric(stage, name, value, batch_size=batch_size)

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
                self._log_metric(stage, name, value, batch_size=batch_size)

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
            self._log_metric(
                stage,
                "loss_nonfinite",
                1.0,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
            if self._consecutive_nan_steps >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Training produced {self._consecutive_nan_steps} consecutive "
                    f"non-finite losses. Halting to prevent silent divergence."
                )
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._consecutive_nan_steps = 0

        metrics_to_log = {
            # Total weighted optimization objective.
            "loss": total_loss,
        }
        if "barlow" in losses:
            # Unweighted Barlow objective term.
            metrics_to_log["barlow"] = losses["barlow"]
        if "vicreg" in losses:
            # Unweighted VICReg objective term.
            metrics_to_log["vicreg"] = losses["vicreg"]
        if "wmse" in losses:
            # Unweighted W-MSE objective term.
            metrics_to_log["wmse"] = losses["wmse"]
        if "pointcontrast" in losses:
            # Unweighted PointContrast objective term.
            metrics_to_log["pointcontrast"] = losses["pointcontrast"]

        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(
                stage,
                name,
                value,
                prog_bar=(name in prog_bar_keys),
                batch_size=batch_size,
            )

        # Cache embeddings for supervised diagnostics.  Skip the encoder
        # forward pass entirely once the sample-count limit has been reached
        # to avoid unnecessary GPU work on remaining validation batches.
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

    def _log_metric(
        self,
        stage: str,
        name: str,
        value,
        *,
        on_step=None,
        on_epoch=None,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
        if batch_size is not None and "batch_size" not in log_kwargs:
            log_kwargs["batch_size"] = int(batch_size)
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


class PointContrastModule(BarlowTwinsModule):
    """Alias for clarity when running PointContrast-specific experiments."""


__all__ = ["BarlowTwinsModule", "PointContrastModule"]
