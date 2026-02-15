import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.pose.head import PoseRotationHead
from src.training_methods.contrastive_learning.pose.utils import (
    cfg_get,
    compute_pose_loss,
    init_pose_components,
    prepare_eq_latent,
    rotation_geodesic_angles,
)
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.training_methods.equivariant_autoencoder.idec import resolve_latent_dim
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_utils import get_optimizers_and_scheduler


class BarlowTwinsModule(pl.LightningModule):
    """
    Self-supervised Barlow Twins / VICReg training for point cloud encoders.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Build encoder (decoder is not used for contrastive training)
        self.encoder, _ = build_model(cfg)

        latent_dim = resolve_latent_dim(cfg)
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

        init_pose_components(self, cfg, latent_dim)
        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))

    @property
    def barlow_projector(self):
        return self.barlow.projector

    @property
    def vicreg_projector(self):
        return self.vicreg.projector if self.vicreg is not None else None

    @staticmethod
    def _cfg_get(obj, name: str, default=None):
        return cfg_get(obj, name, default)

    @staticmethod
    def _prepare_eq_latent(eq_z: torch.Tensor | None) -> torch.Tensor | None:
        return prepare_eq_latent(eq_z)

    @staticmethod
    def _rotation_geodesic_angles(
        pred: torch.Tensor, target: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rotation_geodesic_angles(pred, target, eps)

    def _prepare_encoder_input(self, pc: torch.Tensor) -> torch.Tensor:
        if getattr(self.encoder, "expects_channel_first", False):
            return pc.permute(0, 2, 1).contiguous()
        return pc

    def _split_encoder_output(self, enc_out):
        if isinstance(enc_out, (tuple, list)):
            if not enc_out:
                raise ValueError("Encoder returned empty output")
            inv_z = enc_out[0]
            eq_z = None
            for candidate in enc_out[1:]:
                if not (torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3):
                    continue
                # Accept canonical equivariant outputs (B, C, 3) where C matches
                # invariant width; reject auxiliary transform matrices (B, 3, 3).
                if torch.is_tensor(inv_z) and inv_z.dim() == 2:
                    if candidate.shape[1] == inv_z.shape[1]:
                        eq_z = candidate
                        break
                    if candidate.shape[1] == 3:
                        continue
                # Fallback for encoders that only expose equivariant latents.
                if candidate.shape[1] != 3:
                    eq_z = candidate
                    break
            return inv_z, eq_z
        return enc_out, None

    def _prepare_model_input(self, pc: torch.Tensor) -> torch.Tensor:
        out = pc
        if self.model_points is not None:
            out = crop_to_num_points(out, self.model_points)
        return out

    def _compute_pose_loss(self, pc: torch.Tensor, batch_idx: int, stage: str):
        return compute_pose_loss(self, pc, batch_idx, stage)

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

    def _invariant_latent(self, inv_z, eq_z):
        return self.barlow._invariant(inv_z, eq_z)

    def forward(self, pc: torch.Tensor):
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        inv_z, eq_z = self._split_encoder_output(enc_out)
        z = self._invariant_latent(inv_z, eq_z)
        return z, inv_z, eq_z

    def _step(self, batch, batch_idx, stage: str):
        pc_raw, meta = self._unpack_batch(batch)
        pc_raw = pc_raw.to(device=self.device, dtype=self.dtype, non_blocking=True)
        pc = self._prepare_model_input(pc_raw)

        # Cache embeddings for supervised diagnostics only when requested.
        should_cache_stage = stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )
        if should_cache_stage:
            with torch.no_grad():
                inv_z, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
                z = self._invariant_latent(inv_z, eq_z)
            if z is not None:
                self._cache_supervised_batch(stage, z, meta)

        losses = {}

        # Barlow Twins loss (self-supervised)
        barlow_loss, barlow_metrics = self.barlow.compute_loss(
            pc=pc_raw,
            encoder=self.encoder,
            prepare_input=self._prepare_encoder_input,
            split_output=self._split_encoder_output,
            current_epoch=int(self.current_epoch),
        )
        if barlow_loss is not None:
            losses["barlow"] = barlow_loss
        for name, value in barlow_metrics.items():
            self._log_metric(stage, name, value)

        # VICReg loss (self-supervised, optionally with Radial-VICReg regularization).
        if self.vicreg is not None:
            vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
                pc=pc_raw,
                encoder=self.encoder,
                prepare_input=self._prepare_encoder_input,
                split_output=self._split_encoder_output,
                current_epoch=int(self.current_epoch),
            )
            if vicreg_loss is not None:
                losses["vicreg"] = vicreg_loss
            for name, value in vicreg_metrics.items():
                self._log_metric(stage, name, value)

        # Pose loss (relative-rotation NLL + optional equivariance consistency)
        pose_loss, pose_metrics = self._compute_pose_loss(pc, batch_idx, stage)
        if pose_loss is not None:
            losses["pose"] = pose_loss
            for name, value in pose_metrics.items():
                self._log_metric(stage, name, value)

        total_loss = None
        if "barlow" in losses:
            total_loss = self.barlow.weight * losses["barlow"]
        if "vicreg" in losses and self.vicreg is not None:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "pose" in losses:
            pose_total = self.pose_weight * losses["pose"]
            total_loss = pose_total if total_loss is None else total_loss + pose_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if not torch.isfinite(total_loss).item():
            # loss_nonfinite: indicator that the aggregated loss produced NaN/Inf.
            self._log_metric(stage, "loss_nonfinite", 1.0, on_step=True, on_epoch=False)
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

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
        if "pose" in losses:
            # Unweighted pose-regression objective term.
            metrics_to_log["pose"] = losses["pose"]

        prog_bar_keys = {"loss"}
        for name, value in metrics_to_log.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys))

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

    def _get_wandb_experiment(self):
        logger_obj = getattr(self, "logger", None)
        if logger_obj is None:
            return None

        if hasattr(logger_obj, "experiment"):
            exp = logger_obj.experiment
            if exp is not None and hasattr(exp, "log"):
                return exp

        loggers = getattr(logger_obj, "loggers", None)
        if loggers is not None:
            for lg in loggers:
                exp = getattr(lg, "experiment", None)
                if exp is not None and hasattr(exp, "log"):
                    return exp
        return None

    def _log_pose_ambiguity_histogram(
        self,
        stage: str,
        ambiguity: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        if ambiguity is None:
            return
        if int(getattr(self, "global_rank", 0)) != 0:
            return

        every = int(self.pose_histogram_every_n_steps)
        if stage == "train":
            if every <= 0 or (int(self.global_step) % every) != 0:
                return
        else:
            if int(batch_idx) != 0:
                return

        exp = self._get_wandb_experiment()
        if exp is None:
            return

        try:
            import wandb
        except Exception:
            return

        with torch.no_grad():
            amb_np = ambiguity.detach().to(dtype=torch.float32).cpu().numpy()

        payload = {
            f"{stage}/pose_orientation_ambiguity_hist": wandb.Histogram(amb_np),
        }

        exp.log(payload, step=int(self.global_step))

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

    def _cache_supervised_batch(self, stage: str, z: torch.Tensor, meta: dict) -> None:
        cache_supervised_batch(self, stage, z, meta)

    def _log_supervised_metrics(self, stage: str) -> None:
        log_supervised_metrics(self, stage)


__all__ = ["BarlowTwinsModule", "PoseRotationHead"]
