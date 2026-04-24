import torch
import pytorch_lightning as pl

import sys
import os

sys.path.append(os.getcwd())

from src.models import EncoderAdapter, build_encoder, resolve_encoder_output_dim
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.utils.model_summary import make_model_summary_point_cloud, resolve_model_summary_batch_size
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import get_optimizers_and_scheduler, cached_sample_count


class VICRegModule(pl.LightningModule):
    """
    Self-supervised VICReg training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        if bool(getattr(cfg, "vicreg_use_ri_mae_backbone", False)):
            raise ValueError(
                "vicreg_use_ri_mae_backbone is deprecated. "
                "Use encoder.name='RI_MAE_Invariant'. Contrastive training now uses a fixed norms-only invariant path."
            )

        self.encoder = build_encoder(cfg)
        # Resolve latent dim from the uncompiled module (cheap attribute read).
        latent_dim = resolve_encoder_output_dim(cfg, encoder=self.encoder)
        # Optional torch.compile wrap around the encoder (#8). Off by default:
        # at RI-MAE shapes (short sequences, small head_dim) SDPA falls back
        # to the math kernel and Inductor has little to fuse, while compile
        # adds guard overhead and re-traces on train/val mode flips. Opt in
        # via `compile_encoder: true` only when the encoder + config
        # combination is known to fuse well (e.g. no gradient checkpointing).
        self._compile_encoder = bool(getattr(cfg, "compile_encoder", False))
        self._encoder_compile_mode = str(getattr(cfg, "encoder_compile_mode", "default"))
        self._encoder_compile_fullgraph = bool(getattr(cfg, "encoder_compile_fullgraph", False))
        self._encoder_compile_dynamic = bool(getattr(cfg, "encoder_compile_dynamic", False))
        if self._compile_encoder:
            self.encoder = torch.compile(
                self.encoder,
                mode=self._encoder_compile_mode,
                fullgraph=self._encoder_compile_fullgraph,
                dynamic=self._encoder_compile_dynamic,
            )
        self.encoder_io = EncoderAdapter(self.encoder)

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
        summary_points = self.model_points if self.model_points is not None else self.sample_points
        if summary_points <= 0:
            if data_cfg is not None:
                raise ValueError(
                    "VICRegModule cannot create a PyTorch Lightning FLOP summary input because "
                    f"data.num_points={self.sample_points!r} and data.model_points={self.model_points!r}. "
                    "Set data.num_points or data.model_points to a positive point count."
                )
        else:
            self.example_input_array = {
                "pc": make_model_summary_point_cloud(
                    batch_size=resolve_model_summary_batch_size(cfg),
                    num_points=summary_points,
                ),
                "include_ssl_heads": True,
            }

        self.vicreg = VICRegLoss.from_config(
            cfg,
            input_dim=latent_dim,
        )

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))
        # Device-side NaN counter (#3): avoids a host sync on every train
        # step. We still surface the count to Python on a cheap cadence so
        # divergent runs are halted within O(stride) steps of the first NaN.
        self._nonfinite_step_flag: torch.Tensor | None = None
        self._nonfinite_check_stride = max(1, int(getattr(cfg, "nonfinite_check_stride", 8)))

    @property
    def vicreg_projector(self):
        return self.vicreg.projector

    def _shared_invariant(self, z_inv_model, eq_z):
        # Contrastive training always prefers norms(eq_z) when eq_z exists and
        # otherwise falls back to the encoder invariant branch.
        return self.vicreg._invariant(z_inv_model, eq_z)

    def _forward_ssl_heads_for_summary(self, features: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if self.vicreg.projector is None:
            return {}
        if features is None:
            raise RuntimeError(
                "Cannot profile VICReg FLOPs for the Lightning model summary because "
                "the encoder did not return invariant contrastive features."
            )
        return {"vicreg_projected": self.vicreg(features, profile_projector=True)}

    def _status_print(self, message: str) -> None:
        if getattr(self, "_trainer", None) is not None:
            self.print(message)
            return
        print(message)

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
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        # Keep supervised diagnostics aligned with contrastive objectives: use
        # the invariant latent consumed by losses (prefer eq_z -> invariant).
        features = z_inv_contrastive if z_inv_contrastive is not None else encoded.invariant
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

    def forward(self, pc: torch.Tensor, include_ssl_heads: bool = False):
        if not isinstance(include_ssl_heads, bool):
            raise TypeError(
                "include_ssl_heads must be a bool. It is intended only for PyTorch Lightning "
                f"model-summary FLOP profiling, got {type(include_ssl_heads)}."
            )
        if torch.is_tensor(pc):
            pc = self._prepare_model_input(pc)
            pc = pc.to(device=self.device, dtype=self.dtype)
        encoded = self.encoder_io.encode(pc)
        z_inv_contrastive = self._contrastive_invariant_latent(
            encoded.invariant,
            encoded.equivariant,
        )
        if include_ssl_heads:
            return (
                z_inv_contrastive,
                encoded.invariant,
                encoded.equivariant,
                self._forward_ssl_heads_for_summary(z_inv_contrastive),
            )
        # Forward returns both invariant branches explicitly:
        # (z_inv_contrastive, z_inv_model, eq_z).
        return z_inv_contrastive, encoded.invariant, encoded.equivariant

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

        vicreg_loss, vicreg_metrics = self.vicreg.compute_loss(
            pc=pc_raw,
            encoder=self.encoder,
            prepare_input=self.encoder_io.prepare_input,
            split_output=self.encoder_io.split_output,
            current_epoch=int(self.current_epoch),
            invariant_transform=self._shared_invariant,
        )
        if vicreg_loss is not None:
            losses["vicreg"] = vicreg_loss
        for name, value in vicreg_metrics.items():
            self._log_metric(stage, name, value, batch_size=batch_size)

        total_loss = None
        if "vicreg" in losses:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        # Tensor-only consecutive non-finite tracking (#3). The previous
        # implementation called `torch.isfinite(total_loss).item()` on every
        # micro-step which forces a CUDA->CPU sync per step. Here we keep
        # the counter on-device and only pull it to host every
        # `_nonfinite_check_stride` training steps. `torch.nan_to_num` is a
        # no-op on finite inputs so it is safe to apply unconditionally.
        if self._nonfinite_step_flag is None or self._nonfinite_step_flag.device != total_loss.device:
            self._nonfinite_step_flag = torch.zeros(
                (), dtype=torch.long, device=total_loss.device
            )
        nonfinite_step = (~torch.isfinite(total_loss)).to(dtype=torch.long)
        self._nonfinite_step_flag = torch.where(
            nonfinite_step.bool(),
            self._nonfinite_step_flag + 1,
            torch.zeros_like(self._nonfinite_step_flag),
        )
        self._log_metric(
            stage,
            "loss_nonfinite",
            nonfinite_step.to(dtype=torch.float32),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        if stage == "train" and ((batch_idx + 1) % self._nonfinite_check_stride == 0):
            observed = int(self._nonfinite_step_flag.item())
            self._consecutive_nan_steps = observed
            if observed >= self._max_consecutive_nan_steps:
                raise RuntimeError(
                    f"Training produced {observed} consecutive non-finite losses "
                    f"(checked every {self._nonfinite_check_stride} steps). "
                    "Halting to prevent silent divergence."
                )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        metrics_to_log = {
            # Total weighted optimization objective.
            "loss": total_loss,
        }
        if "vicreg" in losses:
            # Unweighted VICReg objective term.
            metrics_to_log["vicreg"] = losses["vicreg"]

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
                    encoded = self.encoder_io.encode(pc)
                    z_inv_contrastive = self._contrastive_invariant_from_eq_latent(
                        encoded.equivariant,
                        z_inv_model=encoded.invariant,
                        stage=stage,
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
        if "sync_dist" not in log_kwargs:
            # Skip DDP all-reduce for per-step train metrics (#7): they are
            # noisy per-rank samples and the all-reduce stalls the training
            # loop. Epoch-reduced train metrics and validation/test metrics
            # keep sync_dist=True so their aggregates are consistent across
            # ranks. The previous rule `stage != "train"` also skipped the
            # sync for `on_epoch` train metrics, which produced per-rank
            # epoch means in DDP.
            is_train_step_only = (stage == "train") and on_step and not on_epoch
            log_kwargs["sync_dist"] = not is_train_step_only
        if torch.is_tensor(value):
            value = value.detach()
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

__all__ = ["VICRegModule"]
