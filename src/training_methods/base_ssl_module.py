import torch
import pytorch_lightning as pl

from src.models import EncoderAdapter, build_encoder, resolve_encoder_output_dim
from src.training_methods.contrastive_learning.supervised_cache import (
    cache_limit_for_stage,
    cache_supervised_batch,
    init_supervised_cache,
    log_supervised_metrics,
    reset_supervised_cache,
)
from src.training_methods.contrastive_learning.swav import SwAVLoss
from src.training_methods.contrastive_learning.vicreg import VICRegLoss
from src.utils.model_summary import make_model_summary_point_cloud, resolve_model_summary_batch_size
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.training_utils import cached_sample_count, get_optimizers_and_scheduler


class BaseSSLModule(pl.LightningModule):
    """Shared Lightning mechanics for point-cloud SSL modules."""

    test_metrics_on_step = False
    cache_warning_prefix = "ssl"

    def __init__(
        self,
        cfg,
        *,
        module_name: str,
        summary_sequence_length: int | None = None,
        require_summary_points: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder = build_encoder(cfg)
        latent_dim = resolve_encoder_output_dim(cfg, encoder=self.encoder)

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
        self._init_example_input(
            cfg,
            module_name=module_name,
            summary_sequence_length=summary_sequence_length,
            require_summary_points=require_summary_points,
        )

        self.vicreg = VICRegLoss.from_config(cfg, input_dim=latent_dim)
        self.swav = SwAVLoss.from_config(cfg, input_dim=latent_dim)

        init_supervised_cache(self, cfg)
        self.cache_train_supervised_metrics = bool(getattr(cfg, "cache_train_supervised_metrics", False))
        self._warned_cache_eq_fallback = False
        self._consecutive_nan_steps = 0
        self._max_consecutive_nan_steps = int(getattr(cfg, "max_consecutive_nan_steps", 20))
        self._nonfinite_step_flag: torch.Tensor | None = None
        self._nonfinite_check_stride = max(1, int(getattr(cfg, "nonfinite_check_stride", 8)))

    def _init_example_input(
        self,
        cfg,
        *,
        module_name: str,
        summary_sequence_length: int | None,
        require_summary_points: bool,
    ) -> None:
        summary_points = self.model_points if self.model_points is not None else self.sample_points
        if summary_points <= 0:
            if require_summary_points:
                raise ValueError(
                    f"{module_name} cannot create a PyTorch Lightning FLOP summary input because "
                    f"data.num_points={self.sample_points!r} and data.model_points={self.model_points!r}. "
                    "Set data.num_points or data.model_points to a positive point count."
                )
            return

        kwargs = dict(
            batch_size=resolve_model_summary_batch_size(cfg),
            num_points=summary_points,
        )
        if summary_sequence_length is not None:
            kwargs["sequence_length"] = int(summary_sequence_length)
        self.example_input_array = {
            "pc": make_model_summary_point_cloud(**kwargs),
            "include_ssl_heads": True,
        }

    @property
    def vicreg_projector(self):
        return self.vicreg.projector

    def _shared_invariant(self, z_inv_model, eq_z):
        return self.vicreg._invariant(z_inv_model, eq_z)

    def _forward_ssl_heads_for_summary(self, features: torch.Tensor | None) -> dict[str, torch.Tensor]:
        head_outputs = {}
        if self.vicreg.projector is not None:
            if features is None:
                raise RuntimeError(
                    "Cannot profile contrastive FLOPs for the Lightning model summary because "
                    "the encoder did not return invariant contrastive features."
                )
            head_outputs[f"{self.vicreg.metric_prefix}_projected"] = self.vicreg(
                features,
                profile_projector=True,
            )
        if self.swav.projector is not None or self.swav.prototypes is not None:
            if features is None:
                raise RuntimeError(
                    "Cannot profile SwAV FLOPs for the Lightning model summary because "
                    "the encoder did not return invariant contrastive features."
                )
            head_outputs["swav_logits"] = self.swav(features, profile_logits=True)
        return head_outputs

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

    @torch.no_grad()
    def _extract_supervised_features_from_batch(self, batch):
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
                f"[{self.cache_warning_prefix}/cache] eq_z is missing at stage='{stage_name}'. "
                "Falling back to encoder invariant output (z_inv_model) for cached z_inv_contrastive."
            )
            self._warned_cache_eq_fallback = True
        return self._shared_invariant(z_inv_model, None)

    def _should_cache_supervised_stage(self, stage: str) -> bool:
        return stage in self._supervised_cache and (
            stage != "train" or self.cache_train_supervised_metrics
        )

    def _cache_supervised_embeddings_if_needed(
        self,
        *,
        stage: str,
        meta: dict,
        embeddings: torch.Tensor | None,
    ) -> None:
        if embeddings is None:
            return
        limit = self._cache_limit_for_stage(stage)
        cache = self._supervised_cache.get(stage)
        already_cached = cached_sample_count(cache) if cache is not None else 0
        if limit is not None and already_cached >= limit:
            return
        self._cache_supervised_batch(
            stage,
            embeddings,
            meta,
            encoder_features=embeddings,
        )

    def _weighted_total_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = None
        contrastive_key = getattr(self.vicreg, "metric_prefix", "vicreg")
        if contrastive_key in losses:
            vicreg_total = self.vicreg.weight * losses[contrastive_key]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if "swav" in losses:
            swav_total = self.swav.weight * losses["swav"]
            total_loss = swav_total if total_loss is None else total_loss + swav_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
        return total_loss

    def _finish_ssl_step(
        self,
        *,
        stage: str,
        batch_idx: int,
        batch_size: int,
        losses: dict[str, torch.Tensor],
        print_first_eval_batch: bool = False,
    ) -> torch.Tensor:
        total_loss = self._weighted_total_loss(losses)

        if print_first_eval_batch and stage != "train" and batch_idx == 0:
            parts = [f"[{stage}-diag] epoch={self.current_epoch} batch_idx=0"]
            for key, value in losses.items():
                parts.append(f"{key}={value.item():.6f}")
            parts.append(f"total={total_loss.item():.6f}")
            parts.append(f"active_losses={list(losses.keys())}")
            self._status_print(" | ".join(parts))

        if self._nonfinite_step_flag is None or self._nonfinite_step_flag.device != total_loss.device:
            self._nonfinite_step_flag = torch.zeros((), dtype=torch.long, device=total_loss.device)
        nonfinite_step = (~torch.isfinite(total_loss)).to(dtype=torch.long)
        self._nonfinite_step_flag = torch.where(
            nonfinite_step.bool(),
            self._nonfinite_step_flag + 1,
            torch.zeros_like(self._nonfinite_step_flag),
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

        metrics_to_log = {"loss": total_loss}
        metrics_to_log.update(losses)
        for name, value in metrics_to_log.items():
            self._log_metric(
                stage,
                name,
                value,
                prog_bar=(name == "loss"),
                batch_size=batch_size,
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

    def on_after_backward(self) -> None:
        if self.swav.should_freeze_prototypes(global_step=int(self.global_step)):
            self.swav.clear_prototype_gradients()

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
            if stage == "train":
                on_step = True
            elif stage == "val":
                on_step = False
            else:
                on_step = bool(self.test_metrics_on_step)
        if on_epoch is None:
            on_epoch = stage != "train"
        if not on_step and not on_epoch:
            return

        log_kwargs = dict(kwargs)
        if batch_size is not None and "batch_size" not in log_kwargs:
            log_kwargs["batch_size"] = int(batch_size)
        if "sync_dist" not in log_kwargs:
            is_train_step_only = (stage == "train") and on_step and not on_epoch
            log_kwargs["sync_dist"] = not is_train_step_only
        if torch.is_tensor(value):
            value = value.detach()

        if name == "loss":
            callback_kwargs = dict(log_kwargs)
            callback_kwargs["logger"] = False
            self.log(f"{stage}/loss", value, on_step=on_step, on_epoch=on_epoch, **callback_kwargs)

            logger_kwargs = dict(log_kwargs)
            logger_kwargs["prog_bar"] = False
            logger_kwargs["logger"] = True
            self.log(f"loss/{stage}", value, on_step=on_step, on_epoch=on_epoch, **logger_kwargs)
            return

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


__all__ = ["BaseSSLModule"]
