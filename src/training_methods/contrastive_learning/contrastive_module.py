import torch
import pytorch_lightning as pl

import sys, os
sys.path.append(os.getcwd())

from src.models.autoencoders.factory import build_model
from src.utils.pointcloud_ops import crop_to_num_points
from src.utils.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_cluster_metrics,
)
from src.utils.spd_utils import (
    cached_sample_count,
    get_optimizers_and_scheduler,
)
from src.training_methods.equivariant_autoencoder.idec import resolve_latent_dim
from src.training_methods.contrastive_learning.barlow_twins import BarlowTwinsLoss
from src.training_methods.contrastive_learning.vicreg import VICRegLoss


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
        self.barlow = BarlowTwinsLoss.from_config(cfg, input_dim=latent_dim)
        self.vicreg = VICRegLoss.from_config(cfg, input_dim=latent_dim)

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

        # Caches for embedding metrics
        self._supervised_cache = {
            "train": {"latents": [], "class_id": []},
            "val": {"latents": [], "class_id": []},
            "test": {"latents": [], "class_id": []},
        }
        self.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, 'max_supervised_samples') else 8192
        self.max_test_samples = cfg.max_test_samples if hasattr(cfg, 'max_test_samples') else 1000

    @property
    def barlow_projector(self):
        return self.barlow.projector

    @property
    def vicreg_projector(self):
        return self.vicreg.projector

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
            if len(enc_out) > 1:
                candidate = enc_out[1]
                if torch.is_tensor(candidate) and candidate.dim() == 3 and candidate.shape[-1] == 3:
                    if inv_z is not None and inv_z.dim() == 2 and candidate.shape[1] == inv_z.shape[1]:
                        eq_z = candidate
                    elif candidate.shape[1] != 3:
                        eq_z = candidate
            return inv_z, eq_z
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

        # Embeddings for metrics
        inv_z, eq_z = self._split_encoder_output(self.encoder(self._prepare_encoder_input(pc)))
        z = self._invariant_latent(inv_z, eq_z)
        if z is not None and stage in self._supervised_cache:
            self._cache_supervised_batch(stage, z, meta)

        # Barlow Twins loss (self-supervised)
        losses = {}
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

        # VICReg loss (self-supervised)
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

        total_loss = None
        if "barlow" in losses:
            total_loss = self.barlow.weight * losses["barlow"]
        if "vicreg" in losses:
            vicreg_total = self.vicreg.weight * losses["vicreg"]
            total_loss = vicreg_total if total_loss is None else total_loss + vicreg_total
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

        if not torch.isfinite(total_loss).item():
            self._log_metric(stage, "loss_nonfinite", 1.0, on_step=True, on_epoch=False)
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        metrics_to_log = {
            "loss": total_loss,
        }
        if "barlow" in losses:
            metrics_to_log["barlow"] = losses["barlow"]
        if "vicreg" in losses:
            metrics_to_log["vicreg"] = losses["vicreg"]

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
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return
        for key in cache:
            cache[key].clear()

    def _cache_limit_for_stage(self, stage: str):
        if stage == "test":
            return self.max_test_samples
        if stage in {"train", "val"}:
            return self.max_supervised_samples
        return None

    def _cache_supervised_batch(self, stage: str, z: torch.Tensor, meta: dict) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        limit = self._cache_limit_for_stage(stage)
        remaining = None
        if limit is not None and limit > 0:
            cached = cached_sample_count(cache)
            remaining = int(limit - cached)
            if remaining <= 0:
                return

        if z is None:
            return

        batch_size = int(z.shape[0])
        effective_batch = batch_size if remaining is None else min(batch_size, remaining)
        if effective_batch <= 0:
            return

        class_id = meta.get("class_id")
        if class_id is None:
            return
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.detach().view(-1)
        effective_batch = min(effective_batch, class_id.shape[0])
        if effective_batch <= 0:
            return

        cache["latents"].append(z[:effective_batch].detach().to(torch.float32).cpu())
        cache["class_id"].append(class_id[:effective_batch].cpu())

    def _log_supervised_metrics(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        if not cache["latents"] or not cache["class_id"]:
            for key in cache:
                cache[key].clear()
            return

        latents = torch.cat(cache["latents"], dim=0).numpy()
        labels = torch.cat(cache["class_id"], dim=0).numpy()

        metrics = compute_cluster_metrics(latents, labels, stage)
        if metrics:
            for name, value in metrics.items():
                self._log_metric(
                    stage,
                    f"class/{name.lower()}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        try:
            emb_metrics = compute_embedding_quality_metrics(latents, labels, include_expensive=(stage == "test"))
            for name, value in emb_metrics.items():
                self._log_metric(
                    stage,
                    f"embedding/{name}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        except Exception as e:
            print(f"Error computing embedding quality metrics: {e}")

        for key in cache:
            cache[key].clear()
