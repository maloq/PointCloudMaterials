import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys, os
import numpy as np
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance, kl_latent_regularizer
from src.loss.pdist_loss import pairwise_distance_loss
from src.utils.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_canonical_consistency_metrics,
    compute_reconstruction_emd_per_phase,
    compute_cluster_metrics,
)
from src.utils.spd_utils import (
    to_float32,
    cached_sample_count,
    init_sinkhorn_blur_schedule,
    get_current_sinkhorn_blur,
    get_optimizers_and_scheduler,
)
from omegaconf import DictConfig, ListConfig, OmegaConf


# Supported reconstruction loss components (can be combined via '+')
LOSS_COMPONENTS = (
    "sinkhorn",
    "chamfer",
    "pairwise_sorted",
    "pairwise_hist",
)


class EquivariantAutoencoder(pl.LightningModule):
    """Equivariant Autoencoder that decodes from equivariant latent Z_eq while still producing invariant Z_inv."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder, self.decoder = build_model(cfg)

        # Parse loss configuration
        raw_loss = cfg.loss
        if isinstance(raw_loss, (list, tuple, ListConfig)):
            loss_components = [str(part).strip().lower() for part in raw_loss if str(part).strip()]
        else:
            loss_components = [part.strip().lower() for part in str(raw_loss).split("+") if part.strip()]
        if not loss_components:
            raise ValueError("At least one loss component must be specified")
        unknown = [name for name in loss_components if name not in LOSS_COMPONENTS]
        if unknown:
            raise ValueError(
                f"Unknown loss component(s): {unknown}. Available components: {list(LOSS_COMPONENTS)}"
            )
        self.loss_components = loss_components
        self.loss_name = "+".join(loss_components)
        self._sinkhorn_blur_schedule = init_sinkhorn_blur_schedule(cfg)
        raw_loss_params = getattr(cfg, "loss_params", None)
        if raw_loss_params is not None:
            self.loss_params = OmegaConf.to_container(raw_loss_params, resolve=True)
        else:
            self.loss_params = {}

        self.kl_latent_loss_scale = cfg.kl_latent_loss_scale
        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": []},
            "test": {"latents": [], "phase": [], "reconstructions": [], "canonicals": [], "originals": []},
        }

        # Maximum samples to use for metrics caches (to limit memory usage)
        self.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, 'max_supervised_samples') else 8192
        self.max_test_samples = cfg.max_test_samples if hasattr(cfg, 'max_test_samples') else 1000

        # Load reference point clouds for metrics
        self.reference_pcs = None
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_path'):
            ref_path = os.path.join(cfg.data.data_path, 'reference_point_clouds.npy')
            if os.path.exists(ref_path):
                self.reference_pcs = np.load(ref_path, allow_pickle=True).item()
        elif hasattr(cfg, 'synthetic') and hasattr(cfg.synthetic, 'data_dir'):
            ref_path = os.path.join(cfg.synthetic.data_dir, 'reference_point_clouds.npy')
            if os.path.exists(ref_path):
                self.reference_pcs = np.load(ref_path, allow_pickle=True).item()

    def _component_reconstruction_loss(self, component, pred, target, sinkhorn_blur):
        if component == "sinkhorn":
            val, _ = sinkhorn_distance(pred.contiguous(), target, blur=sinkhorn_blur)
            return val
        if component == "chamfer":
            val, _ = chamfer_distance(pred, target)
            return val
        if component in {"pairwise_sorted", "pairwise_hist"}:
            mode = "hist" if component.endswith("hist") else "sorted"
            hist_bins = self._loss_param("pairwise", "hist_bins", 64)
            hist_sigma = self._loss_param("pairwise", "hist_sigma", None)
            reduction = self._loss_param("pairwise", "reduction", "mean")
            return pairwise_distance_loss(
                pred,
                target,
                mode=mode,
                hist_bins=hist_bins,
                hist_sigma=hist_sigma,
                reduction=reduction,
            )

        raise ValueError(f"Unsupported reconstruction loss component: {component}")

    def _reconstruction_loss(self, pred, target, sinkhorn_blur):
        total_loss = None
        for component in self.loss_components:
            comp_val = self._component_reconstruction_loss(component, pred, target, sinkhorn_blur)
            total_loss = comp_val if total_loss is None else (total_loss + comp_val)
        if total_loss is None:
            raise RuntimeError("No reconstruction loss components were applied")
        return total_loss, None

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        pc = batch[0]
        labels = {}
        labels["phase"] = batch[1]
        labels["grain"] = batch[2]
        labels["orientation"] = batch[3]
        labels["quaternion"] = batch[4]
        return pc, labels

    def forward(self, pc: torch.Tensor):
        """
        Forward pass: encode to both inv_z and eq_z, but decode only from eq_z.

        Returns:
            inv_z: Invariant latent (still produced but not used for reconstruction)
            recon: Reconstruction from eq_z
            eq_z: Equivariant latent
        """
        inv_z, eq_z, _ = self.encoder(pc)
        recon = self.decoder(eq_z)
        return inv_z, recon, eq_z

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, **kwargs) -> None:
        """Helper to keep WandB charts grouped by stage."""
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"

        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True

        log_name = f"{stage}/{name}"
        self.log(log_name, value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    def _log_metrics(self, stage: str, metrics: dict, prog_bar_keys=None):
        """Log multiple metrics at once."""
        prog_bar_keys = prog_bar_keys or set()
        for name, value in metrics.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys))

    def _loss_param(self, section: str, key: str, default=None):
        params = getattr(self, "loss_params", None)
        if not isinstance(params, dict):
            return default
        if section:
            section_params = params.get(section)
            if isinstance(section_params, dict) and key in section_params:
                return section_params[key]
        return params.get(key, default)

    def _compute_losses(self, recon, pc, inv_z):
        """Compute all losses and return (loss_dict, current_sinkhorn_blur)."""
        recon_f32, pc_f32 = to_float32(recon, pc)
        sinkhorn_blur = get_current_sinkhorn_blur(self._sinkhorn_blur_schedule, self.current_epoch)

        losses = {}

        # Main reconstruction loss (configurable)
        losses['recon'], _ = self._reconstruction_loss(recon_f32, pc_f32, sinkhorn_blur)

        # Diagnostic metrics (no grad)
        with torch.no_grad():
            losses['chamfer'], _ = chamfer_distance(recon_f32, pc_f32)
            losses['emd'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)

        # KL regularization (optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        return losses, sinkhorn_blur

    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Forward pass
        inv_z, recon, eq_z = self(pc)

        # Cache for metrics if needed
        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, recon, pc)

        # Compute all losses
        losses, sinkhorn_blur = self._compute_losses(recon, pc, inv_z)

        # Build total loss
        total_loss = losses['recon']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        total_loss = total_loss.to(self.dtype)

        # Prepare metrics for logging
        metrics_to_log = {
            'loss': total_loss,
            f'{self.loss_name}_loss': losses['recon'],
            'emd': losses['emd'],
            'chamfer': losses['chamfer'],
        }
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if sinkhorn_blur is not None:
            metrics_to_log['sinkhorn_blur'] = float(sinkhorn_blur)

        # Log all metrics
        self._log_metrics(stage, metrics_to_log, prog_bar_keys={'loss'})

        return total_loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def _handle_epoch_boundary(self, stage: str, is_start: bool):
        """Unified handler for epoch boundaries."""
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

    def _cache_supervised_batch(self, stage: str, inv_z: torch.Tensor, labels: dict,
                                 recon: torch.Tensor = None, pc: torch.Tensor = None) -> None:
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

        batch_size = int(inv_z.shape[0])
        effective_batch = batch_size if remaining is None else min(batch_size, remaining)
        if effective_batch <= 0:
            return

        phase = labels.get("phase")
        if phase is None:
            return
        if not torch.is_tensor(phase):
            phase = torch.as_tensor(phase)
        phase = phase.detach().view(-1)
        effective_batch = min(effective_batch, phase.shape[0])
        if effective_batch <= 0:
            return

        cache["latents"].append(inv_z[:effective_batch].detach().to(torch.float32).cpu())

        # Only cache full point cloud data for test stage to save memory
        if stage == "test":
            if recon is not None:
                cache["reconstructions"].append(recon[:effective_batch].detach().to(torch.float32).cpu())
            # Store reconstruction as "canonical" for compatibility with metrics
            if recon is not None:
                cache["canonicals"].append(recon[:effective_batch].detach().to(torch.float32).cpu())
            if pc is not None:
                cache["originals"].append(pc[:effective_batch].detach().to(torch.float32).cpu())

        cache["phase"].append(phase[:effective_batch].cpu())

    def _log_supervised_metrics(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        if not cache["latents"] or not cache["phase"]:
            for key in cache:
                cache[key].clear()
            return

        latents = torch.cat(cache["latents"], dim=0).numpy()
        labels = torch.cat(cache["phase"], dim=0).numpy()

        # Original clustering metrics
        metrics = compute_cluster_metrics(latents, labels, stage)
        if metrics:
            for name, value in metrics.items():
                self._log_metric(
                    stage,
                    f"phase/{name.lower()}",
                    value,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Embedding quality metrics (lightweight ones for all stages, expensive ones for test only)
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

        # Canonical consistency metrics (test only - expensive pairwise comparisons)
        if stage == "test" and cache["canonicals"] and len(cache["canonicals"]) > 0:
            try:
                canonicals = torch.cat(cache["canonicals"], dim=0).numpy()
                max_samples_consistency = min(1000, len(canonicals))
                if len(canonicals) > max_samples_consistency:
                    indices = np.random.choice(len(canonicals), max_samples_consistency, replace=False)
                    canonicals_subsample = canonicals[indices]
                    latents_subsample = latents[indices]
                    labels_subsample = labels[indices]
                else:
                    canonicals_subsample = canonicals
                    latents_subsample = latents
                    labels_subsample = labels

                consistency_metrics = compute_canonical_consistency_metrics(
                    canonicals_subsample, latents_subsample, labels_subsample
                )
                for name, value in consistency_metrics.items():
                    self._log_metric(
                        stage,
                        f"consistency/{name}",
                        value,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

                # Clear memory
                del canonicals, canonicals_subsample, latents_subsample, labels_subsample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error computing canonical consistency metrics: {e}")

        # Reconstruction EMD per phase
        if stage == "test" and cache["originals"] and cache["reconstructions"] and len(cache["originals"]) > 0:
            try:
                originals = torch.cat(cache["originals"], dim=0).numpy()
                reconstructions = torch.cat(cache["reconstructions"], dim=0).numpy()
                emd_metrics = compute_reconstruction_emd_per_phase(originals, reconstructions, labels)
                for name, value in emd_metrics.items():
                    self._log_metric(
                        stage,
                        f"phase/{name}",
                        value,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

                # Clear memory
                del originals, reconstructions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error computing reconstruction EMD per phase: {e}")

        # Clear cache
        for key in cache:
            cache[key].clear()
