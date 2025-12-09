import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys, os
import numpy as np
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance, kl_latent_regularizer
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
    """
    Equivariant Autoencoder that supports:
    1. Standard Reconstruction (Chamfer/Sinkhorn loss)
    2. Diffusion Probabilistic Models (Score matching loss)
    3. Flow-Matching generative decoders (velocity field training)
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder, self.decoder = build_model(cfg)

        # Detect if we are using the Diffusion Decoder
        # Assumes the decoder class name contains 'Diffusion' or config has a flag
        self.is_diffusion = "Diffusion" in cfg.decoder.name
        self.is_flow_matching = getattr(self.decoder, "is_flow_matching", False) or ("FlowMatching" in cfg.decoder.name)
        self.use_invariant_latent = getattr(self.decoder, "use_invariant_latent", self.is_diffusion or self.is_flow_matching)

        # Parse loss configuration
        raw_loss = cfg.loss
        if isinstance(raw_loss, (list, tuple, ListConfig)):
            loss_components = [str(part).strip().lower() for part in raw_loss if str(part).strip()]
        else:
            loss_components = [part.strip().lower() for part in str(raw_loss).split("+") if part.strip()]
        
        # If diffusion, we don't strictly require standard loss components, 
        # but we keep them for validation metric calculation.
        if not loss_components and not (self.is_diffusion or self.is_flow_matching):
            raise ValueError("At least one loss component must be specified")
            
        self.loss_components = loss_components
        self.loss_name = "+".join(loss_components)
        self._sinkhorn_blur_schedule = init_sinkhorn_blur_schedule(cfg)
        
        raw_loss_params = getattr(cfg, "loss_params", None)
        if raw_loss_params is not None:
            self.loss_params = OmegaConf.to_container(raw_loss_params, resolve=True)
        else:
            self.loss_params = {}

        self.kl_latent_loss_scale = cfg.kl_latent_loss_scale
        
        # Caches
        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": []},
            "test": {"latents": [], "phase": [], "reconstructions": [], "canonicals": [], "originals": []},
        }
        self.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, 'max_supervised_samples') else 8192
        self.max_test_samples = cfg.max_test_samples if hasattr(cfg, 'max_test_samples') else 1000

        # Load reference point clouds if available
        self.reference_pcs = None
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_path'):
            ref_path = os.path.join(cfg.data.data_path, 'reference_point_clouds.npy')
            if os.path.exists(ref_path):
                self.reference_pcs = np.load(ref_path, allow_pickle=True).item()

    def _get_num_points(self):
        """Try to resolve num_points from config."""
        if hasattr(self.hparams, 'num_points'):
            return self.hparams.num_points
        if hasattr(self.hparams, 'data') and hasattr(self.hparams.data, 'num_points'):
            return self.hparams.data.num_points
        if hasattr(self.hparams, 'decoder') and hasattr(self.hparams.decoder, 'kwargs'):
            return self.hparams.decoder.kwargs.get('num_points', 0)
        return 0

    def _component_reconstruction_loss(self, component, pred, target, sinkhorn_blur):
        if component == "sinkhorn":
            val, _ = sinkhorn_distance(pred.contiguous(), target, blur=sinkhorn_blur)
            return val
        if component == "chamfer":
            point_reduction = self.loss_params.get("chamfer", {}).get("point_reduction", "mean")
            val, _ = chamfer_distance(pred, target, point_reduction=point_reduction)
            
            auto_scale = self.loss_params.get("chamfer", {}).get("auto_scale_by_points", False)
            if auto_scale:
                num_points = self._get_num_points()
                if num_points > 0:
                    val = val / float(num_points)
                    
            return val
        if component == "pdist":
            # Assuming _compute_pdist exists in utils or parent, otherwise implement or import
            val = self._compute_pdist(pred, target)
            return val * getattr(self, 'pdist_scale', 1.0)
        raise ValueError(f"Unsupported reconstruction loss component: {component}")

    def _reconstruction_loss(self, pred, target, sinkhorn_blur):
        total_loss = None
        for component in self.loss_components:
            comp_val = self._component_reconstruction_loss(component, pred, target, sinkhorn_blur)
            total_loss = comp_val if total_loss is None else (total_loss + comp_val)
        return total_loss, None

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        pc = batch[0]
        labels = {}
        # Safely unpack labels if they exist
        if len(batch) > 1: labels["phase"] = batch[1]
        if len(batch) > 2: labels["grain"] = batch[2]
        if len(batch) > 3: labels["orientation"] = batch[3]
        if len(batch) > 4: labels["quaternion"] = batch[4]
        return pc, labels

    def forward(self, pc: torch.Tensor):
        """
        Forward pass modified for Diffusion.
        
        Args:
            pc: Input point cloud (B, N, 3)
            
        Returns:
            inv_z: Invariant latent
            recon: Reconstructed PC (or dummy if training diffusion)
            eq_z: Equivariant latent
            diff_loss: Diffusion MSE loss (scalar) or 0.0 if not diffusion
        """
        inv_z, eq_z, _ = self.encoder(pc)
        decoder_input = inv_z if self.use_invariant_latent else eq_z
        
        if self.is_diffusion or self.is_flow_matching:
            # Generative decoders expect invariant latent + gt during training
            recon, diff_loss, _ = self.decoder(decoder_input, gt_pts=pc)
            diff_loss = torch.as_tensor(diff_loss, device=pc.device, dtype=pc.dtype)
        else:
            # Standard Decoder
            recon = self.decoder(decoder_input)
            if isinstance(recon, tuple):
                recon = recon[0]
            diff_loss = torch.tensor(0.0, device=pc.device, dtype=pc.dtype)
            
        return inv_z, recon, eq_z, diff_loss

    def _compute_losses(self, recon, pc, inv_z, diff_loss, stage):
        """
        Compute all losses.
        Logic branches based on whether we are using Diffusion or Standard AE.
        """
        sinkhorn_blur = get_current_sinkhorn_blur(self._sinkhorn_blur_schedule, self.current_epoch)
        losses = {}
        zero_tensor = pc.new_zeros(())

        # 1. PRIMARY LOSS
        if self.is_diffusion or self.is_flow_matching:
            if stage == 'train':
                # Generative decoders provide their own training loss
                losses['recon'] = diff_loss
                # We SKIP geometric metrics (Chamfer/EMD) during training because:
                # a) 'recon' during training is not a clean reconstruction (it's noise or gt)
                # b) Sampling clean reconstructions is too slow for the training loop
                losses['chamfer'] = zero_tensor
                losses['emd'] = zero_tensor
            else:
                # Validation/Test: We have generated samples in 'recon'
                # We calculate geometric metrics for monitoring
                losses['recon'] = diff_loss
                recon_f32, pc_f32 = to_float32(recon, pc)
                
                with torch.no_grad():
                    point_reduction = self.loss_params.get("chamfer", {}).get("point_reduction", "mean")
                    losses['chamfer'], _ = chamfer_distance(recon_f32, pc_f32, squared=False, point_reduction=point_reduction)
                    losses['emd'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
        else:
            # Standard Autoencoder Logic
            recon_f32, pc_f32 = to_float32(recon, pc)
            losses['recon'], _ = self._reconstruction_loss(recon_f32, pc_f32, sinkhorn_blur)
            
            with torch.no_grad():
                point_reduction = self.loss_params.get("chamfer", {}).get("point_reduction", "mean")
                losses['chamfer'], _ = chamfer_distance(recon_f32, pc_f32, squared=False, point_reduction=point_reduction)
                losses['emd'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)

        # 2. LATENT REGULARIZATION (Optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        return losses, sinkhorn_blur

    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Forward pass (now returns diffusion loss if applicable)
        inv_z, recon, eq_z, diff_loss = self(pc)

        # Cache logic:
        # If generative decoder training -> 'recon' is not valid for caching (it's noise/gt)
        # If Val/Test -> 'recon' is a sample, valid for caching
        skip_cache_recon = (self.is_diffusion or self.is_flow_matching) and stage == 'train'
        valid_recon_for_cache = None if skip_cache_recon else recon
        
        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, valid_recon_for_cache, pc)

        # Compute losses
        losses, sinkhorn_blur = self._compute_losses(recon, pc, inv_z, diff_loss, stage)

        # Build total loss
        total_loss = losses['recon']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        
        total_loss = total_loss.to(self.dtype)

        # Logging
        metrics_to_log = {
            'loss': total_loss,
            'recon_loss': losses['recon'],
            'emd': losses.get('emd', 0.0),
            'chamfer': losses.get('chamfer', 0.0),
        }
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if sinkhorn_blur is not None:
            metrics_to_log['sinkhorn_blur'] = float(sinkhorn_blur)

        prog_bar_keys = {'loss'}
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
        if on_step is None: on_step = stage == "train"
        if on_epoch is None: on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

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
