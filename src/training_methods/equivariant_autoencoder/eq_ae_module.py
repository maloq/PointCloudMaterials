import torch
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
)


class EquivariantAutoencoder(pl.LightningModule):
    """
    Equivariant Autoencoder with standard reconstruction (Chamfer/Sinkhorn loss).
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder, self.decoder = build_model(cfg)
        self.use_invariant_latent = bool(getattr(self.decoder, "use_invariant_latent", True))

        # Parse loss configuration
        raw_loss = cfg.loss
        if isinstance(raw_loss, (list, tuple, ListConfig)):
            loss_components = [str(part).strip().lower() for part in raw_loss if str(part).strip()]
        else:
            loss_components = [part.strip().lower() for part in str(raw_loss).split("+") if part.strip()]
        
        if not loss_components:
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
        
        # Caches (using standardized field names)
        self._supervised_cache = {
            "train": {"latents": [], "class_id": []},
            "val": {"latents": [], "class_id": []},
            "test": {"latents": [], "class_id": [], "reconstructions": [], "canonicals": [], "originals": []},
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

    def _loss_param(self, section: str, key: str, default):
        return self.loss_params.get(section, {}).get(key, default)

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

    def _component_reconstruction_loss(self, component, pred, target, sinkhorn_blur):
        if component == "sinkhorn":
            val, _ = sinkhorn_distance(pred.contiguous(), target, blur=sinkhorn_blur)
            return val
        if component == "chamfer":
            point_reduction = self._loss_param("chamfer", "point_reduction", "mean")
            val, _ = chamfer_distance(pred, target, point_reduction=point_reduction)
            
            auto_scale = self._loss_param("chamfer", "auto_scale_by_points", False)
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
        """Unpack batch dict into points and metadata.
        
        Args:
            batch: Dict with keys "points", "class_id", "instance_id", "rotation"
            
        Returns:
            Tuple of (points, meta_dict)
        """
        if isinstance(batch, dict):
            pc = batch["points"]
            meta = {
                "class_id": batch.get("class_id"),
                "instance_id": batch.get("instance_id"),
                "rotation": batch.get("rotation"),
            }
            return pc, meta
        # Fallback for non-dict batches (shouldn't happen with new datasets)
        if not isinstance(batch, (tuple, list)):
            return batch, {}
        return batch[0], {}

    def forward(self, pc: torch.Tensor):
        """
        Forward pass.
        
        Args:
            pc: Input point cloud (B, N, 3)
            
        Returns:
            inv_z: Invariant latent
            recon: Reconstructed point cloud
            eq_z: Equivariant latent
        """
        enc_out = self.encoder(self._prepare_encoder_input(pc))
        inv_z, eq_z = self._split_encoder_output(enc_out)
        decoder_input = inv_z if self.use_invariant_latent else eq_z
        if decoder_input is None:
            latent_kind = "invariant" if self.use_invariant_latent else "equivariant"
            raise ValueError(f"Decoder expects {latent_kind} latent, but encoder did not return it.")
        
        recon = self.decoder(decoder_input)
        if isinstance(recon, tuple):
            recon = recon[0]
            
        return inv_z, recon, eq_z

    def _compute_losses(self, recon, pc, inv_z):
        """
        Compute all losses.
        """
        sinkhorn_blur = get_current_sinkhorn_blur(self._sinkhorn_blur_schedule, self.current_epoch)
        losses = {}

        recon_f32, pc_f32 = to_float32(recon, pc)
        losses['recon'], _ = self._reconstruction_loss(recon_f32, pc_f32, sinkhorn_blur)
        
        with torch.no_grad():
            point_reduction = self.loss_params.get("chamfer", {}).get("point_reduction", "mean")
            losses['chamfer'], _ = chamfer_distance(recon_f32, pc_f32, point_reduction=point_reduction)
            losses['emd'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)

        # LATENT REGULARIZATION (Optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        return losses, sinkhorn_blur

    def _step(self, batch, batch_idx, stage: str):
        pc, meta = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Forward pass
        inv_z, recon, eq_z = self(pc)

        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, meta, recon, pc)

        # Compute losses
        losses, sinkhorn_blur = self._compute_losses(recon, pc, inv_z)

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

    def _cache_supervised_batch(self, stage: str, inv_z: torch.Tensor, meta: dict,
                                 recon: torch.Tensor = None, pc: torch.Tensor = None) -> None:
        """Cache batch data for computing metrics at epoch end.
        
        Args:
            stage: "train", "val", or "test"
            inv_z: Invariant latent representations
            meta: Metadata dict with "class_id", "instance_id", "rotation"
            recon: Reconstructed point clouds (optional)
            pc: Original point clouds (optional)
        """
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

        class_id = meta.get("class_id")
        if class_id is None:
            return
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)
        class_id = class_id.detach().view(-1)
        effective_batch = min(effective_batch, class_id.shape[0])
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

        # Clustering metrics (using "class" prefix instead of "phase")
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

        # Reconstruction EMD per class
        if stage == "test" and cache["originals"] and cache["reconstructions"] and len(cache["originals"]) > 0:
            try:
                originals = torch.cat(cache["originals"], dim=0).numpy()
                reconstructions = torch.cat(cache["reconstructions"], dim=0).numpy()
                emd_metrics = compute_reconstruction_emd_per_phase(originals, reconstructions, labels)
                for name, value in emd_metrics.items():
                    self._log_metric(
                        stage,
                        f"class/{name}",
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
                print(f"Error computing reconstruction EMD per class: {e}")

        # Clear cache
        for key in cache:
            cache[key].clear()
