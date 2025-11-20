import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys,os
import numpy as np
import wandb
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.model_utils import load_supervised_checkpoint, find_best_supervised_checkpoint
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss, angle_triad_loss, rdf_loss
from src.training_methods.spd.rot_heads import build_rot_head, kabsch_rotation
from src.utils.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_canonical_consistency_metrics,
    compute_reconstruction_emd_per_phase,
    test_rotation_equivariance_sample,
    test_reconstruction_consistency_sample,
    compute_cluster_metrics,
    compute_symmetry_aware_rot_metric,
    compute_global_aligned_rot_metric,
)
from src.utils.spd_utils import (
    to_float32,
    rotation_geodesic,
    cached_sample_count,
    init_sinkhorn_blur_schedule,
    get_current_sinkhorn_blur,
    get_optimizers_and_scheduler,
    order_points_for_kabsch,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata


# Supported reconstruction loss components (can be combined via '+')
LOSS_COMPONENTS = (
    "sinkhorn",
    "chamfer",
    "pairwise_sorted",
    "pairwise_hist",
    # "angle_sorted",
    # "angle_sorted_cos",
    # "angle_hist",
    # "angle_hist_cos",
    # "rdf_pdf",
    # "rdf_gr",
)


class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder, self.decoder = build_model(cfg)

        encoder_kwargs = self.hparams.encoder.kwargs if hasattr(self.hparams.encoder, 'kwargs') else {}
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)

        self.rotation_mode = str(cfg.rotation_mode).lower()
        self._use_rot_head = self.rotation_mode in {"sixd_head", "matrix_head", "vn_rotation_head"}
        self.rot_net = build_rot_head(cfg, in_features=self.encoder_latent_size * 3) if self._use_rot_head else None

        # Load pretrained supervised encoder if specified
        if cfg.load_supervised_checkpoint:
            self._load_supervised_checkpoint(cfg)

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

        self.ortho_scale = cfg.ortho_scale
        self.kl_latent_loss_scale = cfg.kl_latent_loss_scale
        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": [], "rotations": [], "gt_rotations": []},
            "test": {"latents": [], "phase": [], "reconstructions": [], "canonicals": [], "rotations": [], "originals": [], "gt_rotations": []},
        }

        # Maximum samples to use for metrics caches (to limit memory usage) - optional with defaults
        self.max_supervised_samples = cfg.max_supervised_samples if hasattr(cfg, 'max_supervised_samples') else 8192
        self.max_test_samples = cfg.max_test_samples if hasattr(cfg, 'max_test_samples') else 1000
        # Enable/disable expensive metrics (equivariance, reconstruction consistency) - optional with default
        self.enable_expensive_metrics = cfg.enable_expensive_metrics if hasattr(cfg, 'enable_expensive_metrics') else True

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

        # Augmentation parameters
        self.aug_noise_scale = 0.0
        self.aug_rotation_scale = 0.0
        if hasattr(cfg, 'augmentation'):
            self.aug_noise_scale = cfg.augmentation.noise_scale
            self.aug_rotation_scale = cfg.augmentation.rotation_scale

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

    def _load_supervised_checkpoint(self, cfg):
        """Load pretrained supervised encoder and rotation network from checkpoint."""
        checkpoint_path = cfg.supervised_checkpoint_path if hasattr(cfg, 'supervised_checkpoint_path') else None

        if checkpoint_path is None:
            # Auto-discover the best checkpoint from lightning_logs
            checkpoint_path = find_best_supervised_checkpoint(cfg)
            if checkpoint_path is None:
                print("Warning: No supervised checkpoint path specified and auto-discovery failed")
                return

        # Delegate to utility function
        load_supervised_checkpoint(checkpoint_path, self.encoder, self.rot_net)

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

    @staticmethod
    def _apply_rotation(points: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """Apply rotation matrices to batched point clouds."""
        return (rot @ points.transpose(1, 2)).transpose(1, 2).contiguous()

    @staticmethod
    def _identity_rotation(batch_size: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(3, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def _unpack_decoder_output(self, output):
        """Helper to handle decoders that return auxiliary info (like VQ loss)."""
        if isinstance(output, tuple):
            # Assuming (pts, vq_loss, ...)
            pts = output[0]
            aux_loss = output[1] if len(output) > 1 else 0.0
            return pts, aux_loss
        return output, 0.0

    def forward(self, pc: torch.Tensor):
        inv_z, eq_z, _ = self.encoder(pc)
        cano, rot, recon, vq_loss = self._decode_with_rotation(inv_z, eq_z, pc)
        return inv_z, recon, cano, rot, vq_loss

    def _decode_with_rotation(self, inv_z, eq_z, pc):
        """Decode and compute rotation based on configured mode."""
        vq_loss = 0.0
        
        if self.rotation_mode == "eq_decoder":
            print("Not standart use of SPD. Z_inv is not used.")
            out = self.decoder(eq_z)
            cano, vq_loss = self._unpack_decoder_output(out)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_no_rot":
            out = self.decoder(inv_z)
            cano, vq_loss = self._unpack_decoder_output(out)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_kabsch":
            out = self.decoder(inv_z)
            cano, vq_loss = self._unpack_decoder_output(out)
            cano_ordered = order_points_for_kabsch(cano)
            pc_ordered = order_points_for_kabsch(pc)
            rot = kabsch_rotation(cano_ordered, pc_ordered)
            recon = self._apply_rotation(cano, rot)
        else:  # rot head modes
            out = self.decoder(inv_z)
            cano, vq_loss = self._unpack_decoder_output(out)
            rot = self.rot_net(eq_z)
            recon = self._apply_rotation(cano, rot)
        return cano, rot, recon, vq_loss

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, legacy: bool = True, batch_size=None, **kwargs) -> None:
        """Helper to keep WandB charts grouped by stage while preserving legacy metric keys."""
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"

        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs:
            if stage != "train" or on_epoch:
                log_kwargs["sync_dist"] = True
        
        if batch_size is not None:
            log_kwargs["batch_size"] = batch_size

        log_name = f"{stage}/{name}"
        self.log(log_name, value, on_step=on_step, on_epoch=on_epoch, **log_kwargs)

    def _log_metrics(self, stage: str, metrics: dict, prog_bar_keys=None, batch_size=None):
        """Log multiple metrics at once."""
        prog_bar_keys = prog_bar_keys or set()
        for name, value in metrics.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys), batch_size=batch_size)

    def _loss_param(self, section: str, key: str, default=None):
        params = getattr(self, "loss_params", None)
        if not isinstance(params, dict):
            return default
        if section:
            section_params = params.get(section)
            if isinstance(section_params, dict) and key in section_params:
                return section_params[key]
        return params.get(key, default)

    def _compute_losses(self, recon, cano, rot, pc, inv_z, vq_loss=0.0, labels=None):
        """Compute all losses and return (loss_dict, current_sinkhorn_blur)."""
        recon_f32, cano_f32, pc_f32 = to_float32(recon, cano, pc)
        sinkhorn_blur = get_current_sinkhorn_blur(self._sinkhorn_blur_schedule, self.current_epoch)

        losses = {}

        # Main reconstruction loss (configurable)
        losses['recon'], _ = self._reconstruction_loss(recon_f32, pc_f32, sinkhorn_blur)

        # Orthogonality loss for rotation matrix
        losses['ortho'] = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                      - torch.eye(3, device=self.device)) ** 2)

        # Existing diagnostic metrics
        losses['chamfer_after'], _ = chamfer_distance(recon_f32, pc_f32)
        losses['emd_after'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
        
        # "Before rotation" metrics
        # 1. Pred Canonical vs Input (Rotated) - previously 'emd_before'
        losses['emd_pred_canonical_vs_input'], _ = sinkhorn_distance(cano_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
        losses['chamfer_pred_canonical_vs_input'], _ = chamfer_distance(cano_f32, pc_f32)

        if labels is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=self.device, dtype=torch.float32)
            # pc_unrotated_gt = R_gt^T @ pc (The original object in canonical frame)
            pc_unrotated_gt = (gt_rot.transpose(1, 2) @ pc_f32.transpose(1, 2)).transpose(1, 2).contiguous()
            
            # 2. Pred Canonical vs GT Canonical (Original Unrotated) - Requested "distance between original and canonical"
            losses['emd_pred_canonical_vs_gt_canonical'], _ = sinkhorn_distance(cano_f32.contiguous(), pc_unrotated_gt, blur=sinkhorn_blur)
            losses['chamfer_pred_canonical_vs_gt_canonical'], _ = chamfer_distance(cano_f32, pc_unrotated_gt)
            
            # 3. Rotation Correctness: dist(R_pred^T @ pc, GT_Canonical)
            # Checks if predicted rotation correctly maps input back to initial position
            if rot is not None:
                pc_derotated = (rot.transpose(1, 2).float() @ pc_f32.transpose(1, 2)).transpose(1, 2).contiguous()
                losses['rot_correctness'], _ = chamfer_distance(pc_derotated, pc_unrotated_gt)

        # KL regularization (optional)
        # KL regularization (optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        # VQ Loss
        if isinstance(vq_loss, torch.Tensor):
            losses['vq'] = vq_loss
        elif vq_loss > 0:
            losses['vq'] = torch.tensor(vq_loss, device=self.device, dtype=self.dtype)

        return losses, sinkhorn_blur


    def _apply_augmentations(self, pc, labels):
        """Apply noise and rotation augmentations."""
        # Noise
        if self.aug_noise_scale > 0:
            noise = torch.randn_like(pc) * self.aug_noise_scale
            pc = pc + noise
        
        # Rotation
        if self.aug_rotation_scale > 0:
            # Generate random rotations
            B = pc.shape[0]
            # Simple random rotation generation in torch
            # Sample random unit vectors for axis and random angle
            # Or just use QR decomposition of random normal matrix
            rand_mat = torch.randn(B, 3, 3, device=pc.device, dtype=pc.dtype)
            q, r = torch.linalg.qr(rand_mat)
            d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
            q *= d.unsqueeze(-1)
            # Ensure det is 1 (rotation, not reflection)
            det = torch.det(q)
            mask = det < 0
            if mask.any():
                q[mask, :, 0] *= -1
            
            rot_aug = q
            
            # Apply rotation
            pc = (rot_aug @ pc.transpose(1, 2)).transpose(1, 2).contiguous()
            
            # Update ground truth orientation
            if labels.get("orientation") is not None:
                # New orientation = R_aug @ Old_orientation
                labels["orientation"] = rot_aug @ labels["orientation"]
                
        return pc, labels

    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        if stage == "train":
            pc, labels = self._apply_augmentations(pc, labels)

        # Forward pass
        inv_z, recon, cano, rot, vq_loss = self(pc)

        # Cache for metrics if needed
        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, recon, cano, rot, pc)

        # Compute all losses
        losses, sinkhorn_blur = self._compute_losses(recon, cano, rot, pc, inv_z, vq_loss, labels=labels)

        # Build total loss
        total_loss = losses['recon'] + self.ortho_scale * losses['ortho']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        if 'vq' in losses:
            total_loss += losses['vq']
        total_loss = total_loss.to(self.dtype)

        # Prepare metrics for logging
        metrics_to_log = {
            'loss': total_loss,
            f'{self.loss_name}_loss': losses['recon'],
            'emd': losses['emd_after'],
            'emd_before_rot': losses['emd_pred_canonical_vs_input'],
            'chamfer': losses['chamfer_after'],
            'chamfer_before_rot': losses['chamfer_pred_canonical_vs_input'],
            'ortho': losses['ortho'],
        }
        if 'emd_pred_canonical_vs_gt_canonical' in losses:
            metrics_to_log['emd_canonical_gt'] = losses['emd_pred_canonical_vs_gt_canonical']
        if 'chamfer_pred_canonical_vs_gt_canonical' in losses:
            metrics_to_log['chamfer_canonical_gt'] = losses['chamfer_pred_canonical_vs_gt_canonical']
        if 'rot_correctness' in losses:
            metrics_to_log['rot_correctness'] = losses['rot_correctness']
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if 'vq' in losses:
            metrics_to_log['vq_loss'] = losses['vq']
        if sinkhorn_blur is not None:
            metrics_to_log['sinkhorn_blur'] = float(sinkhorn_blur)

        # Log all metrics
        self._log_metrics(stage, metrics_to_log, prog_bar_keys={'loss'}, batch_size=pc.shape[0])

        # Optional rotation geodesic when ground truth is available
        if rot is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=rot.device, dtype=torch.float32)
            geodesic = rotation_geodesic(rot.to(torch.float32), gt_rot)
            self._log_metric(stage, "rot_geodesic_deg", geodesic * (180.0 / torch.pi), prog_bar=False)

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
            if stage == "val":
                self._log_validation_metrics()
            elif stage == "test":
                self._log_test_metrics()
            # Train metrics are usually logged per step or handled differently, 
            # but if we wanted epoch-level train metrics, we could add them here.
            if stage == "train":
                 self._log_validation_metrics(stage="train")


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
                                 recon: torch.Tensor = None, cano: torch.Tensor = None,
                                 rot: torch.Tensor = None, pc: torch.Tensor = None) -> None:
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

        # Cache rotations for val and test
        if stage in ["val", "test"]:
            if rot is not None:
                cache["rotations"].append(rot[:effective_batch].detach().to(torch.float32).cpu())
            if labels.get("orientation") is not None:
                cache["gt_rotations"].append(labels["orientation"][:effective_batch].detach().to(torch.float32).cpu())

        # Only cache full point cloud data for test stage to save memory
        if stage == "test":
            if recon is not None:
                cache["reconstructions"].append(recon[:effective_batch].detach().to(torch.float32).cpu())
            if cano is not None:
                cache["canonicals"].append(cano[:effective_batch].detach().to(torch.float32).cpu())
            if pc is not None:
                cache["originals"].append(pc[:effective_batch].detach().to(torch.float32).cpu())

        cache["phase"].append(phase[:effective_batch].cpu())

    def _log_validation_metrics(self, stage="val") -> None:
        """
        Compute and log lightweight metrics for validation/train.
        Uses subsampling to ensure speed.
        """
        cache = self._supervised_cache.get(stage)
        if not cache or not cache["latents"] or not cache["phase"]:
            return

        latents = torch.cat(cache["latents"], dim=0).numpy()
        labels = torch.cat(cache["phase"], dim=0).numpy()

        # Subsample for expensive metrics if dataset is large
        MAX_VAL_SAMPLES = 2048
        if len(latents) > MAX_VAL_SAMPLES:
            indices = np.random.choice(len(latents), MAX_VAL_SAMPLES, replace=False)
            latents_sub = latents[indices]
            labels_sub = labels[indices]
        else:
            latents_sub = latents
            labels_sub = labels

        # Clustering metrics
        metrics = compute_cluster_metrics(latents_sub, labels_sub, stage)
        if metrics:
            for name, value in metrics.items():
                self._log_metric(stage, f"phase/{name.lower()}", value, on_step=False, on_epoch=True)

        # Embedding quality metrics (lightweight only) - skip for train stage
        if stage != "train":
            try:
                emb_metrics = compute_embedding_quality_metrics(latents_sub, labels_sub, include_expensive=False)
                for name, value in emb_metrics.items():
                    if stage == "val":
                        self._log_metric("val_embeddings", name, value, on_step=False, on_epoch=True)
                    else:
                        self._log_metric(stage, f"embedding/{name}", value, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"Error computing embedding quality metrics: {e}")

        # Symmetry-aware rotational metrics (if rotations available)
        if cache.get("rotations") and cache.get("gt_rotations"):
            try:
                pred_rots = torch.cat(cache["rotations"], dim=0).numpy()
                gt_rots = torch.cat(cache["gt_rotations"], dim=0).numpy()
                
                # Subsample if needed
                if len(pred_rots) > MAX_VAL_SAMPLES:
                    indices = np.random.choice(len(pred_rots), MAX_VAL_SAMPLES, replace=False)
                    pred_rots = pred_rots[indices]
                    gt_rots = gt_rots[indices]
                    labels_rot = labels[indices]
                else:
                    labels_rot = labels

                # Symmetry-aware metric (assuming phases 0 and 1 are cubic crystals)
                sym_metrics = compute_symmetry_aware_rot_metric(
                    pred_rots, gt_rots, labels_rot, symmetry_phases=[0, 1]
                )
                # Log average error
                avg_error = np.mean(list(sym_metrics.values()))
                self._log_metric(stage, "rot_sym/avg_error", avg_error, on_step=False, on_epoch=True)
                
            except Exception as e:
                print(f"Error computing validation rotation metrics: {e}")

        # Clear cache to free memory
        self._reset_supervised_cache(stage)

    def _log_per_phase_metrics(self, stage: str, metric_name: str, phase_metrics: dict) -> None:
        """Log per-phase metrics as scalars."""
        for key, value in phase_metrics.items():
            # key is typically something like "metric_phase_0" or just "phase_0" depending on metric
            # We want to log as stage/metric_name/phase_X
            
            # Try to extract phase ID
            try:
                if "phase_" in key:
                    phase_id = key.split("phase_")[-1]
                    log_name = f"{metric_name}/phase_{phase_id}"
                else:
                    # Fallback if no phase in key (shouldn't happen for these metrics but safe fallback)
                    log_name = f"{metric_name}/{key}"
                
                self._log_metric(stage, log_name, value, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"Error logging metric {key}: {e}")

    def _log_test_metrics(self) -> None:
        """
        Compute and log comprehensive metrics for the test stage.
        """
        stage = "test"
        cache = self._supervised_cache.get(stage)
        if not cache or not cache["latents"] or not cache["phase"]:
            return

        latents = torch.cat(cache["latents"], dim=0).numpy()
        labels = torch.cat(cache["phase"], dim=0).numpy()

        # 1. Clustering & Embedding Quality (Full Test Set)
        metrics = compute_cluster_metrics(latents, labels, stage)
        if metrics:
            for name, value in metrics.items():
                self._log_metric(stage, f"phase/{name.lower()}", value, on_step=False, on_epoch=True)

        emb_metrics = compute_embedding_quality_metrics(latents, labels, include_expensive=True)
        for name, value in emb_metrics.items():
            self._log_metric(stage, f"embedding/{name}", value, on_step=False, on_epoch=True)

        # 2. Canonical Consistency
        if cache["canonicals"]:
            canonicals = torch.cat(cache["canonicals"], dim=0).numpy()
            # Subsample for pairwise consistency to avoid O(N^2) explosion
            MAX_CONSISTENCY_SAMPLES = 1000
            if len(canonicals) > MAX_CONSISTENCY_SAMPLES:
                indices = np.random.choice(len(canonicals), MAX_CONSISTENCY_SAMPLES, replace=False)
                canonicals_sub = canonicals[indices]
                latents_sub = latents[indices]
                labels_sub = labels[indices]
            else:
                canonicals_sub = canonicals
                latents_sub = latents
                labels_sub = labels

            consistency_metrics = compute_canonical_consistency_metrics(
                canonicals_sub, latents_sub, labels_sub
            )
            # Log scalar metrics for variance
            self._log_per_phase_metrics(stage, "canonical_pose_variance", 
                                  {k: v for k, v in consistency_metrics.items() if "variance" in k})

        # 3. Reconstruction EMD per Phase
        if cache["originals"] and cache["reconstructions"]:
            originals = torch.cat(cache["originals"], dim=0).numpy()
            reconstructions = torch.cat(cache["reconstructions"], dim=0).numpy()
            emd_metrics = compute_reconstruction_emd_per_phase(originals, reconstructions, labels)
            self._log_per_phase_metrics(stage, "reconstruction_emd", emd_metrics)

        # 4. Rotational Metrics (Global & Symmetry-Aware)
        if cache["rotations"] and cache["gt_rotations"]:
            pred_rots = torch.cat(cache["rotations"], dim=0).numpy()
            gt_rots = torch.cat(cache["gt_rotations"], dim=0).numpy()

            # Global alignment
            global_metrics = compute_global_aligned_rot_metric(pred_rots, gt_rots, labels)
            self._log_per_phase_metrics(stage, "rot_global_error", global_metrics)

            # Symmetry-aware (phases 0 and 1 are cubic)
            sym_metrics = compute_symmetry_aware_rot_metric(
                pred_rots, gt_rots, labels, symmetry_phases=[0, 1]
            )
            self._log_per_phase_metrics(stage, "rot_sym_error", sym_metrics)

        # 5. Expensive Forward-Pass Tests (Equivariance & Consistency)
        if self.reference_pcs is not None:
            self._run_expensive_test_metrics(stage, labels)

        # Clear cache
        self._reset_supervised_cache(stage)

    def _run_expensive_test_metrics(self, stage, labels):
        """Helper for running expensive forward-pass based metrics."""
        # Rotation Equivariance
        equivariance_metrics = test_rotation_equivariance_sample(
            self, self.reference_pcs, labels, n_test_rotations=5, max_samples_per_phase=3
        )
        for name, value in equivariance_metrics.items():
            self._log_metric(stage, f"equivariance/{name}", value, on_step=False, on_epoch=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reconstruction Consistency
        recon_consistency_metrics = test_reconstruction_consistency_sample(
            self, self.reference_pcs, labels, n_rotations=5, max_samples_per_phase=2
        )
        # Log scalar metrics for mean consistency
        self._log_per_phase_metrics(stage, "recon_consistency_mean", 
                              {k: v for k, v in recon_consistency_metrics.items() if "mean" in k})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
