import torch
import torch.nn as nn
import pytorch_lightning as pl

import sys,os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.optimizer_utils import get_optimizers_and_scheduler
from src.utils.model_utils import load_supervised_checkpoint
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss, angle_triad_loss, rdf_loss
from src.training_methods.spd.rot_heads import build_rot_head, kabsch_rotation
from src.training_methods.spd.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_canonical_consistency_metrics,
    compute_reconstruction_emd_per_phase,
    test_rotation_equivariance_sample,
    test_reconstruction_consistency_sample,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata


# Supported reconstruction loss components (can be combined via '+')
LOSS_COMPONENTS = (
    "sinkhorn",
    "chamfer",
    "pairwise_sorted",
    "pairwise_hist",
    "angle_sorted",
    "angle_sorted_cos",
    "angle_hist",
    "angle_hist_cos",
    "rdf_pdf",
    "rdf_gr",
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
        self._use_rot_head = self.rotation_mode in {"sixd_head", "matrix_head"}
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
        self._sinkhorn_blur_schedule = self._init_sinkhorn_blur_schedule(cfg)
        raw_loss_params = getattr(cfg, "loss_params", None)
        if raw_loss_params is not None:
            self.loss_params = OmegaConf.to_container(raw_loss_params, resolve=True)
        else:
            self.loss_params = {}

        self.ortho_scale = cfg.ortho_scale
        self.kl_latent_loss_scale = cfg.kl_latent_loss_scale
        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": []},
            "test": {"latents": [], "phase": [], "reconstructions": [], "canonicals": [], "rotations": [], "originals": []},
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

    def _init_sinkhorn_blur_schedule(self, cfg):
        schedule_cfg = cfg.sinkhorn_blur_schedule
        start = float(schedule_cfg.start)
        end = float(schedule_cfg.end)
        start_epoch = int(schedule_cfg.start_epoch)
        duration = int(schedule_cfg.duration_epochs)
        enabled = bool(schedule_cfg.enable)
        return {
            "enabled": enabled,
            "start": start,
            "end": end,
            "start_epoch": max(0, start_epoch),
            "duration_epochs": duration,
        }

    def _current_sinkhorn_blur(self) -> float:
        schedule = getattr(self, "_sinkhorn_blur_schedule", None)
        if not schedule:
            return 0.02

        start = float(schedule["start"])
        if not schedule.get("enabled", False):
            return start

        duration = int(schedule["duration_epochs"])
        if duration <= 1:
            return float(schedule["end"])

        epoch = max(0, int(self.current_epoch))
        start_epoch = int(schedule["start_epoch"])
        elapsed = max(0, epoch - start_epoch)
        max_elapsed = duration - 1
        if elapsed >= max_elapsed:
            return float(schedule["end"])

        alpha = elapsed / max_elapsed
        return float(start + alpha * (schedule["end"] - start))

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
        if component.startswith("angle_"):
            mode = "hist" if "hist" in component else "sorted"
            k = self._loss_param("angle", "k", 8)
            bins = self._loss_param("angle", "bins", 72)
            sigma = self._loss_param("angle", "sigma", None)
            base_graph = self._loss_param("angle", "base_graph", "x")
            reduction = self._loss_param("angle", "reduction", "mean")
            use_cos_param = self._loss_param("angle", "use_cos", None)
            if use_cos_param is None:
                use_cos = component.endswith("_cos")
            else:
                use_cos = bool(use_cos_param)
            return angle_triad_loss(
                pred,
                target,
                k=k,
                mode=mode,
                bins=bins,
                sigma=sigma,
                base_graph=base_graph,
                use_cos=use_cos,
                reduction=reduction,
            )
        if component in {"rdf_pdf", "rdf_gr"}:
            bins = self._loss_param("rdf", "bins", 64)
            r_max = self._loss_param("rdf", "r_max", None)
            sigma = self._loss_param("rdf", "sigma", None)
            volume = self._loss_param("rdf", "volume", None)
            normalize_mode = "gr" if component.endswith("gr") else "pdf"
            reduction = self._loss_param("rdf", "reduction", "mean")
            return rdf_loss(
                pred,
                target,
                bins=bins,
                r_max=r_max,
                sigma=sigma,
                volume=volume,
                normalize_mode=normalize_mode,
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
            checkpoint_path = self._find_best_supervised_checkpoint(cfg)
            if checkpoint_path is None:
                print("Warning: No supervised checkpoint path specified and auto-discovery failed")
                return

        # Delegate to utility function
        load_supervised_checkpoint(checkpoint_path, self.encoder, self.rot_net)

    def _find_best_supervised_checkpoint(self, cfg):
        """Auto-discover best supervised checkpoint from lightning_logs directory."""
        # Try to find checkpoints directory
        base_dirs = ['lightning_logs', 'outputs/supervised_encoder']

        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                continue

            # Find all checkpoint files
            checkpoint_files = []
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.ckpt'):
                        checkpoint_files.append(os.path.join(root, file))

            if not checkpoint_files:
                continue

            # Sort by modification time and return the most recent
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            print(f"Auto-discovered checkpoint: {checkpoint_files[0]}")
            return checkpoint_files[0]

        return None

    def forward(self, pc: torch.Tensor):
        inv_z, eq_z, _ = self.encoder(pc)
        cano, rot, recon = self._decode_with_rotation(inv_z, eq_z, pc)
        return inv_z, recon, cano, rot

    def _decode_with_rotation(self, inv_z, eq_z, pc):
        """Decode and compute rotation based on configured mode."""
        if self.rotation_mode == "eq_decoder":
            cano = self.decoder(eq_z)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_no_rot":
            cano = self.decoder(inv_z)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_kabsch":
            cano = self.decoder(inv_z)
            rot = kabsch_rotation(cano, pc)
            recon = self._apply_rotation(cano, rot)
        else:  # rot head modes
            cano = self.decoder(inv_z)
            rot = self.rot_net(eq_z)
            recon = self._apply_rotation(cano, rot)
        return cano, rot, recon

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, legacy: bool = True, **kwargs) -> None:
        """Helper to keep WandB charts grouped by stage while preserving legacy metric keys."""
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

    @staticmethod
    def _to_f32(*tensors):
        """Convert multiple tensors to float32 at once."""
        return tuple(t.to(torch.float32) for t in tensors)

    def _compute_losses(self, recon, cano, rot, pc, inv_z):
        """Compute all losses and return (loss_dict, current_sinkhorn_blur)."""
        recon_f32, cano_f32, pc_f32 = self._to_f32(recon, cano, pc)
        sinkhorn_blur = self._current_sinkhorn_blur()

        losses = {}

        # Main reconstruction loss (configurable)
        losses['recon'], _ = self._reconstruction_loss(recon_f32, pc_f32, sinkhorn_blur)

        # Orthogonality loss for rotation matrix
        losses['ortho'] = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                      - torch.eye(3, device=self.device)) ** 2)

        # Diagnostic metrics (no grad)
        with torch.no_grad():
            losses['chamfer_after'], _ = chamfer_distance(recon_f32, pc_f32)
            losses['emd_after'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
            losses['emd_before'], _ = sinkhorn_distance(cano_f32.contiguous(), pc_f32, blur=sinkhorn_blur)
            losses['chamfer_before'], _ = chamfer_distance(cano_f32, pc_f32)

        # KL regularization (optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        return losses, sinkhorn_blur


    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Forward pass
        inv_z, recon, cano, rot = self(pc)

        # Cache for metrics if needed
        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, recon, cano, rot, pc)

        # Compute all losses
        losses, sinkhorn_blur = self._compute_losses(recon, cano, rot, pc, inv_z)

        # Build total loss
        total_loss = losses['recon'] + self.ortho_scale * losses['ortho']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        total_loss = total_loss.to(self.dtype)

        # Prepare metrics for logging
        metrics_to_log = {
            'loss': total_loss,
            f'{self.loss_name}_loss': losses['recon'],
            'emd': losses['emd_after'],
            'emd_before_rot': losses['emd_before'],
            'chamfer': losses['chamfer_after'],
            'chamfer_before_rot': losses['chamfer_before'],
            'ortho': losses['ortho'],
        }
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if sinkhorn_blur is not None:
            metrics_to_log['sinkhorn_blur'] = float(sinkhorn_blur)

        # Log all metrics
        self._log_metrics(stage, metrics_to_log, prog_bar_keys={'loss'})

        # Optional rotation geodesic when ground truth is available
        if rot is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=rot.device, dtype=torch.float32)
            geodesic = self._rotation_geodesic(rot.to(torch.float32), gt_rot)
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

    @staticmethod
    def _cached_sample_count(cache: dict) -> int:
        latents = cache.get("latents") if cache is not None else None
        if not latents:
            return 0
        return sum(t.shape[0] for t in latents)

    def _cache_supervised_batch(self, stage: str, inv_z: torch.Tensor, labels: dict,
                                 recon: torch.Tensor = None, cano: torch.Tensor = None,
                                 rot: torch.Tensor = None, pc: torch.Tensor = None) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        limit = self._cache_limit_for_stage(stage)
        remaining = None
        if limit is not None and limit > 0:
            cached = self._cached_sample_count(cache)
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
            if cano is not None:
                cache["canonicals"].append(cano[:effective_batch].detach().to(torch.float32).cpu())
            if rot is not None:
                cache["rotations"].append(rot[:effective_batch].detach().to(torch.float32).cpu())
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
        metrics = self._compute_cluster_metrics(latents, labels, stage)
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

        # Rotation equivariance and reconstruction consistency tests (test only - expensive forward passes)
        if stage == "test" and self.reference_pcs is not None and self.enable_expensive_metrics:
            try:
                # Reduced number of rotations for memory efficiency
                equivariance_metrics = test_rotation_equivariance_sample(
                    self, self.reference_pcs, labels, n_test_rotations=5, max_samples_per_phase=3
                )
                for name, value in equivariance_metrics.items():
                    self._log_metric(
                        stage,
                        f"equivariance/{name}",
                        value,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

                # Clear memory after equivariance tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error computing rotation equivariance metrics: {e}")

            try:
                # Reduced number of rotations for memory efficiency
                recon_consistency_metrics = test_reconstruction_consistency_sample(
                    self, self.reference_pcs, labels, n_rotations=5, max_samples_per_phase=2
                )
                for name, value in recon_consistency_metrics.items():
                    self._log_metric(
                        stage,
                        f"recon_consistency/{name}",
                        value,
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

                # Clear memory after reconstruction tests
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error computing reconstruction consistency metrics: {e}")

        # Clear cache
        for key in cache:
            cache[key].clear()

    @staticmethod
    def _compute_cluster_metrics(latents: np.ndarray, labels: np.ndarray, stage: str):
        metrics = {}
        unique = np.unique(labels)
        if unique.size >= 2 and latents.shape[0] >= unique.size:
            try:
                assignments = KMeans(n_clusters=unique.size, n_init=10, random_state=0).fit_predict(latents)
                metrics["ARI"] = float(adjusted_rand_score(labels, assignments))
                metrics["NMI"] = float(normalized_mutual_info_score(labels, assignments))
            except Exception:
                pass
        if stage == "val" and latents.shape[0] >= 3:
            try:
                assignments_k3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(latents)
                if np.unique(assignments_k3).size > 1:
                    metrics["Silhouette"] = float(silhouette_score(latents, assignments_k3))
            except Exception:
                pass
        return metrics or None


    @staticmethod
    def _rotation_geodesic(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Mean geodesic angle (radians) between predicted and ground-truth rotations."""
        if pred.shape != target.shape:
            raise ValueError(f"Rotation shapes must match (got {pred.shape} vs {target.shape})")
        delta = pred.transpose(-1, -2) @ target
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
        return torch.arccos(cos_theta).mean()

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
