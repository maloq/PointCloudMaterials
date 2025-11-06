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
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss
from src.training_methods.spd.rot_heads import build_rot_head, kabsch_rotation
from src.training_methods.spd.spd_metrics import (
    compute_embedding_quality_metrics,
    compute_canonical_consistency_metrics,
    compute_reconstruction_emd_per_phase,
    test_rotation_equivariance_sample,
    test_reconstruction_consistency_sample,
)


class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder, self.decoder = build_model(cfg)

        encoder_kwargs = self.hparams.encoder.get('kwargs', {})
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)
        rotation_mode = getattr(cfg, "rotation_mode", None)
        # if hasattr(cfg, "get"):
        #     rotation_mode = cfg.get("rotation_mode", rotation_mode)
        if rotation_mode is None or rotation_mode == "":
            raise ValueError("rotation_mode is required")

        self.rotation_mode = str(rotation_mode).lower()
        self._use_rot_head = self.rotation_mode in {"sixd_head", "matrix_head"}
        self.rot_net = build_rot_head(cfg, in_features=self.encoder_latent_size * 3) if self._use_rot_head else None

        self.ortho_scale = cfg.get("ortho_scale", 0.01)
        self.kl_latent_loss_scale = cfg.get("kl_latent_loss_scale", 0.0)
        self.pdist_loss_scale = cfg.get("pdist_loss_scale", 0.0)
        self.rotation_loss_scale = cfg.get("kabsch_rotation_loss_scale", 0.0)
        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": []},
            "test": {"latents": [], "phase": [], "reconstructions": [], "canonicals": [], "rotations": [], "originals": []},
        }

        # Test evaluation interval (run expensive metrics every N epochs)
        self.test_epoch_interval = cfg.get("test_epoch_interval", 1)
        # Maximum samples to use for test metrics (to limit memory usage)
        self.max_test_samples = cfg.get("max_test_samples", 1000)
        # Enable/disable expensive metrics (equivariance, reconstruction consistency)
        self.enable_expensive_metrics = cfg.get("enable_expensive_metrics", True)

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

    def forward(self, pc: torch.Tensor):
        inv_z, eq_z, _ = self.encoder(pc)

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
        else:
            cano = self.decoder(inv_z)
            rot = self.rot_net(eq_z)
            recon = self._apply_rotation(cano, rot)
        return inv_z, recon, cano, rot

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


    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        inv_z, recon, cano, rot = self(pc)

        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels, recon, cano, rot, pc)
        
        recon_f32 = recon.to(torch.float32)
        cano_f32  = cano.to(torch.float32)
        pc_f32    = pc.to(torch.float32)

        # Metrics after rotation (used for loss)
        emd_after, _      = sinkhorn_distance(recon_f32.contiguous(), pc_f32)
        chamfer_after, _  = chamfer_distance(recon_f32, pc_f32)

        # Metrics before rotation (diagnostics only)
        with torch.no_grad():
            emd_before, _     = sinkhorn_distance(cano_f32.contiguous(), pc_f32)
            chamfer_before, _ = chamfer_distance(cano_f32, pc_f32)

        # Preserve original variable names for compatibility
        loss_recon   = emd_after
        # loss_recon, _ = chamfer_distance(recon_f32, pc_f32)
        loss_chamfer = chamfer_after
        ortho_loss = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                 - torch.eye(3, device=self.device)) ** 2)
        
        # loss_pd = pairwise_distance_loss(pred=recon_f32,target=pc_f32)
        # Rotation supervision via Kabsch teacher (use centered targets)
        # loss_rot = rotation_geodesic_kabsch_loss(rot.to(torch.float32), cano.to(torch.float32), pc_centered_f32)

        # Total loss
        loss = loss_recon + self.ortho_scale * ortho_loss 
        loss = loss.to(self.dtype)
        if False:
            loss += float(self.pdist_loss_scale) * loss_pd
            self._log_metric(stage, "pdist_scaled", float(self.pdist_loss_scale) * loss_pd, prog_bar=False)
        # if self.rotation_loss_scale > 0:
        if False:
            loss += float(self.rotation_loss_scale) * loss_rot
            self._log_metric(stage, "rot_loss_scaled", float(self.rotation_loss_scale) * loss_rot, prog_bar=False)
        if self.kl_latent_loss_scale > 0:
            kl_loss = kl_latent_regularizer(inv_z)
            loss += self.kl_latent_loss_scale * kl_loss
            self._log_metric(stage, "kl_loss", kl_loss)

        self._log_metric(stage, "loss", loss, prog_bar=True)
        self._log_metric(stage, "chamfer", loss_chamfer, prog_bar=False)
        self._log_metric(stage, "chamfer_before_rot", chamfer_before, prog_bar=False)
        # self._log_metric(stage, "pd", loss_pd, prog_bar=False)
        # self._log_metric(stage, "rot", loss_rot)
        self._log_metric(stage, "emd", loss_recon)
        self._log_metric(stage, "emd_before_rot", emd_before, prog_bar=False)
        self._log_metric(stage, "ortho", ortho_loss)

        # Optional supervised diagnostics when synthetic labels are available
        if rot is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=rot.device, dtype=torch.float32)
            geodesic = self._rotation_geodesic(rot.to(torch.float32), gt_rot)
            self._log_metric(stage, "rot_geodesic_deg", geodesic * (180.0 / torch.pi), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._reset_supervised_cache("train")

    def on_train_epoch_end(self) -> None:
        self._log_supervised_metrics("train")
        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_supervised_cache("val")

    def on_validation_epoch_end(self) -> None:
        self._log_supervised_metrics("val")
        super().on_validation_epoch_end()

        # Trigger test evaluation every N epochs (skip during sanity check)
        if self.trainer.sanity_checking:
            return
        if (self.current_epoch + 1) % self.test_epoch_interval == 0:
            if self.trainer is not None and self.trainer.datamodule is not None:
                self.trainer.test(self, dataloaders=self.trainer.datamodule.test_dataloader(), verbose=False)

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._reset_supervised_cache("test")

    def on_test_epoch_end(self) -> None:
        self._log_supervised_metrics("test")
        super().on_test_epoch_end()

    def _reset_supervised_cache(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return
        for key in cache:
            cache[key].clear()

    def _cache_supervised_batch(self, stage: str, inv_z: torch.Tensor, labels: dict,
                                 recon: torch.Tensor = None, cano: torch.Tensor = None,
                                 rot: torch.Tensor = None, pc: torch.Tensor = None) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return

        # Check if we've reached the sample limit (for memory efficiency)
        if stage == "test" and len(cache["latents"]) > 0:
            total_samples = sum(len(batch) for batch in cache["latents"])
            if total_samples >= self.max_test_samples:
                return

        cache["latents"].append(inv_z.detach().to(torch.float32).cpu())

        # Only cache full point cloud data for test stage to save memory
        if stage == "test":
            if recon is not None:
                cache["reconstructions"].append(recon.detach().to(torch.float32).cpu())
            if cano is not None:
                cache["canonicals"].append(cano.detach().to(torch.float32).cpu())
            if rot is not None:
                cache["rotations"].append(rot.detach().to(torch.float32).cpu())
            if pc is not None:
                cache["originals"].append(pc.detach().to(torch.float32).cpu())

        phase = labels.get("phase")
        if phase is None:
            return
        if not torch.is_tensor(phase):
            phase = torch.as_tensor(phase)
        cache["phase"].append(phase.detach().view(-1).cpu())

    def _log_supervised_metrics(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if not cache or not cache["latents"] or not cache["phase"]:
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
                # Subsample for memory efficiency (max 500 samples for pairwise comparisons)
                max_samples_consistency = min(500, len(canonicals))
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
