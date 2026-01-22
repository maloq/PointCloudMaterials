"""
Mo3ENet PyTorch Lightning Training Module.

This module wraps Mo3ENet for training with PyTorch Lightning,
supporting both multi-type point clouds (molecules) and single-type point clouds.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any, Tuple

sys.path.append(os.getcwd())
from src.models.mo3enet import (
    Mo3ENet,
    Mo3ENetSingleType,
    GMLossConfig,
    gaussian_mixture_recon_loss,
)
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.spd_utils import (
    to_float32,
    cached_sample_count,
    get_optimizers_and_scheduler,
)
from src.utils.spd_metrics import (
    compute_cluster_metrics,
    compute_embedding_quality_metrics,
)


class Mo3ENetModule(pl.LightningModule):
    """
    PyTorch Lightning module for training Mo3ENet.
    
    Supports:
    - Multi-type point clouds (molecules) with types tensor
    - Single-type point clouds (automatic dummy types)
    - Gaussian Mixture reconstruction loss with sigma annealing
    - Standard Chamfer/Sinkhorn metrics for monitoring
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        
        # Determine if we're using multi-type or single-type mode
        self.multi_type = cfg.get("multi_type", False)
        self.num_types = cfg.get("num_types", 1)
        
        # Build model
        model_kwargs = self._extract_model_kwargs(cfg)
        
        if self.multi_type or self.num_types > 1:
            self.model = Mo3ENet(num_types=self.num_types, **model_kwargs)
        else:
            self.model = Mo3ENetSingleType(**model_kwargs)
        
        # Loss configuration
        self.use_chamfer_loss = cfg.get("use_chamfer_loss", True)
        self.gm_weight = cfg.get("gm_weight", 0.0)
        self.chamfer_weight = cfg.get("chamfer_weight", 1.0)
        
        self.loss_cfg = GMLossConfig(
            sigma=cfg.get("gm_sigma", 0.3),
            type_scale=cfg.get("gm_type_scale", 0.0),
            use_self_target=cfg.get("gm_use_self_target", False),
            loss_mode=cfg.get("gm_loss_mode", "paper"),
            sphere_penalty=cfg.get("gm_sphere_penalty", 0.01),
            minw_penalty=cfg.get("gm_minw_penalty", 0.01),
            minw_frac=cfg.get("gm_minw_frac", 0.01),
            weight_temperature=cfg.get("gm_weight_temperature", 1.0),
            type_loss_weight=cfg.get("gm_type_loss_weight", 0.0),
            minw_log10_threshold=cfg.get("gm_minw_log10_threshold", 2.0),
            smooth_l1_beta=cfg.get("gm_smooth_l1_beta", 1.0),
            matching_eps=cfg.get("gm_matching_eps", 0.01),
        )
        
        # Sigma annealing schedule
        self.sigma_schedule = cfg.get("sigma_schedule", None)
        if self.sigma_schedule is not None:
            self.sigma_schedule = OmegaConf.to_container(self.sigma_schedule, resolve=True)
        
        # Caches for metrics
        self._cache = {
            "train": {"latents": [], "class_id": []},
            "val": {"latents": [], "class_id": []},
            "test": {"latents": [], "class_id": [], "reconstructions": [], "originals": []},
        }
        self.max_cache_samples = cfg.get("max_cache_samples", 8192)
        self.max_test_samples = cfg.get("max_test_samples", 1000)

    def _extract_model_kwargs(self, cfg: DictConfig) -> Dict[str, Any]:
        """Extract model kwargs from config."""
        return {
            "type_emb_dim": cfg.get("type_emb_dim", 32),
            "h_dim": cfg.get("h_dim", 256),
            "msg_dim": cfg.get("msg_dim", 128),
            "latent_k": cfg.get("latent_k", 128),
            "n_layers": cfg.get("n_layers", 4),
            "num_rbfs": cfg.get("num_rbfs", 32),
            "cutoff": cfg.get("cutoff", 14.0),
            "dropout": cfg.get("dropout", 0.05),
            "swarm_m": cfg.get("swarm_m", 512),
            "attn_dim": cfg.get("attn_dim", 128),
            "query_dim": cfg.get("query_dim", 128),
            "hidden_s": cfg.get("hidden_s", 256),
            "radius_norm": cfg.get("radius_norm", 1.0),
        }

    def _get_current_sigma(self) -> float:
        """Get current sigma based on epoch and schedule."""
        base_sigma = self.loss_cfg.sigma
        
        if self.sigma_schedule is None:
            return base_sigma
        
        schedule = self.sigma_schedule
        if not schedule.get("enable", False):
            return base_sigma
        
        start_sigma = schedule.get("start", base_sigma)
        end_sigma = schedule.get("end", 0.1)
        start_epoch = schedule.get("start_epoch", 0)
        end_epoch = schedule.get("end_epoch", 100)
        
        epoch = self.current_epoch
        if epoch < start_epoch:
            return start_sigma
        if epoch >= end_epoch:
            return end_sigma
        
        # Linear interpolation
        progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
        return start_sigma + progress * (end_sigma - start_sigma)

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Unpack batch dict into points, types, mask, and metadata.
        
        Returns:
            points: (B, N, 3)
            types: (B, N) or None for single-type
            node_mask: (B, N) or None for no padding
            meta: dict with class_id, etc.
        """
        if isinstance(batch, dict):
            pc = batch["points"]
            types = batch.get("types", None)
            node_mask = batch.get("node_mask", batch.get("mask", None))
            meta = {
                "class_id": batch.get("class_id"),
                "instance_id": batch.get("instance_id"),
            }
            return pc, types, node_mask, meta
        
        # Fallback for non-dict batches
        if not isinstance(batch, (tuple, list)):
            return batch, None, None, {}
        
        return batch[0], None, None, {}

    def forward(
        self,
        x: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through Mo3ENet.
        
        Args:
            x: Point cloud (B, N, 3)
            types: Point types (B, N), optional for single-type
            node_mask: Valid point mask (B, N), optional
            
        Returns:
            g: Latent vectors (B, K, 3)
            y: Decoded swarm coords (B, M, 3)
            type_logits: (B, M, C)
            weight_logits: (B, M)
        """
        if isinstance(self.model, Mo3ENetSingleType):
            return self.model(x)
        else:
            B, N, _ = x.shape
            if types is None:
                types = torch.zeros(B, N, dtype=torch.long, device=x.device)
            if node_mask is None:
                node_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            return self.model(x, types, node_mask)

    def _compute_losses(
        self,
        x: torch.Tensor,
        types: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor],
        g: torch.Tensor,
        y: torch.Tensor,
        type_logits: torch.Tensor,
        weight_logits: torch.Tensor,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        B, N, _ = x.shape
        device = x.device
        
        # Default types and mask if not provided
        if types is None:
            types = torch.zeros(B, N, dtype=torch.long, device=device)
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # Get normalized input (same as model does internally)
        if hasattr(self.model, 'model'):
            inner_model = self.model.model
        else:
            inner_model = self.model
        x_norm, _ = inner_model.center_and_normalize(x, node_mask, inner_model.radius_norm)
        
        losses = {}
        
        # Primary: Chamfer loss (scales well to dense point clouds)
        if self.use_chamfer_loss or self.chamfer_weight > 0:
            y_f32, x_f32 = to_float32(y, x_norm)
            chamfer_val, _ = chamfer_distance(y_f32, x_f32, point_reduction="mean")
            losses["chamfer"] = chamfer_val
        
        # Optional: GM loss (mainly for molecules with few atoms)
        if self.gm_weight > 0:
            current_sigma = self._get_current_sigma()
            loss_cfg = GMLossConfig(
                sigma=current_sigma,
                type_scale=self.loss_cfg.type_scale,
                use_self_target=self.loss_cfg.use_self_target,
                loss_mode=self.loss_cfg.loss_mode,
                sphere_penalty=self.loss_cfg.sphere_penalty,
                minw_penalty=self.loss_cfg.minw_penalty,
                minw_frac=self.loss_cfg.minw_frac,
                weight_temperature=self.loss_cfg.weight_temperature,
                type_loss_weight=self.loss_cfg.type_loss_weight,
                minw_log10_threshold=self.loss_cfg.minw_log10_threshold,
                smooth_l1_beta=self.loss_cfg.smooth_l1_beta,
                matching_eps=self.loss_cfg.matching_eps,
            )
            gm_loss, gm_stats = gaussian_mixture_recon_loss(
                x_in=x_norm,
                t_in=types,
                node_mask=node_mask,
                y_out=y,
                type_logits=type_logits,
                weight_logits=weight_logits,
                cfg=loss_cfg,
                return_stats=True,
            )
            losses["gm_loss"] = gm_loss
            for key, value in gm_stats.items():
                losses[f"gm_{key}"] = value
        
        # Monitoring metrics (no gradient)
        with torch.no_grad():
            if "chamfer" not in losses:
                y_f32, x_f32 = to_float32(y, x_norm)
                chamfer_monitor, _ = chamfer_distance(y_f32, x_f32, point_reduction="mean")
                losses["chamfer_monitor"] = chamfer_monitor
            else:
                losses["chamfer_monitor"] = losses["chamfer"]
            
            # Debug: log output stats
            y_radius = torch.linalg.norm(y, dim=-1).mean()
            losses["y_radius"] = y_radius
        
        return losses

    def _step(self, batch, batch_idx: int, stage: str) -> torch.Tensor:
        """Common step logic for train/val/test."""
        pc, types, node_mask, meta = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
        
        if types is not None:
            types = types.to(device=self.device, non_blocking=True)
        if node_mask is not None:
            node_mask = node_mask.to(device=self.device, non_blocking=True)
        
        # Forward pass
        g, y, type_logits, weight_logits = self(pc, types, node_mask)
        
        # Cache latents for metrics
        if stage in self._cache:
            self._cache_batch(stage, g, meta, y if stage == "test" else None, pc if stage == "test" else None)
        
        # Compute losses
        losses = self._compute_losses(pc, types, node_mask, g, y, type_logits, weight_logits, stage)
        
        # Total loss
        total_loss = torch.tensor(0.0, device=pc.device, dtype=pc.dtype)
        
        if "chamfer" in losses and self.chamfer_weight > 0:
            total_loss = total_loss + self.chamfer_weight * losses["chamfer"]
        
        if "gm_loss" in losses and self.gm_weight > 0:
            total_loss = total_loss + self.gm_weight * losses["gm_loss"]
        
        # Logging
        self._log_metric(stage, "loss", total_loss, prog_bar=True)
        self._log_metric(stage, "chamfer", losses["chamfer_monitor"])
        if "gm_loss" in losses:
            self._log_metric(stage, "gm_loss", losses["gm_loss"])
        if "gm_recon" in losses:
            self._log_metric(stage, "gm_recon", losses["gm_recon"])
        if "gm_sphere" in losses:
            self._log_metric(stage, "gm_sphere", losses["gm_sphere"])
        if "gm_minw" in losses:
            self._log_metric(stage, "gm_minw", losses["gm_minw"])
        if "gm_type_loss" in losses:
            self._log_metric(stage, "gm_type_loss", losses["gm_type_loss"])
        if "gm_mean_self_overlap" in losses:
            self._log_metric(stage, "gm_mean_self_overlap", losses["gm_mean_self_overlap"])
        if "gm_matching_nodes_fraction" in losses:
            self._log_metric(stage, "gm_matching_nodes_fraction", losses["gm_matching_nodes_fraction"])
        if "y_radius" in losses:
            self._log_metric(stage, "y_radius", losses["y_radius"])
        
        return total_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())

    def _log_metric(self, stage: str, name: str, value, *, on_step=None, on_epoch=None, prog_bar=False, **kwargs):
        if on_step is None:
            on_step = stage == "train"
        if on_epoch is None:
            on_epoch = stage != "train"
        log_kwargs = dict(kwargs)
        if "sync_dist" not in log_kwargs and stage != "train":
            log_kwargs["sync_dist"] = True
        self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, **log_kwargs)

    # -------------------------
    # Cache and metrics
    # -------------------------

    def _cache_limit_for_stage(self, stage: str) -> int:
        if stage == "test":
            return self.max_test_samples
        return self.max_cache_samples

    def _cache_batch(
        self,
        stage: str,
        g: torch.Tensor,
        meta: Dict,
        recon: Optional[torch.Tensor] = None,
        pc: Optional[torch.Tensor] = None,
    ):
        """Cache batch data for epoch-end metrics."""
        cache = self._cache.get(stage)
        if cache is None:
            return
        
        limit = self._cache_limit_for_stage(stage)
        cached = cached_sample_count(cache)
        remaining = max(0, limit - cached)
        if remaining <= 0:
            return
        
        batch_size = g.shape[0]
        effective_batch = min(batch_size, remaining)
        
        # Get invariant latent (norms of equivariant vectors)
        inv_latent = torch.linalg.norm(g, dim=-1)  # (B, K)
        cache["latents"].append(inv_latent[:effective_batch].detach().cpu().float())
        
        class_id = meta.get("class_id")
        if class_id is not None:
            if not torch.is_tensor(class_id):
                class_id = torch.as_tensor(class_id)
            cache["class_id"].append(class_id[:effective_batch].detach().cpu())
        
        if stage == "test":
            if recon is not None:
                cache["reconstructions"].append(recon[:effective_batch].detach().cpu().float())
            if pc is not None:
                cache["originals"].append(pc[:effective_batch].detach().cpu().float())

    def _reset_cache(self, stage: str):
        cache = self._cache.get(stage)
        if cache is None:
            return
        for key in cache:
            cache[key].clear()

    def _log_epoch_metrics(self, stage: str):
        cache = self._cache.get(stage)
        if cache is None or not cache["latents"]:
            return
        
        latents = torch.cat(cache["latents"], dim=0).numpy()
        
        if cache["class_id"]:
            labels = torch.cat(cache["class_id"], dim=0).numpy()
            
            # Clustering metrics
            metrics = compute_cluster_metrics(latents, labels, stage)
            if metrics:
                for name, value in metrics.items():
                    self._log_metric(stage, f"class/{name.lower()}", value, on_step=False, on_epoch=True)
            
            # Embedding quality metrics
            try:
                emb_metrics = compute_embedding_quality_metrics(
                    latents, labels, include_expensive=(stage == "test")
                )
                for name, value in emb_metrics.items():
                    self._log_metric(stage, f"embedding/{name}", value, on_step=False, on_epoch=True)
            except Exception as e:
                print(f"Error computing embedding metrics: {e}")

    def on_train_epoch_start(self):
        self._reset_cache("train")

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_start(self):
        self._reset_cache("val")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_start(self):
        self._reset_cache("test")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
