import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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


class IDECClusteringHead(nn.Module):
    def __init__(self, num_clusters: int, embed_dim: int, eps: float = 1e-8):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)
        self.centers = nn.Parameter(torch.empty(self.num_clusters, self.embed_dim))
        nn.init.normal_(self.centers, std=0.02)

    def soft_assign_q(self, z: torch.Tensor) -> torch.Tensor:
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)
        q = 1.0 / (1.0 + dist_sq)
        q = q / (q.sum(dim=1, keepdim=True) + self.eps)
        return q

    def target_distribution_p(self, q: torch.Tensor) -> torch.Tensor:
        f_j = q.sum(dim=0)
        numerator = (q * q) / (f_j + self.eps)
        p = numerator / (numerator.sum(dim=1, keepdim=True) + self.eps)
        return p

    def kl_pq(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_safe = p.clamp(min=self.eps)
        q_safe = q.clamp(min=self.eps)
        return torch.mean(torch.sum(p_safe * torch.log(p_safe / q_safe), dim=1))


# IDEC workflow example:
# 1) Pretrain AE with idec_enabled=False for N epochs
# 2) model.init_idec_centers_from_kmeans(trainer.datamodule.train_dataloader())
# 3) Finetune with idec_enabled=True, idec_gamma=0.1, idec_use_global_p=True, idec_update_interval=T


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
        self.log_chamfer = bool(getattr(cfg, "log_chamfer", True))
        default_log_emd = "sinkhorn" in self.loss_components
        self.log_emd = bool(getattr(cfg, "log_emd", default_log_emd))
        
        raw_loss_params = getattr(cfg, "loss_params", None)
        if raw_loss_params is not None:
            self.loss_params = OmegaConf.to_container(raw_loss_params, resolve=True)
        else:
            self.loss_params = {}

        self.kl_latent_loss_scale = cfg.kl_latent_loss_scale

        self.load_pretrained_ae = bool(getattr(cfg, "load_pretrained_ae", False))
        self.pretrained_ae_checkpoint = getattr(cfg, "pretrained_ae_checkpoint", None)

        self.idec_enabled = bool(getattr(cfg, "idec_enabled", False))
        self.idec_num_clusters = int(getattr(cfg, "idec_num_clusters", 0) or 0)
        self.idec_gamma = float(getattr(cfg, "idec_gamma", 0.1))
        self.idec_update_interval = int(getattr(cfg, "idec_update_interval", 200))
        self.idec_delta = float(getattr(cfg, "idec_delta", 0.001))
        self.idec_max_iter = getattr(cfg, "idec_max_iter", None)
        self.idec_use_global_p = bool(getattr(cfg, "idec_use_global_p", True))
        self.idec_normalize_z = bool(getattr(cfg, "idec_normalize_z", True))
        self.idec_init_kmeans = bool(getattr(cfg, "idec_init_kmeans", True))

        self.barlow_enabled = bool(getattr(cfg, "barlow_enabled", False))
        self.barlow_weight = float(getattr(cfg, "barlow_weight", 0.0))
        self.barlow_lambda = float(getattr(cfg, "barlow_lambda", 5e-3))
        self.barlow_embed_dim = int(getattr(cfg, "barlow_embed_dim", 8192))

        self._idec_embed_dim = self._resolve_idec_embed_dim(cfg)
        self.idec_head = None
        if self.idec_num_clusters > 0:
            if self._idec_embed_dim is None:
                raise ValueError("IDEC requires latent_size to set clustering embed_dim")
            self.idec_head = IDECClusteringHead(
                num_clusters=self.idec_num_clusters,
                embed_dim=self._idec_embed_dim,
            )
        elif self.idec_enabled:
            raise ValueError("idec_num_clusters must be set when idec_enabled=True")

        self._barlow_inv_dim = self._idec_embed_dim
        if self._barlow_inv_dim is None:
            if self.barlow_enabled and self.barlow_weight > 0:
                raise ValueError("Barlow Twins requires latent_size to set projector input dim")
            self.barlow_projector = None
        else:
            self.barlow_projector = nn.Sequential(
                nn.Linear(self._barlow_inv_dim, self.barlow_embed_dim, bias=False),
                nn.BatchNorm1d(self.barlow_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.barlow_embed_dim, self.barlow_embed_dim, bias=False),
                nn.BatchNorm1d(self.barlow_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.barlow_embed_dim, self.barlow_embed_dim, bias=False),
            )

        if self.load_pretrained_ae:
            self._load_pretrained_ae(self.pretrained_ae_checkpoint)

        self._idec_p_cache = None
        self._idec_p_cache_is_tensor = False
        self._idec_p_cache_ready = False
        self._idec_prev_labels = None
        self._idec_last_p_update_step = None
        self._idec_last_change_rate = None
        self._idec_should_stop = False
        self._idec_cached_train_loader = None
        self._idec_global_p_invalid = False
        
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

    @staticmethod
    def _resolve_idec_embed_dim(cfg):
        if hasattr(cfg, "latent_size"):
            return int(cfg.latent_size)
        if hasattr(cfg, "encoder") and hasattr(cfg.encoder, "kwargs"):
            latent_size = cfg.encoder.kwargs.get("latent_size", None)
            if latent_size is not None:
                return int(latent_size)
        return None

    def _load_pretrained_ae(self, checkpoint_path) -> None:
        if not checkpoint_path:
            print("Warning: load_pretrained_ae=True but no pretrained_ae_checkpoint provided")
            return
        if not os.path.exists(checkpoint_path):
            print(f"Warning: pretrained AE checkpoint not found at {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        stripped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith("model.") else key
            stripped_state_dict[new_key] = value
        missing, unexpected = self.load_state_dict(stripped_state_dict, strict=False)
        print(f"Loaded pretrained AE checkpoint: {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def _idec_active(self) -> bool:
        return bool(self.idec_enabled and self.idec_head is not None)

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

    def _barlow_augment(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Produce a label-preserving augmented view of a point cloud.
        pc: (B, N, 3)
        """
        x = pc

        jitter_std = float(getattr(self.hparams, "barlow_jitter_std", 0.01))
        drop_ratio = float(getattr(self.hparams, "barlow_drop_ratio", 0.2))

        if jitter_std > 0:
            x = x + torch.randn_like(x) * jitter_std

        if drop_ratio > 0:
            B, N, _ = x.shape
            keep = (torch.rand(B, N, device=x.device) > drop_ratio)
            keep[:, 0] = True
            w = keep.float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            idx = torch.multinomial(w, num_samples=N, replacement=True)
            x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

        return x

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError("Input must be a square matrix")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _barlow_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        z_a, z_b are embeddings after projector: (N, D)
        """
        N, D = z_a.shape
        if N < 2:
            return z_a.new_tensor(0.0)

        z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-9)
        z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-9)

        c = (z_a_norm.T @ z_b_norm) / N
        c_diff = (c - torch.eye(D, device=c.device, dtype=c.dtype)).pow(2)

        off = self._off_diagonal(c_diff)
        off.mul_(self.barlow_lambda)

        loss = torch.diagonal(c_diff).sum() + off.sum()
        return loss

    def _barlow_invariant(self, inv_z, eq_z):
        if eq_z is None and inv_z is not None and inv_z.dim() == 3 and inv_z.shape[-1] == 3:
            eq_z = inv_z
            inv_z = None
        if eq_z is not None:
            return eq_z.norm(dim=-1)
        return inv_z

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
            if self.log_chamfer:
                if self.loss_components == ["chamfer"]:
                    losses['chamfer'] = losses['recon']
                else:
                    point_reduction = self.loss_params.get("chamfer", {}).get("point_reduction", "mean")
                    losses['chamfer'], _ = chamfer_distance(recon_f32, pc_f32, point_reduction=point_reduction)
            if self.log_emd:
                if self.loss_components == ["sinkhorn"]:
                    losses['emd'] = losses['recon']
                else:
                    losses['emd'], _ = sinkhorn_distance(recon_f32.contiguous(), pc_f32, blur=sinkhorn_blur)

        # LATENT REGULARIZATION (Optional)
        if self.kl_latent_loss_scale > 0:
            losses['kl'] = kl_latent_regularizer(inv_z)

        return losses, sinkhorn_blur

    def _compute_idec_loss(self, inv_z: torch.Tensor, meta: dict):
        if not self._idec_active():
            return None, None
        if inv_z is None:
            return None, None
        z = inv_z
        if self.idec_normalize_z:
            z = F.normalize(z, dim=1)
        q = self.idec_head.soft_assign_q(z).float()
        p = None
        if self.idec_use_global_p and self._idec_p_cache_ready:
            p = self._get_idec_p_for_batch(meta.get("instance_id"))
        if p is None:
            p = self.idec_head.target_distribution_p(q).detach()
        p = p.to(device=q.device, dtype=q.dtype)
        cluster_kl = self.idec_head.kl_pq(p, q)
        q_entropy = (-q * q.clamp_min(self.idec_head.eps).log()).sum(dim=1).mean()
        return cluster_kl, q_entropy

    def _maybe_update_idec_targets(self) -> None:
        if not self._idec_active():
            return
        if not self.idec_use_global_p or self._idec_global_p_invalid:
            return
        if self.idec_update_interval <= 0:
            return
        if self._idec_should_stop:
            return
        step = int(self.global_step)
        if self._idec_last_p_update_step is None or (step - self._idec_last_p_update_step) >= self.idec_update_interval:
            change_rate = self._update_idec_p_cache()
            if change_rate is not None:
                self._log_metric("train", "idec_label_change_rate", change_rate, on_step=True, on_epoch=False)

    @torch.no_grad()
    def _update_idec_p_cache(self):
        loader = self._get_train_dataloader()
        if loader is None:
            return None
        was_training = self.training
        self.eval()
        q_batches = []
        id_batches = []
        for batch in loader:
            pc, meta = self._unpack_batch(batch)
            pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
            enc_out = self.encoder(self._prepare_encoder_input(pc))
            inv_z, _ = self._split_encoder_output(enc_out)
            if inv_z is None:
                continue
            z = inv_z
            if self.idec_normalize_z:
                z = F.normalize(z, dim=1)
            q = self.idec_head.soft_assign_q(z)
            q_batches.append(q.detach().to(torch.float32).cpu())
            ids = meta.get("instance_id")
            if ids is None:
                ids = torch.arange(z.shape[0], dtype=torch.long)
            elif not torch.is_tensor(ids):
                ids = torch.as_tensor(ids, dtype=torch.long)
            ids = ids.view(-1).detach().cpu()
            id_batches.append(ids)
        if was_training:
            self.train()
        if not q_batches:
            self._idec_p_cache_ready = False
            return None
        q_all = torch.cat(q_batches, dim=0)
        ids_all = torch.cat(id_batches, dim=0)
        if (ids_all < 0).any():
            self._idec_global_p_invalid = True
            self._idec_p_cache_ready = False
            return None
        p_all = self.idec_head.target_distribution_p(q_all).detach()
        self._set_idec_p_cache(ids_all, p_all)
        labels = torch.argmax(q_all, dim=1)
        change_rate = None
        if self._idec_prev_labels is not None and self._idec_prev_labels.numel() == labels.numel():
            change_rate = (self._idec_prev_labels != labels).float().mean().item()
            self._idec_last_change_rate = change_rate
            if self.idec_delta > 0 and change_rate < self.idec_delta:
                self._idec_should_stop = True
                if self.trainer is not None:
                    self.trainer.should_stop = True
        self._idec_prev_labels = labels
        self._idec_last_p_update_step = int(self.global_step)
        return change_rate

    def _set_idec_p_cache(self, instance_ids: torch.Tensor, p_all: torch.Tensor) -> None:
        instance_ids = instance_ids.to(torch.long).view(-1)
        p_all = p_all.to(torch.float32)
        if instance_ids.numel() == 0:
            self._idec_p_cache = None
            self._idec_p_cache_is_tensor = False
            self._idec_p_cache_ready = False
            return
        unique_ids, _ = torch.unique(instance_ids, return_counts=True)
        has_duplicates = unique_ids.numel() != instance_ids.numel()
        min_id = int(unique_ids.min().item())
        max_id = int(unique_ids.max().item())
        contiguous = min_id >= 0 and (max_id + 1 == unique_ids.numel()) and not has_duplicates
        if contiguous:
            cache = torch.zeros((max_id + 1, p_all.shape[1]), dtype=torch.float32)
            cache[instance_ids] = p_all
            self._idec_p_cache = cache
            self._idec_p_cache_is_tensor = True
            self._idec_p_cache_ready = True
            return
        sums = {}
        counts_map = {}
        for idx, key in enumerate(instance_ids.tolist()):
            key = int(key)
            if key not in sums:
                sums[key] = p_all[idx].clone()
                counts_map[key] = 1
            else:
                sums[key] += p_all[idx]
                counts_map[key] += 1
        cache = {key: sums[key] / float(counts_map[key]) for key in sums}
        self._idec_p_cache = cache
        self._idec_p_cache_is_tensor = False
        self._idec_p_cache_ready = True

    def _get_idec_p_for_batch(self, instance_ids):
        if instance_ids is None or self._idec_p_cache is None or not self._idec_p_cache_ready:
            return None
        if not torch.is_tensor(instance_ids):
            instance_ids = torch.as_tensor(instance_ids, dtype=torch.long)
        instance_ids = instance_ids.view(-1)
        if instance_ids.numel() == 0 or (instance_ids < 0).any():
            return None
        if self._idec_p_cache_is_tensor:
            ids_cpu = instance_ids.to(torch.long).cpu()
            if ids_cpu.max().item() >= self._idec_p_cache.shape[0]:
                return None
            p = self._idec_p_cache[ids_cpu]
            return p.to(device=self.device)
        p_list = []
        for key in instance_ids.tolist():
            val = self._idec_p_cache.get(int(key))
            if val is None:
                return None
            p_list.append(val)
        return torch.stack(p_list, dim=0).to(device=self.device)

    def _clone_dataloader_no_drop(self, loader):
        if not isinstance(loader, DataLoader):
            return loader
        kwargs = dict(
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=False,
            collate_fn=loader.collate_fn,
            worker_init_fn=loader.worker_init_fn,
            timeout=loader.timeout,
            generator=loader.generator,
        )
        if loader.num_workers > 0:
            kwargs["persistent_workers"] = loader.persistent_workers
            prefetch_factor = getattr(loader, "prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(loader.dataset, **kwargs)

    def _get_train_dataloader(self):
        if self._idec_cached_train_loader is not None:
            return self._idec_cached_train_loader
        if self.trainer is None:
            return None
        loader = None
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "train_dataloader"):
            try:
                loader = dm.train_dataloader()
            except Exception:
                loader = None
        if loader is None:
            loader = getattr(self.trainer, "train_dataloader", None)
        if isinstance(loader, (list, tuple)):
            loader = loader[0] if loader else None
        if loader is None:
            return None
        loader = self._clone_dataloader_no_drop(loader)
        self._idec_cached_train_loader = loader
        return loader

    def _run_kmeans(self, z: torch.Tensor, num_clusters: int, *, kmeans_kwargs=None) -> torch.Tensor:
        kmeans_kwargs = {} if kmeans_kwargs is None else dict(kmeans_kwargs)
        z_np = z.detach().cpu().numpy()
        try:
            from sklearn.cluster import KMeans

            n_init = kmeans_kwargs.pop("n_init", 10)
            random_state = kmeans_kwargs.pop("random_state", 0)
            kmeans = KMeans(
                n_clusters=num_clusters,
                n_init=n_init,
                random_state=random_state,
                **kmeans_kwargs,
            )
            kmeans.fit(z_np)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=z.dtype)
            return centers
        except Exception as exc:
            print(f"Falling back to torch k-means: {exc}")
        return self._torch_kmeans(z, num_clusters)

    def _torch_kmeans(self, z: torch.Tensor, num_clusters: int, num_iters: int = 20) -> torch.Tensor:
        if z.shape[0] < num_clusters:
            raise ValueError("Not enough samples to initialize k-means centers")
        z = z.to(torch.float32)
        indices = torch.randperm(z.shape[0], device=z.device)[:num_clusters]
        centers = z[indices].clone()
        for _ in range(num_iters):
            diff = z.unsqueeze(1) - centers.unsqueeze(0)
            dist_sq = (diff * diff).sum(dim=2)
            labels = dist_sq.argmin(dim=1)
            new_centers = centers.clone()
            for k in range(num_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = z[mask].mean(dim=0)
            if torch.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        return centers

    @torch.no_grad()
    def init_idec_centers_from_kmeans(self, train_dataloader=None, *, max_samples=None, kmeans_kwargs=None):
        if self.idec_head is None:
            raise ValueError("IDEC head is not initialized")
        loader = train_dataloader or self._get_train_dataloader()
        if loader is None:
            raise ValueError("Training dataloader is required for k-means initialization")
        was_training = self.training
        self.eval()
        z_batches = []
        total = 0
        for batch in loader:
            pc, meta = self._unpack_batch(batch)
            pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)
            enc_out = self.encoder(self._prepare_encoder_input(pc))
            inv_z, _ = self._split_encoder_output(enc_out)
            if inv_z is None:
                continue
            z = inv_z
            if self.idec_normalize_z:
                z = F.normalize(z, dim=1)
            z_batches.append(z.detach().to(torch.float32).cpu())
            total += int(z.shape[0])
            if max_samples is not None and total >= max_samples:
                break
        if was_training:
            self.train()
        if not z_batches:
            raise RuntimeError("No latents collected for k-means initialization")
        z_all = torch.cat(z_batches, dim=0)
        if max_samples is not None and z_all.shape[0] > max_samples:
            z_all = z_all[:max_samples]
        centers = self._run_kmeans(z_all, self.idec_num_clusters, kmeans_kwargs=kmeans_kwargs)
        self.idec_head.centers.data.copy_(centers.to(self.idec_head.centers.device))
        return centers

    def _step(self, batch, batch_idx, stage: str):
        pc, meta = self._unpack_batch(batch)
        pc = pc.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Forward pass
        inv_z, recon, eq_z = self(pc)

        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, meta, recon, pc)

        # Compute losses
        losses, sinkhorn_blur = self._compute_losses(recon, pc, inv_z)
        if stage == "train" and self._idec_active():
            cluster_kl, q_entropy = self._compute_idec_loss(inv_z, meta)
            if cluster_kl is not None:
                losses['cluster_kl'] = cluster_kl
            if q_entropy is not None:
                losses['idec_q_entropy'] = q_entropy
        if stage == "train" and self.barlow_enabled and self.barlow_weight > 0 and self.barlow_projector is not None:
            y_a = self._barlow_augment(pc)
            y_b = self._barlow_augment(pc)

            enc_a = self.encoder(self._prepare_encoder_input(y_a))
            inv_a, eq_a = self._split_encoder_output(enc_a)
            inv_a = self._barlow_invariant(inv_a, eq_a)

            enc_b = self.encoder(self._prepare_encoder_input(y_b))
            inv_b, eq_b = self._split_encoder_output(enc_b)
            inv_b = self._barlow_invariant(inv_b, eq_b)

            if inv_a is not None and inv_b is not None:
                z_a = self.barlow_projector(inv_a.float())
                z_b = self.barlow_projector(inv_b.float())
                barlow = self._barlow_loss(z_a.float(), z_b.float())
                losses['barlow'] = barlow

        # Build total loss
        total_loss = losses['recon']
        if 'kl' in losses:
            total_loss += self.kl_latent_loss_scale * losses['kl']
        if 'cluster_kl' in losses:
            total_loss += self.idec_gamma * losses['cluster_kl']
        if 'barlow' in losses:
            total_loss += self.barlow_weight * losses['barlow']
        
        total_loss = total_loss.to(self.dtype)

        # Logging
        metrics_to_log = {
            'loss': total_loss,
            'recon_loss': losses['recon'],
        }
        if 'emd' in losses:
            metrics_to_log['emd'] = losses['emd']
        if 'chamfer' in losses:
            metrics_to_log['chamfer'] = losses['chamfer']
        if 'kl' in losses:
            metrics_to_log['kl_loss'] = losses['kl']
        if 'cluster_kl' in losses:
            metrics_to_log['cluster_kl'] = losses['cluster_kl']
        if 'idec_q_entropy' in losses:
            metrics_to_log['idec_q_entropy'] = losses['idec_q_entropy']
        if 'barlow' in losses:
            metrics_to_log['barlow'] = losses['barlow']
        if sinkhorn_blur is not None:
            metrics_to_log['sinkhorn_blur'] = float(sinkhorn_blur)

        prog_bar_keys = {'loss'}
        for name, value in metrics_to_log.items():
            self._log_metric(stage, name, value, prog_bar=(name in prog_bar_keys))

        return total_loss

    
    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if self._idec_active():
            if self.idec_max_iter is not None:
                try:
                    max_iter = int(self.idec_max_iter)
                    if max_iter > 0 and int(self.global_step) >= max_iter:
                        self._idec_should_stop = True
                        if self.trainer is not None:
                            self.trainer.should_stop = True
                except (TypeError, ValueError):
                    pass
            self._maybe_update_idec_targets()
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
