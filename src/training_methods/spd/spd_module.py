import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.functional import normalize

import sys,os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.optimizer_utils import get_optimizers_and_scheduler
import src.models.autoencoders.encoders
import src.models.autoencoders.decoders
from .vn_models import PointNetEncoderVN, SimpleRot, ComplexRot
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss
from src.loss.neighbor_latent_loss import neighbor_pair_latent_loss
from src.training_methods.spd.rot_heads import Rot6DHead, sixd_to_so3

class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.decoder = build_model(cfg, only_decoder=True)
        
        encoder_kwargs = self.hparams.encoder.get('kwargs', {}) 
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)
        self.bypass_rot_head = cfg.get("bypass_rot_head", False)
        encoder_cfg = self.hparams.encoder
        encoder_name = encoder_cfg.get('name', 'PnE_VN') if hasattr(encoder_cfg, 'get') else 'PnE_VN'
        if encoder_name == 'PnE_VN':
            # Use the configured latent size (no unintended halving)
            self.encoder = PointNetEncoderVN(
                latent_size=self.encoder_latent_size//2,
                n_knn=20,
                hidden_dim1=encoder_kwargs.get('hidden_dim1', 256),
                hidden_dim2=encoder_kwargs.get('hidden_dim2', 1024),
                pooling=encoder_kwargs.get('pooling', 'mean'),
                feature_transform=encoder_kwargs.get('feature_transform', False),
                # blocks=encoder_kwargs.get('blocks', [2, 2, 2]),
                # bottleneck_ratio=encoder_kwargs.get('bottleneck_ratio', 0.5)
            )
        # elif encoder_kwargs.get('name', 'PnE_VN') == 'PnE_ResVN':
        #     self.encoder = PointNetEncoderResVN(
        #         latent_size=self.encoder_latent_size // 2,
        #         n_knn=32,
        #     )

        # self.rot_net = ComplexRot(self.encoder_latent_size // 6)
        if not self.bypass_rot_head:
            rot_net_cfg = getattr(self.hparams, 'rot_net', {})
            rot_net_kwargs = rot_net_cfg.get('kwargs', {}) if hasattr(rot_net_cfg, 'get') else {}
            self.rot_head_in_features = self._infer_eq_latent_dim(cfg)
            self.rot_net = Rot6DHead(
                in_features=self.rot_head_in_features,
                hidden=rot_net_kwargs.get('hidden', 256),
                use_attention=rot_net_kwargs.get('use_attention', True),
            )
        else:
            self.rot_head_in_features = None
            self.rot_net = None

        self.ortho_scale = cfg.get("ortho_scale", 0.01)
        self.kl_latent_loss_scale = cfg.get("kl_latent_loss_scale", 0.0)
        self.pdist_loss_scale = cfg.get("pdist_loss_scale", 0.0)
        self.rotation_loss_scale = cfg.get("kabsch_rotation_loss_scale", 0.0)
        # Neighbor latent smoothness configuration (applies to invariant latent inv_z)
        self.neighbor_loss_scale = cfg.get('neighbor_loss_scale', 0.0)
        self.neighbor_weight = cfg.get('neighbor_weight', 'binary')
        self.neighbor_sigma = cfg.get('neighbor_sigma', 1.0)

        self._supervised_cache = {
            "train": {"latents": [], "phase": []},
            "val": {"latents": [], "phase": []},
        }

    def forward(self, pc: torch.Tensor):
        inv_z, eq_z, _ = self.encoder(pc)

        if self.bypass_rot_head:
            eq_latent = self._eq_to_decoder_latent(eq_z)
            cano = self.decoder(eq_latent)
            if cano.shape[1] == 3:
                cano = cano.permute(0, 2, 1)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        else:
            cano = self.decoder(inv_z)
            if cano.shape[1] == 3:
                cano = cano.permute(0, 2, 1)
            rot = self.rot_net(eq_z)

            # rot = _orthogonalize(rot)

            recon = (rot @ cano.transpose(1, 2)).transpose(1, 2)
        return inv_z, recon, cano, rot 

    def _step(self, batch, batch_idx, stage: str):
        pc, labels = self._unpack_batch(batch)
        inv_z, recon, cano, rot = self(pc)

        if stage in self._supervised_cache:
            self._cache_supervised_batch(stage, inv_z, labels)
        
        recon_f32 = recon.to(torch.float32)
        pc_f32    = pc.to(torch.float32)

        loss_recon, _   = sinkhorn_distance(recon_f32.contiguous(), pc_f32)
        # loss_recon, _ = chamfer_distance(recon_f32, pc_f32)
        loss_chamfer, _ = chamfer_distance(recon_f32, pc_f32)

        ortho_loss = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                 - torch.eye(3, device=self.device)) ** 2)
        
        # loss_pd = pairwise_distance_loss(pred=recon_f32,target=pc_f32)
        # Rotation supervision via Kabsch teacher (use centered targets)
        # loss_rot = rotation_geodesic_kabsch_loss(rot.to(torch.float32), cano.to(torch.float32), pc_centered_f32)

        # Total loss
        loss = loss_recon + self.ortho_scale * ortho_loss 
        if False:
            loss += float(self.pdist_loss_scale) * loss_pd
            self.log(f"{stage}_pdist_scaled", float(self.pdist_loss_scale) * loss_pd, prog_bar=False)
        # if self.rotation_loss_scale > 0:
        if False:
            loss += float(self.rotation_loss_scale) * loss_rot
            self.log(f"{stage}_rot_loss_scaled", float(self.rotation_loss_scale) * loss_rot, prog_bar=False)
        if self.kl_latent_loss_scale > 0:
            kl_loss = kl_latent_regularizer(inv_z)
            loss += self.kl_latent_loss_scale * kl_loss
            self.log(f"{stage}_kl_loss", kl_loss)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_chamfer", loss_chamfer, prog_bar=False)
        # self.log(f"{stage}_pd", loss_pd, prog_bar=False)
        # self.log(f"{stage}_rot", loss_rot)
        self.log(f"{stage}_recon", loss_recon)
        self.log(f"{stage}_emd", loss_recon)
        self.log(f"{stage}_ortho", ortho_loss)

        # Optional supervised diagnostics when synthetic labels are available
        if rot is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=rot.device, dtype=torch.float32)
            geodesic = self._rotation_geodesic(rot.to(torch.float32), gt_rot)
            self.log(f"{stage}_rot_geodesic_deg", geodesic * (180.0 / torch.pi), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # Compatibility with PL versions that pass a combined list of batches
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and (
            batch[0] is not None or batch[1] is not None
        ):
            total_loss = None
            # Main AE/SPD branch
            if batch[0] is not None:
                main_loss = self._step(batch[0], batch_idx, "train")
                total_loss = main_loss if torch.is_tensor(main_loss) else main_loss["loss"]
            # Neighbor-pair branch
            if batch[1] is not None and self.neighbor_loss_scale > 0:
                try:
                    pts_i, pts_j, dists = batch[1]
                except Exception as e:
                    raise ValueError("Neighbor pair dataloader must return (points_i, points_j, distances)") from e
                inv_i, _, _ = self.encoder(pts_i)
                inv_j, _, _ = self.encoder(pts_j)
                loss_pairs, stats = neighbor_pair_latent_loss(
                    inv_i, inv_j, dists.to(self.device),
                    weight=self.neighbor_weight,
                    sigma=float(self.neighbor_sigma),
                )
                scaled = float(self.neighbor_loss_scale) * loss_pairs
                self.log('train_neighbor_loss', loss_pairs, prog_bar=False, on_step=True, on_epoch=True, batch_size=inv_i.shape[0])
                self.log('train_neighbor_loss_scaled', scaled, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
                self.log('train_neighbor_pairs', float(inv_i.shape[0]), on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
                for k, v in stats.items():
                    self.log(f'train_{k}', v, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
                total_loss = scaled if total_loss is None else (total_loss + scaled)
            return { 'loss': total_loss } if isinstance(total_loss, dict) else total_loss

        # Neighbor-pair branch (if DataModule provides a second dataloader index)
        if dataloader_idx == 1:
            try:
                pts_i, pts_j, dists = batch
            except Exception as e:
                raise ValueError("Neighbor pair dataloader must return (points_i, points_j, distances)") from e

            # Encode invariant latents (B, N, 3) → inv_z (B, D)
            inv_i, _, _ = self.encoder(pts_i)
            inv_j, _, _ = self.encoder(pts_j)

            loss_pairs, stats = neighbor_pair_latent_loss(
                inv_i, inv_j, dists.to(self.device),
                weight=self.neighbor_weight,
                sigma=float(self.neighbor_sigma),
            )
            scaled = float(self.neighbor_loss_scale) * loss_pairs

            # Logging (train-only branch)
            self.log('train_neighbor_loss', loss_pairs, prog_bar=True, on_step=True, on_epoch=True, batch_size=inv_i.shape[0])
            self.log('train_neighbor_loss_scaled', scaled, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
            self.log('train_neighbor_pairs', float(inv_i.shape[0]), on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
            for k, v in stats.items():
                self.log(f'train_{k}', v, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])

            return { 'loss': scaled }

        # Default SPD reconstruction branch
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

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

    def _reset_supervised_cache(self, stage: str) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return
        cache["latents"].clear()
        cache["phase"].clear()

    def _cache_supervised_batch(self, stage: str, inv_z: torch.Tensor, labels: dict) -> None:
        cache = self._supervised_cache.get(stage)
        if cache is None:
            return
        cache["latents"].append(inv_z.detach().to(torch.float32).cpu())
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
        metrics = self._compute_cluster_metrics(latents, labels, stage)
        if not metrics:
            return
        for name, value in metrics.items():
            self.log(
                f"{stage}_phase_{name.lower()}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        cache["latents"].clear()
        cache["phase"].clear()

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
        if len(batch) > 1 and batch[1] is not None:
            labels["phase"] = batch[1]
        if len(batch) > 2 and batch[2] is not None:
            labels["grain"] = batch[2]
        if len(batch) > 3 and batch[3] is not None:
            labels["orientation"] = batch[3]
        if len(batch) > 4 and batch[4] is not None:
            labels["quaternion"] = batch[4]
        if len(batch) > 5 and batch[5] is not None:
            labels["meta"] = batch[5]
        return pc, labels

    def _eq_to_decoder_latent(self, eq_z: torch.Tensor) -> torch.Tensor:
        """Pool per-point equivariant features to match decoder latent expectations."""
        pooled = eq_z.mean(dim=-1)  # (B, C, 3)
        return pooled.reshape(pooled.size(0), -1)

    @staticmethod
    def _identity_rotation(batch_size: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(3, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    def _infer_eq_latent_dim(self, cfg) -> int:
        """Determine flattened equivariant latent size to initialize the rotation head."""
        num_points = None
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is not None and hasattr(data_cfg, "get"):
            num_points = data_cfg.get("num_points", None)
        if num_points is None:
            decoder_cfg = getattr(cfg, "decoder", None)
            if decoder_cfg is not None and hasattr(decoder_cfg, "get"):
                decoder_kwargs = decoder_cfg.get("kwargs", {})
                if decoder_kwargs and hasattr(decoder_kwargs, "get"):
                    num_points = decoder_kwargs.get("num_points", None)
        if num_points is None:
            num_points = 1024

        dummy_points = torch.zeros(1, int(num_points), 3)
        encoder_was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            _, eq_z, _ = self.encoder(dummy_points)
        if encoder_was_training:
            self.encoder.train()
        return int(eq_z.shape[1] * eq_z.shape[2])


def _orthogonalize(mat: torch.Tensor) -> torch.Tensor:
    """Return closest orthogonal matrix using SVD."""
    orig_dtype = mat.dtype
    u, _, v = torch.linalg.svd(mat.to(torch.float32))
    return (u @ v.transpose(-1, -2)).to(orig_dtype)
