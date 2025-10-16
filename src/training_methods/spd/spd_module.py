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
from .vn_models import PointNetEncoderVN, VNDGCNNEncoder, SimpleRot, ComplexRot
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss
from src.loss.neighbor_latent_loss import neighbor_pair_latent_loss
from src.training_methods.spd.rot_heads import Rot6DHead, RotMatrixHead

class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.decoder = build_model(cfg, only_decoder=True)

        encoder_kwargs = self.hparams.encoder.get('kwargs', {})
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size)
        self.rotation_mode = self._resolve_rotation_mode(cfg)
        self._validate_rotation_mode()

        encoder_cfg = self.hparams.encoder
        encoder_name = encoder_cfg.get('name', 'PnE_VN') if hasattr(encoder_cfg, 'get') else 'PnE_VN'
        if encoder_name == 'PnE_VN':
            self.encoder = PointNetEncoderVN(
                latent_size=self.encoder_latent_size,
                n_knn=20,
                hidden_dim1=encoder_kwargs.get('hidden_dim1', 256),
                hidden_dim2=encoder_kwargs.get('hidden_dim2', 512),
                pooling=encoder_kwargs.get('pooling', 'mean'),
                feature_transform=encoder_kwargs.get('feature_transform', False),

            )
        elif encoder_name == 'VN_DGCNN':
            feature_dims = encoder_kwargs.get('feature_dims', (96, 96, 192, 384, 576))
            global_mlp_dims = encoder_kwargs.get('global_mlp_dims', (256, 128))
            self.encoder = VNDGCNNEncoder(
                latent_size=self.encoder_latent_size,
                n_knn=encoder_kwargs.get('n_knn', 20),
                pooling=encoder_kwargs.get('pooling', 'mean'),
                feature_dims=tuple(feature_dims),
                global_mlp_dims=tuple(global_mlp_dims),
                global_dropout=encoder_kwargs.get('global_dropout', 0.5),
                share_nonlinearity=encoder_kwargs.get('share_nonlinearity', True),
                std_feature_hidden_dims=encoder_kwargs.get('std_feature_hidden_dims'),
                use_batchnorm=encoder_kwargs.get('use_batchnorm', True),
            )

        if self.rotation_mode in {"sixd_head", "matrix_head"}:
            rot_net_cfg = getattr(self.hparams, 'rot_net', {})
            rot_net_kwargs = rot_net_cfg.get('kwargs', {}) if hasattr(rot_net_cfg, 'get') else {}
            self.rot_head_in_features = self._infer_eq_latent_dim(cfg)
            if self.rotation_mode == "sixd_head":
                self.rot_net = Rot6DHead(
                    in_features=self.rot_head_in_features,
                    hidden=rot_net_kwargs.get('hidden', 256),
                    use_attention=rot_net_kwargs.get('use_attention', True),
                )
            else:
                self.rot_net = RotMatrixHead(
                    in_features=self.rot_head_in_features,
                    hidden=rot_net_kwargs.get('hidden', 256),
                    use_attention=rot_net_kwargs.get('use_attention', True),
                    orthogonalize=rot_net_kwargs.get('orthogonalize', False),
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

        if self.rotation_mode == "eq_decoder":
            decoder_latent = self._eq_to_decoder_latent(eq_z)
            cano = self._decode(decoder_latent)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_no_rot":
            cano = self._decode(inv_z)
            rot = self._identity_rotation(cano.size(0), cano.device, cano.dtype)
            recon = cano
        elif self.rotation_mode == "inv_kabsch":
            cano = self._decode(inv_z)
            rot = self._estimate_kabsch_rotation(cano, pc)
            recon = self._apply_rotation(cano, rot)
        else:
            cano = self._decode(inv_z)
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

        if legacy:
            legacy_name = f"{stage}_{name.replace('/', '_')}"
            legacy_kwargs = dict(log_kwargs)
            self.log(legacy_name, value, on_step=on_step, on_epoch=on_epoch, **legacy_kwargs)

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
        # self._log_metric(stage, "pd", loss_pd, prog_bar=False)
        # self._log_metric(stage, "rot", loss_rot)
        self._log_metric(stage, "recon", loss_recon)
        self._log_metric(stage, "emd", loss_recon)
        self._log_metric(stage, "ortho", ortho_loss)

        # Optional supervised diagnostics when synthetic labels are available
        if rot is not None and labels.get("orientation") is not None:
            gt_rot = labels["orientation"].to(device=rot.device, dtype=torch.float32)
            geodesic = self._rotation_geodesic(rot.to(torch.float32), gt_rot)
            self._log_metric(stage, "rot_geodesic_deg", geodesic * (180.0 / torch.pi), prog_bar=False)

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
                self._log_metric('train', 'neighbor_loss', loss_pairs, prog_bar=False, on_step=True, on_epoch=True, batch_size=inv_i.shape[0])
                self._log_metric('train', 'neighbor_loss_scaled', scaled, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
                self._log_metric('train', 'neighbor_pairs', float(inv_i.shape[0]), on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
                for k, v in stats.items():
                    self._log_metric('train', k, v, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
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
            self._log_metric('train', 'neighbor_loss', loss_pairs, prog_bar=True, on_step=True, on_epoch=True, batch_size=inv_i.shape[0])
            self._log_metric('train', 'neighbor_loss_scaled', scaled, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
            self._log_metric('train', 'neighbor_pairs', float(inv_i.shape[0]), on_step=True, on_epoch=False, batch_size=inv_i.shape[0])
            for k, v in stats.items():
                self._log_metric('train', k, v, on_step=True, on_epoch=False, batch_size=inv_i.shape[0])

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
            self._log_metric(
                stage,
                f"phase/{name.lower()}",
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

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent code and ensure (B, N, 3) layout."""
        cano = self.decoder(latent)
        if cano.ndim != 3:
            raise ValueError(f"Decoder output must be 3D tensor (got {cano.shape})")
        if cano.shape[1] == 3 and cano.shape[2] != 3:
            cano = cano.permute(0, 2, 1)
        return cano.contiguous()

    @staticmethod
    def _apply_rotation(points: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """Apply rotation matrices to batched point clouds."""
        return (rot @ points.transpose(1, 2)).transpose(1, 2).contiguous()

    @staticmethod
    def _identity_rotation(batch_size: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(3, device=device, dtype=dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _ensure_points_last_dim(points: torch.Tensor) -> torch.Tensor:
        """Ensure points are shaped as (B, N, 3)."""
        if points.ndim != 3:
            raise ValueError(f"Expected point cloud of shape (B, N, 3) or (B, 3, N), got {points.shape}")
        if points.shape[-1] == 3:
            return points
        if points.shape[1] == 3:
            return points.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot interpret point cloud shape {points.shape}")

    def _estimate_kabsch_rotation(self, cano: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute optimal rotation aligning canonical output to target using Kabsch algorithm."""
        target_points = self._ensure_points_last_dim(target).to(dtype=cano.dtype, device=cano.device)
        cano_points = cano.to(torch.float32)
        target_points32 = target_points.to(torch.float32)

        cano_centered = cano_points - cano_points.mean(dim=1, keepdim=True)
        target_centered = target_points32 - target_points32.mean(dim=1, keepdim=True)

        cov = cano_centered.transpose(1, 2) @ target_centered
        U, _, Vh = torch.linalg.svd(cov, full_matrices=False)
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)

        det = torch.linalg.det(R)
        neg_mask = det < 0
        if torch.any(neg_mask):
            Vh_adjusted = Vh.clone()
            Vh_adjusted[neg_mask, -1, :] *= -1
            R = Vh_adjusted.transpose(-1, -2) @ U.transpose(-1, -2)

        return R.to(dtype=cano.dtype)

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

    def _resolve_rotation_mode(self, cfg) -> str:
        """Derive rotation handling mode, preserving backward-compatible flags."""
        mode = None
        if hasattr(cfg, "get"):
            mode = cfg.get("rotation_mode", None)
        if mode is None and hasattr(cfg, "rotation_mode"):
            mode = cfg.rotation_mode
        if mode is None:
            bypass = False
            if hasattr(cfg, "get"):
                bypass = cfg.get("bypass_rot_head", False)
            elif hasattr(cfg, "bypass_rot_head"):
                bypass = cfg.bypass_rot_head
            return "eq_decoder" if bypass else "sixd_head"
        return str(mode).lower()

    def _validate_rotation_mode(self) -> None:
        valid_modes = {"sixd_head", "matrix_head", "eq_decoder", "inv_no_rot", "inv_kabsch"}
        if self.rotation_mode not in valid_modes:
            raise ValueError(
                f"Unsupported rotation_mode '{self.rotation_mode}'. "
                f"Valid options: {sorted(valid_modes)}"
            )
