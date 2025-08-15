import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.functional import normalize

import sys,os
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.optimizer_utils import get_optimizers_and_scheduler
import src.models.autoencoders.encoders
import src.models.autoencoders.decoders
from .vn_models import PointNetEncoderVN, SimpleRot, ComplexRot
from src.loss.reconstruction_loss import kl_latent_regularizer, rotation_geodesic_kabsch_loss
from src.loss.pdist_loss import pairwise_distance_loss
from src.training_methods.spd.rot_heads import Rot6DHead, sixd_to_so3

class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.decoder = build_model(cfg, only_decoder=True)
        
        encoder_kwargs = self.hparams.encoder.get('kwargs', {})
        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size) 

        if encoder_kwargs.get('name', 'PnE_VN') == 'PnE_VN':
            self.encoder = PointNetEncoderVN(
                latent_size=self.encoder_latent_size // 2,
                n_knn=32,
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
        rot_net_kwargs = self.hparams.rot_net.get('kwargs', {})
        self.rot_net = Rot6DHead(hidden=rot_net_kwargs.get('hidden', 256), use_attention=rot_net_kwargs.get('use_attention', True))

        self.ortho_scale = cfg.get("ortho_scale", 0.01)
        self.kl_latent_loss_scale = cfg.get("kl_latent_loss_scale", 0.0)
        self.pdist_loss_scale = cfg.get("pdist_loss_scale", 0.0)
        self.rotation_loss_scale = cfg.get("kabsch_rotation_loss_scale", 0.0)

    def forward(self, pc: torch.Tensor):
        # pc: (B, N, 3)
        inv_z, eq_z, _ = self.encoder(pc)

        cano = self.decoder(inv_z)
        if cano.shape[1] == 3:
            cano = cano.permute(0, 2, 1)
        rot = self.rot_net(eq_z)

        # rot = _orthogonalize(rot)

        recon = (rot @ cano.transpose(1, 2)).transpose(1, 2)
        return inv_z, recon, cano, rot 

    def _step(self, batch, batch_idx, stage: str):
        pc = batch
        inv_z, recon, cano, rot = self(pc)
        
        recon_f32 = recon.to(torch.float32)
        pc_f32    = pc.to(torch.float32)

        loss_recon, _   = sinkhorn_distance(recon_f32.contiguous(), pc_f32)
        # loss_recon, _ = chamfer_distance(recon_f32, pc_f32)
        loss_chamfer, _ = chamfer_distance(recon_f32, pc_f32)

        ortho_loss = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                 - torch.eye(3, device=self.device)) ** 2)
        
        # loss_pd = pairwise_distance_loss(pred=recon_f32,target=pc_f32)
        # loss_rot = rotation_geodesic_kabsch_loss(rot.to(torch.float32), cano.to(torch.float32), pc_f32)
        loss = loss_recon + self.ortho_scale * ortho_loss 
        if self.kl_latent_loss_scale > 0:
            kl_loss = kl_latent_regularizer(inv_z)
            loss += self.kl_latent_loss_scale * kl_loss
            self.log(f"{stage}_kl_loss", kl_loss)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_chamfer", loss_chamfer, prog_bar=False)
        # self.log(f"{stage}_rot", loss_rot)
        self.log(f"{stage}_recon", loss_recon)
        self.log(f"{stage}_ortho", ortho_loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())


def _orthogonalize(mat: torch.Tensor) -> torch.Tensor:
    """Return closest orthogonal matrix using SVD."""
    orig_dtype = mat.dtype
    u, _, v = torch.linalg.svd(mat.to(torch.float32))
    return (u @ v.transpose(-1, -2)).to(orig_dtype)