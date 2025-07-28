import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys,os
sys.path.append(os.getcwd())
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_distance, sinkhorn_distance
from src.utils.optimizer_utils import get_optimizers_and_scheduler
import src.models.autoencoders.encoders
import src.models.autoencoders.decoders
from .vn_models import PointNetEncoderVN, SimpleRot, ComplexRot
from src.loss.reconstruction_loss import kl_latent_regularizer





class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.decoder = build_model(cfg, only_decoder=True)
        
        encoder_kwargs = self.hparams.encoder.get('kwargs', {})

        self.encoder_latent_size = encoder_kwargs.get('latent_size', self.hparams.latent_size) 

        self.encoder = PointNetEncoderVN(
            latent_size=self.encoder_latent_size // 2,
            n_knn=32,
            hidden_dim1=encoder_kwargs.get('hidden_dim1', 256),
            hidden_dim2=encoder_kwargs.get('hidden_dim2', 1024),
            pooling=encoder_kwargs.get('pooling', 'mean'),
            feature_transform=encoder_kwargs.get('feature_transform', False)
        )

        # Align rotation network input with the equivariant latent vector produced
        # by the encoder.  The channel count equals encoder_latent_size // 3.
        # The encoder returns an equivariant latent tensor whose channel
        # dimension equals (latent_size // 2) // 3 = latent_size // 6.
        # Align the rotation head to this size to avoid shape mismatches.
        self.rot_net = ComplexRot(self.encoder_latent_size // 6)

        # Map the encoder-predicted translation (which has encoder_latent_size//3
        # channels) to a proper 3-D translation vector that can be broadcast
        # with the reconstructed points.

        self.ortho_scale = cfg.get("ortho_scale", 0.01)
        self.kl_latent_loss_scale = cfg.get("kl_latent_loss_scale", 0.0)

    def forward(self, pc: torch.Tensor):
        # pc: (B, N, 3)
        inv_z, eq_z, _ = self.encoder(pc)

        cano = self.decoder(inv_z)
        if cano.shape[1] == 3:
            cano = cano.permute(0, 2, 1)

        rot = _orthogonalize(self.rot_net(eq_z))
        # Flatten any extra singleton dimensions that may appear and project
        # the high-dim translation estimate down to 3-D.  This guarantees an
        # input shape of (B, C) for the linear layer even if the encoder
        # returns (B, C, 1).

        recon = (rot @ cano.transpose(1, 2)).transpose(1, 2)
        return inv_z, recon, cano, rot 

    def _step(self, batch, batch_idx, stage: str):
        pc = batch
        inv_z, recon, cano, rot = self(pc)

        # ------------------------------------------------------------------
        # Losses are computed with external libraries (geomloss & PyTorch3D)
        # which currently do NOT support bf16 on CUDA.  When running with AMP
        # (autocast="bf16") the model outputs `recon`/`rot` in bf16 while the
        # input `pc` stays in fp32, leading to dtype mismatches.  We therefore
        # cast everything to fp32 **just for the loss computation**.
        # ------------------------------------------------------------------

        # Cast to fp32 for external loss functions which do not support bf16
        recon_f32 = recon.to(torch.float32)
        pc_f32    = pc.to(torch.float32)

        loss_recon, _   = sinkhorn_distance(recon_f32.contiguous(), pc_f32)
        loss_chamfer, _ = chamfer_distance(recon_f32, pc_f32)

        ortho_loss = torch.mean((rot.transpose(1, 2).float() @ rot.float()
                                 - torch.eye(3, device=self.device)) ** 2)
        
        loss = loss_recon + self.ortho_scale * ortho_loss

        if self.kl_latent_loss_scale > 0:
            kl_loss = kl_latent_regularizer(inv_z)
            loss += self.kl_latent_loss_scale * kl_loss
            self.log(f"{stage}_kl_loss", kl_loss)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_chamfer", loss_chamfer, prog_bar=True)
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
    # The SVD does not seem to support bfloat16 on CUDA.
    # We cast to float32 and then back to the original type.
    orig_dtype = mat.dtype
    u, _, v = torch.linalg.svd(mat.to(torch.float32))
    return (u @ v.transpose(-1, -2)).to(orig_dtype)