import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.autoencoders.factory import build_model
from src.loss.reconstruction_loss import chamfer_loss
from src.utils.optimizer_utils import get_optimizers_and_scheduler
from .vn_models import PointNetEncoder, SimpleRot


def _orthogonalize(mat: torch.Tensor) -> torch.Tensor:
    """Return closest orthogonal matrix using SVD."""
    u, _, v = torch.linalg.svd(mat)
    return u @ v.transpose(-1, -2)


class ShapePoseDisentanglement(pl.LightningModule):
    """Simplified Shape‑Pose Disentanglement module."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Decoder is taken from existing autoencoder setup
        _, self.decoder = build_model(cfg)

        latent_dim = cfg.latent_size
        self.encoder = PointNetEncoder(latent_size=latent_dim)
        self.rot_net = SimpleRot(latent_dim // 3)

        # translation is predicted directly from the encoder

        self.ortho_scale = cfg.get("ortho_scale", 0.01)

    def forward(self, pc: torch.Tensor):
        # pc: (B, N, 3)
        inv_z, eq_z, trans = self.encoder(pc)
        cano = self.decoder(inv_z)
        if cano.shape[1] == 3:
            cano = cano.permute(0, 2, 1)

        rot = _orthogonalize(self.rot_net(eq_z))
        trans = trans.unsqueeze(1)

        recon = (rot @ cano.transpose(1, 2)).transpose(1, 2) + trans
        return recon, cano, rot, trans

    def _step(self, batch, batch_idx, stage: str):
        pc = batch
        recon, cano, rot, _ = self(pc)
        loss_recon, _ = chamfer_loss(recon, pc)
        ortho_loss = torch.mean((rot.transpose(1, 2) @ rot - torch.eye(3, device=self.device)) ** 2)
        loss = loss_recon + self.ortho_scale * ortho_loss
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_recon", loss_recon)
        self.log(f"{stage}_ortho", ortho_loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())
