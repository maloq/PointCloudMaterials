# dimenet_contrastive.py
"""Contrastive pre‑training of point‑cloud graphs with **DimeNet**.

The module builds two stochastic augmentations of each point cloud,
turns them into *PyG* graphs, embeds them with **DimeNet**, pools the
node representations into a graph vector and applies a SimCLR‑style
NT‑Xent objective.

Changes from v1
---------------
* **Graph‑level pooling** – we now mean‑pool node embeddings with
  `global_mean_pool`, so the NT‑Xent operates on a (B×B) similarity
  matrix instead of (BN×BN). This both reduces memory and avoids the
  earlier `NaN` explosion.
* **FP32 loss path** – the similarity matrix and cross‑entropy are
  computed outside autocast for numeric stability when training with
  AMP.
* **Sentinel mask** – still uses the dtype‑safe `‑torch.finfo(dtype).max`.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn.models import DimeNet
from torch_geometric.nn import global_mean_pool

# -----------------------------------------------------------------------------
# Hyper‑parameters specific to DimeNet (kept in cfg.dimenet)
# -----------------------------------------------------------------------------
@dataclass
class DimeNetHP:
    hidden_channels: int = 128
    num_blocks: int = 4
    num_bilinear: int = 8
    num_radial: int = 6
    num_spherical: int = 7
    envelope_exponent: int = 5


class DimeNetContrastive(pl.LightningModule):
    """Graph‑level contrastive pre‑training with DimeNet."""

    def __init__(self, cfg: Any):
        super().__init__()
        self.save_hyperparameters(cfg)

        #  ‑‑‑ data / augmentation params
        self.num_points: int = cfg.data.num_points
        self.cutoff: float = getattr(cfg.data, "radius_graph", 5.0)
        self.temperature: float = cfg.get("temperature", 0.1)
        self.jitter_std: float = cfg.get("jitter_std", 0.01)

        #  ‑‑‑ model
        hp = DimeNetHP(**cfg.dimenet) if hasattr(cfg, "dimenet") else DimeNetHP()
        self.encoder = DimeNet(
            hidden_channels=hp.hidden_channels,
            out_channels=cfg.latent_size,
            num_blocks=hp.num_blocks,
            num_bilinear=hp.num_bilinear,
            num_radial=hp.num_radial,
            num_spherical=hp.num_spherical,
            cutoff=self.cutoff,
            envelope_exponent=hp.envelope_exponent,
        )

    # ------------------------------------------------------------------
    #  basic utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def random_rotate_point_cloud_batch(self, pcs: torch.Tensor) -> torch.Tensor:
        """Random 3‑D rotation per example (uses Z‑Y‑X Euler)."""
        B, _, _ = pcs.shape
        device = pcs.device
        angles = torch.rand(B, 3, device=device) * 2 * torch.pi  # α,β,γ

        c, s = torch.cos(angles), torch.sin(angles)
        Rz = torch.stack([
            torch.stack([ c[:, 2],‑s[:, 2], torch.zeros_like(c[:, 2])], dim=‑1),
            torch.stack([ s[:, 2],  c[:, 2], torch.zeros_like(c[:, 2])], dim=‑1),
            torch.tensor([0,0,1], device=device).repeat(B,1),
        ], dim=‑2)
        Ry = torch.stack([
            torch.stack([ c[:, 1], torch.zeros_like(c[:, 1]), s[:, 1]], dim=‑1),
            torch.stack([ torch.zeros_like(c[:, 1]), torch.ones_like(c[:, 1]), torch.zeros_like(c[:, 1])], dim=‑1),
            torch.stack([‑s[:, 1], torch.zeros_like(c[:, 1]), c[:, 1]], dim=‑1),
        ], dim=‑2)
        Rx = torch.stack([
            torch.stack([ torch.ones_like(c[:, 0]), torch.zeros_like(c[:, 0]), torch.zeros_like(c[:, 0])], dim=‑1),
            torch.stack([ torch.zeros_like(c[:, 0]), c[:, 0],‑s[:, 0]], dim=‑1),
            torch.stack([ torch.zeros_like(c[:, 0]), s[:, 0], c[:, 0]], dim=‑1),
        ], dim=‑2)

        R = Rz.matmul(Ry).matmul(Rx)  # (B,3,3)
        return torch.bmm(pcs, R)      # (B,N,3)·(B,3,3) → (B,N,3)

    # ------------------------------------------------------------------
    #  data wrangling
    # ------------------------------------------------------------------
    def _pc_to_graph(self, pcs: torch.Tensor) -> Data:
        """Flatten (B,N,3) → PyG *Data* so DimeNet can build edges."""
        B, N, _ = pcs.shape
        pos = pcs.reshape(‑1, 3)
        z = torch.ones(pos.size(0), dtype=torch.long, device=pos.device)  # dummy atomic number
        batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        return Data(z=z, pos=pos, batch=batch)

    # ------------------------------------------------------------------
    #  contrastive loss (NT‑Xent)
    # ------------------------------------------------------------------
    @staticmethod
    def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, *, tau: float) -> torch.Tensor:
        """Compute NT‑Xent in **FP32** for stability."""
        with torch.cuda.amp.autocast(enabled=False):
            z1 = F.normalize(z1.float(), dim=‑1)
            z2 = F.normalize(z2.float(), dim=‑1)
            B = z1.size(0)
            z = torch.cat([z1, z2], dim=0)                    # (2B,D)
            sim = torch.mm(z, z.t()) / tau                   # cosine sim
            sim = sim.masked_fill(torch.eye(2*B, device=z.device, dtype=torch.bool),
                                  ‑torch.finfo(sim.dtype).max)
            targets = torch.arange(B, device=z.device)
            loss = 0.5 * (F.cross_entropy(sim[:B, B:], targets) +
                           F.cross_entropy(sim[B:, :B], targets))
        return loss

    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        pcs, _ = batch  # (B,N,3)
        noise = self.jitter_std * torch.randn_like(pcs)
        aug1 = self.random_rotate_point_cloud_batch(pcs) + noise
        aug2 = self.random_rotate_point_cloud_batch(pcs) + noise

        g1 = self._pc_to_graph(aug1)
        g2 = self._pc_to_graph(aug2)

        z1_nodes = self.encoder(g1.z, g1.pos, g1.batch)
        z2_nodes = self.encoder(g2.z, g2.pos, g2.batch)
        z1 = global_mean_pool(z1_nodes, g1.batch)  # (B,D)
        z2 = global_mean_pool(z2_nodes, g2.batch)

        loss = self._nt_xent(z1, z2, tau=self.temperature)
        self.log("train_loss", loss, prog_bar=True, batch_size=pcs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        pcs, _ = batch
        g1 = self._pc_to_graph(self.random_rotate_point_cloud_batch(pcs))
        g2 = self._pc_to_graph(self.random_rotate_point_cloud_batch(pcs))
        z1 = global_mean_pool(self.encoder(g1.z, g1.pos, g1.batch), g1.batch)
        z2 = global_mean_pool(self.encoder(g2.z, g2.pos, g2.batch), g2.batch)
        val_loss = self._nt_xent(z1, z2, tau=self.temperature)
        self.log("val_loss", val_loss, prog_bar=True, batch_size=pcs.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.get("lr", 1e‑3))
