from __future__ import annotations

import torch

from ..base import Decoder
from ..registry import register_decoder
from src.training_methods.eq_graph.equivariant_decoders import FlowMatchingDecoder


@register_decoder("FlowMatchingEquivariant")
class FlowMatchingEquivariantDecoder(Decoder):
    """
    Wrapper around the flow-matching decoder so it can be instantiated from configs.

    Training mode: returns (dummy_recon, loss, None) similar to DiffusionDecoder.
    Eval mode: samples reconstructions for metrics/visualization.
    """

    def __init__(
        self,
        latent_size: int,
        num_points: int,
        scalar_dim: int = 128,
        vector_dim: int = 32,
        n_layers: int = 6,
        sigma_min: float = 0.001,
        sample_steps: int = 50,
    ):
        super().__init__()
        self.model = FlowMatchingDecoder(
            latent_dim=latent_size,
            n_atoms=num_points,
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            n_layers=n_layers,
            sigma_min=sigma_min,
        )
        self.sample_steps = sample_steps

        # Flags consumed by the training module
        self.is_flow_matching = True
        self.use_invariant_latent = True

    def forward(self, z: torch.Tensor, gt_pts: torch.Tensor | None = None):
        if self.training and gt_pts is not None:
            # Flow matching loss uses a noise sample paired with the ground truth
            noise = torch.randn_like(gt_pts)
            loss = self.model.compute_loss(noise, gt_pts, z)
            return gt_pts, loss, None

        # In eval, generate reconstructions for metrics/visualization
        recon = self.model.sample(z, n_steps=self.sample_steps)
        zero = torch.zeros((), device=z.device, dtype=z.dtype)
        return recon, zero, None
