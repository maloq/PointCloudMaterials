from __future__ import annotations

import torch

from ..base import Encoder
from ..registry import register_encoder
from src.training_methods.eq_graph.vn_graph_encoder import (
    VNGraphEncoderDelaunay,
    VNGraphEncoderMultiScale,
)


@register_encoder("VN_Graph_Delaunay")
class VNGraphDelaunayEncoder(Encoder):
    """
    Thin wrapper to expose the graph-based VN encoder through the project registry.
    Returns invariant and equivariant latents plus the anisotropy score.
    """

    def __init__(
        self,
        latent_size: int = 256,
        hidden_channels: tuple[int, ...] = (32, 64, 128, 256),
        use_batchnorm: bool = True,
        graph_type: str = "delaunay",
        radius_cutoff: float = 5.0,
        use_voronoi_features: bool = False,
        use_invariant_edges: bool = True,
    ):
        super().__init__()
        self.model = VNGraphEncoderDelaunay(
            latent_size=latent_size,
            hidden_channels=hidden_channels,
            use_batchnorm=use_batchnorm,
            graph_type=graph_type,
            radius_cutoff=radius_cutoff,
            use_voronoi_features=use_voronoi_features,
            use_invariant_edges=use_invariant_edges,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)


@register_encoder("VN_Graph_MultiScale")
class VNGraphMultiScaleEncoder(Encoder):
    """
    Multi-scale variant of the graph encoder, exposed for configuration-driven use.
    """

    def __init__(
        self,
        latent_size: int = 256,
        scales: tuple[float, ...] = (3.0, 5.0, 8.0),
        hidden_channels: int = 64,
        n_layers_per_scale: int = 2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.model = VNGraphEncoderMultiScale(
            latent_size=latent_size,
            scales=list(scales),
            hidden_channels=hidden_channels,
            n_layers_per_scale=n_layers_per_scale,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)
