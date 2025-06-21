from abc import ABC, abstractmethod
import torch, torch.nn as nn

class Encoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (latent, *aux)"""

class Decoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed point cloud (B, N, 3)"""