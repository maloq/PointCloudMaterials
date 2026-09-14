"""Contrastive Lightning workflow and its public objective imports."""

from src.training_methods.shared.swav import SwAVLoss
from src.training_methods.shared.vicreg import VICRegLoss

from .vicreg_module import FactorVAELoss, VICRegModule

__all__ = ["VICRegModule", "VICRegLoss", "FactorVAELoss", "SwAVLoss"]
