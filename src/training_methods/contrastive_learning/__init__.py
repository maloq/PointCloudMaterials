__all__ = [
    "VICRegModule",
    "VICRegLoss",
    "FactorVAELoss",
    "SwAVLoss",
]


def __getattr__(name):
    if name == "VICRegModule":
        from .vicreg_module import VICRegModule

        return VICRegModule
    if name == "FactorVAELoss":
        from .vicreg_module import FactorVAELoss

        return FactorVAELoss
    if name == "VICRegLoss":
        from src.training_methods.shared.vicreg import VICRegLoss

        return VICRegLoss
    if name == "SwAVLoss":
        from src.training_methods.shared.swav import SwAVLoss

        return SwAVLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
