__all__ = [
    "VICRegModule",
    "VICRegLoss",
    "SwAVLoss",
]


def __getattr__(name):
    if name == "VICRegModule":
        from .vicreg_module import VICRegModule

        return VICRegModule
    if name == "VICRegLoss":
        from .vicreg import VICRegLoss

        return VICRegLoss
    if name == "SwAVLoss":
        from .swav import SwAVLoss

        return SwAVLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
