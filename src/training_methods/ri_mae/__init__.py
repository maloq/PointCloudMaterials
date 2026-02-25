"""RI-MAE package exports with lazy loading to avoid import cycles."""

from __future__ import annotations

from typing import Any

__all__ = ["RIMAEModule", "RIMAEInvariantEncoderForContrastive"]


def __getattr__(name: str) -> Any:
    if name == "RIMAEModule":
        from .ri_mae_module import RIMAEModule

        return RIMAEModule
    if name == "RIMAEInvariantEncoderForContrastive":
        from .contrastive_backbone import RIMAEInvariantEncoderForContrastive

        return RIMAEInvariantEncoderForContrastive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
