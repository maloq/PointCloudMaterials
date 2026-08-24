from __future__ import annotations

from .registry import ENCODERS


_REGISTRY_INITIALIZED = False


def _ensure_registry_loaded() -> None:
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return

    from . import dgcnn, pointnet, ri_mae_encoder, vn_encoders  # noqa: F401

    _REGISTRY_INITIALIZED = True


def build_encoder(name: str, **kwargs):
    """Build one retained paper encoder by its explicit registry name."""
    _ensure_registry_loaded()
    encoder_name = str(name)
    if encoder_name not in ENCODERS:
        raise KeyError(
            f"Unknown encoder {encoder_name!r}. Available encoders: {sorted(ENCODERS)}"
        )
    return ENCODERS[encoder_name](**kwargs)


def available_encoder_names() -> tuple[str, ...]:
    _ensure_registry_loaded()
    return tuple(sorted(ENCODERS))
