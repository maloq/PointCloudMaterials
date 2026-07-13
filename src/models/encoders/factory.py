from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from .registry import ENCODERS


_REGISTRY_INITIALIZED = False


def _ensure_registry_loaded() -> None:
    """Import the repository's explicit encoder implementation modules."""
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return

    from . import (  # noqa: F401
        dgcnn,
        egnn_encoder,
        geo_frame_transformer,
        mace_encoder,
        mlp,
        nequip_encoder,
        pointnet,
        ri_mae_encoder,
        vn_encoders,
    )

    _REGISTRY_INITIALIZED = True


def build_encoder(cfg: DictConfig):
    """Build the encoder named by the repository Hydra config."""
    _ensure_registry_loaded()
    encoder_name = cfg.encoder.name
    if encoder_name not in ENCODERS:
        raise KeyError(
            f"Unknown encoder {encoder_name!r}. Registered encoders: {sorted(ENCODERS)}"
        )
    encoder_kwargs = OmegaConf.to_container(cfg.encoder.kwargs, resolve=True)
    return ENCODERS[encoder_name](**encoder_kwargs)


def available_encoder_names() -> tuple[str, ...]:
    _ensure_registry_loaded()
    return tuple(sorted(ENCODERS))
