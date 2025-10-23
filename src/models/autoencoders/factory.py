from omegaconf import DictConfig
from .registry import ENCODERS, DECODERS

_REGISTRY_INITIALIZED = False


def _ensure_registry_loaded() -> None:
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return
    # Import encoder/decoder packages so their registration side effects run.
    import src.models.autoencoders.encoders  # noqa: F401
    import src.models.autoencoders.decoders  # noqa: F401

    _REGISTRY_INITIALIZED = True


def build_model(cfg: DictConfig, only_decoder: bool = False):
    _ensure_registry_loaded()
    enc_cfg, dec_cfg = cfg.encoder, cfg.decoder

    if only_decoder:
        dec_cls = DECODERS[dec_cfg.name]
        return dec_cls(**dec_cfg.kwargs)
    else:
        enc_cls = ENCODERS[enc_cfg.name]
        dec_cls = DECODERS[dec_cfg.name]
        
    encoder = enc_cls(**enc_cfg.kwargs)
    decoder = dec_cls(**dec_cfg.kwargs)
    return encoder, decoder
