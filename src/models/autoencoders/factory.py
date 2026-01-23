import inspect
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


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Filter kwargs to only include parameters accepted by the class __init__."""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    # Check if class accepts **kwargs
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_keyword:
        return kwargs
    # Filter to only accepted parameter names
    valid_names = set(params.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_names}


def build_model(cfg: DictConfig, only_decoder: bool = False):
    _ensure_registry_loaded()
    enc_cfg, dec_cfg = cfg.encoder, cfg.decoder

    if only_decoder:
        dec_cls = DECODERS[dec_cfg.name]
        dec_kwargs = _filter_kwargs(dec_cls, dict(dec_cfg.kwargs))
        return dec_cls(**dec_kwargs)
    else:
        enc_cls = ENCODERS[enc_cfg.name]
        dec_cls = DECODERS[dec_cfg.name]
        
    enc_kwargs = _filter_kwargs(enc_cls, dict(enc_cfg.kwargs))
    dec_kwargs = _filter_kwargs(dec_cls, dict(dec_cfg.kwargs))
    encoder = enc_cls(**enc_kwargs)
    decoder = dec_cls(**dec_kwargs)
    return encoder, decoder
