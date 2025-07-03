from omegaconf import DictConfig
from .registry import ENCODERS, DECODERS

def build_model(cfg: DictConfig):
    enc_cfg, dec_cfg = cfg.encoder, cfg.decoder   # <- clearer than 'type'
    enc_cls = ENCODERS[enc_cfg.name]
    dec_cls = DECODERS[dec_cfg.name]
    encoder = enc_cls(**enc_cfg.kwargs)
    decoder = dec_cls(**dec_cfg.kwargs)
    return encoder, decoder