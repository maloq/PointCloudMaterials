"""Explicit selection for the two maintained Lightning workflows."""

from typing import Any


_MODEL_TYPE_ALIASES = {
    "vicreg": "contrastive",
    "visreg": "contrastive",
    "contrastive": "contrastive",
    "temporal_vicreg": "temporal_ssl",
    "temporal_ssl": "temporal_ssl",
}



def _method_name_from_cfg(cfg: Any) -> str | None:
    method_cfg = getattr(cfg, "method", None)
    if method_cfg is not None:
        raw_name = getattr(method_cfg, "name", None)
        if raw_name is None and isinstance(method_cfg, dict):
            raw_name = method_cfg.get("name")
        if raw_name is not None:
            return str(raw_name).strip().lower()

    raw_name = getattr(cfg, "training_method", None)
    if raw_name is not None:
        return str(raw_name).strip().lower()

    model_type = getattr(cfg, "model_type", None)
    if model_type is not None:
        return _MODEL_TYPE_ALIASES.get(str(model_type).strip().lower())
    return None



def resolve_training_method(cfg: Any = None, *, method_name: str | None = None):
    """Return the concrete module class and its default analysis policy."""
    resolved_name = (
        str(method_name).strip().lower() if method_name is not None else None
    )
    if not resolved_name and cfg is not None:
        resolved_name = _method_name_from_cfg(cfg)
    available = "contrastive, temporal_ssl"
    if not resolved_name:
        raise ValueError(
            "Could not resolve training method. Set cfg.method.name, "
            "cfg.training_method, or cfg.model_type. "
            f"Registered methods: [{available}]"
        )
    resolved_name = _MODEL_TYPE_ALIASES.get(resolved_name, resolved_name)
    if resolved_name == "contrastive":
        from src.training_methods.contrastive_learning.vicreg_module import (
            VICRegModule,
        )

        return VICRegModule, True
    if resolved_name == "temporal_ssl":
        from src.training_methods.temporal_ssl.temporal_ssl_module import (
            TemporalSSLModule,
        )

        return TemporalSSLModule, False
    raise KeyError(
        f"Unknown training method {resolved_name!r}. "
        f"Registered methods: [{available}]"
    )
