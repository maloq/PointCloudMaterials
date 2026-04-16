from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings

from omegaconf import DictConfig

from .registry import ENCODERS

_REGISTRY_INITIALIZED = False
_WARNED_DROPPED_KWARGS: set[tuple[str, tuple[str, ...]]] = set()
_NON_ENCODER_MODULES = {"__init__", "base", "factory", "registry", "runtime"}


def _ensure_registry_loaded() -> None:
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return

    package_name = __name__.rsplit(".", 1)[0]
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        raise RuntimeError(f"Encoder package {package_name!r} does not expose __path__ for discovery.")

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name in _NON_ENCODER_MODULES:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _REGISTRY_INITIALIZED = True


def _filter_kwargs(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
    )
    if accepts_var_keyword:
        return kwargs

    valid_names = set(params.keys()) - {"self"}
    filtered = {key: value for key, value in kwargs.items() if key in valid_names}
    dropped = sorted(key for key in kwargs.keys() if key not in valid_names)
    if dropped:
        warn_key = (cls.__name__, tuple(dropped))
        if warn_key not in _WARNED_DROPPED_KWARGS:
            _WARNED_DROPPED_KWARGS.add(warn_key)
            warnings.warn(
                f"Ignoring unsupported kwargs for {cls.__name__}: {', '.join(dropped)}",
                stacklevel=2,
            )
    return filtered


def build_encoder(cfg: DictConfig):
    _ensure_registry_loaded()

    encoder_cfg = getattr(cfg, "encoder", None)
    if encoder_cfg is None:
        raise ValueError("Configuration is missing required section 'encoder'.")

    encoder_name = getattr(encoder_cfg, "name", None)
    if encoder_name is None:
        raise ValueError("Configuration is missing required field 'encoder.name'.")
    encoder_name = str(encoder_name)

    encoder_cls = ENCODERS.get(encoder_name)
    if encoder_cls is None:
        available = ", ".join(sorted(ENCODERS))
        raise KeyError(
            f"Unknown encoder {encoder_name!r}. Registered encoders: [{available}]"
        )

    encoder_kwargs_cfg = getattr(encoder_cfg, "kwargs", {}) or {}
    encoder_kwargs = _filter_kwargs(encoder_cls, dict(encoder_kwargs_cfg))
    return encoder_cls(**encoder_kwargs)


def available_encoder_names() -> tuple[str, ...]:
    _ensure_registry_loaded()
    return tuple(sorted(ENCODERS))

