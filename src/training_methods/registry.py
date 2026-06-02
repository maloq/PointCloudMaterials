from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class TrainingMethodSpec:
    name: str
    module_path: str
    class_name: str
    default_config: str
    run_post_training_analysis: bool = False

    def load_module_class(self):
        module = import_module(self.module_path)
        try:
            return getattr(module, self.class_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Training method {self.name!r} expected class {self.class_name!r} "
                f"in module {self.module_path!r}, but it was not found."
            ) from exc


_METHODS: dict[str, TrainingMethodSpec] = {}
_MODEL_TYPE_ALIASES = {
    "vicreg": "contrastive",
    "contrastive": "contrastive",
    "temporal_vicreg": "temporal_ssl",
    "temporal_ssl": "temporal_ssl",
    "temporal_rigs_vicreg": "temporal_rigs_ssl",
    "temporal_rigs_ssl": "temporal_rigs_ssl",
}


def register_training_method(spec: TrainingMethodSpec) -> TrainingMethodSpec:
    existing = _METHODS.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"Training method {spec.name!r} is already registered as {existing}; "
            f"cannot replace it with {spec}."
        )
    _METHODS[spec.name] = spec
    return spec


def available_training_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


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


def resolve_training_method(cfg: Any = None, *, method_name: str | None = None) -> TrainingMethodSpec:
    resolved_name = str(method_name).strip().lower() if method_name is not None else None
    if not resolved_name and cfg is not None:
        resolved_name = _method_name_from_cfg(cfg)
    if not resolved_name:
        available = ", ".join(available_training_methods())
        raise ValueError(
            "Could not resolve training method. Set cfg.method.name, cfg.training_method, "
            f"or cfg.model_type. Registered methods: [{available}]"
        )

    resolved_name = _MODEL_TYPE_ALIASES.get(resolved_name, resolved_name)
    spec = _METHODS.get(resolved_name)
    if spec is None:
        available = ", ".join(available_training_methods())
        raise KeyError(
            f"Unknown training method {resolved_name!r}. Registered methods: [{available}]"
        )
    return spec


register_training_method(
    TrainingMethodSpec(
        name="contrastive",
        module_path="src.training_methods.contrastive_learning.vicreg_module",
        class_name="VICRegModule",
        default_config="vicreg_vn_molecular.yaml",
        run_post_training_analysis=True,
    )
)
register_training_method(
    TrainingMethodSpec(
        name="temporal_ssl",
        module_path="src.training_methods.temporal_ssl.temporal_ssl_module",
        class_name="TemporalSSLModule",
        default_config="temporal_vicreg_lammps.yaml",
    )
)
register_training_method(
    TrainingMethodSpec(
        name="temporal_rigs_ssl",
        module_path="src.training_methods.temporal_ssl.temporal_rigs_ssl_module",
        class_name="TemporalRIGSSSLModule",
        default_config="temporal_rigs_vicreg_lammps.yaml",
    )
)


__all__ = [
    "TrainingMethodSpec",
    "available_training_methods",
    "register_training_method",
    "resolve_training_method",
]
