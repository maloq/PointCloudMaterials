"""Small, explicit quality/runtime trade-offs for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class AnalysisRuntimeProfile:
    name: str
    lazy_static_dataset_on_cache_hit: bool
    clustering_fit_max_samples: int | None
    snapshot_figure_limit: int | None
    md_num_views: int | None
    raytrace_enabled: bool
    tsne_max_samples: int | None
    equivariance_enabled: bool
    real_md_projection_method: str | None
    directional_line_jepa_enabled: bool | None
    directional_max_directions: int | None
    directional_max_atoms_total: int | None

    @property
    def is_fast(self) -> bool:
        return self.name == "fast"


_DEFAULTS = {
    "full": (None, None, None, True, None, True, None, None, None, None),
    "fast": (None, None, None, False, None, False, "pca", True, None, None),
}
_FIELDS = (
    "clustering_fit_max_samples", "snapshot_figure_limit", "md_num_views",
    "raytrace_enabled", "tsne_max_samples", "equivariance_enabled",
    "real_md_projection_method", "directional_line_jepa_enabled",
    "directional_max_directions", "directional_max_atoms_total",
)


def _positive_or_none(value: object, field: str) -> int | None:
    if value in {None, "", 0, "0"}:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"runtime.{field} must be positive or null, got {value}.")
    return value


def resolve_analysis_runtime_profile(analysis_cfg: DictConfig) -> AnalysisRuntimeProfile:
    name = str(OmegaConf.select(analysis_cfg, "runtime.profile", default="full")).lower()
    if name not in _DEFAULTS:
        raise ValueError(f"runtime.profile must be 'full' or 'fast', got {name!r}.")
    defaults = dict(zip(_FIELDS, _DEFAULTS[name], strict=True))
    raw = OmegaConf.select(analysis_cfg, f"runtime.{name}", default={})
    overrides = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    values = defaults | dict(overrides or {})
    projection = values["real_md_projection_method"]
    projection = None if projection in {None, ""} else str(projection).lower()
    if projection not in {None, "pca", "umap", "tsne"}:
        raise ValueError(f"Invalid real MD projection method {projection!r}.")
    return AnalysisRuntimeProfile(
        name=name,
        lazy_static_dataset_on_cache_hit=bool(values.get("lazy_static_dataset_on_cache_hit", True)),
        clustering_fit_max_samples=_positive_or_none(values["clustering_fit_max_samples"], "clustering_fit_max_samples"),
        snapshot_figure_limit=_positive_or_none(values["snapshot_figure_limit"], "snapshot_figure_limit"),
        md_num_views=_positive_or_none(values["md_num_views"], "md_num_views"),
        raytrace_enabled=bool(values["raytrace_enabled"]),
        tsne_max_samples=_positive_or_none(values["tsne_max_samples"], "tsne_max_samples"),
        equivariance_enabled=bool(values["equivariance_enabled"]),
        real_md_projection_method=projection,
        directional_line_jepa_enabled=(
            None if values["directional_line_jepa_enabled"] is None
            else bool(values["directional_line_jepa_enabled"])
        ),
        directional_max_directions=_positive_or_none(
            values["directional_max_directions"], "directional_max_directions"
        ),
        directional_max_atoms_total=_positive_or_none(
            values["directional_max_atoms_total"], "directional_max_atoms_total"
        ),
    )


def subsample_clustering_reference(
    latents: np.ndarray,
    phases: np.ndarray,
    *,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    latents = np.asarray(latents, dtype=np.float32)
    phases = np.asarray(phases, dtype=int).reshape(-1)
    if max_samples is None or len(latents) <= max_samples:
        return latents, phases, None
    indices = np.sort(
        np.random.default_rng(random_state).choice(len(latents), max_samples, replace=False)
    ).astype(np.int64)
    sampled_phases = phases[indices] if len(phases) == len(latents) else np.empty(0, dtype=int)
    return latents[indices], sampled_phases, indices


def select_evenly_spaced_names(names: list[str], limit: int | None) -> list[str]:
    names = [str(name) for name in names]
    if limit is None or len(names) <= limit:
        return names
    return [names[index] for index in np.linspace(0, len(names) - 1, limit, dtype=int)]


__all__ = [
    "AnalysisRuntimeProfile", "resolve_analysis_runtime_profile",
    "select_evenly_spaced_names", "subsample_clustering_reference",
]
