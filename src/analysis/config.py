from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils.model_utils import resolve_config_path


def _resolve_project_root() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(
        "Could not resolve project root from analysis config module path. "
        f"Expected an ancestor of {this_file} containing both 'src' and 'configs'."
    )


PROJECT_ROOT = _resolve_project_root()
DEFAULT_ANALYSIS_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "analysis" / "checkpoint_analysis.yaml"
)


@dataclass(frozen=True)
class RunSettings:
    checkpoint_path: str
    output_dir: Path
    cuda_device: int


@dataclass(frozen=True)
class InputSettings:
    real_data_files: list[str] | None
    dataloader_num_workers: int
    inference_batch_size: int | None
    max_batches_latent: int | None
    max_samples_total: int | None


@dataclass(frozen=True)
class ClusteringFitSettings:
    enabled: bool
    data_config_path: str | None
    input_settings: InputSettings
    cache_enabled: bool
    cache_force_recompute: bool
    cache_file: str


@dataclass(frozen=True)
class HDBSCANSettings:
    enabled: bool
    fit_fraction: float
    max_fit_samples: int
    target_k_min: int
    target_k_max: int
    min_samples: int | None
    min_samples_candidates: list[int] | None
    cluster_selection_epsilon: float
    cluster_selection_method: str
    min_cluster_size_candidates: list[int] | None
    refit_full_data: bool


@dataclass(frozen=True)
class DynamicMotifRenderSettings:
    heatmaps: bool
    timelines: bool
    representatives: bool
    event_gallery: bool
    sankey: bool


@dataclass(frozen=True)
class DynamicMotifFieldSettings:
    enabled: bool
    top_neighbors: int


@dataclass(frozen=True)
class DynamicMotifSettings:
    enabled: bool
    export_per_sample_arrays: bool
    use_model_outputs: bool
    stable_k: int | None
    bridge_k: int | None
    representative_samples_per_motif: int
    transition_top_k: int
    transition_snapshot_flow_count: int
    bridge_min_support: int
    dwell_min_length: int
    recurrence_max_gap: int
    render: DynamicMotifRenderSettings
    field: DynamicMotifFieldSettings


@dataclass(frozen=True)
class AnalysisSettings:
    primary_k: int
    tsne_max_samples: int
    tsne_n_iter: int
    interactive_max_points: int | None
    cluster_method: str
    cluster_compare_methods: list[str]
    cluster_l2_normalize: bool
    cluster_standardize: bool
    cluster_pca_var: float
    cluster_pca_max_components: int
    cluster_k_values: list[int]
    data_overlap_fraction: float
    md_overlap_fraction: float
    md_use_all_points: bool
    progress_every_batches: int
    inference_cache_enabled: bool
    inference_cache_force_recompute: bool
    inference_cache_file: str
    seed_base: int
    cluster_fit: ClusteringFitSettings | None
    hdbscan: HDBSCANSettings
    dynamic_motif: DynamicMotifSettings


@dataclass(frozen=True)
class FigureSetSettings:
    enabled: bool
    figure_only: bool
    k: int
    md_max_points: int | None
    md_point_size: float
    md_alpha: float
    md_halo_scale: float
    md_halo_alpha: float
    md_saturation_boost: float
    md_view_elev: float
    md_view_azim: float
    visible_cluster_sets: list[list[int]] | None
    cluster_color_assignment: dict[int, int | str] | None
    profile_point_scale_enabled: bool
    icl_enabled: bool
    icl_k_min: int
    icl_k_max: int
    icl_max_samples: int | None
    icl_covariance: str
    representative_points: int
    representative_orientation: str
    representative_view_elev: float
    representative_view_azim: float
    representative_projection: str
    representative_ptm_enabled: bool
    representative_cna_enabled: bool
    representative_cna_max_signatures: int
    representative_center_atom_tolerance: float
    representative_shell_min_neighbors: int
    representative_shell_max_neighbors: int
    real_md_profile_target_points: int
    raytrace_enabled: bool
    raytrace_kwargs: dict[str, Any]

    def build_run_kwargs(
        self,
        *,
        dataset: Any,
        latents: np.ndarray,
        coords: np.ndarray,
        point_scale: float,
        random_state: int,
        l2_normalize: bool,
        standardize: bool,
        pca_variance: float | None,
        pca_max_components: int,
    ) -> dict[str, Any]:
        return {
            "dataset": dataset,
            "latents": latents,
            "coords": coords,
            "k_value": int(self.k),
            "point_scale": float(point_scale),
            "l2_normalize": bool(l2_normalize),
            "standardize": bool(standardize),
            "pca_variance": pca_variance,
            "pca_max_components": int(pca_max_components),
            "md_max_points": self.md_max_points,
            "icl_enabled": bool(self.icl_enabled),
            "icl_k_min": int(self.icl_k_min),
            "icl_k_max": int(self.icl_k_max),
            "icl_max_samples": self.icl_max_samples,
            "icl_covariance_type": str(self.icl_covariance),
            "representative_points": int(self.representative_points),
            "md_point_size": float(self.md_point_size),
            "md_point_alpha": float(self.md_alpha),
            "md_halo_scale": float(self.md_halo_scale),
            "md_halo_alpha": float(self.md_halo_alpha),
            "md_saturation_boost": float(self.md_saturation_boost),
            "md_view_elev": float(self.md_view_elev),
            "md_view_azim": float(self.md_view_azim),
            "representative_orientation_method": str(self.representative_orientation),
            "representative_view_elev": float(self.representative_view_elev),
            "representative_view_azim": float(self.representative_view_azim),
            "representative_projection": str(self.representative_projection),
            "representative_ptm_enabled": bool(self.representative_ptm_enabled),
            "representative_cna_enabled": bool(self.representative_cna_enabled),
            "representative_cna_max_signatures": int(self.representative_cna_max_signatures),
            "representative_center_atom_tolerance": float(
                self.representative_center_atom_tolerance
            ),
            "representative_shell_min_neighbors": int(
                self.representative_shell_min_neighbors
            ),
            "representative_shell_max_neighbors": int(
                self.representative_shell_max_neighbors
            ),
            "visible_cluster_sets": self.visible_cluster_sets,
            "cluster_color_assignment": self.cluster_color_assignment,
            "random_state": int(random_state),
            "raytrace_render_enabled": bool(self.raytrace_enabled),
            **self.raytrace_kwargs,
        }


def _as_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in list(value)]


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_select(cfg: Any, key: str, *, default: Any = None) -> Any:
    if cfg is None:
        return default
    return OmegaConf.select(cfg, key, default=default)


def _as_list_of_int(value: Any, *, field_name: str = "value") -> list[int] | None:
    if value is None:
        return None
    values = list(value)
    if not values:
        return None
    return [int(v) for v in values]


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _validate_overlap_fraction(value: Any) -> float:
    overlap = float(value)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap}.")
    return overlap


def _resolve_optional_cluster_k(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 2:
        raise ValueError(f"{field_name} must be >= 2, got {resolved}.")
    return resolved


def _resolve_input_path(
    path: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    expanded = Path(os.path.expanduser(path))
    if expanded.is_absolute():
        return expanded
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / expanded)
    candidates.append(Path(os.getcwd()) / expanded)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_override_config_file(
    path: str,
    *,
    field_name: str,
) -> DictConfig:
    resolved = _resolve_input_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    loaded_cfg = OmegaConf.load(resolved)
    if not isinstance(loaded_cfg, DictConfig):
        raise TypeError(
            f"{field_name} must load to a DictConfig, got {type(loaded_cfg)!r} from {resolved}."
        )
    return loaded_cfg


def _normalize_visible_cluster_sets(value: Any) -> list[list[int]] | None:
    raw = _to_plain(value)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError(
            "figure_set.visible_cluster_sets must be a list of cluster-ID lists, "
            f"got {type(raw)!r}."
        )
    normalized: list[list[int]] = []
    for set_idx, cluster_set in enumerate(raw):
        if isinstance(cluster_set, str):
            tokens = [token.strip() for token in cluster_set.split(",") if token.strip()]
        else:
            tokens = list(cluster_set)
        if not tokens:
            raise ValueError(
                "figure_set.visible_cluster_sets entries must be non-empty. "
                f"Invalid entry at index {set_idx}: {cluster_set!r}."
            )
        normalized.append([int(v) for v in tokens])
    return normalized


def _parse_color_value(value: Any) -> int | str:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text.lstrip("+-").isdigit():
        return int(text)
    return text


def _normalize_cluster_color_assignment(
    value: Any,
    *,
    field_name: str,
) -> dict[int, int | str] | None:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {int(k): _parse_color_value(v) for k, v in value.items()} or None
    entries = [value] if isinstance(value, str) else list(value)
    result: dict[int, int | str] = {}
    for entry in entries:
        entry = str(entry).strip()
        if "=" not in entry:
            raise ValueError(f"{field_name}: expected CLUSTER=VALUE syntax, got {entry!r}")
        cluster_part, value_part = entry.split("=", 1)
        result[int(cluster_part.strip())] = _parse_color_value(value_part.strip())
    return result or None


def _load_cluster_color_assignment_file(
    path: str,
    *,
    base_dir: Path | None = None,
) -> dict[int, int | str]:
    resolved = _resolve_input_path(path, base_dir=base_dir)
    with resolved.open("r") as handle:
        payload = json.load(handle)
    raw_assignment = payload.get("assignment", payload) if isinstance(payload, dict) else payload
    assignment = _normalize_cluster_color_assignment(
        raw_assignment,
        field_name=f"cluster_color_assignment_file({resolved})",
    )
    if assignment is None:
        raise ValueError(f"No assignments found in {resolved}")
    return assignment


def _merge_cluster_color_assignments(
    *assignments: dict[int, int | str] | None,
) -> dict[int, int | str] | None:
    merged: dict[int, int | str] = {}
    for assignment in assignments:
        if assignment:
            merged.update(assignment)
    return merged or None


def _resolve_run_settings(
    analysis_cfg: DictConfig,
    *,
    checkpoint_path_override: str | None,
    output_dir_override: str | None,
    cuda_device_override: int | None,
) -> RunSettings:
    checkpoint_path_raw = (
        checkpoint_path_override
        if checkpoint_path_override is not None
        else OmegaConf.select(analysis_cfg, "checkpoint.path", default=None)
    )
    if checkpoint_path_raw is None or str(checkpoint_path_raw).strip() == "":
        raise ValueError(
            "Missing checkpoint path. Set checkpoint.path in "
            f"{DEFAULT_ANALYSIS_CONFIG_PATH} or pass a runtime override."
        )
    checkpoint_path = str(_resolve_input_path(str(checkpoint_path_raw))).strip()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir_raw = (
        output_dir_override
        if output_dir_override is not None
        else OmegaConf.select(analysis_cfg, "checkpoint.output_dir", default=None)
    )
    output_dir = (
        Path(checkpoint_path).resolve().parent / "analysis"
        if output_dir_raw is None or str(output_dir_raw).strip() == ""
        else _resolve_input_path(str(output_dir_raw))
    )
    cuda_device = (
        int(cuda_device_override)
        if cuda_device_override is not None
        else int(OmegaConf.select(analysis_cfg, "checkpoint.cuda_device", default=0))
    )
    return RunSettings(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        output_dir=Path(output_dir).resolve(),
        cuda_device=int(cuda_device),
    )


def _resolve_input_settings(analysis_cfg: DictConfig) -> InputSettings:
    return InputSettings(
        real_data_files=_as_list_of_str(
            OmegaConf.select(analysis_cfg, "inputs.real_data_files", default=None)
        ),
        dataloader_num_workers=int(
            OmegaConf.select(analysis_cfg, "inputs.dataloader_num_workers", default=4)
        ),
        inference_batch_size=_positive_int_or_none(
            OmegaConf.select(analysis_cfg, "inputs.inference_batch_size", default=None)
        ),
        max_batches_latent=_positive_int_or_none(
            OmegaConf.select(analysis_cfg, "inputs.max_batches_latent", default=None)
        ),
        max_samples_total=_positive_int_or_none(
            OmegaConf.select(analysis_cfg, "inputs.max_samples_total", default=None)
        ),
    )


def _resolve_clustering_fit_settings(
    analysis_cfg: DictConfig,
) -> ClusteringFitSettings | None:
    fit_cfg = OmegaConf.select(analysis_cfg, "clustering.fit_inputs", default=None)
    enabled = bool(_cfg_select(fit_cfg, "enabled", default=False))
    if not enabled:
        return None

    if bool(_cfg_select(fit_cfg, "temporal_real.enabled", default=False)):
        raise ValueError(
            "clustering.fit_inputs.temporal_real is not supported. "
            "Provide a static/synthetic fit dataset through "
            "clustering.fit_inputs.data_config and optional "
            "clustering.fit_inputs.real_data_files."
        )

    parent_inputs = _resolve_input_settings(analysis_cfg)
    data_config_path = _cfg_select(
        fit_cfg,
        "data_config",
        default=OmegaConf.select(analysis_cfg, "inputs.data_config", default=None),
    )
    cache_cfg = _cfg_select(fit_cfg, "cache", default=None)
    cache_file = str(
        _cfg_select(cache_cfg, "file", default="clustering_fit_inference_cache.npz")
    ).strip()
    if cache_file == "":
        raise ValueError("clustering.fit_inputs.cache.file must be a non-empty file name.")

    return ClusteringFitSettings(
        enabled=True,
        data_config_path=(
            None if data_config_path is None else str(data_config_path).strip() or None
        ),
        input_settings=InputSettings(
            real_data_files=_as_list_of_str(
                _cfg_select(
                    fit_cfg,
                    "real_data_files",
                    default=parent_inputs.real_data_files,
                )
            ),
            dataloader_num_workers=int(
                _cfg_select(
                    fit_cfg,
                    "dataloader_num_workers",
                    default=parent_inputs.dataloader_num_workers,
                )
            ),
            inference_batch_size=_positive_int_or_none(
                _cfg_select(
                    fit_cfg,
                    "inference_batch_size",
                    default=parent_inputs.inference_batch_size,
                )
            ),
            max_batches_latent=_positive_int_or_none(
                _cfg_select(
                    fit_cfg,
                    "max_batches_latent",
                    default=parent_inputs.max_batches_latent,
                )
            ),
            max_samples_total=_positive_int_or_none(
                _cfg_select(
                    fit_cfg,
                    "max_samples_total",
                    default=parent_inputs.max_samples_total,
                )
            ),
        ),
        cache_enabled=bool(_cfg_select(cache_cfg, "enabled", default=True)),
        cache_force_recompute=bool(_cfg_select(cache_cfg, "force_recompute", default=False)),
        cache_file=cache_file,
    )


def _resolve_analysis_files(
    model_cfg: DictConfig,
    input_settings: InputSettings,
) -> list[str] | None:
    if model_cfg.data.kind != "real":
        return None
    if input_settings.real_data_files:
        return input_settings.real_data_files
    data_files = _as_list_of_str(OmegaConf.select(model_cfg, "data.data_files", default=None))
    if not data_files:
        raise ValueError(
            "Cannot resolve analysis data files: data.data_files is missing or empty, "
            "and inputs.real_data_files was not provided in the analysis config."
        )
    return data_files


def _resolve_dynamic_motif_settings(analysis_cfg: DictConfig) -> DynamicMotifSettings:
    dynamic_cfg = OmegaConf.select(analysis_cfg, "dynamic_motif", default=None)
    render_cfg = _cfg_select(dynamic_cfg, "render", default=None)
    field_cfg = _cfg_select(dynamic_cfg, "field", default=None)
    return DynamicMotifSettings(
        enabled=bool(_cfg_select(dynamic_cfg, "enabled", default=False)),
        export_per_sample_arrays=bool(_cfg_select(dynamic_cfg, "export_per_sample_arrays", default=True)),
        use_model_outputs=bool(_cfg_select(dynamic_cfg, "use_model_outputs", default=True)),
        stable_k=_resolve_optional_cluster_k(
            _cfg_select(dynamic_cfg, "stable_k", default=None),
            field_name="dynamic_motif.stable_k",
        ),
        bridge_k=_resolve_optional_cluster_k(
            _cfg_select(dynamic_cfg, "bridge_k", default=None),
            field_name="dynamic_motif.bridge_k",
        ),
        representative_samples_per_motif=int(
            _cfg_select(dynamic_cfg, "representative_samples_per_motif", default=12)
        ),
        transition_top_k=int(
            _cfg_select(dynamic_cfg, "transition_top_k", default=20)
        ),
        transition_snapshot_flow_count=max(
            0,
            int(_cfg_select(dynamic_cfg, "transition_snapshot_flow_count", default=0)),
        ),
        bridge_min_support=int(
            _cfg_select(dynamic_cfg, "bridge_min_support", default=50)
        ),
        dwell_min_length=max(
            1,
            int(_cfg_select(dynamic_cfg, "dwell_min_length", default=1)),
        ),
        recurrence_max_gap=max(
            1,
            int(_cfg_select(dynamic_cfg, "recurrence_max_gap", default=64)),
        ),
        render=DynamicMotifRenderSettings(
            heatmaps=bool(_cfg_select(render_cfg, "heatmaps", default=True)),
            timelines=bool(_cfg_select(render_cfg, "timelines", default=True)),
            representatives=bool(_cfg_select(render_cfg, "representatives", default=True)),
            event_gallery=bool(_cfg_select(render_cfg, "event_gallery", default=True)),
            sankey=bool(_cfg_select(render_cfg, "sankey", default=True)),
        ),
        field=DynamicMotifFieldSettings(
            enabled=bool(_cfg_select(field_cfg, "enabled", default=False)),
            top_neighbors=max(
                1,
                int(_cfg_select(field_cfg, "top_neighbors", default=8)),
            ),
        ),
    )


def _resolve_analysis_settings(
    analysis_cfg: DictConfig,
    model_cfg: DictConfig,
) -> AnalysisSettings:
    clustering_cfg = OmegaConf.select(analysis_cfg, "clustering", default=None)
    figure_cfg = OmegaConf.select(analysis_cfg, "figure_set", default=None)
    md_cfg = OmegaConf.select(analysis_cfg, "md", default=None)
    tsne_cfg = OmegaConf.select(analysis_cfg, "tsne", default=None)
    cache_cfg = OmegaConf.select(analysis_cfg, "cache", default=None)
    runtime_cfg = OmegaConf.select(analysis_cfg, "runtime", default=None)
    hdbscan_cfg = OmegaConf.select(analysis_cfg, "clustering.hdbscan", default=None)

    primary_k = _resolve_optional_cluster_k(
        _cfg_select(clustering_cfg, "primary_k", default=None),
        field_name="clustering.primary_k",
    )
    figure_set_k = _resolve_optional_cluster_k(
        _cfg_select(figure_cfg, "k", default=None),
        field_name="figure_set.k",
    )
    real_md_selected_k = _resolve_optional_cluster_k(
        OmegaConf.select(analysis_cfg, "real_md.selected_k", default=None),
        field_name="real_md.selected_k",
    )
    cluster_k_values_raw = _as_list_of_int(
        _cfg_select(clustering_cfg, "k_values", default=None),
        field_name="clustering.k_values",
    )
    cluster_k_values = (
        []
        if cluster_k_values_raw is None
        else list(dict.fromkeys(int(k) for k in cluster_k_values_raw if int(k) >= 2))
    )

    legacy_selected_k_candidates = {
        "figure_set.k": figure_set_k,
        "real_md.selected_k": real_md_selected_k,
    }
    if primary_k is None:
        for field_name, field_value in legacy_selected_k_candidates.items():
            if field_value is not None:
                primary_k = int(field_value)
                break
    if primary_k is None and cluster_k_values:
        primary_k = int(cluster_k_values[0])
    if primary_k is None:
        raise ValueError(
            "Could not resolve a unified clustering k. Set clustering.primary_k explicitly, "
            "or provide one of the legacy fields figure_set.k / real_md.selected_k, "
            "or provide clustering.k_values with at least one integer >= 2."
        )

    for field_name, field_value in legacy_selected_k_candidates.items():
        if field_value is not None and int(field_value) != int(primary_k):
            raise ValueError(
                f"{field_name}={int(field_value)} conflicts with clustering.primary_k="
                f"{int(primary_k)}. Use clustering.primary_k as the single selected "
                "clustering k and leave per-section overrides null."
            )

    if cluster_k_values_raw is None:
        cluster_k_values = [int(primary_k)]
    elif not cluster_k_values:
        raise ValueError(
            "clustering.k_values was provided, but it does not contain any integers >= 2."
        )
    cluster_k_values = [int(primary_k)] + [
        int(k) for k in cluster_k_values if int(k) != int(primary_k)
    ]

    data_overlap_fraction = _validate_overlap_fraction(
        getattr(model_cfg.data, "overlap_fraction", 0.0)
    )
    md_overlap_fraction_raw = _cfg_select(md_cfg, "overlap_fraction", default=None)
    md_overlap_fraction = (
        min(0.95, data_overlap_fraction + float(_cfg_select(md_cfg, "overlap_boost", default=0.25)))
        if md_overlap_fraction_raw is None
        else _validate_overlap_fraction(md_overlap_fraction_raw)
    )
    model_cfg.data.overlap_fraction = float(md_overlap_fraction)

    hdbscan_min_samples_candidates = _as_list_of_int(
        _cfg_select(hdbscan_cfg, "min_samples_candidates", default=None),
        field_name="clustering.hdbscan.min_samples_candidates",
    )
    hdbscan_min_cluster_size_candidates = _as_list_of_int(
        _cfg_select(hdbscan_cfg, "min_cluster_size_candidates", default=None),
        field_name="clustering.hdbscan.min_cluster_size_candidates",
    )
    hdbscan = HDBSCANSettings(
        enabled=bool(_cfg_select(hdbscan_cfg, "enabled", default=True)),
        fit_fraction=float(_cfg_select(hdbscan_cfg, "fit_fraction", default=0.75)),
        max_fit_samples=int(_cfg_select(hdbscan_cfg, "max_fit_samples", default=50000)),
        target_k_min=int(_cfg_select(hdbscan_cfg, "target_k_min", default=5)),
        target_k_max=int(_cfg_select(hdbscan_cfg, "target_k_max", default=6)),
        min_samples=_positive_int_or_none(
            _cfg_select(hdbscan_cfg, "min_samples", default=None)
        ),
        min_samples_candidates=hdbscan_min_samples_candidates,
        cluster_selection_epsilon=float(
            _cfg_select(hdbscan_cfg, "cluster_selection_epsilon", default=0.0)
        ),
        cluster_selection_method=str(
            _cfg_select(hdbscan_cfg, "cluster_selection_method", default="auto")
        ).lower(),
        min_cluster_size_candidates=hdbscan_min_cluster_size_candidates,
        refit_full_data=bool(_cfg_select(hdbscan_cfg, "refit_full_data", default=True)),
    )
    dynamic_motif = _resolve_dynamic_motif_settings(analysis_cfg)

    inference_cache_file = str(
        OmegaConf.select(cache_cfg, "file", default="analysis_inference_cache.npz")
    ).strip()
    if inference_cache_file == "":
        raise ValueError("cache.file must be a non-empty file name.")

    compare_methods_raw = _as_list_of_str(
        _cfg_select(clustering_cfg, "compare_methods", default=None)
    ) or []
    compare_methods: list[str] = []
    seen_compare_methods: set[str] = set()
    for method_name in compare_methods_raw:
        normalized_method = str(method_name).strip().lower()
        if normalized_method == "":
            raise ValueError("clustering.compare_methods entries must be non-empty strings.")
        if normalized_method in seen_compare_methods:
            continue
        seen_compare_methods.add(normalized_method)
        compare_methods.append(normalized_method)

    return AnalysisSettings(
        primary_k=int(primary_k),
        tsne_max_samples=int(OmegaConf.select(tsne_cfg, "max_samples", default=8000)),
        tsne_n_iter=int(OmegaConf.select(tsne_cfg, "n_iter", default=1000)),
        interactive_max_points=_positive_int_or_none(
            _cfg_select(md_cfg, "interactive_max_points", default=None)
        ),
        cluster_method=str(
            _cfg_select(clustering_cfg, "method", default="spherical_kmeans")
        ).lower(),
        cluster_compare_methods=compare_methods,
        cluster_l2_normalize=bool(_cfg_select(clustering_cfg, "l2_normalize", default=True)),
        cluster_standardize=bool(_cfg_select(clustering_cfg, "standardize", default=True)),
        cluster_pca_var=float(_cfg_select(clustering_cfg, "pca_variance", default=0.98)),
        cluster_pca_max_components=int(
            _cfg_select(clustering_cfg, "pca_max_components", default=32)
        ),
        cluster_k_values=cluster_k_values,
        data_overlap_fraction=data_overlap_fraction,
        md_overlap_fraction=float(md_overlap_fraction),
        md_use_all_points=bool(_cfg_select(md_cfg, "use_all_points", default=True)),
        progress_every_batches=int(
            _cfg_select(runtime_cfg, "progress_every_batches", default=25)
        ),
        inference_cache_enabled=bool(_cfg_select(cache_cfg, "enabled", default=True)),
        inference_cache_force_recompute=bool(_cfg_select(cache_cfg, "force_recompute", default=False)),
        inference_cache_file=inference_cache_file,
        seed_base=int(_cfg_select(runtime_cfg, "seed_base", default=123)),
        cluster_fit=_resolve_clustering_fit_settings(analysis_cfg),
        hdbscan=hdbscan,
        dynamic_motif=dynamic_motif,
    )


def _resolve_figure_set_settings(
    analysis_cfg: DictConfig,
    model_cfg: DictConfig,
    *,
    out_dir: Path,
    primary_k: int,
) -> FigureSetSettings:
    figure_cfg = OmegaConf.select(analysis_cfg, "figure_set", default=None)
    figure_md_cfg = OmegaConf.select(analysis_cfg, "figure_set.md", default=None)
    icl_cfg = OmegaConf.select(analysis_cfg, "figure_set.icl", default=None)
    rep_cfg = OmegaConf.select(analysis_cfg, "figure_set.representatives", default=None)
    raytrace_cfg = OmegaConf.select(analysis_cfg, "figure_set.raytrace", default=None)
    real_md_profile_cfg = OmegaConf.select(analysis_cfg, "real_md.profiles", default=None)

    figure_only = bool(_cfg_select(figure_cfg, "figure_only", default=False))
    enabled = bool(_cfg_select(figure_cfg, "enabled", default=True)) or figure_only
    cluster_k = _resolve_optional_cluster_k(
        _cfg_select(figure_cfg, "k", default=None),
        field_name="figure_set.k",
    )
    if cluster_k is not None and int(cluster_k) != int(primary_k):
        raise ValueError(
            f"figure_set.k={int(cluster_k)} conflicts with clustering.primary_k="
            f"{int(primary_k)}. Use clustering.primary_k as the single selected "
            "clustering k and keep figure_set.k unset."
        )
    cluster_k = int(primary_k)

    cluster_color_assignment_cfg = _normalize_cluster_color_assignment(
        _cfg_select(figure_cfg, "color_assignment", default=None),
        field_name="figure_set.color_assignment",
    )
    cluster_color_assignment_file_cfg = _cfg_select(
        figure_cfg,
        "color_assignment_file",
        default=None,
    )
    cluster_color_assignment_cfg_file = (
        _load_cluster_color_assignment_file(str(cluster_color_assignment_file_cfg), base_dir=out_dir)
        if cluster_color_assignment_file_cfg is not None
        else None
    )

    representative_points_default = int(
        getattr(model_cfg.data, "model_points", getattr(model_cfg.data, "num_points", 48))
    )
    representative_points_cfg = _cfg_select(rep_cfg, "points", default=None)
    representative_points = max(
        16,
        representative_points_default
        if representative_points_cfg is None
        else int(representative_points_cfg),
    )
    representative_orientation = str(
        _cfg_select(rep_cfg, "orientation", default="pca")
    ).strip().lower()
    if representative_orientation not in {"pca", "none"}:
        raise ValueError(
            "figure_set.representatives.orientation must be one of ['pca', 'none'], "
            f"got {representative_orientation!r}."
        )

    representative_cna_max_signatures = int(
        _cfg_select(rep_cfg, "cna_max_signatures", default=5)
    )
    representative_shell_min_neighbors = int(
        _cfg_select(rep_cfg, "shell_min_neighbors", default=8)
    )
    representative_shell_max_neighbors = int(
        _cfg_select(rep_cfg, "shell_max_neighbors", default=24)
    )
    if representative_cna_max_signatures <= 0:
        raise ValueError(
            "figure_set.representatives.cna_max_signatures must be > 0, "
            f"got {representative_cna_max_signatures}."
        )
    if representative_shell_min_neighbors < 2:
        raise ValueError(
            "figure_set.representatives.shell_min_neighbors must be >= 2, "
            f"got {representative_shell_min_neighbors}."
        )
    if representative_shell_max_neighbors <= representative_shell_min_neighbors:
        raise ValueError(
            "figure_set.representatives.shell_max_neighbors must exceed "
            "figure_set.representatives.shell_min_neighbors, got "
            f"{representative_shell_max_neighbors} <= {representative_shell_min_neighbors}."
        )

    return FigureSetSettings(
        enabled=enabled,
        figure_only=figure_only,
        k=cluster_k,
        md_max_points=_positive_int_or_none(
            _cfg_select(figure_md_cfg, "max_points", default=None)
        ),
        md_point_size=float(_cfg_select(figure_md_cfg, "point_size", default=5.6)),
        md_alpha=float(_cfg_select(figure_md_cfg, "alpha", default=0.62)),
        md_halo_scale=float(_cfg_select(figure_md_cfg, "halo_scale", default=1.0)),
        md_halo_alpha=float(_cfg_select(figure_md_cfg, "halo_alpha", default=0.0)),
        md_saturation_boost=float(
            _cfg_select(figure_md_cfg, "saturation_boost", default=1.18)
        ),
        md_view_elev=float(_cfg_select(figure_md_cfg, "view_elev", default=24.0)),
        md_view_azim=float(_cfg_select(figure_md_cfg, "view_azim", default=35.0)),
        visible_cluster_sets=_normalize_visible_cluster_sets(
            _cfg_select(figure_cfg, "visible_cluster_sets", default=None)
        ),
        cluster_color_assignment=_merge_cluster_color_assignments(
            cluster_color_assignment_cfg_file,
            cluster_color_assignment_cfg,
        ),
        profile_point_scale_enabled=bool(
            _cfg_select(figure_cfg, "profile_point_scale_enabled", default=False)
        ),
        icl_enabled=bool(_cfg_select(icl_cfg, "enabled", default=False)),
        icl_k_min=int(_cfg_select(icl_cfg, "k_min", default=2)),
        icl_k_max=int(_cfg_select(icl_cfg, "k_max", default=20)),
        icl_max_samples=_positive_int_or_none(
            _cfg_select(icl_cfg, "max_samples", default=20000)
        ),
        icl_covariance=str(_cfg_select(icl_cfg, "covariance", default="diag")).lower(),
        representative_points=representative_points,
        representative_orientation=representative_orientation,
        representative_view_elev=float(_cfg_select(rep_cfg, "view_elev", default=22.0)),
        representative_view_azim=float(_cfg_select(rep_cfg, "view_azim", default=38.0)),
        representative_projection=str(
            _cfg_select(rep_cfg, "projection", default="ortho")
        ).strip().lower(),
        representative_ptm_enabled=bool(
            _cfg_select(rep_cfg, "ptm_enabled", default=False)
        ),
        representative_cna_enabled=bool(
            _cfg_select(rep_cfg, "cna_enabled", default=False)
        ),
        representative_cna_max_signatures=representative_cna_max_signatures,
        representative_center_atom_tolerance=float(
            _cfg_select(rep_cfg, "center_atom_tolerance", default=1e-6)
        ),
        representative_shell_min_neighbors=representative_shell_min_neighbors,
        representative_shell_max_neighbors=representative_shell_max_neighbors,
        real_md_profile_target_points=int(
            _cfg_select(
                real_md_profile_cfg,
                "target_points",
                default=max(
                    32,
                    int(
                        getattr(
                            model_cfg.data,
                            "model_points",
                            getattr(model_cfg.data, "num_points", 64),
                        )
                    ),
                ),
            )
        ),
        raytrace_enabled=bool(_cfg_select(raytrace_cfg, "enabled", default=False)),
        raytrace_kwargs={
            "raytrace_blender_executable": str(
                _cfg_select(raytrace_cfg, "blender_executable", default="blender")
            ).strip(),
            "raytrace_render_resolution": int(
                _cfg_select(raytrace_cfg, "resolution", default=1600)
            ),
            "raytrace_render_max_points": _positive_int_or_none(
                _cfg_select(raytrace_cfg, "max_points", default=None)
            ),
            "raytrace_render_samples": int(
                _cfg_select(raytrace_cfg, "samples", default=64)
            ),
            "raytrace_render_projection": str(
                _cfg_select(raytrace_cfg, "projection", default="perspective")
            ).strip().lower(),
            "raytrace_render_fov_deg": float(
                _cfg_select(raytrace_cfg, "fov_deg", default=34.0)
            ),
            "raytrace_render_camera_distance_factor": float(
                _cfg_select(raytrace_cfg, "camera_distance_factor", default=2.8)
            ),
            "raytrace_render_sphere_radius_fraction": float(
                _cfg_select(raytrace_cfg, "sphere_radius_fraction", default=0.0105)
            ),
            "raytrace_render_timeout_sec": int(
                _cfg_select(raytrace_cfg, "timeout_sec", default=1200)
            ),
            "raytrace_render_use_gpu": bool(
                _cfg_select(raytrace_cfg, "use_gpu", default=False)
            ),
            "raytrace_parallel_views": bool(
                _cfg_select(raytrace_cfg, "parallel_views", default=False)
            ),
            "raytrace_parallel_max_workers": _positive_int_or_none(
                _cfg_select(raytrace_cfg, "parallel_max_workers", default=None)
            ),
        },
    )


def _print_resolved_analysis_settings(
    analysis_settings: AnalysisSettings,
    figure_settings: FigureSetSettings,
) -> None:
    print(
        "Unified selected clustering k: "
        f"k={analysis_settings.primary_k} "
        "(driven by clustering.primary_k; figure_set.k and real_md.selected_k inherit it)"
    )
    print(f"t-SNE sample cap: {analysis_settings.tsne_max_samples}")
    print(f"Available clustering keys: {analysis_settings.cluster_k_values}")
    print(
        "Clustering backend settings: "
        f"method={analysis_settings.cluster_method}, "
        f"compare_methods={analysis_settings.cluster_compare_methods}, "
        f"l2_normalize={analysis_settings.cluster_l2_normalize}, "
        f"standardize={analysis_settings.cluster_standardize}, "
        f"pca_variance={analysis_settings.cluster_pca_var}, "
        f"pca_max_components={analysis_settings.cluster_pca_max_components}"
    )
    if analysis_settings.cluster_fit is not None:
        print(
            "Clustering fit-transfer settings: "
            f"data_config={analysis_settings.cluster_fit.data_config_path}, "
            f"real_data_files={analysis_settings.cluster_fit.input_settings.real_data_files}, "
            f"cache_enabled={analysis_settings.cluster_fit.cache_enabled}, "
            f"cache_file={analysis_settings.cluster_fit.cache_file}"
        )
    print(
        "MD overlap fraction (analysis): "
        f"{analysis_settings.data_overlap_fraction:.3f} -> {analysis_settings.md_overlap_fraction:.3f}"
    )
    print(
        "HDBSCAN settings: "
        f"fit_fraction={analysis_settings.hdbscan.fit_fraction:.3f}, "
        f"max_fit_samples={analysis_settings.hdbscan.max_fit_samples}, "
        f"target_k=[{analysis_settings.hdbscan.target_k_min}, "
        f"{analysis_settings.hdbscan.target_k_max}], "
        f"selection_method={analysis_settings.hdbscan.cluster_selection_method}, "
        f"refit_full_data={analysis_settings.hdbscan.refit_full_data}"
    )
    print(
        "Fixed-k figure set: "
        f"enabled={figure_settings.enabled}, "
        f"figure_only={figure_settings.figure_only}, "
        f"k={figure_settings.k}, "
        f"visible_sets={figure_settings.visible_cluster_sets or []}, "
        f"md_saturation={figure_settings.md_saturation_boost:.2f}, "
        f"raytrace_enabled={figure_settings.raytrace_enabled}, "
        f"raytrace_projection={figure_settings.raytrace_kwargs['raytrace_render_projection']}, "
        f"raytrace_samples={figure_settings.raytrace_kwargs['raytrace_render_samples']}, "
        f"raytrace_res={figure_settings.raytrace_kwargs['raytrace_render_resolution']}, "
        f"raytrace_max_points={figure_settings.raytrace_kwargs['raytrace_render_max_points']}, "
        f"raytrace_gpu={figure_settings.raytrace_kwargs['raytrace_render_use_gpu']}, "
        f"raytrace_parallel_views={figure_settings.raytrace_kwargs['raytrace_parallel_views']}, "
        "raytrace_parallel_max_workers="
        f"{figure_settings.raytrace_kwargs['raytrace_parallel_max_workers']}, "
        f"icl_enabled={figure_settings.icl_enabled}, "
        f"profile_point_scale_enabled={figure_settings.profile_point_scale_enabled}, "
        f"rep_orientation={figure_settings.representative_orientation}, "
        f"rep_view=({figure_settings.representative_view_elev:.1f},"
        f"{figure_settings.representative_view_azim:.1f}), "
        f"rep_projection={figure_settings.representative_projection}, "
        f"rep_ptm={figure_settings.representative_ptm_enabled}, "
        f"rep_cna={figure_settings.representative_cna_enabled}, "
        "rep_cna_shell="
        f"({figure_settings.representative_shell_min_neighbors},"
        f"{figure_settings.representative_shell_max_neighbors}), "
        "cluster_color_overrides="
        f"{sorted((figure_settings.cluster_color_assignment or {}).items())}"
    )
    print(
        "Dynamic motif analysis: "
        f"enabled={analysis_settings.dynamic_motif.enabled}, "
        f"use_model_outputs={analysis_settings.dynamic_motif.use_model_outputs}, "
        f"stable_k={analysis_settings.dynamic_motif.stable_k}, "
        f"bridge_k={analysis_settings.dynamic_motif.bridge_k}, "
        f"transition_snapshot_flow_count={analysis_settings.dynamic_motif.transition_snapshot_flow_count}, "
        "render="
        f"(heatmaps={analysis_settings.dynamic_motif.render.heatmaps}, "
        f"timelines={analysis_settings.dynamic_motif.render.timelines}, "
        f"representatives={analysis_settings.dynamic_motif.render.representatives}, "
        f"event_gallery={analysis_settings.dynamic_motif.render.event_gallery}, "
        f"sankey={analysis_settings.dynamic_motif.render.sankey}), "
        f"field_enabled={analysis_settings.dynamic_motif.field.enabled}"
    )
def load_checkpoint_training_config(checkpoint_path: str) -> DictConfig:
    config_dir, config_name = resolve_config_path(checkpoint_path)
    config_path = Path(config_dir) / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "Resolved checkpoint config file does not exist: "
            f"{config_path} for checkpoint {checkpoint_path}."
        )
    checkpoint_cfg = OmegaConf.load(config_path)
    if not isinstance(checkpoint_cfg, DictConfig):
        raise TypeError(
            "Checkpoint config must load to a DictConfig, "
            f"got {type(checkpoint_cfg)!r} from {config_path}."
        )
    return checkpoint_cfg


def load_checkpoint_analysis_config(config_path: str | None = None) -> DictConfig:
    resolved_path = (
        DEFAULT_ANALYSIS_CONFIG_PATH
        if config_path is None
        else _resolve_input_path(config_path)
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Analysis config does not exist: {resolved_path}")
    analysis_cfg = OmegaConf.load(resolved_path)
    if not isinstance(analysis_cfg, DictConfig):
        raise TypeError(
            "Analysis config must load to a DictConfig, "
            f"got {type(analysis_cfg)!r} from {resolved_path}."
        )
    return analysis_cfg


def build_runtime_model_config(
    checkpoint_path: str,
    analysis_cfg: DictConfig,
    *,
    data_config_path_override: str | None = None,
) -> DictConfig:
    model_cfg = load_checkpoint_training_config(checkpoint_path)
    data_config_path = (
        data_config_path_override
        if data_config_path_override is not None
        else OmegaConf.select(analysis_cfg, "inputs.data_config", default=None)
    )
    if data_config_path is not None:
        data_cfg = _load_override_config_file(
            str(data_config_path),
            field_name=(
                "inputs.data_config"
                if data_config_path_override is None
                else "clustering.fit_inputs.data_config"
            ),
        )
        override_piece = data_cfg if "data" in data_cfg else OmegaConf.create({"data": data_cfg})
        model_cfg = OmegaConf.merge(model_cfg, override_piece)
        if not isinstance(model_cfg, DictConfig):
            raise TypeError(
                "Merged runtime model config must be a DictConfig, "
                f"got {type(model_cfg)!r}."
            )
    return model_cfg
