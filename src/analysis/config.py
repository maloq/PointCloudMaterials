"""Configuration for the repository's explicit analysis protocols."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils.model_utils import resolve_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_CONFIG_PATH = PROJECT_ROOT / "configs/analysis/static.yaml"


def default_analysis_config_for_checkpoint(checkpoint_path: str) -> Path:
    cfg = load_checkpoint_training_config(checkpoint_path)
    if cfg.data.kind == "relaxed_histories":
        name = (
            "relaxed_histories.yaml"
            if cfg.encoder.name == "PretrainedMACEHistoryGeometry"
            else "static_topology.yaml"
        )
        return DEFAULT_ANALYSIS_CONFIG_PATH.parent / name
    return DEFAULT_ANALYSIS_CONFIG_PATH


@dataclass(frozen=True)
class RunSettings:
    checkpoint_path: str
    output_dir: Path
    cuda_device: int


@dataclass(frozen=True)
class InputSettings:
    static_data_files: list[str] | None
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


@dataclass(frozen=True)
class DynamicMotifSettings:
    enabled: bool
    export_per_sample_arrays: bool
    use_model_outputs: bool
    stable_k: int | None
    bridge_k: int | None
    representative_samples_per_motif: int
    bridge_min_support: int
    dwell_min_length: int
    recurrence_max_gap: int
    transition_snapshot_flow_count: int
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
    k: int
    md_max_points: int | None
    md_point_size: float
    md_alpha: float
    md_halo_scale: float
    md_halo_alpha: float
    md_saturation_boost: float
    md_view_elev: float
    md_view_azim: float
    md_num_views: int
    cluster_color_assignment: dict[int, int | str] | None
    profile_point_scale_enabled: bool
    icl_enabled: bool
    icl_k_min: int
    icl_k_max: int
    icl_max_samples: int | None
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
            "k_value": self.k,
            "point_scale": point_scale,
            "l2_normalize": l2_normalize,
            "standardize": standardize,
            "pca_variance": pca_variance,
            "pca_max_components": pca_max_components,
            "md_max_points": self.md_max_points,
            "icl_enabled": self.icl_enabled,
            "icl_k_min": self.icl_k_min,
            "icl_k_max": self.icl_k_max,
            "icl_max_samples": self.icl_max_samples,
            "representative_points": self.representative_points,
            "md_point_size": self.md_point_size,
            "md_point_alpha": self.md_alpha,
            "md_halo_scale": self.md_halo_scale,
            "md_halo_alpha": self.md_halo_alpha,
            "md_saturation_boost": self.md_saturation_boost,
            "md_view_elev": self.md_view_elev,
            "md_view_azim": self.md_view_azim,
            "md_num_views": self.md_num_views,
            "representative_orientation_method": self.representative_orientation,
            "representative_view_elev": self.representative_view_elev,
            "representative_view_azim": self.representative_view_azim,
            "representative_projection": self.representative_projection,
            "representative_ptm_enabled": self.representative_ptm_enabled,
            "representative_cna_enabled": self.representative_cna_enabled,
            "representative_cna_max_signatures": self.representative_cna_max_signatures,
            "representative_center_atom_tolerance": self.representative_center_atom_tolerance,
            "representative_shell_min_neighbors": self.representative_shell_min_neighbors,
            "representative_shell_max_neighbors": self.representative_shell_max_neighbors,
            "cluster_color_assignment": self.cluster_color_assignment,
            "random_state": random_state,
            "raytrace_render_enabled": self.raytrace_enabled,
            **self.raytrace_kwargs,
        }


def _positive_int_or_none(value):
    return value or None


def _resolve_input_path(path, *, base_dir=None):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base_dir or PROJECT_ROOT) / path


def _resolve_run_settings(
    analysis_cfg,
    *,
    checkpoint_path_override=None,
    output_dir_override=None,
    cuda_device_override=None,
):
    cfg = analysis_cfg.checkpoint
    checkpoint = _resolve_input_path(checkpoint_path_override or cfg.path).resolve()
    output = output_dir_override or cfg.output_dir
    return RunSettings(
        str(checkpoint),
        _resolve_input_path(output).resolve() if output else checkpoint.parent / "analysis",
        cfg.cuda_device if cuda_device_override is None else cuda_device_override,
    )


def _resolve_input_settings(analysis_cfg):
    cfg = analysis_cfg.inputs
    return InputSettings(
        cfg.get("static_data_files"),
        cfg.get("dataloader_num_workers", 4),
        cfg.get("inference_batch_size") or None,
        cfg.get("max_batches_latent") or None,
        cfg.get("max_samples_total") or None,
    )


def _resolve_clustering_fit_settings(analysis_cfg):
    cfg = analysis_cfg.clustering.get("fit_inputs", {})
    if not cfg.get("enabled", False):
        return None
    inputs = asdict(_resolve_input_settings(analysis_cfg))
    inputs.update({key: cfg[key] for key in inputs if key in cfg})
    for key in ("inference_batch_size", "max_batches_latent", "max_samples_total"):
        inputs[key] = inputs[key] or None
    cache = cfg.get("cache", {})
    return ClusteringFitSettings(
        True,
        cfg.get("data_config", analysis_cfg.inputs.get("data_config")),
        InputSettings(**inputs),
        cache.get("enabled", True),
        cache.get("force_recompute", False),
        cache.get("file", "clustering_fit_inference_cache.npz"),
    )


def _resolve_analysis_files(model_cfg, input_settings):
    if model_cfg.data.kind != "static":
        return None
    return list(input_settings.static_data_files or model_cfg.data.data_files)


def _resolve_dynamic_motif_settings(analysis_cfg):
    cfg = analysis_cfg.get("dynamic_motif", {})
    render = cfg.get("render", {})
    return DynamicMotifSettings(
        enabled=cfg.get("enabled", False),
        export_per_sample_arrays=cfg.get("export_per_sample_arrays", True),
        use_model_outputs=cfg.get("use_model_outputs", True),
        stable_k=cfg.get("stable_k"),
        bridge_k=cfg.get("bridge_k"),
        representative_samples_per_motif=cfg.get("representative_samples_per_motif", 12),
        bridge_min_support=cfg.get("bridge_min_support", 50),
        dwell_min_length=cfg.get("dwell_min_length", 1),
        recurrence_max_gap=cfg.get("recurrence_max_gap", 64),
        transition_snapshot_flow_count=cfg.get("transition_snapshot_flow_count", 0),
        render=DynamicMotifRenderSettings(
            **{
                key: render.get(key, True)
                for key in ("heatmaps", "timelines", "representatives", "event_gallery", "sankey")
            }
        ),
        field=DynamicMotifFieldSettings(cfg.get("field", {}).get("enabled", False)),
    )


def _resolve_analysis_settings(analysis_cfg, model_cfg):
    c, md, tsne, cache, runtime = (
        analysis_cfg.get(key, {}) for key in ("clustering", "md", "tsne", "cache", "runtime")
    )
    hdbscan_values = dict(c["hdbscan"])
    for key in ("min_samples", "min_samples_candidates", "min_cluster_size_candidates"):
        hdbscan_values[key] = hdbscan_values[key] or None
    hdbscan = HDBSCANSettings(**hdbscan_values)
    primary_k = c["primary_k"]
    ks = list(dict.fromkeys([primary_k, *(c.get("k_values") or [])]))
    data_overlap = model_cfg.data.get("overlap_fraction", 0.0)
    overlap = md.get("overlap_fraction")
    if overlap is None:
        overlap = min(0.95, data_overlap + md.get("overlap_boost", 0.25))
    model_cfg.data.overlap_fraction = overlap
    return AnalysisSettings(
        primary_k=primary_k,
        tsne_max_samples=tsne.get("max_samples", 8000),
        tsne_n_iter=tsne.get("n_iter", 1000),
        interactive_max_points=md.get("interactive_max_points") or None,
        cluster_method=c.get("method", "spherical_kmeans"),
        cluster_compare_methods=list(dict.fromkeys(c.get("compare_methods") or [])),
        cluster_l2_normalize=c.get("l2_normalize", True),
        cluster_standardize=c.get("standardize", True),
        cluster_pca_var=c.get("pca_variance", 0.98),
        cluster_pca_max_components=c.get("pca_max_components", 32),
        cluster_k_values=ks,
        data_overlap_fraction=data_overlap,
        md_overlap_fraction=overlap,
        md_use_all_points=md.get("use_all_points", True),
        progress_every_batches=runtime.get("progress_every_batches", 25),
        inference_cache_enabled=cache.get("enabled", True),
        inference_cache_force_recompute=cache.get("force_recompute", False),
        inference_cache_file=cache.get("file", "analysis_inference_cache.npz"),
        seed_base=runtime.get("seed_base", 123),
        cluster_fit=_resolve_clustering_fit_settings(analysis_cfg),
        hdbscan=hdbscan,
        dynamic_motif=_resolve_dynamic_motif_settings(analysis_cfg),
    )


def _resolve_figure_set_settings(analysis_cfg, model_cfg, *, out_dir, primary_k):
    cfg = analysis_cfg.figure_set
    md, icl, rep, ray = (cfg[key] for key in ("md", "icl", "representatives", "raytrace"))
    assignment = {}
    if cfg.get("color_assignment_file"):
        assignment.update(
            json.loads(_resolve_input_path(cfg.color_assignment_file).read_text())["assignment"]
        )
    assignment.update(cfg.color_assignment)
    assignment = {int(k): v for k, v in assignment.items()} or None
    points = rep.points or model_cfg.data.get("model_points", model_cfg.data.get("num_points", 48))
    quality = ray.high_quality
    return FigureSetSettings(
        enabled=cfg.enabled,
        k=primary_k,
        md_max_points=md.max_points or None,
        md_point_size=md.point_size,
        md_alpha=md.alpha,
        md_halo_scale=md.halo_scale,
        md_halo_alpha=md.halo_alpha,
        md_saturation_boost=md.saturation_boost,
        md_view_elev=md.view_elev,
        md_view_azim=md.view_azim,
        md_num_views=md.num_views,
        cluster_color_assignment=assignment,
        profile_point_scale_enabled=cfg.profile_point_scale_enabled,
        icl_enabled=icl.enabled,
        icl_k_min=icl.k_min,
        icl_k_max=icl.k_max,
        icl_max_samples=icl.max_samples or None,
        representative_points=points,
        representative_orientation=rep.orientation,
        representative_view_elev=rep.view_elev,
        representative_view_azim=rep.view_azim,
        representative_projection=rep.projection,
        representative_ptm_enabled=rep.ptm_enabled,
        representative_cna_enabled=rep.cna_enabled,
        representative_cna_max_signatures=rep.cna_max_signatures,
        representative_center_atom_tolerance=rep.center_atom_tolerance,
        representative_shell_min_neighbors=rep.shell_min_neighbors,
        representative_shell_max_neighbors=rep.shell_max_neighbors,
        real_md_profile_target_points=analysis_cfg.real_md.profiles.target_points,
        raytrace_enabled=ray.enabled,
        raytrace_kwargs=dict(
            raytrace_blender_executable=ray.blender_executable,
            raytrace_render_resolution=1600 if quality else ray.resolution,
            raytrace_render_max_points=ray.max_points or None,
            raytrace_render_samples=64 if quality else ray.samples,
            raytrace_render_denoise=ray.denoise,
            raytrace_render_high_quality=quality,
            raytrace_render_projection=ray.projection,
            raytrace_render_fov_deg=ray.fov_deg,
            raytrace_render_camera_distance_factor=ray.camera_distance_factor,
            raytrace_render_sphere_radius_fraction=ray.sphere_radius_fraction,
            raytrace_render_timeout_sec=ray.timeout_sec,
            raytrace_render_use_gpu=ray.use_gpu,
        ),
    )


def _print_resolved_analysis_settings(analysis_settings, figure_settings):
    print(
        "Clustering:", analysis_settings.cluster_method, "k =", analysis_settings.cluster_k_values
    )
    print(
        "Features:",
        dict(
            l2_normalize=analysis_settings.cluster_l2_normalize,
            standardize=analysis_settings.cluster_standardize,
            pca_variance=analysis_settings.cluster_pca_var,
            pca_max_components=analysis_settings.cluster_pca_max_components,
        ),
    )
    print(
        "Figures:",
        dict(
            enabled=figure_settings.enabled,
            points=figure_settings.representative_points,
            ptm=figure_settings.representative_ptm_enabled,
            cna=figure_settings.representative_cna_enabled,
        ),
    )


def load_checkpoint_training_config(checkpoint_path):
    directory, name = resolve_config_path(checkpoint_path)
    return OmegaConf.load(Path(directory) / f"{name}.yaml")


def load_checkpoint_analysis_config(config_path=None):
    path = _resolve_input_path(config_path or DEFAULT_ANALYSIS_CONFIG_PATH)
    cfg = OmegaConf.load(path)
    parent = cfg.pop("extends", None)
    return (
        OmegaConf.merge(load_checkpoint_analysis_config(path.parent / parent), cfg)
        if parent
        else cfg
    )


def build_runtime_model_config(checkpoint_path, analysis_cfg, *, data_config_path_override=None):
    model_cfg = load_checkpoint_training_config(checkpoint_path)
    data_path = data_config_path_override or analysis_cfg.inputs.data_config
    if data_path:
        data_cfg = OmegaConf.load(_resolve_input_path(data_path))
        model_cfg = OmegaConf.merge(model_cfg, {"data": data_cfg})
    _apply_analysis_inference_overrides(model_cfg)
    return model_cfg


def _apply_analysis_inference_overrides(model_cfg: DictConfig) -> None:
    if OmegaConf.select(model_cfg, "encoder.name") in {
        "PretrainedMACEGeometry",
        "PretrainedMACEHistoryGeometry",
    }:
        # Analysis adds new graph sizes and inference contexts after training has
        # already populated Dynamo's shared code cache. fullgraph=True then hits
        # the recompilation limit. Eager radial layers retain the weights/BF16 math.
        with open_dict(model_cfg):
            model_cfg.encoder.kwargs.performance.compile_radial_mlp = False
            model_cfg.encoder.kwargs.activation_checkpointing = False
        print("[analysis] Using eager MACE radial layers for variable-size inference batches.")
    if OmegaConf.select(model_cfg, "data.kind") == "relaxed_histories":
        with open_dict(model_cfg):
            model_cfg.data.radius = model_cfg.data.normalization_radius_A
            model_cfg.data.analysis_identity = "atom_id_v1"
    if (
        bool(OmegaConf.select(model_cfg, "vicreg_temporal_view", default=False))
        and model_cfg.data.kind != "spatiotemporal_binary"
    ):
        print(
            "[analysis] Disabling training-only temporal view construction for the overridden inference dataset; encoder/projector weights are unchanged."
        )
        with open_dict(model_cfg):
            model_cfg.vicreg_temporal_view = False
    if bool(
        OmegaConf.select(
            model_cfg,
            "vicreg_projector_bn_eval_batch_stats",
            default=False,
        )
    ):
        print(
            "[analysis] Replacing vicreg_projector_bn_eval_batch_stats=true with "
            "false: exported projector embeddings must use checkpoint running "
            "statistics so they do not depend on inference-batch composition."
        )
        with open_dict(model_cfg):
            model_cfg.vicreg_projector_bn_eval_batch_stats = False

    compile_enabled = bool(OmegaConf.select(model_cfg, "compile_encoder", default=False))
    compile_mode = str(OmegaConf.select(model_cfg, "encoder_compile_mode", default="default"))
    if compile_enabled and compile_mode == "reduce-overhead":
        print(
            "[analysis] Replacing encoder_compile_mode='reduce-overhead' with "
            "'default': the CUDA-graph mode is not numerically stable for "
            "GeoFrame inference on this PyTorch/H100 stack."
        )
        with open_dict(model_cfg):
            model_cfg.encoder_compile_mode = "default"

    encoder_kwargs = OmegaConf.select(model_cfg, "encoder.kwargs", default=None)
    if encoder_kwargs is None:
        return
    if "deterministic_fps" in encoder_kwargs:
        with open_dict(model_cfg):
            model_cfg.encoder.kwargs.deterministic_fps = True
