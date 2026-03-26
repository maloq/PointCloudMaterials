from __future__ import annotations

from dataclasses import asdict, dataclass, field
from os import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class DomainConfig:
    box_size: float
    avg_nn_distance: float
    neighborhood_radius: float
    atoms_per_site: int
    site_count: int
    layout: str = "grid"
    site_spacing: float = 16.0
    padding: float = 8.0
    random_min_site_distance: float = 16.0


@dataclass(frozen=True)
class TimeConfig:
    num_frames: int
    delta_t: float = 1.0


@dataclass(frozen=True)
class DwellTimeConfig:
    distribution: str = "uniform"
    min_steps: int = 2
    max_steps: int = 6
    mean_steps: float = 4.0
    std_steps: float = 1.0
    fixed_steps: Optional[int] = None


@dataclass(frozen=True)
class StateConfig:
    name: str
    template_kind: str
    crystal_structure: Optional[str] = None
    template_params: Dict[str, Any] = field(default_factory=dict)
    base_thermal_jitter: float = 0.05
    base_defect_amplitude: float = 0.0
    base_strain_scale: float = 0.0
    metastable: bool = False
    grain_bearing: bool = False
    dwell: DwellTimeConfig = field(default_factory=DwellTimeConfig)


@dataclass(frozen=True)
class TransitionConfig:
    source: str
    target: str
    weight: float


@dataclass(frozen=True)
class DynamicsConfig:
    mode: str = "independent"
    neighbor_count: int = 6
    initial_state_probs: Dict[str, float] = field(default_factory=dict)
    primary_path: list[str] = field(default_factory=list)
    coupling_strength: float = 0.0
    reverse_coupling_strength: float = 0.0
    nucleation_seed_fraction: float = 0.0
    nucleation_state: Optional[str] = None
    nucleation_start_frame: int = 0
    nucleation_blob_hops: int = 0
    metastable_min_dwell: int = 4
    boundary_state_name: Optional[str] = None
    boundary_min_distinct_grains: int = 2
    boundary_state_probability: float = 0.35
    crystal_variant_probability: float = 0.015
    crystal_variant_persistence: float = 0.985
    liquid_to_precursor_front_threshold: int = 1
    liquid_to_precursor_front_probability: float = 1.0
    precursor_to_interface_blob_threshold: int = 2
    precursor_to_interface_blob_probability: float = 1.0
    interface_to_crystal_blob_threshold: int = 4
    interface_to_crystal_blob_probability: float = 1.0


@dataclass(frozen=True)
class NuisanceConfig:
    orientation_drift_deg: float = 3.0
    orientation_alignment: float = 0.9
    strain_relaxation: float = 0.85
    strain_drift_scale: float = 0.01
    thermal_relaxation: float = 0.85
    thermal_noise_scale: float = 0.02
    defect_relaxation: float = 0.85
    defect_noise_scale: float = 0.02


@dataclass(frozen=True)
class RenderingConfig:
    frame_style: str = "dense_persistent_box"
    target_density: float = 0.0849
    default_crystal_structure: str = "fcc"
    save_frame_dirs: bool = True
    save_atom_tables: bool = True
    neighborhood_sampling_method: str = "closest_to_center"
    parallel_workers: int = field(default_factory=lambda: max(1, cpu_count() or 1))
    site_assignment_candidate_count: int = 24
    phase_region_radius_nn: float = 2.0
    phase_region_target_atoms: Optional[int] = None
    phase_field_jitter_fraction: float = 0.15
    fast_mode: bool = False


@dataclass(frozen=True)
class TrajectoryConfig:
    center_neighborhoods: bool = True
    save_combined_npz: bool = True


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    overwrite: bool = False
    compress: bool = True
    frame_storage: str = "frame_dirs"


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    output_subdir: str = "visualizations"
    max_frames_to_plot: int = 6
    max_sites_in_gallery: int = 8
    max_atoms_per_frame: int = 12000
    frame_slice_axis: int = 2
    frame_slice_relative_thickness: float = 0.22
    write_interactive_html: bool = True
    write_animations: bool = True
    animation_max_atoms_per_frame: int = 18000
    animation_diagonal_cut_fraction: float = 0.85
    animation_diagonal_visible_depth_nn: float = 3.0
    animation_diagonal_marker_size: float = 18.0
    animation_full_box_marker_size: float = 2.6
    parallel_workers: int = field(default_factory=lambda: max(1, cpu_count() or 1))


@dataclass(frozen=True)
class TemporalBenchmarkConfig:
    dataset_name: str
    seed: int
    domain: DomainConfig
    time: TimeConfig
    states: list[StateConfig]
    transitions: list[TransitionConfig]
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    nuisance: NuisanceConfig = field(default_factory=NuisanceConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    trajectories: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(output_dir=Path("output/synthetic_data/temporal"))
    )
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def state_map(self) -> dict[str, StateConfig]:
        return {state.name: state for state in self.states}

    def to_serializable_dict(self) -> dict[str, Any]:
        return _make_serializable(asdict(self))

    def to_config_dict(self) -> dict[str, Any]:
        dynamics_dict = asdict(self.dynamics)
        transition_graph = {
            "initial_state_probs": dynamics_dict.pop("initial_state_probs"),
            "primary_path": dynamics_dict.pop("primary_path"),
            "transitions": [asdict(transition) for transition in self.transitions],
        }
        return _make_serializable(
            {
                "dataset_name": self.dataset_name,
                "seed": self.seed,
                "domain": asdict(self.domain),
                "time": asdict(self.time),
                "states": [asdict(state) for state in self.states],
                "transition_graph": transition_graph,
                "dynamics": dynamics_dict,
                "nuisance": asdict(self.nuisance),
                "rendering": asdict(self.rendering),
                "trajectories": asdict(self.trajectories),
                "output": asdict(self.output),
                "visualization": asdict(self.visualization),
            }
        )


def _make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_serializable(item) for item in value]
    return value


def load_temporal_config(path: str | Path) -> TemporalBenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Temporal config at {config_path} must decode to a mapping, got {type(raw)}.")

    domain_raw = _require_mapping(raw, "domain", config_path)
    time_raw = _require_mapping(raw, "time", config_path)
    graph_raw = _require_mapping(raw, "transition_graph", config_path)
    output_raw = dict(raw.get("output", {}))

    states_raw = raw.get("states")
    if not isinstance(states_raw, list) or not states_raw:
        raise ValueError(f"Temporal config at {config_path} must define a non-empty 'states' list.")

    states = [_load_state_config(item, config_path) for item in states_raw]
    transitions_raw = graph_raw.get("transitions")
    if not isinstance(transitions_raw, list) or not transitions_raw:
        raise ValueError(
            f"Temporal config at {config_path} must define a non-empty transition_graph.transitions list."
        )
    transitions = [
        TransitionConfig(
            source=str(_require_key(item, "source", config_path, "transition entry")),
            target=str(_require_key(item, "target", config_path, "transition entry")),
            weight=float(_require_key(item, "weight", config_path, "transition entry")),
        )
        for item in transitions_raw
    ]

    dynamics = DynamicsConfig(
        mode=str(raw.get("dynamics", {}).get("mode", "independent")),
        neighbor_count=int(raw.get("dynamics", {}).get("neighbor_count", 6)),
        initial_state_probs=dict(graph_raw.get("initial_state_probs", {})),
        primary_path=[str(item) for item in graph_raw.get("primary_path", [])],
        coupling_strength=float(raw.get("dynamics", {}).get("coupling_strength", 0.0)),
        reverse_coupling_strength=float(raw.get("dynamics", {}).get("reverse_coupling_strength", 0.0)),
        nucleation_seed_fraction=float(raw.get("dynamics", {}).get("nucleation_seed_fraction", 0.0)),
        nucleation_state=raw.get("dynamics", {}).get("nucleation_state"),
        nucleation_start_frame=int(raw.get("dynamics", {}).get("nucleation_start_frame", 0)),
        nucleation_blob_hops=int(raw.get("dynamics", {}).get("nucleation_blob_hops", 0)),
        metastable_min_dwell=int(raw.get("dynamics", {}).get("metastable_min_dwell", 4)),
        boundary_state_name=raw.get("dynamics", {}).get("boundary_state_name"),
        boundary_min_distinct_grains=int(raw.get("dynamics", {}).get("boundary_min_distinct_grains", 2)),
        boundary_state_probability=float(raw.get("dynamics", {}).get("boundary_state_probability", 0.35)),
        crystal_variant_probability=float(raw.get("dynamics", {}).get("crystal_variant_probability", 0.015)),
        crystal_variant_persistence=float(raw.get("dynamics", {}).get("crystal_variant_persistence", 0.985)),
        liquid_to_precursor_front_threshold=int(
            raw.get("dynamics", {}).get("liquid_to_precursor_front_threshold", 1)
        ),
        liquid_to_precursor_front_probability=float(
            raw.get("dynamics", {}).get("liquid_to_precursor_front_probability", 1.0)
        ),
        precursor_to_interface_blob_threshold=int(
            raw.get("dynamics", {}).get("precursor_to_interface_blob_threshold", 2)
        ),
        precursor_to_interface_blob_probability=float(
            raw.get("dynamics", {}).get("precursor_to_interface_blob_probability", 1.0)
        ),
        interface_to_crystal_blob_threshold=int(
            raw.get("dynamics", {}).get("interface_to_crystal_blob_threshold", 4)
        ),
        interface_to_crystal_blob_probability=float(
            raw.get("dynamics", {}).get("interface_to_crystal_blob_probability", 1.0)
        ),
    )
    nuisance = NuisanceConfig(**dict(raw.get("nuisance", {})))
    rendering = RenderingConfig(**dict(raw.get("rendering", {})))
    if rendering.frame_style != "dense_persistent_box":
        raise ValueError(
            "Temporal rendering supports only rendering.frame_style='dense_persistent_box'. "
            f"Got rendering.frame_style={rendering.frame_style!r} in {config_path}."
        )
    trajectories = TrajectoryConfig(**dict(raw.get("trajectories", {})))
    visualization = VisualizationConfig(**dict(raw.get("visualization", {})))

    output_dir = Path(output_raw.get("output_dir", "output/synthetic_data/temporal"))
    output = OutputConfig(
        output_dir=output_dir,
        overwrite=bool(output_raw.get("overwrite", False)),
        compress=bool(output_raw.get("compress", True)),
        frame_storage=str(output_raw.get("frame_storage", "frame_dirs")),
    )

    return TemporalBenchmarkConfig(
        dataset_name=str(raw.get("dataset_name", config_path.stem)),
        seed=int(raw.get("seed", 0)),
        domain=DomainConfig(
            box_size=float(_require_key(domain_raw, "box_size", config_path, "domain")),
            avg_nn_distance=float(_require_key(domain_raw, "avg_nn_distance", config_path, "domain")),
            neighborhood_radius=float(_require_key(domain_raw, "neighborhood_radius", config_path, "domain")),
            atoms_per_site=int(_require_key(domain_raw, "atoms_per_site", config_path, "domain")),
            site_count=int(_require_key(domain_raw, "site_count", config_path, "domain")),
            layout=str(domain_raw.get("layout", "grid")),
            site_spacing=float(domain_raw.get("site_spacing", 16.0)),
            padding=float(domain_raw.get("padding", 8.0)),
            random_min_site_distance=float(
                domain_raw.get("random_min_site_distance", domain_raw.get("site_spacing", 16.0))
            ),
        ),
        time=TimeConfig(
            num_frames=int(_require_key(time_raw, "num_frames", config_path, "time")),
            delta_t=float(time_raw.get("delta_t", 1.0)),
        ),
        states=states,
        transitions=transitions,
        dynamics=dynamics,
        nuisance=nuisance,
        rendering=rendering,
        trajectories=trajectories,
        output=output,
        visualization=visualization,
    )


def dump_temporal_config(config: TemporalBenchmarkConfig, path: str | Path) -> None:
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_config_dict(), handle, sort_keys=False)


def _load_state_config(raw: Any, config_path: Path) -> StateConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"Each state entry in {config_path} must be a mapping, got {type(raw)}.")
    dwell_raw = dict(raw.get("dwell", {}))
    return StateConfig(
        name=str(_require_key(raw, "name", config_path, "state entry")),
        template_kind=str(_require_key(raw, "template_kind", config_path, "state entry")),
        crystal_structure=raw.get("crystal_structure"),
        template_params=dict(raw.get("template_params", {})),
        base_thermal_jitter=float(raw.get("base_thermal_jitter", 0.05)),
        base_defect_amplitude=float(raw.get("base_defect_amplitude", 0.0)),
        base_strain_scale=float(raw.get("base_strain_scale", 0.0)),
        metastable=bool(raw.get("metastable", False)),
        grain_bearing=bool(raw.get("grain_bearing", False)),
        dwell=DwellTimeConfig(**dwell_raw),
    )


def _require_mapping(raw: dict[str, Any], key: str, config_path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Temporal config at {config_path} must define '{key}' as a mapping.")
    return value


def _require_key(raw: dict[str, Any], key: str, config_path: Path, section: str) -> Any:
    if key not in raw:
        raise KeyError(f"Temporal config at {config_path} is missing required key '{key}' in {section}.")
    return raw[key]
