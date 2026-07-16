from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from os import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


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
    transition_thermal_spike: float = 0.0
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
    contact_relaxation_max_iterations: int = 128
    fast_mode: bool = False
    source_generator_config: Optional[Path] = None
    initial_positions_file: Optional[Path] = None
    source_frame_step: Optional[int] = None
    source_generator_config_sha256: Optional[str] = None
    source_manifest_sha256: Optional[str] = None
    source_metadata_sha256: Optional[str] = None
    source_trajectory_sha256: Optional[str] = None
    initial_positions_sha256: Optional[str] = None
    source_chemical_symbol: Optional[str] = None


@dataclass(frozen=True)
class StructuralAuditConfig:
    frame_indices: list[int]
    state_names: list[str]
    ptm_rmsd_cutoff: float
    minimum_aggregate_atoms_per_state: int
    crystal_state_name: str
    liquid_state_name: str
    minimum_crystal_crystalline_fraction: float
    minimum_crystal_fcc_fraction: float
    maximum_liquid_crystalline_fraction: float
    minimum_crystal_liquid_fraction_margin: float


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
    structural_audit: StructuralAuditConfig
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
                "structural_audit": asdict(self.structural_audit),
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
    config_path = Path(path).expanduser().resolve()
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
    time = TimeConfig(
        num_frames=int(_require_key(time_raw, "num_frames", config_path, "time")),
        delta_t=float(time_raw.get("delta_t", 1.0)),
    )
    structural_audit = _load_structural_audit_config(
        raw=_require_mapping(raw, "structural_audit", config_path),
        states=states,
        num_frames=time.num_frames,
        config_path=config_path,
    )
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
    state_names = {state.name for state in states}
    missing_outgoing_states = sorted(
        state_name
        for state_name in state_names
        if not any(transition.source == state_name for transition in transitions)
    )
    if missing_outgoing_states:
        raise ValueError(
            f"Temporal config at {config_path} gives finite dwell distributions to states "
            f"without outgoing transitions: {missing_outgoing_states}."
        )
    if dynamics.nucleation_seed_fraction > 0.0:
        num_frames = int(_require_key(time_raw, "num_frames", config_path, "time"))
        if not 0 <= dynamics.nucleation_start_frame < num_frames:
            raise ValueError(
                f"Temporal config at {config_path} requires dynamics.nucleation_start_frame "
                f"in [0, {num_frames - 1}], got {dynamics.nucleation_start_frame}."
            )
        nucleation_state = dynamics.nucleation_state
        if nucleation_state is None and len(dynamics.primary_path) >= 2:
            nucleation_state = dynamics.primary_path[1]
        if nucleation_state not in state_names:
            raise ValueError(
                f"Temporal config at {config_path} has nucleation_state="
                f"{nucleation_state!r}, which is not one of {sorted(state_names)}."
            )
        if (
            dynamics.nucleation_start_frame > 0
            and float(dynamics.initial_state_probs.get(nucleation_state, 0.0)) != 0.0
        ):
            raise ValueError(
                f"Temporal config at {config_path} delays nucleation until frame "
                f"{dynamics.nucleation_start_frame}, so initial_state_probs for "
                f"nucleation_state={nucleation_state!r} must be exactly 0."
            )
    nuisance = NuisanceConfig(**dict(raw.get("nuisance", {})))
    rendering_raw = dict(raw.get("rendering", {}))
    source_generator_config = rendering_raw.get("source_generator_config")
    if source_generator_config is None:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "rendering.source_generator_config so the force-driven source manifest, "
            "Hamiltonian, runtime, and producer identity can be checked exactly."
        )
    source_generator_config_path = Path(str(source_generator_config)).expanduser()
    if not source_generator_config_path.is_absolute():
        source_generator_config_path = REPOSITORY_ROOT / source_generator_config_path
    source_generator_config_path = source_generator_config_path.resolve()
    rendering_raw["source_generator_config"] = source_generator_config_path
    initial_positions = rendering_raw.get("initial_positions_file")
    if initial_positions is None:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "rendering.initial_positions_file pointing to an (N, 3) atoms.npy produced "
            "by a repository-owned force-driven simulation. The procedural renderer no "
            "longer invents a liquid packing. Generate the atomistic phase-context dataset "
            "first and configure its bulk-liquid atoms.npy path."
        )
    initial_positions_path = Path(str(initial_positions)).expanduser()
    if not initial_positions_path.is_absolute():
        initial_positions_path = REPOSITORY_ROOT / initial_positions_path
    initial_positions_path = initial_positions_path.resolve()
    rendering_raw["initial_positions_file"] = initial_positions_path
    derived_source_fields = (
        "source_frame_step",
        "source_generator_config_sha256",
        "source_manifest_sha256",
        "source_metadata_sha256",
        "source_trajectory_sha256",
        "initial_positions_sha256",
        "source_chemical_symbol",
    )
    configured_source_values = {
        field_name: rendering_raw.pop(field_name, None)
        for field_name in derived_source_fields
    }
    rendering = RenderingConfig(**rendering_raw)
    if rendering.frame_style != "dense_persistent_box":
        raise ValueError(
            "Temporal rendering supports only rendering.frame_style='dense_persistent_box'. "
            f"Got rendering.frame_style={rendering.frame_style!r} in {config_path}."
        )
    if not 0.0 <= rendering.phase_field_jitter_fraction <= 0.45:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "rendering.phase_field_jitter_fraction in [0, 0.45], got "
            f"{rendering.phase_field_jitter_fraction}."
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

    domain = DomainConfig(
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
    )
    _validate_initial_positions(
        path=initial_positions_path,
        box_size=domain.box_size,
        target_density=rendering.target_density,
        config_path=config_path,
    )
    observed_source_values = _validate_force_driven_source(
        source_generator_config_path=source_generator_config_path,
        initial_positions_path=initial_positions_path,
        box_size=domain.box_size,
        config_path=config_path,
    )
    missing_source_values = sorted(
        set(derived_source_fields) - set(observed_source_values)
    )
    unexpected_source_values = sorted(
        set(observed_source_values) - set(derived_source_fields)
    )
    if missing_source_values or unexpected_source_values:
        raise RuntimeError(
            "Temporal force-driven source validation returned an incompatible identity "
            f"record: missing={missing_source_values}, unexpected={unexpected_source_values}."
        )
    mismatched_source_values = {
        field_name: {
            "configured": configured_source_values[field_name],
            "observed": observed_source_values[field_name],
        }
        for field_name in derived_source_fields
        if configured_source_values[field_name] is not None
        and configured_source_values[field_name] != observed_source_values[field_name]
    }
    if mismatched_source_values:
        raise RuntimeError(
            f"Temporal config at {config_path} contains stale force-driven source identity "
            f"fields: {mismatched_source_values}. Reload it from the current producer."
        )
    rendering = replace(rendering, **observed_source_values)

    return TemporalBenchmarkConfig(
        dataset_name=str(raw.get("dataset_name", config_path.stem)),
        seed=int(raw.get("seed", 0)),
        domain=domain,
        time=time,
        states=states,
        transitions=transitions,
        dynamics=dynamics,
        nuisance=nuisance,
        rendering=rendering,
        structural_audit=structural_audit,
        trajectories=trajectories,
        output=output,
        visualization=visualization,
    )


def _validate_initial_positions(
    *,
    path: Path,
    box_size: float,
    target_density: float,
    config_path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Temporal config at {config_path} references missing force-driven initial "
            f"positions: {path}. Generate that producer dataset first or set "
            "rendering.initial_positions_file to its atoms.npy."
        )
    positions = np.load(path, mmap_mode="r")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Temporal initial positions at {path} must have concrete shape (N, 3), got "
            f"{positions.shape}."
        )
    if positions.shape[0] < 2:
        raise ValueError(
            f"Temporal initial positions at {path} must contain at least two atoms, got "
            f"shape={positions.shape}."
        )
    if not np.isfinite(positions).all():
        raise ValueError(f"Temporal initial positions contain non-finite coordinates: {path}.")
    minima = np.min(positions, axis=0)
    maxima = np.max(positions, axis=0)
    if np.any(minima < 0.0) or np.any(maxima >= box_size):
        raise ValueError(
            f"Temporal initial positions at {path} do not fit the configured periodic box "
            f"[0, {box_size}); minima={minima.tolist()}, maxima={maxima.tolist()}. Use the "
            "producer's exact cubic cell length as domain.box_size; do not rescale an MD "
            "snapshot."
        )
    observed_density = positions.shape[0] / box_size**3
    relative_density_error = abs(observed_density - target_density) / observed_density
    if relative_density_error > 1.0e-8:
        raise ValueError(
            f"Temporal rendering.target_density={target_density:.9f} is inconsistent with "
            f"the force-driven source density {observed_density:.9f} atoms/A^3 from "
            f"N={positions.shape[0]} and box_size={box_size} A at {path}; relative_error="
            f"{relative_density_error:.3%}. Configure the producer's concrete density."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_force_driven_source(
    *,
    source_generator_config_path: Path,
    initial_positions_path: Path,
    box_size: float,
    config_path: Path,
) -> dict[str, int | str]:
    from ..atomistic.artifacts import PHASE_TO_ID
    from ..atomistic.config import load_config
    from ..atomistic.provenance import validate_configured_source_manifest
    from ..atomistic.simulation import build_initial_solid

    source_config = load_config(source_generator_config_path)
    source_root = initial_positions_path.parent.parent
    expected_source_root = source_config.output.root_dir.resolve()
    if source_root != expected_source_root:
        raise RuntimeError(
            f"Temporal config at {config_path} selects initial positions below {source_root}, "
            f"but source_generator_config declares output.root_dir={expected_source_root}. "
            "The source file and producer configuration must describe the same dataset."
        )
    source_environment = initial_positions_path.parent.name
    if not source_environment.endswith("_bulk_liquid"):
        raise RuntimeError(
            f"Temporal initial positions must come from a bulk-liquid environment, got "
            f"{source_environment!r} at {initial_positions_path}."
        )
    manifest_path = source_root / "manifest.json"
    metadata_path = initial_positions_path.parent / "metadata.json"
    atom_table_path = initial_positions_path.parent / "atoms_full.npy"
    trajectory_path = initial_positions_path.parent / "trajectory.npz"
    for required_path in (
        source_generator_config_path,
        manifest_path,
        metadata_path,
        atom_table_path,
        trajectory_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Temporal force-driven source is missing required file: {required_path}."
            )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: source manifest root must be a mapping.")
    if manifest.get("schema_version") != 4:
        raise RuntimeError(
            f"{manifest_path}: expected current phase-context schema_version=4, got "
            f"{manifest.get('schema_version')!r}. Legacy sources must be regenerated."
        )
    validate_configured_source_manifest(
        manifest,
        config=source_config,
        manifest_path=manifest_path,
    )
    environment_dirs = manifest.get("environment_dirs")
    if not isinstance(environment_dirs, list) or source_environment not in environment_dirs:
        raise RuntimeError(
            f"{manifest_path}: source environment {source_environment!r} is not declared in "
            f"environment_dirs={environment_dirs!r}."
        )

    atom_table = np.load(atom_table_path, mmap_mode="r")
    if atom_table.dtype.names is None or "position" not in atom_table.dtype.names or (
        "phase_id" not in atom_table.dtype.names
    ):
        raise ValueError(
            f"{atom_table_path}: expected repository atom table fields 'position' and "
            f"'phase_id', got dtype={atom_table.dtype}."
        )
    if len(atom_table) != len(build_initial_solid(source_config)):
        raise RuntimeError(
            f"{atom_table_path}: atom count {len(atom_table)} does not match the concrete "
            f"source producer count {len(build_initial_solid(source_config))}."
        )
    non_liquid = np.flatnonzero(atom_table["phase_id"] != PHASE_TO_ID["liquid_bulk"])
    if len(non_liquid):
        raise RuntimeError(
            f"{atom_table_path}: {len(non_liquid)} atoms are not labelled liquid_bulk; "
            "the temporal benchmark must start from the validated bulk-liquid endpoint."
        )
    positions = np.load(initial_positions_path, mmap_mode="r")
    if positions.shape != atom_table["position"].shape or not np.array_equal(
        positions, atom_table["position"]
    ):
        raise RuntimeError(
            f"{initial_positions_path}: positions differ from {atom_table_path}; the source "
            "artifact is internally inconsistent."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 3 or metadata.get(
        "environment_name"
    ) != source_environment:
        raise RuntimeError(
            f"{metadata_path}: expected schema_version=3 and environment_name="
            f"{source_environment!r}, got schema_version={metadata.get('schema_version')!r}, "
            f"environment_name={metadata.get('environment_name')!r}."
        )
    global_metadata = metadata.get("global")
    physics_metadata = metadata.get("physics")
    if not isinstance(global_metadata, dict) or not isinstance(physics_metadata, dict):
        raise TypeError(f"{metadata_path}: global and physics must be mappings.")
    if physics_metadata.get("chemical_symbol") != source_config.system.chemical_symbol:
        raise RuntimeError(
            f"{metadata_path}: chemical_symbol={physics_metadata.get('chemical_symbol')!r} "
            f"does not match source configuration {source_config.system.chemical_symbol!r}."
        )

    with np.load(trajectory_path) as trajectory:
        required_arrays = {
            "step",
            "positions_A",
            "cell_vectors_A",
            "volume_A3",
        }
        missing_arrays = sorted(required_arrays - set(trajectory.files))
        if missing_arrays:
            raise RuntimeError(
                f"{trajectory_path}: missing source trajectory arrays {missing_arrays}."
            )
        step = np.asarray(trajectory["step"], dtype=np.int64)
        trajectory_positions = np.asarray(trajectory["positions_A"][-1], dtype=np.float64)
        final_cell = np.asarray(trajectory["cell_vectors_A"][-1], dtype=np.float64)
        stored_volume = float(trajectory["volume_A3"][-1])
    if step.ndim != 1 or len(step) == 0 or np.any(np.diff(step) <= 0):
        raise RuntimeError(
            f"{trajectory_path}: step must be a non-empty strictly increasing vector, got "
            f"shape={step.shape}, values={step.tolist()}."
        )
    expected_cell = np.eye(3, dtype=np.float64) * box_size
    if final_cell.shape != (3, 3) or not np.allclose(
        final_cell, expected_cell, rtol=0.0, atol=1.0e-8
    ):
        raise RuntimeError(
            f"{trajectory_path}: final source cell={final_cell.tolist()} is not the exact "
            f"configured cubic temporal cell={expected_cell.tolist()}. The renderer cannot "
            "discard tilt or anisotropic cell information."
        )
    final_volume = float(np.linalg.det(final_cell))
    if not np.isclose(final_volume, stored_volume, rtol=1.0e-12, atol=1.0e-6):
        raise RuntimeError(
            f"{trajectory_path}: det(final_cell)={final_volume:.12g} A^3 differs from "
            f"stored final volume={stored_volume:.12g} A^3."
        )
    if trajectory_positions.shape != positions.shape:
        raise RuntimeError(
            f"{trajectory_path}: final positions shape={trajectory_positions.shape} differs "
            f"from {initial_positions_path} shape={positions.shape}."
        )
    position_delta = trajectory_positions - np.asarray(positions, dtype=np.float64)
    position_delta -= box_size * np.rint(position_delta / box_size)
    maximum_position_difference = float(np.linalg.norm(position_delta, axis=1).max())
    if maximum_position_difference > 1.0e-5:
        raise RuntimeError(
            f"{initial_positions_path}: positions do not match the final source trajectory "
            f"under periodic wrapping; maximum_difference={maximum_position_difference:.6g} A."
        )
    metadata_cell = np.asarray(global_metadata.get("box_vectors_A"), dtype=np.float64)
    metadata_atom_count = global_metadata.get("N_final")
    if metadata_cell.shape != (3, 3) or not np.allclose(
        metadata_cell, final_cell, rtol=0.0, atol=1.0e-8
    ) or metadata_atom_count != len(positions):
        raise RuntimeError(
            f"{metadata_path}: global box/count does not match the selected endpoint; "
            f"box_vectors_A={metadata_cell.tolist()}, N_final={metadata_atom_count!r}, "
            f"expected_cell={final_cell.tolist()}, expected_N={len(positions)}."
        )

    return {
        "source_frame_step": int(step[-1]),
        "source_generator_config_sha256": _sha256_file(source_generator_config_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_metadata_sha256": _sha256_file(metadata_path),
        "source_trajectory_sha256": _sha256_file(trajectory_path),
        "initial_positions_sha256": _sha256_file(initial_positions_path),
        "source_chemical_symbol": source_config.system.chemical_symbol,
    }


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


def _load_structural_audit_config(
    *,
    raw: dict[str, Any],
    states: list[StateConfig],
    num_frames: int,
    config_path: Path,
) -> StructuralAuditConfig:
    frame_indices_raw = _require_key(
        raw, "frame_indices", config_path, "structural_audit"
    )
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise TypeError(
            f"Temporal config at {config_path} requires structural_audit.frame_indices "
            "to be a non-empty list of explicit zero-based frame indices."
        )
    if any(type(frame_index) is not int for frame_index in frame_indices_raw):
        raise TypeError(
            f"Temporal config at {config_path} requires integer "
            "structural_audit.frame_indices, got "
            f"{frame_indices_raw!r}."
        )
    frame_indices = [int(frame_index) for frame_index in frame_indices_raw]
    if frame_indices != sorted(set(frame_indices)):
        raise ValueError(
            f"Temporal config at {config_path} requires strictly increasing unique "
            "structural_audit.frame_indices, got "
            f"{frame_indices}."
        )
    invalid_frame_indices = [
        frame_index
        for frame_index in frame_indices
        if frame_index < 0 or frame_index >= num_frames
    ]
    if invalid_frame_indices:
        raise ValueError(
            f"Temporal config at {config_path} has structural audit frame indices outside "
            f"[0, {num_frames}): {invalid_frame_indices}."
        )

    state_names_raw = _require_key(
        raw, "state_names", config_path, "structural_audit"
    )
    if not isinstance(state_names_raw, list) or not state_names_raw or any(
        not isinstance(state_name, str) or not state_name
        for state_name in state_names_raw
    ):
        raise TypeError(
            f"Temporal config at {config_path} requires structural_audit.state_names "
            f"to be a non-empty list of state-name strings, got {state_names_raw!r}."
        )
    state_names = [str(state_name) for state_name in state_names_raw]
    if len(set(state_names)) != len(state_names):
        raise ValueError(
            f"Temporal config at {config_path} contains duplicate "
            f"structural_audit.state_names: {state_names}."
        )
    declared_state_names = {state.name for state in states}
    unknown_state_names = sorted(set(state_names) - declared_state_names)
    if unknown_state_names:
        raise ValueError(
            f"Temporal config at {config_path} audits undeclared states "
            f"{unknown_state_names}; declared states are {sorted(declared_state_names)}."
        )

    ptm_rmsd_cutoff = float(
        _require_key(raw, "ptm_rmsd_cutoff", config_path, "structural_audit")
    )
    if not np.isfinite(ptm_rmsd_cutoff) or not 0.0 < ptm_rmsd_cutoff <= 1.0:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.ptm_rmsd_cutoff in (0, 1], got "
            f"{ptm_rmsd_cutoff}."
        )
    minimum_atoms = int(
        _require_key(
            raw,
            "minimum_aggregate_atoms_per_state",
            config_path,
            "structural_audit",
        )
    )
    if minimum_atoms <= 0:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.minimum_aggregate_atoms_per_state > 0, got "
            f"{minimum_atoms}."
        )

    crystal_state_name = str(
        _require_key(raw, "crystal_state_name", config_path, "structural_audit")
    )
    liquid_state_name = str(
        _require_key(raw, "liquid_state_name", config_path, "structural_audit")
    )
    if crystal_state_name == liquid_state_name:
        raise ValueError(
            f"Temporal config at {config_path} requires distinct structural audit crystal "
            f"and liquid states, got {crystal_state_name!r} for both."
        )
    required_comparison_states = {crystal_state_name, liquid_state_name}
    missing_comparison_states = sorted(required_comparison_states - set(state_names))
    if missing_comparison_states:
        raise ValueError(
            f"Temporal config at {config_path} must include the comparison states "
            f"{missing_comparison_states} in structural_audit.state_names."
        )
    crystal_state = next(
        state for state in states if state.name == crystal_state_name
    )
    if crystal_state.crystal_structure is None or (
        crystal_state.crystal_structure.lower() != "fcc"
    ):
        raise ValueError(
            f"Temporal config at {config_path} uses the FCC-specific structural audit for "
            f"state {crystal_state_name!r}, but that state declares "
            f"crystal_structure={crystal_state.crystal_structure!r}."
        )
    margin = float(
        _require_key(
            raw,
            "minimum_crystal_liquid_fraction_margin",
            config_path,
            "structural_audit",
        )
    )
    if not np.isfinite(margin) or not 0.0 < margin <= 1.0:
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.minimum_crystal_liquid_fraction_margin in (0, 1], got "
            f"{margin}."
        )
    minimum_crystal_fraction = float(
        _require_key(
            raw,
            "minimum_crystal_crystalline_fraction",
            config_path,
            "structural_audit",
        )
    )
    maximum_liquid_fraction = float(
        _require_key(
            raw,
            "maximum_liquid_crystalline_fraction",
            config_path,
            "structural_audit",
        )
    )
    minimum_crystal_fcc_fraction = float(
        _require_key(
            raw,
            "minimum_crystal_fcc_fraction",
            config_path,
            "structural_audit",
        )
    )
    if not np.isfinite(minimum_crystal_fraction) or not (
        0.0 < minimum_crystal_fraction <= 1.0
    ):
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.minimum_crystal_crystalline_fraction in (0, 1], got "
            f"{minimum_crystal_fraction}."
        )
    if not np.isfinite(maximum_liquid_fraction) or not (
        0.0 <= maximum_liquid_fraction < 1.0
    ):
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.maximum_liquid_crystalline_fraction in [0, 1), got "
            f"{maximum_liquid_fraction}."
        )
    if not np.isfinite(minimum_crystal_fcc_fraction) or not (
        0.0 < minimum_crystal_fcc_fraction <= 1.0
    ):
        raise ValueError(
            f"Temporal config at {config_path} requires "
            "structural_audit.minimum_crystal_fcc_fraction in (0, 1], got "
            f"{minimum_crystal_fcc_fraction}."
        )
    if minimum_crystal_fraction <= maximum_liquid_fraction:
        raise ValueError(
            f"Temporal config at {config_path} requires the absolute crystal PTM floor "
            f"({minimum_crystal_fraction}) to exceed the liquid PTM ceiling "
            f"({maximum_liquid_fraction})."
        )
    absolute_implied_margin = minimum_crystal_fraction - maximum_liquid_fraction
    if margin < absolute_implied_margin:
        raise ValueError(
            f"Temporal config at {config_path} requires the explicit crystal/liquid PTM "
            f"margin ({margin}) to be at least the separation implied by the absolute "
            f"phase gates ({absolute_implied_margin})."
        )
    return StructuralAuditConfig(
        frame_indices=frame_indices,
        state_names=state_names,
        ptm_rmsd_cutoff=ptm_rmsd_cutoff,
        minimum_aggregate_atoms_per_state=minimum_atoms,
        crystal_state_name=crystal_state_name,
        liquid_state_name=liquid_state_name,
        minimum_crystal_crystalline_fraction=minimum_crystal_fraction,
        minimum_crystal_fcc_fraction=minimum_crystal_fcc_fraction,
        maximum_liquid_crystalline_fraction=maximum_liquid_fraction,
        minimum_crystal_liquid_fraction_margin=margin,
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
