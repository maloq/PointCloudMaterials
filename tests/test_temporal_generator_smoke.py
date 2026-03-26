from __future__ import annotations

import json

import numpy as np

from src.data_utils.synthetic.temporal.api import generate_temporal_dataset
from src.data_utils.synthetic.temporal.config import (
    DomainConfig,
    DwellTimeConfig,
    DynamicsConfig,
    NuisanceConfig,
    OutputConfig,
    RenderingConfig,
    StateConfig,
    TemporalBenchmarkConfig,
    TimeConfig,
    TrajectoryConfig,
    TransitionConfig,
    VisualizationConfig,
)


def test_temporal_generator_smoke(tmp_path):
    config = TemporalBenchmarkConfig(
        dataset_name="temporal_smoke",
        seed=123,
        domain=DomainConfig(
            box_size=52.0,
            avg_nn_distance=2.49,
            neighborhood_radius=6.4,
            atoms_per_site=48,
            site_count=8,
            layout="random",
            padding=6.0,
            random_min_site_distance=12.5,
        ),
        time=TimeConfig(num_frames=7, delta_t=0.5),
        states=[
            StateConfig(
                name="L",
                template_kind="liquid",
                template_params={"min_pair_distance": 1.35},
                base_thermal_jitter=0.18,
                base_defect_amplitude=0.04,
                dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=3),
            ),
            StateConfig(
                name="P",
                template_kind="precursor",
                crystal_structure="fcc",
                template_params={"min_pair_distance": 1.35, "ordered_fraction": 0.6, "local_noise": 0.2},
                base_thermal_jitter=0.12,
                base_defect_amplitude=0.08,
                grain_bearing=True,
                metastable=True,
                dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=3),
            ),
            StateConfig(
                name="I",
                template_kind="interface",
                crystal_structure="fcc",
                template_params={
                    "min_pair_distance": 1.35,
                    "interface_width": 3.6,
                    "solid_fraction_start": 0.22,
                    "solid_fraction_end": 0.82,
                },
                base_thermal_jitter=0.09,
                base_defect_amplitude=0.06,
                grain_bearing=True,
                dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=2),
            ),
            StateConfig(
                name="C",
                template_kind="crystal",
                crystal_structure="fcc",
                base_thermal_jitter=0.03,
                base_defect_amplitude=0.01,
                grain_bearing=True,
                metastable=True,
                dwell=DwellTimeConfig(distribution="uniform", min_steps=2, max_steps=4),
            ),
            StateConfig(
                name="G",
                template_kind="grain_boundary",
                crystal_structure="fcc",
                template_params={"boundary_width": 2.6, "plane_jitter": 0.25},
                base_thermal_jitter=0.05,
                base_defect_amplitude=0.08,
                grain_bearing=True,
                metastable=True,
                dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=3),
            ),
        ],
        transitions=[
            TransitionConfig(source="L", target="P", weight=1.0),
            TransitionConfig(source="P", target="L", weight=1.0),
            TransitionConfig(source="P", target="I", weight=1.0),
            TransitionConfig(source="I", target="C", weight=1.0),
            TransitionConfig(source="I", target="G", weight=1.0),
            TransitionConfig(source="I", target="P", weight=1.0),
            TransitionConfig(source="I", target="L", weight=1.0),
        ],
        dynamics=DynamicsConfig(
            mode="coupled",
            neighbor_count=4,
            initial_state_probs={"L": 0.7, "P": 0.3},
            primary_path=["L", "P", "I", "C"],
            coupling_strength=1.8,
            reverse_coupling_strength=0.3,
            nucleation_seed_fraction=0.05,
            nucleation_state="I",
            nucleation_blob_hops=1,
            metastable_min_dwell=2,
            boundary_state_name="G",
            boundary_state_probability=0.55,
            crystal_variant_probability=0.04,
            crystal_variant_persistence=0.98,
            liquid_to_precursor_front_threshold=1,
            liquid_to_precursor_front_probability=0.90,
            precursor_to_interface_blob_threshold=2,
            precursor_to_interface_blob_probability=1.0,
            interface_to_crystal_blob_threshold=3,
            interface_to_crystal_blob_probability=1.0,
        ),
        nuisance=NuisanceConfig(
            orientation_drift_deg=2.0,
            orientation_alignment=0.9,
            strain_relaxation=0.9,
            strain_drift_scale=0.003,
            thermal_relaxation=0.85,
            thermal_noise_scale=0.01,
            defect_relaxation=0.85,
            defect_noise_scale=0.01,
        ),
        rendering=RenderingConfig(
            frame_style="dense_persistent_box",
            target_density=0.0849,
            default_crystal_structure="fcc",
            parallel_workers=4,
            site_assignment_candidate_count=8,
            phase_region_target_atoms=16,
        ),
        trajectories=TrajectoryConfig(center_neighborhoods=True, save_combined_npz=True),
        output=OutputConfig(
            output_dir=tmp_path / "temporal_smoke",
            overwrite=True,
            compress=True,
            frame_storage="single_chunk_npz",
        ),
        visualization=VisualizationConfig(
            enabled=True,
            output_subdir="visualizations",
            max_frames_to_plot=4,
            max_sites_in_gallery=4,
            max_atoms_per_frame=1500,
            frame_slice_axis=2,
            frame_slice_relative_thickness=0.3,
            write_interactive_html=False,
            write_animations=True,
            animation_max_atoms_per_frame=1800,
            animation_diagonal_cut_fraction=0.85,
            parallel_workers=1,
        ),
    )

    result = generate_temporal_dataset(config, progress=False)

    assert result.output_dir.exists()
    assert result.manifest_path.exists()
    assert result.validation_summary_path.exists()
    assert result.frame_chunk_path is not None
    assert result.frame_chunk_path.exists()
    assert len(result.frame_dirs) == 0
    assert result.visualization_dir is not None
    assert result.visualization_manifest_path is not None
    assert result.visualization_manifest_path.exists()

    with result.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["num_frames"] == config.time.num_frames
    assert manifest["site_count"] == config.domain.site_count
    assert manifest["phase_site_count"] > manifest["site_count"]
    assert manifest["frame_storage"] == "single_chunk_npz"

    site_layout = np.load(result.output_dir / "site_layout.npz")
    assert site_layout["centers"].shape == (config.domain.site_count, 3)
    assert site_layout["phase_centers"].shape[0] > config.domain.site_count

    frame_chunk = np.load(result.frame_chunk_path)
    frame_atoms = frame_chunk["atoms"][0]
    assert frame_atoms.ndim == 2
    assert frame_atoms.shape[1] == 3
    assert frame_atoms.shape[0] >= config.domain.site_count * config.domain.atoms_per_site

    latent = np.load(result.output_dir / "latent" / "site_latent_trajectories.npz")
    assert latent["state_ids"].shape == (config.time.num_frames, config.domain.site_count)
    assert latent["crystal_variant_ids"].shape == (config.time.num_frames, config.domain.site_count)
    phase_latent = np.load(result.output_dir / "latent" / "phase_latent_trajectories.npz")
    assert phase_latent["state_ids"].shape[0] == config.time.num_frames
    assert phase_latent["state_ids"].shape[1] == manifest["phase_site_count"]
    interface_idx = next(idx for idx, state in enumerate(config.states) if state.name == "I")
    assert int(np.count_nonzero(phase_latent["state_ids"] == interface_idx)) > 0

    neighborhoods = np.load(result.output_dir / "neighborhoods" / "trajectory_pack.npz")
    assert neighborhoods["points"].shape == (
        config.time.num_frames,
        config.domain.site_count,
        config.domain.atoms_per_site,
        3,
    )

    for viz_name in [
        "state_occupancy_over_time.png",
        "site_state_raster.png",
        "transition_matrix.png",
        "frame_snapshots.png",
        "local_trajectory_gallery.png",
        "all_phases_diagonal_cut.gif",
        "solid_only_full_box.gif",
    ]:
        assert (result.visualization_dir / viz_name).exists()


def test_temporal_generator_fast_mode_smoke(tmp_path):
    config = TemporalBenchmarkConfig(
        dataset_name="temporal_fast_smoke",
        seed=456,
        domain=DomainConfig(
            box_size=48.0,
            avg_nn_distance=2.49,
            neighborhood_radius=6.0,
            atoms_per_site=32,
            site_count=6,
            layout="random",
            padding=6.0,
            random_min_site_distance=11.0,
        ),
        time=TimeConfig(num_frames=4, delta_t=0.5),
        states=[
            StateConfig(name="L", template_kind="liquid", dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=2)),
            StateConfig(name="P", template_kind="precursor", crystal_structure="fcc", grain_bearing=True, dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=2)),
            StateConfig(name="I", template_kind="interface", crystal_structure="fcc", grain_bearing=True, dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=2)),
            StateConfig(name="C", template_kind="crystal", crystal_structure="fcc", grain_bearing=True, dwell=DwellTimeConfig(distribution="uniform", min_steps=2, max_steps=3)),
            StateConfig(name="G", template_kind="grain_boundary", crystal_structure="fcc", grain_bearing=True, dwell=DwellTimeConfig(distribution="uniform", min_steps=1, max_steps=2)),
        ],
        transitions=[
            TransitionConfig(source="L", target="P", weight=1.0),
            TransitionConfig(source="P", target="L", weight=1.0),
            TransitionConfig(source="P", target="I", weight=1.0),
            TransitionConfig(source="I", target="C", weight=1.0),
            TransitionConfig(source="I", target="G", weight=1.0),
            TransitionConfig(source="I", target="P", weight=1.0),
            TransitionConfig(source="I", target="L", weight=1.0),
        ],
        dynamics=DynamicsConfig(
            mode="coupled",
            neighbor_count=4,
            initial_state_probs={"L": 0.7, "P": 0.3},
            primary_path=["L", "P", "I", "C"],
            nucleation_seed_fraction=0.05,
            nucleation_state="I",
            nucleation_blob_hops=1,
            liquid_to_precursor_front_threshold=1,
            liquid_to_precursor_front_probability=0.90,
            precursor_to_interface_blob_threshold=2,
            precursor_to_interface_blob_probability=1.0,
            interface_to_crystal_blob_threshold=3,
            interface_to_crystal_blob_probability=1.0,
        ),
        nuisance=NuisanceConfig(),
        rendering=RenderingConfig(
            frame_style="dense_persistent_box",
            target_density=0.0849,
            parallel_workers=2,
            site_assignment_candidate_count=8,
            phase_region_target_atoms=16,
            fast_mode=True,
        ),
        trajectories=TrajectoryConfig(center_neighborhoods=True, save_combined_npz=True),
        output=OutputConfig(
            output_dir=tmp_path / "temporal_fast_smoke",
            overwrite=True,
            compress=False,
            frame_storage="single_chunk_npz",
        ),
        visualization=VisualizationConfig(
            enabled=True,
            output_subdir="visualizations",
            max_frames_to_plot=3,
            max_sites_in_gallery=3,
            max_atoms_per_frame=800,
            frame_slice_axis=2,
            frame_slice_relative_thickness=0.3,
            write_interactive_html=False,
            write_animations=False,
            parallel_workers=1,
        ),
    )

    result = generate_temporal_dataset(config, progress=False)

    frame_chunk = np.load(result.frame_chunk_path)
    atoms = frame_chunk["atoms"]
    assert np.allclose(atoms[0], atoms[-1])

    with result.validation_summary_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    assert validation["frame_atom_count_min"] == validation["frame_atom_count_max"]
    assert result.visualization_dir is not None
    assert (result.visualization_dir / "frame_snapshots.png").exists()
