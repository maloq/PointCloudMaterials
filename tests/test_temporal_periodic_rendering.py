from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk

from src.data_utils.synthetic.temporal.config import (
    StructuralAuditConfig,
    load_temporal_config,
)
from src.data_utils.synthetic.temporal.dynamics import (
    _phase_axis_values,
    build_site_layout,
    simulate_latent_trajectories,
)
from src.data_utils.synthetic.temporal.graph import TransitionGraph
from src.data_utils.synthetic.temporal.rendering import (
    FrameRenderer,
    _minimum_image,
)
from src.data_utils.synthetic.temporal.templates import TemplateLibrary
from src.data_utils.synthetic.temporal.validation import (
    _ptm_structure_types,
    audit_rendered_frame,
    summarize_structural_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_CONFIG = (
    REPOSITORY_ROOT / "configs/simulation/temporal/realistic_metal.yaml"
)


def _source_identity_stub() -> dict[str, int | str]:
    return {
        "source_frame_step": 0,
        "source_generator_config_sha256": "test-generator",
        "source_manifest_sha256": "test-manifest",
        "source_metadata_sha256": "test-metadata",
        "source_trajectory_sha256": "test-trajectory",
        "initial_positions_sha256": "test-positions",
        "source_chemical_symbol": "Al",
    }


def _stub_production_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.config._validate_initial_positions",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.config._validate_force_driven_source",
        lambda **_kwargs: _source_identity_stub(),
    )


def test_phase_field_axis_tiles_periodic_box_without_boundary_gap() -> None:
    box_size = 108.72
    values = _phase_axis_values(box_size=box_size, spacing=5.72)
    cyclic_gaps = np.diff(np.concatenate((values, values[:1] + box_size)))

    assert len(values) == 19
    np.testing.assert_allclose(cyclic_gaps, box_size / len(values), rtol=0.0, atol=1.0e-5)


def test_latent_dwells_and_delayed_seed_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_production_source(monkeypatch)
    config = load_temporal_config(TEMPORAL_CONFIG)
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    layout = build_site_layout(config)
    latent = simulate_latent_trajectories(
        config, graph, layout, np.random.default_rng(config.seed)
    )

    seed_mask = latent.phase_seed_site_mask
    nucleation_state_id = graph.index(str(config.dynamics.nucleation_state))
    start_frame = config.dynamics.nucleation_start_frame
    assert not np.any(latent.phase_state_ids[:start_frame, seed_mask] == nucleation_state_id)
    assert np.all(latent.phase_state_ids[start_frame, seed_mask] == nucleation_state_id)
    assert len(np.unique(latent.phase_grain_ids[start_frame, seed_mask])) == 1

    complete_segment_count = 0
    for site_id in range(layout.phase_site_count):
        segment_start = 0
        for frame_index in range(1, config.time.num_frames):
            previous_state = int(latent.phase_state_ids[frame_index - 1, site_id])
            current_state = int(latent.phase_state_ids[frame_index, site_id])
            if current_state == previous_state:
                continue
            is_seed_intervention = bool(seed_mask[site_id]) and (
                frame_index == start_frame
            )
            if not is_seed_intervention:
                dwell = graph.state_config(graph.name(previous_state)).dwell
                segment_length = frame_index - segment_start
                assert dwell.min_steps <= segment_length <= dwell.max_steps
                complete_segment_count += 1
            segment_start = frame_index
    assert complete_segment_count > 0


def test_temporal_config_requires_force_driven_initial_positions(tmp_path: Path) -> None:
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    del raw["rendering"]["initial_positions_file"]
    path = tmp_path / "missing_source.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="repository-owned force-driven simulation"):
        load_temporal_config(path)


def test_temporal_config_requires_force_driven_source_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    del raw["rendering"]["source_generator_config"]
    path = tmp_path / "missing_source_config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="source_generator_config"):
        load_temporal_config(path)


def test_temporal_config_requires_explicit_structural_audit(tmp_path: Path) -> None:
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    del raw["structural_audit"]
    path = tmp_path / "missing_structural_audit.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(TypeError, match="structural_audit"):
        load_temporal_config(path)


def test_temporal_config_rejects_structural_audit_frame_outside_trajectory(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    raw["structural_audit"]["frame_indices"] = [0, raw["time"]["num_frames"]]
    path = tmp_path / "invalid_structural_audit_frame.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        load_temporal_config(path)


def test_temporal_config_rejects_out_of_range_phase_field_jitter(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    raw["rendering"]["phase_field_jitter_fraction"] = 0.46
    path = tmp_path / "invalid_jitter.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="phase_field_jitter_fraction in"):
        load_temporal_config(path)


def test_temporal_config_rejects_source_outside_periodic_box(tmp_path: Path) -> None:
    positions_path = tmp_path / "atoms.npy"
    np.save(
        positions_path,
        np.array([[0.2, 0.2, 0.2], [9.8, 0.2, 0.2]], dtype=np.float32),
    )
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    raw["domain"]["box_size"] = 5.0
    raw["rendering"]["target_density"] = 2 / 5.0**3
    raw["rendering"]["initial_positions_file"] = str(positions_path)
    path = tmp_path / "incompatible_source.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="do not fit the configured periodic box"):
        load_temporal_config(path)


def test_periodic_contact_relaxation_sees_pairs_across_box_boundary() -> None:
    renderer = FrameRenderer.__new__(FrameRenderer)
    renderer.box_size = 10.0
    renderer._hardcore_distance = 1.0
    renderer._kdtree_workers = 1
    renderer._atom_reference_directions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    points = np.array(
        [[0.2, 0.2, 0.2], [9.8, 0.2, 0.2], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )

    assert renderer._find_periodic_close_pairs(points).tolist() == [[0, 1]]
    relaxed = renderer._relax_close_contacts(points, max_iterations=128)
    periodic_separation = np.linalg.norm(
        _minimum_image(relaxed[1] - relaxed[0], renderer.box_size)
    )

    assert np.all(relaxed >= 0.0)
    assert np.all(relaxed < renderer.box_size)
    assert periodic_separation >= renderer._hardcore_distance
    assert renderer._last_minimum_pair_distance >= renderer._hardcore_distance


def test_small_periodic_renderer_keeps_atoms_and_minimum_distance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = np.array([0.5, 3.0, 5.5, 8.0], dtype=np.float32)
    positions = np.stack(
        np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    positions_path = tmp_path / "atoms.npy"
    np.save(positions_path, positions)
    raw = yaml.safe_load(TEMPORAL_CONFIG.read_text(encoding="utf-8"))
    raw["domain"].update(
        {
            "box_size": 10.0,
            "avg_nn_distance": 2.5,
            "neighborhood_radius": 4.0,
            "atoms_per_site": 4,
            "site_count": 2,
            "padding": 1.0,
            "random_min_site_distance": 2.0,
        }
    )
    raw["time"]["num_frames"] = 2
    raw["structural_audit"]["frame_indices"] = [0, 1]
    raw["dynamics"]["nucleation_start_frame"] = 1
    raw["rendering"]["initial_positions_file"] = str(positions_path)
    raw["rendering"]["target_density"] = len(positions) / 10.0**3
    raw["rendering"]["fast_mode"] = False
    raw["rendering"]["parallel_workers"] = 1
    raw["visualization"]["enabled"] = False
    config_path = tmp_path / "small_periodic.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.config._validate_force_driven_source",
        lambda **_kwargs: _source_identity_stub(),
    )

    config = load_temporal_config(config_path)
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    layout = build_site_layout(config)
    latent = simulate_latent_trajectories(
        config, graph, layout, np.random.default_rng(config.seed)
    )
    templates = TemplateLibrary(
        config.domain, config.rendering, config.states, seed=config.seed
    )
    renderer = FrameRenderer(config, graph, templates, layout, latent)

    first = renderer.render_frame(0)
    np.testing.assert_array_equal(first.site_ids, renderer._assign_sites(first.atoms))
    assert first.local_atom_indices.shape == (
        config.domain.site_count,
        config.domain.atoms_per_site,
    )
    assert first.state_ids[first.local_atom_indices].shape == first.local_atom_indices.shape
    second = renderer.render_frame(1)
    np.testing.assert_array_equal(second.site_ids, renderer._assign_sites(second.atoms))

    assert first.atoms.shape == positions.shape
    assert second.atoms.shape == positions.shape
    assert np.all(first.atoms >= 0.0)
    assert np.all(first.atoms < config.domain.box_size)
    assert second.metadata["minimum_periodic_pair_distance"] >= 2.05
    assert np.max(np.abs(first.local_points)) <= config.domain.box_size / 2.0


def test_state_conditioned_structural_audit_uses_per_atom_labels_and_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_production_source(monkeypatch)
    config = load_temporal_config(TEMPORAL_CONFIG)
    config = replace(
        config,
        structural_audit=StructuralAuditConfig(
            frame_indices=[0],
            state_names=["L", "C"],
            ptm_rmsd_cutoff=0.10,
            minimum_aggregate_atoms_per_state=2,
            crystal_state_name="C",
            liquid_state_name="L",
            minimum_crystal_crystalline_fraction=0.50,
            minimum_crystal_fcc_fraction=0.50,
            maximum_liquid_crystalline_fraction=0.10,
            minimum_crystal_liquid_fraction_margin=0.25,
        ),
    )
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    true_state_ids = np.asarray(
        [
            graph.index("L"),
            graph.index("L"),
            graph.index("P"),
            graph.index("C"),
            graph.index("C"),
            graph.index("C"),
        ],
        dtype=np.int16,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.validation._ptm_structure_types",
        lambda **_kwargs: np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
    )

    frame = audit_rendered_frame(
        config=config,
        graph=graph,
        frame_index=0,
        positions_A=np.zeros((6, 3), dtype=np.float32),
        true_state_ids=true_state_ids,
    )
    summary = summarize_structural_audit(
        config=config,
        graph=graph,
        frames=[frame],
    )

    assert summary["passed"] is True
    assert summary["aggregate_by_true_state"]["L"]["crystalline_fraction"] == 0.0
    assert summary["aggregate_by_true_state"]["C"]["crystalline_fraction"] == 1.0
    assert frame.by_true_state["P"]["atom_count"] == 1


def test_structural_audit_fails_on_crystal_liquid_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_production_source(monkeypatch)
    config = load_temporal_config(TEMPORAL_CONFIG)
    config = replace(
        config,
        structural_audit=StructuralAuditConfig(
            frame_indices=[0],
            state_names=["L", "C"],
            ptm_rmsd_cutoff=0.10,
            minimum_aggregate_atoms_per_state=2,
            crystal_state_name="C",
            liquid_state_name="L",
            minimum_crystal_crystalline_fraction=0.50,
            minimum_crystal_fcc_fraction=0.50,
            maximum_liquid_crystalline_fraction=0.10,
            minimum_crystal_liquid_fraction_margin=0.05,
        ),
    )
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    true_state_ids = np.asarray(
        [graph.index("L"), graph.index("L"), graph.index("C"), graph.index("C")],
        dtype=np.int16,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.validation._ptm_structure_types",
        lambda **_kwargs: np.asarray([1, 1, 0, 0], dtype=np.int32),
    )
    frame = audit_rendered_frame(
        config=config,
        graph=graph,
        frame_index=0,
        positions_A=np.zeros((4, 3), dtype=np.float32),
        true_state_ids=true_state_ids,
    )

    with pytest.raises(RuntimeError, match="crystal/liquid ordering"):
        summarize_structural_audit(config=config, graph=graph, frames=[frame])


@pytest.mark.parametrize(
    ("structure_types", "error_pattern"),
    [
        ([0, 0, 0, 0, 1, 0, 0, 0], "absolute crystal-state"),
        ([0, 0, 0, 0, 2, 2, 2, 2], "declared FCC"),
        ([1, 0, 0, 0, 1, 1, 1, 1], "absolute liquid-state"),
    ],
)
def test_structural_audit_enforces_absolute_and_fcc_phase_gates(
    monkeypatch: pytest.MonkeyPatch,
    structure_types: list[int],
    error_pattern: str,
) -> None:
    _stub_production_source(monkeypatch)
    config = load_temporal_config(TEMPORAL_CONFIG)
    config = replace(
        config,
        structural_audit=StructuralAuditConfig(
            frame_indices=[0],
            state_names=["L", "C"],
            ptm_rmsd_cutoff=0.10,
            minimum_aggregate_atoms_per_state=4,
            crystal_state_name="C",
            liquid_state_name="L",
            minimum_crystal_crystalline_fraction=0.50,
            minimum_crystal_fcc_fraction=0.50,
            maximum_liquid_crystalline_fraction=0.10,
            minimum_crystal_liquid_fraction_margin=0.05,
        ),
    )
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    true_state_ids = np.asarray(
        [graph.index("L")] * 4 + [graph.index("C")] * 4,
        dtype=np.int16,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.validation._ptm_structure_types",
        lambda **_kwargs: np.asarray(structure_types, dtype=np.int32),
    )
    frame = audit_rendered_frame(
        config=config,
        graph=graph,
        frame_index=0,
        positions_A=np.zeros((8, 3), dtype=np.float32),
        true_state_ids=true_state_ids,
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        summarize_structural_audit(config=config, graph=graph, frames=[frame])


def test_structural_audit_fails_when_an_audited_state_has_too_few_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_production_source(monkeypatch)
    config = load_temporal_config(TEMPORAL_CONFIG)
    config = replace(
        config,
        structural_audit=StructuralAuditConfig(
            frame_indices=[0],
            state_names=["L", "C"],
            ptm_rmsd_cutoff=0.10,
            minimum_aggregate_atoms_per_state=3,
            crystal_state_name="C",
            liquid_state_name="L",
            minimum_crystal_crystalline_fraction=0.50,
            minimum_crystal_fcc_fraction=0.50,
            maximum_liquid_crystalline_fraction=0.10,
            minimum_crystal_liquid_fraction_margin=0.05,
        ),
    )
    graph = TransitionGraph(
        config.states, config.transitions, config.dynamics.primary_path
    )
    true_state_ids = np.asarray(
        [graph.index("L"), graph.index("L"), graph.index("C"), graph.index("C")],
        dtype=np.int16,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.temporal.validation._ptm_structure_types",
        lambda **_kwargs: np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    frame = audit_rendered_frame(
        config=config,
        graph=graph,
        frame_index=0,
        positions_A=np.zeros((4, 3), dtype=np.float32),
        true_state_ids=true_state_ids,
    )

    with pytest.raises(RuntimeError, match="inadequate atom support"):
        summarize_structural_audit(config=config, graph=graph, frames=[frame])


def test_temporal_ptm_backend_recognizes_periodic_fcc_aluminum() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    structure_types = _ptm_structure_types(
        positions_A=atoms.get_positions(),
        box_size_A=float(atoms.cell.lengths()[0]),
        chemical_symbol="Al",
        rmsd_cutoff=0.10,
    )

    assert np.mean(structure_types == 1) > 0.95
