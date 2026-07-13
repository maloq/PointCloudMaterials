from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis.directional_line_jepa import (
    DirectionalLineJEPASettings,
    _selected_analysis_indices,
    _valid_directional_line_anchor_mask,
    apply_directional_runtime_limits,
    compute_directional_error_summaries,
    compute_directional_error_summaries_chunked,
    compute_joint_hardness_score,
    disable_directional_for_non_line_jepa,
    fibonacci_sphere_directions,
    phase_conditioned_z_scores,
)
from src.analysis.directional_line_jepa_vis import (
    _DISCOVERY_METRICS,
    render_directional_line_jepa_visualizations,
)
from src.data_utils.line_static_dataset import (
    FixedSlotDirectionError,
    LineStaticPointCloudDataset,
)
from src.training_methods.line_jepa.line_jepa_module import LineJEPAModule


def test_runtime_limits_can_enable_and_cap_directional_sweep() -> None:
    settings = DirectionalLineJEPASettings(
        enabled=False,
        num_directions=64,
        max_atoms_total=10_000,
        task_batch_size=512,
        selection_seed=123,
        target_position="checkpoint",
        relative_error_clip=100.0,
        joint_weights={"mean_error": 1.0},
    )

    effective = apply_directional_runtime_limits(
        settings,
        enabled=True,
        max_directions=32,
        max_atoms=2_048,
    )

    assert effective.enabled
    assert effective.num_directions == 32
    assert effective.max_atoms_total == 2_048


def test_non_line_jepa_checkpoint_disables_directional_analysis() -> None:
    settings = DirectionalLineJEPASettings(
        enabled=True,
        num_directions=64,
        max_atoms_total=10_000,
        task_batch_size=512,
        selection_seed=123,
        target_position="checkpoint",
        relative_error_clip=100.0,
        joint_weights={"mean_error": 1.0},
    )

    effective, reason = disable_directional_for_non_line_jepa(
        settings,
        model_type="vicreg",
    )

    assert not effective.enabled
    assert reason is not None and "LineJEPAModule" in reason


def test_line_jepa_checkpoint_keeps_directional_analysis_enabled() -> None:
    settings = DirectionalLineJEPASettings(
        enabled=True,
        num_directions=64,
        max_atoms_total=10_000,
        task_batch_size=512,
        selection_seed=123,
        target_position="checkpoint",
        relative_error_clip=100.0,
        joint_weights={"mean_error": 1.0},
    )

    effective, reason = disable_directional_for_non_line_jepa(
        settings,
        model_type="line_jepa",
    )

    assert effective.enabled
    assert reason is None


def test_direction_grid_contains_exact_antipodal_pairs() -> None:
    directions = fibonacci_sphere_directions(64)
    np.testing.assert_allclose(directions[:32], -directions[32:], atol=1.0e-7)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-7)


def test_cached_feature_evaluation_matches_line_evaluation() -> None:
    class MeanPredictor(torch.nn.Module):
        def forward(self, context, context_positions, target_position):
            return context.mean(dim=1)

    class DummyDirectionalModel:
        line_atoms = 3
        target_line_index = 1
        target_encoder_mode = "online"
        directional_feature_mode = "none"
        prediction_target = "residual"
        prediction_normalization = "none"
        device = torch.device("cpu")
        dtype = torch.float32
        predictor = MeanPredictor()
        _line_positions_for_targets = LineJEPAModule._line_positions_for_targets
        _prediction_target = LineJEPAModule._prediction_target
        _prediction_loss_inputs = LineJEPAModule._prediction_loss_inputs
        evaluate_directional_feature_batch = LineJEPAModule.evaluate_directional_feature_batch
        evaluate_directional_line_batch = LineJEPAModule.evaluate_directional_line_batch

        def encode_directional_environment_batch(self, points, *, target_encoder=False):
            return points.mean(dim=1)

    model = DummyDirectionalModel()
    points = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3) / 72.0
    line_t = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    line_perp = torch.zeros_like(line_t)
    direct = model.evaluate_directional_line_batch(
        {
            "points": points,
            "line_t": line_t,
            "line_perp": line_perp,
            "line_direction": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ),
        },
        target_index=1,
    )
    features = points.mean(dim=2)
    cached = model.evaluate_directional_feature_batch(
        line_features=features,
        line_t=line_t,
        line_perp=line_perp,
        target_features=features[:, 1],
        target_index=1,
    )
    for field in (
        "residual_prediction_error", "reconstruction_cosine_error",
        "residual_norm", "relative_residual_error", "reconstruction",
    ):
        torch.testing.assert_close(cached[field], direct[field])


def test_directional_per_snapshot_visualizations(tmp_path) -> None:
    rng = np.random.default_rng(5)
    atom_count, direction_count = 12, 8
    directions = fibonacci_sphere_directions(direction_count)
    source_a = tmp_path / "a.npy"
    source_b = tmp_path / "b.npy"
    np.save(source_a, rng.normal(size=(80, 3)).astype(np.float32))
    np.save(source_b, rng.normal(size=(80, 3)).astype(np.float32))
    cosine_error = rng.random((atom_count, direction_count), dtype=np.float32) + 0.1
    summaries = compute_directional_error_summaries(cosine_error, directions)
    arrays = {
        "coords": rng.normal(size=(atom_count, 3)).astype(np.float32),
        "source_slots": np.asarray([0] * 6 + [1] * 6, dtype=np.int64),
        "source_names_by_slot": np.asarray(["sample_a", "sample_b"]),
        "source_paths_by_slot": np.asarray([str(source_a), str(source_b)]),
        "analysis_sample_indices": np.arange(atom_count, dtype=np.int64),
        "atom_ids": np.arange(100, 100 + atom_count, dtype=np.int64),
        "directions": directions,
        "prediction_cosine_error": cosine_error,
        "mean_prediction_cosine_error": summaries["mean_error"],
        "relative_directional_variation": summaries["relative_directional_variation"],
        "context_coords": rng.normal(size=(14, 3)).astype(np.float32),
        "context_source_slots": np.asarray([0] * 7 + [1] * 7, dtype=np.int64),
        "context_atom_ids": np.arange(200, 214, dtype=np.int64),
        "context_analysis_sample_indices": np.arange(14, dtype=np.int64),
        "context_evaluated_mask": np.asarray(
            [True] * 6 + [False] + [True] * 6 + [False]
        ),
    }
    artifacts = render_directional_line_jepa_visualizations(
        arrays,
        tmp_path / "interactive",
        include_plotlyjs="cdn",
    )

    assert len(artifacts) == 7
    assert "interactive_report_index" in artifacts
    for path in artifacts.values():
        assert Path(path).is_file()
    index_html = Path(artifacts["interactive_report_index"]).read_text()
    assert "Prediction error" in index_html
    assert "Directional sensitivity" in index_html
    assert "All snapshots" not in index_html
    prediction_html = Path(
        artifacts["snapshot_sample_a_prediction_error"]
    ).read_text()
    assert "evaluated=6, excluded=1" in prediction_html
    profiles_html = Path(artifacts["snapshot_sample_a_directional_profiles"]).read_text()
    assert "Azimuth (degrees)" in profiles_html
    assert "Unit-direction sphere" in profiles_html
    assert '\"type\":\"cone\"' not in profiles_html
    assert '\"type\":\"mesh3d\"' not in profiles_html


def test_obsolete_visualization_schema_is_rejected(tmp_path) -> None:
    rng = np.random.default_rng(9)
    directions = fibonacci_sphere_directions(8)
    arrays = {
        "coords": rng.normal(size=(5, 3)).astype(np.float32),
        "source_names": np.asarray(["obsolete"] * 5),
        "source_paths": np.asarray([str(tmp_path / "obsolete.npy")] * 5),
        "analysis_sample_indices": np.arange(5, dtype=np.int64),
        "atom_ids": np.arange(5, dtype=np.int64),
        "directions": directions,
        "reconstruction_cosine_error": rng.random((5, 8), dtype=np.float32),
    }

    with pytest.raises(KeyError, match="current directional_line_jepa schema"):
        render_directional_line_jepa_visualizations(
            arrays, tmp_path / "obsolete_report", include_plotlyjs="cdn"
        )


def test_directional_summaries_recover_max_axis_and_anisotropy() -> None:
    directions = fibonacci_sphere_directions(256)
    errors = np.square(directions[:, 0])[None, :]

    summaries = compute_directional_error_summaries(errors, directions)

    max_direction = summaries["max_error_direction"][0]
    assert abs(float(max_direction[0])) > 0.99
    assert float(summaries["anisotropy_scalar"][0]) > 0.4
    np.testing.assert_allclose(
        summaries["anisotropy_tensor"][0],
        summaries["anisotropy_tensor"][0].T,
        atol=1.0e-7,
    )


def test_directional_harmonics_separate_polar_and_axial_signatures() -> None:
    directions = fibonacci_sphere_directions(256)
    polar_errors = (1.0 + 0.4 * directions[:, 0])[None, :]
    axial_errors = (1.0 + 0.5 * (directions[:, 1] ** 2 - 1.0 / 3.0))[None, :]

    polar = compute_directional_error_summaries(polar_errors, directions)
    axial = compute_directional_error_summaries(axial_errors, directions)

    assert float(polar["polar_anisotropy"][0]) > 10.0 * float(polar["axial_anisotropy"][0])
    assert abs(float(polar["polar_direction"][0, 0])) > 0.99
    assert float(polar["polar_share"][0]) > 0.99
    assert float(axial["axial_anisotropy"][0]) > 10.0 * float(axial["polar_anisotropy"][0])
    assert abs(float(axial["axial_direction"][0, 1])) > 0.99
    assert float(axial["polar_share"][0]) < 0.01
    assert float(polar["harmonic_explained_fraction"][0]) > 0.999
    assert float(axial["harmonic_explained_fraction"][0]) > 0.999


def test_discovery_metric_set_is_compact_and_nonredundant() -> None:
    assert set(_DISCOVERY_METRICS) == {
        "mean_prediction_cosine_error", "relative_directional_variation",
    }


def test_cosine_error_summary_has_clear_prediction_and_directional_meaning() -> None:
    directions = fibonacci_sphere_directions(4)
    errors = np.asarray(
        [[0.2, 0.2, 0.2, 0.2], [0.1, 0.3, 0.5, 0.7]], dtype=np.float32
    )

    summaries = compute_directional_error_summaries(errors, directions)

    np.testing.assert_allclose(summaries["mean_error"], [0.2, 0.4])
    np.testing.assert_allclose(summaries["max_error"], [0.2, 0.7])
    np.testing.assert_allclose(summaries["relative_directional_variation"][0], 0.0)
    np.testing.assert_allclose(
        summaries["relative_directional_variation"][1], np.std(errors[1]) / 0.4
    )


def test_chunked_directional_summaries_match_single_batch() -> None:
    rng = np.random.default_rng(17)
    directions = fibonacci_sphere_directions(16)
    errors = rng.random((11, 16), dtype=np.float32)

    expected = compute_directional_error_summaries(errors, directions)
    actual = compute_directional_error_summaries_chunked(
        errors, directions, atom_chunk_size=3
    )

    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1.0e-6, atol=1.0e-7)


def test_uncapped_selection_keeps_every_cache_row() -> None:
    selected, names = _selected_analysis_indices(
        source_groups=[
            ("a", np.asarray([0, 1, 2], dtype=np.int64)),
            ("b", np.asarray([3, 4], dtype=np.int64)),
        ],
        sample_count=5,
        max_atoms_total=0,
        seed=123,
    )

    np.testing.assert_array_equal(selected, np.arange(5))
    np.testing.assert_array_equal(names, ["a", "a", "a", "b", "b"])


def test_phase_z_scores_and_joint_hardness_are_finite() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 2.0], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    z_scores = phase_conditioned_z_scores(values, labels)
    np.testing.assert_allclose(z_scores, [-1.0, 1.0, 0.0, 0.0])
    hardness = compute_joint_hardness_score(
        {"mean_error": values, "phase_conditioned_z": z_scores},
        weights={"mean_error": 1.0, "phase_conditioned_z": 0.5},
    )
    assert np.isfinite(hardness).all()
    assert np.all((hardness >= 0.0) & (hardness <= 1.0))


def test_explicit_static_lines_keep_requested_atom_at_center(tmp_path) -> None:
    grid = np.arange(-4, 5, dtype=np.float32)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    source_path = tmp_path / "lattice.npy"
    np.save(source_path, points)
    dataset = LineStaticPointCloudDataset(
        root=str(tmp_path),
        data_files=[source_path.name],
        radius=1.8,
        num_points=16,
        line_atoms=5,
        line_candidate_atoms=200,
        line_samples_per_file=1,
        drop_edge_samples=False,
        line_min_separation_radius_factor=0.0,
        normalize=False,
    )
    source_slots, atom_ids, _ = dataset.resolve_explicit_centers(
        source_names=[tmp_path.name],
        center_coords=np.zeros((1, 3), dtype=np.float32),
    )

    batch = dataset.build_explicit_direction_batch(
        source_slots=np.repeat(source_slots, 3),
        center_atom_ids=np.repeat(atom_ids, 3),
        directions=np.eye(3, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        batch["line_atom_ids"][:, dataset.target_line_index].numpy(),
        np.repeat(atom_ids, 3),
    )
    np.testing.assert_allclose(
        batch["line_t"].numpy(),
        np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]] * 3, dtype=np.float32),
    )
    geometry_only = dataset.build_explicit_direction_batch(
        source_slots=np.repeat(source_slots, 3),
        center_atom_ids=np.repeat(atom_ids, 3),
        directions=np.eye(3, dtype=np.float32),
        materialize_points=False,
    )
    assert "points" not in geometry_only
    np.testing.assert_array_equal(geometry_only["line_atom_ids"], batch["line_atom_ids"])
    environments = dataset.build_atom_environment_batch(
        source_slots=np.repeat(source_slots, 3 * dataset.line_atoms),
        atom_ids=batch["line_atom_ids"].numpy().reshape(-1),
    )
    np.testing.assert_allclose(
        environments.reshape_as(batch["points"]).numpy(), batch["points"].numpy()
    )

    endpoint_batch = dataset.build_explicit_direction_batch(
        source_slots=source_slots,
        center_atom_ids=atom_ids,
        directions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        target_index=0,
    )
    assert int(endpoint_batch["line_atom_ids"][0, 0]) == int(atom_ids[0])
    endpoint_t = endpoint_batch["line_t"].numpy()[0]
    assert float(endpoint_t[0]) == 0.0
    assert np.all(endpoint_t[1:] > 0.0)
    assert np.all(np.diff(endpoint_t) >= 0.0)

    cached_candidates = dataset.query_explicit_line_candidates(
        source_slots=source_slots,
        center_atom_ids=atom_ids,
    )
    cached_endpoint_batch = dataset.build_explicit_direction_batch(
        source_slots=source_slots,
        center_atom_ids=atom_ids,
        directions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        target_index=0,
        candidate_indices=cached_candidates,
    )
    np.testing.assert_array_equal(
        cached_endpoint_batch["line_atom_ids"],
        endpoint_batch["line_atom_ids"],
    )


def test_fixed_slot_static_lines_control_radial_positions(tmp_path) -> None:
    grid = np.arange(-5, 6, dtype=np.float32)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    source_path = tmp_path / "fixed_lattice.npy"
    np.save(source_path, points)
    dataset = LineStaticPointCloudDataset(
        root=str(tmp_path),
        data_files=[source_path.name],
        radius=2.0,
        num_points=16,
        line_atoms=5,
        line_candidate_atoms=500,
        line_samples_per_file=1,
        drop_edge_samples=False,
        line_selection_method="fixed_slots",
        line_min_separation_radius_factor=0.0,
        line_slot_spacing_radius_factor=0.5,
        line_fixed_slot_max_deviation_radius_factor=0.1,
        normalize=False,
    )
    source_slots, atom_ids, _ = dataset.resolve_explicit_centers(
        source_names=[tmp_path.name],
        center_coords=np.zeros((1, 3), dtype=np.float32),
    )
    centered = dataset.build_explicit_direction_batch(
        source_slots=source_slots,
        center_atom_ids=atom_ids,
        directions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        target_index=dataset.target_line_index,
        materialize_points=False,
    )
    endpoint = dataset.build_explicit_direction_batch(
        source_slots=source_slots,
        center_atom_ids=atom_ids,
        directions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        target_index=0,
        materialize_points=False,
    )

    np.testing.assert_allclose(centered["line_t"].numpy()[0], [-2, -1, 0, 1, 2])
    np.testing.assert_allclose(endpoint["line_t"].numpy()[0], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(centered["line_perp"].numpy(), 0.0)
    np.testing.assert_allclose(endpoint["line_perp"].numpy(), 0.0)


def test_fixed_slot_training_resamples_an_invalid_random_direction(tmp_path) -> None:
    grid = np.arange(-5, 6, dtype=np.float32)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    source_path = tmp_path / "retry_lattice.npy"
    np.save(source_path, points)
    dataset = LineStaticPointCloudDataset(
        root=str(tmp_path),
        data_files=[source_path.name],
        radius=2.0,
        num_points=16,
        line_atoms=5,
        line_candidate_atoms=500,
        line_samples_per_file=1,
        drop_edge_samples=False,
        line_selection_method="fixed_slots",
        line_slot_spacing_radius_factor=0.5,
        line_fixed_slot_max_deviation_radius_factor=None,
        line_direction_max_retries=2,
        deterministic_lines=True,
        normalize=False,
    )
    original_selector = dataset._select_line_atoms_from_candidates
    selection_calls = 0

    def fail_first_direction(**kwargs):
        nonlocal selection_calls
        selection_calls += 1
        if selection_calls == 1:
            raise FixedSlotDirectionError("forced invalid ray for retry test")
        return original_selector(**kwargs)

    dataset._select_line_atoms_from_candidates = fail_first_direction
    batch = dataset[0]

    assert selection_calls == 2
    assert batch["points"].shape == (5, 16, 3)
    assert int(batch["line_atom_ids"][dataset.target_line_index]) == int(
        batch["anchor_atom_id"]
    )


def test_directional_anchor_filter_excludes_boundary_atoms(tmp_path) -> None:
    grid = np.arange(-5, 6, dtype=np.float32)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    source_path = tmp_path / "anchor_domain_lattice.npy"
    np.save(source_path, points)
    dataset = LineStaticPointCloudDataset(
        root=str(tmp_path),
        data_files=[source_path.name],
        radius=2.0,
        num_points=16,
        line_atoms=5,
        line_candidate_atoms=500,
        line_samples_per_file=1,
        drop_edge_samples=True,
        edge_drop_layers=2,
        line_selection_method="fixed_slots",
        line_slot_spacing_radius_factor=0.5,
        line_fixed_slot_max_deviation_radius_factor=0.1,
        normalize=False,
    )
    source_slots, atom_ids, _ = dataset.resolve_explicit_centers(
        source_names=[tmp_path.name, tmp_path.name],
        center_coords=np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
    )

    valid = _valid_directional_line_anchor_mask(
        dataset,
        source_slots=source_slots,
        atom_ids=atom_ids,
    )

    np.testing.assert_array_equal(valid, [True, False])
