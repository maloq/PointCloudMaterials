from __future__ import annotations

import numpy as np

from src.temporal_vamp.geoframe_temporal_variability import (
    compute_increment_alignment,
    compute_smoothing_metrics,
    compute_time_metrics,
    embedding_rms_radius,
)


def test_sibling_decomposition_separates_shared_and_random_motion() -> None:
    # Two parents, two shots, three frames, one center, two embedding dimensions.
    values = np.zeros((4, 3, 1, 2), dtype=np.float32)
    branch_parent = np.asarray([0, 0, 1, 1], dtype=np.int64)
    # Shared x motion is exactly 1 and 2. Shot-specific y motion is +/- 0.5 and +/- 1.
    values[:, 1, 0, 0] = 1.0
    values[:, 2, 0, 0] = 2.0
    values[[0, 2], 1, 0, 1] = 0.5
    values[[1, 3], 1, 0, 1] = -0.5
    values[[0, 2], 2, 0, 1] = 1.0
    values[[1, 3], 2, 0, 1] = -1.0
    rows = compute_time_metrics(
        values,
        times_ps=np.asarray([0.0, 0.3, 0.6]),
        branch_parent=branch_parent,
        parent_temperatures=np.asarray([400.0, 500.0]),
        parent_roles=np.asarray(["transition_candidate", "transition_candidate"], dtype=object),
        rms_radius=embedding_rms_radius(values),
    )
    final = [
        row
        for row in rows
        if row["group_kind"] == "all" and row["time_ps"] == 0.6
    ][0]
    assert np.isclose(final["coherent_rms"], 2.0)
    assert np.isclose(final["sibling_dispersion_rms"], 1.0)
    assert np.isclose(final["stochastic_energy_fraction"], 0.2)
    assert np.isclose(final["conditional_variance_rms_unbiased"], np.sqrt(2.0))
    assert np.isclose(final["conditional_mean_change_rms_debiased"], np.sqrt(3.0))
    assert np.isclose(final["conditional_variance_energy_fraction_debiased"], 0.4)


def test_smoothing_reduces_alternating_embedding_jitter() -> None:
    values = np.zeros((1, 9, 1, 2), dtype=np.float32)
    values[0, :, 0, 0] = np.arange(9, dtype=np.float32)
    values[0, :, 0, 1] = np.asarray([1, -1, 1, -1, 1, -1, 1, -1, 1])
    rows = compute_smoothing_metrics(values, [1, 3], 0.3)
    assert rows[1]["adjacent_step_rms"] < rows[0]["adjacent_step_rms"]
    alignment = compute_increment_alignment(values)
    assert alignment["mean"] < 0.0
