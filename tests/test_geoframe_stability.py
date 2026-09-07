from __future__ import annotations

import numpy as np

from src.temporal_vamp.geoframe_stability import (
    row_cosine,
    same_atom_retrieval,
    sibling_distances,
    temporal_stability_table,
)


def test_stability_metrics_preserve_identical_atoms() -> None:
    initial = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    top1, rank = same_atom_retrieval(initial, initial)
    assert top1 == 1.0
    assert rank == 1.0
    np.testing.assert_allclose(row_cosine(initial, initial), 1.0)


def test_temporal_stability_reports_known_drift_and_sibling_spread() -> None:
    initial = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    future = initial.copy()
    future[:, :, 0] += np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
    embeddings = np.stack([initial, future], axis=1)
    scalar_shape = embeddings.shape[:3]
    retention = np.ones(scalar_shape, dtype=np.float32)
    rms = np.zeros(scalar_shape, dtype=np.float32)
    rows, distances, sibling = temporal_stability_table(
        embeddings,
        np.asarray([0.0, 0.03]),
        retention,
        rms,
        np.zeros(4, dtype=np.int64),
        np.asarray([1.0, 2.0, 3.0]),
        seed=7,
    )
    assert rows[0]["embedding_distance_mean"] == 0.0
    np.testing.assert_allclose(
        distances[:, 1, :],
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
        rtol=1e-6,
        atol=1e-7,
    )
    assert sibling[0] == 0.0
    assert sibling[1] > 0.0
    np.testing.assert_allclose(
        sibling_distances(initial, np.zeros(4, dtype=np.int64)), 0.0
    )
