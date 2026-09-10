"""Identity and periodic geometry of histories paired to relaxed anchor targets."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data_utils.mace_history import history_clouds
from src.data_utils.mace_relaxed import paired_clouds


def test_history_tracks_anchor_ids_across_periodic_crossings():
    rng = np.random.default_rng(12)
    low = np.array([-2., 3., -1.], dtype=np.float32)
    lengths = np.array([12., 12., 12.], dtype=np.float32)
    base = rng.uniform(0, 12, (100, 3))
    positions = np.stack([np.mod(base + step * np.array([1.1, .4, .2]), lengths) + low
                          for step in range(5)]).astype(np.float16)
    source = SimpleNamespace(positions=positions, box_low=np.tile(low, (5, 1)),
        box_high=np.tile(low + lengths, (5, 1)), frame_count=5, root=Path('test_trajectory'))
    centers = np.array([0, 1])
    anchor = positions[-1].astype(float) - low
    hot, _, neighbor_rows, _ = paired_clouds(anchor, anchor, lengths, centers)
    history, error = history_clouds(source, centers, neighbor_rows + 1, 4, np.arange(-4, 1))
    np.testing.assert_array_equal(history[:, -1], hot)
    np.testing.assert_array_equal(history[:, :, 0], 0)
    assert history.dtype == np.float16 and error < .004
    # Every frame is gathered by the anchor's identities, rather than reselected.
    expected = positions[0, neighbor_rows].astype(float) - positions[0, centers, None].astype(float)
    expected -= lengths * np.round(expected/lengths)
    np.testing.assert_array_equal(history[:, 0], expected.astype(np.float16))
    with pytest.raises(ValueError, match='exceeds trajectory'):
        history_clouds(source, centers, neighbor_rows + 1, 2, np.arange(-4, 1))
