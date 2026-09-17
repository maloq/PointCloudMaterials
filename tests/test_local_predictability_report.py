import numpy as np
import pytest

from src.research.local_predictability.report import aligned, paired_rows


def test_alignment_reorders_and_checks_labels():
    a = dict(source=np.array([1, 1, 2]), center=np.array([10, 11, 12]),
             anchor=np.array([4, 4, 4]), event_bin=np.array([0, 6, 2]))
    b = {k: v[[2, 0, 1]] for k, v in a.items()}
    np.testing.assert_array_equal(aligned(a, b, ('event_bin',)), [1, 2, 0])
    b['event_bin'][0] = 5
    with pytest.raises(AssertionError):
        aligned(a, b, ('event_bin',))


def test_alignment_rejects_duplicates_and_missing_rows():
    a = dict(source=np.array([1, 2]), center=np.array([10, 10]), anchor=np.array([4, 4]))
    b = {k: v[[0, 0]] for k, v in a.items()}
    with pytest.raises(ValueError, match='Unpaired or duplicate'):
        aligned(a, b, ())
    with pytest.raises(ValueError, match='Unpaired or duplicate'):
        aligned(a, {k: v[:1] for k, v in a.items()}, ())


def test_paired_difference_weights_sources_not_windows():
    reference = dict(source=np.array([1, 1, 1, 2]))
    result = paired_rows('candidate', reference, np.array([-1., -1., -1., 3.]),
                         {1: 400, 2: 500}, metric='mse', horizon=9)
    assert result['difference'] == 1.
    # One source per temperature stratum gives a degenerate source interval.
    assert result['ci95_lower'] == result['ci95_upper'] == 1.
