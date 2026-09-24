import numpy as np
from sklearn.metrics import average_precision_score

from src.research.structural_state_onset_review import weighted_ap


def test_ap_matches_sklearn_with_ties_and_resampled_root_weights():
    actual = np.array([0,1,0,1,1,0], dtype=bool)
    risk = np.array([.2,.2,.7,.7,.1,.1])
    weights = np.array([[1,1,1,1,1,1], [0,0,2,2,1,1], [3,3,0,0,0,0]], dtype=float)
    np.testing.assert_allclose(weighted_ap(actual,risk,weights),
        [average_precision_score(actual,risk,sample_weight=w) for w in weights])
    tied = weighted_ap(actual, np.ones(6), weights)
    np.testing.assert_allclose(tied, weights@actual/weights.sum(1))


def test_no_positive_ap_is_undefined():
    result = weighted_ap(np.array([False,True]), np.array([.3,.2]), np.array([[1.,0.]]))
    assert np.isnan(result[0])
