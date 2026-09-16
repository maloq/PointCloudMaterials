import numpy as np

from src.research.mace_context.cluster_diagnosis import agreement


def test_agreement_corrects_for_cluster_occupancy():
    labels = np.array([0, 0, 1, 1])
    independent = np.tile(np.arange(4), (4, 1))
    score = agreement(labels, independent, np.ones(4, bool))
    assert score['same_label'] == score['chance'] == .5
    assert score['adjusted_agreement'] == 0
    grouped = np.array([[1], [0], [3], [2]])
    assert agreement(labels, grouped, np.ones(4, bool))['adjusted_agreement'] == 1


def test_single_label_is_not_reported_as_perfect_adjusted_coherence():
    result = agreement(np.zeros(3, int), np.array([[1], [2], [0]]), np.ones(3, bool))
    assert result['same_label'] == result['chance'] == 1
    assert result['adjusted_agreement'] is None


def test_region_mask_applies_to_both_edge_endpoints():
    result = agreement(np.array([0, 1, 2]), np.array([[1,2], [0,2], [0,1]]), np.array([True,True,False]))
    assert result['edges'] == 2 and result['centers'] == 2
    assert result['same_label'] == 0 and result['chance'] == .5
