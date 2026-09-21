from types import SimpleNamespace
import numpy as np
from src.research.structured_context.event_offsets import cohort, timing, evaluate


def test_cohort_keeps_same_event_and_control_and_never_uses_late_origin():
    anchors = list(range(64, 161, 4)); rows = [(1, a, c, 500) for a in range(len(anchors)) for c in range(3)]
    arrays = {1:dict(onset=np.array([143, 180, 240]), labels=np.zeros((3, 240)))}
    c = SimpleNamespace(plan=dict(anchors=anchors), rows=rows, arrays=arrays)
    m = cohort(c, np.arange(len(rows)), [3, 6, 12, 48], 7)
    assert len(m['records']) == 1
    assert m['records'][0]['center'] == 0
    assert np.all(m['leads'][0]-[3, 6, 12, 48] == 2.25)
    pair = m['rows'][0]
    assert len({rows[i][2] for i in pair[:, 1]}) == 1
    for case, control in pair:
        assert rows[case][:2] == rows[control][:2]


def test_timing_penalizes_survival_mass_without_true_time_truncation():
    cdf = np.zeros((3, 128)); cdf[0, 15:] = 1; cdf[1, 15:] = .5
    restricted, conditional = timing(cdf)
    np.testing.assert_allclose(restricted, [12, 54, 96])
    np.testing.assert_allclose(conditional[:2], [12, 12])
    assert np.isnan(conditional[2])


def test_balanced_ap_and_source_bootstrap_preserve_pairs():
    cdf = np.zeros((4, 128)); cdf[[0, 2], 15:] = 1; cdf[[1, 3], 15:] = .01
    m = dict(rows=np.array([[[0, 1]], [[2, 3]]]), sources=np.array([1, 2]), leads=np.full((2, 1), 12.))
    result, _ = evaluate({'model':dict(test_cdf=cdf)}, m, [12], draws=20)
    assert result['model']['12']['matched_ap']['value'] == 1
    assert result['model']['12']['restricted_timing_mae_ps']['value'] == 0
    np.testing.assert_allclose(result['model']['12']['matched_ap']['ci95'], [1, 1])


def test_duplicate_events_within_one_source_do_not_change_its_total_weight():
    cdf = np.zeros((4, 128)); cdf[:, 15:] = np.array([.9, .2, .3, .8])[:, None]
    original = dict(rows=np.array([[[0, 1]], [[2, 3]]]), sources=np.array([1, 2]), leads=np.full((2, 1), 12.))
    duplicate = dict(rows=np.array([[[0, 1]], [[0, 1]], [[2, 3]]]), sources=np.array([1, 1, 2]), leads=np.full((3, 1), 12.))
    a, _ = evaluate({'model':dict(test_cdf=cdf)}, original, [12], draws=20)
    b, _ = evaluate({'model':dict(test_cdf=cdf)}, duplicate, [12], draws=20)
    for key in ('matched_ap', 'restricted_timing_mae_ps', 'conditional_timing_mae_ps'):
        np.testing.assert_allclose(a['model']['12'][key]['value'], b['model']['12'][key]['value'])
        np.testing.assert_allclose(a['model']['12'][key]['ci95'], b['model']['12'][key]['ci95'])
