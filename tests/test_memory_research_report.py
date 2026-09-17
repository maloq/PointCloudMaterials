import pytest

from src.research.memory_report import cross_seed_mean, numeric


def test_matched_seed_mean_excludes_other_models_and_preserves_pairing():
    rows = [dict(model='history', seed=17, nll=0.8),
            dict(model='history', seed=18, nll=1.0),
            dict(model='snapshot', seed=17, nll=3.0)]
    assert cross_seed_mean(rows, 'history', 'nll', [17, 18]) == pytest.approx(0.9)
    with pytest.raises(ValueError, match='incomplete/duplicated'):
        cross_seed_mean(rows, 'snapshot', 'nll', [17, 18])
    with pytest.raises(ValueError, match='incomplete/duplicated'):
        cross_seed_mean(rows + [rows[0]], 'history', 'nll', [17, 18])


def test_undefined_source_interval_is_never_converted_to_zero():
    parsed = numeric(dict(value='0.1', ci95_low='', ci95_high='', seeds='3', sources='1'))
    assert parsed == dict(value=0.1, ci95_low=None, ci95_high=None, seeds=3, sources=1)
