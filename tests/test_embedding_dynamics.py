"""Controls distinguishing drift, collapse, observation gaps and state rank."""
import numpy as np
import pytest

from src.research.trajectory_stability.spectrum import analyze, spectrum, tracks, lag_pairs


def test_isotropic_and_rank_one_spectra():
    isotropic = spectrum(np.r_[np.eye(5), -np.eye(5)])
    assert isotropic['participation_rank'] == pytest.approx(5)
    assert isotropic['entropy_rank'] == pytest.approx(5)
    assert isotropic['d95'] == isotropic['numerical_rank'] == 5
    line = spectrum(np.arange(20.)[:, None]*np.array([[1., 2., -3.]]))
    assert line['participation_rank'] == pytest.approx(1)
    assert line['numerical_rank'] == line['d95'] == 1


def test_constant_movement_is_rank_one_but_centered_fluctuation_is_collapsed():
    velocity = np.tile([1., 3., 2.], (10, 1))
    movement, fluctuation = spectrum(velocity, centered=False), spectrum(velocity)
    assert movement['participation_rank'] == pytest.approx(1)
    assert fluctuation['collapsed']
    assert fluctuation['participation_rank'] is None
    assert fluctuation['numerical_rank'] == 0


def test_rank_is_invariant_to_rotation_scale_translation_but_not_feature_rescaling():
    rng = np.random.default_rng(91)
    z = rng.normal(size=(100, 4))*[1, 2, 3, 4]
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a, b = spectrum(z), spectrum(13*z@q+1e5)
    assert a['participation_rank'] == pytest.approx(b['participation_rank'])
    assert a['entropy_rank'] == pytest.approx(b['entropy_rank'])
    assert a['participation_rank'] != pytest.approx(spectrum(z/[1, 2, 3, 4])['participation_rank'])


def test_exact_lags_never_bridge_source_atom_or_interpolate_gaps():
    source = np.array(['a', 'b', 'a', 'a', 'b', 'a'])
    atom = np.array([1, 1, 2, 1, 1, 1])
    time = np.array([0., 0., 1., 2., 1., 5.])
    groups = tracks(source, atom, time)
    np.testing.assert_array_equal(lag_pairs(groups, time, 1.), [[1, 4]])
    np.testing.assert_array_equal(lag_pairs(groups, time, 3.), [[3, 5]])
    assert lag_pairs(groups, time, .75).shape == (0, 2)
    with pytest.raises(ValueError, match='Duplicate'):
        tracks(['a', 'a'], [1, 1], [0., 0.])


def example():
    # Fit varies in two coordinates; test tracks move linearly with distinct offsets.
    z = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                  [0, 0], [1, 0], [2, 0], [1000, 1000], [1001, 1000], [1002, 1000]], float)
    return z, np.array(['fit']*4+['dev']*6), np.array([0]*4+[1]*3+[2]*3), np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2], float)


def test_track_offsets_do_not_create_movement_and_reference_scale_is_frozen():
    z, source, atom, times = example()
    result = analyze(z, source, atom, times, np.arange(4), np.arange(4, 10), lags_ps=[1, 2, .75])
    block = result['domains']['all']
    assert block['within_track']['spectrum']['participation_rank'] == pytest.approx(1)
    assert block['lags']['1.0']['pairs'] == 4
    assert block['lags']['1.0']['rms_jump'] == pytest.approx(1)
    assert block['lags']['2.0']['rms_jump'] == pytest.approx(2)
    assert block['lags']['1.0']['movement']['participation_rank'] == pytest.approx(1)
    assert block['lags']['1.0']['fluctuation']['participation_rank'] is None
    assert block['lags']['0.75']['pairs'] == 0
    assert all(t['velocity_roughness'] == 0 and t['increment_cosine'] == pytest.approx(1) for t in result['per_track'])


def test_domains_require_both_endpoints_and_do_not_hide_missing_reference():
    z, source, atom, times = example()
    mask = np.array([False]*4+[True, True, False, True, False, False])
    out = analyze(z, source, atom, times, np.arange(4), np.arange(4, 10), lags_ps=[1], domains={'selected': mask})
    row = out['domains']['selected']['lags']['1.0']
    assert row['pairs'] == 1
    assert row['domain_reference_rms_jump'] is None
    assert row['rms_jump'] == pytest.approx(1)


def test_equal_source_weighting_and_centering_do_not_weight_longer_sources_more():
    # The third source is larger but still has weight 1/2 in test measurements.
    z = np.array([[0.], [2.], [0.], [2.], [0.], [4.], [8.]])
    source = np.array(['fit']*2+['a']*2+['b']*3)
    out = analyze(z, source, np.zeros(7), np.array([0., 1., 0., 1., 0., 1., 2.]),
                  np.arange(2), np.arange(2, 7), lags_ps=[1])
    assert out['domains']['all']['lags']['1.0']['rms_jump'] == pytest.approx(np.sqrt((4+16)/4))


def test_rank_ceiling_and_invalid_normalization_are_explicit():
    rng = np.random.default_rng(12)
    assert spectrum(rng.normal(size=(4, 128)))['rank_ceiling'] == 3
    z, source, atom, times = example()
    with pytest.raises(ValueError, match='disjoint'):
        analyze(z, source, atom, times, np.arange(4), np.arange(10), lags_ps=[1])
    z[:4] = 1
    with pytest.raises(ValueError, match='Collapsed training'):
        analyze(z, source, atom, times, np.arange(4), np.arange(4, 10), lags_ps=[1])


def test_circle_has_two_dimensional_state_and_movement():
    t = np.linspace(0, 2*np.pi, 100, endpoint=False)
    z = np.c_[np.cos(t), np.sin(t)]
    assert spectrum(z)['participation_rank'] == pytest.approx(2)
    assert spectrum(np.roll(z, -1, axis=0)-z, centered=False)['participation_rank'] == pytest.approx(2)
