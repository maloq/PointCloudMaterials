import numpy as np
import pytest
from src.data.structural_pretraining.prepare import geometry_packet, offsets, registry_sources
from src.data.predictive_memory.targets import physical_packet


def test_geometry_packet_matches_actual_physical_producer():
    rng=np.random.default_rng(10)
    x=rng.uniform(-7,7,(250,3)); x[0]=0
    reference=physical_packet(x,rng.normal(size=x.shape))
    np.testing.assert_allclose(geometry_packet(x),np.r_[reference[:80],reference[112:117]],rtol=2e-6,atol=1e-6)


def test_geometry_invariant_and_preserves_scale():
    rng=np.random.default_rng(12); x=rng.uniform(-4,4,(90,3)); x[0]=0
    q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    np.testing.assert_allclose(geometry_packet(x),geometry_packet(x@q),rtol=2e-6,atol=1e-6)
    assert np.linalg.norm(geometry_packet(x)-geometry_packet(x*1.1))>.1


def test_periodic_chart_retains_center_and_crossing_atoms():
    x=np.array([[.1,0,0],[9.9,0,0],[1.,1.,1.]])
    np.testing.assert_allclose(offsets(x,0,[0,1,2],np.full(3,10)),[[0,0,0],[-.2,0,0],[.9,1,1]],atol=1e-6)


def test_bad_geometry_fails():
    with pytest.raises(ValueError):geometry_packet(np.zeros((2,3)))
