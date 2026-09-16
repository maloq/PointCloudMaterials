"""Scientific failure modes: source centering, block weighting and PTM sentinels."""

import numpy as np

from src.research.mace_encoder_diagnostics.analyze import balanced_errors, centered, source_gain


def test_equal_homology_weight_not_equal_pixel_weight():
    target=np.zeros((3,144)); prediction=target.copy(); prediction[:,:16]=3
    np.testing.assert_allclose(balanced_errors(prediction,target,np.array([3.,1.,1.])),1/3)


def test_context_trend_cannot_explain_centered_local_variation():
    y=np.array([[99.,101.],[101.,99.],[-101.,-99.],[-99.,-101.]])
    groups=np.array([0,0,1,1]); p=np.array([[100.,100.],[100.,100.],[-100.,-100.],[-100.,-100.]])
    yc=centered(y,groups);pc=centered(p,groups)
    np.testing.assert_array_equal(pc,0)
    assert 1-np.sum((pc-yc)**2)/np.sum(yc**2)==0


def test_source_bootstrap_does_not_weight_duplicate_patch_rows():
    reference=np.array([1.,1.,4.]);candidate=np.array([.5,.5,1.]);sources=np.array([0,0,1])
    result=source_gain(reference,candidate,sources,np.random.default_rng(8),1000)
    assert result['sources']==2
    assert result['skill']==.7


def test_ptm_fcc_margin_matches_existing_labels():
    from ase.build import bulk
    from scipy.spatial import cKDTree
    from src.research.mace_encoder_diagnostics.extract import structural
    from src.research.smooth_temporal_encoder.prepare import ptm_labels
    atoms=bulk('Al','fcc',a=4.05,cubic=True).repeat((5,5,5))
    x=atoms.positions;lengths=atoms.cell.lengths()
    centers=np.array([0,10]);ids=cKDTree(x,boxsize=lengths).query(x[centers],k=80)[1]
    clouds=x[ids]-x[centers,None];clouds-=lengths*np.round(clouds/lengths)
    obs,labels=structural(clouds,.1)
    np.testing.assert_array_equal(labels,ptm_labels(clouds[:,1:]/10,.1))
    np.testing.assert_array_equal(labels,1)
    np.testing.assert_allclose(obs[:,0],0,atol=1e-7)
    np.testing.assert_allclose(obs[:,1],.1,atol=1e-7)
