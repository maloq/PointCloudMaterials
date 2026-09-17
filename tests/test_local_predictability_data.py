import numpy as np
from src.research.local_predictability.data import shell_features
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows


def test_endpoint_event_needs_confirmation_after_horizon():
    crystal=np.zeros((1,16),bool);crystal[0,10:13]=True
    onset=first_sustained_onset(crystal,3)
    assert onset[0]==10
    assert first_sustained_onset(crystal[:,:11],3)[0]==11
    assert risk_windows(crystal,onset,np.array([4,7]),3).all()


def test_shell_invariants_and_smooth_boundary():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(1000,3));x*=rng.uniform(7,25,1000)[:,None]/np.linalg.norm(x,axis=1)[:,None]
    u=rng.normal(size=x.shape)
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    np.testing.assert_allclose(shell_features(x@rotation,u@rotation),shell_features(x,u),atol=1e-5)
    a=shell_features(np.vstack((x,[25-1e-5,0,0])),np.vstack((u,[1,0,0])))
    b=shell_features(np.vstack((x,[25+1e-5,0,0])),np.vstack((u,[1,0,0])))
    np.testing.assert_allclose(a,b,atol=1e-5)
