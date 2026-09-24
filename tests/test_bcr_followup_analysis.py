import numpy as np
import pytest

from src.research.bcr_followup_analysis import paired_change


def test_paired_rmse_bootstrap_matches_known_scaling_and_is_deterministic():
    a=np.arange(1,25,dtype=float).reshape(8,3)
    roots=np.repeat(['a','b','c','d'],2); temp=np.repeat([400,400,500,500],2)
    score=paired_change(a,2*a,roots,temp,1000,14)
    assert score==paired_change(a,2*a,roots,temp,1000,14)
    assert score['reference_rmse']==pytest.approx(np.sqrt(np.square(a).mean()))
    assert score['change_percent']==pytest.approx(100)
    assert score['ci95_lower_percent']==pytest.approx(100)
    assert score['ci95_upper_percent']==pytest.approx(100)
    assert score['roots_improved']==0
    identical=paired_change(a,a,roots,temp,1000,14)
    assert identical['change_percent']==identical['ci95_lower_percent']==identical['ci95_upper_percent']==0


def test_rmse_keeps_observation_weighting_with_unequal_root_counts():
    a=np.array([[1.],[1.],[1.],[3.]])
    b=np.array([[2.],[2.],[2.],[0.]])
    result=paired_change(a,b,['a','a','a','b'],[400]*4,1000,14)
    assert result['reference_rmse']==pytest.approx(np.sqrt(3))
    assert result['candidate_rmse']==pytest.approx(np.sqrt(3))
    assert result['change_percent']==0
    assert result['roots_improved']==1


def test_root_cannot_cross_temperature_strata():
    a=np.ones((2,1))
    with pytest.raises(ValueError,match='multiple temperatures'):
        paired_change(a,a,['a','a'],[400,500],100,1)
    with pytest.raises(ValueError,match='align'):
        paired_change(a,a[:1],['a','a'],[400,400],100,1)
