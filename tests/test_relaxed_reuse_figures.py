"""Fixed-event cohort coverage and actual archived lead-time semantics."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.research.structured_context.reuse_figures import matched_events


def corpus(frames):
    return SimpleNamespace(plan={'anchors':frames},rows=[(7,a,c,520) for a in range(len(frames)) for c in (0,1)],arrays={7:{'onset':np.array([100,801])}})


def test_fixed_event_actual_origins_and_surviving_control():
    c=corpus([32,48,64,80])
    result=matched_events(c,list(range(len(c.rows))),[12,24,36,48],12,9)
    assert len(result['records'])==1
    assert result['records'][0]['origins_frames']==[80,64,48,32]
    assert result['records'][0]['control_center']==1
    np.testing.assert_array_equal(result['leads'],[[15,27,39,51]])
    assert len(set(result['rows'][0,:,0]))==4


def test_missing_offset_is_not_filled_or_interpolated():
    c=corpus([32,64,80])
    with pytest.raises(ValueError,match='No common archived'):
        matched_events(c,list(range(len(c.rows))),[12,24,36,48],12,9)
