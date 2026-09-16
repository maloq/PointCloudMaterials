"""Scientifically consequential split and gradient boundaries for the native study."""
import numpy as np
import pytest
import torch

from src.research.mace_velocity.data_amount_data import batch, nested_selection
from src.research.mace_velocity.data_amount_eval import reference_trace
from src.research.mace_velocity.data_amount_model import Directions, objective
from src.research.mace_velocity.train import Heads


def test_nested_independent_sources_keep_common_core_and_never_include_holdout():
    records=[dict(source=dict(id=i,lineage=f'independent_melt_{i}',split='train' if i<90 else 'val',
                              temperature_K=[400,450,500,510,520][i%5])) for i in range(120)]
    result=nested_selection(records,[10,25,45,90],[17,18],19)
    for seed in (17,18):
        previous=set(result['core_source_ids'])
        for fit in [x for x in result['fits'] if x['seed']==seed]:
            ids=set(fit['source_ids'])
            assert previous<=ids and len(ids)==fit['count'] and not ids.intersection(result['validation_source_ids'])
            assert all(sum(i%5==t for i in ids)==fit['count']//5 for t in range(5))
            previous=ids
    records[1]['source']['lineage']=records[0]['source']['lineage']
    with pytest.raises(ValueError,match='distinct preparation'):
        nested_selection(records,[10,25,45,90],[17],19)


def test_clouds_labels_and_times_share_center_major_order():
    sources={7:dict(clouds=[[(c,t) for t in range(9)] for c in range(4)],
                    raw_target=np.arange(4*9*169).reshape(4,9,169),time_ps=np.arange(9)*.75)}
    clouds,y,times=batch(sources,[7],[3])
    assert clouds==[(c,t) for c in range(4) for t in (3,4,5)]
    np.testing.assert_array_equal(y,sources[7]['raw_target'][:,3:6])
    np.testing.assert_array_equal(times,np.tile(np.array([2.25,3.,3.75]),(4,1)))


def test_auxiliary_direction_fit_cannot_update_embedding_when_motion_terms_disabled():
    torch.manual_seed(31)
    z=torch.randn(24,304,requires_grad=True);targets=torch.randn(8,3,169)
    times=torch.arange(3,dtype=torch.float64).repeat(8,1)*.75
    heads=Heads();directions=Directions(8)
    c=dict(motion_weight=1.,temporal_weight=.03,curvature_weight=.03,direction_weight=.1,variance_weight=.01)
    value,_=objective(c,heads,directions,z,targets,times,0.)
    expected=torch.autograd.grad(value,z,retain_graph=False)[0]
    with torch.no_grad():
        for p in directions.parameters(): p.add_(torch.randn_like(p)*.05)
    actual=torch.autograd.grad(objective(c,heads,directions,z,targets,times,0.)[0],z)[0]
    torch.testing.assert_close(actual,expected)
    changed=torch.autograd.grad(objective(c,heads,directions,z,targets,times,1.)[0],z)[0]
    assert not torch.allclose(changed,expected)


def test_within_context_normalizer_removes_phase_offset_and_detects_collapse():
    rng=np.random.default_rng(0);z=rng.normal(size=(2,4,9,8));mask=np.ones((2,4,9),bool)
    baseline=reference_trace(z,mask,True);shifted=z.copy();shifted[1]+=100
    assert reference_trace(shifted,mask,True)==pytest.approx(baseline)
    assert reference_trace(shifted,mask)>reference_trace(z,mask)*100
    with pytest.raises(ValueError,match='Collapsed'):
        reference_trace(np.zeros_like(z),mask,True)
