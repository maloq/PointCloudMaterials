"""Checkpoint-support and periodic-identity tests for the relaxed forecast arm."""
import numpy as np
from src.research.structured_context.relaxed import samples
from src.research.relaxed_encoder.prepare import graph_arrays, paired_clouds


def test_samples_preserve_training_graphs():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(3,80,3)).astype(np.float32)*3
    x[:,0]=0
    arrays,_=graph_arrays(x,9.121389139452193)
    graphs=samples(x,9.121389139452193)
    for i,g in enumerate(graphs):
        lo,hi=arrays['offsets'][i:i+2];elo,ehi=arrays['edge_offsets'][i:i+2]
        np.testing.assert_array_equal(g['positions'][0],arrays['positions'][lo:hi])
        np.testing.assert_array_equal(g['edges'],arrays['edges'][:,elo:ehi])
        assert g['center']==0 and np.linalg.norm(g['positions'][0],axis=-1).max()<8


def test_quench_retains_observed_neighbor_identity_across_periodic_boundary():
    rng=np.random.default_rng(72);box=np.full(3,20.)
    hot=rng.uniform(0,20,(300,3));cold=np.mod(hot+rng.normal(0,.5,hot.shape),box)
    centers=np.array([0,119,208]);shift=np.array([19.,-.5,23.])
    a,b,ids=paired_clouds(hot,cold,box,centers)
    aa,bb,ids2=paired_clouds(np.mod(hot+shift,box),np.mod(cold+shift,box),box,centers)
    np.testing.assert_array_equal(ids,ids2)
    np.testing.assert_allclose(a,aa,atol=1e-6);np.testing.assert_allclose(b,bb,atol=1e-6)
    np.testing.assert_array_equal(ids[:,0],centers)
    expected=cold[ids]-cold[centers,None];expected-=box*np.rint(expected/box)
    np.testing.assert_allclose(b,expected,atol=1e-6)


def test_reused_histories_are_causal_real_and_bounded():
    from src.research.structured_context.reuse import histories
    result=histories([32,64,80,128,176,224,400],[64,80,128,176,224,400],3,96)
    assert result=={80:[32,64,80],128:[64,80,128],176:[80,128,176],224:[128,176,224]}
    for anchor,frames in result.items():
        assert len(frames)==len(set(frames))==3
        assert frames[-1]==anchor and anchor-frames[0]<=96


def test_restrict_corpus_keeps_original_anchor_and_event_identity():
    from types import SimpleNamespace
    from src.research.structured_context.reuse_data import restrict_corpus
    corpus=SimpleNamespace(plan={'anchors':[64,80,128]},rows=[(s,a,0,500) for s in range(4) for a in range(3)],
        splits={role:list(range(i*3,(i+1)*3)) for i,role in enumerate(['train','selection','calibration','test'])},events=np.arange(12))
    restrict_corpus(corpus,{str(s):{'80':[32,64,80]} for s in range(4)})
    assert corpus.rows==[(s,1,0,500) for s in range(4)]
    np.testing.assert_array_equal(corpus.events,[1,4,7,10])
    np.testing.assert_array_equal(corpus.groups[0],[0])


def test_reference_target_anchor_does_not_copy_relaxed_coordinates():
    from types import SimpleNamespace
    import torch
    from src.research.structured_context.model import StructuredForecaster
    context=torch.ones(2,265)
    model=SimpleNamespace(initial=lambda c:c,spec={'target_encoder':'reference_mace'},target_mean=torch.zeros(265),target_scale=torch.ones(265))
    result=StructuredForecaster.anchor(model,{'features':torch.full((2,75,128),99.)},context)
    torch.testing.assert_close(result[:,:264],torch.ones(2,264))
    torch.testing.assert_close(result[:,264],torch.zeros(2))
