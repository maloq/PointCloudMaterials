import copy
import numpy as np
import pytest
from src.research.crystallization_information.data import event_bins,observed_features,BLOCKS,permutation_control
from src.research.crystallization_information.runtime import standardize,model_for,score_hazard,decoder_score,fit


def test_exact_short_horizons_and_censoring():
    np.testing.assert_array_equal(event_bins(np.array([1,4,5,8,12,16,17,801])+20,20),[0,1,2,2,3,4,5,5])
    with pytest.raises(ValueError):event_bins([20],20)


def test_features_are_causal_and_partition_matches_producer():
    rng=np.random.default_rng(1);packet=rng.normal(size=(2,50,128));order=rng.normal(size=(2,50,8));shell=rng.normal(size=(2,50,12))
    centers=np.array([0,1]);frames=np.array([20,20]);before=observed_features(packet,order,shell,centers,frames)
    packet[:,21:]=1e9;order[:,21:]=1e9;shell[:,21:]=1e9
    np.testing.assert_array_equal(before,observed_features(packet,order,shell,centers,frames))
    np.testing.assert_array_equal(np.sort(np.concatenate(list(BLOCKS.values()))),np.arange(427))
    np.testing.assert_allclose(before[:,:128],packet[centers,frames],rtol=1e-6)


def test_scaling_never_fits_test_and_shuffle_keeps_roles():
    x=np.arange(80,dtype=np.float32).reshape(20,4);s=np.repeat(np.arange(4),5);train=np.arange(10)
    a,m,v=standardize(x,train,s);x[10:]=1e8;b,m2,v2=standardize(x,train,s)
    np.testing.assert_array_equal(m,m2);np.testing.assert_array_equal(v,v2)
    role=np.repeat(['train','test'],10);t=np.full(20,500);permuted=permutation_control(x,role,t,9)
    np.testing.assert_array_equal(np.sort(permuted[:10,0]),np.sort(x[:10,0]))


def test_matched_parameter_capacity_and_perfect_decoding():
    a=model_for(562,5,'mlp',128);b=model_for(562,5,'mlp',128)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    rng=np.random.default_rng(4);y=rng.normal(size=(40,148));d=decoder_score(y,y,np.repeat([0,1],20))
    assert all(v['r2']==pytest.approx(1) for v in d['groups'].values())


def test_real_cpu_readout_smoke(tmp_path):
    rng=np.random.default_rng(8);n=192
    pop=dict(role=np.repeat(['train','selection','calibration','test'],48),source=np.repeat(np.arange(8),24),event=np.tile(np.arange(6),32).astype(np.int64),delay=np.tile([.75,3,6,9,12,15],32),temperature=np.full(n,500))
    config=dict(seed=1,device='cpu',width=8,lr=.001,updates=2,evaluate_every=1,batch_size=16)
    x=rng.normal(size=(n,562)).astype(np.float32)
    fit(config,dict(encoder='synthetic',variant='z',readout='mlp',task='hazard'),pop,x,pop['event'],tmp_path)
    import json
    m=json.loads((tmp_path/'technical/fits/synthetic/z/mlp/metrics.json').read_text())
    assert m['updates']==2 and set(m['classification'])=={'0.75','3.0','6.0','9.0','12.0'}
    fit(config,dict(encoder='synthetic',variant='decode',readout='strong',task='decoder'),pop,x,rng.normal(size=(n,148)).astype(np.float32),tmp_path)
