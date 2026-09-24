import numpy as np
import pytest
from src.research.encoder_parameter_search.metrics import resolution
from src.research.structural_state.common import Study


def test_fit_scaled_error_is_defined_with_constant_test_targets():
    rng=np.random.default_rng(31);z=rng.normal(size=(160,4));target=z[:,:2].copy();target[80:,1]=0
    split=np.r_[np.zeros(80),np.ones(80)];liquid=np.ones(160,dtype=bool)
    scores=resolution(z,target,z[:,3],split,liquid)
    assert np.isfinite(scores['embedding_nmse']) and scores['embedding_nmse']<2
    assert scores['columns']==[0,1]


def test_retrieval_requires_useful_liquid_geometry():
    rng=np.random.default_rng(32);z=rng.normal(size=(500,2));density=rng.normal(size=500)
    split=np.r_[np.zeros(300),np.ones(200)];liquid=np.ones(500,dtype=bool)
    useful=resolution(z,z,density,split,liquid)
    shuffled=resolution(z[rng.permutation(500)],z,density,split,liquid)
    assert useful['embedding_neighbor_nmse'] < .3*shuffled['embedding_neighbor_nmse']
    assert useful['retrieval_gain_vs_density']>0


def test_parameter_search_keeps_historical_factorial_strict(tmp_path):
    import json
    from pathlib import Path
    config=json.loads(Path('configs/structural_state/future_metric_seed20260923.json').read_text())
    config['arms'][0]['relation_weight']=1
    path=tmp_path/'bad.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='four predeclared'):Study(path)


def test_v4_requires_declared_current_and_future_weights(tmp_path):
    import json
    from pathlib import Path
    config=json.loads(Path('configs/encoder_parameter_search/mace-s20260923.json').read_text())
    config['arms'][0]['future_weight']=.25
    path=tmp_path/'bad.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='no future loss'):Study(path)
