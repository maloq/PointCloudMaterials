import numpy as np
import pytest
from src.research.relaxed_encoder.prepare import arm_clouds,paired_clouds,target_cloud,graph_arrays
from src.research.relaxed_encoder.report import compare


def test_relaxed_inputs_and_all_target_views():
    hot=np.arange(2*2*7*80*3,dtype=np.float32).reshape(2,2,7,80,3)
    cold=hot+10
    x,y=arm_clouds(hot,cold,'relaxed_to_relaxed')
    np.testing.assert_array_equal(x,cold);np.testing.assert_array_equal(y,cold)
    x,y=arm_clouds(hot,cold,'hot_to_relaxed')
    np.testing.assert_array_equal(x[:,0,0],hot[:,0,0]);np.testing.assert_array_equal(x[:,1],cold[:,1]);np.testing.assert_array_equal(x[:,0,1:],cold[:,0,1:]);np.testing.assert_array_equal(y,cold)
    x,y=arm_clouds(hot,cold,'instantaneous');np.testing.assert_array_equal(x,hot);np.testing.assert_array_equal(y,hot)
    with pytest.raises(ValueError):arm_clouds(hot,cold,'unknown')


def test_periodic_relaxation_preserves_ids_and_translation():
    rng=np.random.default_rng(77);hot=rng.uniform(0,20,(100,3));box=np.array([20.,20.,20.]);queries=np.array([[0,2],[4,8]])
    cold=np.mod(hot+[19,3,1],box)
    a,b,ids=paired_clouds(hot,cold,box,queries)
    np.testing.assert_array_equal(ids[...,0],queries)
    np.testing.assert_allclose(a,b,atol=2e-6)
    np.testing.assert_array_equal(a[...,0,:],0)
    # Large local changes must not cause target neighbor reselection.
    cold[ids[0,0,-1]]=hot[0]+.0001
    _,b2,ids2=paired_clouds(hot,cold,box,queries)
    np.testing.assert_array_equal(ids,ids2)
    assert not np.allclose(b,b2)


def test_unpaired_report_rejected():
    with pytest.raises(ValueError):compare({'per_source':{'one':{}}},{'per_source':{'two':{}}})


def test_input_and_target_use_identical_support():
    from src.data.structural_pretraining.support import REFERENCE_RADIUS
    rng=np.random.default_rng(7);x=rng.normal(size=(80,3)).astype(np.float32);x[0]=0;x[-1]=[9,0,0]
    target=target_cloud(x,REFERENCE_RADIUS)
    graphs,moments=graph_arrays(x[None],REFERENCE_RADIUS)
    assert len(target)==79
    np.testing.assert_array_equal(graphs['positions'],target)
    np.testing.assert_array_equal(graphs['offsets'],[0,79])
    assert moments.shape==(1,120)


def test_paired_cache_reuse_subsets_centers_and_rejects_insufficient_views(tmp_path):
    import json
    from src.research.relaxed_encoder.prepare import reuse_paired
    from src.data.structural_pretraining.prepare import file_hash
    old=tmp_path/'old';cell=old/'cells/1-64';cell.mkdir(parents=True)
    clouds=np.arange(3*7*80*3,dtype=np.float32).reshape(3,7,80,3)
    ids=np.tile(np.array([2,4,6])[:,None],(1,7))
    np.savez(cell/'clouds.npz',hot=clouds,cold=clouds+1,query_atom_ids=ids)
    (cell/'complete.json').write_text(json.dumps(dict(identity='old',clouds_sha256=file_hash(cell/'clouds.npz'),relaxation=dict(fmax_eV_per_A=.005))))
    common=dict(seed=1,scale=9.,potential_sha256=['potential'])
    parent=tmp_path/'parent.json';parent.write_text(json.dumps(dict(config=dict(common,cache=str(old)),identity='old',sources=[dict(id=1,manifest_sha256='source')])))
    plan=dict(config=dict(common,paired_parent_plan=str(parent)),identity='new');task=dict(id='1-64')
    source=dict(id=1,manifest_sha256='source',pilot_fit=False,center_atom_ids=[4,6]);root=tmp_path/'new';root.mkdir()
    receipt=reuse_paired(plan,task,source,root)
    assert receipt['identity']=='new' and receipt['reused_paired_sha256']==file_hash(cell/'clouds.npz')
    with np.load(root/'clouds.npz') as a:np.testing.assert_array_equal(a['hot'],clouds[1:,:1])
    source.update(pilot_fit=True,split='train',pool_atom_ids=[2,4,6,8])
    assert reuse_paired(plan,task,source,root) is None
