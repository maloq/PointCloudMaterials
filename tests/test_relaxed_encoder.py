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


def test_gpu_execution_preserves_relaxation_protocol(tmp_path):
    from src.research.relaxed_encoder.accelerated import accelerator_settings
    from src.data.structural_pretraining.prepare import file_hash
    binary=tmp_path/'lmp';binary.write_bytes(b'pinned CUDA executable')
    profile=dict(binary=str(binary),binary_sha256=file_hash(binary),backend='h100')
    base=dict(minimizer='fire',force_tolerance=.01,timestep_ps=.001,max_iterations=10000,
              pair_commands=['pair_style meam','pair_coeff immutable-potential'],lammps_command=['mpi','cpu-lmp'])
    result=accelerator_settings(base,profile)
    for key in base.keys()-{'lammps_command'}:assert result[key]==base[key]
    assert result['lammps_command']==[str(binary),'-k','on','g','1','-sf','kk','-pk','kokkos','neigh','half','newton','on','gpu/aware','off']
    assert base['lammps_command']==['mpi','cpu-lmp']
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binary changed'):accelerator_settings(base,profile)


def test_timeout_skips_whole_training_pair_but_not_corruption(tmp_path):
    import json
    from src.research.relaxed_encoder.availability import training_availability,fatal_failures
    root=tmp_path/'out/technical/failures';root.mkdir(parents=True)
    source=dict(id=1,pilot_fit=True,split='train',pool_atom_ids=[1,2],center_atom_ids=[1])
    plan=dict(config=dict(output=str(tmp_path/'out'),cache=str(tmp_path/'cache'),frames=[0,2]),sources=[source])
    task=dict(id='1-1',source=1,frame=1)
    (root/'1-1.json').write_text(json.dumps(dict(task=task,error="TimeoutExpired(['lmp'], 2400)")))
    for frame in (2,3):
        p=tmp_path/f'cache/cells/1-{frame}/complete.json';p.parent.mkdir(parents=True);p.write_text('{}')
    included,excluded,pending=training_availability(plan)
    assert [f for _,f in included]==[2] and excluded[0]['frame']==0
    assert excluded[0]['anchors']==2 and not pending and not fatal_failures(plan)
    # The unavailable other half of an excluded pair must not block the cache.
    (root/'1-4.json').write_text(json.dumps(dict(task=dict(id='1-4'),error="ValueError('source checksum changed')")))
    assert [p.name for p in fatal_failures(plan)]==['1-4.json']


def test_assay_timeout_keeps_matched_rows_and_reindexes_frames(tmp_path,monkeypatch):
    import json
    from src.research.relaxed_encoder import assay
    cache=tmp_path/'cache';out=tmp_path/'out';original_cache=tmp_path/'original';(original_cache/'1').mkdir(parents=True)
    np.save(original_cache/'1/onset.npy',np.array([10,11]))
    original=tmp_path/'original.json';original.write_text(json.dumps(dict(anchors=[0,1,2],config=dict(cache=str(original_cache)))))
    popfile=tmp_path/'population.npz';rows=np.array([[1,f,i] for f in range(3) for i in range(2)])
    np.savez(popfile,source=np.ones(6,int),rows=rows,condition=np.zeros((6,1)),role=np.array(['train']*6),descriptor=np.arange(6)[:,None])
    for frame in (0,2):
        folder=cache/f'cells/1-{frame}';folder.mkdir(parents=True)
        x=np.full((2,1,80,3),frame,np.float32)
        np.savez(folder/'clouds.npz',hot=x,cold=x,query_atom_ids=np.array([[1],[2]]));(folder/'complete.json').write_text('{}')
    failures=out/'technical/failures';failures.mkdir(parents=True)
    (failures/'1-1.json').write_text(json.dumps(dict(task=dict(id='1-1',source=1,frame=1),error="TimeoutExpired(['lmp'], 2400)")))
    monkeypatch.setattr(assay,'graph_arrays',lambda x,scale:({'positions':x.reshape(-1,3)},None))
    monkeypatch.setattr(assay,'target_cloud',lambda x,scale:x)
    monkeypatch.setattr(assay,'geometry_packet',lambda x:np.full(85,x[0,0]))
    monkeypatch.setattr(assay,'persistence_image',lambda x:np.full(144,x[0,0]))
    monkeypatch.setattr(assay,'order_targets',lambda x,scale:np.full(8,x[0,0]))
    plan=dict(identity='test',sources=[dict(id=1,center_atom_ids=[1,2],temperature_K=500)],config=dict(output=str(out),cache=str(cache),assay_plan=str(original),population=str(popfile),frames=[0,1,2],scale=8))
    assert assay.prepare(plan)
    result=dict(np.load(out/'technical/assay/population.npz'))
    np.testing.assert_array_equal(result['rows'][:,1],[0,0,2,2])
    np.testing.assert_array_equal(result['graph'],[0,1,2,3])
    np.testing.assert_array_equal(result['original_geometry'][:,0],[0,1,4,5])
    for domain in ['hot','cold']:
        np.testing.assert_array_equal(np.load(out/f'technical/assay/{domain}-descriptors.npy')[:,0],[0,0,2,2])
    ready=json.loads((out/'technical/assay/ready.json').read_text())
    assert ready['timeout_excluded_windows']==2 and ready['skipped_cells']==['1-1']
