import numpy as np
import pytest
from src.data.structural_pretraining.prepare import geometry_packet, offsets, registry_sources
from src.data.predictive_memory.targets import physical_packet


def test_geometry_packet_matches_actual_physical_producer():
    rng=np.random.default_rng(10)
    x=rng.uniform(-7,7,(250,3)); x[0]=0
    reference=physical_packet(x,rng.normal(size=x.shape))
    np.testing.assert_allclose(geometry_packet(x),np.r_[reference[:80],reference[112:117]],rtol=2e-6,atol=1e-6)


def test_geometry_invariant_and_preserves_scale():
    rng=np.random.default_rng(12); x=rng.uniform(-4,4,(90,3)); x[0]=0
    q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    np.testing.assert_allclose(geometry_packet(x),geometry_packet(x@q),rtol=2e-6,atol=1e-6)
    assert np.linalg.norm(geometry_packet(x)-geometry_packet(x*1.1))>.1


def test_periodic_chart_retains_center_and_crossing_atoms():
    x=np.array([[.1,0,0],[9.9,0,0],[1.,1.,1.]])
    np.testing.assert_allclose(offsets(x,0,[0,1,2],np.full(3,10)),[[0,0,0],[-.2,0,0],[.9,1,1]],atol=1e-6)


def test_bad_geometry_fails():
    with pytest.raises(ValueError):geometry_packet(np.zeros((2,3)))


def test_material_subset_filters_inputs_and_refits_only_training_targets(tmp_path):
    import json
    from src.data.structural_pretraining.batches import Release
    shards=[]
    for name,material,split,offset in [('train','Al','train',0.),('heldout','Al','selection',1000.),
                                      ('excluded','Mg','train',10000.)]:
        shards.append(dict(task=dict(id=name,split=split),material=material,potential='test',static=True,anchors=1))
        if material!='Al':continue  # Excluded arrays must never even be opened.
        folder=tmp_path/'shards'/name;folder.mkdir(parents=True)
        arrays=dict(views=np.array([[-1,-1,0,-1,1]]),
            physical=np.array([[1.,2.],[3.,6.]])+offset,
            tda=np.array([[4.,8.],[6.,12.],[999.,999.]])+offset,
            tda_valid=np.array([True,True,False]))
        for key,value in arrays.items():np.save(folder/f'{key}.npy',value)
    original=dict(state='complete',identity={'version':'parent'},shards=shards,
        normalization={'physical':{'mean':[999,999],'std':[999,999]}})
    (tmp_path/'manifest.json').write_text(json.dumps(original))
    release=Release(tmp_path,materials=['Al'])
    assert release.group_keys==[('Al','test',True)]
    assert release.groups[release.group_keys[0]]==[0] and release.selection==[1]
    assert set(release.arrays)=={'train','heldout'}
    assert release.manifest['normalization']=={
        'physical':{'mean':[2.,4.],'std':[1.,2.]},'tda':{'mean':[5.,10.],'std':[1.,2.]}}
    assert release.manifest['identity']['parent']['parent']==original['identity']
    assert release.manifest['identity']['parent']['observation_support']['outer_radius']==8.
    assert json.loads((tmp_path/'manifest.json').read_text())==original
    with pytest.raises(ValueError,match='Requested materials'):Release(tmp_path,materials=['Ti'])
    original['shards'][1]['task']['split']='test'
    (tmp_path/'manifest.json').write_text(json.dumps(original))
    heldout=Release(tmp_path,materials=['Al'])
    assert heldout.groups[heldout.group_keys[0]]==[0] and heldout.selection==[]
