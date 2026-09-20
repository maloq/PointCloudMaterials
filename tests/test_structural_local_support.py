"""Local inputs cannot leak the discarded halo into either backbone."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from src.analysis.structural_adapter import snapshot_batch, StructuralGATrAnalysis
from src.data.structural_pretraining.support import REFERENCE_RADIUS, local_crop, support_weights
from src.research.trajectory_stability.encode import observation
from src.data.structural_pretraining.batches import collate
from src.models.encoders.structural import StructuralGATr, StructuralMACE


@pytest.mark.parametrize('architecture',['gatr','mace'])
def test_train_inference_crop_parity_and_no_outer_context(architecture):
    rng=np.random.default_rng(87)
    local=rng.uniform(-5,5,(95,3)).astype(np.float32);local[12]=0
    raw=np.concatenate((local,[[12,0,0],[-40,-40,-40],[40,40,40]])).astype(np.float32)
    native=collate([observation(raw,12,REFERENCE_RADIUS,architecture)],architecture)
    static=snapshot_batch(raw,cKDTree(raw),raw[[12]],scale=REFERENCE_RADIUS,material='Al',architecture=architecture)
    for key,value in static.items():torch.testing.assert_close(value,native[key],rtol=0,atol=0)
    assert native['positions'].shape[2]<len(raw)
    changed=raw.copy();changed[-3]=[15,2,1]
    other=collate([observation(changed,12,REFERENCE_RADIUS,architecture)],architecture)
    for key,value in native.items():torch.testing.assert_close(value,other[key],rtol=0,atol=0)
    torch.manual_seed(6)
    net=(StructuralGATr() if architecture=='gatr' else StructuralMACE(backend='e3nn')).eval()
    with torch.no_grad():torch.testing.assert_close(net(native),net(other),rtol=0,atol=0)


def test_material_scale_and_smooth_outer_boundary():
    x=np.array([[0,0,0],[6,0,0],[7,0,0],[8-1e-4,0,0],[8,0,0],[17,0,0]],np.float32)
    local,rows=local_crop(x*1.12,REFERENCE_RADIUS*1.12)
    np.testing.assert_array_equal(rows,[0,1,2,3])
    w=support_weights(local)
    np.testing.assert_allclose(w[:3],[1,1,.5],atol=1e-6)
    assert w[-1]<1e-7


def test_oversized_analysis_revision_is_rejected():
    from types import SimpleNamespace
    cfg=SimpleNamespace(protocol='structural_gatr_snapshot_static_v6')
    with pytest.raises(ValueError,match='Unsupported structural analysis protocol'):
        StructuralGATrAnalysis(cfg)


def test_new_local_shards_keep_spatial_partners_outside_two_units(tmp_path):
    from src.data.structural_pretraining.prepare import prepare_task
    axes=np.arange(-12,13,dtype=np.float32)*2.4
    positions=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),-1).reshape(-1,3)
    path=tmp_path/'lattice.npy';np.save(path,positions)
    source=dict(kind='static',path=str(path),bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,id='test',material='Al',potential='test')
    task=dict(id='000000',seed=3,frame=0,count=2,split='train')
    receipt=prepare_task((tmp_path/'release',source,task,REFERENCE_RADIUS,'test-local'))
    folder=tmp_path/'release/shards/000000'
    views=np.load(folder/'views.npy');centers=np.load(folder/'center_ids.npy')
    assert receipt['views']==4
    delta=positions[centers[views[:,2]]]-positions[centers[views[:,4]]]
    distance=np.linalg.norm(delta,axis=-1)
    assert np.all((distance>2.)&(distance<=4.25))
    assert np.linalg.norm(np.load(folder/'positions.npy'),axis=-1).max()<=8.
