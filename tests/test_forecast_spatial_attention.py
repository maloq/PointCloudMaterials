"""Neighbor identities, invariant geometry, learned attention and accumulated updates."""

import copy
import json
import numpy as np
import pytest
import torch

from src.experiment_runner.registry import sha256,write_json
from src.training_methods.embedding_forecast.attention_data import AttentionWindowDataset,AttentionResidentWindows,AttentionSource
from src.training_methods.embedding_forecast.context_mixture import build_forecaster,mixture_nll,call_forecaster
from src.training_methods.embedding_forecast.run import optimize_batch,train
from src.training_methods.embedding_forecast.runtime import forecast_loader,validation_loader_on_device
from src.research.forecast_spatial_mixture.evaluate import score_source
from test_embedding_forecast import cache_fixture
from test_forecast_context_mixture import variant


def attention_variant(k=3):
    result=variant(spatial=k)
    result['spatial_attention']=dict(width=8,heads=2,blocks=2,radial_bins=8,radial_max_A=20.,precision='float32')
    return result


def attention_fixture(root):
    cache=root/'cache';manifest=cache_fixture(cache)
    spatial=root/'spatial';geometry=root/'geometry'; records=[];geometry_records=[]
    for record in manifest['shards']:
        s=spatial/record['directory'];g=geometry/record['directory'];s.mkdir(parents=True);g.mkdir(parents=True)
        n,t=record['centers'],record['frames']
        neighbors=np.broadcast_to(np.array([1,0],dtype=np.uint16)[:,None,None],(n,t,1)).copy()
        np.save(s/'neighbors.npy',neighbors)
        positions=np.zeros((n,t,3),dtype=np.float32);positions[0,:,0]=.5;positions[1,:,0]=9.5
        positions[:,:,1]=np.arange(t)[None]*.1
        np.save(g/'center_positions_A.npy',positions)
        np.save(g/'box_lengths_A.npy',np.full((t,3),10,dtype=np.float32))
        records.append(dict(directory=record['directory'],checksums={'neighbors.npy':sha256(s/'neighbors.npy')}))
        geometry_records.append(dict(directory=record['directory'],checksums={p.name:sha256(p) for p in g.iterdir()}))
    digest=sha256(cache/'manifest.json')
    write_json(spatial/'manifest.json',dict(state='complete',config=dict(neighbors=1),records=records,base_cache_manifest_sha256=digest))
    write_json(geometry/'manifest.json',dict(state='complete',records=geometry_records,base_cache_manifest_sha256=digest))
    return manifest,cache,spatial,geometry


def test_attention_is_permutation_and_rotation_invariant_with_nonzero_gradients():
    torch.manual_seed(23)
    model=build_forecaster(3,4,.75,[.75,1.5,2.25],attention_variant()).eval()
    history=torch.randn(2,4,3);neighbors=torch.randn(2,4,3,3,requires_grad=True)
    relative=torch.randn_like(neighbors,requires_grad=True)
    expected=model(history,neighbors,relative)
    permutation=torch.tensor([2,0,1])
    permuted=model(history,neighbors[:,:,permutation],relative[:,:,permutation])
    rotation=torch.linalg.qr(torch.randn(3,3)).Q
    rotated=model(history,neighbors,relative@rotation)
    for key in expected:
        torch.testing.assert_close(permuted[key],expected[key],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(rotated[key],expected[key],rtol=1e-5,atol=1e-6)
    mixture_nll(expected,torch.randn(2,3,3)).mean().backward()
    assert neighbors.grad.abs().sum()>0 and relative.grad.abs().sum()>0
    for layer in (model.neighbor_attention.key,model.neighbor_attention.queries[0]):
        assert layer.weight.grad.abs().sum()>0
    assert torch.all(expected['spatial_attention_effective_neighbors']<=3.00001)


def test_directional_arrangement_is_preserved_beyond_radial_information():
    torch.manual_seed(5)
    model=build_forecaster(3,3,.75,[.75,1.5,2.25],attention_variant()).eval()
    history=torch.randn(1,3,3);neighbors=torch.randn(1,3,3,3)
    a=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.]]).expand(1,3,3,3)
    b=torch.tensor([[1.,0,0],[1.,0,0],[1.,0,0]]).expand_as(a)
    torch.testing.assert_close(a.norm(dim=-1),b.norm(dim=-1))
    assert not torch.allclose(model(history,neighbors,a)['mean'],model(history,neighbors,b)['mean'],rtol=0,atol=1e-8)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA'))])
def test_attention_loader_matches_same_frame_other_centers_and_periodic_geometry(tmp_path,device):
    manifest,cache,spatial,geometry=attention_fixture(tmp_path)
    dataset=AttentionWindowDataset(cache,manifest,'train',1.5,2.25,.75,3.,spatial_root=spatial,geometry_root=geometry)
    resident=AttentionResidentWindows(dataset,device)
    indices=[0,resident.windows,2*resident.windows+3]
    batch=resident[indices]
    neighbors,relative=resident.neighbor_batch(batch['spatial_centers'],batch['spatial_columns'])
    for row,index in enumerate(indices):
        center,origin=divmod(index,resident.windows);source,within=divmod(center,2)
        columns=dataset.history_skip+origin*dataset.stride+np.arange(dataset.history_steps)
        actual=np.load(cache/dataset.records[source]['directory']/'embeddings.npy')[1-within,columns]
        torch.testing.assert_close(neighbors[row].cpu()[:,0],torch.from_numpy(actual),rtol=0,atol=0)
        torch.testing.assert_close(relative[row].cpu()[:,0,0],torch.full((3,),-1. if within==0 else 1.))
    # Changing future values cannot change the individual observed-neighbor history.
    resident.embeddings[:,int(batch['spatial_columns'].max())+1:]=999
    after,_=resident.neighbor_batch(batch['spatial_centers'],batch['spatial_columns'])
    torch.testing.assert_close(after,neighbors,rtol=0,atol=0)


def test_microbatch_gradients_and_partial_tail_match_whole_batch():
    torch.manual_seed(8)
    model=build_forecaster(3,4,.75,[.75,1.5,2.25],variant(components=2))
    micro=copy.deepcopy(model)
    batch=dict(history=torch.randn(11,4,3),future=torch.randn(11,3,3))
    settings=dict(gradient_clip=5.)
    a=torch.optim.SGD(model.parameters(),lr=.01);b=torch.optim.SGD(micro.parameters(),lr=.01)
    totals,_,count=optimize_batch(model,batch,torch.zeros(3),torch.ones(3),'cpu',settings,variant(),a)
    chunks,_,chunk_count=optimize_batch(micro,batch,torch.zeros(3),torch.ones(3),'cpu',dict(settings,micro_batch_size=4),variant(),b)
    assert count==chunk_count==11
    for x,y in zip(model.parameters(),micro.parameters()):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
    for key in totals:torch.testing.assert_close(totals[key],chunks[key],rtol=1e-6,atol=1e-6)


def test_attention_physical_projection_matches_direct_forward(tmp_path):
    manifest,cache,spatial,geometry=attention_fixture(tmp_path)
    name=manifest['shards'][0]['directory']
    z=torch.from_numpy(np.load(cache/name/'embeddings.npy'))
    store=AttentionSource(z,torch.from_numpy(np.load(spatial/name/'neighbors.npy').astype(np.int16)),
        torch.from_numpy(np.load(geometry/name/'center_positions_A.npy')),torch.from_numpy(np.load(geometry/name/'box_lengths_A.npy')))
    model=build_forecaster(3,3,.75,[.75,1.5,2.25],attention_variant(1)).eval()
    rows=torch.tensor([1]);anchors=np.array([4,7]);mean=torch.zeros(3);scale=torch.ones(3);w=torch.ones(3,dtype=torch.float64)
    scores=score_source(z[rows],None,None,anchors,model,mean,scale,w,torch.tensor(0.,dtype=torch.float64),1,store,rows)
    for i,anchor in enumerate(anchors):
        columns=torch.arange(anchor-2,anchor+1)[None]
        with torch.no_grad():prediction=call_forecaster(model,z[rows[:,None],columns],dict(spatial_store=store,spatial_centers=rows,spatial_columns=columns),mean,scale)
        np.testing.assert_allclose(scores['mean_margin'][0,i],prediction['mean'][0].sum(-1).numpy(),atol=1e-6)


def test_attention_training_and_swap_validation_export_complete_scores(tmp_path):
    _,cache,spatial,geometry=attention_fixture(tmp_path)
    v=attention_variant(1)
    config=dict(data=dict(cache=str(cache),spatial_cache=str(spatial),geometry_cache=str(geometry)),
        output=str(tmp_path/'fits'),history_ps=1.5,anchor_history_ps=3.,horizons_ps=[.75,1.5,2.25],stride_ps=.75,
        training=dict(epochs=2,patience=2,batch_size=13,micro_batch_size=4,workers=0,cpu_threads=1,
            scale_floor_fraction=.05,learning_rate=.001,weight_decay=.0001,minimum_lr_fraction=.1,gradient_clip=5.),
        variants=[v],seeds=[4],comparisons=[],bin_comparisons=[])
    metrics=train(config,v,4,'cpu',runtime=dict(loader='resident',log_every_steps=0,validation_residency='swap_device',evaluation_batch_size=5))
    assert np.isfinite(metrics['source_mean']['nll'])
    assert metrics['source_mean']['spatial_attention_effective_neighbors']==pytest.approx(1.)
    status=json.loads((tmp_path/'fits/context_mixture-seed4/technical/status.json').read_text())
    assert status['state']=='complete'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_cuda_swap_restores_training_after_interruption_and_bfloat16_backward(tmp_path):
    manifest,cache,spatial,geometry=attention_fixture(tmp_path)
    datasets={split:AttentionWindowDataset(cache,manifest,split,1.5,2.25,.75,3.,
        spatial_root=spatial,geometry_root=geometry) for split in ('train','val')}
    runtime=dict(loader='resident',log_every_steps=0,validation_residency='swap_device')
    train_loader=forecast_loader(datasets['train'],dict(batch_size=5),False,4,'cuda',runtime)
    val_loader=forecast_loader(datasets['val'],dict(batch_size=5),False,4,'cpu',runtime)
    before={name:getattr(train_loader.dataset,name).clone() for name in train_loader.dataset.tensor_names}
    original={name:getattr(val_loader.dataset,name) for name in val_loader.dataset.tensor_names}
    with pytest.raises(RuntimeError,match='interrupted'):
        with validation_loader_on_device(val_loader,'cuda',runtime,training_loader=train_loader):
            assert train_loader.dataset.embeddings.device.type=='cpu'
            assert val_loader.dataset.embeddings.is_cuda
            raise RuntimeError('interrupted')
    for name,value in before.items():
        torch.testing.assert_close(getattr(train_loader.dataset,name),value,rtol=0,atol=0)
    assert all(getattr(val_loader.dataset,name) is value for name,value in original.items())
    v=attention_variant(1);v['spatial_attention']['precision']='bfloat16'
    model=build_forecaster(3,3,.75,[.75,1.5,2.25],v).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    terms,_,_=optimize_batch(model,next(iter(train_loader)),torch.zeros(3,device='cuda'),
        torch.ones(3,device='cuda'),'cuda',dict(micro_batch_size=2,gradient_clip=5.),v,optimizer)
    assert torch.isfinite(terms['loss'])
