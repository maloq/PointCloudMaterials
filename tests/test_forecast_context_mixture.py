"""Full-path mixture likelihood/sampling and causal spatial-window gathering."""

import json
import math
import numpy as np
import pytest
import torch

from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.context_mixture import (
    ContextMixtureForecaster, mixture_nll, mixture_metrics, sample_trajectories, crystal_probability,
)
from src.training_methods.embedding_forecast.model import EmbeddingForecaster
from src.training_methods.embedding_forecast.spatial import SpatialWindowDataset, SpatialResidentWindows
from src.training_methods.embedding_forecast.run import train
from src.training_methods.embedding_forecast.runtime import forecast_loader, validation_loader_on_device, execution_settings
from test_embedding_forecast import cache_fixture


def variant(spatial=0, components=2):
    return dict(name='context_mixture', target='trajectory', architecture='mean_residual_gru',
        history_mode='real', distribution='trajectory_mixture' if components else 'deterministic',
        width=16, layers=1, heads=2, dropout=0., covariance_rank=2, minimum_std=.03,
        spatial_neighbors=spatial, spatial_distance_unit_A=10., mixture_components=components,
        loss=dict(mse=0. if components else 1., nll=1. if components else 0., bin_mse=0., increment_mse=0.))


def test_nonspatial_deterministic_control_is_identical_to_original():
    config = variant(components=0)
    torch.manual_seed(4); old = EmbeddingForecaster(3, 4, .75, [.75,1.5,2.25], config)
    torch.manual_seed(4); new = ContextMixtureForecaster(3, 4, .75, [.75,1.5,2.25], config)
    history = torch.randn(5,4,3)
    torch.testing.assert_close(old(history)['mean'], new(history)['mean'], rtol=0, atol=0)


def test_joint_mixture_likelihood_matches_distribution_and_rejects_mode_switches():
    means = torch.tensor([[[[-1.],[1.]], [[1.],[-1.]]]])
    output = dict(component_means=means, component_std=torch.full_like(means,.1), mixture_logits=torch.zeros(1,2))
    target = torch.tensor([[[-1.],[-1.]]])
    reference = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=output['mixture_logits']),
        torch.distributions.Independent(torch.distributions.Normal(means,output['component_std']),2))
    torch.testing.assert_close(mixture_nll(output,target), -reference.log_prob(target)/2)
    assert mixture_nll(output,target).item() > 90
    assert mixture_nll(output,means[:,0]).item() < 0


def test_sample_component_identity_is_shared_across_future_frames():
    means = torch.tensor([[[[-1.],[1.]], [[1.],[-1.]]]])
    output = dict(component_means=means, component_std=torch.full_like(means,1e-6), mixture_logits=torch.zeros(1,2))
    paths = sample_trajectories(output,1000)
    assert torch.all(paths[:,:,0,0]*paths[:,:,1,0] < 0)
    assert (paths[:,:,0,0]>0).any() and (paths[:,:,0,0]<0).any()


def test_gaussian_control_crps_and_crystal_probability_have_known_values():
    output = dict(component_means=torch.zeros(3,1,2,1), component_std=torch.ones(3,1,2,1), mixture_logits=torch.zeros(3,1))
    metrics = mixture_metrics(output,torch.zeros(3,2,1),False)
    torch.testing.assert_close(metrics['marginal_crps'],torch.full((3,),math.sqrt(2/math.pi)-1/math.sqrt(math.pi)))
    torch.testing.assert_close(metrics['coverage90'],torch.ones(3))
    probability = crystal_probability(output,torch.ones(1,dtype=torch.float64),torch.tensor(0.,dtype=torch.float64))
    torch.testing.assert_close(probability,torch.full((3,2),.5,dtype=torch.float64))


def test_spatial_mixture_gradients_reach_both_history_and_distribution():
    model = ContextMixtureForecaster(3,4,.75,[.75,1.5,2.25],variant(spatial=2))
    history = torch.randn(5,4,3,requires_grad=True)
    spatial = torch.randn(5,4,3,requires_grad=True)
    output = model(history,spatial,torch.ones(5,4,2)*10)
    mixture_nll(output,torch.randn(5,3,3)).mean().backward()
    assert history.grad.abs().sum() > 0 and spatial.grad.abs().sum() > 0
    for layer in (model.component_head,model.scale_head,model.gate,model.spatial_input):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('prepooled', [False, True])
def test_spatial_gather_uses_same_source_and_only_history_frames(tmp_path, prepooled):
    cache = tmp_path/'cache'; manifest = cache_fixture(cache)
    spatial = tmp_path/'spatial'; spatial.mkdir()
    records = []
    for record in manifest['shards']:
        directory = spatial/record['directory']; directory.mkdir()
        n, t = record['centers'],record['frames']
        neighbors = np.repeat(((np.arange(n)[:,None]+1)%n)[:,None,:],t,axis=1).astype(np.uint16)
        np.save(directory/'neighbors.npy',neighbors)
        np.save(directory/'radii_A.npy',np.full((n,t,2),5.,dtype=np.float32))
        files = ['neighbors.npy','radii_A.npy']
        if prepooled:
            values = np.load(cache/record['directory']/'embeddings.npy')
            np.save(directory/'pooled_embeddings.npy', values[(np.arange(n)+1)%n])
            files.append('pooled_embeddings.npy')
        records.append(dict(directory=record['directory'],checksums={name:sha256(directory/name) for name in files}))
    write_json(spatial/'manifest.json',dict(state='complete',base_cache_manifest_sha256=sha256(cache/'manifest.json'),records=records))
    dataset = SpatialWindowDataset(cache,manifest,'train',1.5,2.25,.75,3.,spatial_root=spatial)
    resident = SpatialResidentWindows(dataset,'cpu',spatial)
    batch = resident[[0,2,resident.windows]]
    values = np.load(cache/dataset.records[0]['directory']/'embeddings.npy')
    for row, index in enumerate([0,2,resident.windows]):
        center, origin = divmod(index,resident.windows)
        columns = dataset.history_skip+origin*dataset.stride+np.arange(dataset.history_steps)
        expected = values[(center+1)%len(values),columns]
        torch.testing.assert_close(batch['spatial'][row],torch.from_numpy(expected),rtol=0,atol=0)
    if torch.cuda.is_available():
        runtime = dict(loader='resident', log_every_steps=0, validation_residency='staged_device')
        loader = forecast_loader(dataset, dict(batch_size=3), False, 4, 'cpu', runtime)
        originals = {name: getattr(loader.dataset, name) for name in ('embeddings', 'offsets', 'spatial', 'radii')}
        expected = next(iter(loader))
        rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match='evaluation interrupted'):
            with validation_loader_on_device(loader, 'cuda', runtime):
                assert all(getattr(loader.dataset, name).is_cuda for name in originals)
                for key, value in next(iter(loader)).items():
                    torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
                raise RuntimeError('evaluation interrupted')
        assert all(getattr(loader.dataset, name) is value for name, value in originals.items())
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_mixture_fits_and_exports_with_existing_training_entry_point(tmp_path):
    cache_fixture(tmp_path/'cache')
    model = variant()
    config = dict(data=dict(cache=str(tmp_path/'cache')),output=str(tmp_path/'fits'),history_ps=1.5,
        anchor_history_ps=1.5,horizons_ps=[.75,1.5,2.25],stride_ps=.75,seeds=[4],variants=[model],
        comparisons=[],bin_comparisons=[],training=dict(epochs=2,patience=2,batch_size=16,workers=0,cpu_threads=1,
        scale_floor_fraction=.05,learning_rate=.002,weight_decay=.0001,minimum_lr_fraction=.1,gradient_clip=5.))
    result = train(config,model,4,'cpu')
    assert np.isfinite(result['sample_mean']['nll'])
    assert result['sample_mean']['energy_score'] >= 0
    saved = json.loads((tmp_path/'fits/context_mixture-seed4/technical/status.json').read_text())
    assert saved['state']=='complete'


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is required for transfer/training parity'))])
def test_host_validation_execution_retains_training_and_selection(tmp_path, device):
    cache_fixture(tmp_path/'cache')
    model = variant()
    config = dict(data=dict(cache=str(tmp_path/'cache')),output=str(tmp_path/'device'),history_ps=1.5,
        anchor_history_ps=1.5,horizons_ps=[.75,1.5,2.25],stride_ps=.75,seeds=[4],variants=[model],
        comparisons=[],bin_comparisons=[],training=dict(epochs=2,patience=2,batch_size=16,workers=0,cpu_threads=1,
        scale_floor_fraction=.05,learning_rate=.002,weight_decay=.0001,minimum_lr_fraction=.1,gradient_clip=5.))
    ordinary = train(config,model,4,device,runtime=dict(loader='resident',log_every_steps=0))
    config['output'] = str(tmp_path/'host')
    host = train(config,model,4,device,runtime=dict(loader='resident',log_every_steps=0,validation_residency='host'))
    assert ordinary == host
    config['output'] = str(tmp_path/'staged')
    staged = train(config,model,4,device,runtime=dict(loader='resident',log_every_steps=0,validation_residency='staged_device'))
    assert ordinary == staged


def test_staged_validation_requires_resident_loader():
    with pytest.raises(ValueError, match='requires the resident loader'):
        execution_settings(dict(loader='mmap',log_every_steps=0,validation_residency='staged_device'))


@pytest.mark.parametrize('neighbors_count', [8, 32, 128, 512])
def test_broad_context_pooling_keeps_same_frame_means_and_storage_precision(neighbors_count):
    from src.training_methods.embedding_forecast.spatial import pool_neighbors
    count = neighbors_count+1; frames = 5
    values = (torch.arange(count)[:, None, None]+torch.arange(frames)[None, :, None]*2).expand(-1, -1, 256).half()
    # Every other center, cyclically ordered; all times have exactly the same identities.
    indices = (torch.arange(count)[:, None]+torch.arange(1, count)[None]) % count
    indices = indices[:, None].expand(-1, frames, -1)
    pooled = pool_neighbors(values, indices)
    expected = ((torch.arange(count).float().sum()-torch.arange(count))/(count-1))[:, None, None]+torch.arange(frames)[None, :, None]*2
    torch.testing.assert_close(pooled, expected.expand_as(pooled).half(), rtol=0, atol=0)


def test_spatial_preparation_failure_is_visible_to_queued_training(tmp_path):
    from src.training_methods.embedding_forecast.spatial import prepare
    cache = tmp_path/'cache'; cache.mkdir()
    write_json(cache/'manifest.json', dict(state='incomplete', shards=[]))
    output = tmp_path/'spatial'
    with pytest.raises(ValueError, match='completed embedding producer'):
        prepare(dict(output=str(output), embedding_cache=str(cache), neighbors=32))
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and 'completed embedding producer' in status['error']
