"""History identity, normalized inputs and unchanged VICReg/TDA gradient paths."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from src.data_utils.relaxed_histories import RelaxedHistoryDataset
from src.data_utils.topology_targets import fit_targets
from src.models import EncoderAdapter
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(Path('configs').resolve())):
        return compose(config_name='vicreg_mace_relaxed', overrides=list(overrides))


def test_prepared_history_loader_preserves_identity_units_and_source_split(tmp_path):
    cfg = config()
    cfg.data.cache_dir = str(tmp_path)
    cfg.data.input_mode = 'history'
    rng = np.random.default_rng(71)
    records = []
    targets = rng.normal(size=(48, 144)).astype(np.float32)
    for context, (split, count) in enumerate((('train', 32), ('val', 8), ('test', 8))):
        views = rng.normal(size=(count, 3, 5, 80, 3)).astype(np.float16)
        views[:, :, :, 0] = 0
        filename = f'{context}.npy'
        np.save(tmp_path/filename, views)
        records.append(dict(views=filename, split=split, samples=count, source=context,
                            frame=40, temperature_K=400))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete', shards=records)))
    np.save(tmp_path/'targets.npy', targets)
    np.savez(tmp_path/'scaling.npz', **fit_targets(targets[:32], 32, .05))
    dataset = RelaxedHistoryDataset(cfg, 'test')
    item = dataset[0]
    physical = np.load(tmp_path/'2.npy')[0, 0].astype(np.float32)
    np.testing.assert_allclose(item['points']*cfg.data.normalization_radius_A, physical, atol=3e-7)
    assert item['row'] == 40 and item['source'] == 2
    assert item['points'].shape == (5, 80, 3)
    assert item['tda_targets'].shape == (144,)
    cfg.data.input_mode = 'anchor'
    np.testing.assert_array_equal(RelaxedHistoryDataset(cfg, 'test')[0]['points'], item['points'][-1])


class TinyHistory(nn.Module):
    input_layout = 'btn3'
    output_contract = 'invariant'
    invariant_dim = 256
    equivariant_dim = None

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 256)

    def forward(self, points):
        return self.linear(points.mean((1, 2)))


def test_history_vicreg_exact_loss_and_tda_supervises_only_anchor(monkeypatch):
    monkeypatch.setattr('src.training_methods.shared.base_ssl_module.build_encoder', lambda cfg:TinyHistory())
    cfg = config(['data.input_mode=history', 'vicreg_jitter_std=0', 'vicreg_mirror_prob=0'])
    torch.manual_seed(7)
    model = VICRegModule(cfg)
    metrics = {}
    monkeypatch.setattr(model, '_log_metric', lambda stage,name,value,**kwargs:metrics.update({name:value}))
    keys = ('points', 'spatial_points', 'temporal_points')
    batch = {key:torch.randn(12, 5, 80, 3, requires_grad=True) for key in keys}
    batch['tda_targets'] = torch.randn(12, 144)
    loss = model._spatiotemporal_step(batch, 0, 'train')
    encoded = model.encoder_io.encode(torch.cat([batch[key] for key in keys])).invariant.chunk(3)
    expected = model.vicreg.compute_spatiotemporal_loss(features=encoded, temporal_weight=1.)[0]
    torch.testing.assert_close(loss, expected+cfg.tda.weight*metrics['tda_mse'])
    derivatives = torch.autograd.grad(metrics['tda_mse'], [batch[key] for key in keys], retain_graph=True)
    assert (derivatives[0].abs().sum((0, 2, 3))>0).all()
    assert derivatives[1].count_nonzero() == derivatives[2].count_nonzero() == 0
    loss.backward()
    assert all(batch[key].grad.abs().sum()>0 for key in keys)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    torch.testing.assert_close(model(batch['points'])[0], model.vicreg.project_features(model.encoder(batch['points'])))
    assert EncoderAdapter(model.encoder).encode(batch['points']).invariant.shape == (12, 256)


def test_history_mirror_is_shared_over_time_and_preserves_membership(monkeypatch):
    monkeypatch.setattr('src.training_methods.shared.base_ssl_module.build_encoder', lambda cfg:TinyHistory())
    cfg = config(['data.input_mode=history', 'vicreg_jitter_std=0', 'vicreg_mirror_prob=1'])
    model = VICRegModule(cfg)
    points = torch.randn(12, 1, 80, 3).expand(-1, 5, -1, -1).clone()
    augmented = model.vicreg.apply_view_postprocessing(points.flatten(1, 2), use_neighbor=False,
        apply_occlusion=False, view_points=None).reshape_as(points)
    torch.testing.assert_close(augmented, augmented[:, :1].expand_as(augmented), rtol=0, atol=0)
    torch.testing.assert_close(augmented.square(), points.square(), rtol=0, atol=0)


@pytest.mark.parametrize('flat_reports', [False, True])
def test_collection_pairs_sources_after_averaging_seeds(tmp_path, flat_reports):
    from src.analysis.topology import collect
    for name, error in [('anchor', 1.), ('history', .8)]:
        for seed in [7, 8]:
            root = tmp_path/'plan'/'default'/f'{name}_{seed}'/'analysis_standard'
            root.mkdir(parents=True)
            metrics = dict(variant=name, seed=seed,
                results=dict(test=dict(prediction=dict(balanced_mse=error, mean_within_frame_r2=.5),
                                       projector_ridge=dict(balanced_mse=error*.5))))
            (root/'analysis_metrics.json').write_text(json.dumps({'topology':metrics}))
            if flat_reports:
                report = tmp_path/f'{name}-seed{seed}'
                report.mkdir()
                (report/'metrics.json').write_text(json.dumps({'topology':metrics}))
                (report/'source.json').write_text(json.dumps(dict(analysis_directory=str(root))))
            (root/'topology').mkdir()
            np.savez(root/'topology/test_predictions.npz', errors=np.full(12, error),
                ridge_errors=np.full(12, error*.5),
                sources=np.repeat(np.arange(6), 2), indices=np.arange(12))
    spec = tmp_path/'spec.json'
    settings = dict(variants=['anchor','history'], seeds=[7,8], bootstrap_seed=9,
                    comparisons=[['anchor','history']])
    if flat_reports:
        settings['report_root'] = str(tmp_path)
    spec.write_text(json.dumps(settings))
    collect(tmp_path, spec)
    # A second collection must not treat its own technical metrics as a model.
    collect(tmp_path, spec)
    result = json.loads((tmp_path/'comparison/technical/metrics.json').read_text())
    gain = result['comparisons']['history_versus_anchor']
    np.testing.assert_allclose(gain['relative_mse_reduction'], .2)
    assert gain['source_count'] == 6
    np.testing.assert_allclose(result['ridge_comparisons']['history_versus_anchor']['relative_mse_reduction'], .2)
