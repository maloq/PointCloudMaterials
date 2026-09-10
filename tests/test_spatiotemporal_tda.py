"""TDA target alignment, training-only scaling, and the original VICReg gradient path."""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from src.analysis.liquid_structure import persistence_image
from src.analysis.config import _apply_analysis_inference_overrides
from src.data_utils.spatiotemporal_tda import prepare
from src.data_utils.spatiotemporal_views import SpatiotemporalViewDataset
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule


def test_targets_use_same_normalized_eighty_atoms_and_training_only_scaling(tmp_path):
    root = tmp_path/'views'
    root.mkdir()
    rng = np.random.default_rng(8)
    shards = []
    for split in ('train', 'val'):
        x = (rng.normal(size=(3, 3, 80, 3))*.2).astype(np.float16)
        x[:, :, 0] = 0
        pairs = np.zeros((3, 4), dtype=np.int64)
        pairs[:, 3] = [1, 5, 1]
        np.save(root/f'Al_{split}.views.npy', x)
        np.save(root/f'Al_{split}.pairs.npy', pairs)
        shards.append(dict(split=split, views=f'Al_{split}.views.npy', pairs=f'Al_{split}.pairs.npy'))
    (root/'manifest.json').write_text(json.dumps(dict(state='complete', shards=shards)))
    cfg = OmegaConf.create(dict(data=dict(cache_dir=str(root), temporal_lag_steps=1),
        encoder=dict(kwargs=dict(reference_radius_A=9.192189)), seed_everything=9,
        tda=dict(cache_dir=str(tmp_path/'targets'), components=2, fit_anchors=2, workers=1)))
    prepare(cfg)
    out = Path(cfg.tda.cache_dir)
    original = np.load(root/'Al_train.views.npy').astype(np.float32)
    images = np.load(out/'Al_train.images.npy')
    direct = persistence_image(original[2, 1]*9.192189)
    np.testing.assert_allclose(images[1, 1], direct, rtol=1e-6, atol=1e-7)
    changed = original[2, 1].copy()
    changed[-1] *= .1
    assert np.linalg.norm(direct-persistence_image(changed*9.192189)) > 1e-6
    scaling = np.load(out/'scaling.npz')
    np.testing.assert_allclose(scaling['mean'], images.reshape(-1, 144).mean(0), atol=1e-7)
    ds = SpatiotemporalViewDataset(root, 'train', 1, out)
    item = ds[1]
    np.testing.assert_array_equal(item['spatial_points'].numpy(), original[2, 1])
    expected = (direct-scaling['mean']) @ scaling['components'].T/scaling['std']
    np.testing.assert_allclose(item['tda_targets'][1].numpy(), expected, atol=1e-5)
    assert set(item) == {'points', 'spatial_points', 'temporal_points', 'tda_targets'}
    prepare(cfg)  # Completed cache must verify and reuse its original targets.


class TinyEncoder(nn.Module):
    input_layout = 'bn3'
    output_contract = 'invariant'
    invariant_dim = 256
    equivariant_dim = None

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(240, 256)

    def forward(self, points):
        return self.linear(points.flatten(1))


def test_tda_preserves_vicreg_and_reuses_projector_once_per_view(monkeypatch):
    monkeypatch.setattr('src.training_methods.base_ssl_module.build_encoder', lambda cfg:TinyEncoder())
    cfg = OmegaConf.load('configs/vicreg_pretrained_mace_geometry_tda.yaml')
    cfg.vicreg_jitter_std = 0.
    cfg.vicreg_mirror_prob = 0.
    cfg.tda.enabled = False
    torch.manual_seed(13)
    baseline = VICRegModule(cfg)
    rng_without_head = torch.get_rng_state()
    cfg.tda.enabled = True
    torch.manual_seed(13)
    model = VICRegModule(cfg)
    assert torch.equal(torch.get_rng_state(), rng_without_head)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    metrics = {}
    monkeypatch.setattr(model, '_log_metric', lambda stage,name,value,**kwargs:metrics.update({name:value}))
    monkeypatch.setattr(baseline, '_log_metric', lambda *args,**kwargs:None)
    batch = {key:torch.randn(12, 80, 3) for key in ('points', 'spatial_points', 'temporal_points')}
    batch['tda_targets'] = torch.randn(12, 3, 32)
    optimizer = model.configure_optimizers()[0][0]
    loss = model._spatiotemporal_step(batch, 0, 'train')
    pure = baseline._spatiotemporal_step(batch, 0, 'train')
    torch.testing.assert_close(metrics['vicreg'], pure)
    torch.testing.assert_close(loss, pure+cfg.tda.weight*metrics['tda_mse'])
    torch.testing.assert_close(metrics['tda_mean_baseline_mse'], batch['tda_targets'].square().mean())
    assert model.vicreg.projector[1].num_batches_tracked == 3
    assert model.vicreg.projector[4].num_batches_tracked == 3
    loss.backward()
    for part in (model.encoder, model.vicreg.projector, model.tda_head):
        grads = [p.grad for p in part.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
    before = model.tda_head[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.tda_head[0].weight)
    # Strict inference loading uses only the saved config/weights, not the target cache.
    state = model.state_dict()
    cfg.tda.cache_dir = '/not-used-during-inference'
    cfg.data.kind = 'static'
    _apply_analysis_inference_overrides(cfg)
    restored = VICRegModule(cfg)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch['points']), model(batch['points']))
    batch['tda_targets'][:] = float('nan')
    with pytest.raises(FloatingPointError, match='Nonfinite TDA'):
        model._spatiotemporal_step(batch, 0, 'train')
