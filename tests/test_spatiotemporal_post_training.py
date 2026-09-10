import torch
import json
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.training_methods.spatiotemporal import ProgressCheckpoint
from src.analysis.config import _apply_analysis_inference_overrides, load_checkpoint_analysis_config
from src.data_utils.spatiotemporal_views import SpatiotemporalViewDataset


def test_default_post_training_analysis_uses_existing_static_config():
    cfg = load_checkpoint_analysis_config()
    assert cfg.inputs.data_config == 'configs/data/loaders/static_al_80.yaml'


def test_temporal_lag_selection_keeps_aligned_views_across_shards(tmp_path):
    shards = []
    for shard_index, lags in enumerate(([5, 1, 5], [5, 5], [1, 5, 1])):
        views = np.zeros((len(lags), 3, 80, 3), dtype=np.float16)
        views[:, :, :, 0] = (10*shard_index + np.arange(len(lags)))[:, None, None]
        views[:, :, :, 1] = np.arange(3)[None, :, None]
        pairs = np.zeros((len(lags), 4), dtype=np.int64)
        pairs[:, 3] = lags
        np.save(tmp_path/f'views_{shard_index}.npy', views)
        np.save(tmp_path/f'pairs_{shard_index}.npy', pairs)
        shards.append(dict(split='train', views=f'views_{shard_index}.npy', pairs=f'pairs_{shard_index}.npy'))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(state='complete', shards=shards)))
    filtered = SpatiotemporalViewDataset(tmp_path, 'train', temporal_lag_steps=1)
    assert len(filtered) == 3
    for index, source_row in enumerate((1, 20, 22)):
        sample = filtered[index]
        assert set(sample) == {'points', 'spatial_points', 'temporal_points'}
        for view, key in enumerate(('points', 'spatial_points', 'temporal_points')):
            assert sample[key].dtype == torch.float32
            assert sample[key].shape == (80, 3)
            assert torch.all(sample[key][:, 0] == source_row)
            assert torch.all(sample[key][:, 1] == view)
    assert len(SpatiotemporalViewDataset(tmp_path, 'train')) == 8


def test_static_analysis_disables_temporal_batch_requirement():
    cfg = OmegaConf.create(dict(data=dict(kind="static"), vicreg_temporal_view=True))
    _apply_analysis_inference_overrides(cfg)
    assert not cfg.vicreg_temporal_view


def test_mace_analysis_disables_only_radial_compilation():
    cfg = OmegaConf.load('configs/vicreg_pretrained_mace_geometry.yaml')
    cfg.data.kind = 'static'
    before = OmegaConf.to_container(cfg.encoder.kwargs.performance)
    _apply_analysis_inference_overrides(cfg)
    assert OmegaConf.to_container(cfg.encoder.kwargs.performance) == dict(before, compile_radial_mlp=False)
    assert cfg.encoder.kwargs.reference_radius_A == 9.192189
    training_cfg = OmegaConf.load('configs/vicreg_pretrained_mace_geometry.yaml')
    assert training_cfg.encoder.kwargs.performance.compile_radial_mlp


def test_last_checkpoint_tracks_epoch_when_validation_worsens(tmp_path):
    class EpochCheckpoint(ProgressCheckpoint):
        def on_fit_start(self, trainer, pl_module):
            self.progress(trainer, state="training")

    class TinyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("loss/val", float(self.current_epoch + 1), batch_size=2)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=.01)

    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)
    callback = EpochCheckpoint(tmp_path)
    trainer = pl.Trainer(max_epochs=3, accelerator="cpu", logger=False, callbacks=[callback],
                         enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(TinyModel(), loader, loader)
    best = torch.load(callback.best_model_path, weights_only=False)
    last = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert best["epoch"] == 0
    assert last["epoch"] == 2
    assert last["global_step"] == 3
