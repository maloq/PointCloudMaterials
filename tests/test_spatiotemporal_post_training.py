import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.training_methods.spatiotemporal import ProgressCheckpoint
from src.analysis.config import _apply_analysis_inference_overrides


def test_static_analysis_disables_temporal_batch_requirement():
    cfg = OmegaConf.create(dict(data=dict(kind="static"), vicreg_temporal_view=True))
    _apply_analysis_inference_overrides(cfg)
    assert not cfg.vicreg_temporal_view


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
