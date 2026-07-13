import pytorch_lightning as pl

from src.data_utils.data_kinds import normalize_data_kind
from src.data_utils.data_modules.line_lammps import LineLAMMPSDataModule
from src.data_utils.data_modules.line_static import LineStaticDataModule
from src.data_utils.data_modules.static import StaticPointCloudDataModule
from src.data_utils.data_modules.synthetic import SyntheticPointCloudDataModule
from src.data_utils.data_modules.temporal_lammps import TemporalLAMMPSDataModule


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kind = normalize_data_kind(cfg.data.kind)
        if kind == "synthetic":
            self.impl = SyntheticPointCloudDataModule(cfg)
        elif kind == "line_static":
            self.impl = LineStaticDataModule(cfg)
        elif kind == "line_lammps":
            self.impl = LineLAMMPSDataModule(cfg)
        elif kind == "temporal_lammps":
            self.impl = TemporalLAMMPSDataModule(cfg)
        elif kind == "static":
            self.impl = StaticPointCloudDataModule(cfg)
        else:
            raise ValueError(
                "Unsupported data.kind. Expected one of "
                "['static', 'synthetic', 'temporal_lammps', 'line_lammps', 'line_static'] "
                f"got {cfg.data.kind!r}."
            )

    def setup(self, stage=None):
        return self.impl.setup(stage)

    def train_dataloader(self):
        return self.impl.train_dataloader()

    def val_dataloader(self):
        return self.impl.val_dataloader()

    def test_dataloader(self):
        return self.impl.test_dataloader()
