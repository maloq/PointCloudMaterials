import pytorch_lightning as pl

from src.data_utils.data_kinds import normalize_data_kind
from src.data_utils.data_modules.line_lammps import LineLAMMPSDataModule
from src.data_utils.data_modules.line_static import LineStaticDataModule


class LineJEPADataModule(pl.LightningDataModule):
    """Route Line-JEPA to static or LAMMPS line samplers based on data.kind."""

    def __init__(self, cfg):
        super().__init__()
        kind = normalize_data_kind(getattr(cfg.data, "kind", None), default="static")
        if kind in {"static", "line_static"}:
            self.impl = LineStaticDataModule(cfg)
        elif kind in {"temporal_lammps", "line_lammps"}:
            self.impl = LineLAMMPSDataModule(cfg)
        else:
            raise ValueError(
                "Line-JEPA data.kind must be one of "
                "['static', 'line_static', 'temporal_lammps', 'line_lammps'], "
                f"got {getattr(cfg.data, 'kind', None)!r}."
            )

    def setup(self, stage=None):
        return self.impl.setup(stage)

    def train_dataloader(self):
        return self.impl.train_dataloader()

    def val_dataloader(self):
        return self.impl.val_dataloader()

    def test_dataloader(self):
        return self.impl.test_dataloader()


__all__ = ["LineJEPADataModule"]
