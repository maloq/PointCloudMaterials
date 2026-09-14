"""Select concrete datamodules for ordinary training workflows."""

from src.data.kinds import normalize_data_kind
from src.data.data_modules.static import StaticPointCloudDataModule
from src.data.data_modules.synthetic import SyntheticPointCloudDataModule
from src.data.data_modules.temporal_lammps import TemporalLAMMPSDataModule


def create_datamodule(cfg, model_class=None):
    """Construct one datamodule, checking the method override first."""
    datamodule_class = getattr(model_class, "data_module_class", None)
    if datamodule_class is not None:
        return datamodule_class(cfg)

    kind = normalize_data_kind(cfg.data.kind)
    if kind == "synthetic":
        return SyntheticPointCloudDataModule(cfg)
    if kind == "temporal_lammps":
        return TemporalLAMMPSDataModule(cfg)
    if kind == "spatiotemporal_binary":
        from src.data.spatiotemporal import (
            SpatiotemporalViewDataModule,
        )

        return SpatiotemporalViewDataModule(cfg)
    if kind == "relaxed_histories":
        from src.data.relaxed_histories import RelaxedHistoryDataModule

        return RelaxedHistoryDataModule(cfg)
    if kind == "static":
        return StaticPointCloudDataModule(cfg)
    raise ValueError(
        "Unsupported data.kind. Expected one of "
        "['static', 'synthetic', 'temporal_lammps', "
        "'spatiotemporal_binary', 'relaxed_histories'] "
        f"got {cfg.data.kind!r}."
    )


# Descriptor baselines still use this public constructor spelling.
PointCloudDataModule = create_datamodule
