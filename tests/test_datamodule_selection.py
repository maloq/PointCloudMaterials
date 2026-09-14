"""Characterize ordinary selection independently of numerical training."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.data_utils.data_modules import registry
from src.training_methods import trainer


KINDS = (
    ("static", "StaticPointCloudDataModule"),
    ("synthetic", "SyntheticPointCloudDataModule"),
    ("temporal_lammps", "TemporalLAMMPSDataModule"),
    ("spatiotemporal_binary", "SpatiotemporalViewDataModule"),
    ("relaxed_histories", "RelaxedHistoryDataModule"),
)


class ModelConstructionReached(Exception):
    """Stop after selection, before trainer/model side effects."""


@pytest.mark.parametrize("kind,class_name", KINDS)
@pytest.mark.parametrize("override", [False, True])
def test_trainer_selection_precedes_model_construction(
    monkeypatch, tmp_path, kind, class_name, override
):
    from src.data_utils import relaxed_histories, spatiotemporal_views

    calls = []

    def construct(cfg):
        calls.append((class_name, cfg))
        return SimpleNamespace(cfg=cfg)

    for module in (registry, trainer):
        if hasattr(module, class_name):
            monkeypatch.setattr(module, class_name, construct)
    if kind == "relaxed_histories":
        monkeypatch.setattr(relaxed_histories, class_name, construct)
    if kind == "spatiotemporal_binary":
        monkeypatch.setattr(spatiotemporal_views, class_name, construct)

    def custom(cfg):
        calls.append(("custom", cfg))
        return SimpleNamespace(cfg=cfg)

    class Model:
        data_module_class = staticmethod(custom) if override else None

        def __init__(self, cfg):
            raise ModelConstructionReached

    monkeypatch.setattr(trainer, "_seed_training_run", lambda cfg: None)
    monkeypatch.setattr(trainer, "init_wandb", lambda cfg, path: None)
    cfg = OmegaConf.create({"data": {"kind": f" {kind.upper()} "}})
    with pytest.raises(ModelConstructionReached):
        trainer.train_model(cfg, Model, run_dir=str(tmp_path))
    assert calls == [("custom" if override else class_name, cfg)]


def test_trainer_rejects_unknown_kind(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "_seed_training_run", lambda cfg: None)
    monkeypatch.setattr(trainer, "init_wandb", lambda cfg, path: None)
    cfg = OmegaConf.create({"data": {"kind": "unknown"}})
    with pytest.raises(ValueError, match="Unsupported data.kind.*"):
        trainer.train_model(cfg, object, run_dir=str(tmp_path))


@pytest.mark.parametrize("drop_last", [False, True])
def test_temporal_sampler_distributed_order_and_subset_boundary(drop_last):
    from torch.utils.data import DistributedSampler, SequentialSampler, Subset

    from src.data_utils.data_modules.temporal_window import (
        TemporalWindowBatchSampler,
    )

    class DenseWindows(list):
        window_count = 3
        center_count = 2

    dataset = DenseWindows(range(6))
    sampler = TemporalWindowBatchSampler(
        SequentialSampler(dataset), 4, drop_last, dataset=dataset
    )
    assert list(sampler) == ([[0, 1, 2, 3]] if drop_last else [
        [0, 1, 2, 3], [4, 5],
    ])
    assert len(sampler) == (1 if drop_last else 2)
    ranks = []
    for rank in range(2):
        distributed = DistributedSampler(
            dataset, num_replicas=2, rank=rank, shuffle=False
        )
        batches = TemporalWindowBatchSampler(
            distributed, 2, drop_last, dataset=dataset
        )
        ranks.append(list(batches))
    assert ranks == (
        [[[0, 1]], [[2, 3]]] if drop_last else
        [[[0, 1], [4, 5]], [[2, 3], [0, 1]]]
    )
    subset = Subset(dataset, [0, 1, 3])
    with pytest.raises(TypeError, match="window_count and center_count"):
        TemporalWindowBatchSampler(
            SequentialSampler(subset), 2, drop_last, dataset=subset
        )


def test_temporal_sampler_epoch_replay():
    from torch.utils.data import SequentialSampler

    from src.data_utils.data_modules.temporal_window import (
        TemporalWindowBatchSampler,
    )

    class DenseWindows(list):
        window_count = 4
        center_count = 3

    dataset = DenseWindows(range(12))
    sampler = TemporalWindowBatchSampler(
        SequentialSampler(dataset), 5, False, dataset=dataset,
        shuffle_windows=True, shuffle_centers=True,
        mixed_windows_per_batch=2,
    )
    first = list(sampler)
    sampler.sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    assert sorted(i for batch in second for i in batch) == list(range(12))
    assert [len(batch) for batch in second] == [5, 5, 2]
    sampler.sampler.set_epoch(0)
    assert list(sampler) == first


@pytest.mark.parametrize("kind,class_name", KINDS)
def test_public_constructors_return_concrete_datamodule(kind, class_name):
    from src.data_utils import relaxed_histories, spatiotemporal_views
    from src.data_utils.data_module import PointCloudDataModule

    classes = {
        "StaticPointCloudDataModule": registry.StaticPointCloudDataModule,
        "SyntheticPointCloudDataModule": (
            registry.SyntheticPointCloudDataModule
        ),
        "TemporalLAMMPSDataModule": registry.TemporalLAMMPSDataModule,
        "SpatiotemporalViewDataModule": (
            spatiotemporal_views.SpatiotemporalViewDataModule
        ),
        "RelaxedHistoryDataModule": relaxed_histories.RelaxedHistoryDataModule,
    }
    cfg = OmegaConf.create({
        "batch_size": 4, "num_workers": 0, "max_samples": 0,
        "data": {"kind": kind},
    })
    for constructor in (registry.create_datamodule, PointCloudDataModule):
        dm = constructor(cfg)
        assert type(dm) is classes[class_name]
        assert dm.cfg is cfg
        assert dm.batch_size == 4
        dm.batch_size = 2
        assert dm.batch_size == 2
        assert not hasattr(dm, "impl")


def test_override_does_not_require_ordinary_data_config():
    cfg = OmegaConf.create({})
    concrete = SimpleNamespace(cfg=cfg)
    calls = []

    def custom(received):
        calls.append(received)
        return concrete

    method = SimpleNamespace(data_module_class=custom)
    assert registry.create_datamodule(cfg, method) is concrete
    assert calls == [cfg]


def test_selector_reports_unknown_kind():
    cfg = OmegaConf.create({"data": {"kind": "misspelled"}})
    with pytest.raises(ValueError, match="got 'misspelled'"):
        registry.create_datamodule(cfg)


@pytest.mark.parametrize("kind,class_name", KINDS)
def test_checkpoint_evaluation_uses_concrete_selection(monkeypatch, kind,
                                                      class_name):
    from src.training_methods.contrastive_learning import eval_checkpoint

    concrete = SimpleNamespace(setup=lambda stage: stages.append(stage))
    stages = []
    received = []

    def select(cfg, model_class):
        received.append((cfg.data.kind, model_class))
        return concrete

    monkeypatch.setattr(eval_checkpoint, "create_datamodule", select)
    cfg = OmegaConf.create({"data": {"kind": kind}})
    assert eval_checkpoint.build_datamodule(cfg) is concrete
    assert received == [(kind, eval_checkpoint.VICRegModule)]
    assert stages == ["test"]
