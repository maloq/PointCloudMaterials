from types import SimpleNamespace

import pytest
import torch

from src.training_methods.base_ssl_module import BaseSSLModule
from src.training_methods.contrastive_learning.supervised_cache import (
    init_supervised_cache,
    log_supervised_metrics,
)


def test_metric_stage_controls_skip_validation_cache_and_keep_test_metrics():
    cfg = SimpleNamespace(
        supervised_metric_stages=["test"],
        embedding_metric_stages=["test"],
        probe_metric_stages=["test"],
        enable_supervised_metrics=True,
        enable_embedding_metrics=False,
        enable_probe_metrics=True,
    )
    module = SimpleNamespace()
    init_supervised_cache(module, cfg)
    module.cache_train_supervised_metrics = False

    assert not BaseSSLModule._should_cache_supervised_stage(module, "train")
    assert not BaseSSLModule._should_cache_supervised_stage(module, "val")
    assert BaseSSLModule._should_cache_supervised_stage(module, "test")

    module._supervised_cache["val"]["latents"].append(torch.ones(2, 3))
    log_supervised_metrics(module, "val")
    assert module._supervised_cache["val"]["latents"] == []


def test_metric_stage_controls_reject_unknown_stage():
    cfg = SimpleNamespace(probe_metric_stages=["validation"])
    with pytest.raises(ValueError, match="probe_metric_stages contains unsupported stages"):
        init_supervised_cache(SimpleNamespace(), cfg)
