from types import SimpleNamespace

import pytest
import torch

from src.utils.training_utils import get_optimizers_and_scheduler


def _hparams(**overrides):
    values = {
        "scheduler_name": "Cosine",
        "learning_rate": 1.0e-3,
        "decay_rate": 0.04,
        "enable_swa": False,
        "swa_epoch_start": 1000,
        "epochs": 100,
        "scheduler_min_lr": 1.0e-6,
        "warmup_enabled": True,
        "warmup_epochs": 10,
        "warmup_start_factor": 0.05,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _lr_sequence(hparams) -> tuple[list[float], dict]:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizers, scheduler_configs = get_optimizers_and_scheduler(hparams, [parameter])
    optimizer = optimizers[0]
    scheduler_config = scheduler_configs[0]
    scheduler = scheduler_config["scheduler"]

    learning_rates = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(int(hparams.epochs)):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
    return learning_rates, scheduler_config


def test_warmup_cosine_reaches_base_lr_and_configured_final_lr() -> None:
    learning_rates, scheduler_config = _lr_sequence(_hparams())

    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["frequency"] == 1
    assert learning_rates[0] == pytest.approx(5.0e-5)
    assert learning_rates[10] == pytest.approx(1.0e-3)
    assert learning_rates[11] < learning_rates[10]
    assert learning_rates[100] == pytest.approx(1.0e-6)
    assert learning_rates[:11] == sorted(learning_rates[:11])
    assert learning_rates[10:] == sorted(learning_rates[10:], reverse=True)


def test_cosine_without_warmup_uses_the_full_training_duration() -> None:
    learning_rates, _ = _lr_sequence(
        _hparams(
            epochs=20,
            warmup_enabled=False,
        )
    )

    assert learning_rates[0] == pytest.approx(1.0e-3)
    assert learning_rates[1] < learning_rates[0]
    assert learning_rates[20] == pytest.approx(1.0e-6)
    assert learning_rates == sorted(learning_rates, reverse=True)


@pytest.mark.parametrize(
    ("field_overrides", "message"),
    [
        ({"warmup_epochs": 100}, "leave at least one epoch"),
        ({"warmup_epochs": 101}, "leave at least one epoch"),
        ({"scheduler_min_lr": -1.0e-6}, "between zero and learning_rate"),
        ({"scheduler_min_lr": 2.0e-3}, "between zero and learning_rate"),
    ],
)
def test_scheduler_rejects_ambiguous_or_invalid_ranges(field_overrides, message) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    with pytest.raises(ValueError, match=message):
        get_optimizers_and_scheduler(_hparams(**field_overrides), [parameter])
