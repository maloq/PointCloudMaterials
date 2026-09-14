from types import SimpleNamespace

import pytest

from src.training_methods.registry import resolve_training_method


@pytest.mark.parametrize(
    "name,expected,analysis",
    [
        ("vicreg", "VICRegModule", True),
        ("visreg", "VICRegModule", True),
        ("contrastive", "VICRegModule", True),
        ("temporal_ssl", "TemporalSSLModule", False),
        ("temporal_vicreg", "TemporalSSLModule", False),
    ],
)
def test_method_aliases_and_analysis(name, expected, analysis):
    for cfg in (
        SimpleNamespace(model_type=name),
        SimpleNamespace(training_method=name),
        SimpleNamespace(method={"name": name}),
    ):
        module, policy = resolve_training_method(cfg)
        assert module.__name__ == expected
        assert policy is analysis


def test_method_selection_precedence_and_errors():
    cfg = SimpleNamespace(
        method={"name": "temporal_ssl"},
        training_method="vicreg",
        model_type="vicreg",
    )
    assert resolve_training_method(cfg)[0].__name__ == "TemporalSSLModule"
    assert resolve_training_method(cfg, method_name=" VICREG ")[1] is True
    with pytest.raises(ValueError, match="Could not resolve"):
        resolve_training_method(SimpleNamespace())
    with pytest.raises(KeyError, match="Unknown training method"):
        resolve_training_method(method_name="unknown")
