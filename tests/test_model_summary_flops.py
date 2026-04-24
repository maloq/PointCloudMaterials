from __future__ import annotations

from omegaconf import OmegaConf
from pytorch_lightning.utilities.model_summary import ModelSummary

from src.training_methods.temporal_ssl.temporal_ssl_module import TemporalSSLModule


def test_temporal_ssl_exposes_example_input_for_lightning_flops() -> None:
    cfg = OmegaConf.create(
        {
            "latent_size": 4,
            "learning_rate": 1e-3,
            "data": {
                "kind": "temporal_lammps",
                "sequence_length": 3,
                "num_points": 8,
                "model_points": 6,
            },
            "encoder": {
                "name": "MLP",
                "kwargs": {
                    "num_points": 6,
                    "latent_size": 4,
                },
            },
            "vicreg_enabled": True,
            "vicreg_weight": 1.0,
            "vicreg_embed_dim": 8,
            "vicreg_view_points": 6,
            "swav_enabled": True,
            "swav_weight": 0.1,
            "swav_projection_dim": 4,
            "swav_hidden_dim": 0,
            "swav_num_prototypes": 3,
            "swav_view_points": 6,
        }
    )

    model = TemporalSSLModule(cfg)
    summary = ModelSummary(model, max_depth=1)

    assert tuple(model.example_input_array["pc"].shape) == (1, 3, 6, 3)
    assert summary.total_flops > 0
    assert sum(summary.flop_counts["encoder"].values()) > 0
    assert sum(summary.flop_counts["vicreg"].values()) > 0
    assert sum(summary.flop_counts["swav"].values()) > 0
