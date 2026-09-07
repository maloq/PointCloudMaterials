"""Workflow-stage regressions for the grouped research commands."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from scripts.run_shooting_ablation import main
from src.temporal_vamp.commands import ablation_multiscale


def test_multiscale_extract_builds_cache_and_stops_before_training(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/shooting_multiscale_ablation1_geoframe_v2_20260901.yaml")
    cfg.output_dir = str(tmp_path / "run")
    cfg.device = "cpu"
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)
    snapshot = SimpleNamespace(parents=list(range(40)), branches=list(range(440)), to_dict=lambda: {"parents": 40, "branches": 440})
    monkeypatch.setattr(ablation_multiscale, "load_shooting_campaigns_snapshot", lambda *a, **k: snapshot)
    base_cache = object()
    monkeypatch.setattr(ablation_multiscale.ShootingEmbeddingCache, "load", lambda path: base_cache)
    monkeypatch.setattr(ablation_multiscale, "load_frozen_encoder", lambda *a, **k: SimpleNamespace(checkpoint_path="fixture.ckpt"))
    calls = []

    def extract(snapshot_arg, base_arg, **kwargs):
        assert snapshot_arg is snapshot
        assert base_arg is base_cache
        calls.append(kwargs["cache_path"])
        return SimpleNamespace(satellite_z=np.empty((40, 2, 16, 128)))

    def unexpected_training(*args, **kwargs):
        raise AssertionError("extract stage must not prepare training targets")

    monkeypatch.setattr(ablation_multiscale, "extract_shooting_context_token_cache", extract)
    monkeypatch.setattr(ablation_multiscale, "prepare_dynamic_target_data", unexpected_training)
    main(["multiscale", "--config", str(config_path), "--stage", "extract"])
    assert calls == [tmp_path / "run/context_tokens"]
    summary = json.loads((tmp_path / "run/extraction_summary.json").read_text())
    assert summary == dict(device="cpu", parents=40, center_atoms=2, satellites_per_center=16, embedding_dim=128)
