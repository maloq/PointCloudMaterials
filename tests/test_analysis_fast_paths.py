import json

import numpy as np
from omegaconf import OmegaConf

from src.analysis.lazy_static_dataset import LazyStaticAnalysisDataset
from src.analysis.config import (
    _apply_analysis_inference_overrides,
    _resolve_figure_set_settings,
)
from src.analysis.inference_cache import (
    _build_inference_cache_spec,
    _inference_cache_spec_hash,
)
from src.analysis.runtime_profile import resolve_analysis_runtime_profile
from src.analysis.figure_sets import write_figure_only_metrics


def test_fast_runtime_profile_preserves_analysis_counts() -> None:
    profile = resolve_analysis_runtime_profile(
        OmegaConf.create({"runtime": {"profile": "fast"}})
    )

    assert profile.clustering_fit_max_samples is None
    assert profile.snapshot_figure_limit is None
    assert profile.md_num_views is None
    assert not profile.raytrace_enabled
    assert not profile.equivariance_enabled
    assert profile.real_md_projection_method == "pca"
    assert profile.tsne_max_samples is None


def test_analysis_replaces_unsafe_reduce_overhead_compile_mode() -> None:
    cfg = OmegaConf.create(
        {
            "compile_encoder": True,
            "encoder_compile_mode": "reduce-overhead",
            "encoder": {"kwargs": {"deterministic_fps": False}},
        }
    )

    _apply_analysis_inference_overrides(cfg)

    assert cfg.encoder_compile_mode == "default"
    assert cfg.encoder.kwargs.deterministic_fps is True


def test_analysis_disables_batch_dependent_projector_eval() -> None:
    cfg = OmegaConf.create(
        {
            "vicreg_projector_bn_eval_batch_stats": True,
            "compile_encoder": False,
            "encoder": {"kwargs": {"deterministic_fps": False}},
        }
    )

    _apply_analysis_inference_overrides(cfg)

    assert cfg.vicreg_projector_bn_eval_batch_stats is False
    assert cfg.encoder.kwargs.deterministic_fps is True


def test_inference_cache_fingerprints_projector_eval_policy(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = OmegaConf.create(
        {
            "model_type": "vicreg",
            "representation_source": "vicreg_projector",
            "vicreg_projector_mode": "mlp",
            "vicreg_projector_bn_eval_batch_stats": False,
            "batch_size": 32,
            "data": {"kind": "static", "data_files": ["frame.npy"]},
        }
    )

    running_stats_spec = _build_inference_cache_spec(
        checkpoint_path=str(checkpoint),
        cfg=cfg,
        inference_batch_size=16,
        max_batches_latent=None,
        max_samples_total=None,
        seed_base=123,
    )
    cfg.vicreg_projector_bn_eval_batch_stats = True
    batch_stats_spec = _build_inference_cache_spec(
        checkpoint_path=str(checkpoint),
        cfg=cfg,
        inference_batch_size=16,
        max_batches_latent=None,
        max_samples_total=None,
        seed_base=123,
    )

    assert running_stats_spec["version"] == 8
    assert running_stats_spec["representation"] == {
        "source": "vicreg_projector",
        "vicreg_projector_mode": "mlp",
        "vicreg_projector_bn_eval_batch_stats": False,
    }
    assert _inference_cache_spec_hash(running_stats_spec) != _inference_cache_spec_hash(
        batch_stats_spec
    )


def test_lazy_static_file_counts_come_from_sample_cache_metadata(tmp_path) -> None:
    metadata = {
        "total_samples": 18,
        "shards": [
            {"source": "Al", "file": "a.npy", "count": 10},
            {"source": "Al", "file": "b.npy", "count": 8},
        ],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    source = {"name": "Al"}
    entries, counts = LazyStaticAnalysisDataset._cached_file_entries(
        OmegaConf.create({"sample_cache": {"cache_dir": str(tmp_path)}}),
        entries=[(source, "a.npy"), (source, "b.npy")],
        row_count=14,
    )

    assert [file_name for _, file_name in entries] == ["a.npy", "b.npy"]
    assert counts == [10, 4]


def test_figure_only_metrics_keep_runtime_profile(tmp_path) -> None:
    metrics = write_figure_only_metrics(
        metrics_path=tmp_path / "metrics.json",
        all_metrics={
            "clustering": {"primary_k": 3},
            "inference_cache": {"loaded_from_cache": True},
            "runtime_profile": {"name": "fast"},
        },
        multi_snapshot_real=False,
    )

    assert metrics["runtime_profile"] == {"name": "fast"}


def test_raytrace_standard_and_high_quality_presets(tmp_path) -> None:
    model_cfg = OmegaConf.create({"data": {"model_points": 16}})
    standard = _resolve_figure_set_settings(
        OmegaConf.create(
            {
                "figure_set": {
                    "enabled": True,
                    "representatives": {"points": 16, "ptm_enabled": True},
                    "raytrace": {"high_quality": False},
                }
            }
        ),
        model_cfg,
        out_dir=tmp_path,
        primary_k=7,
    )
    high_quality = _resolve_figure_set_settings(
        OmegaConf.create(
            {
                "figure_set": {
                    "enabled": True,
                    "representatives": {"points": 16, "ptm_enabled": True},
                    "raytrace": {
                        "high_quality": True,
                        "resolution": 1200,
                        "samples": 32,
                    },
                }
            }
        ),
        model_cfg,
        out_dir=tmp_path,
        primary_k=7,
    )

    assert standard.md_num_views == 2
    assert standard.raytrace_kwargs["raytrace_render_resolution"] == 1200
    assert standard.raytrace_kwargs["raytrace_render_samples"] == 32
    assert standard.raytrace_kwargs["raytrace_render_denoise"] is True
    assert standard.raytrace_kwargs["raytrace_render_high_quality"] is False
    assert high_quality.raytrace_kwargs["raytrace_render_resolution"] == 1600
    assert high_quality.raytrace_kwargs["raytrace_render_samples"] == 64
    assert high_quality.raytrace_kwargs["raytrace_render_high_quality"] is True
