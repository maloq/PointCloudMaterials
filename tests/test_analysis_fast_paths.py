import json

import numpy as np
from omegaconf import OmegaConf

from src.analysis.lazy_static_dataset import LazyStaticAnalysisDataset
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
    assert profile.directional_line_jepa_enabled
    assert profile.tsne_max_samples is None
    assert profile.directional_max_directions is None
    assert profile.directional_max_atoms_total is None


def test_full_runtime_profile_does_not_override_directional_settings() -> None:
    profile = resolve_analysis_runtime_profile(
        OmegaConf.create({"runtime": {"profile": "full"}})
    )

    assert profile.directional_line_jepa_enabled is None
    assert profile.directional_max_directions is None
    assert profile.directional_max_atoms_total is None


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
