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


def test_cached_static_file_boundaries_follow_coordinate_resets() -> None:
    first = np.column_stack(
        (np.repeat(np.arange(3, dtype=np.float32), 4), np.zeros(12), np.zeros(12))
    )
    second = first + np.asarray([0.25, 1.0, 0.0], dtype=np.float32)
    third = first + np.asarray([0.5, 2.0, 0.0], dtype=np.float32)
    coords = np.concatenate((first, second, third), axis=0)

    counts = LazyStaticAnalysisDataset._infer_cached_file_counts(coords, file_count=3)

    assert counts == [12, 12, 12]


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
