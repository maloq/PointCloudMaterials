from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.analysis.connected_regimes import (
    ConnectedRegimeSettings,
    resolve_connected_regime_settings,
    run_connected_regime_analysis,
)
from src.analysis.representative_transitions import (
    select_transition_representative_rows,
    select_within_cluster_representative_rows,
)


def _connected_arc_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(18)
    angle_a = np.linspace(-0.90, -0.015, 140)
    angle_b = np.linspace(0.015, 0.90, 140)
    angle_c = np.linspace(2.70, 3.55, 140)
    angles = np.concatenate([angle_a, angle_b, angle_c])
    labels = np.repeat(np.arange(3), 140)
    latents = np.column_stack(
        [
            np.cos(angles),
            np.sin(angles),
            0.03 * rng.normal(size=angles.size),
            0.03 * rng.normal(size=angles.size),
        ]
    ).astype(np.float32)
    latents /= np.linalg.norm(latents, axis=1, keepdims=True)
    return latents, labels


def test_connected_regime_analysis_finds_touching_distinct_pair(tmp_path: Path) -> None:
    latents, labels = _connected_arc_data()
    settings = ConnectedRegimeSettings(
        enabled=True,
        explicit_pairs=(),
        auto_detect=True,
        auto_max_pairs=1,
        max_samples_per_cluster=140,
        neighbor_k=12,
        min_cross_neighbor_fraction=0.001,
        pca_components=4,
        histogram_bins=24,
        random_state=7,
    )

    metrics = run_connected_regime_analysis(
        latents=latents,
        labels=labels,
        out_dir=tmp_path,
        settings=settings,
        cluster_color_map={0: "#d1495b", 1: "#4c956c", 2: "#3a86ff"},
    )

    assert metrics["selection"]["selected_pairs"] == [[0, 1]]
    pair = metrics["pairs"][0]
    assert pair["symmetric_cross_neighbor_fraction"] > 0.0
    assert pair["boundary_sample_fraction"]["0"] > 0.0
    assert pair["boundary_sample_fraction"]["1"] > 0.0
    assert pair["cohen_d"] > 1.0
    assert pair["wasserstein_order_parameter"] > 0.0
    assert 0.0 <= pair["distribution_overlap"] <= 1.0
    assert (tmp_path / pair["artifacts"]["figure"]).is_file()
    assert (tmp_path / pair["artifacts"]["order_parameter_csv"]).is_file()
    assert (tmp_path / metrics["artifacts"]["overview_figure"]).is_file()
    assert (tmp_path / metrics["artifacts"]["metrics_json"]).is_file()


def test_connected_regime_settings_reject_ambiguous_disabled_selection() -> None:
    cfg = OmegaConf.create(
        {
            "clustering": {
                "connected_regimes": {
                    "enabled": True,
                    "pairs": [],
                    "auto_detect": False,
                }
            }
        }
    )

    try:
        resolve_connected_regime_settings(cfg, default_random_state=0)
    except ValueError as exc:
        assert "pairs is empty" in str(exc)
    else:
        raise AssertionError("Expected an explicit configuration error for no pair selection.")


def test_connected_regime_explicit_pair_requires_available_clusters(tmp_path: Path) -> None:
    latents, labels = _connected_arc_data()
    settings = ConnectedRegimeSettings(
        enabled=True,
        explicit_pairs=((0, 9),),
        auto_detect=False,
        auto_max_pairs=1,
        max_samples_per_cluster=140,
        neighbor_k=12,
        min_cross_neighbor_fraction=0.001,
        pca_components=4,
        histogram_bins=24,
        random_state=7,
    )

    try:
        run_connected_regime_analysis(
            latents=latents,
            labels=labels,
            out_dir=tmp_path,
            settings=settings,
            cluster_color_map={0: "#d1495b", 1: "#4c956c", 2: "#3a86ff"},
        )
    except ValueError as exc:
        assert "unavailable cluster IDs" in str(exc)
    else:
        raise AssertionError("Expected unavailable explicit cluster IDs to fail loudly.")


def test_transition_selection_follows_centroid_direction_with_real_rows() -> None:
    latents = np.asarray(
        [
            [-1.1, 0.1],
            [-0.9, -0.1],
            [-0.2, 0.0],
            [0.2, 0.0],
            [0.9, 0.1],
            [1.1, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    rows, targets, actual = select_transition_representative_rows(
        latents, labels, cluster_a=0, cluster_b=1, steps=5
    )

    assert len(np.unique(rows)) == 5
    np.testing.assert_allclose(targets, np.linspace(0.0, 1.0, 5))
    assert labels[rows[0]] == 0
    assert labels[rows[-1]] == 1
    assert actual[rows[0]] < actual[rows[-1]]


def test_within_cluster_selection_follows_dominant_direction_with_real_rows() -> None:
    parameter = np.linspace(-2.0, 2.0, 80, dtype=np.float32)
    latents = np.column_stack(
        [parameter, 0.05 * np.sin(parameter), 0.02 * np.cos(parameter)]
    )

    rows, targets, actual = select_within_cluster_representative_rows(
        latents, steps=30
    )

    assert len(rows) == 30
    assert len(np.unique(rows)) == 30
    np.testing.assert_allclose(targets, np.linspace(0.0, 1.0, 30))
    assert actual[rows[0]] < actual[rows[-1]]


def test_connected_regime_renders_manual_real_structure_paths(
    tmp_path: Path,
) -> None:
    latents, labels = _connected_arc_data()

    class LocalStructureDataset:
        def __len__(self) -> int:
            return len(latents)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            rng = np.random.default_rng(int(index))
            points = rng.normal(size=(24, 3)).astype(np.float32)
            points[0] = 0.0
            points[:, 0] *= 0.7 + 0.4 * float(index) / float(len(latents) - 1)
            return {"points": torch.from_numpy(points)}

    settings = ConnectedRegimeSettings(
        enabled=True,
        explicit_pairs=((0, 1),),
        auto_detect=False,
        auto_max_pairs=1,
        max_samples_per_cluster=140,
        neighbor_k=12,
        min_cross_neighbor_fraction=0.001,
        pca_components=4,
        histogram_bins=24,
        random_state=7,
        interactive_3d=True,
        representative_steps=30,
    )
    representatives_dir = tmp_path / "real_md" / "representatives"

    metrics = run_connected_regime_analysis(
        latents=latents,
        labels=labels,
        out_dir=tmp_path,
        settings=settings,
        cluster_color_map={0: "#d1495b", 1: "#4c956c", 2: "#3a86ff"},
        dataset=LocalStructureDataset(),
        representatives_out_dir=representatives_dir,
        representative_target_points=18,
    )

    pair = metrics["pairs"][0]
    interactive_path = tmp_path / pair["artifacts"]["interactive_3d_representatives"]
    index_path = tmp_path / metrics["artifacts"]["interactive_3d_index"]
    gallery_path = tmp_path / metrics["artifacts"]["interactive_3d_cluster_gallery"]
    assert interactive_path.is_file()
    assert index_path.is_file()
    assert gallery_path.is_file()
    assert len(pair["interactive_3d_representatives"]["steps"]) == 30
    html = interactive_path.read_text(encoding="utf-8")
    assert "latent path t" in html
    assert "Play" not in html and "Pause" not in html
    assert '"sliders"' in html
    assert '"mode":"lines"' in html
    assert '"width":1.0' in html
    assert '"size":8.5' in html

    within = metrics["interactive_3d"]["within_cluster_transitions"]
    assert len(within) == 1
    assert within[0]["cluster_ids"] == [2]
    assert len(within[0]["steps"]) == 30
    within_path = tmp_path / within[0]["artifact"]
    assert within_path.is_file()
    within_html = within_path.read_text(encoding="utf-8")
    assert "dominant within-cluster direction" in within_html
    assert '"sliders"' in within_html
    assert "Play" not in within_html and "Pause" not in within_html
    assert '"mode":"lines"' in within_html
    assert '"width":1.0' in within_html
    assert '"size":8.5' in within_html
    assert "C3 within-cluster path" in index_path.read_text(encoding="utf-8")

    gallery_html = gallery_path.read_text(encoding="utf-8")
    assert "Interactive 3D cluster representatives" in gallery_html
    assert '"mode":"lines"' in gallery_html
    assert '"width":1.0' in gallery_html
    assert '"size":8.5' in gallery_html
