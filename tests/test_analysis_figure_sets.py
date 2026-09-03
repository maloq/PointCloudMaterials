from pathlib import Path

import numpy as np
import pytest

import src.analysis.cluster_figures as cluster_figures
import src.analysis.cluster_blender as cluster_blender
import src.analysis.figure_sets as figure_sets
from src.analysis.figure_sets import (
    _build_unique_snapshot_output_names,
    _sanitize_snapshot_output_name,
)
from src.analysis.representative_structures import detect_crystal_like_cluster_ids


def test_snapshot_output_name_preserves_decimal_time_labels() -> None:
    assert _sanitize_snapshot_output_name("182.8ps") == "182_8ps"


def test_snapshot_output_name_strips_repository_data_suffixes() -> None:
    assert _sanitize_snapshot_output_name("000000step.npy") == "000000step"
    assert _sanitize_snapshot_output_name("trajectory.lammpstrj") == "trajectory"


def test_snapshot_output_names_remain_unique_after_sanitization() -> None:
    assert _build_unique_snapshot_output_names(["182.8ps", "182_8ps"]) == {
        "182.8ps": "182_8ps",
        "182_8ps": "182_8ps_2",
    }


def test_crystal_like_cluster_detection_uses_repository_ptm_definition() -> None:
    summary = {
        "ptm_enabled": True,
        "representatives": [
            {"cluster_id": 0, "ptm": {"center_structure_type_id": 1}},  # FCC
            {"cluster_id": 1, "ptm": {"center_structure_type_id": 0}},  # Other
            {"cluster_id": 2, "ptm": {"center_structure_type_id": 2}},  # HCP
            {"cluster_id": 3, "ptm": {"center_structure_type_id": 3}},  # BCC
            {"cluster_id": 4, "ptm": {"center_structure_type_id": 4}},  # ICO
        ],
    }

    assert detect_crystal_like_cluster_ids(summary) == [0, 2, 3]


def test_crystal_like_cluster_detection_requires_ptm() -> None:
    with pytest.raises(RuntimeError, match="ptm_enabled=true"):
        detect_crystal_like_cluster_ids(
            {"ptm_enabled": False, "representatives": []}
        )


def test_snapshot_gallery_uses_per_snapshot_crystal_like_clusters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def panel(path: Path) -> dict[str, object]:
        path.write_bytes(b"raytrace")
        return {
            "view_name": "view1",
            "raytrace_render": {"out_file": str(path)},
        }

    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    first_all = panel(tmp_path / "first_all.png")
    first_crystal = panel(tmp_path / "first_crystal.png")
    second_all = panel(tmp_path / "second_all.png")
    saved_galleries: list[tuple[list[Path], Path, list[str]]] = []

    def save_gallery(
        panel_paths: list[Path],
        *,
        out_file: Path,
        panel_titles: list[str],
    ) -> None:
        saved_galleries.append((panel_paths, out_file, panel_titles))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"gallery")

    monkeypatch.setattr(figure_sets, "_save_horizontal_image_gallery", save_gallery)
    summary = figure_sets._save_snapshot_raytrace_galleries_by_view(
        {
            "root_dir": str(snapshot_root),
            "k_value": 7,
            "snapshots": [
                {
                    "source_name": "175ps",
                    "output_name": "175ps",
                    "figure_set": {
                        "panel_all_clusters_views": [first_all],
                        "crystal_like_cluster_ids": [0, 1, 2],
                        "panel_crystal_like_views": [first_crystal],
                    },
                },
                {
                    "source_name": "166ps",
                    "output_name": "166ps",
                    "figure_set": {
                        "panel_all_clusters_views": [second_all],
                        "crystal_like_cluster_ids": [],
                        "panel_crystal_like_views": [],
                    },
                },
            ],
        }
    )

    assert len(saved_galleries) == 2
    assert saved_galleries[1][0] == [tmp_path / "first_crystal.png"]
    assert saved_galleries[1][1].name == (
        "02_md_clusters_crystal_like_k7_view1_raytrace_gallery.png"
    )
    assert summary["crystal_like"]["snapshots"] == [
        {
            "source_name": "175ps",
            "output_name": "175ps",
            "cluster_ids": [0, 1, 2],
            "rendered": True,
        },
        {
            "source_name": "166ps",
            "output_name": "166ps",
            "cluster_ids": [],
            "rendered": False,
        },
    ]


def test_fixed_k_figure_set_renders_only_detected_crystal_like_clusters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure_analysis = {
        "ptm_enabled": True,
        "representatives": [
            {"cluster_id": 0, "ptm": {"center_structure_type_id": 1}},
            {"cluster_id": 1, "ptm": {"center_structure_type_id": 0}},
            {"cluster_id": 2, "ptm": {"center_structure_type_id": 2}},
        ],
    }
    monkeypatch.setattr(
        cluster_figures,
        "_save_cluster_representatives_figure",
        lambda *args, **kwargs: {
            "structure_analysis": structure_analysis,
            "pca_two_shell_figures": {},
        },
    )
    rendered: list[tuple[str, list[int] | None]] = []

    def save_snapshot(
        coords: np.ndarray,
        labels: np.ndarray,
        color_map: dict[int, str],
        out_file: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        visible = kwargs["visible_cluster_ids"]
        rendered.append(
            (
                out_file.name,
                None if visible is None else [int(v) for v in visible],
            )
        )
        return {"out_file": str(out_file)}

    monkeypatch.setattr(cluster_figures, "_save_md_cluster_snapshot", save_snapshot)
    result = cluster_figures._save_fixed_k_cluster_figure_set(
        out_dir=tmp_path,
        dataset=object(),
        latents=np.zeros((4, 2), dtype=np.float32),
        coords=np.zeros((4, 3), dtype=np.float32),
        cluster_labels=np.asarray([0, 1, 2, 0], dtype=int),
        k_value=3,
        point_scale=1.0,
        l2_normalize=True,
        standardize=True,
        pca_variance=0.99,
        pca_max_components=2,
        md_max_points=None,
        icl_enabled=False,
        icl_k_min=2,
        icl_k_max=4,
        icl_max_samples=None,
        representative_points=16,
        md_point_size=5.6,
        md_point_alpha=0.62,
        md_halo_scale=1.0,
        md_halo_alpha=0.0,
        md_saturation_boost=1.18,
        md_view_elev=24.0,
        md_view_azim=35.0,
        md_num_views=1,
        representative_orientation_method="pca",
        representative_view_elev=22.0,
        representative_view_azim=38.0,
        representative_projection="ortho",
        representative_ptm_enabled=True,
        representative_cna_enabled=False,
        representative_cna_max_signatures=5,
        representative_center_atom_tolerance=1e-6,
        representative_shell_min_neighbors=8,
        representative_shell_max_neighbors=24,
    )

    assert rendered == [
        ("01_md_clusters_all_k3.png", None),
        ("02_md_clusters_crystal_like_k3.png", [0, 2]),
    ]
    assert result["crystal_like_cluster_ids"] == [0, 2]


def test_raytrace_views_are_batched_into_one_blender_process_per_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure_analysis = {
        "ptm_enabled": True,
        "representatives": [
            {"cluster_id": 0, "ptm": {"center_structure_type_id": 1}},
            {"cluster_id": 1, "ptm": {"center_structure_type_id": 0}},
            {"cluster_id": 2, "ptm": {"center_structure_type_id": 2}},
        ],
    }
    monkeypatch.setattr(
        cluster_figures,
        "_save_cluster_representatives_figure",
        lambda *args, **kwargs: {
            "structure_analysis": structure_analysis,
            "pca_two_shell_figures": {},
        },
    )
    monkeypatch.setattr(
        cluster_figures,
        "_save_md_cluster_snapshot",
        lambda *args, **kwargs: {"out_file": str(args[3])},
    )
    batch_calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def save_raytrace_batch(
        coords: np.ndarray,
        labels: np.ndarray,
        color_map: dict[int, str],
        render_jobs: list[dict[str, object]],
        **kwargs: object,
    ) -> list[dict[str, object]]:
        batch_calls.append((render_jobs, kwargs))
        return [
            {"out_file": str(job["out_file"]), "batch_size": len(render_jobs)}
            for job in render_jobs
        ]

    monkeypatch.setattr(
        cluster_figures,
        "_save_md_cluster_snapshots_raytrace_blender",
        save_raytrace_batch,
    )
    result = cluster_figures._save_fixed_k_cluster_figure_set(
        out_dir=tmp_path,
        dataset=object(),
        latents=np.zeros((4, 2), dtype=np.float32),
        coords=np.zeros((4, 3), dtype=np.float32),
        cluster_labels=np.asarray([0, 1, 2, 0], dtype=int),
        k_value=3,
        point_scale=1.0,
        l2_normalize=True,
        standardize=True,
        pca_variance=0.99,
        pca_max_components=2,
        md_max_points=None,
        icl_enabled=False,
        icl_k_min=2,
        icl_k_max=4,
        icl_max_samples=None,
        representative_points=16,
        md_point_size=5.6,
        md_point_alpha=0.62,
        md_halo_scale=1.0,
        md_halo_alpha=0.0,
        md_saturation_boost=1.18,
        md_view_elev=24.0,
        md_view_azim=35.0,
        md_num_views=2,
        representative_orientation_method="pca",
        representative_view_elev=22.0,
        representative_view_azim=38.0,
        representative_projection="ortho",
        representative_ptm_enabled=True,
        representative_cna_enabled=False,
        representative_cna_max_signatures=5,
        representative_center_atom_tolerance=1e-6,
        representative_shell_min_neighbors=8,
        representative_shell_max_neighbors=24,
        raytrace_render_enabled=True,
    )

    assert len(batch_calls) == 1
    jobs, kwargs = batch_calls[0]
    assert [Path(job["out_file"]).name for job in jobs] == [
        "01_md_clusters_all_k3_raytrace.png",
        "01_md_clusters_all_k3_view2_raytrace.png",
        "02_md_clusters_crystal_like_k3_raytrace.png",
        "02_md_clusters_crystal_like_k3_view2_raytrace.png",
    ]
    assert [job["visible_cluster_ids"] for job in jobs] == [
        None,
        None,
        [0, 2],
        [0, 2],
    ]
    assert kwargs["image_width"] == 1200
    assert kwargs["cycles_samples"] == 32
    assert kwargs["use_denoise"] is True
    assert result["raytrace_render_settings"]["blender_processes_per_snapshot"] == 1
    assert result["raytrace_render_settings"]["renders_per_blender_process"] == 4


def test_blender_batch_builds_cluster_scene_once_for_multiple_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run_batch(
        blender_exec: str,
        blender_script: str,
        payload: dict[str, object],
        out_files: list[Path],
        **kwargs: object,
    ) -> None:
        captured.update(
            {
                "blender_exec": blender_exec,
                "script": blender_script,
                "payload": payload,
                "out_files": out_files,
            }
        )

    monkeypatch.setattr(
        cluster_blender,
        "_resolve_blender_executable",
        lambda executable: "/usr/bin/blender",
    )
    monkeypatch.setattr(cluster_blender, "_run_blender_render_batch", run_batch)
    outputs = [tmp_path / "all.png", tmp_path / "crystal.png"]
    metadata = cluster_blender._save_md_cluster_snapshots_raytrace_blender(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 0, 1, 1], dtype=int),
        {0: "#ff0000", 1: "#0000ff"},
        [
            {
                "out_file": outputs[0],
                "title": "all",
                "visible_cluster_ids": None,
                "view_elev": 24.0,
                "view_azim": 35.0,
            },
            {
                "out_file": outputs[1],
                "title": "crystal",
                "visible_cluster_ids": [1],
                "view_elev": 24.0,
                "view_azim": 125.0,
            },
        ],
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert len(payload["clusters"]) == 2
    assert len(payload["renders"]) == 2
    assert captured["out_files"] == outputs
    assert "for render_index, render_spec in enumerate(payload[\"renders\"])" in captured[
        "script"
    ]
    assert metadata[0]["num_points_rendered"] == 4
    assert metadata[1]["num_points_rendered"] == 2
    assert metadata[0]["batch_size"] == 2
