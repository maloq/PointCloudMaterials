from pathlib import Path

import numpy as np

from src.data_utils.synthetic import load_config
from src.vis_tools.md_cluster_plot import (
    load_coords_clusters,
    render_interactive_md_clusters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_polycrystal_config_is_balanced_and_paper_sized() -> None:
    global_config, phase_configs, grain_assignment = load_config(
        REPO_ROOT / "configs/data/paper_demo_polycrystal.yaml"
    )

    assert global_config.L == 64.0
    assert global_config.grain_count == 8
    assert set(phase_configs) == {
        "amorphous_pure",
        "bcc_iron",
        "fcc_iron",
        "hcp_iron",
    }
    assert grain_assignment.mode == "explicit"
    assert grain_assignment.assignments is not None
    assert {
        phase: grain_assignment.assignments.count(phase)
        for phase in phase_configs
    } == {phase: 2 for phase in phase_configs}

    for phase_name in ("bcc_iron", "fcc_iron", "hcp_iron"):
        perturbations = phase_configs[phase_name].perturbations
        assert perturbations.temperature_K == 150.0
        assert perturbations.p_dropout == 0.003
        assert perturbations.rot_bubble_prob == 0.10
        assert perturbations.density_bubbles == []

    amorphous = phase_configs["amorphous_pure"].perturbations
    assert amorphous.sigma_thermal == 0.16
    assert amorphous.p_dropout == 0.003
    assert amorphous.rot_bubble_prob == 0.05
    assert amorphous.density_bubbles == []


def test_post_training_md_renderer_uses_physical_coordinates(tmp_path: Path) -> None:
    coords = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    clusters = np.array([0, 0, 1, 1], dtype=np.int64)
    np.savez(
        tmp_path / "local_structure_coords_clusters.npz",
        coords=coords,
        clusters=clusters,
    )
    loaded_coords, loaded_clusters = load_coords_clusters(tmp_path)

    output = render_interactive_md_clusters(
        tmp_path,
        aspect_mode="data",
    )

    html = output.read_text(encoding="utf-8")
    assert output.name == "md_space_clusters.html"
    np.testing.assert_array_equal(loaded_coords, coords)
    np.testing.assert_array_equal(loaded_clusters, clusters)
    assert "scatter3d" in html
    assert "MD local-structure clusters" in html
