from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/plot_homogeneous_checkpoint.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "pointcloudmaterials_plot_homogeneous_checkpoint",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load checkpoint visualization script from {SCRIPT_PATH}.")
plot_homogeneous_checkpoint = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(plot_homogeneous_checkpoint)


def test_render_checkpoint_visualizations_writes_live_and_step_stamped_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "checkpoints" / "replica_000" / "step_000000000135"
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text(
        json.dumps(
            {
                "replica_name": "replica_000",
                "completed_global_step": 135,
                "equilibration_steps": 5,
                "planned_measurement_steps": 130,
            }
        ),
        encoding="utf-8",
    )
    trace = SimpleNamespace(positions_A=np.zeros((3, 4, 3), dtype=np.float32))
    homogeneous = SimpleNamespace(
        equilibration_steps=5,
        steps=130,
        sample_interval=10,
        temperature_K=500.0,
        generator=SimpleNamespace(
            potential=SimpleNamespace(model_name="test-model"),
            system=SimpleNamespace(chemical_symbol="Al"),
            dynamics=SimpleNamespace(timestep_fs=1.0, pressure_GPa=0.0),
            validation=SimpleNamespace(maximum_liquid_crystalline_fraction=0.1),
        ),
        analysis=SimpleNamespace(
            nucleus_size_threshold_atoms=100,
            threshold_persistence_frames=3,
            ptm_rmsd_cutoff=0.1,
        ),
    )
    config = SimpleNamespace(output_root=tmp_path, homogeneous=homogeneous)

    monkeypatch.setattr(
        plot_homogeneous_checkpoint,
        "_latest_verified_snapshot",
        lambda checkpoint_directory: (snapshot, 135),
    )
    monkeypatch.setattr(
        plot_homogeneous_checkpoint,
        "_load_trace",
        lambda checkpoint_snapshot: trace,
    )
    monkeypatch.setattr(
        plot_homogeneous_checkpoint,
        "_load_online_arrays",
        lambda checkpoint_snapshot: {
            "measurement_step": np.array([130], dtype=np.int64),
            "crystalline_fraction": np.array([0.5]),
            "crystalline_cluster_count": np.array([2], dtype=np.int64),
            "largest_crystalline_cluster_atoms": np.array([200], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        plot_homogeneous_checkpoint,
        "_retained_checkpoint_steps",
        lambda checkpoint_directory: (135,),
    )

    def write_dashboard(path: Path, **kwargs: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dashboard")

    def write_structure_slices(path: Path, **kwargs: object) -> None:
        path.write_bytes(b"structure")

    monkeypatch.setattr(
        plot_homogeneous_checkpoint,
        "_plot_dashboard",
        write_dashboard,
    )
    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.transition_analysis."
        "write_structure_slice_visualization",
        write_structure_slices,
    )

    observed_snapshot, observed_step, outputs = (
        plot_homogeneous_checkpoint.render_checkpoint_visualizations(
            config,
            include_structure_slices=True,
            step_stamped=True,
        )
    )

    dashboard = tmp_path / "visualizations/replica_000_checkpoint_dashboard.png"
    structure = (
        tmp_path / "visualizations/replica_000_checkpoint_structure_slices.png"
    )
    stamped_dashboard = (
        tmp_path
        / "visualizations/replica_000_checkpoint_dashboard_step_000000000135.png"
    )
    stamped_structure = (
        tmp_path
        / "visualizations/"
        "replica_000_checkpoint_structure_slices_step_000000000135.png"
    )
    assert observed_snapshot == snapshot
    assert observed_step == 135
    assert outputs == (
        dashboard,
        structure,
        stamped_dashboard,
        stamped_structure,
    )
    assert dashboard.read_bytes() == stamped_dashboard.read_bytes() == b"dashboard"
    assert structure.read_bytes() == stamped_structure.read_bytes() == b"structure"
