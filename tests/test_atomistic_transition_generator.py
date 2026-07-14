from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic import (
    add_phase_rdf_to_transition_dataset,
    generate_transition_dataset,
    load_transition_config,
)
from src.data_utils.synthetic.atomistic.simulation import ThermodynamicTrace
from src.data_utils.synthetic.atomistic.transition_analysis import analyze_phase_rdf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPOSITORY_ROOT / "configs/data/atomistic_al_phase_context.yaml"
TRANSITION_CONFIG = REPOSITORY_ROOT / "configs/data/atomistic_al_phase_transition.yaml"


def test_phase_rdf_resolves_the_fcc_first_neighbor_shell() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 4))
    atom_count = len(atoms)
    trace = ThermodynamicTrace(
        step=np.array([0], dtype=np.int64),
        temperature_K=np.array([0.0]),
        pressure_GPa=np.array([0.0]),
        volume_A3=np.array([atoms.get_volume()]),
        potential_energy_eV_per_atom=np.array([0.0]),
        positions_A=np.asarray([atoms.positions]),
        cell_vectors_A=np.asarray([atoms.cell.array]),
    )
    rdf = analyze_phase_rdf(
        trace,
        chemical_symbol="Al",
        prepared_phase_ids=np.arange(atom_count, dtype=np.int64) % 3,
        timestep_fs=1.0,
        cutoff_A=4.5,
        bins=90,
        branch_name="fcc_test",
        progress=lambda _message: None,
    )

    assert np.all(rdf.g_r[:, :, rdf.distance_A < 2.8] == 0.0)
    first_peak_A = rdf.distance_A[np.argmax(rdf.g_r[0, 0])]
    assert 2.8 < first_peak_A < 2.9


def test_production_transition_config_is_direct_coexistence() -> None:
    config = load_transition_config(TRANSITION_CONFIG)
    assert config.generator.system.repetitions == (26, 26, 26)
    assert config.crystallization.expected_direction == "growth"
    assert config.crystallization.temperature_K == 650.0
    assert config.melting.expected_direction == "melting"
    assert config.melting.temperature_K == 1100.0
    assert config.sample_interval == 50
    assert config.analysis.rdf_cutoff_A == 8.0
    assert config.analysis.rdf_bins == 160
    assert config.crystallization.steps // config.sample_interval + 1 == 101
    assert config.melting.steps // config.sample_interval + 1 == 101


def test_small_direct_coexistence_round_trip(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    interface_dir = source_root / "replica_000_solid_liquid_interface"
    solid_dir = source_root / "replica_000_bulk_solid"
    interface_dir.mkdir(parents=True)
    solid_dir.mkdir()

    generator_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    generator_raw["system"]["repetitions"] = [3, 3, 4]
    generator_raw["system"]["interface_half_width_A"] = 1.0
    generator_raw["validation"]["maximum_pressure_error_GPa"] = 100.0
    generator_raw["validation"]["maximum_temperature_error_K"] = 1000.0
    generator_raw["output"]["root_dir"] = str(source_root)
    generator_path = tmp_path / "generator.yaml"
    generator_path.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 4))
    atom_count = len(atoms)
    with (interface_dir / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=np.array([0], dtype=np.int64),
            positions_A=np.asarray([atoms.positions], dtype=np.float32),
            cell_vectors_A=np.asarray([atoms.cell.array], dtype=np.float64),
        )
    (interface_dir / "metadata.json").write_text(
        json.dumps(
            {
                "intermediate_regions": [
                    {
                        "definition": {
                            "slab_bounds_fractional": [0.25, 0.75],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (solid_dir / "metadata.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    "number_density_per_A3": atom_count / atoms.get_volume(),
                }
            }
        ),
        encoding="utf-8",
    )
    potential_sha256 = generator_raw["potential"]["sha256"]
    (source_root / "manifest.json").write_text(
        json.dumps({"potential_sha256": potential_sha256}), encoding="utf-8"
    )

    output_root = tmp_path / "transitions"
    transition_raw = {
        "dataset_name": "test_direct_coexistence",
        "source_generator_config": str(generator_path),
        "source_dataset": str(source_root),
        "source_interface_environment": "replica_000_solid_liquid_interface",
        "source_solid_environment": "replica_000_bulk_solid",
        "source_frame_step": 0,
        "random_seed": 24680,
        "sample_interval": 1,
        "analysis": {
            "profile_bins": 4,
            "rdf_cutoff_A": 5.0,
            "rdf_bins": 50,
        },
        "crystallization": {
            "temperature_K": 300.0,
            "steps": 2,
            "minimum_crystalline_fraction_change": 0.0,
        },
        "melting": {
            "temperature_K": 600.0,
            "steps": 2,
            "minimum_crystalline_fraction_change": 0.0,
        },
        "output": {
            "root_dir": str(output_root),
            "overwrite": False,
            "save_extxyz": True,
            "create_visualizations": True,
        },
    }
    transition_path = tmp_path / "transition.yaml"
    transition_path.write_text(yaml.safe_dump(transition_raw), encoding="utf-8")
    config = load_transition_config(transition_path)
    result = generate_transition_dataset(
        config,
        calculator=EMT(),
        progress=lambda _message: None,
    )

    assert [path.name for path in result.branch_dirs] == ["crystallization", "melting"]
    assert (result.output_root / "transition_overview.png").is_file()
    assert (result.output_root / "phase_rdf_overview.png").is_file()
    assert (result.output_root / "structure_slice_overview.png").is_file()
    for branch_dir in result.branch_dirs:
        assert np.load(branch_dir / "atoms.npy").shape == (atom_count, 3)
        with np.load(branch_dir / "trajectory.npz") as trajectory:
            assert trajectory["step"].tolist() == [0, 1, 2]
            assert trajectory["positions_A"].shape == (3, atom_count, 3)
            np.testing.assert_allclose(
                np.linalg.det(trajectory["cell_vectors_A"]),
                trajectory["volume_A3"],
            )
        with np.load(branch_dir / "transition_progress.npz") as transition:
            assert transition["crystalline_profile"].shape == (3, 4)
            assert np.isfinite(transition["front_displacement_A"]).all()
        with np.load(branch_dir / "phase_rdf.npz") as rdf:
            assert rdf["phase_names"].tolist() == [
                "solid_bulk",
                "liquid_bulk",
                "interface",
            ]
            assert rdf["g_r"].shape == (3, 3, 50)
            assert np.isfinite(rdf["g_r"]).all()
        metadata = json.loads((branch_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["physics"]["method"] == "planar direct solid-liquid coexistence"
        assert metadata["rdf"]["frames"] == 3
        assert (branch_dir / "visualizations/transition_progress.png").is_file()
        assert (branch_dir / "visualizations/phase_rdf.png").is_file()
        assert (branch_dir / "visualizations/structure_slice.png").is_file()

    (result.output_root / "phase_rdf_overview.png").unlink()
    for branch_dir in result.branch_dirs:
        (branch_dir / "phase_rdf.npz").unlink()
        (branch_dir / "visualizations/phase_rdf.png").unlink()
    add_phase_rdf_to_transition_dataset(config, progress=lambda _message: None)
    assert (result.output_root / "phase_rdf_overview.png").is_file()
    for branch_dir in result.branch_dirs:
        assert (branch_dir / "phase_rdf.npz").is_file()
        assert (branch_dir / "visualizations/phase_rdf.png").is_file()
