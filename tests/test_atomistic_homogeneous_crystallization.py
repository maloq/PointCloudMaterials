from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic import (
    generate_homogeneous_crystallization_dataset,
    load_homogeneous_crystallization_config,
)
from src.data_utils.synthetic.atomistic.artifacts import (
    build_atom_table,
    label_bulk,
)
from src.data_utils.synthetic.atomistic.homogeneous_analysis import (
    analyze_homogeneous_crystallization,
)
from src.data_utils.synthetic.atomistic.simulation import ThermodynamicTrace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_CONFIG = REPOSITORY_ROOT / "configs/data/atomistic_al_phase_context.yaml"
HOMOGENEOUS_CONFIG = (
    REPOSITORY_ROOT / "configs/data/atomistic_al_homogeneous_crystallization.yaml"
)


def test_production_homogeneous_config_is_unseeded_and_twice_as_long() -> None:
    config = load_homogeneous_crystallization_config(HOMOGENEOUS_CONFIG)
    assert config.source_environment == "replica_000_bulk_liquid"
    assert config.source_frame_step == 3000
    assert config.generator.system.repetitions == (26, 26, 26)
    assert config.temperature_K == 500.0
    assert config.steps == 10000
    assert config.sample_interval == 100
    assert config.steps // config.sample_interval + 1 == 101
    assert config.analysis.nucleus_size_threshold_atoms == 100


def test_connected_crystalline_cluster_analysis_resolves_fcc() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
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
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol="Al",
        timestep_fs=1.0,
        crystalline_cluster_cutoff_A=3.5,
        nucleus_size_threshold_atoms=100,
        rdf_cutoff_A=5.0,
        rdf_bins=50,
        progress=lambda _message: None,
    )

    assert analysis.crystalline_fraction.tolist() == [1.0]
    assert analysis.largest_crystalline_cluster_atoms.tolist() == [atom_count]
    assert analysis.nucleation_observed is True
    assert analysis.nucleation_step == 0
    assert analysis.rdf_g_r.shape == (1, 50)


def test_small_homogeneous_crystallization_round_trip(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_dir = source_root / "replica_000_bulk_liquid"
    source_dir.mkdir(parents=True)

    generator_raw = yaml.safe_load(GENERATOR_CONFIG.read_text(encoding="utf-8"))
    generator_raw["system"]["repetitions"] = [3, 3, 4]
    generator_raw["validation"]["maximum_force_eV_per_A"] = 100.0
    generator_raw["validation"]["maximum_pressure_error_GPa"] = 100.0
    generator_raw["validation"]["maximum_temperature_error_K"] = 1000.0
    generator_raw["validation"]["minimum_pair_distance_A"] = 0.5
    generator_raw["validation"]["maximum_liquid_crystalline_fraction"] = 1.0
    generator_raw["output"]["root_dir"] = str(source_root)
    generator_path = tmp_path / "generator.yaml"
    generator_path.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 4))
    atom_count = len(atoms)
    source_table = build_atom_table(
        atoms,
        label_bulk(atom_count, "liquid_bulk", grain_id=0),
    )
    np.save(source_dir / "atoms_full.npy", source_table)
    with (source_dir / "trajectory.npz").open("wb") as handle:
        np.savez(
            handle,
            step=np.array([3000], dtype=np.int64),
            positions_A=np.asarray([atoms.positions], dtype=np.float32),
            cell_vectors_A=np.asarray([atoms.cell.array], dtype=np.float64),
            temperature_K=np.array([300.0]),
            pressure_GPa=np.array([0.0]),
            volume_A3=np.array([atoms.get_volume()]),
            potential_energy_eV_per_atom=np.array([0.0]),
        )
    (source_dir / "metadata.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    "ptm_structure_fractions": {
                        "other": 0.0,
                        "fcc": 1.0,
                        "hcp": 0.0,
                        "bcc": 0.0,
                        "ico": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (source_root / "manifest.json").write_text(
        json.dumps({"potential_sha256": generator_raw["potential"]["sha256"]}),
        encoding="utf-8",
    )

    output_root = tmp_path / "homogeneous"
    homogeneous_raw = {
        "dataset_name": "test_homogeneous_crystallization",
        "source_generator_config": str(generator_path),
        "source_dataset": str(source_root),
        "source_environment": "replica_000_bulk_liquid",
        "source_frame_step": 3000,
        "random_seed": 35791,
        "temperature_K": 300.0,
        "steps": 2,
        "sample_interval": 1,
        "analysis": {
            "crystalline_cluster_cutoff_A": 3.5,
            "nucleus_size_threshold_atoms": atom_count + 1,
            "rdf_cutoff_A": 5.0,
            "rdf_bins": 50,
        },
        "output": {
            "root_dir": str(output_root),
            "overwrite": False,
            "save_extxyz": True,
            "create_visualizations": True,
        },
    }
    homogeneous_path = tmp_path / "homogeneous.yaml"
    homogeneous_path.write_text(
        yaml.safe_dump(homogeneous_raw), encoding="utf-8"
    )
    result = generate_homogeneous_crystallization_dataset(
        load_homogeneous_crystallization_config(homogeneous_path),
        calculator=EMT(),
        progress=lambda _message: None,
    )

    assert result.run_dir.name == "homogeneous_crystallization"
    assert (result.output_root / "crystallization_overview.png").is_file()
    assert np.load(result.run_dir / "atoms.npy").shape == (atom_count, 3)
    with np.load(result.run_dir / "trajectory.npz") as trajectory:
        assert trajectory["step"].tolist() == [0, 1, 2]
        assert trajectory["positions_A"].shape == (3, atom_count, 3)
    with np.load(result.run_dir / "crystallization_progress.npz") as progress:
        assert progress["structure_fractions"].shape == (3, 5)
        assert progress["largest_crystalline_cluster_atoms"].shape == (3,)
    with np.load(result.run_dir / "total_rdf.npz") as rdf:
        assert rdf["g_r"].shape == (3, 50)
        assert np.isfinite(rdf["g_r"]).all()
    metadata = json.loads(
        (result.run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["physics"]["seeded"] is False
    assert metadata["physics"]["duration_ps"] == 0.002
    assert metadata["nucleation"]["observed"] is False
    assert (result.run_dir / "visualizations/crystallization_progress.png").is_file()
    assert (result.run_dir / "visualizations/total_rdf.png").is_file()
    assert (result.run_dir / "visualizations/structure_slice.png").is_file()
