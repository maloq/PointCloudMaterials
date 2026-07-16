from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic import (
    generate_homogeneous_crystallization_dataset,
    load_config,
    load_homogeneous_crystallization_config,
)
from src.data_utils.synthetic.atomistic.artifacts import (
    build_atom_table,
    label_bulk,
)
from src.data_utils.synthetic.atomistic.homogeneous_analysis import (
    ReplicaObservation,
    analyze_homogeneous_crystallization,
    analyze_replica_survival,
    first_persistent_threshold_run,
)
from src.data_utils.synthetic.atomistic.simulation import ThermodynamicTrace
from src.data_utils.synthetic.atomistic.provenance import _producer_code_provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)
HOMOGENEOUS_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/homogeneous_16384_mpa.yaml"
)


def test_production_homogeneous_config_has_independent_replicas() -> None:
    config = load_homogeneous_crystallization_config(HOMOGENEOUS_CONFIG)
    assert config.source_environment == "replica_000_bulk_liquid"
    assert config.source_frame_step == 3000
    assert config.generator.system.repetitions == (16, 16, 16)
    assert config.temperature_K == 500.0
    assert len(config.random_seeds) == 10
    assert len(set(config.random_seeds)) == 10
    assert config.equilibration_steps == 5000
    assert config.steps == 200000
    assert config.sample_interval == 1000
    assert config.steps // config.sample_interval + 1 == 201
    assert config.analysis.ptm_rmsd_cutoff == 0.1
    assert config.analysis.nucleus_size_threshold_atoms == 100
    assert config.analysis.threshold_persistence_frames == 3


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
        ptm_rmsd_cutoff=0.1,
        crystalline_cluster_cutoff_A=3.5,
        nucleus_size_threshold_atoms=100,
        threshold_persistence_frames=1,
        rdf_cutoff_A=5.0,
        rdf_bins=50,
        progress=lambda _message: None,
    )

    assert analysis.crystalline_fraction.tolist() == [1.0]
    assert analysis.largest_crystalline_cluster_atoms.tolist() == [atom_count]
    assert analysis.nucleation_observed is True
    assert analysis.nucleation_step == 0
    assert analysis.rdf_g_r.shape == (1, 50)


def test_persistent_cluster_threshold_rejects_one_frame_spike() -> None:
    values = np.array([4, 101, 8, 100, 105, 109, 3], dtype=np.int64)
    assert first_persistent_threshold_run(
        values, threshold=100, persistence_frames=3
    ) == (3, 5)


def test_replica_survival_records_events_and_right_censoring() -> None:
    survival = analyze_replica_survival(
        (
            ReplicaObservation("replica_000", 11, True, 4.0),
            ReplicaObservation("replica_001", 12, False, 10.0),
            ReplicaObservation("replica_002", 13, True, 8.0),
        )
    )
    assert survival.time_ps.tolist() == [4.0, 8.0, 10.0]
    assert survival.replicas_at_risk.tolist() == [3, 2, 1]
    assert survival.events.tolist() == [1, 1, 0]
    assert survival.censored.tolist() == [0, 0, 1]
    assert np.allclose(survival.survival_probability, [2 / 3, 1 / 3, 1 / 3])
    assert survival.to_dict()["rate_estimate"] is None


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
    source_generator_config = load_config(generator_path)
    potential = source_generator_config.potential
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": source_generator_config.to_dict(),
                "potential_sha256": potential.sha256,
                "execution_provenance": {
                    "calculator": {
                        "source": "configured_mace_model",
                        "identity": (
                            f"{potential.model_name}:{potential.sha256}:{potential.head}"
                        ),
                        "implementation_class": (
                            "src.data_utils.synthetic.atomistic.calculator."
                            "VerletSkinMACECalculator"
                        ),
                        "model_name": potential.model_name,
                        "family": potential.family,
                        "model_path": str(potential.model_path),
                        "model_sha256": potential.sha256,
                        "head": potential.head,
                        "available_heads": [potential.head],
                        "source_url": potential.source_url,
                        "license_identifier": potential.license_identifier,
                        "usage_mode": potential.usage_mode,
                        "validation_report_path": None,
                        "validation_report_sha256": None,
                        "validation_report_type": None,
                        "scientifically_qualified": False,
                        "qualification_scope": None,
                        "settings": {
                            "device": potential.device,
                            "default_dtype": potential.default_dtype,
                            "enable_cueq": potential.enable_cueq,
                            "neighbor_skin_A": potential.neighbor_skin_A,
                        },
                    },
                    "runtime": {
                        "python": "test",
                        "numpy": "test",
                        "ase": "test",
                        "torch": "test",
                        "platform": "test",
                        "machine": "test",
                        "mace_torch": "test",
                        "torch_cuda": "test",
                        "cudnn": "test",
                        "cuequivariance": "test",
                        "cuequivariance_torch": "test",
                        "cuequivariance_ops_torch_cu12": "test",
                        "cuda_device_index": 0,
                        "cuda_device_name": "test",
                    },
                    "producer_code": _producer_code_provenance(),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "homogeneous"
    homogeneous_raw = {
        "dataset_name": "test_homogeneous_crystallization",
        "source_generator_config": str(generator_path),
        "source_dataset": str(source_root),
        "source_environment": "replica_000_bulk_liquid",
        "source_frame_step": 3000,
        "random_seeds": [35791],
        "temperature_K": 300.0,
        "equilibration_steps": 2,
        "steps": 2,
        "sample_interval": 1,
        "analysis": {
            "ptm_rmsd_cutoff": 0.1,
            "crystalline_cluster_cutoff_A": 3.5,
            "nucleus_size_threshold_atoms": atom_count + 1,
            "threshold_persistence_frames": 2,
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
        injected_calculator_identity="test-only ASE EMT",
        progress=lambda _message: None,
    )

    assert len(result.replicas) == 1
    replica = result.replicas[0]
    assert replica.run_dir.name == "replica_000"
    assert (result.output_root / "replica_000_overview.png").is_file()
    assert np.load(replica.run_dir / "atoms.npy").shape == (atom_count, 3)
    with np.load(replica.run_dir / "equilibration_trajectory.npz") as trajectory:
        assert trajectory["step"].tolist() == [0, 1, 2]
    with np.load(replica.run_dir / "trajectory.npz") as trajectory:
        assert trajectory["step"].tolist() == [0, 1, 2]
        assert trajectory["positions_A"].shape == (3, atom_count, 3)
    with np.load(replica.run_dir / "crystallization_progress.npz") as progress:
        assert progress["structure_fractions"].shape == (3, 5)
        assert progress["largest_crystalline_cluster_atoms"].shape == (3,)
    with np.load(replica.run_dir / "total_rdf.npz") as rdf:
        assert rdf["g_r"].shape == (3, 50)
        assert np.isfinite(rdf["g_r"]).all()
    metadata = json.loads(
        (replica.run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["physics"]["seeded"] is False
    assert metadata["physics"]["measurement_duration_ps"] == 0.002
    assert metadata["physics"]["target_temperature_equilibration"]["duration_ps"] == 0.002
    assert metadata["threshold_event"]["observed"] is False
    assert metadata["threshold_event"]["ptm_normalized_rmsd_cutoff"] == 0.1
    assert result.survival.to_dict()["right_censored_count"] == 1
    assert (result.output_root / "survival_summary.json").is_file()
    assert (replica.run_dir / "visualizations/crystallization_progress.png").is_file()
    assert (replica.run_dir / "visualizations/total_rdf.png").is_file()
    assert (replica.run_dir / "visualizations/structure_slice.png").is_file()

    with pytest.raises(FileExistsError, match="before loading the source"):
        generate_homogeneous_crystallization_dataset(
            load_homogeneous_crystallization_config(homogeneous_path),
            progress=lambda _message: None,
        )
