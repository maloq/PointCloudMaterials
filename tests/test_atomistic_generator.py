from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic import generate_dataset, load_config
from src.data_utils.synthetic.atomistic.artifacts import PHASE_NAMES, label_interface
from src.data_utils.data_load import SyntheticPointCloudDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/data/data_synth_polycrystalline_balanced_geometries.yaml"
)


def test_production_config_has_no_density_control() -> None:
    config = load_config(PRODUCTION_CONFIG)
    assert config.system.chemical_symbol == "Al"
    assert config.dynamics.pressure_GPa == 0.0
    assert config.validation.reference_density_cache is not None


def test_density_control_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["system"]["density_target"] = 0.0849
    raw["potential"]["model_path"] = str(
        REPOSITORY_ROOT / "datasets/potentials/mace-mpa-0-medium.model"
    )
    config_path = tmp_path / "invalid_density.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="NPT simulation result"):
        load_config(config_path)


def test_unknown_physics_control_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["dynamics"]["helpful_magic"] = True
    config_path = tmp_path / "unknown_control.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(KeyError, match="unsupported keys in dynamics"):
        load_config(config_path)


def test_interface_labels_cover_real_slabs() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))
    labels = label_interface(
        atoms,
        slab_bounds_fractional=(0.25, 0.75),
        interface_half_width_A=3.0,
    )
    assert set(np.unique(labels.phase_names)) == set(PHASE_NAMES)
    assert sum(len(indices) for indices in labels.grain_atom_indices.values()) == len(atoms)
    assert labels.intermediate_atom_indices.size > 0
    assert np.all(labels.phase_names[labels.intermediate_atom_indices] == "interface")


def test_small_force_driven_generation_round_trip(tmp_path: Path) -> None:
    config = load_config(PRODUCTION_CONFIG)
    dummy_model = tmp_path / "test-calculator.model"
    dummy_model.write_bytes(b"EMT test calculator; production uses the configured MACE model")
    config = replace(
        config,
        potential=replace(
            config.potential,
            model_path=dummy_model,
            sha256=hashlib.sha256(dummy_model.read_bytes()).hexdigest(),
            device="cpu",
        ),
        system=replace(config.system, repetitions=(3, 3, 4), interface_half_width_A=2.0),
        dynamics=replace(
            config.dynamics,
            solid_equilibration_steps=2,
            melt_steps=2,
            quench_steps=2,
            quench_stages=1,
            target_equilibration_steps=2,
            interface_evolution_steps=2,
            sample_interval=1,
            barostat_time_fs=200.0,
        ),
        validation=replace(
            config.validation,
            maximum_force_eV_per_A=100.0,
            maximum_pressure_error_GPa=100.0,
            maximum_temperature_error_K=1000.0,
            minimum_pair_distance_A=0.5,
            reference_density_cache=None,
            maximum_relative_density_error=None,
            minimum_solid_fcc_fraction=0.0,
            maximum_liquid_crystalline_fraction=1.0,
            minimum_interface_crystalline_fraction=0.0,
            maximum_interface_crystalline_fraction=1.0,
        ),
        output=replace(
            config.output,
            root_dir=tmp_path / "generated",
            overwrite=False,
            save_extxyz=False,
        ),
    )

    result = generate_dataset(config, calculator=EMT(), progress=lambda _message: None)

    assert [path.name for path in result.environment_dirs] == [
        "bulk_solid",
        "bulk_liquid",
        "solid_liquid_interface",
    ]
    manifest = json.loads((result.output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment_dirs"] == [
        "bulk_solid",
        "bulk_liquid",
        "solid_liquid_interface",
    ]
    for environment_dir in result.environment_dirs:
        atoms = np.load(environment_dir / "atoms.npy")
        atom_table = np.load(environment_dir / "atoms_full.npy")
        metadata = json.loads((environment_dir / "metadata.json").read_text(encoding="utf-8"))
        assert atoms.shape == (144, 3)
        assert np.array_equal(atoms, atom_table["position"])
        assert metadata["global"]["N_final"] == 144
        assert "rho_target" not in metadata["global"]
        assert metadata["physics"]["label_policy"].startswith("Labels encode preparation")

    interface_metadata = json.loads(
        (result.output_root / "solid_liquid_interface/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(interface_metadata["intermediate_regions"]) == 1
    assert interface_metadata["phase_statistics"]["interface"]["n_atoms"] > 0

    dataset = SyntheticPointCloudDataset(
        env_dirs=result.environment_dirs,
        radius=4.0,
        sample_type="random",
        overlap_fraction=0.0,
        n_samples=2,
        num_points=16,
        drop_edge_samples=False,
    )
    assert len(dataset) == 6
    assert set(dataset.class_names.values()).issubset(set(PHASE_NAMES))
