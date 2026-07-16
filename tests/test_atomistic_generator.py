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

from src.data_utils.synthetic.atomistic import (
    generate_dataset,
    load_config,
    load_homogeneous_crystallization_config,
    load_transition_config,
)
from src.data_utils.synthetic.atomistic.artifacts import PHASE_NAMES, label_interface
from src.data_utils.synthetic.atomistic.generator import build_calculator, select_calculator
from src.data_utils.synthetic.atomistic.provenance import configured_mace_provenance
from src.data_utils.data_load import SyntheticPointCloudDataset
from src.data_utils.prepare_data import get_regular_samples


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)
TRANSITION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_transition_70304_mpa.yaml"
)
HOMOGENEOUS_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/homogeneous_16384_mpa.yaml"
)


def _qualified_report(raw: dict[str, object]) -> dict[str, object]:
    potential = raw["potential"]
    return {
        "schema_version": 1,
        "report_type": "al_crystallization_mlip_qualification",
        "model_name": potential["model_name"],
        "model_sha256": potential["sha256"],
        "head": potential["head"],
        "scientifically_qualified": True,
        "qualification_failures": [],
        "scope": {
            "chemical_symbol": "Al",
            "pressure_range_GPa": [-0.1, 0.1],
            "state_temperature_ranges_K": {
                "solid_bulk": [600.0, 1000.0],
                "liquid_bulk": [500.0, 1600.0],
                "interface": [500.0, 1600.0],
                "strained_solid": [600.0, 1000.0],
                "nucleus": [500.0, 700.0],
                "hcp": [300.0, 700.0],
                "bcc": [300.0, 700.0],
            },
            "maximum_timestep_fs": 1.0,
            "authorized_claims": {
                "phase_context_structure": True,
                "equilibrium_thermodynamics": True,
                "kinetics": False,
            },
            "calculator_settings": {
                "implementation_class": (
                    "src.data_utils.synthetic.atomistic.calculator."
                    "VerletSkinMACECalculator"
                ),
                "device": potential["device"],
                "default_dtype": potential["default_dtype"],
                "enable_cueq": potential["enable_cueq"],
                "neighbor_skin_A": potential["neighbor_skin_A"],
            },
        },
    }


def _write_qualified_report_chain(
    tmp_path: Path,
    raw: dict[str, object],
    *,
    report: dict[str, object] | None = None,
) -> Path:
    child = _qualified_report(raw) if report is None else report
    model_name = raw["potential"]["model_name"]
    benchmark_config_path = tmp_path / "benchmark_config.yaml"
    benchmark_config_path.write_text("fixture: qualification-chain\n", encoding="utf-8")
    dft_path = tmp_path / "dft_reference.extxyz"
    dft_path.write_text("fixture DFT evidence\n", encoding="utf-8")
    melting_path = tmp_path / "velocity_summary.json"
    melting_path.write_text('{"fixture": "melting evidence"}\n', encoding="utf-8")
    child_path = tmp_path / "qualification.json"
    parent_path = tmp_path / "benchmark_report.json"
    child_scope = child["scope"]
    benchmark_config = {
        "config_path": str(benchmark_config_path),
        "chemical_symbol": child_scope["chemical_symbol"],
        "qualification_scope": {
            "pressure_range_GPa": child_scope["pressure_range_GPa"],
            "state_temperature_ranges_K": child_scope[
                "state_temperature_ranges_K"
            ],
            "maximum_timestep_fs": child_scope["maximum_timestep_fs"],
            "authorized_claims": child_scope["authorized_claims"],
        },
        "fixture": "qualification-chain",
    }
    dft_evidence = {
        "path": str(dft_path),
        "sha256": hashlib.sha256(dft_path.read_bytes()).hexdigest(),
    }
    melting_evidence = {
        "scans": [
            {
                "path": str(melting_path),
                "sha256": hashlib.sha256(melting_path.read_bytes()).hexdigest(),
            }
        ]
    }
    model_result = {
        "identity": {
            "model_name": model_name,
            "sha256": raw["potential"]["sha256"],
            "head": raw["potential"]["head"],
        },
        "equation_of_state": {},
        "phase_energy_difference_from_fcc_meV_per_atom": {},
        "nve": {},
        "dft_reference_errors": {"fixture": True},
        "scientifically_qualified": True,
        "qualification_failures": [],
    }
    parent = {
        "schema_version": 1,
        "benchmark_config": benchmark_config,
        "models": {model_name: model_result},
        "dft_reference_evidence": dft_evidence,
        "melting_scans": {model_name: melting_evidence},
        "qualification_reports": {model_name: str(child_path)},
        "selection": {
            "qualified_models": [model_name],
            "automatic_replacement_allowed": False,
            "manual_comparison_required": True,
        },
    }
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    child.update(
        {
            "benchmark_report": str(parent_path),
            "benchmark_report_sha256": hashlib.sha256(
                parent_path.read_bytes()
            ).hexdigest(),
            "evidence": {
                "benchmark_config_canonical_sha256": hashlib.sha256(
                    json.dumps(
                        benchmark_config, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "benchmark_config_file_sha256": hashlib.sha256(
                    benchmark_config_path.read_bytes()
                ).hexdigest(),
                "dft_reference": dft_evidence,
                "melting_scans": melting_evidence,
            },
            "equation_of_state": model_result["equation_of_state"],
            "phase_energy_difference_from_fcc_meV_per_atom": model_result[
                "phase_energy_difference_from_fcc_meV_per_atom"
            ],
            "nve": model_result["nve"],
            "dft_reference_errors": model_result["dft_reference_errors"],
            "melting_scan": melting_evidence,
        }
    )
    child_path.write_text(json.dumps(child), encoding="utf-8")
    return child_path


def test_production_config_has_no_density_control() -> None:
    config = load_config(PRODUCTION_CONFIG)
    assert config.system.chemical_symbol == "Al"
    assert config.dynamics.pressure_GPa == 0.0
    assert config.validation.reference_density_cache is None
    assert config.random_seeds == (12345,)
    assert config.potential.model_name == "mace-mpa-0-medium"
    assert config.potential.family == "MACE-MPA-0"
    assert config.potential.head == "default"
    assert config.potential.usage_mode == "exploratory"
    assert config.potential.scientifically_qualified is False
    assert config.potential.enable_cueq is True
    assert config.potential.neighbor_skin_A == 0.3
    assert config.system.repetitions == (26, 26, 26)
    assert config.system.liquid_slab_fraction == 0.5


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


def test_quantitative_usage_requires_matching_qualified_report(tmp_path: Path) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["potential"]["usage_mode"] = "quantitative"
    missing_report_path = tmp_path / "missing_report.yaml"
    missing_report_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientifically_qualified=true"):
        load_config(missing_report_path)

    report_path = _write_qualified_report_chain(tmp_path, raw)
    raw["potential"]["validation_report"] = str(report_path)
    qualified_path = tmp_path / "qualified.yaml"
    qualified_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    qualified = load_config(qualified_path)
    assert qualified.potential.usage_mode == "quantitative"
    assert qualified.potential.scientifically_qualified is True
    assert qualified.potential.qualification is not None
    assert qualified.potential.qualification.chemical_symbol == "Al"

    ConfiguredCalculatorStub = type(
        "VerletSkinMACECalculator",
        (),
        {"available_heads": ["default"], "head": "default"},
    )
    ConfiguredCalculatorStub.__module__ = (
        "src.data_utils.synthetic.atomistic.calculator"
    )

    provenance = configured_mace_provenance(
        qualified, ConfiguredCalculatorStub()
    )
    assert provenance.qualification_scope is not None
    assert provenance.qualification_scope["pressure_range_GPa"] == [-0.1, 0.1]
    assert provenance.qualification_scope["authorized_claims"] == {
        "phase_context_structure": True,
        "equilibrium_thermodynamics": True,
        "kinetics": False,
    }
    calculator_settings = provenance.qualification_scope["calculator_settings"]
    assert isinstance(calculator_settings, dict)
    assert calculator_settings["default_dtype"] == "float32"

    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after configuration load"):
        configured_mace_provenance(qualified, ConfiguredCalculatorStub())
    with pytest.raises(RuntimeError, match="cannot inherit that qualification"):
        select_calculator(
            qualified,
            calculator=EMT(),
            injected_calculator_identity="ase-emt:test-only",
        )


@pytest.mark.parametrize(
    ("scope_change", "expected_error"),
    [
        (("chemical_symbol", "Ni"), "covers chemical_symbol='Ni'"),
        (("pressure_range_GPa", [0.1, 1.0]), "outside qualification range"),
        (("maximum_timestep_fs", 0.5), "exceeds qualified maximum"),
        (("liquid_maximum_temperature_K", 1500.0), "state='liquid_bulk'.*1600"),
        (("default_dtype", "float64"), "calculator settings do not match"),
        (("model_sha256", "0" * 64), "validation report identifies model_sha256"),
        (("head", "unrelated_head"), "validation report identifies model_sha256"),
        (("report_type", "unrelated_schema_1_report"), "expected report_type"),
    ],
)
def test_quantitative_usage_rejects_out_of_scope_conditions(
    tmp_path: Path,
    scope_change: tuple[str, object],
    expected_error: str,
) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["potential"]["usage_mode"] = "quantitative"
    report = _qualified_report(raw)
    key, value = scope_change
    scope = report["scope"]
    if key == "liquid_maximum_temperature_K":
        scope["state_temperature_ranges_K"]["liquid_bulk"][1] = value
    elif key == "default_dtype":
        scope["calculator_settings"][key] = value
    elif key in {"model_sha256", "head", "report_type"}:
        report[key] = value
    else:
        scope[key] = value
    report_path = _write_qualified_report_chain(tmp_path, raw, report=report)
    raw["potential"]["validation_report"] = str(report_path)
    config_path = tmp_path / f"{key}.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected_error):
        load_config(config_path)


def test_derived_protocols_check_their_own_quantitative_temperature_scope(
    tmp_path: Path,
) -> None:
    generator_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    generator_raw["potential"]["usage_mode"] = "quantitative"
    report = _qualified_report(generator_raw)
    report["scope"]["authorized_claims"]["kinetics"] = True
    report["scope"]["state_temperature_ranges_K"]["solid_bulk"] = [600.0, 700.0]
    report["scope"]["state_temperature_ranges_K"]["interface"] = [600.0, 1600.0]
    report_path = _write_qualified_report_chain(
        tmp_path, generator_raw, report=report
    )
    generator_raw["potential"]["validation_report"] = str(report_path)
    generator_path = tmp_path / "generator.yaml"
    generator_path.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")

    transition_raw = yaml.safe_load(TRANSITION_CONFIG.read_text(encoding="utf-8"))
    transition_raw["source_generator_config"] = str(generator_path)
    transition_path = tmp_path / "transition.yaml"
    transition_path.write_text(yaml.safe_dump(transition_raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="state='solid_bulk'.*800"):
        load_transition_config(transition_path)

    homogeneous_raw = yaml.safe_load(HOMOGENEOUS_CONFIG.read_text(encoding="utf-8"))
    homogeneous_raw["source_generator_config"] = str(generator_path)
    homogeneous_path = tmp_path / "homogeneous.yaml"
    homogeneous_path.write_text(yaml.safe_dump(homogeneous_raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="state='interface'.*500"):
        load_homogeneous_crystallization_config(homogeneous_path)


def test_quantitative_workflows_require_claim_specific_authorization(
    tmp_path: Path,
) -> None:
    generator_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    generator_raw["potential"]["usage_mode"] = "quantitative"

    kinetics_dir = tmp_path / "kinetics"
    kinetics_dir.mkdir()
    kinetics_report = _qualified_report(generator_raw)
    kinetics_report_path = _write_qualified_report_chain(
        kinetics_dir, generator_raw, report=kinetics_report
    )
    generator_raw["potential"]["validation_report"] = str(kinetics_report_path)
    kinetics_generator_path = kinetics_dir / "generator.yaml"
    kinetics_generator_path.write_text(yaml.safe_dump(generator_raw), encoding="utf-8")
    homogeneous_raw = yaml.safe_load(HOMOGENEOUS_CONFIG.read_text(encoding="utf-8"))
    homogeneous_raw["source_generator_config"] = str(kinetics_generator_path)
    homogeneous_path = kinetics_dir / "homogeneous.yaml"
    homogeneous_path.write_text(yaml.safe_dump(homogeneous_raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not authorize required claim='kinetics'"):
        load_homogeneous_crystallization_config(homogeneous_path)

    equilibrium_dir = tmp_path / "equilibrium"
    equilibrium_dir.mkdir()
    equilibrium_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    equilibrium_raw["potential"]["usage_mode"] = "quantitative"
    equilibrium_report = _qualified_report(equilibrium_raw)
    equilibrium_report["scope"]["authorized_claims"][
        "equilibrium_thermodynamics"
    ] = False
    equilibrium_report_path = _write_qualified_report_chain(
        equilibrium_dir, equilibrium_raw, report=equilibrium_report
    )
    equilibrium_raw["potential"]["validation_report"] = str(
        equilibrium_report_path
    )
    equilibrium_generator_path = equilibrium_dir / "generator.yaml"
    equilibrium_generator_path.write_text(
        yaml.safe_dump(equilibrium_raw), encoding="utf-8"
    )
    transition_raw = yaml.safe_load(TRANSITION_CONFIG.read_text(encoding="utf-8"))
    transition_raw["source_generator_config"] = str(equilibrium_generator_path)
    transition_path = equilibrium_dir / "transition.yaml"
    transition_path.write_text(yaml.safe_dump(transition_raw), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="does not authorize required claim='equilibrium_thermodynamics'",
    ):
        load_transition_config(transition_path)

    structure_dir = tmp_path / "structure"
    structure_dir.mkdir()
    structure_raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    structure_raw["potential"]["usage_mode"] = "quantitative"
    structure_report = _qualified_report(structure_raw)
    structure_report["scope"]["authorized_claims"][
        "phase_context_structure"
    ] = False
    structure_report_path = _write_qualified_report_chain(
        structure_dir, structure_raw, report=structure_report
    )
    structure_raw["potential"]["validation_report"] = str(structure_report_path)
    structure_path = structure_dir / "generator.yaml"
    structure_path.write_text(yaml.safe_dump(structure_raw), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="does not authorize required claim='phase_context_structure'",
    ):
        load_config(structure_path)


def test_qualification_claims_cannot_expand_beyond_bound_benchmark(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["potential"]["usage_mode"] = "quantitative"
    report_path = _write_qualified_report_chain(tmp_path, raw)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scope"]["authorized_claims"]["kinetics"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    raw["potential"]["validation_report"] = str(report_path)
    config_path = tmp_path / "expanded-claim.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="scope expands or differs"):
        load_config(config_path)


def test_mace_head_fallback_is_rejected(tmp_path: Path, monkeypatch) -> None:
    config = load_config(PRODUCTION_CONFIG)
    dummy_model = tmp_path / "multihead.model"
    dummy_model.write_bytes(b"multi-head test model")
    config = replace(
        config,
        potential=replace(
            config.potential,
            model_path=dummy_model,
            sha256=hashlib.sha256(dummy_model.read_bytes()).hexdigest(),
            head="missing_head",
            device="cpu",
            enable_cueq=False,
        ),
    )

    class FallbackCalculator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.available_heads = ["omat_pbe", "matpes_r2scan"]
            self.head = self.available_heads[-1]

    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.calculator.VerletSkinMACECalculator",
        FallbackCalculator,
    )
    with pytest.raises(RuntimeError, match="does not exist.*available_heads"):
        build_calculator(config)


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


def test_regular_sampling_starts_at_single_radius_padding() -> None:
    axis = np.arange(21, dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    samples = get_regular_samples(
        points,
        size=2.0,
        overlap_fraction=0.0,
        return_coords=True,
        n_points=8,
        max_samples=1,
        drop_edge_samples=False,
    )
    assert len(samples) == 1
    _, center = samples[0]
    assert np.array_equal(center, np.array([2.0, 2.0, 2.0]))


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

    with pytest.raises(ValueError, match="requires injected_calculator_identity"):
        generate_dataset(config, calculator=EMT(), progress=lambda _message: None)

    result = generate_dataset(
        config,
        calculator=EMT(),
        injected_calculator_identity="ase-emt-default:test-only",
        progress=lambda _message: None,
    )

    assert [path.name for path in result.environment_dirs] == [
        "replica_000_bulk_solid",
        "replica_000_bulk_liquid",
        "replica_000_solid_liquid_interface",
    ]
    manifest = json.loads((result.output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment_dirs"] == [
        "replica_000_bulk_solid",
        "replica_000_bulk_liquid",
        "replica_000_solid_liquid_interface",
    ]
    assert manifest["potential_sha256"] is None
    assert manifest["execution_provenance"]["calculator"]["source"] == (
        "injected_calculator"
    )
    assert manifest["execution_provenance"]["calculator"]["identity"] == (
        "ase-emt-default:test-only"
    )
    assert manifest["execution_provenance"]["calculator"]["usage_mode"] == (
        "injected_unqualified"
    )
    assert (result.output_root / "benchmark_overview.png").is_file()
    for environment_dir in result.environment_dirs:
        atoms = np.load(environment_dir / "atoms.npy")
        atom_table = np.load(environment_dir / "atoms_full.npy")
        with np.load(environment_dir / "trajectory.npz") as trajectory:
            trajectory_steps = trajectory["step"]
            trajectory_positions = trajectory["positions_A"]
        metadata = json.loads((environment_dir / "metadata.json").read_text(encoding="utf-8"))
        assert atoms.shape == (144, 3)
        assert np.array_equal(atoms, atom_table["position"])
        assert metadata["global"]["N_final"] == 144
        assert metadata["global"]["random_seed"] in config.random_seeds
        assert trajectory_steps.tolist() == [0, 1, 2]
        assert trajectory_positions.shape == (3, 144, 3)
        assert np.allclose(trajectory_positions[-1], atoms)
        assert "rho_target" not in metadata["global"]
        assert metadata["physics"]["label_policy"].startswith("Labels encode preparation")
        assert metadata["physics"]["calculator"]["source"] == "injected_calculator"
        assert (environment_dir / "visualizations/structure_slice.png").is_file()
        assert (environment_dir / "visualizations/thermodynamic_trace.png").is_file()
        assert (environment_dir / "visualizations/structure_diagnostics.png").is_file()

    interface_metadata = json.loads(
        (result.output_root / "replica_000_solid_liquid_interface/metadata.json").read_text(
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

    with pytest.raises(FileExistsError, match="before calculator construction or MD"):
        generate_dataset(config, progress=lambda _message: None)
