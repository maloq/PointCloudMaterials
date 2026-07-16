from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT

from src.data_utils.synthetic.atomistic import (
    add_phase_rdf_to_transition_dataset,
    generate_transition_dataset,
    load_transition_config,
)
from src.data_utils.synthetic.atomistic.simulation import ThermodynamicTrace
from src.data_utils.synthetic.atomistic.transition_analysis import (
    analyze_phase_rdf,
    analyze_transition,
)
from src.data_utils.synthetic.atomistic.transition_config import TransitionBranchConfig
from src.data_utils.synthetic.atomistic.transition_generator import _resolve_zero_velocity
from src.data_utils.synthetic.atomistic.provenance import _producer_code_provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)
TRANSITION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_transition_70304_mpa.yaml"
)
MH1_TRANSITION_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/phase_transition_70304_mh1.yaml"
)


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


def test_spatial_front_fit_tracks_two_interfaces_not_global_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_bins = 20
    atoms_per_bin = 20
    scaled_z = np.repeat(
        (np.arange(profile_bins, dtype=np.float64) + 0.5) / profile_bins,
        atoms_per_bin,
    )
    atom_count = len(scaled_z)
    positions_A = np.column_stack(
        (
            np.arange(atom_count, dtype=np.float64) % 10,
            (np.arange(atom_count, dtype=np.float64) // 10) % 10,
            scaled_z * 20.0,
        )
    )
    step = np.arange(0, 50, 10, dtype=np.int64)
    trace = ThermodynamicTrace(
        step=step,
        temperature_K=np.full(len(step), 300.0),
        pressure_GPa=np.zeros(len(step)),
        volume_A3=np.full(len(step), 2000.0),
        potential_energy_eV_per_atom=np.zeros(len(step)),
        positions_A=np.repeat(positions_A[None, :, :], len(step), axis=0),
        cell_vectors_A=np.repeat(np.diag([10.0, 10.0, 20.0])[None, :, :], len(step), axis=0),
    )
    equilibration_trace = ThermodynamicTrace(
        step=np.array([0, 10, 20], dtype=np.int64),
        temperature_K=np.full(3, 300.0),
        pressure_GPa=np.zeros(3),
        volume_A3=np.full(3, 2000.0),
        potential_energy_eV_per_atom=np.zeros(3),
        positions_A=np.repeat(positions_A[None, :, :], 3, axis=0),
        cell_vectors_A=np.repeat(np.diag([10.0, 10.0, 20.0])[None, :, :], 3, axis=0),
    )
    lower_positions = (0.25, 0.30, 0.35, 0.40, 0.45)
    upper_positions = (0.75, 0.70, 0.65, 0.60, 0.55)
    frame_index = 0

    def moving_interface_structure_types(frame_atoms, _rmsd_cutoff):
        nonlocal frame_index
        frame_scaled_z = frame_atoms.get_scaled_positions(wrap=True)[:, 2]
        crystalline = (frame_scaled_z < lower_positions[frame_index]) | (
            frame_scaled_z >= upper_positions[frame_index]
        )
        frame_index += 1
        return crystalline.astype(np.int32)

    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.transition_analysis._ptm_structure_types",
        moving_interface_structure_types,
    )
    branch = TransitionBranchConfig(
        name="spatial_growth_test",
        expected_direction="growth",
        temperature_K=300.0,
        equilibration_steps=20,
        production_steps=40,
        steady_state_start_step=10,
        steady_state_end_step=40,
        minimum_crystalline_fraction_change=0.1,
    )
    analysis = analyze_transition(
        trace,
        equilibration_trace=equilibration_trace,
        chemical_symbol="Al",
        timestep_fs=1.0,
        slab_bounds_fractional=(0.25, 0.75),
        profile_bins=profile_bins,
        profile_smoothing_bins=1,
        ptm_rmsd_cutoff=0.1,
        minimum_profile_contrast=0.2,
        minimum_velocity_fit_r_squared=0.99,
        target_pressure_GPa=0.0,
        maximum_temperature_error_K=1.0,
        maximum_pressure_error_GPa=0.1,
        branch=branch,
        progress=lambda _message: None,
    )

    np.testing.assert_allclose(
        analysis.interface_positions_fractional,
        np.column_stack((lower_positions, upper_positions)),
    )
    np.testing.assert_allclose(
        analysis.signed_interface_advance_A,
        np.repeat(np.arange(5, dtype=np.float64)[:, None], 2, axis=1),
    )
    assert analysis.fitted_interface_velocity_m_per_s == pytest.approx(10000.0)
    assert analysis.velocity_fit_r_squared == pytest.approx(1.0)

    frame_index = 0
    with pytest.raises(RuntimeError, match="profile contrast falls"):
        analyze_transition(
            trace,
            equilibration_trace=equilibration_trace,
            chemical_symbol="Al",
            timestep_fs=1.0,
            slab_bounds_fractional=(0.25, 0.75),
            profile_bins=profile_bins,
            profile_smoothing_bins=1,
            ptm_rmsd_cutoff=0.1,
            minimum_profile_contrast=0.5,
            minimum_velocity_fit_r_squared=0.99,
            target_pressure_GPa=0.0,
            maximum_temperature_error_K=1.0,
            maximum_pressure_error_GPa=0.1,
            branch=branch,
            progress=lambda _message: None,
        )


def test_production_transition_config_is_direct_coexistence() -> None:
    config = load_transition_config(TRANSITION_CONFIG)
    assert config.generator.system.repetitions == (26, 26, 26)
    assert config.generator.system.liquid_slab_fraction == 0.5
    assert config.source_frame_step == 1000
    assert config.random_seeds == (24680, 24681, 24682, 24683)
    assert config.temperature_runs[0].expected_direction == "growth"
    assert config.temperature_runs[0].temperature_K == 650.0
    assert config.temperature_runs[-1].expected_direction == "melting"
    assert config.temperature_runs[-1].temperature_K == 1000.0
    assert [branch.temperature_K for branch in config.temperature_runs] == [
        650.0,
        800.0,
        850.0,
        900.0,
        950.0,
        1000.0,
    ]
    assert config.sample_interval == 200
    assert config.analysis.rdf_cutoff_A == 8.0
    assert config.analysis.rdf_bins == 160
    assert config.analysis.ptm_rmsd_cutoff == 0.1
    assert all(
        branch.production_steps // config.sample_interval + 1 == 101
        for branch in config.temperature_runs
    )
    assert all(branch.equilibration_steps == 5000 for branch in config.temperature_runs)
    assert all(branch.production_steps == 20000 for branch in config.temperature_runs)
    assert all(
        (branch.steady_state_start_step, branch.steady_state_end_step)
        == (5000, 20000)
        for branch in config.temperature_runs
    )


def test_zero_velocity_bracket_ignores_unrelated_zero_overlapping_temperature() -> None:
    summary = _resolve_zero_velocity(
        [
            {
                "temperature_K": 800.0,
                "mean_velocity_m_per_s": 1.0,
                "standard_error_m_per_s": 1.0,
                "confidence_interval_95_m_per_s": [-1.0, 3.0],
            },
            {
                "temperature_K": 900.0,
                "mean_velocity_m_per_s": 6.0,
                "standard_error_m_per_s": 0.5,
                "confidence_interval_95_m_per_s": [5.0, 7.0],
            },
            {
                "temperature_K": 950.0,
                "mean_velocity_m_per_s": -4.0,
                "standard_error_m_per_s": 0.5,
                "confidence_interval_95_m_per_s": [-5.0, -3.0],
            },
        ]
    )

    assert summary["status"] == "resolved_for_this_finite_protocol"
    assert summary["bracket_temperature_K"] == [900.0, 950.0]
    assert summary["interpolated_temperature_K"] == pytest.approx(930.0)
    assert summary[
        "other_temperatures_with_zero_overlapping_confidence_interval_K"
    ] == [800.0]


def test_zero_velocity_remains_unresolved_without_confidence_resolved_sign_pair() -> None:
    summary = _resolve_zero_velocity(
        [
            {
                "temperature_K": 900.0,
                "mean_velocity_m_per_s": 1.0,
                "standard_error_m_per_s": 1.0,
                "confidence_interval_95_m_per_s": [-1.0, 3.0],
            },
            {
                "temperature_K": 950.0,
                "mean_velocity_m_per_s": -2.0,
                "standard_error_m_per_s": 1.0,
                "confidence_interval_95_m_per_s": [-4.0, 0.5],
            },
        ]
    )

    assert summary["status"] == "unresolved"
    assert summary["interpolated_temperature_K"] is None
    assert summary["temperatures_with_zero_overlapping_confidence_interval_K"] == [
        900.0,
        950.0,
    ]


def test_zero_velocity_rejects_robust_negative_to_positive_reversal() -> None:
    summary = _resolve_zero_velocity(
        [
            {
                "temperature_K": 850.0,
                "mean_velocity_m_per_s": 5.0,
                "standard_error_m_per_s": 0.5,
                "confidence_interval_95_m_per_s": [4.0, 6.0],
            },
            {
                "temperature_K": 900.0,
                "mean_velocity_m_per_s": -4.0,
                "standard_error_m_per_s": 0.5,
                "confidence_interval_95_m_per_s": [-5.0, -3.0],
            },
            {
                "temperature_K": 950.0,
                "mean_velocity_m_per_s": 3.0,
                "standard_error_m_per_s": 0.5,
                "confidence_interval_95_m_per_s": [2.0, 4.0],
            },
        ]
    )

    assert summary["status"] == "unresolved"
    assert summary["interpolated_temperature_K"] is None
    assert summary["reverse_sign_temperature_pairs_K"] == [[900.0, 950.0]]
    assert summary["candidate_positive_to_negative_brackets_K"] == [
        [850.0, 900.0]
    ]


def test_mh1_transition_grid_matches_baseline_protocol() -> None:
    baseline = load_transition_config(TRANSITION_CONFIG)
    candidate = load_transition_config(MH1_TRANSITION_CONFIG)

    assert candidate.generator.potential.model_name == "mace-mh-1-omat-pbe"
    assert candidate.generator.potential.head == "omat_pbe"
    assert candidate.generator.potential.usage_mode == "exploratory"
    assert candidate.generator.system.liquid_slab_fraction == 0.5
    assert candidate.random_seeds == baseline.random_seeds
    assert candidate.sample_interval == baseline.sample_interval
    assert candidate.analysis == baseline.analysis
    assert candidate.temperature_runs == baseline.temperature_runs
    assert candidate.source_dataset != baseline.source_dataset
    assert candidate.output.root_dir != baseline.output.root_dir


def test_small_direct_coexistence_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    interface_dir = source_root / "replica_000_solid_liquid_interface"
    interface_dir.mkdir(parents=True)

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
            volume_A3=np.asarray([atoms.get_volume()], dtype=np.float64),
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
    output_root = tmp_path / "transitions"
    transition_raw = {
        "dataset_name": "test_direct_coexistence",
        "source_generator_config": str(generator_path),
        "source_dataset": str(source_root),
        "source_interface_environment": "replica_000_solid_liquid_interface",
        "source_frame_step": 0,
        "random_seeds": [24680, 24681],
        "sample_interval": 1,
        "analysis": {
            "profile_bins": 4,
            "profile_smoothing_bins": 1,
            "ptm_rmsd_cutoff": 0.1,
            "minimum_profile_contrast": 0.1,
            "minimum_velocity_fit_r_squared": 0.0,
            "rdf_cutoff_A": 5.0,
            "rdf_bins": 50,
        },
        "temperature_runs": [
            {
                "name": "temperature_0300K",
                "temperature_K": 300.0,
                "expected_direction": "growth",
                "equilibration_steps": 2,
                "production_steps": 3,
                "steady_state_start_step": 1,
                "steady_state_end_step": 3,
                "minimum_crystalline_fraction_change": 0.0,
            },
            {
                "name": "temperature_0600K",
                "temperature_K": 600.0,
                "expected_direction": "melting",
                "equilibration_steps": 2,
                "production_steps": 3,
                "steady_state_start_step": 1,
                "steady_state_end_step": 3,
                "minimum_crystalline_fraction_change": 0.0,
            },
        ],
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
    potential = config.generator.potential
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": config.generator.to_dict(),
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
                        "scientifically_qualified": potential.scientifically_qualified,
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

    def spatial_test_structure_types(frame_atoms, _rmsd_cutoff):
        scaled_z = frame_atoms.get_scaled_positions(wrap=True)[:, 2]
        structure_types = np.zeros(len(frame_atoms), dtype=np.int32)
        structure_types[(scaled_z < 0.25) | (scaled_z >= 0.75)] = 1
        return structure_types

    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.transition_analysis._ptm_structure_types",
        spatial_test_structure_types,
    )
    result = generate_transition_dataset(
        config,
        calculator=EMT(),
        injected_calculator_identity="ASE EMT direct-coexistence functional test",
        progress=lambda _message: None,
    )

    assert len(result.branch_dirs) == 4
    assert [path.name for path in result.branch_dirs] == ["replica_000", "replica_001"] * 2
    assert (result.output_root / "transition_overview.png").is_file()
    assert (result.output_root / "phase_rdf_overview.png").is_file()
    assert (result.output_root / "structure_slice_overview.png").is_file()
    for branch_dir in result.branch_dirs:
        assert np.load(branch_dir / "atoms.npy").shape == (atom_count, 3)
        with np.load(branch_dir / "trajectory.npz") as trajectory:
            assert trajectory["step"].tolist() == [0, 1, 2, 3]
            assert trajectory["positions_A"].shape == (4, atom_count, 3)
            np.testing.assert_allclose(
                np.linalg.det(trajectory["cell_vectors_A"]),
                trajectory["volume_A3"],
            )
        with np.load(branch_dir / "transition_progress.npz") as transition:
            assert transition["crystalline_profile"].shape == (4, 4)
            assert transition["interface_positions_fractional"].shape == (4, 2)
            assert transition["profile_contrast"].shape == (4,)
            assert np.isfinite(transition["mean_interface_advance_A"]).all()
        with np.load(branch_dir / "phase_rdf.npz") as rdf:
            assert rdf["phase_names"].tolist() == [
                "solid_bulk",
                "liquid_bulk",
                "interface",
            ]
            assert rdf["g_r"].shape == (4, 3, 50)
            assert np.isfinite(rdf["g_r"]).all()
        metadata = json.loads((branch_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["physics"]["method"] == "planar direct solid-liquid coexistence"
        assert metadata["physics"]["equilibration_excluded_from_production"] is True
        assert metadata["rdf"]["frames"] == 4
        assert (branch_dir / "visualizations/transition_progress.png").is_file()
        assert (branch_dir / "visualizations/phase_rdf.png").is_file()
        assert (branch_dir / "visualizations/structure_slice.png").is_file()

    velocity_summary = json.loads(
        (result.output_root / "velocity_summary.json").read_text(encoding="utf-8")
    )
    assert velocity_summary["schema_version"] == 2
    assert velocity_summary["zero_velocity_bracket_K"] is None
    assert velocity_summary["zero_velocity"]["status"] == "unresolved"
    assert velocity_summary["protocol"]["chemical_symbol"] == "Al"
    assert velocity_summary["protocol"]["atom_count"] == atom_count
    assert velocity_summary["protocol"]["interface_normal_crystal_direction"] == "[001]"
    for temperature in velocity_summary["temperatures"]:
        for run in temperature["runs"]:
            assert set(run["artifacts"]) == {
                "trajectory.npz",
                "equilibration_trajectory.npz",
                "transition_progress.npz",
                "metadata.json",
            }
            for artifact in run["artifacts"].values():
                artifact_path = Path(artifact["path"])
                assert artifact_path.is_file()
                assert (
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == artifact["sha256"]
                )

    (result.output_root / "phase_rdf_overview.png").unlink()
    for branch_dir in result.branch_dirs:
        (branch_dir / "phase_rdf.npz").unlink()
        (branch_dir / "visualizations/phase_rdf.png").unlink()
    add_phase_rdf_to_transition_dataset(config, progress=lambda _message: None)
    assert (result.output_root / "phase_rdf_overview.png").is_file()
    for branch_dir in result.branch_dirs:
        assert (branch_dir / "phase_rdf.npz").is_file()
        assert (branch_dir / "visualizations/phase_rdf.png").is_file()

    with pytest.raises(FileExistsError, match="before loading the source"):
        generate_transition_dataset(config, progress=lambda _message: None)
