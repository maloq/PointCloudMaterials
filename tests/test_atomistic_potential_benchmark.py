from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from src.data_utils.synthetic.atomistic.potential_benchmark import (
    DFTReferenceConfig,
    QualificationScopeConfig,
    _read_reference_frames,
    dft_error_metrics,
    equation_of_state_metrics,
    load_potential_benchmark_config,
    nve_energy_conservation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/potential_benchmark.yaml"
)


def test_verlet_cache_rebuilds_when_atom_count_changes() -> None:
    from src.data_utils.synthetic.atomistic.calculator import VerletSkinMACECalculator

    calculator = object.__new__(VerletSkinMACECalculator)
    calculator._reference_cell_A = np.eye(3)
    calculator._reference_scaled_positions = np.zeros((4, 3))
    calculator.r_max = 5.0
    calculator.neighbor_skin_A = 0.3

    assert not calculator._graph_is_valid(np.zeros((2, 3)), np.eye(3))


def test_al_fcc_equation_of_state_is_fitted_inside_scan() -> None:
    metrics = equation_of_state_metrics(
        EMT(),
        symbol="Al",
        phase="fcc",
        reference_fcc_a_A=4.05,
        volume_scales=(0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10),
        repetitions=(2, 2, 2),
    )
    assert 3.9 < metrics["equilibrium_lattice_constant_A"] < 4.1
    assert metrics["bulk_modulus_GPa"] > 0.0


def test_nve_diagnostic_records_the_requested_duration() -> None:
    metrics = nve_energy_conservation(
        EMT(),
        symbol="Al",
        lattice_constant_A=4.0,
        repetitions=(2, 2, 2),
        temperature_K=300.0,
        timestep_fs=1.0,
        duration_ps=0.01,
        sample_interval_fs=2.0,
        random_seed=17,
    )
    assert metrics["duration_ps"] == 0.01
    assert metrics["sample_count"] == 6
    assert np.isfinite(metrics["drift_meV_per_atom_ps"])


def test_dft_error_metrics_are_zero_for_the_reference_calculator() -> None:
    frames = []
    for state, scale in (("solid_bulk", 1.0), ("liquid_bulk", 1.03), ("interface", 0.98)):
        atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
        atoms.set_cell(np.asarray(atoms.cell) * scale, scale_atoms=True)
        atoms.info["state"] = state
        atoms.calc = EMT()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=forces,
            stress=stress,
        )
        frames.append(atoms)

    metrics = dft_error_metrics(frames, EMT())

    assert metrics[
        "energy_rmse_meV_per_atom_after_global_constant_offset"
    ] < 1.0e-10
    assert metrics["force_rmse_eV_per_A"] < 1.0e-12
    assert metrics["stress_rmse_GPa"] < 1.0e-10
    assert set(metrics["by_state"]) == {"solid_bulk", "liquid_bulk", "interface"}


def test_dft_error_metrics_reject_nonfinite_reference() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.info["state"] = "solid_bulk"
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    forces[0, 0] = np.nan
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=forces,
        stress=np.zeros(6),
    )

    with pytest.raises(FloatingPointError, match="Non-finite residual"):
        dft_error_metrics([atoms], EMT())


def test_dft_reference_rejects_wrong_element(tmp_path) -> None:
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.info.update(
        {
            "state": "solid_bulk",
            "configuration_id": "cu-not-al",
            "temperature_K": 650.0,
            "target_pressure_GPa": 0.0,
            "reference_code": "test-code",
            "reference_level_of_theory": "test-theory",
            "reference_pseudopotential": "test-pseudo",
            "reference_plane_wave_cutoff_eV": 500.0,
            "reference_kpoint_spacing_per_A": 0.2,
        }
    )
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    path = tmp_path / "wrong-element.extxyz"
    write(path, atoms)
    reference = DFTReferenceConfig(
        extxyz=path,
        code="test-code",
        level_of_theory="test-theory",
        pseudopotential="test-pseudo",
        plane_wave_cutoff_eV=500.0,
        kpoint_spacing_per_A=0.2,
        source_url="https://example.invalid/reference",
        minimum_frames_per_state=5,
    )
    scope = QualificationScopeConfig(
        pressure_range_GPa=(-0.1, 0.1),
        state_temperature_ranges_K={"solid_bulk": (650.0, 650.0)},
        maximum_timestep_fs=1.0,
        authorized_claims={
            "phase_context_structure": True,
            "equilibrium_thermodynamics": True,
            "kinetics": False,
        },
    )

    with pytest.raises(ValueError, match="only Al atoms"):
        _read_reference_frames(
            reference,
            chemical_symbol="Al",
            required_states=("solid_bulk",),
            scope=scope,
        )


def _al_reference_frame(
    configuration_id: str,
    *,
    scale: float,
    target_pressure_GPa: float,
) -> Atoms:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.set_cell(np.asarray(atoms.cell) * scale, scale_atoms=True)
    atoms.info.update(
        {
            "state": "solid_bulk",
            "configuration_id": configuration_id,
            "temperature_K": 650.0,
            "target_pressure_GPa": target_pressure_GPa,
            "reference_code": "test-code",
            "reference_level_of_theory": "test-theory",
            "reference_pseudopotential": "test-pseudo",
            "reference_plane_wave_cutoff_eV": 500.0,
            "reference_kpoint_spacing_per_A": 0.2,
        }
    )
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    return atoms


def _test_reference(path: Path) -> DFTReferenceConfig:
    return DFTReferenceConfig(
        extxyz=path,
        code="test-code",
        level_of_theory="test-theory",
        pseudopotential="test-pseudo",
        plane_wave_cutoff_eV=500.0,
        kpoint_spacing_per_A=0.2,
        source_url="https://example.invalid/reference",
        minimum_frames_per_state=5,
    )


def _test_scope(pressure_range_GPa: tuple[float, float]) -> QualificationScopeConfig:
    return QualificationScopeConfig(
        pressure_range_GPa=pressure_range_GPa,
        state_temperature_ranges_K={"solid_bulk": (650.0, 650.0)},
        maximum_timestep_fs=1.0,
        authorized_claims={
            "phase_context_structure": True,
            "equilibrium_thermodynamics": True,
            "kinetics": False,
        },
    )


def test_dft_reference_rejects_duplicate_geometry_with_different_ids(tmp_path) -> None:
    frames = [
        _al_reference_frame(
            f"configuration-{index}",
            scale=1.0 if index < 2 else 1.0 + 0.01 * index,
            target_pressure_GPa=0.0,
        )
        for index in range(5)
    ]
    source = frames[0]
    permutation = np.arange(len(source) - 1, -1, -1)
    reordered_and_wrapped = source[permutation]
    reordered_and_wrapped.positions[0] += np.asarray(reordered_and_wrapped.cell[0])
    reordered_and_wrapped.info = dict(source.info)
    reordered_and_wrapped.info["configuration_id"] = "configuration-1"
    reordered_and_wrapped.calc = SinglePointCalculator(
        reordered_and_wrapped,
        energy=source.get_potential_energy(),
        forces=source.get_forces()[permutation],
        stress=source.get_stress(voigt=True),
    )
    frames[1] = reordered_and_wrapped
    path = tmp_path / "duplicate-geometry.extxyz"
    write(path, frames)

    with pytest.raises(ValueError, match="duplicates the exact periodic geometry"):
        _read_reference_frames(
            _test_reference(path),
            chemical_symbol="Al",
            required_states=("solid_bulk",),
            scope=_test_scope((0.0, 0.0)),
        )


def test_dft_reference_pressure_must_span_scope_for_every_state(tmp_path) -> None:
    frames = [
        _al_reference_frame(
            f"configuration-{index}",
            scale=1.0 + 0.01 * index,
            target_pressure_GPa=0.0,
        )
        for index in range(5)
    ]
    path = tmp_path / "missing-pressure-coverage.extxyz"
    write(path, frames)

    with pytest.raises(ValueError, match="target-pressure coverage.*does not span"):
        _read_reference_frames(
            _test_reference(path),
            chemical_symbol="Al",
            required_states=("solid_bulk",),
            scope=_test_scope((-0.1, 0.1)),
        )


def test_supplied_benchmark_has_claim_and_timestep_scope() -> None:
    config = load_potential_benchmark_config(BENCHMARK_CONFIG)

    assert config.qualification_scope.pressure_range_GPa == (0.0, 0.0)
    assert config.qualification_scope.maximum_timestep_fs in config.nve_timesteps_fs
    assert config.qualification_scope.authorized_claims == {
        "phase_context_structure": True,
        "equilibrium_thermodynamics": True,
        "kinetics": False,
    }


def test_benchmark_requires_scoped_timestep_in_nve_grid(tmp_path) -> None:
    raw = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["qualification_scope"]["maximum_timestep_fs"] = 0.75
    path = tmp_path / "unvalidated-timestep.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must be explicitly included in nve.timesteps_fs"):
        load_potential_benchmark_config(path)


def test_benchmark_requires_at_least_five_dft_frames_per_state(tmp_path) -> None:
    raw = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    dft_path = tmp_path / "reference.extxyz"
    dft_path.write_text("placeholder\n", encoding="utf-8")
    raw["dft_reference"] = {
        "extxyz": str(dft_path),
        "code": "test-code",
        "level_of_theory": "test-theory",
        "pseudopotential": "test-pseudo",
        "plane_wave_cutoff_eV": 500.0,
        "kpoint_spacing_per_A": 0.2,
        "source_url": "https://example.invalid/reference",
        "minimum_frames_per_state": 4,
    }
    path = tmp_path / "too-few-dft-frames.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_frames_per_state must be >= 5"):
        load_potential_benchmark_config(path)


@pytest.mark.parametrize(
    ("claims", "error_type", "error_match"),
    [
        (
            {
                "phase_context_structure": True,
                "equilibrium_thermodynamics": True,
            },
            KeyError,
            "must contain exactly",
        ),
        (
            {
                "phase_context_structure": 1,
                "equilibrium_thermodynamics": True,
                "kinetics": False,
            },
            TypeError,
            "exact boolean",
        ),
        (
            {
                "phase_context_structure": True,
                "equilibrium_thermodynamics": True,
                "kinetics": True,
            },
            ValueError,
            "cannot authorize kinetics",
        ),
    ],
)
def test_benchmark_claim_authorizations_are_exact(
    tmp_path,
    claims,
    error_type,
    error_match,
) -> None:
    raw = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["qualification_scope"]["authorized_claims"] = claims
    path = tmp_path / "invalid-claims.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(error_type, match=error_match):
        load_potential_benchmark_config(path)
