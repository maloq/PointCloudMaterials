from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from src.data_utils.synthetic.atomistic.config import load_config
from src.data_utils.synthetic.atomistic.simulation import run_nvt


PRODUCTION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
)


class _ModeTrackingCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self) -> None:
        super().__init__()
        self.md_property_mode = "forces_stress"
        self.calls: list[tuple[bool, bool]] = []

    def set_md_property_mode(self, mode: str) -> None:
        if mode not in {"forces", "forces_stress"}:
            raise ValueError(f"Unsupported test calculator mode={mode!r}.")
        self.md_property_mode = mode
        self.results = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        requested = {"energy"} if properties is None else set(properties)
        effective = set(requested)
        if self.md_property_mode == "forces_stress" and requested.intersection(
            {"forces", "stress"}
        ):
            effective.update({"forces", "stress"})
        compute_forces = "forces" in effective
        compute_stress = "stress" in effective
        self.calls.append((compute_forces, compute_stress))
        self.results = {"energy": 0.0}
        if compute_forces:
            if atoms is None:
                raise RuntimeError("The test force request did not provide ASE atoms.")
            self.results["forces"] = np.zeros((len(atoms), 3), dtype=np.float64)
        if compute_stress:
            self.results["stress"] = np.zeros(6, dtype=np.float64)


def _nvt_test_system() -> tuple[Atoms, _ModeTrackingCalculator, FixAtoms]:
    atoms = Atoms(
        "Al4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [0.0, 2.8, 0.0],
            [0.0, 0.0, 2.8],
        ],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    atoms.set_velocities(np.zeros((4, 3), dtype=np.float64))
    original_constraint = FixAtoms(indices=[0])
    atoms.set_constraint(original_constraint)
    calculator = _ModeTrackingCalculator()
    atoms.calc = calculator
    return atoms, calculator, original_constraint


def _nvt_config(*, property_mode: str | None):
    config = load_config(PRODUCTION_CONFIG)
    return replace(
        config,
        potential=replace(
            config.potential,
            nvt_md_property_mode=property_mode,
        ),
        dynamics=replace(config.dynamics, sample_interval=2),
    )


def test_force_only_nvt_uses_sparse_stress_and_restores_runtime_state() -> None:
    atoms, calculator, original_constraint = _nvt_test_system()

    trace = run_nvt(
        atoms,
        config=_nvt_config(property_mode="forces"),
        temperature_K=650.0,
        steps=3,
        stage="force-only-test",
        initialize_velocities=False,
        rng=np.random.default_rng(123),
        progress=lambda _message: None,
    )

    assert trace.step.tolist() == [0, 2, 3]
    assert (True, True) not in calculator.calls
    assert calculator.calls.count((False, True)) == len(trace.step)
    assert calculator.calls.count((True, False)) >= 4
    assert set(calculator.calls) == {(True, False), (False, True)}
    assert calculator.md_property_mode == "forces_stress"
    assert calculator.results == {}
    assert atoms.constraints == [original_constraint]


def test_force_only_nvt_restores_mode_and_constraints_after_failure() -> None:
    atoms, calculator, original_constraint = _nvt_test_system()

    def fail_progress(_message: str) -> None:
        raise RuntimeError("intentional progress failure")

    with pytest.raises(RuntimeError, match="intentional progress failure"):
        run_nvt(
            atoms,
            config=_nvt_config(property_mode="forces"),
            temperature_K=650.0,
            steps=1,
            stage="failure-test",
            initialize_velocities=False,
            rng=np.random.default_rng(456),
            progress=fail_progress,
        )

    assert calculator.md_property_mode == "forces_stress"
    assert calculator.results == {}
    assert atoms.constraints == [original_constraint]


def test_default_nvt_mode_preserves_combined_force_stress_evaluations() -> None:
    atoms, calculator, _original_constraint = _nvt_test_system()

    run_nvt(
        atoms,
        config=_nvt_config(property_mode=None),
        temperature_K=650.0,
        steps=1,
        stage="legacy-mode-test",
        initialize_velocities=False,
        rng=np.random.default_rng(789),
        progress=lambda _message: None,
    )

    force_calls = [call for call in calculator.calls if call[0]]
    assert force_calls
    assert all(compute_stress for _compute_force, compute_stress in force_calls)
    assert calculator.md_property_mode == "forces_stress"
