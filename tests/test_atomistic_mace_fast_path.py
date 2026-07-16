from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from mace import data as mace_data
from mace.data.padding_tools import build_fake_padding_graph
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools

from src.data_utils.synthetic.atomistic.calculator import VerletSkinMACECalculator
from src.data_utils.synthetic.atomistic.config import (
    load_config,
    mace_kernel_backend,
)


class _TensorBatch(dict[str, torch.Tensor]):
    def to_dict(self) -> dict[str, torch.Tensor]:
        return dict(self)


class _PropertyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calls: list[tuple[bool, bool]] = []

    def forward(
        self,
        data: dict[str, torch.Tensor],
        *,
        compute_force: bool,
        compute_stress: bool,
        training: bool,
        compute_edge_forces: bool,
        compute_atomic_stresses: bool,
    ) -> dict[str, torch.Tensor | None]:
        del data, training, compute_edge_forces, compute_atomic_stresses
        self.calls.append((compute_force, compute_stress))
        return {
            "energy": torch.tensor([7.5]),
            "node_energy": torch.tensor([3.0, 4.5]),
            "forces": (
                torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
                if compute_force or compute_stress
                else None
            ),
            "stress": (
                torch.tensor(
                    [[[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]]
                )
                if compute_stress
                else None
            ),
        }


def _property_calculator(mode: str) -> tuple[VerletSkinMACECalculator, _PropertyModel]:
    calculator = object.__new__(VerletSkinMACECalculator)
    Calculator.__init__(calculator)
    model = _PropertyModel()
    calculator.models = [model]
    calculator.md_property_mode = mode
    calculator.use_compile = False
    calculator._enable_oeq = False
    calculator._enable_cueq = True
    calculator.energy_units_to_eV = 1.0
    calculator.length_units_to_A = 1.0
    calculator._real_atom_count = 2
    calculator.model_evaluation_count = 0
    calculator.force_evaluation_count = 0
    calculator.stress_evaluation_count = 0
    batch = _TensorBatch(
        positions=torch.zeros((2, 3)),
        node_attrs=torch.ones((2, 1)),
        head=torch.tensor([0], dtype=torch.long),
        batch=torch.tensor([0, 0], dtype=torch.long),
        ptr=torch.tensor([0, 2], dtype=torch.long),
    )
    calculator._atoms_to_batch = MethodType(
        lambda _self, _atoms: batch,
        calculator,
    )
    return calculator, model


def test_force_only_fast_path_skips_stress_and_reuses_computed_energy() -> None:
    calculator, model = _property_calculator("forces")
    atoms = Atoms("Al2", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])

    calculator.calculate(atoms, ["forces"], all_changes)

    assert model.calls == [(True, False)]
    assert set(calculator.results) == {"energy", "free_energy", "forces"}
    assert "stress" not in calculator.results
    assert "node_energy" not in calculator.results
    assert np.array_equal(
        calculator.results["forces"],
        np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    )

    # Energy was already formed by the force derivative. The one-scalar cache
    # prevents a second complete MACE call at a thermodynamic sample.
    calculator.calculate(atoms, ["energy"], [])
    assert model.calls == [(True, False)]
    assert calculator.results["energy"] == pytest.approx(7.5)


def test_npt_fast_path_returns_exact_force_and_stress_together() -> None:
    calculator, model = _property_calculator("forces_stress")
    atoms = Atoms("Al2", positions=[[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])

    calculator.calculate(atoms, ["stress"], all_changes)

    assert model.calls == [(True, True)]
    assert set(calculator.results) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }
    assert np.allclose(
        calculator.results["stress"],
        np.array([1.0, 2.0, 3.0, 0.3, 0.2, 0.1]),
        atol=1.0e-7,
    )


def _pair_outputs(batch: _TensorBatch) -> tuple[np.ndarray, np.ndarray]:
    sender = int(batch["edge_index"][0, 0])
    receiver = int(batch["edge_index"][1, 0])
    vector = (
        batch["positions"][receiver]
        - batch["positions"][sender]
        + batch["shifts"][0]
    )
    forces = torch.zeros_like(batch["positions"])
    forces[sender] = vector
    forces[receiver] = -vector
    stress = torch.outer(vector, vector)
    return forces.numpy(), stress.numpy()


def test_periodic_image_crossing_preserves_cached_force_and_stress() -> None:
    calculator = object.__new__(VerletSkinMACECalculator)
    calculator.r_max = 1.0
    calculator.neighbor_skin_A = 0.5
    calculator._reference_cell_A = np.diag([10.0, 10.0, 10.0])
    calculator._reference_scaled_positions = np.array(
        [[0.99, 0.5, 0.5], [0.02, 0.5, 0.5]]
    )
    calculator._reference_pbc = np.array([True, False, False])
    calculator._real_atom_count = 2
    calculator._real_edge_count = 1
    calculator.graph_request_count = 0
    calculator.graph_rebuild_count = 1
    calculator.graph_reuse_count = 0
    cached_batch = _TensorBatch(
        positions=torch.tensor([[9.9, 5.0, 5.0], [0.2, 5.0, 5.0]]),
        cell=torch.diag(torch.tensor([10.0, 10.0, 10.0])),
        unit_shifts=torch.tensor([[1.0, 0.0, 0.0]]),
        shifts=torch.tensor([[10.0, 0.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    calculator._cached_batch = cached_batch
    calculator._rebuild_graph = MethodType(
        lambda _self, _atoms, _positions, _cell: pytest.fail(
            "a pure periodic image crossing must not rebuild the graph"
        ),
        calculator,
    )

    wrapped = Atoms(
        "Al2",
        positions=[[0.1, 5.0, 5.0], [0.2, 5.0, 5.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=[True, False, False],
    )
    reused_batch = calculator._atoms_to_batch(wrapped)
    fresh_batch = _TensorBatch(
        positions=torch.tensor([[0.1, 5.0, 5.0], [0.2, 5.0, 5.0]]),
        shifts=torch.zeros((1, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )

    cached_forces, cached_stress = _pair_outputs(reused_batch)
    fresh_forces, fresh_stress = _pair_outputs(fresh_batch)
    assert calculator.graph_reuse_count == 1
    assert reused_batch["positions"][0, 0].item() == pytest.approx(10.1)
    assert np.allclose(cached_forces, fresh_forces, atol=1.0e-6)
    assert np.allclose(cached_stress, fresh_stress, atol=1.0e-6)


def _al_atomic_data(*, cell_scale: float):
    atoms = Atoms(
        "Al4",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        cell=np.eye(3) * 4.05 * cell_scale,
        pbc=True,
    )
    configuration = mace_data.config_from_atoms(atoms, head_name="default")
    with torch_tools.default_dtype("float32"):
        return mace_data.AtomicData.from_config(
            configuration,
            z_table=AtomicNumberTable([13]),
            cutoff=3.2,
            heads=["default"],
        )


def _assert_same_padding_graph(observed: object, expected: object) -> None:
    observed_values = observed.to_dict()
    expected_values = expected.to_dict()
    assert set(observed_values) == set(expected_values)
    for key in expected_values:
        observed_value = observed_values[key]
        expected_value = expected_values[key]
        if torch.is_tensor(expected_value):
            assert torch.equal(observed_value, expected_value), key
        else:
            assert observed_value == expected_value, key
    assert observed.num_nodes == expected.num_nodes


def test_padding_template_matches_full_graph_reference_after_cell_change() -> None:
    calculator = object.__new__(VerletSkinMACECalculator)
    calculator.r_max = 3.2
    calculator._padding_graph_template = None

    first = _al_atomic_data(cell_scale=1.0)
    expected_first = build_fake_padding_graph(
        first, num_atoms=3, num_edges=11, r_max=calculator.r_max
    )
    observed_first = calculator._build_padding_graph(
        first, fake_atom_count=3, fake_edge_count=11
    )
    _assert_same_padding_graph(observed_first, expected_first)
    template = calculator._padding_graph_template
    assert template.num_nodes == 2
    assert template["edge_index"].shape == (2, 0)

    deformed = _al_atomic_data(cell_scale=1.01)
    expected_deformed = build_fake_padding_graph(
        deformed, num_atoms=3, num_edges=11, r_max=calculator.r_max
    )
    observed_deformed = calculator._build_padding_graph(
        deformed, fake_atom_count=3, fake_edge_count=11
    )
    assert calculator._padding_graph_template is template
    _assert_same_padding_graph(observed_deformed, expected_deformed)


def test_compiled_graph_refill_matches_fresh_fixed_shape_batch() -> None:
    first_atoms = Atoms(
        "Al4",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        cell=np.eye(3) * 4.05,
        pbc=True,
    )
    deformed_atoms = first_atoms.copy()
    deformed_atoms.set_cell(np.diag([4.07, 4.03, 4.06]), scale_atoms=True)
    deformed_atoms.positions += np.array(
        [
            [0.010, -0.005, 0.002],
            [-0.006, 0.004, -0.003],
            [0.002, 0.003, 0.005],
            [-0.004, -0.002, -0.004],
        ]
    )

    z_table = AtomicNumberTable([13])

    def atomic_data(atoms: Atoms):
        configuration = mace_data.config_from_atoms(atoms, head_name="default")
        with torch_tools.default_dtype("float32"):
            return mace_data.AtomicData.from_config(
                configuration,
                z_table=z_table,
                cutoff=3.2,
                heads=["default"],
            )

    pad_num_edges = 100
    first = atomic_data(first_atoms)
    first_real_edges = int(first["edge_index"].shape[1])
    first_padding = build_fake_padding_graph(
        first,
        num_atoms=1,
        num_edges=pad_num_edges - first_real_edges,
        r_max=3.0,
    )
    cached_batch = torch_geometric.Batch.from_data_list([first, first_padding])

    calculator = object.__new__(VerletSkinMACECalculator)
    calculator.r_max = 3.0
    calculator.neighbor_skin_A = 0.2
    calculator.pad_num_atoms = 4
    calculator.pad_num_edges = pad_num_edges
    calculator._cached_batch = cached_batch
    calculator._reference_atomic_numbers = np.asarray(first_atoms.numbers).copy()
    calculator._real_atom_count = 4
    calculator._real_edge_count = first_real_edges
    calculator._maximum_real_edge_count = first_real_edges
    calculator.graph_rebuild_count = 1
    calculator.compiled_graph_refill_count = 0

    observed = calculator._refill_compiled_graph(
        deformed_atoms,
        np.asarray(deformed_atoms.positions, dtype=np.float64),
        np.asarray(deformed_atoms.cell.array, dtype=np.float64),
    )

    deformed = atomic_data(deformed_atoms)
    deformed_real_edges = int(deformed["edge_index"].shape[1])
    deformed_padding = build_fake_padding_graph(
        deformed,
        num_atoms=1,
        num_edges=pad_num_edges - deformed_real_edges,
        r_max=3.0,
    )
    expected = torch_geometric.Batch.from_data_list([deformed, deformed_padding])
    _assert_same_padding_graph(observed, expected)
    assert calculator.compiled_graph_refill_count == 1
    assert calculator._real_edge_count == deformed_real_edges


def test_compile_and_kernel_controls_are_explicit_and_strict(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/simulation/atomistic/al/phase_context_70304_mpa.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["potential"].update(
        {
            "compile_mode": "reduce-overhead",
            "compile_fullgraph": False,
            "pad_num_atoms": 70_304,
            "pad_num_edges": 6_000_000,
            "enable_cueq": True,
            "enable_oeq": True,
            "md_property_mode": "forces_stress",
        }
    )
    compiled_path = tmp_path / "compiled.yaml"
    compiled_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    compiled = load_config(compiled_path)
    assert compiled.potential.compile_mode == "reduce-overhead"
    assert compiled.potential.pad_num_atoms == 70_304
    assert mace_kernel_backend(
        enable_cueq=compiled.potential.enable_cueq,
        enable_oeq=compiled.potential.enable_oeq,
    ) == "hybrid_cueq_oeq"

    raw["potential"]["pad_num_edges"] = 0
    invalid_path = tmp_path / "invalid-padding.yaml"
    invalid_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="requires explicit positive fixed-shape"):
        load_config(invalid_path)


def test_mace_0315_rejects_every_compiled_or_oeq_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.data_utils.synthetic.atomistic.calculator.version",
        lambda _distribution: "0.3.15",
    )
    with pytest.raises(
        RuntimeError,
        match="permitted only for the existing uncompiled zero-padding",
    ):
        VerletSkinMACECalculator(
            neighbor_skin_A=0.3,
            md_property_mode="forces_stress",
            compile_mode="reduce-overhead",
            fullgraph=False,
            enable_cueq=True,
            enable_oeq=False,
            pad_num_atoms=16_384,
            pad_num_edges=1_200_000,
            default_dtype="float32",
        )
