from __future__ import annotations

import math
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.stress import full_3x3_to_voigt_6_stress
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools import torch_geometric, torch_tools


MACE_TORCH_VERSION = "0.3.16"
LEGACY_UNCOMPILED_MACE_TORCH_VERSION = "0.3.15"
SUPPORTED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
SUPPORTED_MD_PROPERTY_MODES = {"forces", "forces_stress"}
SUPPORTED_FAST_PROPERTIES = {
    "energy",
    "free_energy",
    "energies",
    "node_energy",
    "forces",
    "stress",
}


class VerletSkinMACECalculator(MACECalculator):
    """MACE calculator optimized for the repository's fixed-composition MD.

    The real neighbor graph is built to ``r_max + neighbor_skin_A`` while the
    model retains its trained cutoff ``r_max``.  A rigorous deformation and
    non-affine-displacement bound controls graph reuse.  Compiled execution
    requires explicit atom and edge padding budgets; exceeding either budget is
    a hard error rather than a silent recompile.

    ``md_property_mode='forces_stress'`` evaluates and retains force and stress
    together.  This matches ASE's MTK NPT access pattern and avoids two model
    calls per state.  ``'forces'`` omits stress for NVT work.  Derivative calls
    cache the already-computed scalar energy for sparse thermodynamic sampling;
    per-atom energy tensors are transferred only when ASE requests them.
    """

    def __init__(
        self,
        *args: object,
        neighbor_skin_A: float,
        md_property_mode: str,
        compile_mode: str | None = None,
        fullgraph: bool = False,
        enable_cueq: bool = False,
        enable_oeq: bool = False,
        pad_num_atoms: int = 0,
        pad_num_edges: int = 0,
        **kwargs: object,
    ) -> None:
        installed_mace = version("mace-torch")
        supported_versions = {
            MACE_TORCH_VERSION,
            LEGACY_UNCOMPILED_MACE_TORCH_VERSION,
        }
        if installed_mace not in supported_versions:
            raise RuntimeError(
                "VerletSkinMACECalculator supports mace-torch "
                f"{MACE_TORCH_VERSION}, plus the explicitly constrained uncompiled legacy "
                f"version {LEGACY_UNCOMPILED_MACE_TORCH_VERSION}; the active environment "
                f"contains {installed_mace}. Install requirements.txt in a fresh "
                "environment rather than running against an unreviewed private API."
            )
        if not isinstance(neighbor_skin_A, (int, float)) or isinstance(
            neighbor_skin_A, bool
        ):
            raise TypeError(
                f"neighbor_skin_A must be an explicit number, got {neighbor_skin_A!r}."
            )
        if not math.isfinite(neighbor_skin_A) or neighbor_skin_A <= 0.0:
            raise ValueError(
                f"neighbor_skin_A must be finite and > 0, got {neighbor_skin_A!r}."
            )
        if not isinstance(md_property_mode, str) or (
            md_property_mode not in SUPPORTED_MD_PROPERTY_MODES
        ):
            raise ValueError(
                f"md_property_mode={md_property_mode!r} is unsupported; expected one of "
                f"{sorted(SUPPORTED_MD_PROPERTY_MODES)}."
            )
        if compile_mode is not None and (
            not isinstance(compile_mode, str)
            or compile_mode not in SUPPORTED_COMPILE_MODES
        ):
            raise ValueError(
                f"compile_mode={compile_mode!r} is unsupported; expected null or one of "
                f"{sorted(SUPPORTED_COMPILE_MODES)}."
            )
        if not isinstance(pad_num_atoms, int) or isinstance(pad_num_atoms, bool):
            raise TypeError(
                f"pad_num_atoms must be a nonnegative integer, got {pad_num_atoms!r}."
            )
        if pad_num_atoms < 0:
            raise ValueError(
                f"pad_num_atoms must be >= 0, got {pad_num_atoms}."
            )
        if not isinstance(pad_num_edges, int) or isinstance(pad_num_edges, bool):
            raise TypeError(
                f"pad_num_edges must be a nonnegative integer, got {pad_num_edges!r}."
            )
        if pad_num_edges < 0:
            raise ValueError(f"pad_num_edges must be >= 0, got {pad_num_edges}.")
        if not isinstance(fullgraph, bool):
            raise TypeError(f"fullgraph must be a boolean, got {fullgraph!r}.")
        if not isinstance(enable_cueq, bool) or not isinstance(enable_oeq, bool):
            raise TypeError(
                "enable_cueq and enable_oeq must be booleans, got "
                f"enable_cueq={enable_cueq!r}, enable_oeq={enable_oeq!r}."
            )
        if compile_mode is None and (pad_num_atoms != 0 or pad_num_edges != 0):
            raise ValueError(
                "Fixed atom/edge padding is reserved for compiled execution: "
                f"compile_mode=None, pad_num_atoms={pad_num_atoms}, "
                f"pad_num_edges={pad_num_edges}."
            )
        if compile_mode is not None and (pad_num_atoms == 0 or pad_num_edges == 0):
            raise ValueError(
                "Compiled MACE requires explicit positive fixed-shape budgets for both "
                f"atoms and edges; got pad_num_atoms={pad_num_atoms}, "
                f"pad_num_edges={pad_num_edges}. Automatic padding growth is forbidden "
                "because it causes expensive, difficult-to-audit recompilation."
            )
        if fullgraph and (enable_cueq or enable_oeq):
            raise ValueError(
                "mace-torch 0.3.16 compiles accelerated CuEq/OEq kernels with "
                "fullgraph=False. Set compile_fullgraph=false explicitly instead of "
                "requesting an option the upstream calculator ignores."
            )
        legacy_uncompiled = installed_mace == LEGACY_UNCOMPILED_MACE_TORCH_VERSION
        if legacy_uncompiled and (
            compile_mode is not None
            or enable_oeq
            or pad_num_atoms != 0
            or pad_num_edges != 0
        ):
            raise RuntimeError(
                f"mace-torch {LEGACY_UNCOMPILED_MACE_TORCH_VERSION} is permitted only for "
                "the existing uncompiled zero-padding e3nn/CuEq path. Requested settings "
                f"were compile_mode={compile_mode!r}, enable_oeq={enable_oeq}, "
                f"pad_num_atoms={pad_num_atoms}, pad_num_edges={pad_num_edges}. Install "
                f"mace-torch {MACE_TORCH_VERSION} for fixed-shape compilation or OEq/hybrid."
            )
        legacy_default_dtype = kwargs.get("default_dtype")
        if legacy_uncompiled and legacy_default_dtype not in {"float32", "float64"}:
            raise ValueError(
                f"The reviewed mace-torch {LEGACY_UNCOMPILED_MACE_TORCH_VERSION} path "
                "requires an explicit default_dtype='float32' or 'float64', got "
                f"{legacy_default_dtype!r}."
            )

        self.neighbor_skin_A = float(neighbor_skin_A)
        self.md_property_mode = md_property_mode
        if legacy_uncompiled:
            super().__init__(
                *args,
                compile_mode=None,
                fullgraph=fullgraph,
                enable_cueq=enable_cueq,
                enable_oeq=False,
                **kwargs,
            )
            # These attributes were introduced upstream in 0.3.16. They are
            # zero by construction in the reviewed 0.3.15 execution mode.
            self.default_dtype = str(legacy_default_dtype)
            self.pad_num_atoms = 0
            self.pad_num_edges = 0
        else:
            super().__init__(
                *args,
                compile_mode=compile_mode,
                fullgraph=fullgraph,
                enable_cueq=enable_cueq,
                enable_oeq=enable_oeq,
                pad_num_atoms=pad_num_atoms,
                pad_num_edges=pad_num_edges,
                **kwargs,
            )
        if self.model_type != "MACE":
            raise TypeError(
                "VerletSkinMACECalculator supports the repository's energy/force/stress "
                f"MACE models only, got model_type={self.model_type!r}."
            )
        if self.num_models != 1:
            raise ValueError(
                "Repository MD uses one checksum-bound MACE Hamiltonian at a time; "
                f"calculator loaded num_models={self.num_models}. Committee output and "
                "its additional transfers are intentionally unsupported by the fast path."
            )

        self.graph_rebuild_count = 0
        self.compiled_graph_refill_count = 0
        self.graph_reuse_count = 0
        self.graph_request_count = 0
        self.model_evaluation_count = 0
        self.force_evaluation_count = 0
        self.stress_evaluation_count = 0
        self._cached_batch: Any | None = None
        self._padding_graph_template: Any | None = None
        self._reference_cell_A: np.ndarray | None = None
        self._reference_scaled_positions: np.ndarray | None = None
        self._reference_pbc: np.ndarray | None = None
        self._reference_atomic_numbers: np.ndarray | None = None
        self._real_atom_count = 0
        self._real_edge_count = 0
        self._maximum_real_edge_count = 0

    @property
    def kernel_backend(self) -> str:
        if self._enable_cueq and self._enable_oeq:
            return "hybrid_cueq_oeq"
        if self._enable_cueq:
            return "cueq"
        if self._enable_oeq:
            return "oeq"
        return "e3nn"

    def set_md_property_mode(self, mode: str) -> None:
        """Select the exact property pair used by subsequent MD evaluations."""

        if not isinstance(mode, str) or mode not in SUPPORTED_MD_PROPERTY_MODES:
            raise ValueError(
                f"md_property_mode={mode!r} is unsupported; expected one of "
                f"{sorted(SUPPORTED_MD_PROPERTY_MODES)}."
            )
        self.md_property_mode = mode
        self.results = {}

    def graph_cache_metrics(self) -> dict[str, float | int]:
        requests = self.graph_request_count
        return {
            "requests": requests,
            "rebuilds": self.graph_rebuild_count,
            "compiled_buffer_refills": self.compiled_graph_refill_count,
            "reuses": self.graph_reuse_count,
            "reuse_fraction": (
                float(self.graph_reuse_count / requests) if requests else 0.0
            ),
            "neighbor_skin_A": self.neighbor_skin_A,
            "real_atom_count": self._real_atom_count,
            "real_edge_count": self._real_edge_count,
            "maximum_real_edge_count": self._maximum_real_edge_count,
            "maximum_edge_budget_fraction": (
                float(self._maximum_real_edge_count / self.pad_num_edges)
                if self.pad_num_edges
                else 0.0
            ),
            "pad_num_atoms": int(self.pad_num_atoms),
            "pad_num_edges": int(self.pad_num_edges),
            "model_evaluations": self.model_evaluation_count,
            "force_evaluations": self.force_evaluation_count,
            "stress_evaluations": self.stress_evaluation_count,
        }

    def clear_neighbor_cache(self) -> None:
        self._cached_batch = None
        self._padding_graph_template = None
        self._reference_cell_A = None
        self._reference_scaled_positions = None
        self._reference_pbc = None
        self._reference_atomic_numbers = None
        self._real_atom_count = 0
        self._real_edge_count = 0
        self.reset()

    def _build_padding_graph(
        self,
        atomic_data: Any,
        *,
        fake_atom_count: int,
        fake_edge_count: int,
    ) -> Any:
        """Build fixed-shape padding without cloning the full real graph.

        MACE's padding helper clones its reference graph before replacing every
        per-atom and per-edge tensor.  Using the full MD graph as that reference
        needlessly copies tens of megabytes on every neighbor rebuild.  A
        two-atom, zero-edge template retains the same schema and tensor dtypes
        while making that clone constant-size.  Two atoms are intentional: the
        upstream generic resizer distinguishes per-atom tensors from graph-level
        tensors by their leading dimension, and graph-level tensors commonly
        have a leading dimension of one.
        """

        from mace.data.padding_tools import build_fake_padding_graph

        if self._padding_graph_template is None:
            self._padding_graph_template = build_fake_padding_graph(
                atomic_data,
                num_atoms=2,
                num_edges=0,
                r_max=self.r_max,
            )
        template = self._padding_graph_template
        for key in ("rcell", "volume"):
            template[key].copy_(atomic_data[key])
        return build_fake_padding_graph(
            template,
            num_atoms=fake_atom_count,
            num_edges=fake_edge_count,
            r_max=self.r_max,
        )

    def _atoms_to_batch(self, atoms: Atoms):
        positions_A = np.asarray(atoms.positions, dtype=np.float64)
        cell_A = np.asarray(atoms.cell.array, dtype=np.float64)
        if positions_A.shape[0] == 0:
            raise ValueError("MACE MD received an empty Atoms object.")
        if cell_A.shape != (3, 3):
            raise ValueError(
                f"MACE MD requires cell shape=(3, 3), got shape={cell_A.shape}."
            )
        self.graph_request_count += 1
        current_pbc = np.asarray(atoms.pbc, dtype=bool)
        if self._reference_pbc is not None and not np.array_equal(
            current_pbc, self._reference_pbc
        ):
            return self._rebuild_graph(atoms, positions_A, cell_A)
        if self._cached_batch is None or not self._graph_is_valid(positions_A, cell_A):
            return self._rebuild_graph(atoms, positions_A, cell_A)

        self.graph_reuse_count += 1
        batch = self._cached_batch
        canonical_positions_A = self._canonical_positions(positions_A, cell_A)
        position_tensor = torch.as_tensor(
            canonical_positions_A,
            dtype=batch["positions"].dtype,
            device=batch["positions"].device,
        )
        cell_tensor = torch.as_tensor(
            cell_A,
            dtype=batch["cell"].dtype,
            device=batch["cell"].device,
        )
        with torch.no_grad():
            batch["positions"][: self._real_atom_count].copy_(position_tensor)
            batch["cell"][:3].copy_(cell_tensor)
            batch["shifts"][: self._real_edge_count].copy_(
                torch.matmul(
                    batch["unit_shifts"][: self._real_edge_count], cell_tensor
                )
            )
        return batch

    def _canonical_positions(
        self, positions_A: np.ndarray, cell_A: np.ndarray
    ) -> np.ndarray:
        reference_scaled_positions = self._reference_scaled_positions
        reference_pbc = self._reference_pbc
        if reference_scaled_positions is None or reference_pbc is None:
            raise RuntimeError(
                "Cannot canonicalize periodic positions without a cached reference graph."
            )
        try:
            current_scaled_positions = np.linalg.solve(
                cell_A.T, positions_A.T
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Cannot canonicalize positions for singular cell_A={cell_A.tolist()}."
            ) from exc
        image_offsets = np.zeros_like(current_scaled_positions)
        image_offsets[:, reference_pbc] = np.rint(
            current_scaled_positions[:, reference_pbc]
            - reference_scaled_positions[:, reference_pbc]
        )
        return positions_A - image_offsets @ cell_A

    def _graph_is_valid(self, positions_A: np.ndarray, cell_A: np.ndarray) -> bool:
        reference_cell_A = self._reference_cell_A
        reference_scaled_positions = self._reference_scaled_positions
        if reference_cell_A is None or reference_scaled_positions is None:
            return False
        if positions_A.shape != reference_scaled_positions.shape:
            return False

        try:
            deformation = np.linalg.solve(reference_cell_A, cell_A)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Cannot validate the cached neighbor graph because its reference cell is "
                f"singular: reference_cell_A={reference_cell_A.tolist()}."
            ) from exc
        stretches = np.linalg.svd(deformation, compute_uv=False)
        minimum_stretch = float(stretches[-1])
        if not math.isfinite(minimum_stretch) or minimum_stretch <= 0.0:
            raise ValueError(
                "Cannot validate the cached neighbor graph for a singular/non-finite cell "
                f"deformation: singular_values={stretches.tolist()}, "
                f"cell_A={cell_A.tolist()}."
            )
        canonical_positions_A = self._canonical_positions(positions_A, cell_A)
        affine_reference_positions_A = reference_scaled_positions @ cell_A
        maximum_nonaffine_displacement_A = float(
            np.linalg.norm(
                canonical_positions_A - affine_reference_positions_A, axis=1
            ).max()
        )
        shortest_omitted_distance_A = minimum_stretch * (
            self.r_max + self.neighbor_skin_A
        ) - 2.0 * maximum_nonaffine_displacement_A
        return shortest_omitted_distance_A > self.r_max

    def _refill_compiled_graph(
        self,
        atoms: Atoms,
        positions_A: np.ndarray,
        cell_A: np.ndarray,
    ) -> Any:
        """Refill an existing fixed-shape GPU batch after a neighbor rebuild."""

        batch = self._cached_batch
        reference_atomic_numbers = self._reference_atomic_numbers
        if batch is None or reference_atomic_numbers is None:
            raise RuntimeError(
                "Compiled graph refill requires an initialized batch and fixed "
                "composition reference."
            )
        atomic_numbers = np.asarray(atoms.numbers, dtype=np.int64)
        if not np.array_equal(atomic_numbers, reference_atomic_numbers):
            if atomic_numbers.shape == reference_atomic_numbers.shape:
                changed_indices = np.flatnonzero(
                    atomic_numbers != reference_atomic_numbers
                )[:20].tolist()
            else:
                changed_indices = []
            raise RuntimeError(
                "Compiled MACE graph buffers cannot be reused after composition or atom "
                "ordering changes. The repository workload has fixed composition; got "
                f"reference_shape={reference_atomic_numbers.shape}, "
                f"current_shape={atomic_numbers.shape}, first_changed_indices="
                f"{changed_indices}."
            )

        edge_index, shifts_A, unit_shifts, _neighbor_cell_A = (
            mace_data.get_neighborhood(
                positions=positions_A,
                cutoff=self.r_max + self.neighbor_skin_A,
                pbc=tuple(bool(value) for value in atoms.pbc),
                cell=cell_A.copy(),
            )
        )
        real_atom_count = len(atomic_numbers)
        real_edge_count = int(edge_index.shape[1])
        if real_atom_count != self._real_atom_count:
            raise RuntimeError(
                "Compiled MACE graph refill changed the fixed real atom count: "
                f"cached={self._real_atom_count}, current={real_atom_count}."
            )
        if real_edge_count >= self.pad_num_edges:
            raise RuntimeError(
                "Compiled MACE fixed edge budget was exhausted while refilling the "
                f"r_max+skin graph: real_edge_count={real_edge_count}, "
                f"pad_num_edges={self.pad_num_edges}, r_max_A={self.r_max}, "
                f"neighbor_skin_A={self.neighbor_skin_A}. Increase the explicit edge "
                "budget and restart from the last checkpoint; this calculator will not "
                "recompile silently."
            )

        position_tensor = torch.as_tensor(
            positions_A, dtype=batch["positions"].dtype
        )
        cell_tensor = torch.as_tensor(cell_A, dtype=batch["cell"].dtype)
        edge_index_tensor = torch.as_tensor(
            edge_index, dtype=batch["edge_index"].dtype
        )
        unit_shift_tensor = torch.as_tensor(
            unit_shifts, dtype=batch["unit_shifts"].dtype
        )
        shift_tensor = torch.as_tensor(shifts_A, dtype=batch["shifts"].dtype)
        volume_tensor = torch.linalg.det(cell_tensor)
        reciprocal_cell_tensor = 2.0 * torch.pi * torch.linalg.inv(cell_tensor.mT)
        pbc_tensor = torch.as_tensor(
            np.asarray(atoms.pbc, dtype=bool), dtype=batch["pbc"].dtype
        )
        fake_atom_count = self.pad_num_atoms - real_atom_count
        if fake_atom_count == 0:
            fake_atom_count = 1
        fake_atom_index = real_atom_count + fake_atom_count - 1
        fake_edge_start = real_edge_count
        fake_cell_scale = max(float(self.r_max) * 2.0, 1.0)

        with torch.no_grad():
            batch["positions"][:real_atom_count].copy_(position_tensor)
            batch["cell"][:3].copy_(cell_tensor)
            batch["rcell"][:3].copy_(reciprocal_cell_tensor)
            batch["rcell"][3:6].copy_(reciprocal_cell_tensor)
            batch["volume"].fill_(float(volume_tensor.item()))
            batch["pbc"][0].copy_(pbc_tensor)
            batch["edge_index"][:, :real_edge_count].copy_(edge_index_tensor)
            batch["edge_index"][:, fake_edge_start:].fill_(fake_atom_index)
            batch["unit_shifts"][:real_edge_count].copy_(unit_shift_tensor)
            batch["unit_shifts"][fake_edge_start:].zero_()
            batch["unit_shifts"][fake_edge_start:, 0].fill_(1.0)
            batch["shifts"][:real_edge_count].copy_(shift_tensor)
            batch["shifts"][fake_edge_start:].zero_()
            batch["shifts"][fake_edge_start:, 0].fill_(fake_cell_scale)

        self._reference_cell_A = cell_A.copy()
        try:
            self._reference_scaled_positions = np.linalg.solve(
                cell_A.T,
                positions_A.T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Cannot cache a neighbor graph for singular cell_A={cell_A.tolist()}."
            ) from exc
        self._reference_pbc = np.asarray(atoms.pbc, dtype=bool).copy()
        self._real_edge_count = real_edge_count
        self._maximum_real_edge_count = max(
            self._maximum_real_edge_count, real_edge_count
        )
        self.graph_rebuild_count += 1
        self.compiled_graph_refill_count += 1
        return batch

    def _rebuild_graph(
        self,
        atoms: Atoms,
        positions_A: np.ndarray,
        cell_A: np.ndarray,
    ):
        if (
            self.use_compile
            and self._cached_batch is not None
            and bool(np.all(atoms.pbc))
        ):
            return self._refill_compiled_graph(atoms, positions_A, cell_A)
        self.arrays_keys.update({self.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(
            info_keys=self.info_keys,
            arrays_keys=self.arrays_keys,
        )
        with torch_tools.default_dtype(self.default_dtype):
            configuration = mace_data.config_from_atoms(
                atoms,
                key_specification=keyspec,
                head_name=self.head,
            )
            atomic_data = mace_data.AtomicData.from_config(
                configuration,
                z_table=self.z_table,
                cutoff=self.r_max + self.neighbor_skin_A,
                heads=self.available_heads,
            )

        real_atom_count = int(atomic_data["node_attrs"].shape[0])
        real_edge_count = int(atomic_data["edge_index"].shape[1])
        data_list = [atomic_data]
        if self.use_compile:
            if real_atom_count > self.pad_num_atoms:
                raise RuntimeError(
                    "Compiled MACE fixed atom budget was exceeded while rebuilding the "
                    f"neighbor graph: real_atom_count={real_atom_count}, "
                    f"pad_num_atoms={self.pad_num_atoms}. Use the exact repository-produced "
                    "atom count in a dedicated workload config."
                )
            if real_edge_count >= self.pad_num_edges:
                raise RuntimeError(
                    "Compiled MACE fixed edge budget was exhausted while rebuilding the "
                    f"r_max+skin graph: real_edge_count={real_edge_count}, "
                    f"pad_num_edges={self.pad_num_edges}, r_max_A={self.r_max}, "
                    f"neighbor_skin_A={self.neighbor_skin_A}. Increase the explicit edge "
                    "budget and restart; this calculator will not recompile silently."
                )
            fake_atom_count = self.pad_num_atoms - real_atom_count
            if fake_atom_count == 0:
                fake_atom_count = 1
            data_list.append(
                self._build_padding_graph(
                    atomic_data,
                    fake_atom_count=fake_atom_count,
                    fake_edge_count=self.pad_num_edges - real_edge_count,
                )
            )

        batch = torch_geometric.Batch.from_data_list(data_list).to(self.device)
        model_dtype = next(self.models[0].parameters()).dtype
        mismatched_dtypes = {
            key: str(value.dtype)
            for key, value in batch.to_dict().items()
            if torch.is_tensor(value)
            and torch.is_floating_point(value)
            and value.dtype != model_dtype
        }
        if mismatched_dtypes:
            raise TypeError(
                "MACE graph tensors do not match the loaded model dtype; rebuilding/casting "
                f"inside every MD step is forbidden. model_dtype={model_dtype}, "
                f"mismatched_tensors={mismatched_dtypes}."
            )
        batch["positions"].requires_grad_(True)
        if self.use_compile:
            batch["node_attrs"].requires_grad_(True)

        self._cached_batch = batch
        self._reference_cell_A = cell_A.copy()
        try:
            self._reference_scaled_positions = np.linalg.solve(
                cell_A.T,
                positions_A.T,
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Cannot cache a neighbor graph for singular cell_A={cell_A.tolist()}."
            ) from exc
        self._reference_pbc = np.asarray(atoms.pbc, dtype=bool).copy()
        self._reference_atomic_numbers = np.asarray(
            atoms.numbers, dtype=np.int64
        ).copy()
        self._real_atom_count = real_atom_count
        self._real_edge_count = real_edge_count
        self._maximum_real_edge_count = max(
            self._maximum_real_edge_count, real_edge_count
        )
        self.graph_rebuild_count += 1
        return batch

    def _effective_properties(self, properties: list[str] | None) -> set[str]:
        requested = {"energy"} if properties is None else set(properties)
        unsupported = requested - SUPPORTED_FAST_PROPERTIES
        if unsupported:
            raise PropertyNotImplementedError(
                "VerletSkinMACECalculator fast path does not implement requested "
                f"properties={sorted(unsupported)}; supported properties are "
                f"{sorted(SUPPORTED_FAST_PROPERTIES)}."
            )
        effective = set(requested)
        if self.md_property_mode == "forces_stress" and requested.intersection(
            {"forces", "stress"}
        ):
            effective.update({"forces", "stress"})
        if requested.intersection({"energy", "free_energy"}):
            effective.update({"energy", "free_energy"})
        if requested.intersection({"energies", "node_energy"}):
            effective.update({"energies", "node_energy"})
        return effective

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if atoms is None:
            if self.atoms is None:
                raise ValueError("MACE calculate() requires an Atoms object.")
            atoms = self.atoms
        effective = self._effective_properties(properties)
        preserved_results = dict(self.results) if not system_changes else {}
        Calculator.calculate(self, atoms)
        missing = effective - set(preserved_results)
        if not missing:
            self.results = preserved_results
            return

        batch = self._atoms_to_batch(atoms)
        model = self.models[0]
        compute_stress = "stress" in missing
        compute_force = "forces" in missing
        oeq_compile = self.use_compile and self._enable_oeq
        batch_dict = batch.to_dict()
        if oeq_compile and compute_stress:
            positions = batch_dict["positions"]
            graph_count = int(batch_dict["ptr"].numel() - 1)
            displacement = torch.zeros(
                (graph_count, 3, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
            batch_dict["displacement"] = displacement + positions.sum() * 0.0

        output = model(
            batch_dict,
            compute_force=compute_force,
            compute_stress=compute_stress,
            training=self.use_compile and not oeq_compile,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )
        self.model_evaluation_count += 1
        if compute_force:
            self.force_evaluation_count += 1
        if compute_stress:
            self.stress_evaluation_count += 1

        results = preserved_results
        energy_conversion = self.energy_units_to_eV
        force_conversion = self.energy_units_to_eV / self.length_units_to_A
        stress_conversion = self.energy_units_to_eV / self.length_units_to_A**3

        # MACE necessarily forms total energy before differentiating forces/stress.
        # Copying this one scalar avoids a second full model call when the sparse
        # thermodynamic recorder requests potential energy at the same state.
        if missing.intersection({"energy", "free_energy", "forces", "stress"}):
            energy = output.get("energy")
            if energy is None:
                raise RuntimeError(
                    "MACE model returned no energy for an energy/force/stress evaluation."
                )
            if energy.ndim != 1 or energy.shape[0] < 1:
                raise RuntimeError(
                    f"MACE energy output must have shape=(graphs,), got {tuple(energy.shape)}."
                )
            energy_eV = float(energy[0].detach().cpu().item() * energy_conversion)
            results["energy"] = energy_eV
            results["free_energy"] = energy_eV

        if missing.intersection({"energies", "node_energy"}):
            node_energy = output.get("node_energy")
            if node_energy is None:
                raise RuntimeError(
                    "MACE model returned no node_energy for a per-atom energy request."
                )
            node_energy = node_energy[: self._real_atom_count]
            total_node_energy = node_energy.detach().cpu().numpy() * energy_conversion
            node_heads = batch["head"][batch["batch"]][: self._real_atom_count]
            atom_indices = torch.arange(
                self._real_atom_count,
                device=batch["node_attrs"].device,
            )
            node_e0 = (
                model.atomic_energies_fn(batch["node_attrs"][: self._real_atom_count])[
                    atom_indices, node_heads
                ]
                .detach()
                .cpu()
                .numpy()
                * energy_conversion
            )
            results["energies"] = total_node_energy
            results["node_energy"] = total_node_energy - node_e0

        if "forces" in missing:
            forces = output.get("forces")
            if forces is None:
                raise RuntimeError("MACE model returned no forces for a force request.")
            expected_force_shape = (self._real_atom_count, 3)
            real_forces = forces[: self._real_atom_count]
            if tuple(real_forces.shape) != expected_force_shape:
                raise RuntimeError(
                    f"MACE force output has real shape={tuple(real_forces.shape)}, "
                    f"expected={expected_force_shape}."
                )
            results["forces"] = (
                real_forces.detach().cpu().numpy() * force_conversion
            )

        if "stress" in missing:
            stress = output.get("stress")
            if stress is None:
                raise RuntimeError("MACE model returned no stress for an NPT stress request.")
            if stress.ndim == 3:
                stress = stress[0]
            if tuple(stress.shape) != (3, 3):
                raise RuntimeError(
                    f"MACE stress output must have real shape=(3, 3), got "
                    f"shape={tuple(stress.shape)}."
                )
            stress_matrix = stress.detach().cpu().numpy() * stress_conversion
            results["stress"] = full_3x3_to_voigt_6_stress(stress_matrix)

        self.results = results
