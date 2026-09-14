"""Production Langevin-NVT engine and PTM cluster CV for jumpy FFS."""

from __future__ import annotations

import copy
from typing import Any

import ase
import numpy as np
from ase import Atoms, units
from ase.constraints import FixCom
from ase.md.langevin import Langevin

from .jumpy_ffs import DynamicalState, require_branchable_integrator
from .simulation import set_maxwell_boltzmann_velocities
from .transition_analysis import CRYSTALLINE_STRUCTURE_TYPES


class LangevinNVTShotEngine:
    """Fixed-cell stochastic shot engine with a complete Markov-state restart.

    A new ASE ``Langevin`` object is constructed for every cadence segment.
    This propagator has no persistent thermostat variables: its dynamical state is
    positions, momenta, the fixed cell, and the NumPy random stream.  The complete
    contract is stored with every state. CUDA/CuEq force kernels are not claimed to
    reproduce a discarded trajectory bit-for-bit after a process restart; jFFS recovery
    discards all uncommitted observations and continues from the exact committed state.
    """

    def __init__(
        self,
        calculator: object,
        *,
        execution_provenance: dict[str, Any],
        temperature_K: float,
        timestep_fs: float,
        friction_time_fs: float,
    ) -> None:
        for name, value in (
            ("temperature_K", temperature_K),
            ("timestep_fs", timestep_fs),
            ("friction_time_fs", friction_time_fs),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}.")
        if not isinstance(execution_provenance, dict) or not execution_provenance:
            raise TypeError(
                "execution_provenance must be the non-empty repository provenance mapping; "
                "jFFS restarts may not silently change Hamiltonian or calculator runtime."
            )
        self.calculator = calculator
        self.temperature_K = float(temperature_K)
        self._timestep_fs = float(timestep_fs)
        self.friction_time_fs = float(friction_time_fs)
        self._contract = {
            "ensemble": "langevin_nvt",
            "fixed_cell": True,
            "integrator": "ase.md.langevin.Langevin",
            "ase_version": ase.__version__,
            "ase_langevin_algorithm_version": int(Langevin._lgv_version),
            "temperature_K": self.temperature_K,
            "timestep_fs": self._timestep_fs,
            "friction_time_fs": self.friction_time_fs,
            "center_of_mass_constraint": "ase.constraints.FixCom",
            "langevin_fixcm_argument": False,
            "rng": "numpy.random.PCG64",
            "complete_restart_state": True,
            "restart_semantics": "exact_committed_markov_state",
            "bitwise_discarded_path_replay_required": False,
            "bitwise_gpu_force_replay_claimed": False,
            "extended_integrator_state_serialized": False,
            "execution_provenance": copy.deepcopy(execution_provenance),
        }

    @property
    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._contract)

    @property
    def timestep_fs(self) -> float:
        return self._timestep_fs

    def initialize_state(
        self,
        atoms: Atoms,
        *,
        random_seed: int,
    ) -> DynamicalState:
        if len(atoms) == 0:
            raise ValueError("Cannot initialize a jFFS engine state with zero atoms.")
        if not bool(np.all(atoms.pbc)):
            raise ValueError(
                f"jFFS homogeneous crystallization requires pbc=[true,true,true], got "
                f"{atoms.pbc.tolist()}."
            )
        rng = np.random.default_rng(random_seed)
        initialized = atoms.copy()
        set_maxwell_boltzmann_velocities(initialized, self.temperature_K, rng)
        initialized.set_constraint(FixCom())
        initialized.set_momenta(initialized.get_momenta(), apply_constraint=True)
        initialized.set_constraint()
        state = self._state_from_atoms(initialized, step=0, rng=rng)
        state.validate(context="initialized Langevin-NVT state")
        return state

    def branch(
        self, state: DynamicalState, *, random_seed: int
    ) -> DynamicalState:
        self._validate_state_contract(state, context="jFFS parent before branching")
        rng = np.random.default_rng(random_seed)
        branched = DynamicalState(
            positions_A=np.asarray(state.positions_A).copy(),
            momenta=np.asarray(state.momenta).copy(),
            cell_A=np.asarray(state.cell_A).copy(),
            atomic_numbers=np.asarray(state.atomic_numbers).copy(),
            masses=np.asarray(state.masses).copy(),
            pbc=np.asarray(state.pbc).copy(),
            step=state.step,
            rng_state=copy.deepcopy(rng.bit_generator.state),
            integrator_contract=self.contract,
            extended_integrator_state=None,
        )
        branched.validate(context="jFFS child immediately after branching")
        return branched

    def advance(self, state: DynamicalState, *, steps: int) -> DynamicalState:
        self._validate_state_contract(state, context="Langevin-NVT state before advance")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError(f"Langevin-NVT advance steps must be positive, got {steps!r}.")
        rng = self._rng_from_state(state.rng_state)
        atoms = Atoms(
            numbers=np.asarray(state.atomic_numbers, dtype=np.int32),
            positions=np.asarray(state.positions_A, dtype=np.float64),
            cell=np.asarray(state.cell_A, dtype=np.float64),
            pbc=np.asarray(state.pbc, dtype=bool),
        )
        atoms.set_masses(np.asarray(state.masses, dtype=np.float64))
        atoms.set_momenta(np.asarray(state.momenta, dtype=np.float64))
        atoms.calc = self.calculator
        atoms.set_constraint(FixCom())
        dynamics = Langevin(
            atoms,
            timestep=self._timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            friction=1.0 / (self.friction_time_fs * units.fs),
            fixcm=False,
            rng=rng,
        )
        dynamics.run(steps)
        atoms.set_constraint()
        result = self._state_from_atoms(
            atoms,
            step=state.step + steps,
            rng=rng,
        )
        if not np.array_equal(result.cell_A, state.cell_A):
            raise RuntimeError(
                "Langevin-NVT shot changed the simulation cell. The jFFS rate uses one "
                "fixed volume and cannot mix cell states."
            )
        result.validate(context="Langevin-NVT state after advance")
        return result

    def _state_from_atoms(
        self,
        atoms: Atoms,
        *,
        step: int,
        rng: np.random.Generator,
    ) -> DynamicalState:
        momenta = atoms.get_momenta()
        if momenta is None:
            raise RuntimeError(
                "Langevin-NVT restart state is missing momenta; initialize velocities "
                "before constructing a shot state."
            )
        return DynamicalState(
            positions_A=np.asarray(atoms.positions, dtype=np.float64).copy(),
            momenta=np.asarray(momenta, dtype=np.float64).copy(),
            cell_A=np.asarray(atoms.cell.array, dtype=np.float64).copy(),
            atomic_numbers=np.asarray(atoms.numbers, dtype=np.int32).copy(),
            masses=np.asarray(atoms.get_masses(), dtype=np.float64).copy(),
            pbc=np.asarray(atoms.pbc, dtype=np.bool_).copy(),
            step=step,
            rng_state=copy.deepcopy(rng.bit_generator.state),
            integrator_contract=self.contract,
            extended_integrator_state=None,
        )

    def _validate_state_contract(self, state: DynamicalState, *, context: str) -> None:
        state.validate(context=context)
        if state.integrator_contract != self._contract:
            raise RuntimeError(
                f"{context}: stored integrator contract differs from the active "
                "Langevin-NVT engine. A branch may not change Hamiltonian, ASE algorithm, "
                "temperature, time step, friction, fixed cell, or constraints."
            )
        require_branchable_integrator(state.integrator_contract, context=context)

    @staticmethod
    def _rng_from_state(state: dict[str, Any]) -> np.random.Generator:
        if state.get("bit_generator") != "PCG64":
            raise RuntimeError(
                "Langevin-NVT restart requires a NumPy PCG64 state, got "
                f"{state.get('bit_generator')!r}."
            )
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = copy.deepcopy(state)
        return rng


class PTMLargestCrystallineClusterCV:
    """Integer size of the largest connected PTM FCC/HCP/BCC cluster."""

    def __init__(self, *, ptm_rmsd_cutoff: float, cluster_cutoff_A: float) -> None:
        if not np.isfinite(ptm_rmsd_cutoff) or not 0.0 < ptm_rmsd_cutoff <= 1.0:
            raise ValueError(
                "ptm_rmsd_cutoff must be a normalized value in (0, 1], got "
                f"{ptm_rmsd_cutoff}."
            )
        if not np.isfinite(cluster_cutoff_A) or cluster_cutoff_A <= 0.0:
            raise ValueError(
                f"cluster_cutoff_A must be finite and > 0, got {cluster_cutoff_A}."
            )
        try:
            import ovito
            from ovito.modifiers import (
                ClusterAnalysisModifier,
                PolyhedralTemplateMatchingModifier,
            )
        except ImportError as exc:
            raise ImportError(
                "Online jFFS cluster evaluation requires OVITO PTM. Install the "
                "repository requirements in the pointnet environment."
            ) from exc
        self.ptm_rmsd_cutoff = float(ptm_rmsd_cutoff)
        self.cluster_cutoff_A = float(cluster_cutoff_A)
        self._ptm = PolyhedralTemplateMatchingModifier()
        self._ptm.rmsd_cutoff = self.ptm_rmsd_cutoff
        self._clusters = ClusterAnalysisModifier(
            cutoff=self.cluster_cutoff_A,
            only_selected=True,
            sort_by_size=True,
        )
        self._contract = {
            "name": "largest_connected_ptm_crystalline_cluster_atoms",
            "value_type": "nonnegative_integer_atom_count",
            "crystalline_structure_types": [
                int(value) for value in CRYSTALLINE_STRUCTURE_TYPES
            ],
            "ptm_rmsd_cutoff": self.ptm_rmsd_cutoff,
            "cluster_cutoff_A": self.cluster_cutoff_A,
            "ovito_version": ovito.version_string,
            "sampling_semantics": (
                "first observation at each fixed CV cadence; jumps land at the highest "
                "configured interface not exceeding the observed integer cluster size"
            ),
        }

    @property
    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._contract)

    def evaluate(self, state: DynamicalState) -> int:
        state.validate(context="state supplied to PTM largest-cluster CV")
        try:
            from ovito.io.ase import ase_to_ovito
        except ImportError as exc:
            raise ImportError(
                "Online jFFS cluster evaluation requires OVITO PTM and cluster analysis."
            ) from exc
        atoms = Atoms(
            numbers=state.atomic_numbers,
            positions=state.positions_A,
            cell=state.cell_A,
            pbc=state.pbc,
        )
        data = ase_to_ovito(atoms)
        data.apply(self._ptm)
        structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
        data.apply(self._clusters)
        largest = int(data.attributes["ClusterAnalysis.largest_size"])
        if largest < 0 or largest > len(state.atomic_numbers):
            raise RuntimeError(
                f"OVITO returned invalid largest crystalline cluster size={largest} for "
                f"atom_count={len(state.atomic_numbers)}."
            )
        return largest
