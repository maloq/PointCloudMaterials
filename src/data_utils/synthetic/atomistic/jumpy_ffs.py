"""Restartable jumpy forward-flux sampling for crystallization.

The implementation uses a weighted branching tree rather than a product of
adjacent-interface probabilities.  That distinction is essential for an integer
largest-cluster order parameter: one observation can jump across several
interfaces.  A child receives ``parent_weight / trials_per_state`` and is queued
at the interface on which it actually lands.  The total weight reaching the last
interface is therefore the complete A-to-B crossing probability, including all
skipped-interface paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


JUMPY_FFS_SCHEMA_VERSION = 5


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True)
class DynamicalState:
    """Complete phase-space and stochastic state at an MD step boundary.

    ASE's Langevin propagator has no extended dynamical variables between step
    calls.  Positions, momenta, cell, species/masses, constraints specified by
    the integrator contract, and the bit-generator state are consequently a
    complete restart.  Barostat/thermostat-chain state is deliberately not
    represented here; MTK-NPT is rejected unless a different engine supplies a
    contract declaring that its extended state is serialized.
    """

    positions_A: np.ndarray
    momenta: np.ndarray
    cell_A: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    pbc: np.ndarray
    step: int
    rng_state: dict[str, Any]
    integrator_contract: dict[str, Any]
    extended_integrator_state: dict[str, Any] | None = None

    def validate(self, *, context: str) -> None:
        atom_count = len(self.atomic_numbers)
        expected_shapes = {
            "positions_A": (atom_count, 3),
            "momenta": (atom_count, 3),
            "cell_A": (3, 3),
            "atomic_numbers": (atom_count,),
            "masses": (atom_count,),
            "pbc": (3,),
        }
        for name, expected_shape in expected_shapes.items():
            values = np.asarray(getattr(self, name))
            if values.shape != expected_shape:
                raise ValueError(
                    f"{context}: {name} has shape={values.shape}, expected "
                    f"shape={expected_shape}."
                )
        for name in ("positions_A", "momenta", "cell_A", "masses"):
            dtype = np.asarray(getattr(self, name)).dtype
            if not np.issubdtype(dtype, np.floating):
                raise TypeError(
                    f"{context}: {name} must have a floating dtype, got {dtype}."
                )
        if not np.issubdtype(np.asarray(self.atomic_numbers).dtype, np.integer):
            raise TypeError(
                f"{context}: atomic_numbers must have an integer dtype, got "
                f"{np.asarray(self.atomic_numbers).dtype}."
            )
        if np.asarray(self.pbc).dtype != np.dtype(np.bool_):
            raise TypeError(
                f"{context}: pbc must have dtype=bool, got "
                f"{np.asarray(self.pbc).dtype}."
            )
        if atom_count == 0:
            raise ValueError(f"{context}: a dynamical state must contain atoms.")
        for name in ("positions_A", "momenta", "cell_A", "masses"):
            values = np.asarray(getattr(self, name))
            if not np.isfinite(values).all():
                indices = np.argwhere(~np.isfinite(values))[:10].tolist()
                raise FloatingPointError(
                    f"{context}: {name} contains non-finite values at indices={indices}."
                )
        if np.any(np.asarray(self.atomic_numbers) <= 0):
            raise ValueError(
                f"{context}: atomic_numbers must be positive, got "
                f"{np.asarray(self.atomic_numbers).tolist()}."
            )
        if np.any(np.asarray(self.masses) <= 0.0):
            raise ValueError(f"{context}: all atomic masses must be positive.")
        if not bool(np.all(np.asarray(self.pbc, dtype=bool))):
            raise ValueError(
                f"{context}: homogeneous nucleation requires a fully periodic fixed cell, "
                f"got pbc={np.asarray(self.pbc).tolist()}."
            )
        volume_A3 = float(np.linalg.det(np.asarray(self.cell_A, dtype=np.float64)))
        if not math.isfinite(volume_A3) or volume_A3 <= 0.0:
            raise ValueError(
                f"{context}: cell determinant must be finite and positive, got "
                f"{volume_A3} A^3."
            )
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError(
                f"{context}: step must be a nonnegative integer, got {self.step!r}."
            )
        if not isinstance(self.rng_state, dict) or "bit_generator" not in self.rng_state:
            raise TypeError(
                f"{context}: rng_state must be a complete NumPy bit-generator state "
                "mapping containing 'bit_generator'."
            )
        require_branchable_integrator(self.integrator_contract, context=context)
        if self.integrator_contract.get("ensemble") in {
            "mtk_npt",
            "isotropic_mtk_npt",
        }:
            if (
                not isinstance(self.extended_integrator_state, dict)
                or not self.extended_integrator_state
            ):
                raise RuntimeError(
                    f"{context}: MTK-NPT state is missing serialized thermostat-chain and "
                    "barostat variables despite its branchable contract."
                )
        elif self.extended_integrator_state is not None:
            raise RuntimeError(
                f"{context}: Langevin-NVT has no hidden extended integrator state, but "
                "extended_integrator_state was populated."
            )


def require_branchable_integrator(
    contract: dict[str, Any], *, context: str
) -> None:
    """Reject ensembles whose hidden state is absent from a branch checkpoint."""

    if not isinstance(contract, dict):
        raise TypeError(f"{context}: integrator_contract must be a mapping.")
    ensemble = contract.get("ensemble")
    if ensemble == "langevin_nvt":
        if contract.get("fixed_cell") is not True:
            raise RuntimeError(
                f"{context}: Langevin-NVT branching requires fixed_cell=true."
            )
        if contract.get("complete_restart_state") is not True:
            raise RuntimeError(
                f"{context}: Langevin-NVT engine did not certify complete_restart_state=true."
            )
        return
    if ensemble in {"mtk_npt", "isotropic_mtk_npt"}:
        if contract.get("extended_integrator_state_serialized") is not True:
            raise RuntimeError(
                f"{context}: refusing MTK-NPT branching because thermostat-chain and "
                "barostat variables are not serialized. Saving only atoms, momenta, and "
                "the cell does not define the same Markov state. Use the repository-owned "
                "Langevin-NVT shot engine or implement and validate a complete MTK state "
                "serializer first."
            )
        return
    raise RuntimeError(
        f"{context}: unsupported branching ensemble={ensemble!r}; the engine must provide "
        "an explicit, audited restart-state contract."
    )


class JumpyCollectiveVariable(Protocol):
    @property
    def contract(self) -> dict[str, Any]: ...

    def evaluate(self, state: DynamicalState) -> int: ...


class JumpyShotEngine(Protocol):
    @property
    def contract(self) -> dict[str, Any]: ...

    @property
    def timestep_fs(self) -> float: ...

    def branch(self, state: DynamicalState, *, random_seed: int) -> DynamicalState: ...

    def advance(self, state: DynamicalState, *, steps: int) -> DynamicalState: ...


@dataclass(frozen=True)
class JumpyFFSAlgorithmConfig:
    interfaces_atoms: tuple[int, ...]
    equilibration_steps: int
    equilibration_checkpoint_interval_steps: int
    basin_target_crossings: int
    basin_max_steps: int
    basin_checkpoint_interval_steps: int
    cv_interval_steps: int
    trials_per_state: int
    shot_max_steps: int
    shot_checkpoint_interval_steps: int
    bootstrap_samples: int
    bootstrap_block_crossings: int
    random_seed: int

    def validate(self) -> None:
        if len(self.interfaces_atoms) < 2:
            raise ValueError(
                "jFFS requires at least two interfaces: an A-basin boundary and a B "
                "target boundary."
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.interfaces_atoms
        ):
            raise ValueError(
                f"interfaces_atoms must be positive integers, got "
                f"{self.interfaces_atoms}."
            )
        if any(
            right <= left
            for left, right in zip(self.interfaces_atoms, self.interfaces_atoms[1:])
        ):
            raise ValueError(
                f"interfaces_atoms must be strictly increasing, got "
                f"{self.interfaces_atoms}."
            )
        positive_fields = {
            "basin_target_crossings": self.basin_target_crossings,
            "basin_max_steps": self.basin_max_steps,
            "basin_checkpoint_interval_steps": (
                self.basin_checkpoint_interval_steps
            ),
            "cv_interval_steps": self.cv_interval_steps,
            "trials_per_state": self.trials_per_state,
            "shot_max_steps": self.shot_max_steps,
            "equilibration_checkpoint_interval_steps": (
                self.equilibration_checkpoint_interval_steps
            ),
            "shot_checkpoint_interval_steps": self.shot_checkpoint_interval_steps,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_block_crossings": self.bootstrap_block_crossings,
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if (
            not isinstance(self.equilibration_steps, int)
            or isinstance(self.equilibration_steps, bool)
            or self.equilibration_steps < 0
        ):
            raise ValueError(
                f"equilibration_steps must be a nonnegative integer, got "
                f"{self.equilibration_steps!r}."
            )
        for name, steps in (
            ("equilibration_steps", self.equilibration_steps),
            (
                "equilibration_checkpoint_interval_steps",
                self.equilibration_checkpoint_interval_steps,
            ),
            ("basin_max_steps", self.basin_max_steps),
            (
                "basin_checkpoint_interval_steps",
                self.basin_checkpoint_interval_steps,
            ),
            ("shot_max_steps", self.shot_max_steps),
            ("shot_checkpoint_interval_steps", self.shot_checkpoint_interval_steps),
        ):
            if steps % self.cv_interval_steps != 0:
                raise ValueError(
                    f"{name}={steps} must be divisible by cv_interval_steps="
                    f"{self.cv_interval_steps}; endpoints must be observed exactly."
                )
        if self.bootstrap_block_crossings > self.basin_target_crossings:
            raise ValueError(
                "bootstrap_block_crossings cannot exceed basin_target_crossings; got "
                f"{self.bootstrap_block_crossings} > {self.basin_target_crossings}."
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError(
                f"random_seed must be an explicit integer, got {self.random_seed!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class JumpyFFSResult:
    basin_crossing_count: int
    basin_elapsed_time_ps: float
    basin_flux_per_ps: float
    volume_A3: float
    crossing_probability: float
    rate_per_A3_ps: float
    rate_per_m3_s: float
    confidence_level: float | None
    rate_confidence_interval_per_A3_ps: tuple[float, float] | None
    rate_bootstrap_standard_error_per_A3_ps: float | None
    uncertainty_status: str
    uncertainty_reason: str | None
    bootstrap_samples_used: int
    interface_statistics: tuple[dict[str, Any], ...]
    initial_landing_probabilities: dict[str, float]
    root_success_probabilities: tuple[float, ...]
    trial_count: int
    scientific_scope: dict[str, Any]
    source_evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _state_metadata(state: DynamicalState) -> dict[str, Any]:
    return {
        "step": state.step,
        "rng_state": state.rng_state,
        "integrator_contract": state.integrator_contract,
        "extended_integrator_state": state.extended_integrator_state,
    }


def _state_sha256(state: DynamicalState) -> str:
    """Canonical digest of complete phase space, RNG, and integrator state."""

    digest = hashlib.sha256()
    digest.update(_canonical_json(_state_metadata(state)).encode("utf-8"))
    for name in (
        "positions_A",
        "momenta",
        "cell_A",
        "atomic_numbers",
        "masses",
        "pbc",
    ):
        values = np.ascontiguousarray(getattr(state, name))
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(_canonical_json(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_fixed_cell(
    state: DynamicalState, reference_cell_A: np.ndarray, *, context: str
) -> None:
    if not np.array_equal(np.asarray(state.cell_A), reference_cell_A):
        maximum_difference_A = float(
            np.max(np.abs(np.asarray(state.cell_A) - reference_cell_A))
        )
        raise RuntimeError(
            f"{context}: fixed-volume jFFS state changed its cell; "
            f"maximum_absolute_difference_A={maximum_difference_A:.12g}."
        )


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes on the Linux research node."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class JumpyFFSArtifactStore:
    """Atomic immutable state blobs plus one atomic workflow journal."""

    def __init__(
        self,
        root: Path,
        *,
        identity: dict[str, Any],
        resume: bool,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.states_directory = self.root / "states"
        self.manifest_path = self.root / "manifest.json"
        self.journal_path = self.root / "journal.json"
        self.state_audit_path = self.root / "state_sha256_audit.log"
        expected_manifest = {
            "schema_version": JUMPY_FFS_SCHEMA_VERSION,
            "identity": _json_value(identity),
            "identity_sha256": hashlib.sha256(
                _canonical_json(identity).encode("utf-8")
            ).hexdigest(),
        }
        if self.root.exists():
            if not resume:
                raise FileExistsError(
                    f"jFFS artifact directory already exists: {self.root}. Pass resume=true "
                    "only to continue the exact same manifest-bound workflow."
                )
            if not self.manifest_path.is_file():
                raise RuntimeError(
                    f"Cannot resume {self.root}: manifest is missing at "
                    f"{self.manifest_path}."
                )
            observed_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if observed_manifest != expected_manifest:
                raise RuntimeError(
                    f"Cannot resume {self.root}: active algorithm, engine, CV, or volume "
                    "does not match the stored manifest. Start a new output directory."
                )
        else:
            if resume:
                raise FileNotFoundError(
                    f"Cannot resume jFFS because output directory does not exist: {self.root}."
                )
            self.states_directory.mkdir(parents=True)
            with self.state_audit_path.open("x", encoding="ascii") as audit_handle:
                audit_handle.flush()
                os.fsync(audit_handle.fileno())
            _fsync_directory(self.root)
            _fsync_directory(self.root.parent)
            self._write_json_atomic(self.manifest_path, expected_manifest)
        if not self.states_directory.is_dir():
            raise RuntimeError(
                f"jFFS state directory is missing or not a directory: "
                f"{self.states_directory}."
            )
        if not self.state_audit_path.is_file():
            raise RuntimeError(
                f"jFFS state SHA audit log is missing: {self.state_audit_path}."
            )
        audit_state_ids = self.state_audit_path.read_text(
            encoding="ascii"
        ).splitlines()
        invalid_audit_ids = sorted(
            state_id
            for state_id in audit_state_ids
            if len(state_id) != 64
            or any(character not in "0123456789abcdef" for character in state_id)
        )
        if invalid_audit_ids:
            raise RuntimeError(
                f"{self.state_audit_path}: invalid state SHA-256 entries="
                f"{invalid_audit_ids}."
            )
        self._audited_state_ids = set(audit_state_ids)
        unaudited_blobs = sorted(
            path.stem
            for path in self.states_directory.glob("*.npz")
            if path.stem not in self._audited_state_ids
        )
        if unaudited_blobs:
            raise RuntimeError(
                f"{self.states_directory}: state blobs are missing from the durable SHA "
                f"audit log: {unaudited_blobs}."
            )

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(value), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)

    def save_state(self, state: DynamicalState) -> str:
        state.validate(context="jFFS state before persistence")
        metadata = _state_metadata(state)
        state_id = _state_sha256(state)
        path = self.states_directory / f"{state_id}.npz"
        if not path.exists():
            temporary = path.with_name(f".{path.name}.tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    positions_A=np.asarray(state.positions_A),
                    momenta=np.asarray(state.momenta),
                    cell_A=np.asarray(state.cell_A),
                    atomic_numbers=np.asarray(state.atomic_numbers),
                    masses=np.asarray(state.masses),
                    pbc=np.asarray(state.pbc),
                    metadata_json=np.asarray(_canonical_json(metadata)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            if state_id not in self._audited_state_ids:
                with self.state_audit_path.open("a", encoding="ascii") as audit_handle:
                    audit_handle.write(f"{state_id}\n")
                    audit_handle.flush()
                    os.fsync(audit_handle.fileno())
                self._audited_state_ids.add(state_id)
            temporary.replace(path)
            _fsync_directory(self.states_directory)
        return state_id

    def load_state(self, state_id: str) -> DynamicalState:
        path = self.states_directory / f"{state_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"jFFS journal references missing dynamical state {state_id}: {path}."
            )
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"].item()))
            state = DynamicalState(
                positions_A=stored["positions_A"].copy(),
                momenta=stored["momenta"].copy(),
                cell_A=stored["cell_A"].copy(),
                atomic_numbers=stored["atomic_numbers"].copy(),
                masses=stored["masses"].copy(),
                pbc=stored["pbc"].copy(),
                step=int(metadata["step"]),
                rng_state=metadata["rng_state"],
                integrator_contract=metadata["integrator_contract"],
                extended_integrator_state=metadata["extended_integrator_state"],
            )
        state.validate(context=f"jFFS state loaded from {path}")
        observed_state_id = _state_sha256(state)
        if observed_state_id != state_id:
            raise RuntimeError(
                f"jFFS state content hash mismatch for {path}: filename={state_id}, "
                f"observed={observed_state_id}. The restart artifact is corrupt."
            )
        return state

    def save_journal(self, journal: dict[str, Any]) -> None:
        self._write_json_atomic(self.journal_path, journal)
        self.garbage_collect_states(journal)

    @staticmethod
    def _restart_state_ids(journal: dict[str, Any]) -> set[str]:
        phase = journal.get("phase")
        required: set[str] = set()
        if phase == "equilibration":
            required.add(str(journal["equilibration"]["current_state_id"]))
        elif phase == "basin":
            basin = journal["basin"]
            required.add(str(basin["current_state_id"]))
            required.update(str(item["state_id"]) for item in basin["exits"])
        elif phase == "shooting":
            shooting = journal["shooting"]
            required.update(str(item["state_id"]) for item in shooting["pending"])
            active = shooting["active_trial"]
            if active is not None:
                required.add(str(active["current_state_id"]))
                required.add(str(active["parent_state_id"]))
        elif phase == "terminal_invalid":
            required.add(str(journal["terminal_invalid"]["state_id"]))
        elif phase not in {"analysis", "complete"}:
            raise RuntimeError(
                f"Cannot determine live jFFS states for journal phase={phase!r}."
            )
        invalid = sorted(
            state_id
            for state_id in required
            if len(state_id) != 64
            or any(character not in "0123456789abcdef" for character in state_id)
        )
        if invalid:
            raise RuntimeError(
                f"jFFS journal contains invalid state SHA-256 identifiers: {invalid}."
            )
        return required

    def garbage_collect_states(self, journal: dict[str, Any]) -> None:
        """Delete only blobs unreachable from the durably committed restart journal.

        ``save_journal`` calls this strictly after atomic journal replacement. A crash
        before replacement leaves an unreferenced extra blob; a crash after replacement
        leaves both old and new blobs. Either case is safe, and the next collection
        removes the orphan. Every blob SHA is durably appended to
        ``state_sha256_audit.log`` before the blob is published, while scientific exit and
        trial records additionally retain the relevant endpoint SHA identifiers.
        """

        required = self._restart_state_ids(journal)
        missing = sorted(
            state_id
            for state_id in required
            if not (self.states_directory / f"{state_id}.npz").is_file()
        )
        if missing:
            raise RuntimeError(
                f"Committed jFFS journal references missing restart states: {missing}."
            )
        directory_changed = False
        for path in self.states_directory.glob("*.npz"):
            if path.stem not in required:
                path.unlink()
                directory_changed = True
        for temporary in self.states_directory.glob(".*.npz.tmp"):
            temporary.unlink()
            directory_changed = True
        if directory_changed:
            _fsync_directory(self.states_directory)

    def load_journal(self) -> dict[str, Any] | None:
        if not self.journal_path.exists():
            return None
        with self.journal_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"{self.journal_path}: journal root must be a mapping.")
        return value


def _landing_interface(cv: int, interfaces: tuple[int, ...]) -> int:
    if cv < interfaces[0]:
        return -1
    return int(np.searchsorted(interfaces, cv, side="right") - 1)


def _evaluate_cv(
    collective_variable: JumpyCollectiveVariable,
    state: DynamicalState,
    *,
    context: str,
) -> int:
    value = collective_variable.evaluate(state)
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(
            f"{context}: largest-cluster collective variable must return an integer atom "
            f"count, got {value!r} ({type(value).__name__})."
        )
    result = int(value)
    if result < 0 or result > len(state.atomic_numbers):
        raise ValueError(
            f"{context}: largest-cluster collective variable returned {result}, outside "
            f"[0, atom_count={len(state.atomic_numbers)}]."
        )
    return result


def _rng_from_state(state: dict[str, Any]) -> np.random.Generator:
    bit_generator_name = state.get("bit_generator")
    if bit_generator_name != "PCG64":
        raise RuntimeError(
            f"jFFS workflow RNG requires PCG64 for exact restart, got "
            f"{bit_generator_name!r}."
        )
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state
    return rng


def _initial_journal(
    *,
    initial_state_id: str,
    initial_cv: int,
    random_seed: int,
    equilibration_steps: int,
    fixed_cell_A: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 0x53484F54]))
    return {
        "schema_version": JUMPY_FFS_SCHEMA_VERSION,
        "phase": "equilibration" if equilibration_steps else "basin",
        "master_rng_state": _json_value(rng.bit_generator.state),
        "fixed_cell_A": np.asarray(fixed_cell_A, dtype=np.float64).tolist(),
        "equilibration": {
            "current_state_id": initial_state_id,
            "completed_steps": 0,
            "cv_trace": [{"step": 0, "cv": initial_cv}],
        },
        "basin": {
            "current_state_id": initial_state_id,
            "total_steps": 0,
            "armed": True,
            "last_cv": initial_cv,
            "last_crossing_step": 0,
            "exits": [],
        },
        "shooting": None,
    }


def _set_terminal_invalid(
    journal: dict[str, Any],
    *,
    origin_phase: str,
    reason_code: str,
    message: str,
    state_id: str,
    observation_step: int,
    dynamical_step: int,
    cv: int,
) -> None:
    """Bind a permanent scientific-invalidity decision to its exact endpoint."""

    journal["phase"] = "terminal_invalid"
    journal["terminal_invalid"] = {
        "origin_phase": origin_phase,
        "reason_code": reason_code,
        "message": message,
        "state_id": state_id,
        "observation_step": observation_step,
        "dynamical_step": dynamical_step,
        "cv": cv,
    }


def _advance_equilibration(
    journal: dict[str, Any],
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    engine: JumpyShotEngine,
    collective_variable: JumpyCollectiveVariable,
    store: JumpyFFSArtifactStore,
    progress: Callable[[str], None],
) -> None:
    data = journal["equilibration"]
    state = store.load_state(data["current_state_id"])
    while data["completed_steps"] < algorithm.equilibration_steps:
        previous_step = state.step
        state = engine.advance(state, steps=algorithm.cv_interval_steps)
        state.validate(context="jFFS NVT equilibration endpoint")
        if state.integrator_contract != engine.contract:
            raise RuntimeError(
                "jFFS NVT equilibration changed the active engine contract."
            )
        if state.step != previous_step + algorithm.cv_interval_steps:
            raise RuntimeError(
                f"jFFS NVT equilibration engine advanced state.step from {previous_step} "
                f"to {state.step}, expected {previous_step + algorithm.cv_interval_steps}."
            )
        _validate_fixed_cell(
            state,
            np.asarray(journal["fixed_cell_A"], dtype=np.float64),
            context="jFFS NVT equilibration endpoint",
        )
        cv = _evaluate_cv(
            collective_variable, state, context="jFFS NVT equilibration"
        )
        data["completed_steps"] += algorithm.cv_interval_steps
        data["cv_trace"].append({"step": data["completed_steps"], "cv": cv})
        reached_basin_b = cv >= algorithm.interfaces_atoms[-1]
        checkpoint_due = (
            data["completed_steps"]
            % algorithm.equilibration_checkpoint_interval_steps
            == 0
            or data["completed_steps"] == algorithm.equilibration_steps
        )
        if reached_basin_b:
            data["current_state_id"] = store.save_state(state)
            message = (
                "The trajectory reached the B interface during pre-flux equilibration "
                f"at step={data['completed_steps']}, largest_cluster_atoms={cv}. The "
                "starting liquid is not a valid metastable-basin state for this jFFS "
                "run."
            )
            _set_terminal_invalid(
                journal,
                origin_phase="equilibration",
                reason_code="reached_b_during_equilibration",
                message=message,
                state_id=data["current_state_id"],
                observation_step=int(data["completed_steps"]),
                dynamical_step=state.step,
                cv=cv,
            )
            store.save_journal(journal)
            raise RuntimeError(message)
        if checkpoint_due:
            data["current_state_id"] = store.save_state(state)
            store.save_journal(journal)
    final_cv = int(data["cv_trace"][-1]["cv"])
    if final_cv >= algorithm.interfaces_atoms[0]:
        raise RuntimeError(
            "Pre-flux equilibration did not finish inside basin A: "
            f"largest_cluster_atoms={final_cv}, basin condition is CV < "
            f"{algorithm.interfaces_atoms[0]}."
        )
    journal["basin"] = {
        "current_state_id": data["current_state_id"],
        "total_steps": 0,
        "armed": True,
        "last_cv": final_cv,
        "last_crossing_step": 0,
        "exits": [],
    }
    journal["phase"] = "basin"
    store.save_journal(journal)
    progress(
        f"jFFS: completed {algorithm.equilibration_steps} Langevin-NVT "
        "pre-flux equilibration steps"
    )


def _sample_basin_flux(
    journal: dict[str, Any],
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    engine: JumpyShotEngine,
    collective_variable: JumpyCollectiveVariable,
    store: JumpyFFSArtifactStore,
    progress: Callable[[str], None],
) -> None:
    """Measure basin exits with frequent CV checks and sparse durable states.

    The complete Langevin Markov state includes its future RNG stream.  We therefore keep
    ordinary CV endpoints in memory and commit only the configured restart cadence.
    Every outward crossing is committed immediately because that exact endpoint seeds the
    branching tree.  A crash discards all later volatile observations and restarts from
    the last journal-bound state; correctness does not require CUDA force kernels to
    reproduce the discarded numerical path bit-for-bit.
    """

    basin = journal["basin"]
    target = algorithm.basin_target_crossings
    state = store.load_state(basin["current_state_id"])
    durable_total_steps = int(basin["total_steps"])
    while len(basin["exits"]) < target:
        if basin["total_steps"] >= algorithm.basin_max_steps:
            if int(basin["total_steps"]) != durable_total_steps:
                basin["current_state_id"] = store.save_state(state)
                store.save_journal(journal)
                durable_total_steps = int(basin["total_steps"])
            raise RuntimeError(
                f"Basin flux sampling reached basin_max_steps="
                f"{algorithm.basin_max_steps} with only {len(basin['exits'])}/{target} "
                "outward crossings. Increase the explicitly reported observation time or "
                "move the first interface after inspecting the basin CV distribution."
            )
        previous_step = state.step
        state = engine.advance(state, steps=algorithm.cv_interval_steps)
        state.validate(context="jFFS basin-flux endpoint")
        if state.integrator_contract != engine.contract:
            raise RuntimeError("jFFS basin propagation changed the active engine contract.")
        if state.step != previous_step + algorithm.cv_interval_steps:
            raise RuntimeError(
                f"jFFS basin engine advanced state.step from {previous_step} to "
                f"{state.step}, expected {previous_step + algorithm.cv_interval_steps}."
            )
        _validate_fixed_cell(
            state,
            np.asarray(journal["fixed_cell_A"], dtype=np.float64),
            context="jFFS basin-flux endpoint",
        )
        basin["total_steps"] += algorithm.cv_interval_steps
        cv = _evaluate_cv(collective_variable, state, context="jFFS basin flux")
        basin["last_cv"] = cv
        crossing_committed = False
        if basin["armed"] and cv >= algorithm.interfaces_atoms[0]:
            landing = _landing_interface(cv, algorithm.interfaces_atoms)
            crossing_step = int(basin["total_steps"])
            interval_steps = crossing_step - int(basin["last_crossing_step"])
            if interval_steps <= 0:
                raise RuntimeError(
                    f"Invalid basin crossing interval={interval_steps} at "
                    f"step={crossing_step}."
                )
            exit_index = len(basin["exits"])
            crossing_state_id = store.save_state(state)
            basin["current_state_id"] = crossing_state_id
            basin["exits"].append(
                {
                    "root_id": f"exit_{exit_index:06d}",
                    "state_id": crossing_state_id,
                    "crossing_step": crossing_step,
                    "interval_steps": interval_steps,
                    "cv": cv,
                    "landing_interface": landing,
                }
            )
            basin["last_crossing_step"] = crossing_step
            basin["armed"] = False
            reached_basin_b_early = (
                landing == len(algorithm.interfaces_atoms) - 1
                and len(basin["exits"]) < target
            )
            terminal_message = None
            if reached_basin_b_early:
                terminal_message = (
                    "The continuous basin trajectory reached B before collecting the "
                    f"requested flux sample ({len(basin['exits'])}/{target} crossings). "
                    "It cannot be continued as metastable liquid without an explicit, "
                    "statistically justified reinjection protocol."
                )
                _set_terminal_invalid(
                    journal,
                    origin_phase="basin",
                    reason_code="reached_b_before_basin_flux_target",
                    message=terminal_message,
                    state_id=crossing_state_id,
                    observation_step=crossing_step,
                    dynamical_step=state.step,
                    cv=cv,
                )
            store.save_journal(journal)
            durable_total_steps = crossing_step
            crossing_committed = True
            progress(
                f"jFFS basin exit {exit_index + 1}/{target}: CV={cv}, "
                f"landing_interface={landing}, step={crossing_step}"
            )
            if terminal_message is not None:
                raise RuntimeError(terminal_message)
        elif not basin["armed"] and cv < algorithm.interfaces_atoms[0]:
            basin["armed"] = True
        if (
            not crossing_committed
            and basin["total_steps"]
            % algorithm.basin_checkpoint_interval_steps
            == 0
        ):
            basin["current_state_id"] = store.save_state(state)
            store.save_journal(journal)
            durable_total_steps = int(basin["total_steps"])

    crossing_count = len(basin["exits"])
    root_weight = 1.0 / crossing_count
    final_interface = len(algorithm.interfaces_atoms) - 1
    pending: list[dict[str, Any]] = []
    terminal_by_root: dict[str, float] = {
        str(item["root_id"]): 0.0 for item in basin["exits"]
    }
    for item in basin["exits"]:
        item["weight"] = root_weight
        if item["landing_interface"] == final_interface:
            terminal_by_root[item["root_id"]] += root_weight
        else:
            pending.append(
                {
                    "state_id": item["state_id"],
                    "source_interface": item["landing_interface"],
                    "weight": root_weight,
                    "root_id": item["root_id"],
                    "source_cv": item["cv"],
                    "next_trial": 0,
                }
            )
    journal["shooting"] = {
        "pending": pending,
        "active_trial": None,
        "trial_records": [],
        "terminal_success_weight_by_root": terminal_by_root,
    }
    journal["phase"] = "shooting"
    store.save_journal(journal)


def _start_trial(
    journal: dict[str, Any],
    *,
    engine: JumpyShotEngine,
    collective_variable: JumpyCollectiveVariable,
    store: JumpyFFSArtifactStore,
) -> DynamicalState:
    shooting = journal["shooting"]
    parent = shooting["pending"][0]
    rng = _rng_from_state(journal["master_rng_state"])
    random_seed = int(rng.integers(0, np.iinfo(np.int64).max, endpoint=False))
    journal["master_rng_state"] = _json_value(rng.bit_generator.state)
    parent_state = store.load_state(parent["state_id"])
    state = engine.branch(parent_state, random_seed=random_seed)
    state.validate(context="new jFFS branch state")
    if state.integrator_contract != engine.contract:
        raise RuntimeError(
            "New jFFS branch state integrator contract does not match the active engine."
        )
    _validate_fixed_cell(
        state,
        np.asarray(journal["fixed_cell_A"], dtype=np.float64),
        context="new jFFS branch state",
    )
    unchanged_fields = (
        "positions_A",
        "momenta",
        "cell_A",
        "atomic_numbers",
        "masses",
        "pbc",
    )
    changed_fields = [
        name
        for name in unchanged_fields
        if not np.array_equal(getattr(state, name), getattr(parent_state, name))
    ]
    if state.step != parent_state.step:
        changed_fields.append("step")
    if state.extended_integrator_state != parent_state.extended_integrator_state:
        changed_fields.append("extended_integrator_state")
    if changed_fields:
        raise RuntimeError(
            "A jFFS branch may replace only the future stochastic RNG stream; it "
            f"changed phase-space fields={changed_fields}."
        )
    if _canonical_json(state.rng_state) == _canonical_json(parent_state.rng_state):
        raise RuntimeError(
            "jFFS branch retained the parent RNG state. Independent conditional shots "
            "must start from identical phase space but distinct future noise streams."
        )
    start_cv = _evaluate_cv(collective_variable, state, context="new jFFS branch")
    observed_landing = _landing_interface(start_cv, tuple(journal["interfaces_atoms"]))
    if observed_landing != parent["source_interface"]:
        raise RuntimeError(
            "Branching changed or reclassified the parent configuration: stored "
            f"source_interface={parent['source_interface']}, branch CV={start_cv}, "
            f"observed_interface={observed_landing}. Branching may replace only the "
            "future stochastic-noise stream, not phase space."
        )
    shooting["active_trial"] = {
        "parent_state_id": parent["state_id"],
        "current_state_id": store.save_state(state),
        "root_id": parent["root_id"],
        "source_interface": parent["source_interface"],
        "parent_weight": parent["weight"],
        "trial_index": parent["next_trial"],
        "random_seed": random_seed,
        "start_cv": start_cv,
        "last_cv": start_cv,
        "elapsed_steps": 0,
    }
    store.save_journal(journal)
    return state


def _finish_active_trial(
    journal: dict[str, Any],
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    outcome: str,
    landing_interface: int,
) -> None:
    shooting = journal["shooting"]
    active = shooting["active_trial"]
    parent = shooting["pending"][0]
    if active["source_interface"] != parent["source_interface"]:
        raise RuntimeError("jFFS journal parent/active source interface is inconsistent.")
    trial_weight = float(active["parent_weight"]) / algorithm.trials_per_state
    record = {
        "root_id": active["root_id"],
        "parent_state_id": active["parent_state_id"],
        "endpoint_state_id": active["current_state_id"],
        "source_interface": active["source_interface"],
        "landing_interface": landing_interface,
        "outcome": outcome,
        "parent_weight": active["parent_weight"],
        "trial_weight": trial_weight,
        "trial_index": active["trial_index"],
        "random_seed": active["random_seed"],
        "start_cv": active["start_cv"],
        "endpoint_cv": active["last_cv"],
        "elapsed_steps": active["elapsed_steps"],
    }
    shooting["trial_records"].append(record)
    if outcome == "higher_interface":
        if landing_interface <= active["source_interface"]:
            raise RuntimeError(
                "Successful jFFS shot did not land above its source interface: "
                f"source={active['source_interface']}, landing={landing_interface}."
            )
        final_interface = len(algorithm.interfaces_atoms) - 1
        if landing_interface == final_interface:
            shooting["terminal_success_weight_by_root"][active["root_id"]] += (
                trial_weight
            )
        else:
            shooting["pending"].append(
                {
                    "state_id": active["current_state_id"],
                    "source_interface": landing_interface,
                    "weight": trial_weight,
                    "root_id": active["root_id"],
                    "source_cv": active["last_cv"],
                    "next_trial": 0,
                }
            )
    elif outcome != "basin_return":
        raise RuntimeError(f"Unsupported jFFS shot outcome={outcome!r}.")
    parent["next_trial"] += 1
    shooting["active_trial"] = None


def _run_shooting(
    journal: dict[str, Any],
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    engine: JumpyShotEngine,
    collective_variable: JumpyCollectiveVariable,
    store: JumpyFFSArtifactStore,
    progress: Callable[[str], None],
) -> None:
    shooting = journal["shooting"]
    active_state = (
        None
        if shooting["active_trial"] is None
        else store.load_state(shooting["active_trial"]["current_state_id"])
    )
    while shooting["pending"] or shooting["active_trial"] is not None:
        if shooting["active_trial"] is None:
            if active_state is not None:
                raise RuntimeError(
                    "jFFS has an in-memory active shot state without an active journal "
                    "record."
                )
            parent = shooting["pending"][0]
            if parent["next_trial"] == algorithm.trials_per_state:
                shooting["pending"].pop(0)
                store.save_journal(journal)
                continue
            active_state = _start_trial(
                journal,
                engine=engine,
                collective_variable=collective_variable,
                store=store,
            )
        active = shooting["active_trial"]
        if active_state is None:
            raise RuntimeError(
                "jFFS active trial is missing its in-memory dynamical state."
            )
        if active["elapsed_steps"] >= algorithm.shot_max_steps:
            active["current_state_id"] = store.save_state(active_state)
            store.save_journal(journal)
            raise RuntimeError(
                "A jFFS shot reached shot_max_steps without returning to basin A or "
                "reaching a higher interface. Treating it as failure would bias the "
                f"crossing probability. root_id={active['root_id']}, source_interface="
                f"{active['source_interface']}, elapsed_steps={active['elapsed_steps']}, "
                f"last_cv={active['last_cv']}. Increase shot_max_steps in a new "
                "manifest-bound run or justify a censoring estimator."
            )
        previous_step = active_state.step
        active_state = engine.advance(
            active_state, steps=algorithm.cv_interval_steps
        )
        active_state.validate(context="jFFS shooting endpoint")
        if active_state.integrator_contract != engine.contract:
            raise RuntimeError("jFFS shot propagation changed the active engine contract.")
        if active_state.step != previous_step + algorithm.cv_interval_steps:
            raise RuntimeError(
                f"jFFS shot engine advanced state.step from {previous_step} to "
                f"{active_state.step}, expected "
                f"{previous_step + algorithm.cv_interval_steps}."
            )
        _validate_fixed_cell(
            active_state,
            np.asarray(journal["fixed_cell_A"], dtype=np.float64),
            context="jFFS shooting endpoint",
        )
        active["elapsed_steps"] += algorithm.cv_interval_steps
        cv = _evaluate_cv(
            collective_variable, active_state, context="jFFS shooting"
        )
        active["last_cv"] = cv
        landing = _landing_interface(cv, algorithm.interfaces_atoms)
        trial_finished = landing == -1 or landing > active["source_interface"]
        if trial_finished:
            active["current_state_id"] = store.save_state(active_state)
        if landing == -1:
            _finish_active_trial(
                journal,
                algorithm=algorithm,
                outcome="basin_return",
                landing_interface=-1,
            )
        elif landing > active["source_interface"]:
            _finish_active_trial(
                journal,
                algorithm=algorithm,
                outcome="higher_interface",
                landing_interface=landing,
            )
        if trial_finished:
            store.save_journal(journal)
            active_state = None
        elif (
            active["elapsed_steps"] % algorithm.shot_checkpoint_interval_steps == 0
        ):
            active["current_state_id"] = store.save_state(active_state)
            store.save_journal(journal)
        if (
            trial_finished
            and len(shooting["trial_records"]) % 25 == 0
            and shooting["trial_records"]
        ):
            progress(
                f"jFFS: completed {len(shooting['trial_records'])} weighted shots; "
                f"pending parent states={len(shooting['pending'])}"
            )
    journal["phase"] = "analysis"
    store.save_journal(journal)


def _interface_statistics(
    records: list[dict[str, Any]], interfaces: tuple[int, ...]
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source in range(len(interfaces) - 1):
        source_records = [item for item in records if item["source_interface"] == source]
        source_weight = float(sum(item["trial_weight"] for item in source_records))
        landing_weight = {
            str(landing): float(
                sum(
                    item["trial_weight"]
                    for item in source_records
                    if item["landing_interface"] == landing
                )
            )
            for landing in (-1, *range(source + 1, len(interfaces)))
        }
        if source_weight:
            probabilities = {
                key: value / source_weight for key, value in landing_weight.items()
            }
            probability_sum = sum(probabilities.values())
            if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"Weighted outcomes from interface {source} sum to "
                    f"{probability_sum}, expected 1."
                )
        else:
            probabilities = {key: 0.0 for key in landing_weight}
        rows.append(
            {
                "source_interface": source,
                "source_threshold_atoms": interfaces[source],
                "weighted_trial_mass": source_weight,
                "trial_count": len(source_records),
                "landing_weight": landing_weight,
                "conditional_landing_probability": probabilities,
            }
        )
    return tuple(rows)


def _bootstrap_rate(
    *,
    intervals_steps: np.ndarray,
    root_success: np.ndarray,
    timestep_fs: float,
    volume_A3: float,
    samples: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    count = len(intervals_steps)
    if root_success.shape != (count,):
        raise ValueError(
            f"Bootstrap root_success has shape={root_success.shape}, expected {(count,)}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x424F4F54]))
    blocks_needed = math.ceil(count / block_size)
    rates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        starts = rng.integers(0, count, size=blocks_needed)
        indices = np.concatenate(
            [
                (np.arange(start, start + block_size, dtype=np.int64) % count)
                for start in starts
            ]
        )[:count]
        elapsed_ps = float(np.sum(intervals_steps[indices])) * timestep_fs / 1000.0
        if elapsed_ps <= 0.0:
            raise RuntimeError(
                f"Bootstrap sample {sample_index} has nonpositive elapsed time "
                f"{elapsed_ps} ps."
            )
        flux = count / elapsed_ps
        probability = float(np.mean(root_success[indices]))
        rates[sample_index] = flux * probability / volume_A3
    return rates


def _analyze(
    journal: dict[str, Any],
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    timestep_fs: float,
    volume_A3: float,
    scientific_scope: dict[str, Any],
    source_evidence: dict[str, Any],
) -> JumpyFFSResult:
    exits = journal["basin"]["exits"]
    crossing_count = len(exits)
    elapsed_steps = int(journal["basin"]["total_steps"])
    elapsed_ps = elapsed_steps * timestep_fs / 1000.0
    flux_per_ps = crossing_count / elapsed_ps
    terminal_by_root = journal["shooting"]["terminal_success_weight_by_root"]
    crossing_probability = float(sum(terminal_by_root.values()))
    if crossing_probability < -1.0e-14 or crossing_probability > 1.0 + 1.0e-12:
        raise RuntimeError(
            f"Weighted jFFS crossing probability={crossing_probability} lies outside [0, 1]."
        )
    crossing_probability = min(1.0, max(0.0, crossing_probability))
    root_weight = 1.0 / crossing_count
    root_success = np.asarray(
        [terminal_by_root[item["root_id"]] / root_weight for item in exits],
        dtype=np.float64,
    )
    if np.any(root_success < -1.0e-14) or np.any(root_success > 1.0 + 1.0e-12):
        raise RuntimeError(
            f"Per-exit descendant success probabilities are invalid: "
            f"{root_success.tolist()}."
        )
    intervals_steps = np.asarray(
        [item["interval_steps"] for item in exits], dtype=np.int64
    )
    if int(np.sum(intervals_steps)) != elapsed_steps:
        raise RuntimeError(
            "Basin-flux accounting is inconsistent: inter-crossing intervals sum to "
            f"{int(np.sum(intervals_steps))} steps but elapsed basin time is "
            f"{elapsed_steps} steps."
        )
    rate = flux_per_ps * crossing_probability / volume_A3
    all_paths_failed = bool(np.allclose(root_success, 0.0, rtol=0.0, atol=1.0e-12))
    all_paths_succeeded = bool(
        np.allclose(root_success, 1.0, rtol=0.0, atol=1.0e-12)
    )
    if all_paths_failed:
        confidence_level = None
        rate_confidence_interval = None
        rate_bootstrap_standard_error = None
        uncertainty_status = "undefined_inadequate_boundary_sample"
        uncertainty_reason = (
            "No A-to-B success was observed across the basin-exit descendant trees. "
            "Resampling an all-zero success sample produces a degenerate zero-width "
            "bootstrap interval and cannot quantify the upper uncertainty or rate bound. "
            "Run enough additional independent shots to observe successes before "
            "reporting a confidence interval."
        )
        bootstrap_samples_used = 0
    elif all_paths_succeeded:
        confidence_level = None
        rate_confidence_interval = None
        rate_bootstrap_standard_error = None
        uncertainty_status = "undefined_inadequate_boundary_sample"
        uncertainty_reason = (
            "No A-to-B failure was observed across the basin-exit descendant trees. "
            "Resampling an all-one success sample cannot quantify the lower uncertainty "
            "of the crossing probability or rate. Run enough additional independent "
            "shots to observe failures before reporting a confidence interval."
        )
        bootstrap_samples_used = 0
    else:
        bootstrap_rates = _bootstrap_rate(
            intervals_steps=intervals_steps,
            root_success=root_success,
            timestep_fs=timestep_fs,
            volume_A3=volume_A3,
            samples=algorithm.bootstrap_samples,
            block_size=algorithm.bootstrap_block_crossings,
            seed=algorithm.random_seed,
        )
        lower, upper = np.quantile(bootstrap_rates, [0.025, 0.975])
        confidence_level = 0.95
        rate_confidence_interval = (float(lower), float(upper))
        rate_bootstrap_standard_error = float(np.std(bootstrap_rates, ddof=1))
        uncertainty_status = "paired_moving_block_bootstrap_available"
        uncertainty_reason = None
        bootstrap_samples_used = algorithm.bootstrap_samples
    result_scientific_scope = dict(scientific_scope)
    result_scientific_scope["rate_uncertainty_status"] = uncertainty_status
    result_scientific_scope["rate_uncertainty_limitation"] = uncertainty_reason
    initial_weights: dict[str, float] = {}
    for landing in range(len(algorithm.interfaces_atoms)):
        initial_weights[str(landing)] = sum(
            item["weight"] for item in exits if item["landing_interface"] == landing
        )
    initial_probability_sum = sum(initial_weights.values())
    if not np.isclose(initial_probability_sum, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"Initial jump landing probabilities sum to {initial_probability_sum}, "
            "expected 1."
        )
    return JumpyFFSResult(
        basin_crossing_count=crossing_count,
        basin_elapsed_time_ps=elapsed_ps,
        basin_flux_per_ps=flux_per_ps,
        volume_A3=volume_A3,
        crossing_probability=crossing_probability,
        rate_per_A3_ps=rate,
        rate_per_m3_s=rate * 1.0e42,
        confidence_level=confidence_level,
        rate_confidence_interval_per_A3_ps=rate_confidence_interval,
        rate_bootstrap_standard_error_per_A3_ps=(
            rate_bootstrap_standard_error
        ),
        uncertainty_status=uncertainty_status,
        uncertainty_reason=uncertainty_reason,
        bootstrap_samples_used=bootstrap_samples_used,
        interface_statistics=_interface_statistics(
            journal["shooting"]["trial_records"], algorithm.interfaces_atoms
        ),
        initial_landing_probabilities=initial_weights,
        root_success_probabilities=tuple(float(value) for value in root_success),
        trial_count=len(journal["shooting"]["trial_records"]),
        scientific_scope=result_scientific_scope,
        source_evidence=source_evidence,
    )


def run_jumpy_ffs(
    initial_state: DynamicalState,
    *,
    algorithm: JumpyFFSAlgorithmConfig,
    engine: JumpyShotEngine,
    collective_variable: JumpyCollectiveVariable,
    output_root: Path,
    resume: bool,
    source_evidence: dict[str, Any],
    scientific_scope: dict[str, Any],
    progress: Callable[[str], None] = print,
) -> JumpyFFSResult:
    """Run jFFS or resume from its exact last committed Markov state."""

    algorithm.validate()
    initial_state.validate(context="initial jFFS state")
    if algorithm.interfaces_atoms[-1] > len(initial_state.atomic_numbers):
        raise ValueError(
            f"Final jFFS interface={algorithm.interfaces_atoms[-1]} atoms exceeds "
            f"system atom_count={len(initial_state.atomic_numbers)} and is unreachable."
        )
    require_branchable_integrator(engine.contract, context="jFFS shot engine")
    if initial_state.integrator_contract != engine.contract:
        raise RuntimeError(
            "Initial state integrator contract does not exactly match the active engine. "
            "Do not change ensemble, time step, thermostat, constraints, or calculator "
            "identity when restarting or branching."
        )
    if not math.isfinite(engine.timestep_fs) or engine.timestep_fs <= 0.0:
        raise ValueError(
            f"jFFS engine timestep_fs must be finite and positive, got "
            f"{engine.timestep_fs}."
        )
    initial_volume_A3 = float(np.linalg.det(initial_state.cell_A))
    if not isinstance(source_evidence, dict) or not source_evidence:
        raise TypeError("jFFS source_evidence must be a non-empty provenance mapping.")
    if not isinstance(scientific_scope, dict) or not scientific_scope:
        raise TypeError("jFFS scientific_scope must be a non-empty claim-scope mapping.")
    identity = {
        "algorithm": algorithm.to_dict(),
        "engine_contract": engine.contract,
        "collective_variable_contract": collective_variable.contract,
        "atom_count": len(initial_state.atomic_numbers),
        "volume_A3": initial_volume_A3,
        "initial_state_sha256": _state_sha256(initial_state),
        "source_evidence": source_evidence,
        "scientific_scope": scientific_scope,
    }
    store = JumpyFFSArtifactStore(output_root, identity=identity, resume=resume)
    journal = store.load_journal()
    if journal is not None and journal.get("schema_version") != JUMPY_FFS_SCHEMA_VERSION:
        raise RuntimeError(
            f"{store.journal_path}: expected schema_version="
            f"{JUMPY_FFS_SCHEMA_VERSION}, got {journal.get('schema_version')!r}."
        )
    if journal is not None:
        store.garbage_collect_states(journal)
    if journal is None:
        initial_cv = _evaluate_cv(
            collective_variable, initial_state, context="initial jFFS state"
        )
        if initial_cv >= algorithm.interfaces_atoms[-1]:
            raise RuntimeError(
                f"Initial jFFS state has already reached basin B: largest_cluster_atoms="
                f"{initial_cv}, B condition is CV >= {algorithm.interfaces_atoms[-1]}."
            )
        if (
            algorithm.equilibration_steps == 0
            and initial_cv >= algorithm.interfaces_atoms[0]
        ):
            raise RuntimeError(
                f"Initial jFFS state is outside basin A with no configured pre-flux "
                f"equilibration: largest_cluster_atoms={initial_cv}, basin condition is "
                f"CV < {algorithm.interfaces_atoms[0]}."
            )
        initial_state_id = store.save_state(initial_state)
        journal = _initial_journal(
            initial_state_id=initial_state_id,
            initial_cv=initial_cv,
            random_seed=algorithm.random_seed,
            equilibration_steps=algorithm.equilibration_steps,
            fixed_cell_A=initial_state.cell_A,
        )
        journal["interfaces_atoms"] = list(algorithm.interfaces_atoms)
        store.save_journal(journal)
    elif journal.get("interfaces_atoms") != list(algorithm.interfaces_atoms):
        raise RuntimeError(
            "Stored jFFS journal interfaces do not match the manifest-bound algorithm."
        )

    if journal["phase"] == "terminal_invalid":
        terminal = journal["terminal_invalid"]
        raise RuntimeError(
            "Cannot resume terminal-invalid jFFS workflow: "
            f"origin_phase={terminal['origin_phase']}, "
            f"reason_code={terminal['reason_code']}, "
            f"observation_step={terminal['observation_step']}, cv={terminal['cv']}. "
            f"Stored terminal failure: {terminal['message']}"
        )

    if journal["phase"] == "equilibration":
        _advance_equilibration(
            journal,
            algorithm=algorithm,
            engine=engine,
            collective_variable=collective_variable,
            store=store,
            progress=progress,
        )
    if journal["phase"] == "basin":
        _sample_basin_flux(
            journal,
            algorithm=algorithm,
            engine=engine,
            collective_variable=collective_variable,
            store=store,
            progress=progress,
        )
    if journal["phase"] == "shooting":
        _run_shooting(
            journal,
            algorithm=algorithm,
            engine=engine,
            collective_variable=collective_variable,
            store=store,
            progress=progress,
        )
    if journal["phase"] == "analysis":
        result = _analyze(
            journal,
            algorithm=algorithm,
            timestep_fs=engine.timestep_fs,
            volume_A3=initial_volume_A3,
            scientific_scope=scientific_scope,
            source_evidence=source_evidence,
        )
        result_path = store.root / "result.json"
        store._write_json_atomic(result_path, result.to_dict())
        journal["phase"] = "complete"
        journal["result"] = result.to_dict()
        store.save_journal(journal)
        progress(f"jFFS: complete; wrote {result_path}")
        return result
    if journal["phase"] == "complete":
        stored_result = journal.get("result")
        if not isinstance(stored_result, dict):
            raise RuntimeError("Completed jFFS journal is missing its result mapping.")
        result_path = store.root / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"Completed jFFS journal references a missing result artifact: "
                f"{result_path}."
            )
        persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
        if persisted_result != stored_result:
            raise RuntimeError(
                f"{result_path}: result does not match the completed restart journal."
            )
        return JumpyFFSResult(
            basin_crossing_count=int(stored_result["basin_crossing_count"]),
            basin_elapsed_time_ps=float(stored_result["basin_elapsed_time_ps"]),
            basin_flux_per_ps=float(stored_result["basin_flux_per_ps"]),
            volume_A3=float(stored_result["volume_A3"]),
            crossing_probability=float(stored_result["crossing_probability"]),
            rate_per_A3_ps=float(stored_result["rate_per_A3_ps"]),
            rate_per_m3_s=float(stored_result["rate_per_m3_s"]),
            confidence_level=(
                None
                if stored_result["confidence_level"] is None
                else float(stored_result["confidence_level"])
            ),
            rate_confidence_interval_per_A3_ps=(
                None
                if stored_result["rate_confidence_interval_per_A3_ps"] is None
                else tuple(
                    float(value)
                    for value in stored_result[
                        "rate_confidence_interval_per_A3_ps"
                    ]
                )
            ),
            rate_bootstrap_standard_error_per_A3_ps=(
                None
                if stored_result[
                    "rate_bootstrap_standard_error_per_A3_ps"
                ]
                is None
                else float(
                    stored_result["rate_bootstrap_standard_error_per_A3_ps"]
                )
            ),
            uncertainty_status=str(stored_result["uncertainty_status"]),
            uncertainty_reason=(
                None
                if stored_result["uncertainty_reason"] is None
                else str(stored_result["uncertainty_reason"])
            ),
            bootstrap_samples_used=int(stored_result["bootstrap_samples_used"]),
            interface_statistics=tuple(stored_result["interface_statistics"]),
            initial_landing_probabilities={
                str(key): float(value)
                for key, value in stored_result[
                    "initial_landing_probabilities"
                ].items()
            },
            root_success_probabilities=tuple(
                float(value)
                for value in stored_result["root_success_probabilities"]
            ),
            trial_count=int(stored_result["trial_count"]),
            scientific_scope=stored_result["scientific_scope"],
            source_evidence=stored_result["source_evidence"],
        )
    raise RuntimeError(f"Unsupported jFFS journal phase={journal['phase']!r}.")
