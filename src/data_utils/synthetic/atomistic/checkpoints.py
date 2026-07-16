from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms
from ase.io import read, write

from .provenance import ExecutionProvenance
from .simulation import ThermodynamicTrace, validate_thermodynamic_trace


CHECKPOINT_SCHEMA_VERSION = 2


class _CheckpointOutput(Protocol):
    root_dir: Path


class CheckpointConfig(Protocol):
    output: _CheckpointOutput

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SimulationCheckpoint:
    atoms: Atoms
    trace: ThermodynamicTrace
    metadata: dict[str, object]


class CheckpointStore:
    def __init__(
        self,
        config: CheckpointConfig,
        execution_provenance: ExecutionProvenance,
    ) -> None:
        config_dict = config.to_dict()
        serialized_config = json.dumps(
            config_dict, sort_keys=True, separators=(",", ":")
        )
        self.config_sha256 = hashlib.sha256(
            serialized_config.encode("utf-8")
        ).hexdigest()
        identity = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "config": config_dict,
            "execution_provenance": execution_provenance.to_dict(),
        }
        serialized_identity = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        )
        self.identity_sha256 = hashlib.sha256(
            serialized_identity.encode("utf-8")
        ).hexdigest()
        self.directory = (
            config.output.root_dir.parent
            / f".{config.output.root_dir.name}.generation-{self.identity_sha256[:12]}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        expected_manifest = {
            **identity,
            "checkpoint_identity_sha256": self.identity_sha256,
        }
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                observed_manifest = json.load(handle)
            if observed_manifest != expected_manifest:
                raise RuntimeError(
                    f"Checkpoint manifest does not match the active configuration: {manifest_path}."
                )
        else:
            temporary = manifest_path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(expected_manifest, handle, indent=2)
            temporary.replace(manifest_path)

    def load(self, stage: str) -> SimulationCheckpoint | None:
        atoms_path = self.directory / f"{stage}.traj"
        trace_path = self.directory / f"{stage}.trace.npz"
        metadata_path = self.directory / f"{stage}.json"
        existing = [path.exists() for path in (atoms_path, trace_path, metadata_path)]
        if not any(existing):
            return None
        if not all(existing):
            raise RuntimeError(
                f"Incomplete checkpoint for stage={stage!r} in {self.directory}; "
                f"atoms={existing[0]}, trace={existing[1]}, metadata={existing[2]}."
            )
        atoms = read(atoms_path, format="traj")
        with np.load(trace_path) as stored:
            trace = ThermodynamicTrace(
                step=stored["step"],
                temperature_K=stored["temperature_K"],
                pressure_GPa=stored["pressure_GPa"],
                volume_A3=stored["volume_A3"],
                potential_energy_eV_per_atom=stored["potential_energy_eV_per_atom"],
                positions_A=stored["positions_A"],
                cell_vectors_A=stored["cell_vectors_A"],
            )
        validate_thermodynamic_trace(
            trace,
            atom_count=len(atoms),
            context=f"checkpoint stage={stage!r} loaded from {trace_path}",
        )
        _validate_checkpoint_endpoint(
            atoms,
            trace,
            context=f"checkpoint stage={stage!r} loaded from {self.directory}",
        )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return SimulationCheckpoint(atoms=atoms, trace=trace, metadata=metadata)

    def save(
        self,
        stage: str,
        atoms: Atoms,
        trace: ThermodynamicTrace,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        validate_thermodynamic_trace(
            trace,
            atom_count=len(atoms),
            context=f"checkpoint stage={stage!r} before save to {self.directory}",
        )
        _validate_checkpoint_endpoint(
            atoms,
            trace,
            context=f"checkpoint stage={stage!r} before save to {self.directory}",
        )
        atoms_path = self.directory / f"{stage}.traj"
        trace_path = self.directory / f"{stage}.trace.npz"
        metadata_path = self.directory / f"{stage}.json"
        atoms_temporary = atoms_path.with_suffix(".traj.tmp")
        trace_temporary = trace_path.with_suffix(".npz.tmp")
        metadata_temporary = metadata_path.with_suffix(".json.tmp")
        write(atoms_temporary, atoms, format="traj")
        with trace_temporary.open("wb") as handle:
            np.savez(
                handle,
                step=trace.step,
                temperature_K=trace.temperature_K,
                pressure_GPa=trace.pressure_GPa,
                volume_A3=trace.volume_A3,
                potential_energy_eV_per_atom=trace.potential_energy_eV_per_atom,
                positions_A=trace.positions_A,
                cell_vectors_A=trace.cell_vectors_A,
            )
        with metadata_temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata or {}, handle, indent=2)
        atoms_temporary.replace(atoms_path)
        trace_temporary.replace(trace_path)
        metadata_temporary.replace(metadata_path)


def _validate_checkpoint_endpoint(
    atoms: Atoms,
    trace: ThermodynamicTrace,
    *,
    context: str,
) -> None:
    atom_cell = np.asarray(atoms.cell.array, dtype=np.float64)
    trace_cell = np.asarray(trace.cell_vectors_A[-1], dtype=np.float64)
    if not np.allclose(atom_cell, trace_cell, rtol=1.0e-12, atol=1.0e-10):
        raise RuntimeError(
            f"{context}: checkpoint .traj cell does not match the last trace cell; "
            f"maximum_absolute_difference_A={float(np.max(np.abs(atom_cell - trace_cell))):.6g}."
        )
    if not bool(np.all(atoms.pbc)):
        raise RuntimeError(
            f"{context}: checkpoint endpoint must be periodic in all axes, got "
            f"pbc={atoms.pbc.tolist()}."
        )
    atom_positions = np.asarray(atoms.positions, dtype=np.float64)
    trace_positions = np.asarray(trace.positions_A[-1], dtype=np.float64)
    fractional_difference = np.linalg.solve(
        trace_cell.T, (atom_positions - trace_positions).T
    ).T
    fractional_difference -= np.rint(fractional_difference)
    minimum_image_difference = fractional_difference @ trace_cell
    maximum_difference_A = float(
        np.max(np.linalg.norm(minimum_image_difference, axis=1))
    )
    trace_position_dtype = np.asarray(trace.positions_A).dtype
    position_precision = (
        np.finfo(trace_position_dtype).eps
        if np.issubdtype(trace_position_dtype, np.floating)
        else np.finfo(np.float64).eps
    )
    coordinate_scale_A = max(
        1.0,
        float(np.max(np.abs(trace_positions))),
        float(np.max(np.abs(trace_cell))),
    )
    endpoint_tolerance_A = 8.0 * position_precision * coordinate_scale_A
    if maximum_difference_A > endpoint_tolerance_A:
        raise RuntimeError(
            f"{context}: checkpoint .traj positions do not match the last trace positions "
            "under the periodic minimum-image convention; "
            f"maximum_atom_displacement_A={maximum_difference_A:.6g}, "
            f"trace_dtype={trace_position_dtype}, "
            f"precision_tolerance_A={endpoint_tolerance_A:.6g}."
        )
