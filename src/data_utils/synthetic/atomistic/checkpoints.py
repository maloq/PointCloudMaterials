from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms
from ase.io import read, write

from .simulation import ThermodynamicTrace


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
    def __init__(self, config: CheckpointConfig) -> None:
        serialized = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        self.config_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.directory = (
            config.output.root_dir.parent
            / f".{config.output.root_dir.name}.generation-{self.config_sha256[:12]}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        expected_manifest = {
            "schema_version": 1,
            "config_sha256": self.config_sha256,
            "config": config.to_dict(),
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
