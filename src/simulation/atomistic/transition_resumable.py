from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from .homogeneous_resumable import MTKState
from .provenance import ExecutionProvenance
from .simulation import ThermodynamicTrace, validate_thermodynamic_trace
from .transition_campaign_config import TransitionCampaignConfig
from .transition_campaign_queue import TransitionCampaignTask
from .transition_config import TransitionBranchConfig


TRANSITION_CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_ARTIFACTS = ("atoms.traj", "trace.npz", "mtk_state.npz", "metadata.json")


@dataclass(frozen=True)
class TransitionCheckpoint:
    atoms: Atoms
    trace: ThermodynamicTrace
    state: MTKState
    metadata: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(
    config: TransitionCampaignConfig,
    provenance: ExecutionProvenance,
    task: TransitionCampaignTask,
) -> dict[str, object]:
    payload = {
        "schema_version": TRANSITION_CHECKPOINT_SCHEMA_VERSION,
        "campaign_config": config.to_dict(),
        "execution_provenance": provenance.to_dict(),
        "task": task.__dict__,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _verify_snapshot(snapshot: Path) -> dict[str, object]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(f"{snapshot}: checkpoint snapshot must be a real directory.")
    manifest_path = snapshot / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{snapshot}: committed checkpoint has no snapshot_manifest.json.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"schema_version", "completed_global_step", "artifacts_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise RuntimeError(
            f"{manifest_path}: keys must be exactly {sorted(required)}, got {manifest!r}."
        )
    step = manifest["completed_global_step"]
    if (
        manifest["schema_version"] != TRANSITION_CHECKPOINT_SCHEMA_VERSION
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 0
        or snapshot.name != f"step_{step:012d}"
    ):
        raise RuntimeError(f"{manifest_path}: invalid checkpoint schema/step identity.")
    digests = manifest["artifacts_sha256"]
    if not isinstance(digests, dict) or set(digests) != set(CHECKPOINT_ARTIFACTS):
        raise RuntimeError(
            f"{manifest_path}: artifacts_sha256 must contain exactly "
            f"{list(CHECKPOINT_ARTIFACTS)}."
        )
    for name, expected in digests.items():
        artifact = snapshot / name
        if not artifact.is_file():
            raise RuntimeError(f"{manifest_path}: missing hashed artifact {artifact}.")
        observed = _sha256(artifact)
        if observed != expected:
            raise RuntimeError(
                f"{manifest_path}: SHA-256 mismatch for {name}: expected={expected}, "
                f"observed={observed}."
            )
    return manifest


class TransitionCheckpointStore:
    def __init__(
        self,
        config: TransitionCampaignConfig,
        provenance: ExecutionProvenance,
        task: TransitionCampaignTask,
    ) -> None:
        self.directory = config.output_root / "checkpoints" / task.run_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention = config.execution.checkpoint_retention
        self.identity = _identity(config, provenance, task)
        manifest_path = self.directory / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                observed = json.load(handle)
            if observed != self.identity:
                raise RuntimeError(
                    f"{manifest_path}: checkpoint identity differs from the active "
                    "campaign, runtime, branch, or seed; refusing an ambiguous resume."
                )
        else:
            _write_json_atomic(manifest_path, self.identity)

    def load(self) -> TransitionCheckpoint | None:
        pointer = self.directory / "LATEST"
        if not pointer.exists():
            snapshots = list(self.directory.glob("step_*"))
            if not snapshots:
                return None
            verified = [
                (int(_verify_snapshot(snapshot)["completed_global_step"]), snapshot)
                for snapshot in snapshots
            ]
            steps = [step for step, _ in verified]
            if len(set(steps)) != len(steps):
                raise RuntimeError(
                    f"{self.directory}: multiple checkpoint snapshots claim one step."
                )
            _, snapshot = max(verified)
            temporary_pointer = self.directory / "LATEST.tmp"
            temporary_pointer.write_text(f"{snapshot.name}\n", encoding="utf-8")
            temporary_pointer.replace(pointer)
        else:
            snapshot_name = pointer.read_text(encoding="utf-8").strip()
            if not snapshot_name or Path(snapshot_name).name != snapshot_name:
                raise RuntimeError(
                    f"{pointer}: expected one snapshot directory name, got "
                    f"{snapshot_name!r}."
                )
            snapshot = self.directory / snapshot_name
        manifest = _verify_snapshot(snapshot)
        atoms = read(snapshot / "atoms.traj", format="traj")
        with np.load(snapshot / "trace.npz") as stored:
            trace = ThermodynamicTrace(**{name: stored[name] for name in stored.files})
        validate_thermodynamic_trace(
            trace, atom_count=len(atoms), context=f"transition checkpoint {snapshot}"
        )
        with np.load(snapshot / "mtk_state.npz") as stored:
            state = MTKState(
                nsteps=int(stored["nsteps"].item()),
                q=stored["q"],
                p=stored["p"],
                eps=float(stored["eps"].item()),
                p_eps=float(stored["p_eps"].item()),
                cell0=stored["cell0"],
                volume0=float(stored["volume0"].item()),
                thermostat_eta=stored["thermostat_eta"],
                thermostat_p_eta=stored["thermostat_p_eta"],
                barostat_xi=stored["barostat_xi"],
                barostat_p_xi=stored["barostat_p_xi"],
            )
        with (snapshot / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected_step = manifest["completed_global_step"]
        if (
            state.nsteps != expected_step
            or metadata.get("completed_global_step") != expected_step
            or int(trace.step[-1]) != expected_step
        ):
            raise RuntimeError(
                f"{snapshot}: state/metadata/trace checkpoint endpoints disagree: "
                f"state={state.nsteps}, metadata={metadata.get('completed_global_step')!r}, "
                f"trace={int(trace.step[-1])}, manifest={expected_step}."
            )
        return TransitionCheckpoint(atoms=atoms, trace=trace, state=state, metadata=metadata)

    def save(
        self,
        *,
        atoms: Atoms,
        trace: ThermodynamicTrace,
        state: MTKState,
        metadata: dict[str, object],
    ) -> None:
        step = state.nsteps
        if metadata.get("completed_global_step") != step or int(trace.step[-1]) != step:
            raise RuntimeError(
                f"Checkpoint endpoint must equal MTK step={step}; metadata="
                f"{metadata.get('completed_global_step')!r}, trace={int(trace.step[-1])}."
            )
        validate_thermodynamic_trace(
            trace, atom_count=len(atoms), context=f"transition checkpoint step={step}"
        )
        snapshot_name = f"step_{step:012d}"
        final = self.directory / snapshot_name
        staging = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_name}.staging-", dir=self.directory)
        )
        try:
            write(staging / "atoms.traj", atoms, format="traj")
            with (staging / "trace.npz").open("wb") as handle:
                np.savez(handle, **trace.__dict__)
            with (staging / "mtk_state.npz").open("wb") as handle:
                np.savez(handle, **state.__dict__)
            with (staging / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            snapshot_manifest = {
                "schema_version": TRANSITION_CHECKPOINT_SCHEMA_VERSION,
                "completed_global_step": step,
                "artifacts_sha256": {
                    name: _sha256(staging / name) for name in CHECKPOINT_ARTIFACTS
                },
            }
            with (staging / "snapshot_manifest.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(snapshot_manifest, handle, indent=2, sort_keys=True)
            if final.exists():
                if _verify_snapshot(final) != snapshot_manifest:
                    raise RuntimeError(
                        f"{final}: same-step checkpoint exists with different hashes."
                    )
                shutil.rmtree(staging)
            else:
                staging.replace(final)
            temporary_pointer = self.directory / "LATEST.tmp"
            temporary_pointer.write_text(f"{snapshot_name}\n", encoding="utf-8")
            temporary_pointer.replace(self.directory / "LATEST")
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        snapshots = sorted(self.directory.glob("step_*"))
        for obsolete in snapshots[: -self.retention]:
            shutil.rmtree(obsolete)


def build_transition_mtk_dynamics(
    atoms: Atoms,
    *,
    config: TransitionCampaignConfig,
    branch: TransitionBranchConfig,
    state: MTKState | None,
) -> IsotropicMTKNPT:
    dynamics_config = config.transition.generator.dynamics
    dynamics = IsotropicMTKNPT(
        atoms,
        timestep=dynamics_config.timestep_fs * units.fs,
        temperature_K=branch.temperature_K,
        pressure_au=dynamics_config.pressure_GPa * units.GPa,
        tdamp=dynamics_config.thermostat_time_fs * units.fs,
        pdamp=dynamics_config.barostat_time_fs * units.fs,
    )
    if state is None:
        return dynamics
    expected_atom_shape = (len(atoms), 3)
    if state.q.shape != expected_atom_shape or state.p.shape != expected_atom_shape:
        raise RuntimeError(
            f"MTK checkpoint q/p shapes {state.q.shape}/{state.p.shape} do not match "
            f"atoms {expected_atom_shape}."
        )
    expected_cell = state.cell0 * np.exp(state.eps)
    if not np.allclose(atoms.cell.array, expected_cell, rtol=1e-12, atol=1e-10):
        raise RuntimeError("MTK checkpoint cell is inconsistent with cell0 and eps.")
    if not np.allclose(atoms.positions, state.q, rtol=1e-12, atol=1e-10):
        raise RuntimeError("MTK checkpoint Atoms positions differ from q state.")
    if not np.allclose(atoms.get_momenta(), state.p, rtol=1e-12, atol=1e-10):
        raise RuntimeError("MTK checkpoint Atoms momenta differ from p state.")
    thermostat = dynamics._thermostat
    barostat = dynamics._barostat
    observed_shapes = (
        state.thermostat_eta.shape,
        state.thermostat_p_eta.shape,
        state.barostat_xi.shape,
        state.barostat_p_xi.shape,
    )
    expected_shapes = (
        thermostat._eta.shape,
        thermostat._p_eta.shape,
        barostat._xi.shape,
        barostat._p_xi.shape,
    )
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            f"MTK chain shapes {observed_shapes} do not match ASE {expected_shapes}."
        )
    dynamics.nsteps = state.nsteps
    dynamics._q = state.q.copy()
    dynamics._p = state.p.copy()
    dynamics._eps = state.eps
    dynamics._p_eps = state.p_eps
    dynamics._cell0 = state.cell0.copy()
    dynamics._volume0 = state.volume0
    thermostat._eta = state.thermostat_eta.copy()
    thermostat._p_eta = state.thermostat_p_eta.copy()
    barostat._xi = state.barostat_xi.copy()
    barostat._p_xi = state.barostat_p_xi.copy()
    dynamics._update_atoms()
    return dynamics


def capture_mtk_state(dynamics: IsotropicMTKNPT) -> MTKState:
    return MTKState(
        nsteps=int(dynamics.nsteps),
        q=np.asarray(dynamics._q, dtype=np.float64).copy(),
        p=np.asarray(dynamics._p, dtype=np.float64).copy(),
        eps=float(dynamics._eps),
        p_eps=float(dynamics._p_eps),
        cell0=np.asarray(dynamics._cell0, dtype=np.float64).copy(),
        volume0=float(dynamics._volume0),
        thermostat_eta=np.asarray(dynamics._thermostat._eta, dtype=np.float64).copy(),
        thermostat_p_eta=np.asarray(
            dynamics._thermostat._p_eta, dtype=np.float64
        ).copy(),
        barostat_xi=np.asarray(dynamics._barostat._xi, dtype=np.float64).copy(),
        barostat_p_xi=np.asarray(
            dynamics._barostat._p_xi, dtype=np.float64
        ).copy(),
    )
