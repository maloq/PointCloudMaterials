from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from .homogeneous_campaign_config import (
    HomogeneousCampaignConfig,
    campaign_config_is_monotonic_measurement_extension,
    campaign_config_matches_after_path_relocation,
)
from .homogeneous_online import (
    OnlineCrystallinityObservation,
    online_observations_from_arrays,
    online_observations_to_arrays,
)
from .provenance import ExecutionProvenance, producer_code_is_compatible
from .simulation import ThermodynamicTrace, validate_thermodynamic_trace


REPLICA_CHECKPOINT_SCHEMA_VERSION = 1
SNAPSHOT_ARTIFACT_NAMES = (
    "atoms.traj",
    "trace.npz",
    "online_crystallinity.npz",
    "mtk_state.npz",
    "metadata.json",
)


@dataclass(frozen=True)
class MTKState:
    nsteps: int
    q: np.ndarray
    p: np.ndarray
    eps: float
    p_eps: float
    cell0: np.ndarray
    volume0: float
    thermostat_eta: np.ndarray
    thermostat_p_eta: np.ndarray
    barostat_xi: np.ndarray
    barostat_p_xi: np.ndarray


@dataclass(frozen=True)
class ResumableReplicaCheckpoint:
    atoms: Atoms
    trace: ThermodynamicTrace
    online_observations: tuple[OnlineCrystallinityObservation, ...]
    integrator_state: MTKState
    metadata: dict[str, object]


class ThermodynamicTraceBuffer:
    def __init__(self, trace: ThermodynamicTrace | None = None) -> None:
        if trace is None:
            self.step: list[int] = []
            self.temperature_K: list[float] = []
            self.pressure_GPa: list[float] = []
            self.volume_A3: list[float] = []
            self.potential_energy_eV_per_atom: list[float] = []
            self.positions_A: list[np.ndarray] = []
            self.cell_vectors_A: list[np.ndarray] = []
        else:
            self.step = trace.step.tolist()
            self.temperature_K = trace.temperature_K.tolist()
            self.pressure_GPa = trace.pressure_GPa.tolist()
            self.volume_A3 = trace.volume_A3.tolist()
            self.potential_energy_eV_per_atom = (
                trace.potential_energy_eV_per_atom.tolist()
            )
            self.positions_A = [frame.copy() for frame in trace.positions_A]
            self.cell_vectors_A = [frame.copy() for frame in trace.cell_vectors_A]

    def sample(self, atoms: Atoms, step: int) -> None:
        if self.step and step <= self.step[-1]:
            raise ValueError(
                f"Thermodynamic trace steps must increase strictly: previous={self.step[-1]}, "
                f"new={step}."
            )
        atom_count = len(atoms)
        self.step.append(step)
        self.temperature_K.append(float(atoms.get_temperature()))
        self.pressure_GPa.append(
            float(
                -np.trace(atoms.get_stress(voigt=False, include_ideal_gas=True))
                / 3.0
                / units.GPa
            )
        )
        self.volume_A3.append(float(atoms.get_volume()))
        self.potential_energy_eV_per_atom.append(
            float(atoms.get_potential_energy() / atom_count)
        )
        self.positions_A.append(
            np.asarray(atoms.get_positions(wrap=True), dtype=np.float32)
        )
        self.cell_vectors_A.append(
            np.asarray(atoms.cell.array, dtype=np.float64).copy()
        )

    def finish(self, *, atom_count: int, context: str) -> ThermodynamicTrace:
        trace = ThermodynamicTrace(
            step=np.asarray(self.step, dtype=np.int64),
            temperature_K=np.asarray(self.temperature_K, dtype=np.float64),
            pressure_GPa=np.asarray(self.pressure_GPa, dtype=np.float64),
            volume_A3=np.asarray(self.volume_A3, dtype=np.float64),
            potential_energy_eV_per_atom=np.asarray(
                self.potential_energy_eV_per_atom, dtype=np.float64
            ),
            positions_A=np.stack(self.positions_A),
            cell_vectors_A=np.stack(self.cell_vectors_A),
        )
        validate_thermodynamic_trace(trace, atom_count=atom_count, context=context)
        return trace


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_digests(snapshot: Path) -> dict[str, str]:
    return {name: _sha256_file(snapshot / name) for name in SNAPSHOT_ARTIFACT_NAMES}


def _load_and_verify_snapshot_manifest(snapshot: Path) -> dict[str, object]:
    manifest_path = snapshot / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"{snapshot}: committed checkpoint has no snapshot_manifest.json with "
            "artifact content hashes."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_keys = {"schema_version", "completed_global_step", "artifacts_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError(
            f"{manifest_path}: keys must be exactly {sorted(expected_keys)}, got "
            f"{sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__}."
        )
    if manifest["schema_version"] != REPLICA_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"{manifest_path}: schema_version={manifest['schema_version']!r}, expected "
            f"{REPLICA_CHECKPOINT_SCHEMA_VERSION}."
        )
    digests = manifest["artifacts_sha256"]
    if not isinstance(digests, dict) or set(digests) != set(SNAPSHOT_ARTIFACT_NAMES):
        raise RuntimeError(
            f"{manifest_path}: artifacts_sha256 must contain exactly "
            f"{list(SNAPSHOT_ARTIFACT_NAMES)}, got "
            f"{sorted(digests) if isinstance(digests, dict) else type(digests).__name__}."
        )
    for name, expected_sha256 in digests.items():
        artifact_path = snapshot / name
        if not artifact_path.is_file():
            raise RuntimeError(
                f"{manifest_path}: hashed checkpoint artifact is missing: {artifact_path}."
            )
        observed_sha256 = _sha256_file(artifact_path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"{manifest_path}: checkpoint artifact SHA-256 mismatch for {name}: "
                f"recorded={expected_sha256}, observed={observed_sha256}. The checkpoint "
                "is corrupt and cannot be resumed."
            )
    return manifest


def _load_and_verify_named_snapshot(snapshot: Path) -> dict[str, object]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(
            f"{snapshot}: checkpoint snapshot must be a real directory, not a file or "
            "symbolic link."
        )
    manifest = _load_and_verify_snapshot_manifest(snapshot)
    completed_step = manifest["completed_global_step"]
    if (
        not isinstance(completed_step, int)
        or isinstance(completed_step, bool)
        or completed_step < 0
    ):
        raise RuntimeError(
            f"{snapshot / 'snapshot_manifest.json'}: completed_global_step must be a "
            f"non-negative integer, got {completed_step!r}."
        )
    expected_name = f"step_{completed_step:012d}"
    if snapshot.name != expected_name:
        raise RuntimeError(
            f"{snapshot}: checkpoint directory name is inconsistent with its verified "
            f"completed_global_step={completed_step}; expected {expected_name!r}."
        )
    with (snapshot / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(snapshot / "mtk_state.npz") as stored:
        state_step = int(stored["nsteps"].item())
    if (
        metadata.get("completed_global_step") != completed_step
        or state_step != completed_step
    ):
        raise RuntimeError(
            f"{snapshot}: verified checkpoint step identities disagree: "
            f"directory/manifest={completed_step}, metadata="
            f"{metadata.get('completed_global_step')!r}, "
            f"MTK state={state_step}."
        )
    return manifest


def _write_latest_pointer(directory: Path, snapshot_name: str) -> None:
    temporary = directory / "LATEST.tmp"
    temporary.write_text(f"{snapshot_name}\n", encoding="utf-8")
    temporary.replace(directory / "LATEST")


def build_mtk_dynamics(
    atoms: Atoms,
    *,
    config: HomogeneousCampaignConfig,
    state: MTKState | None,
) -> IsotropicMTKNPT:
    homogeneous = config.homogeneous
    dynamics = IsotropicMTKNPT(
        atoms,
        timestep=homogeneous.generator.dynamics.timestep_fs * units.fs,
        temperature_K=homogeneous.temperature_K,
        pressure_au=homogeneous.generator.dynamics.pressure_GPa * units.GPa,
        tdamp=homogeneous.generator.dynamics.thermostat_time_fs * units.fs,
        pdamp=homogeneous.generator.dynamics.barostat_time_fs * units.fs,
    )
    if state is None:
        return dynamics
    atom_count = len(atoms)
    expected_atom_shape = (atom_count, 3)
    if state.q.shape != expected_atom_shape or state.p.shape != expected_atom_shape:
        raise RuntimeError(
            "MTK checkpoint atom arrays do not match the source system: "
            f"q={state.q.shape}, p={state.p.shape}, expected={expected_atom_shape}."
        )
    expected_cell = state.cell0 * np.exp(state.eps)
    if not np.allclose(
        np.asarray(atoms.cell.array), expected_cell, rtol=1.0e-12, atol=1.0e-10
    ):
        raise RuntimeError(
            "MTK checkpoint cell is inconsistent with cell0 and eps; maximum "
            f"difference_A={float(np.max(np.abs(atoms.cell.array - expected_cell))):.6g}."
        )
    if not np.allclose(atoms.positions, state.q, rtol=1.0e-12, atol=1.0e-10):
        raise RuntimeError(
            "MTK checkpoint Atoms positions differ from the serialized integrator q state."
        )
    if not np.allclose(atoms.get_momenta(), state.p, rtol=1.0e-12, atol=1.0e-10):
        raise RuntimeError(
            "MTK checkpoint Atoms momenta differ from the serialized integrator p state."
        )
    thermostat = dynamics._thermostat
    barostat = dynamics._barostat
    if (
        state.thermostat_eta.shape != thermostat._eta.shape
        or state.thermostat_p_eta.shape != thermostat._p_eta.shape
        or state.barostat_xi.shape != barostat._xi.shape
        or state.barostat_p_xi.shape != barostat._p_xi.shape
    ):
        raise RuntimeError(
            "MTK checkpoint thermostat/barostat chain shapes do not match this ASE "
            "integrator: thermostat_eta="
            f"{state.thermostat_eta.shape}/{thermostat._eta.shape}, thermostat_p_eta="
            f"{state.thermostat_p_eta.shape}/{thermostat._p_eta.shape}, barostat_xi="
            f"{state.barostat_xi.shape}/{barostat._xi.shape}, barostat_p_xi="
            f"{state.barostat_p_xi.shape}/{barostat._p_xi.shape}."
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
        thermostat_eta=np.asarray(
            dynamics._thermostat._eta, dtype=np.float64
        ).copy(),
        thermostat_p_eta=np.asarray(
            dynamics._thermostat._p_eta, dtype=np.float64
        ).copy(),
        barostat_xi=np.asarray(dynamics._barostat._xi, dtype=np.float64).copy(),
        barostat_p_xi=np.asarray(
            dynamics._barostat._p_xi, dtype=np.float64
        ).copy(),
    )


def _campaign_identity(
    config: HomogeneousCampaignConfig,
    execution_provenance: ExecutionProvenance,
    *,
    replica_name: str,
    random_seed: int,
) -> dict[str, object]:
    identity = {
        "schema_version": REPLICA_CHECKPOINT_SCHEMA_VERSION,
        "campaign_config": config.to_dict(),
        "execution_provenance": execution_provenance.to_dict(),
        "replica_name": replica_name,
        "random_seed": random_seed,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**identity, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _normalized_portable_linux_platform(
    value: object,
    *,
    machine: object,
) -> tuple[str, str, str] | None:
    if not isinstance(value, str) or not isinstance(machine, str):
        return None
    if "-with-" not in value:
        return None
    kernel_and_machine, libc = value.rsplit("-with-", maxsplit=1)
    if (
        not kernel_and_machine.startswith("Linux-")
        or not kernel_and_machine.endswith(f"-{machine}")
        or not libc.startswith("glibc")
    ):
        return None
    return ("Linux", machine, libc)


def _checkpoint_runtime_is_portable(
    observed: object,
    expected: object,
) -> bool:
    """Allow only host placement changes within one exact H100 software stack."""

    if observed == expected:
        return True
    if (
        not isinstance(observed, dict)
        or not isinstance(expected, dict)
        or set(observed) != set(expected)
    ):
        return False
    portable_fields = {"platform", "cuda_device_index", "cuda_device_name"}
    for name in set(observed) - portable_fields:
        if observed[name] != expected[name]:
            return False

    if observed.get("platform") != expected.get("platform"):
        observed_platform = _normalized_portable_linux_platform(
            observed.get("platform"), machine=observed.get("machine")
        )
        expected_platform = _normalized_portable_linux_platform(
            expected.get("platform"), machine=expected.get("machine")
        )
        if observed_platform is None or observed_platform != expected_platform:
            return False

    cuda_host_fields = {"cuda_device_index", "cuda_device_name"}
    if cuda_host_fields & set(observed):
        if not cuda_host_fields <= set(observed):
            return False
        for runtime, name in ((observed, "observed"), (expected, "expected")):
            device_index = runtime["cuda_device_index"]
            if (
                not isinstance(device_index, int)
                or isinstance(device_index, bool)
                or device_index < 0
            ):
                raise RuntimeError(
                    f"Checkpoint {name} cuda_device_index must be a non-negative "
                    f"integer, got {device_index!r}."
                )
        observed_name = observed["cuda_device_name"]
        expected_name = expected["cuda_device_name"]
        if observed_name != expected_name:
            h100_name = re.compile(r"^NVIDIA H100(?:\s|$)", flags=re.IGNORECASE)
            if (
                not isinstance(observed_name, str)
                or not isinstance(expected_name, str)
                or h100_name.match(observed_name) is None
                or h100_name.match(expected_name) is None
            ):
                return False
    return True


def _checkpoint_identity_migration_record(
    observed: dict[str, object],
    expected: dict[str, object],
) -> dict[str, object]:
    observed_execution = observed["execution_provenance"]
    expected_execution = expected["execution_provenance"]
    observed_campaign = observed["campaign_config"]
    expected_campaign = expected["campaign_config"]
    if (
        not isinstance(observed_execution, dict)
        or not isinstance(expected_execution, dict)
        or not isinstance(observed_campaign, dict)
        or not isinstance(expected_campaign, dict)
    ):
        raise TypeError("Checkpoint identity migration inputs must contain mappings.")
    observed_runtime = observed_execution["runtime"]
    expected_runtime = expected_execution["runtime"]
    if not isinstance(observed_runtime, dict) or not isinstance(expected_runtime, dict):
        raise TypeError("Checkpoint identity migration runtime records must be mappings.")
    runtime_host_changes = {
        field: {"observed": observed_runtime.get(field), "active": expected_runtime.get(field)}
        for field in ("platform", "cuda_device_index", "cuda_device_name")
        if observed_runtime.get(field) != expected_runtime.get(field)
    }
    observed_homogeneous = observed_campaign["homogeneous"]
    expected_homogeneous = expected_campaign["homogeneous"]
    if not isinstance(observed_homogeneous, dict) or not isinstance(
        expected_homogeneous, dict
    ):
        raise TypeError("Checkpoint campaign homogeneous records must be mappings.")
    return {
        "schema_version": 2,
        "migration": "certified_checkpoint_identity_migration_v2",
        "observed_identity_sha256": observed["identity_sha256"],
        "active_identity_sha256": expected["identity_sha256"],
        "observed_producer_code": observed_execution["producer_code"],
        "active_producer_code": expected_execution["producer_code"],
        "runtime_host_changes": runtime_host_changes,
        "campaign_extension": {
            "observed_measurement_steps": observed_homogeneous["steps"],
            "active_measurement_steps": expected_homogeneous["steps"],
            "observed_output_root": observed_campaign["output_root"],
            "active_output_root": expected_campaign["output_root"],
        },
        "equivalence_basis": (
            "The model, calculator settings, source evidence, integration state, "
            "software stack, CUDA toolkit, cuDNN, and CuEq versions are exact. Only "
            "the Linux kernel placement, logical CUDA index, H100 product name, "
            "repository output labels/locations, and a strictly longer full-duration "
            "measurement endpoint may differ. The next segment resumes the hashed "
            "MTK-NPT state without reconstructing or modifying checkpoint tensors."
        ),
    }


def _write_checkpoint_identity_migration(
    directory: Path,
    *,
    observed: dict[str, object],
    expected: dict[str, object],
) -> None:
    migration_directory = directory / "identity_migrations"
    migration_directory.mkdir(exist_ok=True)
    migration_path = migration_directory / (
        f"{observed['identity_sha256']}_to_{expected['identity_sha256']}.json"
    )
    record = _checkpoint_identity_migration_record(observed, expected)
    if migration_path.exists():
        with migration_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != record:
            raise RuntimeError(
                f"{migration_path}: existing checkpoint identity migration differs "
                "from the exact requested transition."
            )
        return
    temporary = migration_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    temporary.replace(migration_path)


def _compatible_checkpoint_identity_migration(
    observed: object,
    expected: dict[str, object],
) -> dict[str, object] | None:
    """Migrate only certified code, portable H100 host fields, and a longer endpoint."""

    if not isinstance(observed, dict) or set(observed) != set(expected):
        return None
    observed_payload = {
        key: value for key, value in observed.items() if key != "identity_sha256"
    }
    observed_digest = hashlib.sha256(
        json.dumps(
            observed_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if observed.get("identity_sha256") != observed_digest:
        raise RuntimeError(
            "Checkpoint manifest identity_sha256 does not match its serialized identity."
        )
    observed_execution = observed_payload.get("execution_provenance")
    expected_execution = expected.get("execution_provenance")
    if not isinstance(observed_execution, dict) or not isinstance(
        expected_execution, dict
    ):
        return None
    if set(observed_execution) != set(expected_execution):
        return None
    if observed_execution.get("calculator") != expected_execution.get("calculator"):
        return None
    if not _checkpoint_runtime_is_portable(
        observed_execution.get("runtime"), expected_execution.get("runtime")
    ):
        return None
    if not producer_code_is_compatible(
        observed_execution.get("producer_code"),
        expected_execution["producer_code"],
    ):
        return None
    migrated = deepcopy(observed_payload)
    migrated["execution_provenance"]["producer_code"] = deepcopy(
        expected_execution["producer_code"]
    )
    migrated["execution_provenance"]["runtime"] = deepcopy(
        expected_execution["runtime"]
    )
    observed_campaign_config = migrated.get("campaign_config")
    expected_campaign_config = expected.get("campaign_config")
    if not isinstance(expected_campaign_config, dict):
        return None
    if not (
        campaign_config_matches_after_path_relocation(
            observed_campaign_config, expected_campaign_config
        )
        or campaign_config_is_monotonic_measurement_extension(
            observed_campaign_config, expected_campaign_config
        )
    ):
        return None
    migrated["campaign_config"] = deepcopy(expected_campaign_config)
    migrated_digest = hashlib.sha256(
        json.dumps(migrated, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    migrated["identity_sha256"] = migrated_digest
    return migrated if migrated == expected else None


class ResumableReplicaCheckpointStore:
    def __init__(
        self,
        config: HomogeneousCampaignConfig,
        execution_provenance: ExecutionProvenance,
        *,
        replica_name: str,
        random_seed: int,
    ) -> None:
        self.directory = config.output_root / "checkpoints" / replica_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention = config.execution.checkpoint_retention
        self.identity = _campaign_identity(
            config,
            execution_provenance,
            replica_name=replica_name,
            random_seed=random_seed,
        )
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                observed = json.load(handle)
            if observed != self.identity:
                migrated = _compatible_checkpoint_identity_migration(
                    observed, self.identity
                )
                if migrated is None:
                    raise RuntimeError(
                        f"{manifest_path}: checkpoint identity differs from the active "
                        "campaign, potential/runtime, replica name, or random seed. "
                        "Refusing an ambiguous resume."
                    )
                _write_checkpoint_identity_migration(
                    self.directory,
                    observed=observed,
                    expected=self.identity,
                )
                temporary_manifest = manifest_path.with_suffix(".json.tmp")
                with temporary_manifest.open("w", encoding="utf-8") as handle:
                    json.dump(self.identity, handle, indent=2)
                temporary_manifest.replace(manifest_path)
        else:
            temporary = manifest_path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.identity, handle, indent=2)
            temporary.replace(manifest_path)

    def load(self) -> ResumableReplicaCheckpoint | None:
        latest_path = self.directory / "LATEST"
        restore_latest = False
        if not latest_path.exists():
            snapshots = list(self.directory.glob("step_*"))
            if not snapshots:
                return None
            verified_snapshots: list[tuple[int, Path, dict[str, object]]] = []
            verified_steps: set[int] = set()
            for candidate in snapshots:
                candidate_manifest = _load_and_verify_named_snapshot(candidate)
                candidate_step = int(candidate_manifest["completed_global_step"])
                if candidate_step in verified_steps:
                    raise RuntimeError(
                        f"{self.directory}: multiple verified checkpoint snapshots "
                        f"claim completed_global_step={candidate_step}; recovery is "
                        "ambiguous."
                    )
                verified_steps.add(candidate_step)
                verified_snapshots.append(
                    (candidate_step, candidate, candidate_manifest)
                )
            _, snapshot, snapshot_manifest = max(
                verified_snapshots, key=lambda item: item[0]
            )
            snapshot_name = snapshot.name
            restore_latest = True
        else:
            snapshot_name = latest_path.read_text(encoding="utf-8").strip()
            if not snapshot_name or Path(snapshot_name).name != snapshot_name:
                raise RuntimeError(
                    f"{latest_path}: checkpoint pointer must contain exactly one "
                    f"snapshot directory name, got {snapshot_name!r}."
                )
            snapshot = self.directory / snapshot_name
            snapshot_manifest = _load_and_verify_named_snapshot(snapshot)
        required = {
            "atoms": snapshot / "atoms.traj",
            "trace": snapshot / "trace.npz",
            "online": snapshot / "online_crystallinity.npz",
            "integrator": snapshot / "mtk_state.npz",
            "metadata": snapshot / "metadata.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"{snapshot}: LATEST checkpoint is incomplete; missing files={missing}."
            )
        atoms = read(required["atoms"], format="traj")
        with np.load(required["trace"]) as stored:
            trace = ThermodynamicTrace(
                step=stored["step"],
                temperature_K=stored["temperature_K"],
                pressure_GPa=stored["pressure_GPa"],
                volume_A3=stored["volume_A3"],
                potential_energy_eV_per_atom=stored[
                    "potential_energy_eV_per_atom"
                ],
                positions_A=stored["positions_A"],
                cell_vectors_A=stored["cell_vectors_A"],
            )
        validate_thermodynamic_trace(
            trace,
            atom_count=len(atoms),
            context=f"resumable checkpoint trace loaded from {snapshot}",
        )
        with np.load(required["online"]) as stored:
            online_arrays = {name: stored[name] for name in stored.files}
        observations = online_observations_from_arrays(online_arrays)
        with np.load(required["integrator"]) as stored:
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
        with required["metadata"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"{required['metadata']}: checkpoint metadata must be a JSON mapping, "
                f"got {type(metadata).__name__}."
            )
        if state.nsteps != metadata.get("completed_global_step"):
            raise RuntimeError(
                f"{snapshot}: MTK nsteps={state.nsteps} differs from metadata "
                f"completed_global_step={metadata.get('completed_global_step')!r}."
            )
        if state.nsteps != snapshot_manifest["completed_global_step"]:
            raise RuntimeError(
                f"{snapshot}: MTK nsteps={state.nsteps} differs from hashed snapshot "
                "manifest completed_global_step="
                f"{snapshot_manifest['completed_global_step']!r}."
            )
        if restore_latest:
            _write_latest_pointer(self.directory, snapshot_name)
        return ResumableReplicaCheckpoint(
            atoms=atoms,
            trace=trace,
            online_observations=observations,
            integrator_state=state,
            metadata=metadata,
        )

    def save(
        self,
        *,
        atoms: Atoms,
        trace: ThermodynamicTrace,
        online_observations: tuple[OnlineCrystallinityObservation, ...],
        integrator_state: MTKState,
        metadata: dict[str, object],
    ) -> None:
        completed_step = integrator_state.nsteps
        if not isinstance(metadata, dict):
            raise TypeError(
                "Checkpoint metadata must be a mapping, got "
                f"{type(metadata).__name__}."
            )
        if metadata.get("completed_global_step") != completed_step:
            raise RuntimeError(
                "Checkpoint metadata completed_global_step must equal the captured MTK "
                f"nsteps={completed_step}, got {metadata.get('completed_global_step')!r}."
            )
        validate_thermodynamic_trace(
            trace,
            atom_count=len(atoms),
            context=f"resumable checkpoint before step={completed_step}",
        )
        snapshot_name = f"step_{completed_step:012d}"
        final_snapshot = self.directory / snapshot_name
        staging = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_name}.staging-", dir=self.directory)
        )
        try:
            write(staging / "atoms.traj", atoms, format="traj")
            with (staging / "trace.npz").open("wb") as handle:
                np.savez(handle, **trace.__dict__)
            with (staging / "online_crystallinity.npz").open("wb") as handle:
                np.savez(handle, **online_observations_to_arrays(online_observations))
            with (staging / "mtk_state.npz").open("wb") as handle:
                np.savez(handle, **integrator_state.__dict__)
            with (staging / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            snapshot_manifest = {
                "schema_version": REPLICA_CHECKPOINT_SCHEMA_VERSION,
                "completed_global_step": completed_step,
                "artifacts_sha256": _snapshot_digests(staging),
            }
            with (staging / "snapshot_manifest.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(snapshot_manifest, handle, indent=2, sort_keys=True)
            if final_snapshot.exists():
                committed_manifest = _load_and_verify_named_snapshot(final_snapshot)
                if committed_manifest != snapshot_manifest:
                    raise RuntimeError(
                        f"{final_snapshot}: refusing to replace a committed same-step "
                        "checkpoint with different artifact hashes. This indicates "
                        "non-idempotent replay or state corruption."
                    )
                shutil.rmtree(staging)
            else:
                staging.replace(final_snapshot)
            _write_latest_pointer(self.directory, snapshot_name)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        snapshots = sorted(self.directory.glob("step_*"))
        for obsolete in snapshots[: -self.retention]:
            shutil.rmtree(obsolete)
