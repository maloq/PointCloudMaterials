from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.io import write

from .homogeneous_analysis import (
    HomogeneousCrystallizationAnalysis,
    ReplicaObservation,
    analyze_homogeneous_crystallization,
    analyze_replica_survival,
    first_persistent_threshold_run,
    write_homogeneous_progress_visualization,
    write_homogeneous_rdf_visualization,
)
from .homogeneous_campaign_config import (
    HomogeneousCampaignConfig,
    campaign_config_matches_after_path_relocation,
)
from .homogeneous_campaign_queue import (
    CampaignReplicaTask,
    campaign_row,
    campaign_rows,
    claim_analysis_task,
    claim_md_task,
    complete_analysis_task,
    complete_md_task,
    fail_analysis_task,
    fail_md_task,
    initialize_campaign_queue,
)
from .homogeneous_generator import _load_source_liquid, _runtime_generator_config
from .homogeneous_online import (
    OnlineCrystallinityDetector,
    OnlineThresholdTracker,
    online_observations_from_arrays,
    online_observations_to_arrays,
)
from .homogeneous_resumable import (
    ResumableReplicaCheckpointStore,
    ThermodynamicTraceBuffer,
    build_mtk_dynamics,
    capture_mtk_state,
)
from .generator import select_calculator
from .provenance import ExecutionProvenance
from .simulation import (
    ThermodynamicTrace,
    set_maxwell_boltzmann_velocities,
    validate_thermodynamic_trace,
)
from .transition_analysis import STRUCTURE_NAMES, write_structure_slice_visualization
from .validation import SystemDiagnostics, diagnose_system


CAMPAIGN_SCHEMA_VERSION = 2
MD_OUTCOMES = (
    "event_stopped",
    "event_observed_full_duration",
    "right_censored",
    "left_censored",
    "invalid_initial_liquid",
)
RAW_REPLICA_ARTIFACTS = (
    "endpoint.traj",
    "equilibration_trajectory.npz",
    "trajectory.npz",
    "online_crystallinity.npz",
)


@dataclass(frozen=True)
class CampaignReplicaRunResult:
    outcome: str
    raw_directory: Path
    run_metadata_sha256: str
    online_threshold_event: dict[str, object]


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw_artifact_hashes(
    replica_directory: Path,
    metadata_path: Path,
    metadata: dict[str, object],
) -> None:
    artifact_digests = metadata.get("raw_artifacts_sha256")
    if not isinstance(artifact_digests, dict) or set(artifact_digests) != set(
        RAW_REPLICA_ARTIFACTS
    ):
        raise RuntimeError(
            f"{metadata_path}: raw_artifacts_sha256 must contain exactly "
            f"{list(RAW_REPLICA_ARTIFACTS)}, got {artifact_digests!r}."
        )
    failures: list[dict[str, str]] = []
    for relative_path in RAW_REPLICA_ARTIFACTS:
        expected_sha256 = artifact_digests[relative_path]
        if not isinstance(expected_sha256, str):
            failures.append(
                {
                    "path": str(replica_directory / relative_path),
                    "error": f"non-string expected digest {expected_sha256!r}",
                }
            )
            continue
        artifact_path = replica_directory / relative_path
        if not artifact_path.is_file():
            failures.append({"path": str(artifact_path), "error": "missing"})
            continue
        observed_sha256 = _sha256_file(artifact_path)
        if observed_sha256 != expected_sha256:
            failures.append(
                {
                    "path": str(artifact_path),
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                }
            )
    if failures:
        raise RuntimeError(
            f"{metadata_path}: committed raw replica artifact integrity failed: "
            f"{failures}."
        )


def _trace_to_arrays(trace: ThermodynamicTrace) -> dict[str, np.ndarray]:
    return {
        "step": trace.step,
        "temperature_K": trace.temperature_K,
        "pressure_GPa": trace.pressure_GPa,
        "volume_A3": trace.volume_A3,
        "potential_energy_eV_per_atom": trace.potential_energy_eV_per_atom,
        "positions_A": trace.positions_A,
        "cell_vectors_A": trace.cell_vectors_A,
    }


def _load_trace(path: Path) -> ThermodynamicTrace:
    with np.load(path) as stored:
        return ThermodynamicTrace(
            step=stored["step"],
            temperature_K=stored["temperature_K"],
            pressure_GPa=stored["pressure_GPa"],
            volume_A3=stored["volume_A3"],
            potential_energy_eV_per_atom=stored["potential_energy_eV_per_atom"],
            positions_A=stored["positions_A"],
            cell_vectors_A=stored["cell_vectors_A"],
        )


def _slice_continuous_trace(
    trace: ThermodynamicTrace,
    *,
    equilibration_steps: int,
) -> tuple[ThermodynamicTrace, ThermodynamicTrace]:
    boundary = np.flatnonzero(trace.step == equilibration_steps)
    if len(boundary) != 1:
        raise RuntimeError(
            "Continuous campaign trace must contain exactly one equilibration/measurement "
            f"boundary at global step={equilibration_steps}, found "
            f"indices={boundary.tolist()}."
        )

    def sliced(mask: np.ndarray, offset: int) -> ThermodynamicTrace:
        return ThermodynamicTrace(
            step=trace.step[mask].copy() - offset,
            temperature_K=trace.temperature_K[mask].copy(),
            pressure_GPa=trace.pressure_GPa[mask].copy(),
            volume_A3=trace.volume_A3[mask].copy(),
            potential_energy_eV_per_atom=(
                trace.potential_energy_eV_per_atom[mask].copy()
            ),
            positions_A=trace.positions_A[mask].copy(),
            cell_vectors_A=trace.cell_vectors_A[mask].copy(),
        )

    return (
        sliced(trace.step <= equilibration_steps, 0),
        sliced(trace.step >= equilibration_steps, equilibration_steps),
    )


def _expected_online_steps(
    *,
    completed_global_step: int,
    equilibration_steps: int,
    event_check_interval: int,
) -> list[int]:
    if completed_global_step < equilibration_steps:
        return []
    final_measurement_step = completed_global_step - equilibration_steps
    return list(range(0, final_measurement_step + 1, event_check_interval))


def _validated_raw_event_and_outcome(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    replica_directory: Path,
    metadata_path: Path,
    metadata: dict[str, object],
) -> tuple[str, dict[str, object]]:
    """Recompute the committed event/outcome from hashed raw observables."""
    homogeneous = config.homogeneous
    actual_measurement_steps = metadata.get("actual_measurement_steps")
    if not isinstance(actual_measurement_steps, int) or isinstance(
        actual_measurement_steps, bool
    ):
        raise RuntimeError(
            f"{metadata_path}: actual_measurement_steps must be an integer, got "
            f"{actual_measurement_steps!r}."
        )
    trajectory_path = replica_directory / "trajectory.npz"
    with np.load(trajectory_path) as stored:
        trajectory_steps = np.asarray(stored["step"], dtype=np.int64)
    if (
        trajectory_steps.ndim != 1
        or len(trajectory_steps) == 0
        or int(trajectory_steps[0]) != 0
        or int(trajectory_steps[-1]) != actual_measurement_steps
    ):
        raise RuntimeError(
            f"{metadata_path}: actual_measurement_steps={actual_measurement_steps} is "
            f"inconsistent with {trajectory_path} steps={trajectory_steps.tolist()}."
        )

    online_path = replica_directory / "online_crystallinity.npz"
    with np.load(online_path) as stored:
        observations = online_observations_from_arrays(
            {name: stored[name] for name in stored.files}
        )
    observed_steps = [item.measurement_step for item in observations]
    expected_steps = list(
        range(
            0,
            actual_measurement_steps + 1,
            config.execution.event_check_interval,
        )
    )
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"{online_path}: committed online observation steps={observed_steps} "
            f"differ from expected={expected_steps} through actual_measurement_steps="
            f"{actual_measurement_steps}."
        )
    tracker = OnlineThresholdTracker(
        threshold_atoms=homogeneous.analysis.nucleus_size_threshold_atoms,
        persistence_frames=config.execution.online_persistence_frames,
        event_cadence_steps=homogeneous.sample_interval,
        observations=observations,
    )
    event = tracker.event
    initial_invalid = (
        observations[0].crystalline_fraction
        > homogeneous.generator.validation.maximum_liquid_crystalline_fraction
    )
    if initial_invalid:
        expected_final_step = 0
        expected_outcome = "invalid_initial_liquid"
    elif event is not None and event.onset_step == 0:
        expected_final_step = (
            min(
                homogeneous.steps,
                event.confirmation_step + config.execution.post_event_steps,
            )
            if config.execution.stop_on_event
            else homogeneous.steps
        )
        expected_outcome = "left_censored"
    elif event is not None:
        expected_final_step = (
            min(
                homogeneous.steps,
                event.confirmation_step + config.execution.post_event_steps,
            )
            if config.execution.stop_on_event
            else homogeneous.steps
        )
        expected_outcome = (
            "event_stopped"
            if expected_final_step < homogeneous.steps
            else "event_observed_full_duration"
        )
    else:
        expected_final_step = homogeneous.steps
        expected_outcome = "right_censored"
    if actual_measurement_steps != expected_final_step:
        raise RuntimeError(
            f"{metadata_path}: raw online observables imply final measurement step="
            f"{expected_final_step}, but metadata/trajectory end at "
            f"{actual_measurement_steps}."
        )
    outcome = metadata.get("outcome")
    if outcome != expected_outcome:
        raise RuntimeError(
            f"{metadata_path}: raw online observables imply outcome="
            f"{expected_outcome!r}, but metadata records {outcome!r}."
        )

    timestep_fs = homogeneous.generator.dynamics.timestep_fs
    expected_event_document: dict[str, object] = {
        "observed": event is not None,
        "observable_name": "online_persistent_crystalline_cluster_threshold_event",
        "nucleus_size_threshold_atoms": (
            homogeneous.analysis.nucleus_size_threshold_atoms
        ),
        "ptm_normalized_rmsd_cutoff": homogeneous.analysis.ptm_rmsd_cutoff,
        "cluster_neighbor_cutoff_A": (
            homogeneous.analysis.crystalline_cluster_cutoff_A
        ),
        "event_check_interval_steps": config.execution.event_check_interval,
        "event_check_interval_ps": (
            config.execution.event_check_interval * timestep_fs / 1000.0
        ),
        "event_definition_interval_steps": homogeneous.sample_interval,
        "event_definition_interval_ps": (
            homogeneous.sample_interval * timestep_fs / 1000.0
        ),
        "dense_monitoring_frames_used_for_persistence": False,
        "online_persistence_frames": config.execution.online_persistence_frames,
        "configured_saved_persistence_frames": (
            homogeneous.analysis.threshold_persistence_frames
        ),
        "configured_saved_sample_interval_steps": homogeneous.sample_interval,
        "physical_persistence_span_steps": (
            (homogeneous.analysis.threshold_persistence_frames - 1)
            * homogeneous.sample_interval
        ),
        "onset_step": None if event is None else event.onset_step,
        "confirmation_step": None if event is None else event.confirmation_step,
        "onset_time_ps": (
            None if event is None else event.onset_step * timestep_fs / 1000.0
        ),
        "confirmation_time_ps": (
            None
            if event is None
            else event.confirmation_step * timestep_fs / 1000.0
        ),
    }
    observed_event_document = metadata.get("online_threshold_event")
    if observed_event_document != expected_event_document:
        raise RuntimeError(
            f"{metadata_path}: online_threshold_event does not match the event "
            f"recomputed from hashed raw observations: observed="
            f"{observed_event_document!r}, "
            f"expected={expected_event_document!r}."
        )
    expected_observation_label = {
        "event_stopped": expected_outcome == "event_stopped",
        "right_censored": expected_outcome == "right_censored",
        "full_duration_after_event": (
            expected_outcome == "event_observed_full_duration"
        ),
        "left_censored": expected_outcome == "left_censored",
        "invalid_initial_liquid": expected_outcome == "invalid_initial_liquid",
    }
    expected_metadata_fields = {
        "planned_measurement_steps": homogeneous.steps,
        "actual_measurement_duration_ps": (
            actual_measurement_steps * timestep_fs / 1000.0
        ),
        "observation_label": expected_observation_label,
        "post_event_growth_steps_observed": (
            None
            if event is None
            else actual_measurement_steps - event.confirmation_step
        ),
    }
    mismatches = {
        name: {"observed": metadata.get(name), "expected": expected}
        for name, expected in expected_metadata_fields.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{metadata_path}: event/outcome metadata is internally inconsistent: "
            f"{mismatches}."
        )
    return expected_outcome, expected_event_document


def _validate_database_raw_commit(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    replica_directory: Path,
    metadata_path: Path,
    metadata: dict[str, object],
    row: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    anchored_row = (
        campaign_row(config, replica_index=task.replica_index) if row is None else row
    )
    expected_assignment = {
        "replica_name": task.replica_name,
        "random_seed": task.random_seed,
        "md_status": "complete",
        "raw_directory": str(replica_directory),
    }
    assignment_mismatches = {
        name: {"observed": anchored_row.get(name), "expected": expected}
        for name, expected in expected_assignment.items()
        if anchored_row.get(name) != expected
    }
    if assignment_mismatches:
        raise RuntimeError(
            f"{metadata_path}: SQLite raw-commit assignment differs from the replica: "
            f"{assignment_mismatches}."
        )
    expected_metadata_sha256 = anchored_row.get("run_metadata_sha256")
    observed_metadata_sha256 = _sha256_file(metadata_path)
    if expected_metadata_sha256 != observed_metadata_sha256:
        raise RuntimeError(
            f"{metadata_path}: externally anchored run-metadata SHA-256 mismatch: "
            f"SQLite={expected_metadata_sha256!r}, observed={observed_metadata_sha256}."
        )
    expected_outcome, expected_event = _validated_raw_event_and_outcome(
        config,
        task=task,
        replica_directory=replica_directory,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    if anchored_row.get("outcome") != expected_outcome:
        raise RuntimeError(
            f"{metadata_path}: SQLite outcome={anchored_row.get('outcome')!r} differs "
            f"from hashed raw-observable outcome={expected_outcome!r}."
        )
    event_json = anchored_row.get("online_threshold_event_json")
    if not isinstance(event_json, str):
        raise RuntimeError(
            f"{metadata_path}: SQLite has no online-threshold-event commit anchor."
        )
    try:
        anchored_event = json.loads(event_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{metadata_path}: SQLite online-threshold-event anchor is invalid JSON."
        ) from exc
    if anchored_event != expected_event:
        raise RuntimeError(
            f"{metadata_path}: SQLite online-threshold-event anchor differs from "
            f"hashed raw observables: SQLite={anchored_event!r}, "
            f"expected={expected_event!r}."
        )
    return anchored_row, expected_event


def _termination_target(
    config: HomogeneousCampaignConfig,
    tracker: OnlineThresholdTracker,
) -> int:
    homogeneous = config.homogeneous
    natural_end = homogeneous.equilibration_steps + homogeneous.steps
    observations = tracker.observations
    if (
        observations
        and observations[0].measurement_step == 0
        and observations[0].crystalline_fraction
        > homogeneous.generator.validation.maximum_liquid_crystalline_fraction
    ):
        return homogeneous.equilibration_steps
    event = tracker.event
    if event is None or not config.execution.stop_on_event:
        return natural_end
    return min(
        natural_end,
        homogeneous.equilibration_steps
        + event.confirmation_step
        + config.execution.post_event_steps,
    )


def _checkpoint_metadata(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    completed_global_step: int,
    tracker: OnlineThresholdTracker,
) -> dict[str, object]:
    event = tracker.event
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "replica_name": task.replica_name,
        "random_seed": task.random_seed,
        "completed_global_step": completed_global_step,
        "equilibration_steps": config.homogeneous.equilibration_steps,
        "planned_measurement_steps": config.homogeneous.steps,
        "termination_target_global_step": _termination_target(config, tracker),
        "online_event": (
            None
            if event is None
            else {
                "onset_measurement_step": event.onset_step,
                "confirmation_measurement_step": event.confirmation_step,
            }
        ),
        "source_evidence": config.source_evidence,
    }


def _replica_initial_state_design(
    config: HomogeneousCampaignConfig,
) -> dict[str, object]:
    homogeneous = config.homogeneous
    return {
        "shared_coordinate_configuration": True,
        "independently_sampled_coordinate_configurations": False,
        "source_dataset": str(homogeneous.source_dataset),
        "source_environment": homogeneous.source_environment,
        "source_frame_step": homogeneous.source_frame_step,
        "replica_variation": (
            "independent Maxwell-Boltzmann momenta generated from each replica seed"
        ),
        "pre_measurement_mtk_npt_equilibration_steps": (
            homogeneous.equilibration_steps
        ),
        "pre_measurement_mtk_npt_equilibration_ps": (
            homogeneous.equilibration_steps
            * homogeneous.generator.dynamics.timestep_fs
            / 1000.0
        ),
        "statistical_interpretation": (
            "conditional velocity-replica ensemble from one liquid coordinate/cell "
            "configuration; it is not an independently sampled configuration ensemble"
        ),
    }


def _load_existing_raw_result(
    config: HomogeneousCampaignConfig,
    execution_provenance: ExecutionProvenance,
    task: CampaignReplicaTask,
) -> CampaignReplicaRunResult | None:
    raw_directory = config.output_root / "replicas" / task.replica_name
    if not raw_directory.exists():
        return None
    metadata_path = raw_directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"{raw_directory}: replica directory exists without run_metadata.json."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "replica_name": task.replica_name,
        "random_seed": task.random_seed,
        "campaign_config": config.to_dict(),
        "execution_provenance": execution_provenance.to_dict(),
        "source_evidence": config.source_evidence,
    }
    mismatches = {}
    for name, value in expected.items():
        observed_value = metadata.get(name)
        matches = (
            campaign_config_matches_after_path_relocation(observed_value, value)
            if name == "campaign_config" and isinstance(value, dict)
            else observed_value == value
        )
        if not matches:
            mismatches[name] = {"observed": observed_value, "expected": value}
    if mismatches:
        raise RuntimeError(
            f"{metadata_path}: committed raw replica does not match the active campaign: "
            f"mismatches={mismatches}."
        )
    _validate_raw_artifact_hashes(raw_directory, metadata_path, metadata)
    outcome, online_threshold_event = _validated_raw_event_and_outcome(
        config,
        task=task,
        replica_directory=raw_directory,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    return CampaignReplicaRunResult(
        outcome=outcome,
        raw_directory=raw_directory,
        run_metadata_sha256=_sha256_file(metadata_path),
        online_threshold_event=online_threshold_event,
    )


def _write_raw_replica(
    config: HomogeneousCampaignConfig,
    execution_provenance: ExecutionProvenance,
    *,
    task: CampaignReplicaTask,
    atoms: Atoms,
    equilibration_trace: ThermodynamicTrace,
    measurement_trace: ThermodynamicTrace,
    tracker: OnlineThresholdTracker,
    outcome: str,
    diagnostics: SystemDiagnostics,
) -> CampaignReplicaRunResult:
    replicas_root = config.output_root / "replicas"
    replicas_root.mkdir(parents=True, exist_ok=True)
    raw_directory = replicas_root / task.replica_name
    staging = Path(
        tempfile.mkdtemp(prefix=f".{task.replica_name}.staging-", dir=replicas_root)
    )
    event = tracker.event
    timestep_fs = config.homogeneous.generator.dynamics.timestep_fs
    try:
        write(staging / "endpoint.traj", atoms, format="traj")
        _write_npz_atomic(
            staging / "equilibration_trajectory.npz",
            **_trace_to_arrays(equilibration_trace),
        )
        _write_npz_atomic(
            staging / "trajectory.npz", **_trace_to_arrays(measurement_trace)
        )
        _write_npz_atomic(
            staging / "online_crystallinity.npz",
            **online_observations_to_arrays(tracker.observations),
        )
        raw_artifacts_sha256 = {
            relative_path: _sha256_file(staging / relative_path)
            for relative_path in RAW_REPLICA_ARTIFACTS
        }
        metadata = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "replica_name": task.replica_name,
            "random_seed": task.random_seed,
            "campaign_config": config.to_dict(),
            "execution_provenance": execution_provenance.to_dict(),
            "source_evidence": config.source_evidence,
            "replica_initial_state_design": _replica_initial_state_design(config),
            "raw_artifacts_sha256": raw_artifacts_sha256,
            "outcome": outcome,
            "observation_label": {
                "event_stopped": outcome == "event_stopped",
                "right_censored": outcome == "right_censored",
                "full_duration_after_event": outcome
                == "event_observed_full_duration",
                "left_censored": outcome == "left_censored",
                "invalid_initial_liquid": outcome == "invalid_initial_liquid",
            },
            "actual_measurement_steps": int(measurement_trace.step[-1]),
            "planned_measurement_steps": config.homogeneous.steps,
            "actual_measurement_duration_ps": float(
                measurement_trace.step[-1] * timestep_fs / 1000.0
            ),
            "online_threshold_event": {
                "observed": event is not None,
                "observable_name": "online_persistent_crystalline_cluster_threshold_event",
                "nucleus_size_threshold_atoms": (
                    config.homogeneous.analysis.nucleus_size_threshold_atoms
                ),
                "ptm_normalized_rmsd_cutoff": (
                    config.homogeneous.analysis.ptm_rmsd_cutoff
                ),
                "cluster_neighbor_cutoff_A": (
                    config.homogeneous.analysis.crystalline_cluster_cutoff_A
                ),
                "event_check_interval_steps": config.execution.event_check_interval,
                "event_check_interval_ps": (
                    config.execution.event_check_interval * timestep_fs / 1000.0
                ),
                "event_definition_interval_steps": (
                    config.homogeneous.sample_interval
                ),
                "event_definition_interval_ps": (
                    config.homogeneous.sample_interval * timestep_fs / 1000.0
                ),
                "dense_monitoring_frames_used_for_persistence": False,
                "online_persistence_frames": (
                    config.execution.online_persistence_frames
                ),
                "configured_saved_persistence_frames": (
                    config.homogeneous.analysis.threshold_persistence_frames
                ),
                "configured_saved_sample_interval_steps": (
                    config.homogeneous.sample_interval
                ),
                "physical_persistence_span_steps": (
                    (config.homogeneous.analysis.threshold_persistence_frames - 1)
                    * config.homogeneous.sample_interval
                ),
                "onset_step": None if event is None else event.onset_step,
                "confirmation_step": (
                    None if event is None else event.confirmation_step
                ),
                "onset_time_ps": (
                    None
                    if event is None
                    else event.onset_step * timestep_fs / 1000.0
                ),
                "confirmation_time_ps": (
                    None
                    if event is None
                    else event.confirmation_step * timestep_fs / 1000.0
                ),
            },
            "post_event_growth_steps_requested": config.execution.post_event_steps,
            "post_event_growth_steps_observed": (
                None
                if event is None
                else int(measurement_trace.step[-1] - event.confirmation_step)
            ),
            "full_ptm_rdf_analysis_status": (
                "committed independently by full_analysis.json written last"
            ),
            "endpoint_diagnostics": diagnostics.to_dict(),
            "scientific_label": (
                "event-driven stopping changes only post-confirmation trajectory length; "
                "the waiting-time event is the online onset. Event-stopped trajectories "
                "and no-event right-censored trajectories are explicitly distinct."
            ),
        }
        _write_json_atomic(staging / "run_metadata.json", metadata)
        staging.replace(raw_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    metadata_path = raw_directory / "run_metadata.json"
    _validate_raw_artifact_hashes(raw_directory, metadata_path, metadata)
    validated_outcome, online_threshold_event = _validated_raw_event_and_outcome(
        config,
        task=task,
        replica_directory=raw_directory,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    return CampaignReplicaRunResult(
        outcome=validated_outcome,
        raw_directory=raw_directory,
        run_metadata_sha256=_sha256_file(metadata_path),
        online_threshold_event=online_threshold_event,
    )


def run_campaign_replica(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    source_atoms: Atoms,
    calculator: object,
    execution_provenance: ExecutionProvenance,
    detector: OnlineCrystallinityDetector,
    progress: Callable[[str], None] = print,
) -> CampaignReplicaRunResult:
    existing = _load_existing_raw_result(config, execution_provenance, task)
    if existing is not None:
        progress(f"{task.replica_name}: recovered committed raw replica")
        return existing
    checkpoints = ResumableReplicaCheckpointStore(
        config,
        execution_provenance,
        replica_name=task.replica_name,
        random_seed=task.random_seed,
    )
    checkpoint = checkpoints.load()
    homogeneous = config.homogeneous
    if checkpoint is None:
        atoms = source_atoms.copy()
        atoms.calc = calculator
        set_maxwell_boltzmann_velocities(
            atoms,
            homogeneous.temperature_K,
            np.random.default_rng(task.random_seed),
        )
        dynamics = build_mtk_dynamics(atoms, config=config, state=None)
        trace_buffer = ThermodynamicTraceBuffer()
        trace_buffer.sample(atoms, 0)
        tracker = OnlineThresholdTracker(
            threshold_atoms=homogeneous.analysis.nucleus_size_threshold_atoms,
            persistence_frames=config.execution.online_persistence_frames,
            event_cadence_steps=homogeneous.sample_interval,
        )
        progress(
            f"{task.replica_name}: new continuous MTK-NPT trajectory, "
            f"random_seed={task.random_seed}"
        )
    else:
        atoms = checkpoint.atoms
        atoms.calc = calculator
        dynamics = build_mtk_dynamics(
            atoms, config=config, state=checkpoint.integrator_state
        )
        trace_buffer = ThermodynamicTraceBuffer(checkpoint.trace)
        tracker = OnlineThresholdTracker(
            threshold_atoms=homogeneous.analysis.nucleus_size_threshold_atoms,
            persistence_frames=config.execution.online_persistence_frames,
            event_cadence_steps=homogeneous.sample_interval,
            observations=checkpoint.online_observations,
        )
        expected_online_steps = _expected_online_steps(
            completed_global_step=int(dynamics.nsteps),
            equilibration_steps=homogeneous.equilibration_steps,
            event_check_interval=config.execution.event_check_interval,
        )
        observed_online_steps = [
            item.measurement_step for item in tracker.observations
        ]
        if observed_online_steps != expected_online_steps:
            raise RuntimeError(
                f"{task.replica_name}: checkpoint online observation steps="
                f"{observed_online_steps} differ from expected={expected_online_steps} "
                f"at completed_global_step={dynamics.nsteps}."
            )
        progress(
            f"{task.replica_name}: resumed exact MTK state at global step="
            f"{dynamics.nsteps} from {checkpoints.directory}"
        )

    last_committed_step = (
        -1 if checkpoint is None else checkpoint.integrator_state.nsteps
    )

    def commit_checkpoint(*, reason: str) -> None:
        nonlocal last_committed_step
        completed_step = int(dynamics.nsteps)
        if completed_step == last_committed_step:
            return
        checkpoint_trace = trace_buffer.finish(
            atom_count=len(atoms),
            context=(
                f"{task.replica_name} checkpoint at global step={completed_step}"
            ),
        )
        checkpoints.save(
            atoms=atoms,
            trace=checkpoint_trace,
            online_observations=tracker.observations,
            integrator_state=capture_mtk_state(dynamics),
            metadata=_checkpoint_metadata(
                config,
                task=task,
                completed_global_step=completed_step,
                tracker=tracker,
            ),
        )
        last_committed_step = completed_step
        progress(
            f"{task.replica_name}: committed resumable {reason} checkpoint through "
            f"global step={completed_step}"
        )

    natural_end = homogeneous.equilibration_steps + homogeneous.steps
    sample_interval = homogeneous.sample_interval
    event_interval = config.execution.event_check_interval
    while dynamics.nsteps < _termination_target(config, tracker):
        chunk_end = min(
            int(dynamics.nsteps) + config.execution.chunk_steps,
            _termination_target(config, tracker),
        )
        while dynamics.nsteps < min(chunk_end, _termination_target(config, tracker)):
            current_step = int(dynamics.nsteps)
            next_sample_step = (
                current_step // sample_interval + 1
            ) * sample_interval
            next_event_step = (
                homogeneous.equilibration_steps
                + len(tracker.observations) * event_interval
            )
            next_boundary = min(
                chunk_end,
                _termination_target(config, tracker),
                next_sample_step,
                next_event_step,
            )
            if next_boundary > current_step:
                dynamics.run(next_boundary - current_step)
            current_step = int(dynamics.nsteps)
            if current_step == next_sample_step:
                trace_buffer.sample(atoms, current_step)
            if current_step == next_event_step and current_step <= natural_end:
                measurement_step = current_step - homogeneous.equilibration_steps
                if measurement_step < 0:
                    raise RuntimeError(
                        f"{task.replica_name}: computed negative online measurement_step="
                        f"{measurement_step} at global step={current_step}."
                    )
                observation = detector.evaluate(
                    atoms, measurement_step=measurement_step
                )
                tracker.append(observation)
                progress(
                    f"{task.replica_name}: online step={measurement_step}, "
                    "largest_crystalline_cluster_atoms="
                    f"{observation.largest_crystalline_cluster_atoms}, "
                    f"crystalline_fraction={observation.crystalline_fraction:.6f}"
                )
        if (
            dynamics.nsteps == _termination_target(config, tracker)
            and trace_buffer.step[-1] != dynamics.nsteps
        ):
            trace_buffer.sample(atoms, int(dynamics.nsteps))
        commit_checkpoint(reason="chunk")

    completed_global_step = int(dynamics.nsteps)
    if completed_global_step > natural_end:
        raise RuntimeError(
            f"{task.replica_name}: completed global step={completed_global_step} exceeds "
            f"natural end={natural_end}."
        )
    if trace_buffer.step[-1] != completed_global_step:
        raise RuntimeError(
            f"{task.replica_name}: final checkpoint at global step={completed_global_step} "
            f"does not contain its endpoint trace frame; last trace step="
            f"{trace_buffer.step[-1]}."
        )
    continuous_trace = trace_buffer.finish(
        atom_count=len(atoms), context=f"{task.replica_name} completed continuous trace"
    )
    equilibration_trace, measurement_trace = _slice_continuous_trace(
        continuous_trace, equilibration_steps=homogeneous.equilibration_steps
    )

    initial_observation = tracker.observations[0]
    maximum_initial_fraction = (
        homogeneous.generator.validation.maximum_liquid_crystalline_fraction
    )
    if initial_observation.crystalline_fraction > maximum_initial_fraction:
        outcome = "invalid_initial_liquid"
    elif tracker.event is not None and tracker.event.onset_step == 0:
        outcome = "left_censored"
    elif tracker.event is not None and completed_global_step < natural_end:
        outcome = "event_stopped"
    elif tracker.event is not None:
        outcome = "event_observed_full_duration"
    else:
        if completed_global_step != natural_end:
            raise RuntimeError(
                f"{task.replica_name}: no event but trajectory ended at global step="
                f"{completed_global_step}, before natural end={natural_end}; it cannot be "
                "labeled as right-censored."
            )
        outcome = "right_censored"
    diagnostics = diagnose_system(
        atoms,
        measurement_trace,
        _runtime_generator_config(homogeneous),
        name=f"{task.replica_name}_optimized_homogeneous_crystallization",
        require_pressure_convergence=True,
    )
    return _write_raw_replica(
        config,
        execution_provenance,
        task=task,
        atoms=atoms,
        equilibration_trace=equilibration_trace,
        measurement_trace=measurement_trace,
        tracker=tracker,
        outcome=outcome,
        diagnostics=diagnostics,
    )


def run_md_worker(
    config: HomogeneousCampaignConfig,
    *,
    worker_name: str,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
    progress: Callable[[str], None] = print,
) -> bool:
    """Run a dynamic seed queue while retaining one calculator/model in memory."""
    source = _load_source_liquid(config.homogeneous)
    selected_calculator, execution_provenance = select_calculator(
        config.homogeneous.generator,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    detector = OnlineCrystallinityDetector(
        ptm_rmsd_cutoff=config.homogeneous.analysis.ptm_rmsd_cutoff,
        crystalline_cluster_cutoff_A=(
            config.homogeneous.analysis.crystalline_cluster_cutoff_A
        ),
    )
    progress(
        f"{worker_name}: loaded calculator once for dynamic replica queue; "
        f"source_atoms={len(source.atoms)}"
    )
    while True:
        task = claim_md_task(config, worker_name=worker_name)
        if task is None:
            return False
        try:
            result = run_campaign_replica(
                config,
                task=task,
                source_atoms=source.atoms,
                calculator=selected_calculator,
                execution_provenance=execution_provenance,
                detector=detector,
                progress=progress,
            )
            complete_md_task(
                config,
                task=task,
                outcome=result.outcome,
                raw_directory=result.raw_directory,
                run_metadata_sha256=result.run_metadata_sha256,
                online_threshold_event=result.online_threshold_event,
            )
        except BaseException:
            error = traceback.format_exc()
            fail_md_task(config, task=task, error=error)
            raise RuntimeError(
                f"{worker_name}: replica {task.replica_name} failed; the full traceback "
                "was persisted in campaign.sqlite3."
            )


def _analysis_to_dict(
    analysis: HomogeneousCrystallizationAnalysis,
) -> dict[str, object]:
    return {
        "nucleation_observed": analysis.nucleation_observed,
        "nucleation_step": analysis.nucleation_step,
        "nucleation_time_ps": analysis.nucleation_time_ps,
        "confirmation_step": analysis.confirmation_step,
        "confirmation_time_ps": analysis.confirmation_time_ps,
        "ptm_rmsd_cutoff": analysis.ptm_rmsd_cutoff,
        "nucleus_size_threshold_atoms": analysis.nucleus_size_threshold_atoms,
        "threshold_persistence_frames": analysis.threshold_persistence_frames,
        "maximum_cluster_atoms": int(
            np.max(analysis.largest_crystalline_cluster_atoms)
        ),
        "initial_crystalline_fraction": float(analysis.crystalline_fraction[0]),
        "final_crystalline_fraction": float(analysis.crystalline_fraction[-1]),
    }


def _use_regular_saved_frames_for_offline_event(
    analysis: HomogeneousCrystallizationAnalysis,
    *,
    config: HomogeneousCampaignConfig,
) -> HomogeneousCrystallizationAnalysis:
    regular_indices = np.flatnonzero(
        analysis.step % config.homogeneous.sample_interval == 0
    )
    persistent_run = first_persistent_threshold_run(
        analysis.largest_crystalline_cluster_atoms[regular_indices],
        threshold=config.homogeneous.analysis.nucleus_size_threshold_atoms,
        persistence_frames=(
            config.homogeneous.analysis.threshold_persistence_frames
        ),
    )
    if persistent_run is None:
        return replace(
            analysis,
            nucleation_observed=False,
            nucleation_step=None,
            nucleation_time_ps=None,
            confirmation_step=None,
            confirmation_time_ps=None,
        )
    onset_regular_index, confirmation_regular_index = persistent_run
    onset_index = int(regular_indices[onset_regular_index])
    confirmation_index = int(regular_indices[confirmation_regular_index])
    return replace(
        analysis,
        nucleation_observed=True,
        nucleation_step=int(analysis.step[onset_index]),
        nucleation_time_ps=float(analysis.time_ps[onset_index]),
        confirmation_step=int(analysis.step[confirmation_index]),
        confirmation_time_ps=float(analysis.time_ps[confirmation_index]),
    )


def _audit_online_offline_shared_frames(
    *,
    analysis: HomogeneousCrystallizationAnalysis,
    online_path: Path,
) -> dict[str, object]:
    with np.load(online_path) as stored:
        online = online_observations_from_arrays(
            {name: stored[name] for name in stored.files}
        )
    online_by_step = {item.measurement_step: item for item in online}
    shared_steps: list[int] = []
    for frame_index, step_value in enumerate(analysis.step):
        step = int(step_value)
        observation = online_by_step.get(step)
        if observation is None:
            continue
        shared_steps.append(step)
        offline_largest = int(analysis.largest_crystalline_cluster_atoms[frame_index])
        offline_cluster_count = int(analysis.crystalline_cluster_count[frame_index])
        offline_fraction = float(analysis.crystalline_fraction[frame_index])
        if (
            offline_largest != observation.largest_crystalline_cluster_atoms
            or offline_cluster_count != observation.crystalline_cluster_count
            or offline_fraction != observation.crystalline_fraction
        ):
            raise RuntimeError(
                f"{online_path}: online/offline PTM cluster observables disagree at shared "
                f"measurement step={step}: online=(largest="
                f"{observation.largest_crystalline_cluster_atoms}, clusters="
                f"{observation.crystalline_cluster_count}, fraction="
                f"{observation.crystalline_fraction}), offline=(largest={offline_largest}, "
                f"clusters={offline_cluster_count}, fraction={offline_fraction}). Event "
                "control and offline scientific analysis must use identical observables."
            )
    if not shared_steps:
        raise RuntimeError(
            f"{online_path}: online and offline traces have no shared measurement step."
        )
    return {
        "status": "exact_match",
        "shared_frame_count": len(shared_steps),
        "shared_measurement_steps": shared_steps,
        "comparison": (
            "largest crystalline cluster, crystalline cluster count, and crystalline "
            "fraction are exactly equal at every shared frame"
        ),
    }


def analyze_campaign_replica(
    config: HomogeneousCampaignConfig,
    *,
    task: CampaignReplicaTask,
    progress: Callable[[str], None] = print,
) -> str:
    replica_directory = config.output_root / "replicas" / task.replica_name
    metadata_path = replica_directory / "run_metadata.json"
    analysis_path = replica_directory / "full_analysis.json"
    required = (
        metadata_path,
        *(replica_directory / name for name in RAW_REPLICA_ARTIFACTS),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{replica_directory}: cannot run deferred PTM/RDF analysis; "
            f"missing={missing}."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        metadata.get("schema_version") != CAMPAIGN_SCHEMA_VERSION
        or metadata.get("replica_name") != task.replica_name
        or metadata.get("random_seed") != task.random_seed
        or not campaign_config_matches_after_path_relocation(
            metadata.get("campaign_config"), config.to_dict()
        )
        or metadata.get("source_evidence") != config.source_evidence
    ):
        raise RuntimeError(
            f"{metadata_path}: raw trajectory identity does not match analysis task."
        )
    _validate_raw_artifact_hashes(replica_directory, metadata_path, metadata)
    queue_row, anchored_online_event = _validate_database_raw_commit(
        config,
        task=task,
        replica_directory=replica_directory,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    run_metadata_sha256 = str(queue_row["run_metadata_sha256"])
    expected_analysis_artifacts = {
        "crystallization_progress.npz",
        "total_rdf.npz",
    }
    if config.homogeneous.output.create_visualizations:
        expected_analysis_artifacts.update(
            {
                "visualizations/crystallization_progress.png",
                "visualizations/total_rdf.png",
                "visualizations/structure_slice.png",
            }
        )
    if (
        analysis_path.is_file()
        and queue_row.get("analysis_status") in ("pending", "running")
        and queue_row.get("full_analysis_sha256") is None
    ):
        with analysis_path.open("r", encoding="utf-8") as handle:
            unanchored_analysis = json.load(handle)
        if unanchored_analysis.get("run_metadata_sha256") is None:
            progress(
                f"{task.replica_name}: replacing pre-anchor full analysis from "
                "externally verified raw artifacts"
            )
            analysis_path.unlink()
    if analysis_path.is_file():
        if queue_row.get("analysis_status") == "complete":
            anchored_analysis_sha256 = queue_row.get("full_analysis_sha256")
            observed_analysis_sha256 = _sha256_file(analysis_path)
            if anchored_analysis_sha256 != observed_analysis_sha256:
                raise RuntimeError(
                    f"{analysis_path}: externally anchored full-analysis SHA-256 "
                    f"mismatch: SQLite={anchored_analysis_sha256!r}, "
                    f"observed={observed_analysis_sha256}."
                )
        with analysis_path.open("r", encoding="utf-8") as handle:
            committed_analysis = json.load(handle)
        expected_identity = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "replica_name": task.replica_name,
            "random_seed": task.random_seed,
            "campaign_config": config.to_dict(),
            "source_evidence": config.source_evidence,
            "raw_artifacts_sha256": metadata["raw_artifacts_sha256"],
            "run_metadata_sha256": run_metadata_sha256,
            "queue_outcome": queue_row["outcome"],
            "queue_online_threshold_event": anchored_online_event,
        }
        mismatches = {}
        for name, value in expected_identity.items():
            observed_value = committed_analysis.get(name)
            matches = (
                campaign_config_matches_after_path_relocation(observed_value, value)
                if name == "campaign_config" and isinstance(value, dict)
                else observed_value == value
            )
            if not matches:
                mismatches[name] = {
                    "observed": observed_value,
                    "expected": value,
                }
        artifact_digests = committed_analysis.get("artifacts_sha256")
        observed_artifact_names = (
            sorted(artifact_digests)
            if isinstance(artifact_digests, dict)
            else artifact_digests
        )
        if not isinstance(artifact_digests, dict) or set(artifact_digests) != (
            expected_analysis_artifacts
        ):
            raise RuntimeError(
                f"{analysis_path}: committed analysis artifacts_sha256 must contain "
                f"exactly {sorted(expected_analysis_artifacts)}, got "
                f"{observed_artifact_names!r}."
            )
        artifact_failures: list[dict[str, str]] = []
        for relative_path, expected_sha256 in artifact_digests.items():
            artifact_path = replica_directory / relative_path
            if not artifact_path.is_file():
                artifact_failures.append(
                    {"path": str(artifact_path), "error": "missing"}
                )
                continue
            observed_sha256 = _sha256_file(artifact_path)
            if observed_sha256 != expected_sha256:
                artifact_failures.append(
                    {
                        "path": str(artifact_path),
                        "expected_sha256": str(expected_sha256),
                        "observed_sha256": observed_sha256,
                    }
                )
        if mismatches or artifact_failures:
            raise RuntimeError(
                f"{analysis_path}: committed analysis is inconsistent; identity "
                f"mismatches={mismatches}, artifact failures={artifact_failures}."
            )
        return _sha256_file(analysis_path)
    trace = _load_trace(replica_directory / "trajectory.npz")
    validate_thermodynamic_trace(
        trace,
        atom_count=trace.positions_A.shape[1],
        context=f"offline analysis input {replica_directory / 'trajectory.npz'}",
    )
    homogeneous = config.homogeneous
    analysis = analyze_homogeneous_crystallization(
        trace,
        chemical_symbol=homogeneous.generator.system.chemical_symbol,
        timestep_fs=homogeneous.generator.dynamics.timestep_fs,
        ptm_rmsd_cutoff=homogeneous.analysis.ptm_rmsd_cutoff,
        crystalline_cluster_cutoff_A=(
            homogeneous.analysis.crystalline_cluster_cutoff_A
        ),
        nucleus_size_threshold_atoms=(
            homogeneous.analysis.nucleus_size_threshold_atoms
        ),
        threshold_persistence_frames=(
            homogeneous.analysis.threshold_persistence_frames
        ),
        rdf_cutoff_A=homogeneous.analysis.rdf_cutoff_A,
        rdf_bins=homogeneous.analysis.rdf_bins,
        progress=progress,
    )
    analysis = _use_regular_saved_frames_for_offline_event(
        analysis, config=config
    )
    shared_audit = _audit_online_offline_shared_frames(
        analysis=analysis,
        online_path=replica_directory / "online_crystallinity.npz",
    )
    _write_npz_atomic(
        replica_directory / "crystallization_progress.npz",
        step=analysis.step,
        time_ps=analysis.time_ps,
        structure_names=np.asarray(STRUCTURE_NAMES),
        structure_fractions=analysis.structure_fractions,
        crystalline_fraction=analysis.crystalline_fraction,
        crystalline_cluster_count=analysis.crystalline_cluster_count,
        largest_crystalline_cluster_atoms=(
            analysis.largest_crystalline_cluster_atoms
        ),
    )
    _write_npz_atomic(
        replica_directory / "total_rdf.npz",
        step=analysis.step,
        time_ps=analysis.time_ps,
        distance_A=analysis.rdf_distance_A,
        g_r=analysis.rdf_g_r,
    )
    analysis_document = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "replica_name": task.replica_name,
        "random_seed": task.random_seed,
        "campaign_config": config.to_dict(),
        "source_evidence": config.source_evidence,
        "raw_artifacts_sha256": metadata["raw_artifacts_sha256"],
        "run_metadata_sha256": run_metadata_sha256,
        "queue_outcome": queue_row["outcome"],
        "queue_online_threshold_event": anchored_online_event,
        "analysis_role": (
            "offline full-trajectory PTM/cluster/RDF audit; MD stopping used the "
            "separately persisted online detector"
        ),
        "event_frame_policy": (
            "Only frames on the configured sample_interval cadence enter the offline "
            "persistent-event audit. An extra event-stop endpoint may be structurally "
            "analyzed but never shortens the configured persistence span."
        ),
        "saved_frame_analysis": _analysis_to_dict(analysis),
        "online_offline_shared_frame_audit": shared_audit,
    }
    if homogeneous.output.create_visualizations:
        visualization_directory = replica_directory / "visualizations"
        visualization_directory.mkdir(exist_ok=True)
        write_homogeneous_progress_visualization(
            visualization_directory / "crystallization_progress.png",
            trace=trace,
            analysis=analysis,
            temperature_K=homogeneous.temperature_K,
            pressure_GPa=homogeneous.generator.dynamics.pressure_GPa,
        )
        write_homogeneous_rdf_visualization(
            visualization_directory / "total_rdf.png",
            analysis=analysis,
            temperature_K=homogeneous.temperature_K,
        )
        write_structure_slice_visualization(
            visualization_directory / "structure_slice.png",
            trace=trace,
            chemical_symbol=homogeneous.generator.system.chemical_symbol,
            timestep_fs=homogeneous.generator.dynamics.timestep_fs,
            reference_planes_fractional=(),
            simulation_name="optimized homogeneous crystallization",
            temperature_K=homogeneous.temperature_K,
            ptm_rmsd_cutoff=homogeneous.analysis.ptm_rmsd_cutoff,
        )
    artifact_paths = {
        "crystallization_progress.npz": (
            replica_directory / "crystallization_progress.npz"
        ),
        "total_rdf.npz": replica_directory / "total_rdf.npz",
    }
    if homogeneous.output.create_visualizations:
        artifact_paths.update(
            {
                "visualizations/crystallization_progress.png": (
                    replica_directory
                    / "visualizations"
                    / "crystallization_progress.png"
                ),
                "visualizations/total_rdf.png": (
                    replica_directory / "visualizations" / "total_rdf.png"
                ),
                "visualizations/structure_slice.png": (
                    replica_directory / "visualizations" / "structure_slice.png"
                ),
            }
        )
    analysis_document["artifacts_sha256"] = {
        relative_path: _sha256_file(path)
        for relative_path, path in artifact_paths.items()
    }
    # This JSON is the analysis commit marker and is intentionally written last. A crash
    # before this point leaves only replaceable partial artifacts; a retry recomputes them.
    _write_json_atomic(analysis_path, analysis_document)
    return _sha256_file(analysis_path)


def run_analysis_worker(
    config: HomogeneousCampaignConfig,
    *,
    worker_name: str,
    follow_md: bool,
    progress: Callable[[str], None] = print,
) -> None:
    while True:
        task = claim_analysis_task(config, worker_name=worker_name)
        if task is not None:
            try:
                full_analysis_sha256 = analyze_campaign_replica(
                    config, task=task, progress=progress
                )
                complete_analysis_task(
                    config,
                    task=task,
                    full_analysis_sha256=full_analysis_sha256,
                )
            except BaseException:
                error = traceback.format_exc()
                fail_analysis_task(config, task=task, error=error)
                raise RuntimeError(
                    f"{worker_name}: offline analysis for {task.replica_name} failed; "
                    "the full traceback was persisted in campaign.sqlite3."
                )
            continue
        rows = campaign_rows(config)
        md_unfinished = any(
            row["md_status"] in ("queued", "running") for row in rows
        )
        if follow_md and md_unfinished:
            time.sleep(1.0)
            continue
        return


def finalize_campaign(config: HomogeneousCampaignConfig) -> Path:
    rows = campaign_rows(config)
    incomplete = [
        {
            "replica_name": row["replica_name"],
            "md_status": row["md_status"],
            "analysis_status": row["analysis_status"],
        }
        for row in rows
        if row["md_status"] != "complete" or row["analysis_status"] != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"Campaign cannot be finalized with incomplete replicas: {incomplete}."
        )
    observations: list[ReplicaObservation] = []
    outcomes: dict[str, int] = {name: 0 for name in MD_OUTCOMES}
    replica_records: list[dict[str, object]] = []
    invalid_records: list[dict[str, object]] = []
    for row in rows:
        replica_name = str(row["replica_name"])
        metadata_path = (
            config.output_root / "replicas" / replica_name / "run_metadata.json"
        )
        analysis_path = (
            config.output_root / "replicas" / replica_name / "full_analysis.json"
        )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        _validate_raw_artifact_hashes(metadata_path.parent, metadata_path, metadata)
        task = CampaignReplicaTask(
            replica_index=int(row["replica_index"]),
            replica_name=replica_name,
            random_seed=int(row["random_seed"]),
        )
        anchored_row, anchored_online_event = _validate_database_raw_commit(
            config,
            task=task,
            replica_directory=metadata_path.parent,
            metadata_path=metadata_path,
            metadata=metadata,
            row=row,
        )
        if not analysis_path.is_file():
            raise FileNotFoundError(
                f"{analysis_path}: queue marks analysis complete but the commit marker "
                "is missing."
            )
        analyze_campaign_replica(
            config,
            task=task,
            progress=lambda _message: None,
        )
        with analysis_path.open("r", encoding="utf-8") as handle:
            offline_analysis = json.load(handle)
        outcome = anchored_row.get("outcome")
        if outcome not in outcomes:
            raise RuntimeError(
                f"{metadata_path}: unsupported finalized outcome={outcome!r}."
            )
        outcomes[str(outcome)] += 1
        online_event = anchored_online_event
        if outcome in ("left_censored", "invalid_initial_liquid"):
            invalid_records.append(
                {"replica_name": replica_name, "outcome": outcome}
            )
        elif outcome in ("event_stopped", "event_observed_full_duration"):
            onset_time_ps = online_event.get("onset_time_ps")
            if not isinstance(onset_time_ps, (int, float)):
                raise RuntimeError(
                    f"{metadata_path}: event outcome has no numeric online onset time."
                )
            observations.append(
                ReplicaObservation(
                    replica_name=replica_name,
                    random_seed=int(row["random_seed"]),
                    event_observed=True,
                    observation_time_ps=float(onset_time_ps),
                )
            )
        else:
            observations.append(
                ReplicaObservation(
                    replica_name=replica_name,
                    random_seed=int(row["random_seed"]),
                    event_observed=False,
                    observation_time_ps=float(
                        metadata["actual_measurement_duration_ps"]
                    ),
                )
            )
        replica_records.append(
            {
                "replica_name": replica_name,
                "random_seed": int(row["random_seed"]),
                "outcome": outcome,
                "raw_and_analysis_directory": str(metadata_path.parent),
                "run_metadata_sha256": anchored_row["run_metadata_sha256"],
                "full_analysis_sha256": anchored_row["full_analysis_sha256"],
                "online_threshold_event": online_event,
                "offline_saved_frame_analysis": offline_analysis[
                    "saved_frame_analysis"
                ],
                "online_offline_shared_frame_audit": offline_analysis[
                    "online_offline_shared_frame_audit"
                ],
            }
        )
    if invalid_records:
        invalid_path = config.output_root / "invalid_observations.json"
        _write_json_atomic(
            invalid_path,
            {
                "status": "campaign_not_valid_for_survival_analysis",
                "invalid_replicas": invalid_records,
            },
        )
        raise RuntimeError(
            "Campaign contains left-censored or invalid-initial-liquid replicas and cannot "
            f"produce a Kaplan-Meier curve: {invalid_records}. Details: {invalid_path}."
        )
    survival = analyze_replica_survival(tuple(observations))
    survival_document = survival.to_dict()
    survival_document["replica_initial_state_design"] = (
        _replica_initial_state_design(config)
    )
    survival_document["statistical_scope"] = (
        "The Kaplan-Meier curve is descriptive and conditional on one shared liquid "
        "coordinate/cell configuration with independently randomized momenta. It does "
        "not quantify uncertainty over independently sampled liquid configurations."
    )
    _write_json_atomic(
        config.output_root / "survival_summary.json", survival_document
    )
    _write_npz_atomic(
        config.output_root / "survival_curve.npz",
        time_ps=survival.time_ps,
        replicas_at_risk=survival.replicas_at_risk,
        events=survival.events,
        right_censored=survival.censored,
        survival_probability=survival.survival_probability,
    )
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "dataset_name": config.homogeneous.dataset_name,
        "campaign_config": config.to_dict(),
        "source_evidence": config.source_evidence,
        "potential_selection": {
            "report": (
                None
                if config.potential_selection_report is None
                else str(config.potential_selection_report)
            ),
            "report_sha256": config.potential_selection_report_sha256,
            "direct_generator_model_selection": (
                config.potential_selection_report is None
            ),
        },
        "replica_count": len(replica_records),
        "outcome_counts": outcomes,
        "replicas": replica_records,
        "campaign_artifacts_sha256": {
            "survival_summary.json": _sha256_file(
                config.output_root / "survival_summary.json"
            ),
            "survival_curve.npz": _sha256_file(
                config.output_root / "survival_curve.npz"
            ),
        },
        "execution_optimizations": {
            "immutable_liquid_source_reused": True,
            "interface_source_preparation_required": False,
            "persistent_model_per_gpu_worker": True,
            "dynamic_seed_queue": True,
            "exact_mtk_state_checkpoint_resume": True,
            "event_driven_stop": config.execution.stop_on_event,
            "post_event_growth_steps": config.execution.post_event_steps,
            "full_ptm_rdf_analysis": config.execution.analysis_mode,
        },
        "replica_initial_state_design": _replica_initial_state_design(config),
        "scientific_scope": {
            "supported_claim": (
                "A descriptive conditional waiting-time survival curve across randomized "
                "momenta from one shared liquid coordinate/cell configuration, for the "
                "configured online PTM connected-cluster event. The original saved-frame "
                "persistence definition is preserved while denser frames monitor control, "
                "and stopped/full-duration/right-censored outcomes remain explicit."
            ),
            "unsupported_claim": (
                "Independent-configuration ensemble uncertainty, a converged homogeneous "
                "nucleation rate, a committor-derived critical nucleus, or potential-"
                "independent kinetics without a decorrelated source bank plus separate "
                "stationarity, finite-size, model, and undercooling validation."
            ),
        },
    }
    manifest_path = config.output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _write_campaign_status(
    config: HomogeneousCampaignConfig,
    *,
    status: str,
    detail: object | None = None,
) -> None:
    _write_json_atomic(
        config.output_root / "campaign_status.json",
        {
            "status": status,
            "written_at_epoch": time.time(),
            "detail": detail,
            "replicas": campaign_rows(config),
        },
    )


def _spawn_logged(
    *,
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[bytes], object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    log_handle.write(
        (
            f"\n=== worker invocation started at epoch={time.time():.6f}; "
            f"command={command!r} ===\n"
        ).encode("utf-8")
    )
    log_handle.flush()
    parent_pid = os.getpid()

    def terminate_with_parent() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                "prctl(PR_SET_PDEATHSIG, SIGTERM) failed for campaign worker",
            )
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        preexec_fn=terminate_with_parent,
    )
    return process, log_handle


def _worker_command(
    config: HomogeneousCampaignConfig,
    *,
    role: str,
    worker_name: str,
    follow_md: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.data_utils.synthetic.atomistic_homogeneous_campaign",
        role,
        "--config",
        str(config.config_path),
        "--worker-name",
        worker_name,
    ]
    if follow_md:
        command.append("--follow-md")
    return command


def _failure_records(config: HomogeneousCampaignConfig) -> list[dict[str, object]]:
    return [
        row
        for row in campaign_rows(config)
        if row["md_status"] == "failed" or row["analysis_status"] == "failed"
    ]


def run_optimized_campaign(
    config: HomogeneousCampaignConfig,
    *,
    devices: tuple[str, ...],
    retry_failed: bool = False,
) -> Path | None:
    if not devices or any(not device.strip() for device in devices):
        raise ValueError(f"At least one non-empty CUDA device is required, got {devices}.")
    if len(set(devices)) != len(devices):
        raise ValueError(f"CUDA devices must be unique, got {devices}.")
    selection_report = config.potential_selection_report
    if selection_report is not None:
        observed_selection_sha256 = _sha256_file(selection_report)
        if observed_selection_sha256 != config.potential_selection_report_sha256:
            raise RuntimeError(
                "Potential-selection report changed after campaign configuration load: "
                f"path={selection_report}, loaded_sha256="
                f"{config.potential_selection_report_sha256}, observed_sha256="
                f"{observed_selection_sha256}."
            )
        runtime_controls = config.potential_selection_runtime_controls
        if runtime_controls is None:
            raise RuntimeError(
                f"{selection_report}: optimized campaign has no validated runtime "
                "projection evidence."
            )
    config.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.output_root / "campaign.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Campaign is already controlled by another process: {lock_path}."
            ) from exc
        initialize_campaign_queue(config, retry_failed=retry_failed)
        _write_campaign_status(
            config,
            status="running",
            detail={"devices": list(devices)},
        )
        rows = campaign_rows(config)
        queued_count = sum(row["md_status"] == "queued" for row in rows)
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
        base_environment = os.environ.copy()
        for worker_index, device in enumerate(devices[:queued_count]):
            worker_name = f"md_gpu_{device}"
            environment = base_environment.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            process, log_handle = _spawn_logged(
                command=_worker_command(
                    config,
                    role="worker",
                    worker_name=worker_name,
                ),
                log_path=config.output_root / "logs" / f"{worker_name}.log",
                environment=environment,
            )
            processes.append((worker_name, process, log_handle))
        if config.execution.analysis_mode == "asynchronous":
            analysis_needed = any(
                row["analysis_status"] in ("blocked", "pending", "running")
                for row in rows
            )
            if analysis_needed:
                for worker_index in range(config.execution.analysis_workers):
                    worker_name = f"analysis_{worker_index:02d}"
                    process, log_handle = _spawn_logged(
                        command=_worker_command(
                            config,
                            role="analyzer",
                            worker_name=worker_name,
                            follow_md=True,
                        ),
                        log_path=(
                            config.output_root / "logs" / f"{worker_name}.log"
                        ),
                        environment=base_environment,
                    )
                    processes.append((worker_name, process, log_handle))
        exit_codes: dict[str, int] = {}
        for worker_name, process, log_handle in processes:
            try:
                exit_codes[worker_name] = process.wait()
            finally:
                log_handle.close()
        failures = _failure_records(config)
        nonzero = {
            name: code for name, code in exit_codes.items() if code != 0
        }
        if nonzero or failures:
            detail = {"process_exit_codes": nonzero, "failed_replicas": failures}
            _write_campaign_status(config, status="failed", detail=detail)
            raise RuntimeError(
                f"Optimized campaign workers failed: {detail}. Inspect "
                f"{config.output_root / 'logs'} and campaign.sqlite3."
            )
        rows = campaign_rows(config)
        unfinished_md = [
            row for row in rows if row["md_status"] in ("queued", "running")
        ]
        if unfinished_md:
            detail = {
                "reason": "workers_exited_before_campaign_completion",
                "remaining_replica_count": len(unfinished_md),
                "resume_command": (
                    "invoke the same campaign config again after resolving why the "
                    "workers exited"
                ),
            }
            _write_campaign_status(config, status="paused", detail=detail)
            return None
        if config.execution.analysis_mode == "deferred":
            _write_campaign_status(
                config,
                status="awaiting_offline_analysis",
                detail=(
                    "Run the analyze command with explicit CPU worker count; GPU MD is "
                    "complete and no MACE model is needed for analysis."
                ),
            )
            return None
        manifest_path = finalize_campaign(config)
        _write_campaign_status(config, status="complete")
        return manifest_path


def run_deferred_campaign_analysis(
    config: HomogeneousCampaignConfig,
    *,
    workers: int,
    retry_failed: bool = False,
) -> Path:
    if workers <= 0:
        raise ValueError(f"Offline analysis workers must be positive, got {workers}.")
    config.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.output_root / "campaign.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Campaign is already controlled by another process: {lock_path}."
            ) from exc
        initialize_campaign_queue(config, retry_failed=retry_failed)
        rows = campaign_rows(config)
        unfinished_md = [
            row["replica_name"] for row in rows if row["md_status"] != "complete"
        ]
        if unfinished_md:
            raise RuntimeError(
                f"Offline analysis requires completed MD; unfinished={unfinished_md}."
            )
        _write_campaign_status(config, status="offline_analysis_running")
        pending_count = sum(
            row["analysis_status"] == "pending" for row in rows
        )
        base_environment = os.environ.copy()
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
        for worker_index in range(min(workers, pending_count)):
            worker_name = f"offline_analysis_{worker_index:02d}"
            process, log_handle = _spawn_logged(
                command=_worker_command(
                    config,
                    role="analyzer",
                    worker_name=worker_name,
                ),
                log_path=config.output_root / "logs" / f"{worker_name}.log",
                environment=base_environment,
            )
            processes.append((worker_name, process, log_handle))
        exit_codes: dict[str, int] = {}
        for worker_name, process, log_handle in processes:
            try:
                exit_codes[worker_name] = process.wait()
            finally:
                log_handle.close()
        failures = _failure_records(config)
        nonzero = {
            name: code for name, code in exit_codes.items() if code != 0
        }
        if nonzero or failures:
            detail = {"process_exit_codes": nonzero, "failed_replicas": failures}
            _write_campaign_status(config, status="failed", detail=detail)
            raise RuntimeError(f"Offline campaign analysis failed: {detail}.")
        manifest_path = finalize_campaign(config)
        _write_campaign_status(config, status="complete")
        return manifest_path
