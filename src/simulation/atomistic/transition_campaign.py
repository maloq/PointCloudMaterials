from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from .artifacts import PHASE_TO_ID
from .generator import select_calculator
from .homogeneous_resumable import ThermodynamicTraceBuffer
from .provenance import (
    CalculatorProvenance,
    ExecutionProvenance,
    bind_transition_campaign_execution_provenance,
    build_transition_deferred_analysis_provenance,
)
from .simulation import (
    ThermodynamicTrace,
    set_maxwell_boltzmann_velocities,
    validate_thermodynamic_trace,
)
from .transition_analysis import (
    analyze_phase_rdf,
    analyze_transition,
    write_phase_rdf_overview,
    write_structure_slice_overview,
)
from .transition_campaign_config import (
    TransitionCampaignConfig,
    load_content_bound_prepared_interface,
)
from .transition_campaign_queue import (
    TransitionCampaignTask,
    campaign_row,
    campaign_rows,
    claim_analysis_task,
    claim_md_task,
    complete_analysis_task,
    complete_md_task,
    fail_task,
    initialize_transition_queue,
    validate_transition_queue_identity,
)
from .transition_generator import (
    TransitionBranchResult,
    _resolve_zero_velocity,
    _runtime_generator_config,
    _slice_trace,
    _velocity_summary as _generator_velocity_summary,
    _write_branch,
    _write_velocity_summary_visualization,
)
from .transition_resumable import (
    TransitionCheckpointStore,
    build_transition_mtk_dynamics,
    capture_mtk_state,
)
from .validation import diagnose_system


RAW_ARTIFACTS = (
    "endpoint.traj",
    "endpoint_forces.npy",
    "equilibration_trajectory.npz",
    "trajectory.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_trace(path: Path, trace: ThermodynamicTrace) -> None:
    with path.open("wb") as handle:
        np.savez(handle, **trace.__dict__)


def _load_trace(path: Path, *, atom_count: int, context: str) -> ThermodynamicTrace:
    with np.load(path) as stored:
        trace = ThermodynamicTrace(**{name: stored[name] for name in stored.files})
    validate_thermodynamic_trace(trace, atom_count=atom_count, context=context)
    return trace


def _branch_for_task(config: TransitionCampaignConfig, task: TransitionCampaignTask):
    branch = config.transition.temperature_runs[task.branch_index]
    if branch.name != task.branch_name:
        raise RuntimeError(
            f"{task.run_name}: queue branch={task.branch_name!r} differs from config "
            f"index={task.branch_index} name={branch.name!r}."
        )
    return branch


def _validate_artifact_commit(
    directory: Path,
    *,
    metadata_name: str,
    artifact_key: str,
    expected_artifacts: tuple[str, ...] | None,
) -> tuple[dict[str, object], str]:
    metadata_path = directory / metadata_name
    if not metadata_path.is_file():
        raise RuntimeError(f"{directory}: committed directory has no {metadata_name}.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    digests = metadata.get(artifact_key)
    if not isinstance(digests, dict):
        raise RuntimeError(f"{metadata_path}: {artifact_key} must be a mapping.")
    if expected_artifacts is not None and set(digests) != set(expected_artifacts):
        raise RuntimeError(
            f"{metadata_path}: {artifact_key} must contain exactly "
            f"{list(expected_artifacts)}, got {sorted(digests)}."
        )
    for relative, expected in digests.items():
        artifact = directory / relative
        if not artifact.is_file():
            raise RuntimeError(f"{metadata_path}: missing committed artifact {artifact}.")
        observed = _sha256(artifact)
        if observed != expected:
            raise RuntimeError(
                f"{metadata_path}: SHA-256 mismatch for {relative}: expected={expected}, "
                f"observed={observed}."
            )
    return metadata, _sha256(metadata_path)


def _raw_directory(config: TransitionCampaignConfig, task: TransitionCampaignTask) -> Path:
    return config.output_root / "raw" / task.run_name


def _write_raw_commit(
    config: TransitionCampaignConfig,
    provenance: ExecutionProvenance,
    task: TransitionCampaignTask,
    *,
    atoms,
    equilibration_trace: ThermodynamicTrace,
    production_trace: ThermodynamicTrace,
    endpoint_forces: np.ndarray,
) -> tuple[Path, str]:
    final = _raw_directory(config, task)
    if final.exists():
        metadata, digest = _validate_artifact_commit(
            final,
            metadata_name="raw_commit.json",
            artifact_key="raw_artifacts_sha256",
            expected_artifacts=RAW_ARTIFACTS,
        )
        if metadata.get("campaign_config") != config.to_dict() or metadata.get(
            "execution_provenance"
        ) != provenance.to_dict() or metadata.get(
            "source_evidence"
        ) != config.source_evidence or metadata.get("task") != task.__dict__:
            raise RuntimeError(
                f"{final}: existing raw commit belongs to another campaign/runtime/task."
            )
        return final, digest
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    try:
        endpoint = atoms.copy()
        endpoint.calc = None
        endpoint.wrap()
        write(staging / "endpoint.traj", endpoint, format="traj")
        np.save(staging / "endpoint_forces.npy", endpoint_forces)
        _write_trace(staging / "equilibration_trajectory.npz", equilibration_trace)
        _write_trace(staging / "trajectory.npz", production_trace)
        metadata = {
            "schema_version": 1,
            "campaign_config": config.to_dict(),
            "execution_provenance": provenance.to_dict(),
            "source_evidence": config.source_evidence,
            "task": task.__dict__,
            "raw_artifacts_sha256": {
                name: _sha256(staging / name) for name in RAW_ARTIFACTS
            },
        }
        with (staging / "raw_commit.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        staging.replace(final)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final, _sha256(final / "raw_commit.json")


def run_transition_task(
    config: TransitionCampaignConfig,
    *,
    task: TransitionCampaignTask,
    prepared,
    calculator: object,
    provenance: ExecutionProvenance,
    progress: Callable[[str], None] = print,
) -> tuple[Path, str]:
    validate_transition_queue_identity(config)
    existing = _raw_directory(config, task)
    if existing.exists():
        metadata, digest = _validate_artifact_commit(
            existing,
            metadata_name="raw_commit.json",
            artifact_key="raw_artifacts_sha256",
            expected_artifacts=RAW_ARTIFACTS,
        )
        if metadata.get("campaign_config") != config.to_dict() or metadata.get(
            "execution_provenance"
        ) != provenance.to_dict() or metadata.get(
            "source_evidence"
        ) != config.source_evidence or metadata.get("task") != task.__dict__:
            raise RuntimeError(f"{existing}: committed raw task identity mismatch.")
        progress(f"{task.run_name}: recovered atomically committed raw trajectory")
        return existing, digest

    branch = _branch_for_task(config, task)
    checkpoints = TransitionCheckpointStore(config, provenance, task)
    checkpoint = checkpoints.load()
    if checkpoint is None:
        atoms = prepared.atoms.copy()
        atoms.calc = calculator
        set_maxwell_boltzmann_velocities(
            atoms,
            branch.temperature_K,
            np.random.default_rng(task.simulation_seed),
        )
        dynamics = build_transition_mtk_dynamics(
            atoms, config=config, branch=branch, state=None
        )
        trace_buffer = ThermodynamicTraceBuffer()
        trace_buffer.sample(atoms, 0)
        progress(
            f"{task.run_name}: started continuous MTK-NPT, "
            f"simulation_seed={task.simulation_seed}, temperature={branch.temperature_K} K"
        )
    else:
        atoms = checkpoint.atoms
        atoms.calc = calculator
        dynamics = build_transition_mtk_dynamics(
            atoms, config=config, branch=branch, state=checkpoint.state
        )
        trace_buffer = ThermodynamicTraceBuffer(checkpoint.trace)
        progress(
            f"{task.run_name}: resumed exact MTK state at global step={dynamics.nsteps}"
        )

    total_steps = branch.equilibration_steps + branch.production_steps
    if dynamics.nsteps > total_steps:
        raise RuntimeError(
            f"{task.run_name}: checkpoint step={dynamics.nsteps} exceeds configured "
            f"continuous trajectory end={total_steps}."
        )
    sample_interval = config.transition.sample_interval
    while dynamics.nsteps < total_steps:
        chunk_end = min(
            int(dynamics.nsteps) + config.execution.chunk_steps, total_steps
        )
        while dynamics.nsteps < chunk_end:
            current = int(dynamics.nsteps)
            next_sample = (current // sample_interval + 1) * sample_interval
            boundary = min(chunk_end, next_sample)
            dynamics.run(boundary - current)
            if dynamics.nsteps == next_sample:
                trace_buffer.sample(atoms, int(dynamics.nsteps))
        if trace_buffer.step[-1] != dynamics.nsteps:
            trace_buffer.sample(atoms, int(dynamics.nsteps))
        trace = trace_buffer.finish(
            atom_count=len(atoms),
            context=f"{task.run_name} checkpoint at step={dynamics.nsteps}",
        )
        checkpoints.save(
            atoms=atoms,
            trace=trace,
            state=capture_mtk_state(dynamics),
            metadata={
                "schema_version": 1,
                "completed_global_step": int(dynamics.nsteps),
                "task": task.__dict__,
            },
        )
        progress(
            f"{task.run_name}: committed exact MTK checkpoint at step={dynamics.nsteps}"
        )

    continuous_trace = trace_buffer.finish(
        atom_count=len(atoms), context=f"{task.run_name} completed continuous trace"
    )
    equilibration_mask = continuous_trace.step <= branch.equilibration_steps
    production_mask = continuous_trace.step >= branch.equilibration_steps
    if not np.any(continuous_trace.step == branch.equilibration_steps):
        raise RuntimeError(
            f"{task.run_name}: no stored equilibration-boundary frame at "
            f"step={branch.equilibration_steps}."
        )
    equilibration_trace = _slice_trace(
        continuous_trace, equilibration_mask, step_offset=0
    )
    production_trace = _slice_trace(
        continuous_trace,
        production_mask,
        step_offset=branch.equilibration_steps,
    )
    endpoint_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if endpoint_forces.shape != (len(atoms), 3) or not np.isfinite(
        endpoint_forces
    ).all():
        raise FloatingPointError(
            f"{task.run_name}: endpoint forces must be finite with shape "
            f"{(len(atoms), 3)}, got {endpoint_forces.shape}."
        )
    return _write_raw_commit(
        config,
        provenance,
        task,
        atoms=atoms,
        equilibration_trace=equilibration_trace,
        production_trace=production_trace,
        endpoint_forces=endpoint_forces,
    )


def run_md_worker(
    config: TransitionCampaignConfig,
    *,
    worker_name: str,
    calculator: object | None = None,
    injected_calculator_identity: str | None = None,
    progress: Callable[[str], None] = print,
) -> bool:
    validate_transition_queue_identity(config)
    prepared = load_content_bound_prepared_interface(config)
    selected_calculator, provenance = select_calculator(
        config.runtime_generator,
        calculator=calculator,
        injected_calculator_identity=injected_calculator_identity,
    )
    provenance = bind_transition_campaign_execution_provenance(provenance)
    progress(
        f"{worker_name}: loaded one persistent calculator for the dynamic transition "
        f"queue; source_atoms={len(prepared.atoms)}"
    )
    while True:
        task = claim_md_task(config, worker_name=worker_name)
        if task is None:
            return False
        try:
            directory, digest = run_transition_task(
                config,
                task=task,
                prepared=prepared,
                calculator=selected_calculator,
                provenance=provenance,
                progress=progress,
            )
            complete_md_task(
                config,
                task=task,
                raw_directory=directory,
                raw_commit_sha256=digest,
            )
        except BaseException:
            error = traceback.format_exc()
            fail_task(config, task=task, error=error, analysis=False)
            raise RuntimeError(
                f"{worker_name}: MD task {task.run_name} failed; traceback persisted in "
                "transition_campaign.sqlite3."
            )


def _provenance_from_dict(value: object, *, context: str) -> ExecutionProvenance:
    if not isinstance(value, dict) or set(value) != {
        "calculator",
        "runtime",
        "producer_code",
    }:
        raise RuntimeError(f"{context}: invalid execution_provenance mapping.")
    calculator = value["calculator"]
    if not isinstance(calculator, dict):
        raise RuntimeError(f"{context}: calculator provenance must be a mapping.")
    calculator_value = dict(calculator)
    calculator_value["available_heads"] = tuple(calculator_value["available_heads"])
    return ExecutionProvenance(
        calculator=CalculatorProvenance(**calculator_value),
        runtime=dict(value["runtime"]),
        producer_code=dict(value["producer_code"]),
    )


def _task_identity_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "task_index": int(row["task_index"]),
        "run_name": str(row["run_name"]),
        "branch_index": int(row["branch_index"]),
        "branch_name": str(row["branch_name"]),
        "replica_index": int(row["replica_index"]),
        "configured_replica_seed": int(row["configured_replica_seed"]),
        "simulation_seed": int(row["simulation_seed"]),
    }


def _validated_raw_commit(
    config: TransitionCampaignConfig,
    *,
    row: dict[str, object],
    task_identity: dict[str, object],
) -> tuple[Path, dict[str, object], str]:
    run_name = str(task_identity["run_name"])
    raw_value = row.get("raw_directory")
    anchored_digest = row.get("raw_commit_sha256")
    if not isinstance(raw_value, str) or not isinstance(anchored_digest, str):
        raise RuntimeError(f"{run_name}: queue has no committed raw trajectory anchor.")
    raw = Path(raw_value)
    metadata, observed_digest = _validate_artifact_commit(
        raw,
        metadata_name="raw_commit.json",
        artifact_key="raw_artifacts_sha256",
        expected_artifacts=RAW_ARTIFACTS,
    )
    if observed_digest != anchored_digest:
        raise RuntimeError(
            f"{run_name}: SQLite raw digest={anchored_digest} differs from "
            f"observed={observed_digest}."
        )
    if (
        metadata.get("campaign_config") != config.to_dict()
        or metadata.get("source_evidence") != config.source_evidence
        or metadata.get("task") != task_identity
    ):
        raise RuntimeError(f"{raw}: raw commit identity differs from the active task.")
    return raw, metadata, observed_digest


def _analysis_directory(
    config: TransitionCampaignConfig, task: TransitionCampaignTask
) -> Path:
    return config.output_root / task.run_name


def analyze_transition_task(
    config: TransitionCampaignConfig,
    *,
    task: TransitionCampaignTask,
    prepared,
    prepared_phase_ids: np.ndarray,
    analysis_execution_provenance: dict[str, object],
    progress: Callable[[str], None] = print,
) -> tuple[Path, str]:
    validate_transition_queue_identity(config)
    final = _analysis_directory(config, task)
    row = campaign_row(config, task_index=task.task_index)
    raw, raw_metadata, raw_digest = _validated_raw_commit(
        config,
        row=row,
        task_identity=task.__dict__,
    )
    if final.exists():
        metadata, digest = _validate_artifact_commit(
            final,
            metadata_name="analysis_commit.json",
            artifact_key="analysis_artifacts_sha256",
            expected_artifacts=None,
        )
        if (
            metadata.get("campaign_config") != config.to_dict()
            or metadata.get("source_evidence") != config.source_evidence
            or metadata.get("task") != task.__dict__
            or metadata.get("raw_commit_sha256") != raw_digest
            or metadata.get("analysis_execution_provenance")
            != analysis_execution_provenance
        ):
            raise RuntimeError(f"{final}: committed analysis task identity mismatch.")
        progress(f"{task.run_name}: recovered atomically committed deferred analysis")
        return final, digest

    provenance = _provenance_from_dict(
        raw_metadata.get("execution_provenance"), context=str(raw / "raw_commit.json")
    )
    atoms = read(raw / "endpoint.traj", format="traj")
    forces = np.load(raw / "endpoint_forces.npy")
    atoms.calc = SinglePointCalculator(atoms, forces=forces)
    equilibration_trace = _load_trace(
        raw / "equilibration_trajectory.npz",
        atom_count=len(atoms),
        context=f"{task.run_name} raw equilibration trace",
    )
    trace = _load_trace(
        raw / "trajectory.npz",
        atom_count=len(atoms),
        context=f"{task.run_name} raw production trace",
    )
    branch = _branch_for_task(config, task)
    transition = config.transition
    analysis = analyze_transition(
        trace,
        equilibration_trace=equilibration_trace,
        chemical_symbol=transition.generator.system.chemical_symbol,
        timestep_fs=transition.generator.dynamics.timestep_fs,
        slab_bounds_fractional=prepared.slab_bounds_fractional,
        profile_bins=transition.analysis.profile_bins,
        profile_smoothing_bins=transition.analysis.profile_smoothing_bins,
        ptm_rmsd_cutoff=transition.analysis.ptm_rmsd_cutoff,
        minimum_profile_contrast=transition.analysis.minimum_profile_contrast,
        minimum_velocity_fit_r_squared=(
            transition.analysis.minimum_velocity_fit_r_squared
        ),
        target_pressure_GPa=transition.generator.dynamics.pressure_GPa,
        maximum_temperature_error_K=(
            transition.generator.validation.maximum_temperature_error_K
        ),
        maximum_pressure_error_GPa=(
            transition.generator.validation.maximum_pressure_error_GPa
        ),
        branch=branch,
        progress=progress,
    )
    phase_rdf = analyze_phase_rdf(
        trace,
        chemical_symbol=transition.generator.system.chemical_symbol,
        prepared_phase_ids=prepared_phase_ids,
        timestep_fs=transition.generator.dynamics.timestep_fs,
        cutoff_A=transition.analysis.rdf_cutoff_A,
        bins=transition.analysis.rdf_bins,
        branch_name=task.run_name,
        progress=progress,
    )
    runtime_transition = replace(transition, generator=config.runtime_generator)
    diagnostics = diagnose_system(
        atoms,
        trace,
        _runtime_generator_config(runtime_transition, branch),
        name=task.run_name,
        require_pressure_convergence=True,
    )
    result = TransitionBranchResult(
        branch=branch,
        replica_index=task.replica_index,
        configured_replica_seed=task.configured_replica_seed,
        simulation_seed=task.simulation_seed,
        atoms=atoms,
        equilibration_trace=equilibration_trace,
        trace=trace,
        analysis=analysis,
        phase_rdf=phase_rdf,
        diagnostics=diagnostics,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.analysis-staging-", dir=final.parent)
    )
    staging = staging_root / "run"
    try:
        _write_branch(
            staging,
            prepared=prepared,
            result=result,
            config=transition,
            execution_provenance=provenance,
        )
        artifacts = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        commit = {
            "schema_version": 1,
            "campaign_config": config.to_dict(),
            "source_evidence": config.source_evidence,
            "task": task.__dict__,
            "raw_commit_sha256": raw_digest,
            "analysis_execution_provenance": analysis_execution_provenance,
            "analysis_artifacts_sha256": artifacts,
        }
        with (staging / "analysis_commit.json").open("w", encoding="utf-8") as handle:
            json.dump(commit, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        staging.replace(final)
        staging_root.rmdir()
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return final, _sha256(final / "analysis_commit.json")


def run_analysis_worker(
    config: TransitionCampaignConfig,
    *,
    worker_name: str,
    progress: Callable[[str], None] = print,
) -> bool:
    validate_transition_queue_identity(config)
    prepared = load_content_bound_prepared_interface(config)
    analysis_execution_provenance = (
        build_transition_deferred_analysis_provenance().to_dict()
    )
    prepared_phase_ids = np.fromiter(
        (PHASE_TO_ID[str(name)] for name in prepared.labels.phase_names),
        dtype=np.int64,
        count=len(prepared.atoms),
    )
    while True:
        task = claim_analysis_task(config, worker_name=worker_name)
        if task is None:
            return False
        try:
            directory, digest = analyze_transition_task(
                config,
                task=task,
                prepared=prepared,
                prepared_phase_ids=prepared_phase_ids,
                analysis_execution_provenance=analysis_execution_provenance,
                progress=progress,
            )
            complete_analysis_task(
                config,
                task=task,
                analysis_directory=directory,
                analysis_commit_sha256=digest,
            )
        except BaseException:
            error = traceback.format_exc()
            fail_task(config, task=task, error=error, analysis=True)
            raise RuntimeError(
                f"{worker_name}: analysis task {task.run_name} failed; traceback persisted "
                "in transition_campaign.sqlite3."
            )


def _velocity_summary(
    config: TransitionCampaignConfig,
    rows: list[dict[str, object]],
    *,
    raw_commits: dict[str, dict[str, object]],
    analysis_commits: dict[str, dict[str, object]],
) -> dict[str, object]:
    first_provenance: dict[str, object] | None = None
    analysis_provenance: dict[str, object] | None = None
    prepared = load_content_bound_prepared_interface(config)
    results: dict[str, object] = {}
    for row in rows:
        run_name = str(row["run_name"])
        directory = Path(str(row["analysis_directory"]))
        with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        transition = metadata["transition"]
        replica = metadata["replica"]
        with np.load(directory / "transition_progress.npz") as stored:
            progress_step = np.asarray(stored["step"]).copy()
            profile_contrast = np.asarray(stored["profile_contrast"]).copy()
        with np.load(directory / "trajectory.npz") as stored:
            production_cell = np.asarray(stored["cell_vectors_A"]).copy()
        fit_start, fit_end = transition["velocity_fit_step_interval"]
        branch = config.transition.temperature_runs[int(row["branch_index"])]
        results[run_name] = SimpleNamespace(
            branch=branch,
            replica_index=int(replica["index"]),
            configured_replica_seed=int(replica["configured_replica_seed"]),
            simulation_seed=int(replica["simulation_seed"]),
            atoms=prepared.atoms,
            trace=SimpleNamespace(cell_vectors_A=production_cell),
            analysis=SimpleNamespace(
                fitted_interface_velocity_m_per_s=float(
                    transition["fitted_interface_velocity_m_per_s"]
                ),
                individual_interface_velocities_m_per_s=np.asarray(
                    transition["individual_interface_velocities_m_per_s"],
                    dtype=np.float64,
                ),
                individual_interface_fit_r_squared=np.asarray(
                    transition["individual_interface_fit_r_squared"],
                    dtype=np.float64,
                ),
                velocity_fit_r_squared=float(transition["velocity_fit_r_squared"]),
                velocity_fit_ols_standard_error_m_per_s=float(
                    transition["velocity_fit_ols_standard_error_m_per_s"]
                ),
                velocity_fit_residual_rms_A=float(
                    transition["velocity_fit_residual_rms_A"]
                ),
                profile_contrast=profile_contrast,
                step=progress_step,
                velocity_fit_start_step=int(fit_start),
                velocity_fit_end_step=int(fit_end),
            ),
        )
        provenance = raw_commits[run_name]["execution_provenance"]
        if first_provenance is None:
            first_provenance = provenance
        elif provenance != first_provenance:
            raise RuntimeError(
                "Transition tasks have differing MD execution provenance; refusing to "
                "pool replica velocities."
            )
        observed_analysis = analysis_commits[run_name][
            "analysis_execution_provenance"
        ]
        if analysis_provenance is None:
            analysis_provenance = observed_analysis
        elif observed_analysis != analysis_provenance:
            raise RuntimeError(
                "Transition tasks have differing deferred-analysis execution provenance; "
                "refusing to pool replica velocities."
            )
    if first_provenance is None:
        raise RuntimeError("No completed transition provenance was found.")
    if analysis_provenance is None:
        raise RuntimeError("No completed deferred-analysis provenance was found.")
    md_provenance = _provenance_from_dict(
        first_provenance, context="finalized transition MD execution provenance"
    )
    summary = _generator_velocity_summary(
        config.transition,
        results,
        md_provenance,
        prepared,
        config.output_root,
    )
    summary.update(
        {
            "campaign_config": config.to_dict(),
            "campaign_source_evidence": config.source_evidence,
            "md_execution_provenance": first_provenance,
            "analysis_execution_provenance": analysis_provenance,
            "runtime_generator": {
                "config_file": str(config.runtime_generator.config_path),
                "config_file_sha256": _sha256(config.runtime_generator.config_path),
                "config_sha256": hashlib.sha256(
                    json.dumps(
                        config.runtime_generator.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "campaign_execution": asdict(config.execution),
        }
    )
    return summary


def finalize_transition_campaign(config: TransitionCampaignConfig) -> Path:
    rows = campaign_rows(config)
    unfinished = [
        row["run_name"]
        for row in rows
        if row["md_status"] != "complete" or row["analysis_status"] != "complete"
    ]
    if unfinished:
        raise RuntimeError(f"Cannot finalize transition campaign; unfinished={unfinished}.")
    raw_commits: dict[str, dict[str, object]] = {}
    analysis_commits: dict[str, dict[str, object]] = {}
    active_analysis_provenance = (
        build_transition_deferred_analysis_provenance().to_dict()
    )
    for row in rows:
        run_name = str(row["run_name"])
        task_identity = _task_identity_from_row(row)
        _, raw_metadata, raw_digest = _validated_raw_commit(
            config,
            row=row,
            task_identity=task_identity,
        )
        directory = Path(str(row["analysis_directory"]))
        analysis_metadata, digest = _validate_artifact_commit(
            directory,
            metadata_name="analysis_commit.json",
            artifact_key="analysis_artifacts_sha256",
            expected_artifacts=None,
        )
        if digest != row["analysis_commit_sha256"]:
            raise RuntimeError(
                f"{row['run_name']}: SQLite analysis digest differs from committed output."
            )
        if (
            analysis_metadata.get("campaign_config") != config.to_dict()
            or analysis_metadata.get("source_evidence") != config.source_evidence
            or analysis_metadata.get("task") != task_identity
            or analysis_metadata.get("raw_commit_sha256") != raw_digest
            or not isinstance(
                analysis_metadata.get("analysis_execution_provenance"), dict
            )
            or analysis_metadata.get("analysis_execution_provenance")
            != active_analysis_provenance
        ):
            raise RuntimeError(
                f"{run_name}: analysis commit is not bound to the active campaign, "
                "source, task, raw commit, and current deferred-analysis provenance."
            )
        raw_commits[run_name] = raw_metadata
        analysis_commits[run_name] = analysis_metadata
    summary = _velocity_summary(
        config,
        rows,
        raw_commits=raw_commits,
        analysis_commits=analysis_commits,
    )
    _write_json_atomic(config.output_root / "velocity_summary.json", summary)
    if config.transition.output.create_visualizations:
        _write_velocity_summary_visualization(
            config.output_root / "transition_overview.png", summary
        )
        first = {
            branch.name: config.output_root / branch.name / "replica_000" / "visualizations"
            for branch in config.transition.temperature_runs
        }
        write_phase_rdf_overview(
            config.output_root / "phase_rdf_overview.png",
            {name: directory / "phase_rdf.png" for name, directory in first.items()},
        )
        write_structure_slice_overview(
            config.output_root / "structure_slice_overview.png",
            {name: directory / "structure_slice.png" for name, directory in first.items()},
        )
    manifest = {
        "schema_version": 3,
        "dataset_name": config.transition.dataset_name,
        "run_dirs": [row["run_name"] for row in rows],
        "campaign_config": config.to_dict(),
        "source_evidence": config.source_evidence,
        "md_execution_provenance": summary["md_execution_provenance"],
        "analysis_execution_provenance": summary[
            "analysis_execution_provenance"
        ],
        "potential_sha256": summary["calculator"]["model_sha256"],
        "potential_usage_mode": summary["calculator"]["usage_mode"],
        "scientifically_qualified_potential": summary["calculator"][
            "scientifically_qualified"
        ],
        "velocity_summary": "velocity_summary.json",
        "execution_features": {
            "persistent_model_per_gpu_worker": True,
            "dynamic_temperature_replica_queue": True,
            "exact_mtk_state_checkpoint_resume": True,
            "atomic_per_run_raw_and_analysis_commits": True,
            "deferred_cpu_ptm_rdf_analysis": True,
        },
        "scientific_scope": {
            "supported_claim": (
                "Replica statistics for spatially tracked seeded planar-interface "
                "velocities under the selected calculator and finite protocol."
            ),
            "unsupported_claim": (
                "Homogeneous nucleation rates or potential-independent kinetics. A "
                "zero-velocity interpolation remains conditional on cell size, "
                "orientation, duration, PTM coordinate, and MLIP."
            ),
        },
    }
    manifest_path = config.output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _write_status(
    config: TransitionCampaignConfig, *, status: str, detail: object | None = None
) -> None:
    _write_json_atomic(
        config.output_root / "campaign_status.json",
        {
            "status": status,
            "written_at_epoch": time.time(),
            "detail": detail,
            "tasks": campaign_rows(config),
        },
    )


def _worker_command(
    config: TransitionCampaignConfig, *, role: str, worker_name: str
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "src.data_utils.synthetic.atomistic_transition_campaign",
        role,
        "--config",
        str(config.config_path),
        "--worker-name",
        worker_name,
    ]


def _spawn(
    *, command: list[str], log_path: Path, environment: dict[str, str]
) -> tuple[subprocess.Popen[bytes], object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    handle.write(
        f"\n=== started epoch={time.time():.6f}; command={command!r} ===\n".encode()
    )
    handle.flush()
    parent_pid = os.getpid()

    def terminate_with_parent() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, "prctl(PR_SET_PDEATHSIG, SIGTERM) failed")
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=environment,
        preexec_fn=terminate_with_parent,
    )
    return process, handle


def _run_processes(
    processes: list[tuple[str, subprocess.Popen[bytes], object]]
) -> dict[str, int]:
    codes: dict[str, int] = {}
    for name, process, handle in processes:
        try:
            codes[name] = process.wait()
        finally:
            handle.close()
    return codes


def run_transition_campaign(
    config: TransitionCampaignConfig,
    *,
    devices: tuple[str, ...],
    retry_failed: bool = False,
) -> None:
    if not devices or any(not item.strip() for item in devices):
        raise ValueError(f"At least one non-empty CUDA device is required, got {devices}.")
    if len(set(devices)) != len(devices):
        raise ValueError(f"CUDA devices must be unique, got {devices}.")
    config.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.output_root / "transition_campaign.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Transition campaign is already running: {lock_path}.") from exc
        initialize_transition_queue(config, retry_failed=retry_failed)
        rows = campaign_rows(config)
        queued = sum(row["md_status"] == "queued" for row in rows)
        _write_status(config, status="md_running", detail={"devices": list(devices)})
        base = os.environ.copy()
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
        for device in devices[:queued]:
            name = f"transition_md_gpu_{device}"
            environment = base.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            process, handle = _spawn(
                command=_worker_command(config, role="worker", worker_name=name),
                log_path=config.output_root / "logs" / f"{name}.log",
                environment=environment,
            )
            processes.append((name, process, handle))
        codes = _run_processes(processes)
        rows = campaign_rows(config)
        failures = [row for row in rows if row["md_status"] == "failed"]
        nonzero = {name: code for name, code in codes.items() if code}
        if failures or nonzero:
            detail = {"process_exit_codes": nonzero, "failed_tasks": failures}
            _write_status(config, status="failed", detail=detail)
            raise RuntimeError(
                f"Transition MD workers failed: {detail}. Inspect "
                f"{config.output_root / 'logs'}."
            )
        unfinished = [row["run_name"] for row in rows if row["md_status"] != "complete"]
        if unfinished:
            _write_status(config, status="paused", detail={"unfinished_md": unfinished})
            return
        _write_status(
            config,
            status="awaiting_offline_analysis",
            detail="Run the analyze command with explicit CPU worker count.",
        )


def run_deferred_transition_analysis(
    config: TransitionCampaignConfig,
    *,
    workers: int,
    retry_failed: bool = False,
) -> Path:
    if workers <= 0:
        raise ValueError(f"Analysis workers must be positive, got {workers}.")
    config.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.output_root / "transition_campaign.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Transition campaign is already running: {lock_path}.") from exc
        initialize_transition_queue(config, retry_failed=retry_failed)
        rows = campaign_rows(config)
        unfinished = [row["run_name"] for row in rows if row["md_status"] != "complete"]
        if unfinished:
            raise RuntimeError(
                f"Deferred analysis requires completed MD; unfinished={unfinished}."
            )
        pending = sum(row["analysis_status"] == "pending" for row in rows)
        _write_status(config, status="analysis_running", detail={"workers": workers})
        base = os.environ.copy()
        processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
        for index in range(min(workers, pending)):
            name = f"transition_analysis_{index:02d}"
            process, handle = _spawn(
                command=_worker_command(config, role="analyzer", worker_name=name),
                log_path=config.output_root / "logs" / f"{name}.log",
                environment=base,
            )
            processes.append((name, process, handle))
        codes = _run_processes(processes)
        rows = campaign_rows(config)
        failures = [row for row in rows if row["analysis_status"] == "failed"]
        nonzero = {name: code for name, code in codes.items() if code}
        if failures or nonzero:
            detail = {"process_exit_codes": nonzero, "failed_tasks": failures}
            _write_status(config, status="failed", detail=detail)
            raise RuntimeError(f"Deferred transition analysis failed: {detail}.")
        manifest = finalize_transition_campaign(config)
        _write_status(config, status="complete")
        return manifest
