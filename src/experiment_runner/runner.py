"""Core orchestrator: executes an experiment plan stage by stage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .plan import ExperimentPlan, StageSpec
from .restart import (
    build_scaled_lr_override,
    detect_nan_loss_in_logs,
    resolve_learning_rate,
    validate_nan_restart_settings,
)
from .results import (
    collect_all_results,
    find_best_checkpoint,
    print_summary_table,
    write_csv_tables,
    write_summary_json,
)
from .state import RunState


def resolve_checkpoints(
    stage: StageSpec,
    output_dir: Path,
    run_state: RunState,
) -> Dict[str, List[str]]:
    """For a stage with inherit_checkpoint=True, resolve checkpoints from
    the dependency stage and return extra overrides per experiment name."""
    if not stage.inherit_checkpoint or stage.depends_on is None:
        return {}

    dep_stage_state = run_state.stages.get(stage.depends_on)
    if dep_stage_state is None:
        raise RuntimeError(
            f"Stage {stage.name!r} depends on {stage.depends_on!r}, "
            f"but no state was recorded for it."
        )

    overrides: Dict[str, List[str]] = {}
    for exp in stage.experiments:
        dep_job = dep_stage_state.jobs.get(exp.name)
        if dep_job is None:
            raise RuntimeError(
                f"Stage {stage.name!r}, experiment {exp.name!r}: "
                f"no matching job in dependency stage {stage.depends_on!r}."
            )

        # Try cached checkpoint path first, then search the run dir.
        ckpt_path: Optional[Path] = None
        if dep_job.checkpoint_path:
            ckpt_path = Path(dep_job.checkpoint_path)
            if not ckpt_path.exists():
                ckpt_path = None

        if ckpt_path is None and dep_job.run_dir:
            ckpt_path = find_best_checkpoint(Path(dep_job.run_dir))

        if ckpt_path is None:
            raise RuntimeError(
                f"Could not find a checkpoint for experiment {exp.name!r} "
                f"from stage {stage.depends_on!r} (run_dir={dep_job.run_dir})."
            )

        dep_job.checkpoint_path = str(ckpt_path)
        overrides[exp.name] = [f"++init_from_checkpoint={ckpt_path}"]

    return overrides


def run_plan(
    plan: ExperimentPlan,
    *,
    output_dir: Path,
    repo_root: Path,
    local: bool = False,
    parallel: int = 1,
    dry_run: bool = False,
    continue_on_error: bool = False,
    resume_state: Optional[RunState] = None,
    nan_restart_max_retries: int = 1,
    nan_restart_lr_factor: float = 0.7,
) -> None:
    """Execute an experiment plan end to end."""
    validate_nan_restart_settings(
        max_retries=nan_restart_max_retries,
        lr_factor=nan_restart_lr_factor,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    state = resume_state or RunState(output_dir=output_dir, plan_name=plan.name)
    state.save()

    for stage in plan.stages:
        if state.is_stage_complete(stage.name):
            print(f"\nStage '{stage.name}': already complete (from resumed state), skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Stage: {stage.name}  ({len(stage.experiments)} experiments)")
        print(f"{'='*60}")

        stage_state = state.get_or_create_stage(stage.name)
        stage_state.status = "running"
        stage_state.started_at = datetime.now().isoformat()
        state.save()

        # Resolve checkpoint overrides from dependency stage (skip during dry-run
        # since no actual checkpoints exist yet).
        if dry_run and stage.inherit_checkpoint:
            extra_overrides: Dict[str, List[str]] = {
                exp.name: ["++init_from_checkpoint=<CHECKPOINT_FROM_STAGE:"
                           f"{stage.depends_on}/{exp.name}>"]
                for exp in stage.experiments
            }
        else:
            extra_overrides = resolve_checkpoints(stage, output_dir, state)

        if local:
            _run_stage_local(
                stage=stage, plan=plan, output_dir=output_dir,
                repo_root=repo_root, stage_state=stage_state,
                extra_overrides=extra_overrides,
                parallel=parallel, dry_run=dry_run,
                continue_on_error=continue_on_error,
                nan_restart_max_retries=nan_restart_max_retries,
                nan_restart_lr_factor=nan_restart_lr_factor,
            )
        else:
            _run_stage_slurm(
                stage=stage, plan=plan, output_dir=output_dir,
                repo_root=repo_root, stage_state=stage_state,
                extra_overrides=extra_overrides,
                dry_run=dry_run, continue_on_error=continue_on_error,
                nan_restart_max_retries=nan_restart_max_retries,
                nan_restart_lr_factor=nan_restart_lr_factor,
            )

        stage_state.finished_at = datetime.now().isoformat()
        failed_count = sum(
            1 for j in stage_state.jobs.values()
            if j.status not in {"completed", "dry_run", "COMPLETED"}
        )
        if failed_count > 0:
            stage_state.status = "completed_with_errors"
            print(f"\n  Stage '{stage.name}' finished with {failed_count} failed experiment(s).")
        else:
            stage_state.status = "completed"
            print(f"\n  Stage '{stage.name}' completed successfully.")
        state.save()

    # --- Collect and report results ---
    if not dry_run:
        _collect_and_report(plan, output_dir)


def _run_stage_slurm(
    *,
    stage: StageSpec,
    plan: ExperimentPlan,
    output_dir: Path,
    repo_root: Path,
    stage_state,
    extra_overrides: Dict[str, List[str]],
    dry_run: bool,
    continue_on_error: bool,
    nan_restart_max_retries: int,
    nan_restart_lr_factor: float,
) -> None:
    from .slurm import rebuild_jobs_from_state, submit_stage, wait_for_jobs

    exp_by_name = {exp.name: exp for exp in stage.experiments}
    effective_overrides: Dict[str, List[str]] = {
        exp.name: list(extra_overrides.get(exp.name, []))
        for exp in stage.experiments
    }
    retry_counts = {exp.name: 0 for exp in stage.experiments}

    # Check if jobs were already submitted (resume scenario: the polling loop
    # was interrupted but the SLURM jobs are still running/queued).
    already_submitted = {
        name: js for name, js in stage_state.jobs.items()
        if js.job_id is not None
    }

    if already_submitted:
        saved_dicts = {
            name: {"job_id": js.job_id, "run_dir": js.run_dir, "status": js.status}
            for name, js in already_submitted.items()
        }
        jobs = rebuild_jobs_from_state(stage, saved_dicts)
        for j in jobs:
            print(f"  [RESUMED] {j.experiment.name} (job {j.job_id}, was {j.status})")

        # Submit any experiments that were NOT yet submitted (e.g. runner
        # crashed mid-submission and only some jobs got through).
        submitted_names = set(already_submitted.keys())
        remaining_exps = [e for e in stage.experiments if e.name not in submitted_names]
        if remaining_exps:
            partial_stage = _clone_stage_with_experiments(stage, remaining_exps)
            new_jobs = submit_stage(
                stage=partial_stage, plan=plan, output_dir=output_dir,
                repo_root=repo_root,
                extra_overrides_per_experiment=effective_overrides,
                dry_run=dry_run,
            )
            for job in new_jobs:
                stage_state.record_job(
                    job.experiment.name, job_id=job.job_id,
                    run_dir=job.run_dir, status=job.status,
                )
            jobs.extend(new_jobs)
    else:
        jobs = submit_stage(
            stage=stage, plan=plan, output_dir=output_dir,
            repo_root=repo_root,
            extra_overrides_per_experiment=effective_overrides,
            dry_run=dry_run,
        )
        for job in jobs:
            stage_state.record_job(
                job.experiment.name, job_id=job.job_id,
                run_dir=job.run_dir, status=job.status,
            )

    # Persist job IDs immediately so a second --resume can find them.
    _persist_stage_state(output_dir=output_dir, stage_name=stage.name, stage_state=stage_state)

    if not dry_run:
        print(f"\n  Waiting for {len(jobs)} SLURM job(s) to complete...")
        wait_for_jobs(
            jobs,
            continue_on_error=(continue_on_error or nan_restart_max_retries > 0),
        )

    for job in jobs:
        js = stage_state.jobs.get(job.experiment.name)
        if js:
            js.status = job.status
    _persist_stage_state(output_dir=output_dir, stage_name=stage.name, stage_state=stage_state)

    if not dry_run and nan_restart_max_retries > 0:
        retry_experiments = _collect_nan_retry_experiments_slurm(
            jobs=jobs,
            stage=stage,
            plan=plan,
            output_dir=output_dir,
            repo_root=repo_root,
            effective_overrides=effective_overrides,
            exp_by_name=exp_by_name,
            retry_counts=retry_counts,
            nan_restart_max_retries=nan_restart_max_retries,
            nan_restart_lr_factor=nan_restart_lr_factor,
        )

        while retry_experiments:
            partial_stage = _clone_stage_with_experiments(stage, retry_experiments)
            retry_jobs = submit_stage(
                stage=partial_stage,
                plan=plan,
                output_dir=output_dir,
                repo_root=repo_root,
                extra_overrides_per_experiment=effective_overrides,
                dry_run=False,
            )

            for retry_job in retry_jobs:
                stage_state.record_job(
                    retry_job.experiment.name,
                    job_id=retry_job.job_id,
                    run_dir=retry_job.run_dir,
                    status=retry_job.status,
                )
            _persist_stage_state(output_dir=output_dir, stage_name=stage.name, stage_state=stage_state)

            print(f"\n  Waiting for {len(retry_jobs)} NaN-retry SLURM job(s) to complete...")
            wait_for_jobs(retry_jobs, continue_on_error=True)

            for retry_job in retry_jobs:
                js = stage_state.jobs.get(retry_job.experiment.name)
                if js:
                    js.status = retry_job.status
            _persist_stage_state(output_dir=output_dir, stage_name=stage.name, stage_state=stage_state)

            retry_experiments = _collect_nan_retry_experiments_slurm(
                jobs=retry_jobs,
                stage=stage,
                plan=plan,
                output_dir=output_dir,
                repo_root=repo_root,
                effective_overrides=effective_overrides,
                exp_by_name=exp_by_name,
                retry_counts=retry_counts,
                nan_restart_max_retries=nan_restart_max_retries,
                nan_restart_lr_factor=nan_restart_lr_factor,
            )

    unresolved = [
        js for js in stage_state.jobs.values()
        if not _is_success_status(js.status)
    ]
    if unresolved and not continue_on_error and not dry_run:
        details = ", ".join(
            f"{job.experiment_name} ({job.status})"
            for job in unresolved
        )
        raise RuntimeError(
            f"Stage {stage.name!r} has unresolved SLURM failures after NaN-restart "
            f"handling: {details}. Use --continue-on-error to proceed."
        )


def _run_stage_local(
    *,
    stage: StageSpec,
    plan: ExperimentPlan,
    output_dir: Path,
    repo_root: Path,
    stage_state,
    extra_overrides: Dict[str, List[str]],
    parallel: int,
    dry_run: bool,
    continue_on_error: bool,
    nan_restart_max_retries: int,
    nan_restart_lr_factor: float,
) -> None:
    from .local import run_stage_local

    exp_by_name = {exp.name: exp for exp in stage.experiments}
    effective_overrides: Dict[str, List[str]] = {
        exp.name: list(extra_overrides.get(exp.name, []))
        for exp in stage.experiments
    }
    retry_counts = {exp.name: 0 for exp in stage.experiments}
    pending_experiments = list(stage.experiments)
    final_jobs = {}

    while pending_experiments:
        partial_stage = _clone_stage_with_experiments(stage, pending_experiments)
        jobs = run_stage_local(
            stage=partial_stage,
            plan=plan,
            output_dir=output_dir,
            repo_root=repo_root,
            extra_overrides_per_experiment=effective_overrides,
            parallel=parallel,
            dry_run=dry_run,
            continue_on_error=(continue_on_error or nan_restart_max_retries > 0),
        )

        retry_next: List = []
        for job in jobs:
            exp_name = job.experiment.name
            final_jobs[exp_name] = job
            if dry_run or job.status != "failed":
                continue
            if retry_counts[exp_name] >= nan_restart_max_retries:
                continue

            log_match = detect_nan_loss_in_logs(_local_log_candidates(job.run_dir))
            if log_match is None:
                continue

            context = (
                f"stage={stage.name!r}, experiment={exp_name!r}, "
                f"retry_attempt={retry_counts[exp_name] + 1}"
            )
            current_lr = resolve_learning_rate(
                repo_root=repo_root,
                config_name=partial_stage.config_name or plan.config_name,
                overrides=(
                    list(stage.base_overrides)
                    + list(exp_by_name[exp_name].overrides)
                    + list(effective_overrides[exp_name])
                ),
                context=context,
            )
            lr_override, new_lr = build_scaled_lr_override(
                current_lr=current_lr,
                lr_factor=nan_restart_lr_factor,
            )
            effective_overrides[exp_name] = [
                *effective_overrides[exp_name],
                lr_override,
            ]
            retry_counts[exp_name] += 1

            log_path, pattern = log_match
            print(
                f"  [RETRY] {exp_name}: detected NaN loss in {log_path} "
                f"(pattern: {pattern}). Retry {retry_counts[exp_name]}/"
                f"{nan_restart_max_retries} with learning_rate={new_lr:.12g}"
            )
            retry_next.append(exp_by_name[exp_name])

        if not retry_next:
            break
        pending_experiments = retry_next

    for exp in stage.experiments:
        job = final_jobs.get(exp.name)
        if job is None:
            raise RuntimeError(
                f"Stage {stage.name!r}, experiment {exp.name!r}: missing local job "
                "result after execution."
            )
        stage_state.record_job(
            job.experiment.name,
            run_dir=job.run_dir,
            status=job.status,
        )

    unresolved = [
        js for js in stage_state.jobs.values()
        if not _is_success_status(js.status)
    ]
    if unresolved and not continue_on_error:
        details = ", ".join(
            f"{job.experiment_name} ({job.status})"
            for job in unresolved
        )
        raise RuntimeError(
            f"Stage {stage.name!r} has unresolved local failures after NaN-restart "
            f"handling: {details}. Use --continue-on-error to proceed."
        )


def _collect_nan_retry_experiments_slurm(
    *,
    jobs,
    stage: StageSpec,
    plan: ExperimentPlan,
    output_dir: Path,
    repo_root: Path,
    effective_overrides: Dict[str, List[str]],
    exp_by_name,
    retry_counts: Dict[str, int],
    nan_restart_max_retries: int,
    nan_restart_lr_factor: float,
) -> List:
    retry_experiments = []
    for job in jobs:
        exp_name = job.experiment.name
        if _is_success_status(job.status):
            continue
        if retry_counts[exp_name] >= nan_restart_max_retries:
            continue

        log_match = detect_nan_loss_in_logs(
            _slurm_log_candidates(
                output_dir=output_dir,
                run_dir=job.run_dir,
                job_id=job.job_id,
            )
        )
        if log_match is None:
            continue

        context = (
            f"stage={stage.name!r}, experiment={exp_name!r}, "
            f"retry_attempt={retry_counts[exp_name] + 1}"
        )
        current_lr = resolve_learning_rate(
            repo_root=repo_root,
            config_name=stage.config_name or plan.config_name,
            overrides=(
                list(stage.base_overrides)
                + list(exp_by_name[exp_name].overrides)
                + list(effective_overrides[exp_name])
            ),
            context=context,
        )
        lr_override, new_lr = build_scaled_lr_override(
            current_lr=current_lr,
            lr_factor=nan_restart_lr_factor,
        )
        effective_overrides[exp_name] = [
            *effective_overrides[exp_name],
            lr_override,
        ]
        retry_counts[exp_name] += 1

        log_path, pattern = log_match
        print(
            f"  [RETRY] {exp_name}: detected NaN loss in {log_path} "
            f"(pattern: {pattern}). Retry {retry_counts[exp_name]}/"
            f"{nan_restart_max_retries} with learning_rate={new_lr:.12g}"
        )
        retry_experiments.append(exp_by_name[exp_name])

    return retry_experiments


def _clone_stage_with_experiments(stage: StageSpec, experiments: Sequence) -> StageSpec:
    return StageSpec(
        name=stage.name,
        config_name=stage.config_name,
        train_script=stage.train_script,
        base_overrides=list(stage.base_overrides),
        experiments=list(experiments),
        depends_on=stage.depends_on,
        inherit_checkpoint=stage.inherit_checkpoint,
        slurm=stage.slurm,
    )


def _persist_stage_state(*, output_dir: Path, stage_name: str, stage_state) -> None:
    from .state import RunState

    state_path = output_dir / "state.json"
    if not state_path.exists():
        return
    rs = RunState.load(output_dir)
    rs.stages[stage_name] = stage_state
    rs.save()


def _local_log_candidates(run_dir: Optional[str]) -> List[Path]:
    if run_dir is None:
        return []
    run_path = Path(run_dir)
    return [
        run_path / "train.log",
        run_path / "train_contrastive.log",
    ]


def _slurm_log_candidates(
    *,
    output_dir: Path,
    run_dir: Optional[str],
    job_id: Optional[str],
) -> List[Path]:
    candidates: List[Path] = []

    if run_dir is not None:
        run_path = Path(run_dir)
        candidates.extend([
            run_path / "train.log",
            run_path / "train_contrastive.log",
        ])

    if job_id is None:
        return candidates

    slurm_dir = output_dir / "slurm_logs"
    if not slurm_dir.exists():
        return candidates

    candidates.extend(sorted(slurm_dir.glob(f"*_{job_id}.err"), reverse=True))
    candidates.extend(sorted(slurm_dir.glob(f"*_{job_id}.out"), reverse=True))
    return candidates


def _is_success_status(status: str) -> bool:
    return status in {"completed", "dry_run", "COMPLETED"}


def _collect_and_report(plan: ExperimentPlan, output_dir: Path) -> None:
    print(f"\n{'='*60}")
    print("  Collecting results...")
    print(f"{'='*60}")

    results = collect_all_results(output_dir, plan)
    csv_paths = write_csv_tables(results, output_dir, plan.metrics)
    json_path = write_summary_json(results, output_dir, plan)

    print_summary_table(results, plan.metrics)

    print("Output files:")
    for p in csv_paths:
        print(f"  {p}")
    print(f"  {json_path}")


def collect_only(plan: ExperimentPlan, output_dir: Path) -> None:
    """Re-run only the result collection step on an existing output directory."""
    _collect_and_report(plan, output_dir)
