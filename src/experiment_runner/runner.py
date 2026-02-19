"""Core orchestrator: executes an experiment plan stage by stage."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .plan import ExperimentPlan, StageSpec
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
) -> None:
    """Execute an experiment plan end to end."""
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
            )
        else:
            _run_stage_slurm(
                stage=stage, plan=plan, output_dir=output_dir,
                repo_root=repo_root, stage_state=stage_state,
                extra_overrides=extra_overrides,
                dry_run=dry_run, continue_on_error=continue_on_error,
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
) -> None:
    from .slurm import rebuild_jobs_from_state, submit_stage, wait_for_jobs

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
        n_resume = len(jobs)
        for j in jobs:
            print(f"  [RESUMED] {j.experiment.name} (job {j.job_id}, was {j.status})")

        # Submit any experiments that were NOT yet submitted (e.g. runner
        # crashed mid-submission and only some jobs got through).
        submitted_names = set(already_submitted.keys())
        remaining_exps = [e for e in stage.experiments if e.name not in submitted_names]
        if remaining_exps:
            from .plan import StageSpec as _SS
            partial_stage = StageSpec(
                name=stage.name,
                config_name=stage.config_name,
                train_script=stage.train_script,
                base_overrides=stage.base_overrides,
                experiments=remaining_exps,
                depends_on=stage.depends_on,
                inherit_checkpoint=stage.inherit_checkpoint,
                slurm=stage.slurm,
            )
            new_jobs = submit_stage(
                stage=partial_stage, plan=plan, output_dir=output_dir,
                repo_root=repo_root,
                extra_overrides_per_experiment=extra_overrides,
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
            extra_overrides_per_experiment=extra_overrides,
            dry_run=dry_run,
        )
        for job in jobs:
            stage_state.record_job(
                job.experiment.name, job_id=job.job_id,
                run_dir=job.run_dir, status=job.status,
            )

    # Persist job IDs immediately so a second --resume can find them.
    from .state import RunState
    state_path = output_dir / "state.json"
    if state_path.exists():
        rs = RunState.load(output_dir)
        rs.stages[stage.name] = stage_state
        rs.save()

    if not dry_run:
        print(f"\n  Waiting for {len(jobs)} SLURM job(s) to complete...")
        wait_for_jobs(jobs, continue_on_error=continue_on_error)

        for job in jobs:
            js = stage_state.jobs.get(job.experiment.name)
            if js:
                js.status = job.status


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
) -> None:
    from .local import run_stage_local

    jobs = run_stage_local(
        stage=stage,
        plan=plan,
        output_dir=output_dir,
        repo_root=repo_root,
        extra_overrides_per_experiment=extra_overrides,
        parallel=parallel,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )

    for job in jobs:
        stage_state.record_job(
            job.experiment.name,
            run_dir=job.run_dir,
            status=job.status,
        )


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
