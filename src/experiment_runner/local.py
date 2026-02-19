"""Local subprocess execution backend."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .plan import Experiment, ExperimentPlan, StageSpec


@dataclass
class LocalJob:
    experiment: Experiment
    stage: str
    run_dir: Optional[str] = None
    status: str = "pending"  # pending | running | completed | failed
    return_code: Optional[int] = None
    duration_sec: Optional[float] = None
    error: Optional[str] = None


def build_command(
    *,
    experiment: Experiment,
    plan: ExperimentPlan,
    stage: StageSpec,
    run_dir: Path,
    extra_overrides: Optional[List[str]] = None,
) -> List[str]:
    """Build the full python command for one experiment."""
    train_script = stage.train_script or plan.train_script
    config_name = stage.config_name or plan.config_name
    if config_name is None:
        raise ValueError(
            f"No config_name for experiment {experiment.name!r} in stage {stage.name!r}."
        )

    exp_name = f"{plan.name}_{stage.name}_{experiment.name}"

    cmd = [
        "python",
        train_script,
        "--config-name", config_name,
    ]
    for ov in stage.base_overrides:
        cmd.append(ov)
    for ov in experiment.overrides:
        cmd.append(ov)
    cmd.append(f"hydra.run.dir={run_dir}")
    cmd.append(f"experiment_name={exp_name}")
    if extra_overrides:
        cmd.extend(extra_overrides)

    return cmd


def _run_one(
    *,
    command: List[str],
    run_dir: Path,
    experiment: Experiment,
    stage_name: str,
    repo_root: Path,
) -> LocalJob:
    """Execute a single experiment as a subprocess."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    job = LocalJob(
        experiment=experiment,
        stage=stage_name,
        run_dir=str(run_dir),
    )
    job.status = "running"
    print(f"  [RUNNING] {experiment.name}  ->  {run_dir}")

    start = time.perf_counter()
    try:
        with log_path.open("w") as log_handle:
            log_handle.write("COMMAND:\n" + " ".join(command) + "\n\n")
            log_handle.flush()
            proc = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        job.return_code = proc.returncode
        job.duration_sec = time.perf_counter() - start

        if proc.returncode == 0:
            job.status = "completed"
            print(f"  [OK] {experiment.name} ({job.duration_sec:.0f}s)")
        else:
            job.status = "failed"
            job.error = f"Exit code {proc.returncode}. See {log_path}"
            print(f"  [FAIL] {experiment.name} (exit {proc.returncode}, {job.duration_sec:.0f}s)")
    except Exception as exc:
        job.duration_sec = time.perf_counter() - start
        job.status = "failed"
        job.error = str(exc)
        print(f"  [ERROR] {experiment.name}: {exc}")

    return job


def run_stage_local(
    *,
    stage: StageSpec,
    plan: ExperimentPlan,
    output_dir: Path,
    repo_root: Path,
    extra_overrides_per_experiment: Optional[Dict[str, List[str]]] = None,
    parallel: int = 1,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> List[LocalJob]:
    """Run all experiments in a stage locally. Returns list of LocalJob."""
    jobs: List[LocalJob] = []

    tasks = []
    for exp in stage.experiments:
        run_dir = output_dir / stage.name / exp.name
        extra = (extra_overrides_per_experiment or {}).get(exp.name, [])
        cmd = build_command(
            experiment=exp, plan=plan, stage=stage,
            run_dir=run_dir, extra_overrides=extra,
        )
        tasks.append((cmd, run_dir, exp))

    if dry_run:
        for cmd, run_dir, exp in tasks:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            print(f"  [DRY-RUN] {exp.name}: {' '.join(cmd[:6])} ...")
            jobs.append(LocalJob(
                experiment=exp, stage=stage.name,
                run_dir=str(run_dir), status="dry_run",
            ))
        return jobs

    if parallel <= 1:
        for cmd, run_dir, exp in tasks:
            job = _run_one(
                command=cmd, run_dir=run_dir, experiment=exp,
                stage_name=stage.name, repo_root=repo_root,
            )
            jobs.append(job)
            if job.status == "failed" and not continue_on_error:
                raise RuntimeError(
                    f"Experiment {exp.name!r} failed: {job.error}. "
                    "Use --continue-on-error to proceed."
                )
    else:
        # Parallel local execution via ProcessPoolExecutor.
        # We submit as futures but still collect results as they complete.
        futures_map = {}
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            for cmd, run_dir, exp in tasks:
                future = pool.submit(
                    _run_one,
                    command=cmd,
                    run_dir=run_dir,
                    experiment=exp,
                    stage_name=stage.name,
                    repo_root=repo_root,
                )
                futures_map[future] = exp

            for future in as_completed(futures_map):
                exp = futures_map[future]
                try:
                    job = future.result()
                except Exception as exc:
                    job = LocalJob(
                        experiment=exp, stage=stage.name,
                        status="failed", error=str(exc),
                    )
                    print(f"  [ERROR] {exp.name}: {exc}")
                jobs.append(job)
                if job.status == "failed" and not continue_on_error:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"Experiment {exp.name!r} failed: {job.error}. "
                        "Use --continue-on-error to proceed."
                    )

    return jobs
