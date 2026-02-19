"""SLURM execution backend: sbatch generation, submission, and polling."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .plan import Experiment, ExperimentPlan, SlurmConfig, StageSpec

POLL_INITIAL_INTERVAL_SEC = 15
POLL_MAX_INTERVAL_SEC = 120
POLL_BACKOFF_FACTOR = 1.5


@dataclass
class SlurmJob:
    experiment: Experiment
    stage: str
    job_id: Optional[str] = None
    run_dir: Optional[str] = None
    sbatch_script: Optional[str] = None
    status: str = "pending"  # pending | submitted | running | completed | failed | cancelled


def generate_sbatch_script(
    *,
    experiment: Experiment,
    plan: ExperimentPlan,
    stage: StageSpec,
    run_dir: Path,
    repo_root: Path,
    log_dir: Path,
    extra_overrides: Optional[List[str]] = None,
) -> str:
    """Generate a self-contained sbatch script for one experiment."""
    slurm = stage.slurm or plan.slurm

    train_script = stage.train_script or plan.train_script
    config_name = stage.config_name or plan.config_name
    if config_name is None:
        raise ValueError(
            f"No config_name for experiment {experiment.name!r} in stage {stage.name!r}."
        )

    job_name = f"{plan.name}_{stage.name}_{experiment.name}"
    # SLURM doesn't like very long job names; truncate.
    if len(job_name) > 80:
        job_name = job_name[:77] + "..."

    all_overrides: List[str] = []
    all_overrides.extend(stage.base_overrides)
    all_overrides.extend(experiment.overrides)
    all_overrides.append(f"hydra.run.dir={run_dir}")
    all_overrides.append(f"experiment_name={_safe_experiment_name(plan.name, stage.name, experiment.name)}")
    if extra_overrides:
        all_overrides.extend(extra_overrides)

    overrides_str = " \\\n    ".join(f"'{ov}'" for ov in all_overrides)

    sbatch_lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir / '%x_%j.out'}",
        f"#SBATCH --error={log_dir / '%x_%j.err'}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --gres=gpu:{slurm.gpus}",
        f"#SBATCH --cpus-per-task={slurm.cpus}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --time={slurm.time}",
    ]
    for extra in slurm.extra_sbatch:
        sbatch_lines.append(f"#SBATCH {extra}")

    sbatch_block = "\n".join(sbatch_lines)

    script = f"""\
#!/bin/bash
{sbatch_block}

set -eo pipefail

echo "Experiment: {experiment.name}"
echo "Stage: {stage.name}"
echo "Node: $(hostname)"
echo "Started: $(date)"

set +u
source {slurm.conda_sh}
conda activate {slurm.conda_env}
set -u

cd {repo_root}
export PYTHONPATH="${{PYTHONPATH:-}}:{repo_root}"

python {train_script} \\
    --config-name {config_name} \\
    {overrides_str}

echo "Finished: $(date)"
"""
    return script


def _safe_experiment_name(plan_name: str, stage_name: str, exp_name: str) -> str:
    raw = f"{plan_name}_{stage_name}_{exp_name}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)[:120]


def submit_sbatch(script_content: str, script_path: Path) -> str:
    """Write an sbatch script to disk and submit it. Returns the SLURM job ID."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}\n"
            f"  script: {script_path}"
        )

    # sbatch prints "Submitted batch job 12345"
    match = re.search(r"(\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output: {result.stdout.strip()!r}"
        )
    return match.group(1)


def query_job_statuses(job_ids: Sequence[str]) -> Dict[str, str]:
    """Query SLURM for the status of given job IDs.

    Returns a dict mapping job_id -> state string (PENDING, RUNNING,
    COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.).
    """
    if not job_ids:
        return {}

    id_str = ",".join(job_ids)
    result = subprocess.run(
        ["sacct", "-j", id_str, "--format=JobID,State", "--noheader", "--parsable2"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        # Fallback to squeue if sacct is unavailable.
        return _query_via_squeue(job_ids)

    statuses: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        raw_id = parts[0].strip()
        state = parts[1].strip().split()[0]  # e.g. "COMPLETED by 0" -> "COMPLETED"
        # sacct may include sub-job entries like "12345.batch"; keep only the main entry.
        if "." in raw_id:
            continue
        if raw_id in {str(j) for j in job_ids}:
            statuses[raw_id] = state

    return statuses


def _query_via_squeue(job_ids: Sequence[str]) -> Dict[str, str]:
    id_str = ",".join(job_ids)
    result = subprocess.run(
        ["squeue", "--jobs=" + id_str, "--format=%i %T", "--noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    statuses: Dict[str, str] = {}
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                statuses[parts[0].strip()] = parts[1].strip()

    # Jobs not found in squeue have completed (or been removed).
    for jid in job_ids:
        if jid not in statuses:
            statuses[jid] = "COMPLETED"
    return statuses


def is_terminal_state(state: str) -> bool:
    return state.upper() in {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
        "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
    }


def is_success(state: str) -> bool:
    return state.upper() == "COMPLETED"


def wait_for_jobs(
    jobs: List[SlurmJob],
    *,
    continue_on_error: bool = False,
) -> None:
    """Block until all SLURM jobs reach a terminal state, polling with backoff."""
    active_ids = {j.job_id: j for j in jobs if j.job_id is not None}
    if not active_ids:
        return

    interval = POLL_INITIAL_INTERVAL_SEC
    while True:
        remaining = {jid: j for jid, j in active_ids.items() if not is_terminal_state(j.status)}
        if not remaining:
            break

        time.sleep(interval)
        interval = min(interval * POLL_BACKOFF_FACTOR, POLL_MAX_INTERVAL_SEC)

        statuses = query_job_statuses(list(remaining.keys()))
        for jid, state in statuses.items():
            job = remaining.get(jid)
            if job is None:
                continue
            job.status = state
            if is_terminal_state(state):
                symbol = "OK" if is_success(state) else "FAIL"
                print(f"  [{symbol}] {job.experiment.name} (job {jid}): {state}")

    failed = [j for j in jobs if j.job_id and not is_success(j.status)]
    if failed and not continue_on_error:
        names = ", ".join(f"{j.experiment.name} ({j.status})" for j in failed)
        raise RuntimeError(
            f"The following SLURM jobs did not complete successfully: {names}. "
            "Use --continue-on-error to proceed despite failures."
        )


def submit_stage(
    *,
    stage: StageSpec,
    plan: ExperimentPlan,
    output_dir: Path,
    repo_root: Path,
    extra_overrides_per_experiment: Optional[Dict[str, List[str]]] = None,
    dry_run: bool = False,
) -> List[SlurmJob]:
    """Submit all experiments in a stage as SLURM jobs. Returns list of SlurmJob."""
    log_dir = output_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[SlurmJob] = []
    for exp in stage.experiments:
        run_dir = output_dir / stage.name / exp.name
        run_dir.mkdir(parents=True, exist_ok=True)

        extra = (extra_overrides_per_experiment or {}).get(exp.name, [])
        script = generate_sbatch_script(
            experiment=exp,
            plan=plan,
            stage=stage,
            run_dir=run_dir,
            repo_root=repo_root,
            log_dir=log_dir,
            extra_overrides=extra,
        )
        script_path = run_dir / "job.sbatch"

        job = SlurmJob(
            experiment=exp,
            stage=stage.name,
            run_dir=str(run_dir),
            sbatch_script=str(script_path),
        )

        if dry_run:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script)
            print(f"  [DRY-RUN] {exp.name}: would submit {script_path}")
            job.status = "dry_run"
        else:
            job_id = submit_sbatch(script, script_path)
            job.job_id = job_id
            job.status = "submitted"
            print(f"  [SUBMITTED] {exp.name} -> job {job_id}")

        jobs.append(job)

    return jobs


def rebuild_jobs_from_state(
    stage: StageSpec,
    saved_jobs: Dict[str, "JobStateDict"],
) -> List[SlurmJob]:
    """Reconstruct SlurmJob objects from saved state for resumed polling.

    ``saved_jobs`` maps experiment_name -> dict with keys
    ``job_id``, ``run_dir``, ``status``.
    """
    jobs: List[SlurmJob] = []
    for exp in stage.experiments:
        saved = saved_jobs.get(exp.name)
        if saved is None:
            continue
        job = SlurmJob(
            experiment=exp,
            stage=stage.name,
            job_id=saved.get("job_id"),
            run_dir=saved.get("run_dir"),
            status=saved.get("status", "submitted"),
        )
        jobs.append(job)
    return jobs


# Type alias for the dicts we read back from state.json.
JobStateDict = Dict[str, Optional[str]]
