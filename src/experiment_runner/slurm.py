"""SLURM execution backend: sbatch generation, submission, and polling."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .plan import (
    Experiment,
    ExperimentPlan,
    SlurmConfig,
    StageSpec,
    resolve_experiment_config_name,
    resolve_experiment_train_script,
)

POLL_INITIAL_INTERVAL_SEC = 15
POLL_MAX_INTERVAL_SEC = 120
POLL_BACKOFF_FACTOR = 1.5
JOB_STATUS_FILENAME = "slurm_job_status.txt"
MISSING_STATUS_POLLS_BEFORE_UNKNOWN = 2


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

    train_script = resolve_experiment_train_script(experiment, stage, plan)
    config_name = resolve_experiment_config_name(experiment, stage, plan)

    job_name = f"{plan.name}_{stage.name}_{experiment.name}"
    # SLURM doesn't like very long job names; truncate.
    if len(job_name) > 80:
        job_name = job_name[:77] + "..."
    status_file = run_dir / JOB_STATUS_FILENAME

    all_overrides: List[str] = []
    all_overrides.extend(stage.base_overrides)
    all_overrides.extend(experiment.overrides)
    all_overrides.append(f'hydra.run.dir="{run_dir}"')
    all_overrides.append(
        'experiment_name="'
        f'{_safe_experiment_name(plan.name, stage.name, experiment.name)}'
        '"'
    )
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

STATUS_FILE="{status_file}"
JOB_FINAL_STATE=""

write_job_status() {{
    local state="$1"
    local exit_code="$2"
    local tmp_file="${{STATUS_FILE}}.tmp"
    {{
        printf 'state=%s\n' "$state"
        printf 'exit_code=%s\n' "$exit_code"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    }} > "$tmp_file"
    mv "$tmp_file" "$STATUS_FILE"
}}

on_job_signal() {{
    JOB_FINAL_STATE="CANCELLED"
    exit 143
}}

on_job_exit() {{
    local exit_code="$?"
    local state="$JOB_FINAL_STATE"
    if [ -z "$state" ]; then
        if [ "$exit_code" -eq 0 ]; then
            state="COMPLETED"
        else
            state="FAILED"
        fi
    fi
    write_job_status "$state" "$exit_code"
}}

rm -f "$STATUS_FILE" "${{STATUS_FILE}}.tmp"
trap on_job_signal TERM INT
trap on_job_exit EXIT

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
        ["sacct", "-j", id_str, "--format=JobIDRaw,State,ExitCode", "--noheader", "--parsable2"],
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
        if len(parts) < 3:
            continue
        raw_id = parts[0].strip()
        state = parts[1].strip().split()[0]
        exit_code = parts[2].strip()
        if not raw_id:
            continue
        root_id = raw_id.split(".", 1)[0]
        if root_id not in {str(j) for j in job_ids}:
            continue
        normalized = _normalize_sacct_state(state, exit_code)
        statuses[root_id] = _merge_job_states(statuses.get(root_id), normalized)

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
    return statuses


def is_terminal_state(state: str) -> bool:
    return state.upper() in {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
        "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "UNKNOWN",
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
    missing_status_polls = {jid: 0 for jid in active_ids}
    while True:
        remaining = {jid: j for jid, j in active_ids.items() if not is_terminal_state(j.status)}
        if not remaining:
            break

        time.sleep(interval)
        interval = min(interval * POLL_BACKOFF_FACTOR, POLL_MAX_INTERVAL_SEC)

        statuses = query_job_statuses(list(remaining.keys()))
        for jid, job in remaining.items():
            state = statuses.get(jid)
            if state is None:
                state = _read_terminal_status_from_run_dir(job.run_dir)
                if state is None:
                    missing_status_polls[jid] += 1
                    if missing_status_polls[jid] < MISSING_STATUS_POLLS_BEFORE_UNKNOWN:
                        continue
                    state = "UNKNOWN"
                else:
                    missing_status_polls[jid] = 0
            else:
                missing_status_polls[jid] = 0

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


def read_job_status_file(run_dir: Path) -> Optional[Dict[str, str]]:
    path = run_dir / JOB_STATUS_FILENAME
    if not path.exists():
        return None

    parsed: Dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line == "":
            continue
        if "=" not in line:
            raise RuntimeError(
                f"Malformed SLURM job status file {path}: line {line_no} does not contain '='."
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "" or value == "":
            raise RuntimeError(
                f"Malformed SLURM job status file {path}: line {line_no} has empty key or value."
            )
        parsed[key] = value

    if "state" not in parsed:
        raise RuntimeError(
            f"Malformed SLURM job status file {path}: missing required 'state' entry."
        )
    return parsed


def _read_terminal_status_from_run_dir(run_dir: Optional[str]) -> Optional[str]:
    if run_dir is None:
        return None
    payload = read_job_status_file(Path(run_dir))
    if payload is None:
        return None
    return payload["state"].upper()


def _normalize_sacct_state(state: str, exit_code: str) -> str:
    normalized_state = state.strip().upper()
    normalized_exit = exit_code.strip()
    if normalized_state == "COMPLETED" and normalized_exit not in {"", "0:0"}:
        return "FAILED"
    return normalized_state


def _merge_job_states(existing: Optional[str], candidate: str) -> str:
    if existing is None:
        return candidate

    precedence = {
        "FAILED": 100,
        "CANCELLED": 95,
        "TIMEOUT": 90,
        "OUT_OF_MEMORY": 85,
        "NODE_FAIL": 80,
        "PREEMPTED": 75,
        "UNKNOWN": 70,
        "RUNNING": 40,
        "CONFIGURING": 35,
        "COMPLETING": 30,
        "PENDING": 20,
        "COMPLETED": 10,
    }
    return candidate if precedence.get(candidate, 50) >= precedence.get(existing, 50) else existing


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
