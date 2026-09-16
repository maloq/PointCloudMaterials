"""Timing summaries and bounded, reproducible run metadata."""
import importlib.metadata
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def summarize(seconds, units, unit_name):
    if not seconds or any(not math.isfinite(t) or t <= 0 for t in seconds):
        raise ValueError(f"Expected positive finite timings, got {seconds}")
    median = statistics.median(seconds)
    return dict(trials=len(seconds), seconds_median=median,
                seconds_min=min(seconds), seconds_max=max(seconds),
                seconds_samples=list(seconds), units_per_trial=units,
                **{f"{unit_name}_per_second": units / median})


def command_info(command):
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))
    return dict(returncode=result.returncode, stdout=result.stdout.strip(),
                stderr=result.stderr.strip())


def metadata():
    packages = {}
    for name in ("numpy", "torch", "lammps", "mace-torch", "e3nn", "cuequivariance-torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                executable=sys.executable, logical_cpus=os.cpu_count(),
                cpu_affinity=sorted(os.sched_getaffinity(0)), packages=packages,
                git_commit=command_info(["git", "rev-parse", "HEAD"]),
                git_status=command_info(["git", "status", "--short"]),
                cpu=command_info(["lscpu"]),
                environment={key: os.environ.get(key) for key in (
                    "CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                    "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")})
