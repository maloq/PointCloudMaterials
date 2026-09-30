"""Process environment for the repository's conda LAMMPS executable."""

import os
from pathlib import Path
import sys


def lammps_environment(*, hide_gpus=False) -> dict[str, str]:
    environment = os.environ.copy()
    if hide_gpus:
        for key in (
            "CUDA_VISIBLE_DEVICES", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
            "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK", "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
        ):
            environment.pop(key, None)
    environment.update(
        MPIR_CVAR_CH4_NETMOD="ofi", FI_PROVIDER="tcp",
        OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE",
    )
    environment["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + (
        f":{environment['LD_LIBRARY_PATH']}" if environment.get("LD_LIBRARY_PATH") else ""
    )
    return environment
