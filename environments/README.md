# Execution environments

Validated interpreter: Python 3.12.13. `requirements-core.txt` contains pinned shared
scientific packages. Choose `requirements-cpu.txt` or `requirements-gpu.txt`; do not
install both. The Linux CPU profile uses PyTorch 2.11.0+cpu; the NVIDIA profile retains
PyTorch 2.11.0+cu128 and the cluster's optional equivariance kernels.

`requirements-cpu.lock.txt` captures all installed distributions from the clean CPU
venv tested on 2026-09-13. Install it with the CPU wheel index:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r environments/requirements-cpu.lock.txt
python -m pip check
python scripts/project.py doctor
```

LAMMPS and MPI must be installed separately with the required potential packages.
Set their executable, launcher and runtime library environment explicitly in the
machine profile; verify with `project.py doctor --lammps`. See `docs/portability.md`.
