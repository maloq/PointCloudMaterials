# Execution environments

Validated interpreter: Python 3.12.13. `requirements-core.txt` contains pinned shared
scientific packages. Choose `requirements-cpu.txt` or `requirements-gpu.txt`; do not
install both. Root `requirements.txt` includes the GPU profile, so direct dependency
versions have one owner. The historical Linux CPU profile uses PyTorch 2.11.0+cpu; the current
NVIDIA profile selects PyTorch 2.14.0+cu130, TorchVision 0.29.0+cu130 and the CUDA 13
cuEquivariance kernels. The GPU environment is `pointnet-torch214`; existing runs
and exact resumes retain their original `pointnet` interpreter. See
[upgrade and validation](../docs/pytorch214_upgrade.md).

## Current GPU profile

`requirements-gpu.lock.txt` pins the 141 installed distributions required by the
GPU profile, including transitive dependencies, from `pointnet-torch214` on
2026-10-23 (Linux x86_64, Python 3.12.13). It excludes unrelated host packages and
optional experiment profiles. All selected requirements match installed versions
and dependency metadata; scientific imports passed. The CUDA device is visible,
but its context was busy/unavailable during the tensor check, so GPU execution
still needs checking inside an available allocation.
No local paths or editable packages are included. A fresh installation has not yet
been exercised, so this is a captured environment, not a verified portable build.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-gpu.lock.txt
python -m pip check
python scripts/project.py doctor --profile gpu
```

Use the direct profile for intentional upgrades; update the lock only after
validation in a clean environment. Preserve each frozen run's original environment.
The lock does not capture the NVIDIA driver, compiler, or LAMMPS/MPI installation.

The source environment's full `pip check` still reports five existing conflicts:
Zarr and Rasterio require NumPy 2; Xarray and Wheel require newer Packaging; GATr
declares a historical NumPy upper bound. None is in the GPU profile's dependency
closure. The optional GATr/LeJEPA profiles retain their explicit `--no-deps`
installation instructions and are validated separately; they are not silently
included in the base lock. See [upgrade validation](../docs/pytorch214_upgrade.md)
and the local [capture receipts](../output/maintenance/research-practices-20261023/technical/).

## Historical CPU profile

`requirements-cpu.lock.txt` captures all installed distributions from the clean CPU
venv tested on 2026-09-13. Install it with the CPU wheel index:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r environments/requirements-cpu.lock.txt
python -m pip check
python scripts/project.py doctor --profile cpu
```

## Environment checks

`requirements-check.txt` is the small CPU environment used by GitHub Actions and
`project.py check`: NumPy, SciPy, PyTorch, OmegaConf, PyYAML and Ruff. Its shared
versions come from the core constraints; it does not install the full research
stack. A clean Python 3.12 installation and the complete integration workflow
passed on 2026-10-23. See [development](../docs/development.md).

`project.py doctor` remains a fast machine-path and core-package report.
`--profile cpu` or `--profile gpu` additionally checks the maintained direct pins,
imports NumPy, PyTorch/Lightning, MACE/e3nn and ASE, and runs a tiny tensor operation.
GPU checks also import TorchVision/cuEquivariance/OpenEquivariance and require a
visible CUDA device; run them inside an allocation. Failures produce a JSON report
and exit status 1. This is an explicit environment diagnostic, not a launch gate.

LAMMPS and MPI must be installed separately with the required potential packages.
Set their executable, launcher and runtime library environment explicitly in the
machine profile; verify with `project.py doctor --lammps`. See `docs/portability.md`.
