# Standalone synthetic hardware benchmark

Run storage, LAMMPS CPU and repository GPU workloads without any datasets,
trained checkpoints, potential files or machine-local dataset settings.
Use the existing conda `pointnet` environment. Storage needs NumPy; CPU needs
a working LAMMPS executable. GPU needs CUDA PyTorch and the repo's model
dependencies; MACE additionally uses the installed mace-torch/e3nn stack.
No command installs software or downloads model weights.

```bash
# Quick correctness check of all three components (about 8 MiB of storage).
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py all \
  --smoke \
  --output output/hardware_benchmark/smoke

# Measure a selected filesystem; about 1 GiB of temporary space by default.
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py storage \
  --storage-dir /path/to/test/filesystem \
  --output output/hardware_benchmark/storage-machine-a

# Default CPU baseline: 32,000 atoms, one rank, one thread.
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py cpu \
  --output output/hardware_benchmark/cpu-machine-a

# Fixed-size strong scaling; launch only inside your allocated CPU resources.
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py cpu \
  --mpi-ranks 8 --launcher 'mpiexec -n 8' \
  --output output/hardware_benchmark/cpu-machine-a-r8

# All GPU models, or select only one with --workloads mace (or pointnet/forecast).
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py gpu \
  --device cuda:0 --output output/hardware_benchmark/gpu-machine-a
```

Storage defaults to the repository root, regardless of the current working
directory. Use `--storage-dir` with an **existing** directory to measure another mount.
The benchmark creates and removes only its uniquely named temporary directory.
Results may live on a different filesystem. Output directories must be new;
existing runs are never overwritten. Missing requested components fail with a
nonzero exit and diagnostic context; the command never substitutes CPU for GPU
or silently omits a task. Completed components remain in partial failure reports.

The LAMMPS command generates a periodic Lennard-Jones system internally, warms it
up and measures dynamics without trajectory I/O. It exercises MD computation,
neighbor lists and MPI communication, but is a baseline rather than the repo's
calibrated metal EAM/MEAM potentials. `--lammps /path/to/lmp` selects another build.
`--threads N` explicitly selects its OPENMP package when N > 1. Use a matching
MPI launcher and build; for Slurm a prefix such as `--launcher 'srun -n 8'` is
supported with `--mpi-ranks 8`. No jobs are submitted by this command.

GPU tests use the actual PointNet and MACE encoders plus the transformer embedding
forecaster. PointNet/MACE train with VICReg, which
matches two views while controlling feature variance and covariance. MACE uses
random weights, physical-coordinate graph patches and the current central-feature
API ([tracked-center embedding](research_glossary.md#tracked-center-embedding)).
The [projector](research_glossary.md#projector) maps encoder features into the
space where the VICReg loss is evaluated. Forecasting uses synthetic histories
and future embeddings. All include
backward and optimizer updates, not just matrix multiplication or inference.

Copy/edit [the workload recipe](../configs/benchmarks/hardware.json) and pass
`--config PATH` to change sizes, repeat counts or warmup. Defaults use three
trials; `--smoke` overrides sizes and counts for a tiny functional check. The
recipe's `mace_accelerated` selects cuEquivariance explicitly. Out-of-memory
errors do not trigger automatic batch shrinking. Keep exact sizes, backend,
precision and software versions fixed for comparisons. Scale a storage file
past available RAM when studying cache effects, with enough free disk space.

Storage reports fsynced writes, sequential reads and random float16 mmap batches
materialized as float32. Cache-eviction advice is best effort, so reported
throughput is never labeled guaranteed cold-disk speed. GPU inputs stay resident;
data loading, transfers and setup are outside timed training. Run benchmarks on
idle allocated resources to make the hardware comparison meaningful.

Each run exports `tables/hardware.csv`, frozen `tables/METRICS.md`, source hashes
in `technical/metric-contract.json`, full timing samples and system metadata in
`technical/results.json`, and its resolved recipe in `technical/config.json`.
CPU runs retain the generated input, LAMMPS log and process output. Failures retain
`technical/failure.txt` and a failed status. See the
[exact metric definitions](metrics/hardware_benchmark.md) before comparing rates.
