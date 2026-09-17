# Standalone synthetic hardware benchmark

Run storage, LAMMPS CPU and repository GPU workloads without any datasets,
trained checkpoints, potential files or machine-local dataset settings.
Use the existing conda `pointnet` environment. Storage needs NumPy; CPU needs
a working LAMMPS executable. GPU needs CUDA PyTorch and the repo's model
dependencies; MACE additionally uses the installed mace-torch/e3nn stack.
No command installs software or downloads model weights.

```bash
# Run once: print the final table and save RESULTS.md plus tables/summary.csv.
# With no suite argument, all components run; storage uses the repository root.
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py

# Include the 1/8/24-rank CPU cases in the same invocation (requires 24 allowed CPUs).
conda run --no-capture-output -n pointnet python scripts/benchmark_hardware.py \
  --cpu-ranks 1 8 24

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
For a sweep, use `--cpu-ranks 1 8 24`: one rank launches directly, and larger
cases use `mpiexec -n N` by default. An explicit sweep launcher needs the placeholder,
for example `--launcher 'srun -n {ranks}'`. The command checks rank×thread counts
against the process's allowed logical CPUs; it never silently shrinks a sweep or
oversubscribes that allocation. CPU cases run sequentially at the same atom count.
Each case retains separate input/log/build files under `technical/rN_tT/`.

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

At the end, the command prints a Markdown table and saves the identical report
as `RESULTS.md`. It includes CPU/GPU identity, software versions, storage
filesystem, exact file size, workload sizes, rank/thread and batch counts,
median throughput, observed min–max throughput, trial count, and GPU ms/update.
Smoke runs and incomplete runs are labeled prominently. Copy this report to share
the results; `tables/summary.csv` retains the same rows at full numeric precision.
The report does not rank machines or calculate cross-run comparisons.

Each run also exports `tables/hardware.csv`, frozen `tables/METRICS.md`, source hashes
in `technical/metric-contract.json`, full timing samples and system metadata in
`technical/results.json`, and its resolved recipe in `technical/config.json`.
CPU runs retain the generated input, LAMMPS log and process output. Failures retain
`technical/failure.txt` and a failed status. See the
[exact metric definitions](metrics/hardware_benchmark.md) before comparing rates.

## Standard format for sharing and comparing results

The final table is generated automatically; no manual unit conversions are needed.
It reports **medians over three measured trials by default, excluding warmup**, with
the workload and resources stated in every row. If the trial count differs, say
so. Use these units consistently across machines:

| Workload | Required row details | Primary result | Additional result |
| --- | --- | --- | --- |
| CPU — LAMMPS | Atom count; MPI ranks × OpenMP threads per rank | MD steps/s | Atom-steps/s |
| GPU — PointNet | GPU model; batch size | Examples/s | ms/update |
| GPU — MACE | GPU model; batch size; channels; reference or accelerated backend | Examples/s | ms/update |
| GPU — forecast transformer | GPU model; batch size; history/future lengths | Examples/s | ms/update |
| Storage — sequential read, eviction advised | Filesystem/mount; file size | MiB/s | Keep separate from warm reads |
| Storage — sequential read, warm | Same filesystem and size | MiB/s | Buffered/cache throughput |
| Storage — generation, hashing, write and fsync | Same filesystem and size | MiB/s | Includes CPU data preparation |
| Storage — random mmap, eviction advised | File size; cloud shape; batch size | Clouds/s | Logical MiB/s |
| Storage — random mmap, warm | Same file and batch sizes | Clouds/s | Logical MiB/s |

Use the matching fields in `technical/results.json` and `tables/hardware.csv`:
CPU `steps_per_second` / `atom_steps_per_second`; GPU `examples_per_second` /
`step_ms`; storage `sequential_eviction_advised`, `sequential_warm`,
`write_generate_hash_fsync`, `mmap_eviction_advised` and `mmap_warm`. The
`write_syscalls_fsync` diagnostic may be included as an additional, separately
labeled row; it must not replace the generation/hash/write/fsync result.

For an externally supplied GPU time, convert with
`examples/s = 1000 * batch_size / ms_per_update`. Count original examples, not
the two PointNet/MACE views. State both batch size and latency so this conversion
can be checked. Here an update is one forward/backward/optimizer step without
gradient accumulation. Derive rates from the same median trial duration as the
latency; do not average individual trial rates. Keep full precision in machine
artifacts; round only for display, and mark conversions from rounded inputs as
approximate. Use `forecast transformer` rather than the ambiguous `Transformer`.

Alongside the table, provide the run date, CPU and GPU models, storage mount/type,
software versions, resolved settings, and whether other jobs shared the machine.
Attach or link `technical/results.json`, `technical/config.json`,
`technical/metric-contract.json`, `tables/summary.csv` and `tables/METRICS.md`.
A CPU sweep captures every case in the same report and uses keys such as
`cpu.r8_t1` in the detailed metric export (raw-results schema version 2).
Historical schema-1 files retain their original layout and definitions. Mark missing measurements
as **not measured** and missing metadata as **not supplied**; never infer hardware,
atom counts, precision or model settings from a timing alone. Supply the per-trial
samples or min/max alongside medians when available. Three trials without their
spread do not establish that a small difference is meaningful.

Compare matching atom counts, CPU rank/thread counts, model and batch sizes,
precision/backends, software/source versions, and storage cache modes. Use a
table with `Workload | Unit | Reference | Reported run | Throughput change`.
Compute `throughput change (%) = 100 * (reported_rate / reference_rate - 1)`;
positive means faster. If starting from latency, the speedup is
`reference_ms / reported_ms`, not the inverse. Do not combine the distinct
workloads into one hardware score.

Keep CPU scaling separate from comparisons at equal rank counts:
`speedup(p) = steps_per_second(p) / steps_per_second(1)` and
`parallel efficiency(p) = speedup(p) / p`. These require the same atom count and
threads per rank. If the reference machine has only a one-rank result, its 8/24-rank
cells remain **not measured**; a multi-rank/one-rank ratio is not an equal-resource
hardware comparison. When only a pasted summary is available, label the comparison
as conditional on matching settings and retain which metadata was not supplied.
