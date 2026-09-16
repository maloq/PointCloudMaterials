# Synthetic hardware benchmark metrics (v1)

These are operational hardware measurements, not scientific accuracy scores.
All inputs are generated with the saved seed; no checkpoint or dataset is loaded.
The resolved configuration, package versions, CPU affinity, device details, source
hashes and raw timing samples accompany every export. Never compare `--smoke`
results as performance measurements. Match workload sizes, backend, software,
precision and process/thread placement across machines; use idle allocated resources.

## Common summaries

`seconds_samples` contains the individual measured trial durations in JSON.
`seconds_median`, `seconds_min`, `seconds_max` summarize those durations in seconds.
`trials` is the sample count; `units_per_trial` is the numerator used for throughput.
`*_per_second = units_per_trial / seconds_median`. This is a rate at the median
duration, not a mean of rates. Trials are sequential repeats, not independent
machines or uncertainty intervals. Arrays remain in JSON; scalar summaries go to CSV.
MiB = 2^20 bytes; GiB = 2^30 bytes. Blank CSV values never mean zero.

## Storage

A unique temporary file is created inside the explicitly selected filesystem.
Each record is `(points, 3)` little-endian float16, default `(80, 3)`, matching the
coordinate dtype and local-cloud shape used by the repository's training caches.
This is a raw contiguous synthetic array, not a complete trajectory/manifest.
The requested size is rounded down to a whole record. Uniform random coordinates
in [-9.2, 9.2) are generated chunk by chunk with bounded RAM, without sparse files
or repeating a single buffer. Every written file is fsynced and verified with a
full SHA-256 read outside the write interval; mmap batch checksums must agree.
Only that invocation's temporary directory is removed, including on failure.

- `write_generate_hash_fsync`: wall time for file creation, coordinate generation,
  float16 conversion, hashing, buffered writes, fsync and close. This is application
  throughput including CPU preparation, not pure device write bandwidth.
- `write_syscalls_fsync`: sum of write-call durations and final fsync duration.
  Generation/hash gaps are excluded; the OS can write back during those gaps, so
  this diagnostic rate can overestimate continuous device write bandwidth.
- `sequential_eviction_advised`: sequential buffered reads into one reusable buffer,
  following file-local `POSIX_FADV_DONTNEED`. Timed bytes are the complete file.
  No checksum or float conversion is inside this read interval.
- `sequential_warm`: the same scan immediately after an untimed full scan. If the
  file exceeds available page-cache capacity, some reads can still reach storage.
- `mmap_eviction_advised`: materialized random cloud batches, float16-to-float32
  conversion and float64 checksum reduction, following the same eviction advice.
- `mmap_warm`: identical batches after untimed replay of those exact indices in
  the same mapping. Batch indices are sampled with replacement and generated
  before timing. Repeats and both cache modes use the same indices.

The eviction request is advisory; filesystem, client/server and device caches can
remain warm. These are explicitly buffered application tests, not direct I/O or
guaranteed cold-media measurements. No global cache dropping occurs.
Random gathers report `clouds_per_second` and `logical_MiB_per_second`, where
logical bytes = batches × batch_size × points × 3 × 2. They do not report physical
bytes fetched: page faults, readahead, repeated samples and caches change that count.

## CPU: LAMMPS

Periodic FCC initialization with `4*cells^3` atoms, reduced density 0.8442,
temperature 1, `lj/cut` cutoff 2.5, neighbor skin 0.3, NVE integration at dt=0.005.
There is no trajectory output. This tests force/neighbor/integration/communication
work shared by the repository's simulations. It is not a calibrated Al/Ti/Ta
EAM/MEAM simulation and does not predict their absolute runtime.

A warmup run is discarded, followed by `repeats` consecutive segments of `steps`
timesteps in one process invocation. LAMMPS's own `Loop time` excludes setup.
`steps_per_second = steps / median_loop_seconds`;
`atom_steps_per_second = atoms * steps_per_second`.
`process_wall_seconds` separately includes launch, initialization, warmup and all
segments. Atom, step and MPI×OpenMP counts must match the requested protocol.
Nonfinite thermo output, lost atoms, missing reports and subprocess errors fail.
Single-thread runs use the standard CPU pair style. Multiple threads explicitly
request the OPENMP suffix/package; changing this backend is a protocol change.

The producer semantics are described in the
[LAMMPS run-output documentation](https://docs.lammps.org/Run_output.html) and
[OPENMP documentation](https://docs.lammps.org/Speed_omp.html).

## GPU

The actual repository models are initialized with random weights:

- `pointnet`: `PointNetEncoder` (`PnE_L`, latent 128, feature transform enabled),
  fused two-view inputs `(2*B, 3, points)`, production VICReg projector/loss.
- `mace`: `ReferenceMACEEncoder`, two MACE interactions, configurable channels
  (default 64), ell=2, correlation=3, 4 Å cutoff. Synthetic centered grid patches
  have spacing 2.4 Å and small random displacements. Directed radius edges are
  prepared before timing. Material IDs use the actual 0=Al, 1=Mg, 2=Ta producer
  convention. Default reference e3nn backend; cuEquivariance is opt-in through
  `mace_accelerated`. Trainable central scalar features from both interactions
  feed the production VICReg projector/loss. No force/energy loss is used.
- `forecast`: `EmbeddingForecaster` with two transformer layers, four heads,
  default width 256, 16 observed steps, 16 future steps and 256 input channels.
  Production `forecast_loss` has MSE weight 1; other weights are zero. Synthetic
  random-walk futures retain the expected shape; loss values have no scientific
  interpretation.

VICReg uses the repository's 128-channel projector and coefficients 25/25/1:
paired mean squared difference, mean of the two population-variance penalties
(epsilon 1e-4, target standard deviation 1), and mean of the two off-diagonal
sample-covariance penalties. We call its implementation rather than redefine it.
See the [research glossary](../research_glossary.md) for research-protocol context.

All tasks use float32, eager execution, AdamW (lr=1e-4, weight_decay=1e-2,
foreach=false, fused=false), no gradient clipping, and resident device inputs.
`matmul_precision=highest` is the default; `high` explicitly permits TF32 where
PyTorch supports it. cuDNN TF32 and autotuning are disabled. Host thread count is
recorded. No model download, training cache, pretrained checkpoint or compilation
is involved. These timings do not cover data loading, host-to-device transfer,
augmentation, graph construction, logging, validation or checkpoint saving.

Each timed step includes zeroing gradients, forward/projector/loss, backward,
AdamW update and detaching the loss for later checks. Setup and warmup are reported
separately. CUDA synchronization brackets each multi-step trial; `step_ms` is
1000 × median wall duration / steps. CUDA events additionally retain trial
durations in `cuda_event_seconds_samples`; these include device idle gaps between
launches, not just kernel execution time. Finite loss, gradients and parameters,
and at least one nonzero gradient, are verified outside timing after each trial.

`examples_per_second = B*steps/median_wall_seconds` counts original examples,
not paired views. `cloud_views_per_second` is twice that for PointNet and MACE.
`peak_allocated_MiB` and `peak_reserved_MiB` are process-local PyTorch allocator
peaks since warmup ended, including resident inputs, model, optimizer and current
step storage. They are not total GPU usage. Raw `nvidia-smi` snapshots record
other visible GPU activity; the command never stops other processes.


Table export: 2026-09-16T19:33:52.758280+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
