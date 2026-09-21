# Running the paired relaxation pilot

The larger fixed-grid evaluation uses `python -m
src.research.relaxed_encoder.evaluation freeze|reuse|build|worker|report --config
configs/analysis/relaxed_encoder_large_test.json`. `reuse` copies verified centered
float32 observations from the expanded release. Remaining fixed-box quenches use
the existing GPU producer with this release's `technical/accelerator-config.json`.
`build` prepares matched assay rows and descriptors; `worker` extracts completed
checkpoints and runs frozen readouts, with no encoder training. Per-role grids
increase development/calibration/test coverage without repeating training data.

For preliminary frozen crystallization readouts of completed expanded encoders,
use `python -m src.research.relaxed_encoder.interim prepare|extract|probe|report
--config configs/analysis/relaxed_encoder_interim.json`. Extraction needs a GPU;
probes use CPU. This reuses the complete earlier two-origin assay, checks source
ancestry/splits and checkpoint/feature hashes, and writes a separate report. It
does not select whichever cells relaxed fastest, change the main queue, or replace
the larger 15-origin evaluation. The earlier cohort has eight positive test windows.

Use pointnet-torch214. All scientific settings are in
`configs/analysis/relaxed_encoder_pilot.json`; implementation is
`src/research/relaxed_encoder/`. No metrics-contract gate is added.

Module stages: `freeze`, `cpu --lane N --ranks 32`, `build`, `gpu --lane N`,
`benchmark --ranks 32`, `report`, each with `--config PATH`. Run GPU workers inside
an allocated GPU with its own CUDA_VISIBLE_DEVICES. CPU tasks use file locks,
immutable raw/potential hashes, strict convergence and archived failure evidence.
The builder releases matched caches only when all pilot cells exist; assay release
requires all evaluation cells. GPU workers wait for dependencies and record failures
explicitly. Model checkpoints retain the existing optimizer/RNG resume mechanism.
The queue does not create an allocation automatically or terminate a bash allocation.

Freeze source/config/metric definitions with the existing shared-pretraining
`snapshot` before detaching. Technical logs, per-cell failures, per-lane status,
queue receipts and the frozen plan live below the configured output. Training
caches are on IDS; fresh minimizations on SCRATCH; verified full-cell archives on
STORE. Conversion uses `src.data.conversion.relaxation`, the implementation of
`scripts/convert_trajectory.py`, preserving identity/time and verifying quantization.
Centered local clouds are saved float32 before global float16 conversion. An
interrupted publication resumes from a hash-checked pending-cloud receipt.

Existing expanded relaxed-TDA clouds may be reused for non-training center assays,
with exact source/frame/center/potential/hash checks. Full precision neighboring
query-center clouds were not saved there, so six-neighbor training cells are
re-minimized. Never silently extract new relaxed targets from the archived float16
cell and call those original converged coordinates.

For cells that stopped at the iteration limit, use
`python -m src.research.relaxed_encoder.recovery submit --config configs/analysis/relaxed_encoder_pilot.json --name retry-20260921 --allocation 1001497`.
This verifies and preserves archived failures, restarts only failed cells from
their full-precision final coordinates, and records larger execution limits in
`technical/restarts/NAME/plan.json`. FIRE state resets; potential, fixed box and
0.01 eV/Angstrom tolerance stay unchanged. Completed clouds are reused unchanged.
Successful retries move the original error receipts to `resolved-failures/`.
The detached CPU array gates cache construction, which gates GPU training and
frozen assays. One GPU worker waits inside the nominated existing allocation;
a second GPU job is requested with an after-success cache-build dependency.
Every submission freezes its own code, and the original release remains intact.

Expanded study: `python -m src.research.relaxed_encoder.expanded --config
configs/analysis/relaxed_encoder_expanded.json --allocation JOB_ID` snapshots the
release and submits twelve32-rank CPU workers, one cache builder and three GPU
jobs dependent on cache completion. A fourth GPU worker uses the given existing
allocation as soon as training caches are ready. No new MD trajectories are
integrated: existing frames are quenched, with verified paired/legacy cache reuse.
Iteration-limit failures automatically continue from archived full-precision
coordinates once, at unchanged force tolerance, with the configured larger budget.

## GPU production alongside CPU workers

`python -m src.research.relaxed_encoder.accelerated --config configs/analysis/relaxed_encoder_accelerated.json --backend h100 --lane h100-local --handoff`
uses the existing plan, cell locks, source/potential verification, convergence
checks, full-precision failed-quench retries and verified archive publication.
The `--handoff` worker switches to the encoder queue once training-ready.json
exists, after finishing its current cell. Stop only the older waiting training
step on that same GPU before launching this combined worker; preserve the bash
allocation. Dedicated producer GPUs omit --handoff and finish all missing cells.

Backends are admitted only after the matched benchmark's first cell converges and
its initial forces agree with the same-release CPU at max component error <=1e-8
eV/Angstrom. The pinned binary hash and GPU identity are recorded in every newly
quenched cell's relaxation settings. Existing reused cells retain their original
receipts. Pending tasks are randomized within training/evaluation priority, so
backend assignment does not simply follow whole temperature/source ranges.

GPU and CPU FIRE may converge to different local minima despite initial forces
agreeing to rounding precision. Expanded release therefore contains explicitly
recorded numerical backends with the same physical Hamiltonian, fixed box and
0.01 eV/Angstrom tolerance, not bitwise-equivalent CPU targets. The frozen shared
plan specifies physics/data; technical/accelerated contains this additive execution
release and its code snapshot. Retain the original CPU production and old code.

### Recovering a CPU timeout without a final dump

A frozen recovery recipe may explicitly use `restart_dump: null` and
`restart_sha256: null` to requench the original verified MD frame in a fresh retry
directory. The old timeout archive is retained and verified. The `cuda` recovery
stage accepts `--accelerator-config configs/analysis/relaxed_encoder_accelerated.json
--backend a100` and requires the same benchmark admission as production workers.
A successful recovery archives the old failure receipt before releasing cache
construction. `train-submit` watches training-ready.json and submits the frozen
train-*.sbatch files without waiting for the separate assay release. It records
submitted IDs and refuses a second submission. This watcher needs only a CPU;
training GPUs are requested after caches exist.

## Non-blocking timeout policy

Per-cell subprocess.TimeoutExpired is a terminal skipped sample. Keep the failure
record and verified failure archive; record its identity/hash under technical/skipped.
No automatic timeout retry is required. CPU and GPU workers continue, and builder/
trainer readiness treats explicit skips as resolved work. Other failure types still
fail with context. The availability adapter also recognizes actual TimeoutExpired
repr records from older frozen CPU workers, so they need not be interrupted.

If either cell of a present/future training pair times out, exclude that entire
pair from every arm and train-only normalization. An assay timeout removes only
that source/frame's rows, consistently for all encoders and baselines; graph indices
are rebuilt for remaining frames. Export excluded training-anchor and assay-window
counts. Preserve an already completed recovery when it predates the skip decision.
This is conditional-on-success sampling; retained cohorts and event counts can change.

## Stop encoder fitting and evaluate completed checkpoints

Send SIGUSR1 to the active fit subprocess to save `last.pt` at an update boundary.
Stop other training queue workers before signalling it so they cannot resume the
checkpoint. Preserve both best and last checkpoints; an interrupted fit remains
`checkpointed`, never `complete`. Record its fit queue receipt as `blocked`.

`technical/evaluation-exclusions.json` records the frozen plan identity and an
`excluded` mapping from run name to an explicit reason. It changes which encoders
are evaluated without changing the data release, source split or outcome labels.
Both reports display the exclusion and adjust the number of expected readouts.
Use `python -m src.research.relaxed_encoder.queue evaluate --config CONFIG` for
evaluation only: this stage never schedules encoder fits. The larger assay's
`evaluation worker` also respects these exclusions. Submit GPU extraction jobs
after successful CPU assay construction so they do not hold GPUs while waiting.
