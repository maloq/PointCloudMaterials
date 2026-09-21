# Running the paired relaxation pilot

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
