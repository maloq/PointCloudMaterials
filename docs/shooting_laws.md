# Shooting future-law workflow

Use conda `pointnet-torch214`. This workflow derives observations/physical labels
from existing registered trajectories; it launches no MD simulation. Existing
inputs remain on WORK/STORE, derived observation/target data on IDS, and durable
predictions, checkpoints, provenance and metric bundles on STORE. Frozen encoder
features use the shared six-entry leased cache. No automated tests or test-suite
dependencies are introduced.

```bash
python -m src.research.shooting_laws.queue bind --config configs/shooting_laws/al480_20260930.json
python -m src.research.shooting_laws.queue preflight --config configs/shooting_laws/al480_20260930.json
python -m src.research.shooting_laws.queue submit --config configs/shooting_laws/al480_20260930.json
python -m src.research.shooting_laws.diagnostics bind --config configs/shooting_laws/diagnostics_20260930.json
python -m src.research.shooting_laws.diagnostics submit --config configs/shooting_laws/diagnostics_20260930.json
```

`bind` checks all branch completion records, parent hashes, seed uniqueness,
actual timelines, frozen encoder checksums and source roles. `preflight` computes
one actual parent/branch including full-cell PTM, ancestry and physical descriptors,
then verifies actual model input/target shapes and finite likelihood gradients.
It is local numerical verification, not a scientific training run.

Submission freezes source and metric definitions. The main dependency graph is
40 CPU parent tasks grouped into eight workers, seal, two GPU frozen exports
(one concurrent), 27 CPU fits grouped into three workers (each fit has path and
event heads), then comparison. Packing sequential tasks keeps both dependency
graphs below the observed 30-job per-user Slurm submission limit.
Each completed branch has a checksum-bound receipt and resumes independently.
Partial preparation fails loudly; no subset silently enters a scientific fit.
Readouts use one deterministic data order per seed, with every fitting row visited
once per epoch. They are frozen diagnostic probes, so no W&B runs are created.

The diagnostic array groups 82 parents into four workers: 40 CSLD, 36 nested-Al
and six Ta. It waits
for the main sealed release to reuse exact CSLD center identities. Ta is a
structural-only diagnostic; it does not launch expensive full-cell temporal
event classification on the ten-million-atom cells. All 24 completed new Ta
shots are included. Earlier Ta shots remain preserved but are not silently
pooled into this four-shot diagnostic contract.

Results: `${storage:training_storage}/shooting_laws/al480-20260930/` and
`${storage:training_storage}/shooting_laws/diagnostics-20260930/`.
`technical/launch.json` records scheduler IDs, frozen commands, logs and errors.
Metric CSVs live under named `analyses/` bundles with `tables/METRICS.md` and
frozen implementation hashes. Failed workers retain explicit stage receipts.
Resume a failed stage using its frozen bundle and original index; do not submit
the entire dependency graph twice. A changed scientific producer requires a
new version rather than reusing completed targets.

## Submitted September 30, 2026

The primary chain is `1015983` (eight preparation workers), `1015984` (seal),
`1015985` (two serialized frozen exports), `1015986` (three workers covering
27 fits), and `1015987` (comparison). The diagnostic chain is `1015998`
(four preparation workers), `1015999` (seal), `1016000` (frozen exports), and
`1016001` (evaluation). Primary preparation was observed running on nodecpu05
and nodecpu11; later stages wait on successful dependencies.

Real-data checks passed for primary Al path/event production, both native
128-dimensional encoder exports, and CSLD, nested-Al and Ta target production.
All 40 CSLD parent states exactly match the primary Langevin parent positions,
boxes, atom IDs and generating parent hashes. The corresponding verification
receipt is in the diagnostic run's `technical/` directory.

The rejected initial array submission accepted no jobs. Its source and failure
receipt remain under the primary run's `technical/submission-rejected-job-limit/`.
Preflight diagnostic data from before the scheduling-only revision remain in
the IDS diagnostic cache under `preflight-before-job-packing/`; the submitted
revision derives its own checksum-bound targets.

## October 1 continuation

The original three grouped fit allocations exhausted their eight-hour limits after
nine completed fits. `src.research.shooting_laws.resume` copies the original frozen
source and adds a recorded execution adapter: validate/read the immutable parent
plan once per fit process instead of once per observation. Original fitting code,
configuration, numerical definitions and nine completed fits remain unchanged.
One remaining fit per array task, at most three concurrent, has a four-hour limit.
The continuation receipt records original and adapter hashes. Collector dependency
is rebound to that array after submission; downstream diagnostic dependency remains.
