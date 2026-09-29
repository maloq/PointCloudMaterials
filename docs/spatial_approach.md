# Fixed-snapshot spatial approach workflow

[Scientific protocol](../experiments/spatial_approach_20260926/README.md).

[Results analysis](../experiments/spatial_approach_20260926/RESULTS.md) separates
the original 8 Å alarm from exploratory readouts of the saved distance
probabilities. Reproduce it with `python -m src.research.spatial_approach.review
--run RUN_ROOT --output REVIEW_ROOT`. It uses CPU only, fits nothing and keeps
the original artifacts unchanged. The review exports paired source uncertainty,
every model/score-radius combination, visibility diagnostics and a figure.

Launched 26 September: CPU preparation job **1009413** on nodecpu11;
detached GPU step **1009378.6** on node59. All five fits and evaluations completed
on 26 September. Each fit ran sixteen epochs; the encoder remained frozen.
[Collected results](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/RESULTS.md)
cover 126,545 matched fixed observations and 1,205 scan paths, including 495
held-out approach paths and 292 held-out away controls.
[Launch receipt](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/technical/launch.json)
and [worker log](/work/PERSO/vmorozov/analysis/spatial_approach/al64-20260926/technical/gpu-worker.log)
record the exact frozen code and allocation. Reuse the frozen configuration
for a resume, rather than the editable recipe.
Use `pointnet-torch214`; GPU work uses cuEquivariance. Set the existing runtime
environment `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for trusted local e3nn constants.
Preparation and feature checks create no W&B run. All five scientific predictor
fits use stable online IDs in `teshbek/PointCloudMaterials`.

```bash
python -m src.research.spatial_approach.queue prepare --config configs/analysis/spatial_approach_20260926.json
python -m src.research.spatial_approach.queue worker --config configs/analysis/spatial_approach_20260926.json
python -m src.research.spatial_approach.queue collect --config configs/analysis/spatial_approach_20260926.json
```

Preparation uses twelve CPU workers and completed PTM/ancestry receipts. The GPU
worker waits for completion, extracts only new path features, fits five
predictors sequentially and exports evaluation. Reused features hold a shared
cache lease; new path features use the same global six-entry LRU. Checkpoints,
predictions and metrics are durable and outside that disposable cache.

Results/logs live outside the repository at
`${storage:analysis}/spatial_approach/al64-20260926`. Frozen code and launch receipts
are in `technical/`. `preparation-state.json` and `state.json` record progress
or failure. Per-source preparation and epoch checkpoints support exact resumes.
An interrupted epoch replays from its preceding checkpoint. No storage-budget
rejection is introduced.
