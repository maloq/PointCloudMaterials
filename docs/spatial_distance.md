# Continuous crystal-distance workflow

Use conda `pointnet-torch214` and the
[recipe](../configs/analysis/spatial_distance_20260926.json). Scientific protocol
and fixed-confidence results are indexed in
[the experiment](../experiments/spatial_distance_20260926/README.md).

`python -m src.research.spatial_distance.queue prepare --config CONFIG` samples
uniform spatial centers on CPU. It uses the same fixed train/selection ancestors
and observation frames, and the retained full-cell crystal audit. No simulation.
`python -m src.research.spatial_distance.queue worker --config CONFIG` rebuilds
frozen observed MACE features from retained geometry, extracts the extra centers
and unchanged scan paths, then trains five distance heads in sequence. Checkpoint
and completed-shard receipts permit a restart of the same frozen code/config.

Freeze `src/`, metric documents/contracts and the recipe under the run's
`technical/code` before launching. Export `PCM_PROJECT_ROOT` to the checkout for
machine-local path resolution. Other runtime variables match the spatial-approach
worker: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, single-threaded BLAS/OpenMP and
`PYTORCH_ALLOC_CONF=expandable_segments:True`. Scientific training uses online
W&B in `teshbek/PointCloudMaterials`, group `spatial-distance-20260926`.

Generated geometry resides under `${storage:cache}/spatial-distance-geometry`.
Fixed, augmented and scan feature caches live outside the checkout under the
existing six-entry lease/eviction policy. The old features had been evicted;
retained graph geometry and checkpoints permit their reconstruction. Inputs,
checkpoints, predictions and completed historical scores are not deleted.

Runtime state, preparation/GPU logs, source snapshot and Slurm receipts live in
`${storage:analysis}/spatial_distance/al64-uniform-20260926/technical`.
Each model exports distance and confidence analyses beneath its own `analyses/`.
No new automated tests are added. Before launch the five heads were exercised
on actual scan geometry with zero/positive/censored labels, finite-gradient and
CDF checks, and an independent numerical check of the capped expectation.

Detached on 26 September: CPU step **1009378.11**, GPU step **1009378.12**,
node59. [Receipt](/work/PERSO/vmorozov/analysis/spatial_distance/al64-uniform-20260926/technical/launch.json).
The first stage reconstructs evicted features; training follows automatically.
