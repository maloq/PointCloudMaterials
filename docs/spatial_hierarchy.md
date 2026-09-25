# Spatial hierarchy queue

[Scientific protocol](../experiments/spatial_hierarchy_20260924/README.md).
Use conda `pointnet-torch214`. This extends the tested robust-onset trainer using
explicit model, input and diagnostic interfaces. The preceding queue uses its
frozen source and is unaffected by workspace changes.

The new, unsubmitted `configs/spatial_hierarchy/screen_ap3_20260924.json` uses
3 ps AP for ranking and checkpoint selection, with a separate output. AP6 and
AP12 remain secondary exports. The original12 ps-selected hierarchy models
perform poorly at 3 ps; the [horizon review](../output/encoder_research/onset-horizons-20260924/RESULTS.md)
and [revised priorities](../experiments/onset_ap3_20260924/README.md) should guide
which variants to run after expanding the event cohort.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.spatial_hierarchy.queue preflight --config configs/spatial_hierarchy/screen_20260924.json
python -m src.research.spatial_hierarchy.queue submit --config configs/spatial_hierarchy/screen_20260924.json
```

Preflight prepares and verifies 16 Å current-frame patches and the wider dense
evaluation cache from existing trajectories, then runs unit tests and a production
GPU update for each arm. No MD or minimization is performed. Inputs are on IDS;
fits/results are on WORK. Paths are configured through `machine.local.yaml`.

Four array tasks, one seed, at most two GPUs concurrently, two-hour limit each
(eight GPU-hours maximum; up to four hours task time with two slots available,
plus scheduling). Tasks checkpoint on timeout and declare incomplete status.
The dependent CPU collector runs after all tasks and lists incomplete evaluations.
Launch receipts, source snapshots and status are under the run's `technical/`.

Frozen-source resume uses `worker --config CONFIG --arm NAME` inside an allocation.
Do not rerun an old study from changed workspace code; retain the exact frozen
source or use a new output. Reporting uses the existing physical/onset/stability
metrics plus an outer-context intervention diagnostic.
