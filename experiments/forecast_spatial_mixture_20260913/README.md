# 12 ps history, spatial surroundings and probabilistic trajectories — 2026-09-13

Question: do longer observations, nearby local environments, or multiple possible
future paths improve prediction of when the tracked local structure becomes crystal?
This is a matched exploratory follow-up to the [local crystallization assay](../forecast_crystallization_20260913/README.md).

[Completed results](SPATIAL_CONTEXT_RESULTS.md) cover the original study, shorter
histories and the larger spatial-context sweep. Twelve ps history with 32 cached
neighbors is the strongest observed configuration overall, with 56.86% transition
F1 at 9 ps; timing remains limited. The [11-fit interim report](INTERIM_RESULTS.md)
is retained as the earlier analysis.

## Protocol

| Condition | History | Spatial context | Future distribution |
| --- | --- | --- | --- |
| `history6_deterministic` | 6 ps | none | point trajectory, matched control |
| `history12_deterministic` | 12 ps | none | point trajectory |
| `history12_spatial` | 12 ps | 8 nearby cached centers | point trajectory |
| `history12_gaussian` | 12 ps | none | one Gaussian path |
| `history12_mixture4` | 12 ps | none | four Gaussian path components |
| `history12_spatial_mixture4` | 12 ps | 8 nearby cached centers | four Gaussian path components |

Two seeds per condition: 20260913 and 20260914. Every forecast covers all 12 sampled
future embeddings through 9 ps at 0.75 ps cadence. There are 125 independent cached
sources and 1,024 tracked centers per source: 74 train / 24 validation / 27 test.
All conditions use identical origins with at least 12 ps available history, spaced
6 ps apart: 7,350,272 training, 2,383,872 validation and 2,681,856 test windows.
The 6 ps model receives only the final nine observations at those same origins.

Each fit uses 12 epochs, batch 8,192 (898 batches/epoch), width 512, four GRU layers,
AdamW learning rate 0.0003, two warmup epochs, cosine decay, clipping 5 and weight
decay 0.0001. Central-history augmentation adds standardized noise 0.01 and carries
missing intermediate frames forward with probability 0.15. The clean anchor and
oldest frame cannot be dropped. Spatial observations are not corrupted. Training inputs and
spatial means remain resident on the H100; validation inputs/pools stay in host RAM; original embeddings are reused without
a second large disk cache. Exact-resume checkpoints remain on WORK.

The maintained `mean_residual_gru` is extended with a small spatial projection of
neighbor-minus-central embedding plus mean/outer distance at every observed time.
Eight nearest OTHER cached centers are found separately at each frame under periodic
boundaries. These are sparse sampled centers, not eight nearest physical atoms;
the typical eighth-center distance is about 13 angstrom. No future geometry enters.

Probabilistic heads predict K complete mean/scale paths and a history-conditioned
gate. A single component identity applies to the entire future; conditional noise
is diagonal within each path. Models train with proper joint mixture NLL. K=1
separates the effect of learning uncertainty from adding modes. Deterministic fits
use MSE. Validation selects the corresponding MSE/NLL checkpoint. Monitor MSE,
CRPS, marginal coverage, path energy score, entropy and component responsibilities;
collapsed components must be reported, not interpreted as four learned modes.
The construction follows [mixture density networks](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/)
with a component shared across the whole output path.

The normalization inherited by new methods was checked bit-for-bit against both
fresh controls. It is computed from unique training embeddings only, independent
of the history-window spacing. The original 32-epoch checkpoint supplies only
these normalization vectors; predictor weights start fresh.

## Physical evaluation and outputs

Reuse the frozen local PTM labels and ridge readout, with 64 fixed centers/source.
Measure the first local three-frame crystal episode, with nine-frame sensitivity.
All conditions get the same 765 every-frame forecast origins. Select event and
state thresholds on validation only. Report event precision/recall/F1, false alarms,
conditional timing MAE and all-event recall timed within 1.5 ps, plus one prediction
per eligible event exactly 3/6/9 ps beforehand. Mixtures get both the mean-path
readout and analytical marginal crystal probabilities. The latter are not calibrated
probabilities of sustained physical onset. Bootstrap whole independent sources,
paired across models after seed averaging. These test sources were inspected before;
results remain exploratory. See [full definitions](../../docs/metrics/forecast_spatial_mixture.md).

Output: [`output/embedding_forecast/context-space-mixture-20260913`](../../output/embedding_forecast/context-space-mixture-20260913),
a link to `/work/PERSO/vmorozov/analysis/embedding_forecast/context-space-mixture-20260913`.
Configurations and the 12-fit analysis plan are in [technical](technical/plan.json).
Forecast artifacts are in `fits/<method>-seed<seed>/technical`; physical score arrays,
logs and immutable source snapshots are under `technical/`. Final comparison tables
and frozen metric descriptions go in `tables/`, figures in `plots/`.

## Reproduction

Use the `pointnet` conda environment from the repository root. The embedding cache
and previous local-label assay are declared in the plan and must already exist.

```bash
python -m src.training_methods.embedding_forecast.spatial --config experiments/forecast_spatial_mixture_20260913/technical/spatial.json
python -m src.training_methods.embedding_forecast --config experiments/forecast_spatial_mixture_20260913/technical/history12_spatial_mixture4.json --stage train --seed 20260913 --runtime-config experiments/forecast_spatial_mixture_20260913/technical/runtime-spatial.json
python -m src.research.forecast_spatial_mixture.evaluate --plan experiments/forecast_spatial_mixture_20260913/technical/plan.json --run history12_spatial_mixture4 --seed 20260913
python -m src.research.forecast_spatial_mixture.compare --plan experiments/forecast_spatial_mixture_20260913/technical/plan.json
```

The preparation command creates a fresh output. To reproduce completed artifacts,
choose a new output in configuration; do not overwrite them. Repeat the training
and evaluation commands for each exact method/seed in `technical/plan.json`.
The final compare requires all 12 completed forecasts and local assays. Launched
training uses immutable `technical/baseline-source`, `technical/method-source` (nonspatial probabilistic),
and `technical/host-validation-source` (spatial) snapshots. Physical inference uses
`technical/analysis-source-host-validation`. Set `PCM_PROJECT_ROOT`
to the real project root when executing a snapshot. Do not modify any frozen source.

## Execution / initial validation

The first controls started immediately on the existing H100 allocations:
991149 / node53 (expires 20:56:40 UTC) and 990987 / nodesumo01 (expires September 14,
08:19:56 UTC). Detached serial queues launch the other ten fits as the controls
finish; no new Slurm allocation was requested. Node53 runs the four-component
conditions and the second 12 ps control; nodesumo01 runs spatial deterministic,
K=1 and the second 6 ps control. Queue commands, PIDs, statuses and logs are in
`technical/`. Local physical evaluation is detached and follows training on
nodesumo01's longer allocation.

Numerical and training checks: 47 passed for mixture likelihood/sampling,
causal spatial gathers, gradients, existing forecast regression tests and metric
exports. Another 14 passed for physical projection/identity, local event metrics,
paired source/seed bootstrap and metric documents. The host-validation change passed 17 focused regression checks, including an
identical-results execution comparison. A full-size spatial K=4 H100 preflight
completed three optimizer steps and probabilistic validation at batch 8192: compute
peak 14.29 GiB allocated / 16.84 GiB reserved, plus 57.88 GiB of resident training
embeddings and neighbor means. See `technical/spatial-gpu-preflight.json`. No transition improvement is claimed before the paired assay.

New maintained implementation: `src/training_methods/embedding_forecast/{context_mixture,spatial}.py`,
extensions to the existing training/evaluation/runtime, and
`src/research/forecast_spatial_mixture/{evaluate,compare}.py`.
This directory contains experiment records/configs. Generated allocation drivers,
launch receipts, smoke diagnostics and logs under the output are disposable run
artifacts; checkpoints, paired scores and immutable provenance are retained.

The first spatial attempt exceeded memory on its first backward pass because both
splits were GPU resident. It produced no completed epoch/checkpoint; the entire
attempt is retained under `technical/failed-attempts/`. Spatial queues were replaced
with `*-queue-host-validation.json`, keeping the model, batch, seed and objective.
The active nonspatial mixture and Gaussian fits were retained. The old node53
controller is stopped while its active child completes; its successor retires only
that controller after the child's exit and continues inside allocation 991149.
The detached analysis watcher was also restarted with the revised execution plan.

The first 6 ps and 12 ps control assays run immediately on the completed checkpoints;
remaining physical evaluations are queued after nodesumo01 training. Final collection
requires all 12 completed fits and paired physical assays, including their checkpoint
hashes. Individual results are available earlier under `technical/local/`.

Final combined regression run: **56 passed** in `pointnet`, including the existing
forecast, local crystallization and new spatial/mixture suites. Log:
`technical/final-tests.log`. Initial matched single-seed results are in [RESULTS.md](RESULTS.md).

## Short-history extension

[Four shorter histories](SHORT_HISTORY.md) add 14 fits for anchor-only, 1.5, 3 and
6 ps spatial-mixture forecasts and matched deterministic controls. They reuse six
existing reference fits and publish a separate paired report; the original 12-fit
study continues under its existing plan.

## Measured structures and embedding paths

[Trajectory visualizations](TRAJECTORY_VISUALS.md) now link actual 80-atom local
clouds to their measured embedding trajectories and four-component predicted
futures. The [gallery](../../output/embedding_forecast/structure-embedding-paths-umap-20260913/index.html)
contains static figures, a synchronized animation, and four offline interactive
point-cloud / UMAP time sliders. The original 12-fit study has completed; the shorter-history
extension remains a separate experiment.

## Broader spatial context

[The context-size sweep](SPATIAL_CONTEXT_SCALE.md) adds 32, 128 and 512 nearby
cached centers at 3 and 12 ps history, with two matched seeds and retained 8-center
controls. It uses the same full-trajectory objective, model, data and augmentation.
All twelve new fits and their paired analyses completed. See the
[results and interpretation](SPATIAL_CONTEXT_RESULTS.md) and
[six-figure gallery](../../output/embedding_forecast/spatial-context-analysis-20260914/index.html).
