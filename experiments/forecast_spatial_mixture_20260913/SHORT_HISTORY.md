# Shorter observed history — added 2026-09-13

Question: how much past observation does the spatial full-trajectory mixture need
to predict local crystallization, and does that differ from a deterministic forecast?

Compare **0, 1.5, 3, 6 and 12 ps** of history for two model families:

- Deterministic mean-residual GRU, 8,014,080 parameters.
- The same GRU with eight nearby cached centers and four full-path Gaussian
  mixture components, 9,198,852 parameters.

Each length has two seeds, 20260913 and 20260914. Existing 6/12 ps deterministic
fits and 12 ps spatial-mixture fits supply six retained references. There are
**14 new fits** and 20 fitted models in this history comparison. The original
12-fit spatial/uncertainty ablation retains its own report and execution plan.

The anchor-only condition observes one frame: the central embedding plus the
current spatial pool when spatial context is enabled. Intermediate-frame
augmentation has no effect when there is no earlier frame; the clean anchor
is retained under the existing augmentation protocol.

All other settings match the 12 ps study: predict all 12 future embeddings through
9 ps; all 125 cached sources and 1,024 centers/source; 74/24/27 source split;
common forecast origins requiring 12 ps available history, 6 ps spacing;
12 epochs, batch 8192, width 512, four GRU layers, AdamW schedule and augmentation.
New fits reuse the verified training normalization, with fresh predictor weights.
Parameter counts are identical within each family at all history lengths.
Validation inputs stay in host memory so spatial training fits the H100.

Read out local PTM crystal transitions using the same frozen probe and physical
labels, thresholds selected on validation, and 765 matched every-frame origins.
Report embedding error, probabilistic scores, transition precision/recall/F1,
missed events, timing error and all-event recall timed within 1.5 ps. Pair each
short history with 12 ps within its family, and compare the two families at each
history. Retain the same two-seed averaging and source bootstrap. At 0 ps, the
three-frame physical risk-set criterion is still used to define the common test
population; these labels are not model inputs.

Configurations: [technical/short-history](technical/short-history/plan.json).
Output: [context-space-mixture-short-history-20260913](../../output/embedding_forecast/context-space-mixture-short-history-20260913),
with new fits under `fits/`, exact arrays/logs/source snapshots under `technical/`,
and comparison tables/plots under `tables/` and `plots/`. Retained reference fits
and physical assays are read directly from the original study; they are verified
by checkpoint/normalization/window identity and are not copied or rewritten.
Paths in new recipes use the machine profile and dataset catalog.

## Reproduction and execution

Use conda environment `pointnet` from the repository root:

```bash
python -m src.training_methods.embedding_forecast --config experiments/forecast_spatial_mixture_20260913/technical/short-history/history3_spatial_mixture4.json --stage train --seed 20260913 --runtime-config experiments/forecast_spatial_mixture_20260913/technical/short-history/runtime.json
python -m src.research.forecast_spatial_mixture.evaluate --plan experiments/forecast_spatial_mixture_20260913/technical/short-history/plan.json --run history3_spatial_mixture4 --seed 20260913
python -m src.research.forecast_spatial_mixture.compare --plan experiments/forecast_spatial_mixture_20260913/technical/short-history/plan.json
```

Repeat the train/evaluate commands for the 14 new runs in the plan. The final
comparison also requires the six declared reference assays. To execute the recorded
serial sequence within its active allocation:

```bash
python -m src.training_methods.embedding_forecast.allocation --plan experiments/forecast_spatial_mixture_20260913/technical/short-history/allocation.json
```

The submitted detached execution uses existing tracking (`experiment_registry.py
run --spec`) and frozen source. It waits for nodesumo01's current training queue
(allocation 990987), then runs the short deterministic fits first. The existing
small physical-inference workload may overlap those deterministic fits, as already
verified for the first controls. Before starting spatial training, it requires the
original analysis to finish. Each new fit is followed immediately by its local
physical assay; comparison follows all fits. Node53's current queue continues.

The allocation ends September 14 at 08:19:56 UTC. The driver deadline is 08:09:56 UTC
and refuses to start a fit with under one hour left. Prerequisite failures and
nonzero child return codes fail explicitly, preserving logs and exact-resume state.
This is a fresh-training sequence, not an automatic restart protocol. New forecasts
use the same immutable host-validation trainer as the 12 ps spatial fits. The
analysis snapshot includes configurable comparison pairs and retained-reference
paths. Original job files and source snapshots are unchanged.

Validation: 13 focused tests passed, covering anchor-only spatial gradients and
clean-anchor augmentation, physical scoring, paired statistics, metric contracts,
failed dependencies, duplicate launch rejection and deadline enforcement. A real-cache
preflight verified matched sampled test futures/atom IDs/origins across all seven
new configurations, exact within-family parameter counts, and finite 12-frame forecasts.

No new short-history result is available yet. Maintained additions are the existing
family's allocation executor and configurable reference/comparison inputs. This file
and its JSON files are experiment records. Launch receipts, logs, source snapshots
and preflight output are generated run artifacts under the result's `technical/`.
