# How much surrounding spatial context helps local transition prediction?

[Completed results and plots](SPATIAL_CONTEXT_RESULTS.md): 32 cached neighbors with
12 ps history gives the best observed overall performance; broader pooling does not
bring further transition-F1 gains.

Compare mean embeddings of the 32, 128 and 512 nearest OTHER cached centers with
the retained 8-center reference. Test both 3 and 12 ps observed history, with seeds
20260913 and 20260914 for every condition: twelve new fits. These neighborhoods
cover increasingly broad regions of each periodic simulation. They consist of
cached local-structure centers, not nearest individual atoms. Cache records retain
actual mean and median outer distances in angstrom; do not infer exact radii solely
from the neighbor count. The maximum context uses half of the 1,024 cached centers.

Hold the scientific controls fixed: all 125 source simulations (74 train, 24 val,
27 test), 256-dimensional frozen embeddings, common 12 ps eligibility, 6 ps origin
stride, 9 ps full future (twelve 0.75 ps samples), and inherited train-only scaling.
Each fit uses a 512-wide four-layer GRU, four whole-path Gaussian components,
12 epochs, batch 8192, AdamW, and the same noise/frame-dropout augmentation. Use the
same neighbor mean plus mean/outer distance inputs; only neighborhood count and
observed history vary. This tests wider context without confounding it with capacity,
training duration, source selection or validation selection.

The explicit prepool cache reduces same-frame neighbor embeddings in float32 on
CPU, then stores them in the original float16 cache dtype. Training, validation,
test and local inference use exactly these stored values. This retains causal
observed-frame pooling and bounds memory independently of the neighbor count.
The existing eight-neighbor references used CPU/GPU float32 reduction followed by
float16 storage; bitwise equivalence across reduction devices is not asserted.
Their original checkpoints, assay arrays and source snapshots are preserved.

Select the lowest validation joint-mixture NLL. Reuse the frozen local PTM truth
and crystal readout. Assess first local three-frame crystallization, nine-frame
sensitivity, validation-selected thresholds, F1/precision/recall, timing MAE and
all-event recall within 1.5 ps; retain the fixed-lead event-level assay. Embedding
metrics include MSE, proper NLL, CRPS, marginal coverage and whole-path energy score.
Compare each context against the matching eight-neighbor model, adjacent context
sizes, and 12 versus 3 ps history. Retain deterministic history controls. Bootstrap
independent test sources with matched seed-averaged confusion counts as before.

Publish a paired cohort report after each neighborhood size completes, then the
full paired comparison. No improvement is claimed before those evaluations finish.
The test simulations have already been inspected; this is exploratory research.

Recipe directory: [`configs/embedding_forecast/spatial-context`](../../configs/embedding_forecast/spatial-context/analysis.json).
Output: [`output/embedding_forecast/spatial-context-scale-20260913`](../../output/embedding_forecast/spatial-context-scale-20260913).
Exact cache configurations and all per-fit commands are retained in the allocation
plan and immutable source snapshot under that output. Operational timing and storage
details are in [forecast performance](../../docs/forecast_performance.md).

With conda `pointnet`, an example condition is:

```bash
python -m src.training_methods.embedding_forecast.spatial --config configs/embedding_forecast/spatial-context/spatial32.json
python -m src.training_methods.embedding_forecast --config configs/embedding_forecast/spatial-context/history3_spatial32_mixture4.json --stage train --seed 20260913 --runtime-config configs/embedding_forecast/staged-validation.json
python -m src.research.forecast_spatial_mixture.evaluate --plan configs/embedding_forecast/spatial-context/analysis.json --run history3_spatial32_mixture4 --seed 20260913
python -m src.research.forecast_spatial_mixture.compare --plan configs/embedding_forecast/spatial-context/analysis.json
```

Use a fresh output for reproduction; the actual detached queues execute their
retained source snapshot and resolved recipes. Existing queued short-history jobs
continue with their original source and plans.
