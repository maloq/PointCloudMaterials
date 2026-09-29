# Latest native MACE quality evaluation

The [scientific protocol](../experiments/encoder_quality_20260926/README.md)
applies the [quality criteria](encoder_research/quality_criteria.md) to six
completed supervised encoders and two epoch-12 pretrained encoders. No encoder
training is launched. The [configuration](../configs/analysis/encoder_quality_latest_20260926.json)
pins all eight checkpoint hashes and their original tensor producers.

Use conda `pointnet-torch214` from the repository root:

```bash
python -m src.research.encoder_quality.run --config configs/analysis/encoder_quality_latest_20260926.json --name scratch-hot --check-only
python -m src.research.encoder_quality.queue submit --config configs/analysis/encoder_quality_latest_20260926.json --node node58
python -m src.research.encoder_quality.queue report --config configs/analysis/encoder_quality_latest_20260926.json
```

The check uses real inputs and GPU inference, verifies geometric consistency and
replays saved features, and creates no online run. Submission requires its exact
configuration/source receipt and the completed static reference. It freezes the
source, recipes and metric contracts before submitting one GPU for up to 12 hours.
Each model runs in a separate process; a failed model is recorded and later models
remain eligible to run. Inspect `technical/launch.json`, `queue-state.json`,
per-model logs and failure receipts. Do not resubmit an active queue.

The September26 queue completed all eight evaluations as Slurm job `1009457`
on node58 in about16 minutes, with no failed stage. All completion hashes and
geometry checks were verified. See the
[results and interpretation](../experiments/encoder_quality_20260926/RESULTS.md).

Results resolve to `${storage:analysis}/encoder_quality/latest-mace-20260926`:
`index.html`, `plots/`, `tables/`, and `technical/evaluations/<model>/`. The overview
refreshes after each model; the report command can refresh it earlier. A complete
marker requires every scientific stage and saved metric table to finish.

Execution uses the current native MACE `ir_mul` fused convolution, compiled
tensor encoder, vectorized GPU geometry, and padded 256×80 inference batches.
The actual four inference source files must match the frozen training producers.
Saved all64 features are reused only after checkpoint, row and numerical replay
verification. New generated feature caches share the global six-entry retained
cache with protected leases. Descriptor controls are fitted once per input domain.
Frozen diagnostic readouts and descriptor controls stay local. Their progress and
summaries are saved under `technical/evaluation-tracking/<component>/<kind>/`,
alongside retained readout checkpoints and evaluation predictions/metrics. Cached
readouts return verified saved predictions without opening a tracking session.
No separate W&B run is created for a model, checkpoint, readout or control in
this evaluation workflow. Scientific encoder/predictor fits still run online;
associated final metrics update an explicitly recorded training run through the
API, without creating or restarting a run. Already-running frozen jobs keep
their original logging; this policy applies to subsequent launches.

The missing September23 reference was rebuilt with its unchanged producer and
recipe. Its nine populations reproduce the saved input coordinates and class
counts. Dense spatial panels retain eight times the original slab sample and
marker area 1.25 points². Input hashes and definitions are exported with results.

See [metric definitions](metrics/encoder_quality.md) for restrictions: static
snapshots are relaxed; observed-trained encoders therefore undergo transfer on
those snapshots. Dense relaxed dynamics and an accepted regional-birth prediction
population are unavailable in this release. Neither is silently substituted.
