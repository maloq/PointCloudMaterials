# Learned spatial attention for future local-structure embeddings

Question: does learning from individual neighboring structure embeddings and their
relative geometry improve local-crystallization forecasts over a fixed spatial mean?
The [completed context-size sweep](../forecast_spatial_mixture_20260913/SPATIAL_CONTEXT_RESULTS.md)
identified 32 cached neighbors as the strongest overall pooling extent. This follow-up
keeps that extent and tests a learned spatial encoder at 12 and 3 ps observed history,
with two seeds per history. Twelve ps fits and their paired report take priority.

Each observed frame supplies the tracked center's embedding and its 32 nearest other
cached centers under periodic boundaries. Every neighbor keeps its own embedding
and minimum-image relative position until the learned aggregation. A two-layer SiLU
network embeds its standardized embedding difference from the target; another embeds
16 Gaussian radial features between 0 and 60 angstrom. Four attention heads use the
target temporal token as query, individual neighbor keys/values, and a learned radial
bias. Two residual attention blocks refine the target token. Attention width is 64.

Each head also aggregates neighbor direction vectors. Their 4-by-4 Gram matrix
captures directional arrangement through rotation/reflection-invariant scalars.
The output network combines learned values, that matrix, and mean/outer distance,
then adds its result to the target's temporal token. Permuting neighbor order or
rotating the relative geometry leaves scalar predictions invariant up to numerical
precision. No coordinate axis is given a preferred physical meaning. Scalar frozen
embeddings are assumed to retain their producer's invariance. This custom attention
design is informed by spatial-attention work such as
[Point Transformer](https://arxiv.org/abs/2012.09164); it is not a replication.

The temporal encoder remains a four-layer GRU of width 512, followed by four
Gaussian components over the complete 12-by-256 future path through 9 ps. Component
identity is shared across future frames. Joint mixture NLL selects checkpoints on
validation. The new model has 9,319,876 parameters, compared with 9,198,852 for the
mean-pooling control (approximately 1.3% more). Any gain cannot be attributed purely
to attention weights without further parameter-matched architectural controls.

Use all 125 simulations, split 74/24/27, and all 1,024 cached centers per source.
Keep the 12 ps common eligibility grid, 6 ps training origin stride, 0.75 ps cadence,
train-derived normalization and 9 ps future unchanged. Each fit uses 12 epochs,
effective batch 8,192, AdamW at 0.0003, the original warmup/cosine schedule, clipping 5,
and central-history noise 0.01 / frame dropout 0.15. Neighbors and geometry are
observed inputs and are not corrupted by that central-history augmentation.
Microbatches of 1,024 retain one optimizer update per original batch. The spatial
branch uses bfloat16 autocast on CUDA; the GRU, predictions and likelihood use float32.
Thus spatial architecture and its internal precision differ from the mean control;
the data, temporal architecture, objective and update budget remain fixed.

The physical assay reuses the frozen center PTM labels and train-only readout.
Primary truth is the first three-frame crystalline episode; nine-frame persistence
is a sensitivity analysis. Validation selects thresholds separately per fitted model,
readout and horizon. Report transition F1, precision/recall, detected-event timing
MAE, all-positive-window recall within 1.5 ps, and one-origin-per-event fixed-lead
recall. Compare both mean-path scores and distribution readouts. Paired comparisons
use the same completed seeds and bootstrap whole test sources; the previously
inspected test set makes all results exploratory.

Monitor full-path MSE, joint NLL, CRPS, coverage, energy score and component occupancy.
Additional spatial diagnostics are attention entropy, effective neighbor count,
maximum weight and attention-weighted radius. They test whether the network learns
selectivity but do not establish causal neighbor importance. No improvement is claimed
before the completed assays.

The maintained implementation is in
[`spatial_attention.py`](../../src/training_methods/embedding_forecast/spatial_attention.py)
and [`attention_data.py`](../../src/training_methods/embedding_forecast/attention_data.py).
[Configurations](../../configs/embedding_forecast/spatial-attention/analysis.json),
[outputs](../../output/embedding_forecast/spatial-attention-20260914), and
[execution measurements](../../docs/forecast_performance.md) retain the exact protocol.

With conda `pointnet`, use the existing commands:

```bash
python -m src.training_methods.embedding_forecast.spatial --stage geometry --config configs/embedding_forecast/spatial-attention/geometry.json
python -m src.training_methods.embedding_forecast --config configs/embedding_forecast/spatial-attention/history12_attention32_mixture4.json --stage train --seed 20260913 --runtime-config configs/embedding_forecast/spatial-attention/runtime.json
python -m src.research.forecast_spatial_mixture.evaluate --plan configs/embedding_forecast/spatial-attention/analysis.json --run history12_attention32_mixture4 --seed 20260913
python -m src.research.forecast_spatial_mixture.compare --plan configs/embedding_forecast/spatial-attention/analysis.json
```

Reproduction requires fresh outputs. Active training uses frozen source and recipes;
existing mean-pooling checkpoints and their original analyses remain unchanged.
