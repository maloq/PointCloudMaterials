# How much observed history improves an embedding forecast?

**Completed:** all 24 fits succeeded at 13:43 UTC on September 13; paired
comparison and plots also completed. Longer history improves both models, with
roughly 90% of the anchor-to-24 ps gain already obtained at 6 ps. See
[RESULTS.md](RESULTS.md) for the findings and limitations.

Compare **0, 1.5, 3, 6, 12 and 24 ps** of observed history when predicting the
same next **9 ps** of local-structure embeddings. Zero history means one observed
anchor embedding. At 0.75 ps cadence, the six settings contain 1, 3, 5, 9, 17 and
33 embeddings, respectively. The user confirmed this interpretation on September 13.

## Protocol

The compact sweep contains **24 fresh fits**: six histories, direct and
autoregressive predictors, and seeds 20260913/20260914. Both use width 128 and two
GRU layers, 16 epochs, batch size 8,192, AdamW learning rate 0.001, two warmup
epochs, cosine decay, and gradient clipping at 5. Patience equals the epoch budget.
History augmentation retains standardized noise 0.01 and frame dropout 0.15;
the final anchor stays clean. Models predict all twelve future embeddings by
self-conditioned rollout or direct decoding, with frame-MSE loss only.

AR has 579,072 parameters and direct has 380,160. Parameter counts are constant
across history lengths within each architecture. This compares context lengths
within each method, not direct versus AR at identical model capacity.

All fits reuse the verified float16 embedding cache and independent train/val/test
splits from the enlarged forecast campaign: **74/24/27 sources**, each with
1,024 tracked centers. Anchors are shared at 24, 48, …, 576 ps, with future targets
at +0.75, +1.5, …, +9 ps. Each context is an exact suffix of the same 24 ps history.
This gives **1,818,624 / 589,824 / 663,552** train/validation/test windows and
222 training updates per epoch. The sparse anchor grid reduces training windows
by about 32.5 times relative to the enlarged run, while preserving all sources.
Unique training embeddings determine the same normalization for every context.
No additional simulations, encoder fitting, or embedding quantization is needed.

Checkpoint selection uses source-mean validation MSE. The paired final analysis
reports source-mean path MSE, per-frame errors including +9 ps, separate
(0,3], (3,6], (6,9] ps bin errors, seed spread, and paired whole-source bootstrap
gain over each architecture's trained anchor-only predictor. It also includes
history-mean and persistence baselines and the existing history interventions.
The exporter verifies matching held-out row identities, normalization, cache and
implementation identity, and equal parameter counts across contexts.

The test sources have been examined previously, and two seeds provide an initial
estimate of fit variability. Treat the comparison as exploratory. Longer history
can improve denoising without using temporal order; examine history-mean errors
and reversal sensitivity alongside anchor gains. Fixed compact models and a
16-epoch budget do not establish the best context for the large production fits.

## Execution and reproduction

Training started **2026-09-13 11:57 UTC**, detached in existing H100 allocation
**990987**, step **990987.1**, on **nodesumo01**. It uses this session's idle GPU;
the two enlarged fits continue in their separate allocations. All six contexts
run for the first seed before repeating for the second. Within a seed the order
is 0, 6, 24, 1.5, 3 and 12 ps, each AR then direct.

The existing maintained trainer is reused with configuration changes only:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/forecast_context_20260913/technical/history6ps.json \
  --stage train --seed 20260913 \
  --runtime-config experiments/forecast_context_20260913/technical/runtime.json
```

The submitted commands use the immutable September 13 GPU-resident training
snapshot, recorded in [technical/plan.json](technical/plan.json). They are listed
in the generated `technical/train_matrix.sh` at the output root. Fresh training
refuses existing fit outputs. For an explicit continuation, select the existing
variant/seed, add `--resume`, and use the same frozen source/configuration.

The distinct cross-context analysis reuses forecast errors rather than retraining:

```bash
conda run --no-capture-output -n pointnet python -m src.research.forecast_context.compare \
  --plan experiments/forecast_context_20260913/technical/plan.json
```

A detached CPU step in allocation 990987 waits for the tracked training matrix
to succeed, then runs this comparison from a separate immutable analysis snapshot.
It fails explicitly if training fails or remains incomplete at its 20:00 UTC
dependency deadline. No new Slurm allocations were submitted.

## Results and verification

Outputs: [`output/embedding_forecast/context-pilot-20260913/`](../../output/embedding_forecast/context-pilot-20260913/).
The complete sweep and paired comparison succeeded. Its root links to logs and contains
`RESULTS.md`, `tables/context-quality.csv`, `tables/METRICS.md`,
`plots/context-quality.png` and `plots/horizon-errors.png` after collection.
Per-fit checkpoints, logs, retained paired test errors and metrics live under
`technical/fits/`. The training and analysis execution records are under
`technical/training/` and `technical/comparison/`.

The GPU preflight exercised both models at 1, 9 and 33 history frames with the
actual batch size, augmentation and optimizer. Warmed updates ranged from
0.020–0.046 s for AR and 0.011–0.037 s for direct, excluding gathering/metrics.
Production epoch timings and actual progress remain in the per-fit logs.
Exact history suffixes, future targets and metadata were checked across every
source for all six contexts. **Ten context-comparison/layout tests passed**,
including source weighting, paired gains, identity/normalization mismatch
rejection, and complete table/plot/metric-definition export.

Maintained implementation: the existing forecast trainer plus the distinct
comparison in `src/research/forecast_context/` and its tests. Versioned experiment
records: this README and `technical/*.json`. Disposable diagnostics and generated
launchers/logs live under the output root's `technical/`; checkpoints and paired
test errors remain protected research state. Metric formulas and implementation
hashes are documented in `docs/metrics/forecast_context.md` and `contracts.json`.
