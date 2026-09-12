# Embedding forecasting pilot — September 11, 2026

This records the initial nine-fit pilot. The subsequent
[five-fit autoregressive comparison](AUTOREGRESSIVE_RESULTS.md) finds that
training on predicted rollouts improves the full-path result. The common
collector now includes all fourteen fits.

**Both forecasting tasks are implemented and ran successfully on real MEAM
trajectories. Predicting a correction to the observed history mean is a better
starting architecture than correcting the last embedding. The pilot does not
yet establish a benefit from ordered history over a trained mean-only control.**

Nine fits completed on the available H100 in `pointnet`. Each uses seed 20260911,
the same frozen original-VICReg single-frame MACE checkpoint, 6 ps of embedding
history and 9 ps of future at 0.75 ps cadence. The three mean targets are separate
(0,3], (3,6], (6,9] ps bins. The cache follows eight center atoms per source at
three contexts: 432 training, 144 validation and 144 test examples from 18/6/6
independent source lineages. The test cohort includes two sources per temperature
at 400/450/510 K. All target scaling is fitted on training embeddings only.

This is a small, adaptively developed pilot, including sources already inspected
in earlier research. Intervals below are exploratory resampling of six whole
sources conditional on one fitted seed. There is no optimization-seed uncertainty
estimate and no untouched confirmation cohort.

## Held-out accuracy

MSE is measured in the common training-standardized embedding space. The second
column scores all 12 future frames; the third scores the three bin means. Path
models are averaged into those bins for the third column, allowing direct
comparison with the bin models. Lower is better. Equal-source and sample-weighted
means coincide here because every test source has 24 examples.

| Forecast | Full-path MSE | Three-bin MSE |
| --- | ---: | ---: |
| Repeat last embedding | 0.398486 | 0.264134 |
| Repeat observed history mean | 0.263882 | 0.129530 |
| Anchor-residual GRU, real history | 0.314955 | 0.180411 |
| Anchor-residual GRU, trained anchor control | 0.310510 | 0.176159 |
| Direct bins, anchor-residual GRU | — | 0.178206 |
| Direct bins, trained anchor control | — | 0.175297 |
| **Mean-residual full path, real history** | **0.251330** | 0.117005 |
| Mean-residual full path, trained mean-only control | 0.252864 | 0.118516 |
| **Mean-residual direct bins, real history** | — | **0.111155** |
| Mean-residual direct bins, trained mean-only control | — | 0.117739 |
| Joint Gaussian path, anchor-residual GRU | 0.385650 | 0.251246 |

The initial anchor-residual GRUs improved on persistence but failed to reproduce
the stronger history-mean baseline. Starting directly at the history mean makes
that baseline available in every channel, while the GRU and shared time decoder
learn only a correction. This change was prompted by the initial pilot results;
it is not a pre-registered architecture win. The original five fits are retained
alongside the four mean-residual follow-ups.

| Paired error reduction | Estimate | 95% source-bootstrap interval |
| --- | ---: | ---: |
| Mean-residual path versus persistence | 36.93% | [31.47%, 41.90%] |
| Mean-residual path versus observed history mean | 4.76% | [−0.57%, 12.66%] |
| Mean-residual path versus trained mean-only path | 0.61% | [−2.69%, 3.91%] |
| Mean-residual direct bins versus observed history mean | 14.19% | [1.14%, 29.05%] |
| Mean-residual direct bins versus trained mean-only bins | 5.59% | [−3.19%, 14.30%] |

The full-path model improves over the untrained history mean on three of six
sources; the direct-bin model improves on five of six. The direct-bin model is
slightly better on its own objective than averaging the full-path prediction:
path-derived bin error is 5.26% higher, interval [2.43%,9.84%]. This is a small
single-seed tradeoff: full paths retain time resolution that three means discard.

## Dynamics and uncertainty checks

For the mean-residual full-path model, reversing past observations while keeping
the anchor fixed changes test MSE from 0.251330 to 0.266318. Repeating the anchor
raises it to 0.362549. For direct bins the corresponding errors are 0.111155,
0.126009 and 0.222964. The learned mean-only controls are invariant to reversal
up to floating-point summation. These interventions show sensitivity but are
distribution shifts; the trained-control comparisons above remain inconclusive
about useful order-sensitive information. History can also contribute its spread
and the latest observation, not only its order.

The full-path predicted-change RMS is 0.661 times the observed future-change RMS.
Conditional mean paths need not reproduce thermal noise, so this shrinkage is
not by itself a model defect. It does mean that level MSE alone cannot establish
faithful reconstruction of the stochastic dynamics. Past-only linear
extrapolation performs poorly: path MSE 0.794119, compared with 0.263882 for
history averaging. Extrapolating fluctuations is a weak baseline for this cohort.

The rank-four joint Gaussian completed finite NLL training and produces correlated
whole-path samples. Test metrics are per-coordinate standardized joint NLL 0.443397,
marginal CRPS 0.311396, normalized path energy score 0.430632, 90% marginal coverage
92.09%, and mean 90% interval width 1.964455 standardized units. Its conditional
mean is substantially worse than the deterministic mean-residual model. Coverage
alone does not establish useful uncertainty, and there is no trained probabilistic
baseline or multi-seed calibration comparison yet. Treat it as a working
distributional method, not the selected predictor.

## Next experiment and reproduction

Prioritize the **mean-residual full-path GRU**, its trained mean-only control and
the anchor control on the larger cache with three seeds. Retain the direct-bin
method as the lower-resolution comparison. The [main configuration](main.json)
also contains dense-MLP and transformer alternatives, the original residual
baselines and the Gaussian. The [3 ps history configuration](history3.json)
preserves exactly the same anchor IDs/times/futures. Neither larger campaign has
been launched. See the [full method, losses, metrics and protocol](README.md).

Reproduce the measured pilot in a new configured output directory:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot.json --stage all
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_mean_residual.json --stage train
```

Recollect the already completed fits without retraining:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_comparison.json --stage collect
```

Evidence: [combined metrics](../../output/embedding_forecast_20260911/pilot/runs/comparison.json),
[full-path scores and source gains](../../output/embedding_forecast_20260911/pilot/runs/path_mean_residual-seed20260911/forecast_scores.png),
[full-path metrics](../../output/embedding_forecast_20260911/pilot/runs/path_mean_residual-seed20260911/test_metrics.json),
[direct-bin metrics](../../output/embedding_forecast_20260911/pilot/runs/bins_mean_residual-seed20260911/test_metrics.json).
Each fit retains its selected and last checkpoints, training log, source-level
test errors, example predictions and intervention metrics. Embedding arrays total
about 14.8 MiB; the complete pilot including checkpoints is about 117 MiB.

Verification: **16 tests pass** in `pointnet`, covering physical boundaries,
common anchors across history lengths, batched spawned-worker loading, split
leakage and cache corruption, train-only scaling, historical gradients through
all architectures, dense-versus-low-rank Gaussian likelihoods, uncertainty
scoring, and learning/checkpoint/collection round trips for both objectives.
The real pilot also exercised the maintained trajectory reader, periodic
neighborhood producer, frozen checkpoint loading and GPU training/evaluation.
The report/configurations are experiment records; generated results stay in the
run directory, and the forecast package/tests are maintained code.
