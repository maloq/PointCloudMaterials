# Interim local-crystallization results — September 13, 2026

Spatial context gives the strongest replicated transition improvement. The spatial four-component model has the best first-seed detection/recall result, but its second seed is still training. Adding mixture components without spatial context improves embedding likelihood without a clear transition advantage over one Gaussian.

## Scope and protocol

We assessed **11 completed fits**: all six conditions for seed 20260913 and five conditions for seed 20260914. The second Gaussian fit finished during analysis and is included in the final two-seed comparison. The remaining spatial-mixture fit and new shorter-history sweep are excluded from these fixed interim cohorts.

The target is the **first crystalline episode of the tracked local center**, identified by FCC/HCP/BCC PTM for three consecutive sampled frames. The nine-frame persistence sensitivity is separate. All models predict the same next 9 ps at 0.75 ps cadence. Independent sources are split 74/24/27 for training/validation/test. The physical assay uses the same 64 centers per source and 765 forecast origins; standard embedding scores use all 1,024 cached centers.

Thresholds maximize validation F1 separately for each model/readout/horizon; test labels select neither thresholds nor checkpoints. Event scores use eligible pre-onset origins with three physically noncrystalline observed frames. The probability readout is an embedding-distribution crystal probability, not a calibrated probability of sustained physical onset.

## Matched first-seed comparison at 9 ps

All six rows use seed 20260913. Probabilistic rows use their probability readout; deterministic rows use the mean-path readout. The full exports also retain the mean-path results of every probabilistic model.

| Model | Precision | Recall | F1 | Timing MAE, detected events | All-positive-window recall within 1.5 ps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 ps deterministic | 49.24% | 51.71% | 50.45% | 2.264 ps | 26.18% |
| 12 ps deterministic | 47.76% | 54.44% | 50.88% | 2.237 ps | 28.05% |
| 12 ps + spatial | 51.48% | 58.34% | 54.70% | 2.151 ps | 31.04% |
| 12 ps + Gaussian (K=1) | 46.81% | 59.25% | 52.30% | 2.175 ps | 30.87% |
| 12 ps + mixture (K=4) | 46.52% | 56.51% | 51.03% | 2.297 ps | 28.82% |
| 12 ps + spatial + mixture (K=4) | 49.76% | 63.75% | 55.90% | 2.163 ps | 33.80% |

The spatial mixture improves recall, with a precision tradeoff relative to spatial deterministic forecasting. Its mean-path F1 is 54.62%, almost identical to the spatial deterministic model's 54.70%; the probability readout raises it to 55.90%. Conditional timing MAE is essentially unchanged at about 2.15–2.16 ps. More correct detections, rather than much finer timing of detected events, explain most of the timed-recall gain.

![Matched first-seed transition results](../../output/embedding_forecast/context-space-mixture-interim-seed1-20260913/plots/local-transitions.png)

## Replication across two seeds

Paired confidence intervals resample the 27 independent test sources 2,000 times. Confusion counts are averaged across the same two seeds before computing each paired F1 difference; this is neither a prediction ensemble nor the arithmetic mean of per-seed F1. Differences below are percentage points.

| Change | Readout | F1 difference | Paired 95% interval |
| --- | --- | ---: | ---: |
| 6 → 12 ps history | Mean path | +0.70 | [+0.22, +1.21] |
| Add spatial context | Mean path | +3.74 | [+2.63, +4.98] |
| Gaussian probability vs deterministic | Probability | +1.25 | [+0.34, +2.17] |
| Four components vs one Gaussian | Probability | -0.80 | [-1.55, +0.03] |

The spatial gain repeats in both seeds: F1 is 54.70% and 54.33%, versus 50.88% and 50.67% for their matched 12 ps deterministic controls. Longer history has a smaller positive effect. Four nonspatial components have no demonstrated probability-readout F1 advantage over one Gaussian; the interval includes zero and the point estimate is negative.

For the combined spatial mixture, the first-seed probability-readout gain over spatial deterministic forecasting is +1.20 points, with a source-bootstrap interval of [+0.55, +1.81]. This remains a single-seed result. There is no spatial K=1 control, so it does not isolate the benefit of four components from that of probabilistic forecasting.

## Timing at a fixed 9 ps lead

The rolling-origin table has 767,716 eligible test windows and 14,916 positive event windows: 1.94% prevalence. Always predicting no transition would have 98.06% accuracy and zero recall. Plain accuracy is therefore not an adequate transition metric. These overlapping windows are not 14,916 independent physical transitions.

To assess an actual nine-picosecond warning, we also use one origin exactly 9 ps before each of the same **1,365 eligible local events**. For seed 20260913:

| Model | Event recall at fixed 9 ps lead | Recall timed within 1.5 ps | Conditional timing MAE |
| --- | ---: | ---: | ---: |
| 12 ps deterministic | 27.91% | 5.86% | 4.435 ps |
| 12 ps + spatial | 32.09% | 9.82% | 3.724 ps |
| 12 ps + spatial + mixture (K=4) | 38.90% | 14.73% | 3.321 ps |

Fixed-lead errors can only be early or zero because the true onset lies at the last forecast frame; this assay cannot measure late errors. The spatial mixture is better here, but most events are still missed or timed outside 1.5 ps.

For its probability readout at 3/6/9 ps horizons, rolling-window F1 is 41.44/51.09/55.90%, while conditional timing MAE is 0.836/1.579/2.163 ps. These horizons have different positive prevalences and timing ranges; higher F1 at 9 ps does not mean more accurate long-lead timing.

## Persistence and probabilistic trajectory quality

Requiring nine consecutive crystalline frames reduces first-seed 9 ps F1 to 46.81% for 12 ps deterministic, 50.01% for spatial deterministic, and 50.22% for spatial-mixture probabilities. The extra mixture-versus-spatial gain is only +0.20 points, with interval [−0.46, +0.92]. Spatial context remains useful; the additional mixture gain is not established for this stricter event.

On embedding trajectories, first-seed source-mean MSE falls from 0.206928 to 0.204119 with spatial context. The spatial mixture scores 0.204524: it improves event recall without improving mean-path MSE over spatial deterministic forecasting.

Nonspatial K=4 improves first-seed joint NLL from 0.337453 to 0.281737 nats per coordinate relative to K=1, while CRPS barely changes (0.222663 → 0.222640) and energy score changes only slightly (0.310734 → 0.310558). Marginal 90% coverage stays near 90.2–90.4%. The four-component gates use about 3.55 effective components without spatial context and 3.82 with it; these summaries do not establish four distinct crystallization pathways. Lower embedding NLL alone is not evidence of better transition timing.

## Interpretation and reproducibility

The current evidence favors spatial information more strongly than extra history length or additional nonspatial mixture components. The combined spatial probability forecast is promising for detection and warning recall, while precise onset timing remains limited. These are exploratory results on previously examined test sources; bootstrap intervals condition on the fitted seeds and do not correct for all comparisons.

- [All six conditions, first seed: tables and figure](../../output/embedding_forecast/context-space-mixture-interim-seed1-20260913/README.md).
- [Five conditions, both seeds: paired tables and figure](../../output/embedding_forecast/context-space-mixture-interim-two-seed-20260913/README.md).
- [First-seed analysis recipe](../../configs/embedding_forecast/interim-analysis-seed1-20260913.json) and [two-seed recipe](../../configs/embedding_forecast/interim-analysis-two-seed-20260913.json).
- [Metric definitions](../../docs/metrics/forecast_spatial_mixture.md); each table export retains its frozen definitions and implementation hashes.

The maintained `src.research.forecast_spatial_mixture.evaluate` command produced the local assays. `python -m src.research.forecast_spatial_mixture.compare --plan PLAN.json` reproduces each paired export; choose a fresh output location while retaining the completed `local_directory` inputs. Collection verified matched embedding row identities, exact normalization, cache identity and selected-checkpoint hashes.
