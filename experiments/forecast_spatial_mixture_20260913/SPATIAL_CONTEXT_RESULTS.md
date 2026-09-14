# Completed spatial-context and history results — September 14, 2026

**12 ps history with 32 neighboring cached centers is the strongest observed
configuration overall.** Its improvement over eight centers is modest and replicated.
Increasing the single pooled neighborhood to 128 or 512 centers brings no further
transition-F1 benefit. Detecting the exact transition time remains much harder than
predicting the local crystal/noncrystal state.

The twelve new fits and all physical assays completed. The original twelve-fit
study and fourteen-fit shorter-history extension are also complete. This analysis
uses two seeds per condition, 27 independent test simulations, and the existing
validation-selected checkpoints and thresholds. The target is the first crystalline
episode of the tracked local center: PTM FCC/HCP/BCC for three consecutive sampled
frames. It is not the first nucleus anywhere in the simulation.

[Six-figure gallery](../../output/embedding_forecast/spatial-context-analysis-20260914/index.html)
and [full source tables](../../output/embedding_forecast/spatial-context-scale-20260913/tables/).

## Spatial extent: a moderate context wins

The table uses 12 ps history and the probability readout of the full-path mixture,
with a 9 ps forecast horizon. Values are arithmetic means over the two fitted seeds.
Timing MAE includes correctly detected positive windows only; timed recall also
counts missed positive windows in its denominator.

| Nearby cached centers | Typical outer radius | Transition F1 | Precision | Recall | Timing MAE | Recall within 1.5 ps | Embedding MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 13.0 Å | 55.76% | 49.91% | 63.18% | 2.160 ps | 33.32% | 0.204542 |
| **32** | **20.9 Å** | **56.86%** | **50.04%** | **65.83%** | **2.124 ps** | **35.16%** | **0.204140** |
| 128 | 33.2 Å | 55.78% | 47.63% | 67.34% | 2.176 ps | 35.34% | 0.204409 |
| 512 | 52.7 Å | 54.89% | 48.10% | 63.97% | 2.216 ps | 33.01% | 0.204979 |

Radius is the median of the 27 test-source median outer-neighbor distances. These
are sampled local-structure centers, each with its own embedding, not nearest
individual atoms. The model still receives one neighborhood mean and distance
summaries; the experiment varies its extent while holding model capacity fixed.

![Spatial context comparison](../../output/embedding_forecast/spatial-context-analysis-20260914/plots/spatial-context.png)

At 9 ps, paired differences relative to eight centers are:

| History | 32 centers | 128 centers | 512 centers |
| --- | --- | --- | --- |
| 3 ps | +1.24 pp [0.62, 1.82] | +1.01 pp [−0.06, 1.97] | −0.99 pp [−1.91, −0.08] |
| 12 ps | +1.10 pp [0.38, 1.85] | +0.01 pp [−0.97, 0.96] | −0.88 pp [−1.61, −0.17] |

Brackets are paired 95% source-bootstrap intervals, computed from seed-averaged
confusion counts. They are exploratory, unadjusted for multiple comparisons, and
conditional on these fitted seeds. The 32-center gain is small; it is not a large
jump in transition predictability. At 12 ps history the embedding-MSE decrease
relative to eight centers is 0.000402, about 0.20%, with paired interval
[−0.000482, −0.000323] for the signed difference.

The stronger nine-frame persistence definition gives the same F1 ordering for
12 ps history: **49.99%, 51.45%, 50.28%, 49.09%** for 8/32/128/512 centers.
For 12 ps history, the 32-center transition-F1 point estimate also exceeds eight
centers in all five temperature strata. The largest difference is at 520 K,
66.14% versus 61.83%, but that stratum has only three test sources. The other
temperatures each have six sources; this descriptive breakdown has no separately
tuned thresholds or new confidence intervals.

A plausible interpretation is that very broad averaging dilutes useful nearby
structure. This is a hypothesis: the experiment does not isolate that mechanism
from other consequences of changing the spatial summaries.

## Timing: useful warnings, limited precision

For the 12 ps / 32-center model:

| Forecast horizon | Transition F1 | Precision | Recall | MAE among detections | All-positive-window recall within 1.5 ps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 ps | 41.57% | 33.44% | 54.97% | 0.809 ps | 48.92% |
| 6 ps | 51.55% | 45.04% | 60.28% | 1.552 ps | 39.48% |
| 9 ps | 56.86% | 50.04% | 65.83% | 2.124 ps | 35.16% |

At 9 ps, roughly half of positive warnings are correct. Among correctly detected
positive windows, about 53.4% have timing error at most 1.5 ps; the median absolute
error is 1.5 ps and the 90th percentile is 4.5 ps. Average signed error is −0.207 ps,
so the small mean bias hides substantial early and late errors. Sampling cadence
is 0.75 ps. Event prevalence and thresholds change with horizon, so rising F1 across
horizons does not establish that longer-term timing is easier.

The repeated-window assay includes 767,716 eligible test origins, only 14,916 of
which contain a transition within 9 ps: **1.94% positive**. An always-negative
forecast already achieves **98.06% accuracy**, nearly the model's 98.06%. F1,
recall, precision and timing therefore carry the relevant information. For the
different task of crystal/noncrystal state at exactly +9 ps, the model reaches
96.70% accuracy and 95.67% F1; that includes structures already crystalline before
the forecast and should not be quoted as transition accuracy.

The stricter fixed-lead assay asks for one warning per eligible event, exactly
3/6/9 ps before its onset. Detection recall is **41.25%, 39.27%, 41.21%**;
recall with timing within 1.5 ps is **22.98%, 18.41%, 16.12%**, respectively.
Those evaluations contain 1,212/1,323/1,365 eligible events, so the cohorts differ
slightly. Truth occurs at the last forecast frame, making fixed-lead timing error
early or zero by construction. This assay cannot quantify late predictions.

![Detection versus precise timing](../../output/embedding_forecast/spatial-context-analysis-20260914/plots/timing-and-lead.png)

## History and probabilistic forecasts

The completed eight-center mixture sweep gives these 9 ps transition F1 scores:

| Observed history | Anchor only | 1.5 ps | 3 ps | 6 ps | 12 ps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-center mixture | 47.31% | 51.78% | 53.64% | 54.42% | 55.76% |
| Nonspatial deterministic | 32.08% | 43.02% | 46.87% | 50.08% | 50.78% |

At 32 centers, increasing history from 3 to 12 ps raises F1 from **54.86% to
56.86%**: paired +2.00 pp [1.11, 2.95]. Both short and long history are useful;
longer context retains a measurable benefit. No intermediate 32-center histories
were trained in this sweep.

The completed original two-seed controls refine the earlier interim conclusion.
At 12 ps, nonspatial deterministic, spatial deterministic, nonspatial K=1,
nonspatial K=4 and spatial K=4 achieve **50.78%, 54.51%, 52.02%, 51.22%, 55.76%**
transition F1. Probabilistic models use their probability readout here. Four
components without spatial input do not show a clear advantage over one Gaussian:
paired −0.80 pp [−1.55, 0.03]. Spatial K=4 with its distribution readout beats
spatial deterministic by +1.25 pp [0.57, 1.92], while its mean-path readout has no
clear advantage over spatial deterministic. Spatial information and how uncertainty
is used both matter; component count alone has not solved transitions.

For the 12 ps / 32-center model, marginal 90% embedding intervals cover **90.22%**
of targets and the mean effective component count is approximately **3.82**.
All components receive substantial gate weight, although this does not demonstrate
four distinct physical pathways. These embedding probabilities have not been
calibrated as probabilities of sustained PTM onset.

Full-path MSE is **0.20414**, approximately 45.7% below anchor persistence and
15.7% below the observed-history mean. Replacing central history by the repeated
anchor raises MSE to **0.24689**; reversing the past gives **0.20931**. These
saved interventions keep the anchor and spatial history unchanged, so they test
central temporal information conditional on the retained spatial context.

Every new fit selected epoch 12, and validation NLL was still decreasing. The
ranking is established for this matched training budget; it is not evidence that
all architectures have converged. Both seed-specific test F1 values for the best
condition are close: 57.06% and 56.67%.

## What this supports next

Keep **12 ps / 32 centers / full-path mixture** as the reference for transition
work, with the 3 ps version as a shorter-history control. The current data support
preserving near and far spatial summaries separately in a future experiment,
rather than replacing local information with an increasingly broad mean. Treat
that as a proposed ablation, not a demonstrated improvement.

The larger remaining opportunity is event timing: a future loss/readout should
represent first sustained onset and missed events directly, with probability
calibration measured on validation data. Extra mixture modes or greater spatial
extent alone do not address the gap between good state classification and modest
transition timing. Confirm any selected improvement on new independent test sources;
these sources have been inspected repeatedly and the comparisons are exploratory.

## Reproduction and provenance

```bash
conda run -n pointnet python -m src.research.forecast_spatial_mixture.compare --stage summarize --plan configs/embedding_forecast/spatial-context/report-20260914.json
```

Use a fresh output for reproduction. The report reads the completed collectors'
JSON and verifies score/status consistency against retained producers; it does not
rerun prediction or change thresholds. Figures are available as PNG and PDF, with
summary CSVs, frozen metric definitions, implementation hashes and input hashes.
The spatial source-statistics count interpretation was checked against every
corresponding original pooled confusion matrix. Ten existing analysis/metric-layout
regression tests passed. The original collected tables and frozen execution sources
remain unchanged.
