# Local crystallization prediction — completed 2026-09-13

**The forecasts contain useful advance-warning information about local crystallization,
but they do not yet predict its time reliably at long lead times.** AR reaches event
F1 0.502 for onset within the next 9 ps, compared with 0.196 for holding the last
embedding fixed and 0.264 for extrapolating its trend. Its correct warnings have
2.39 ps mean absolute timing error. However, when a forecast starts exactly 9 ps
before the actual transition, it detects only 25.7% of eligible events, and only
4.0% are both detected and timed within 1.5 ps. These are different denominators:
52% recall over all 9 ps windows does not mean 52% recall at a 9 ps lead.

This addresses the user's clarified event: **the tracked atom's local environment
becoming crystalline**, rather than a nucleus appearing somewhere in the simulation.

The assay used 8,000 local trajectories: 64 existing embedded centers in each of
125 independent simulation sources. Whole-source splits remain 74 training,
24 validation and 27 test. The held-out evaluation contains 1,728 local trajectories,
1,335,744 matched forecast origins for state evaluation, and 781,500 origins still
at risk of their first local onset. The two full-size models finished 32 training
epochs; evaluation uses validation-selected checkpoints from epoch 6 for AR and
epoch 12 for direct prediction. Both receive 6 ps of history and forecast the next
12 frames, spaced by 0.75 ps. This assay does not retrain their forecast weights.

Physical crystal labels come from PTM at the exact tracked center: FCC/HCP/BCC,
RMSD cutoff 0.1. The primary event starts at the first of three consecutive crystal
frames; it needs 1.5 ps of subsequent confirmation. The local patch implementation
agreed with full periodic PTM for all 512 tested selected-center labels. This is a
finite verification. PTM uses the retained float16 positions, as do the embeddings;
this assay did not measure label changes relative to original full-precision positions.
See [OVITO's PTM documentation](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html)
for the structural assay, and [the exact metric definitions](../../docs/metrics/forecast_crystallization.md)
for this experiment's targets and calculations.

The readout is a class-balanced linear ridge classifier trained only on observed
training embeddings. Its regularization and all alert thresholds use validation
sources only. On held-out observed embeddings it achieves 98.3% balanced accuracy;
therefore the embedding/readout combination represents instantaneous crystal state
well. Its scores are margins, not calibrated probabilities.

**Recognizing future crystal state is much easier than predicting a transition.**
All-state accuracy includes environments that remain liquid or remain crystal.

| Method | State accuracy +3 ps | +6 ps | +9 ps | Balanced accuracy +9 ps |
|---|---:|---:|---:|---:|
| AR | 97.35% | 96.87% | 96.40% | 96.18% |
| Direct | 97.55% | 97.10% | 96.68% | 96.45% |
| Last embedding | 97.32% | 96.79% | 96.28% | 95.80% |
| History mean | 97.12% | 96.58% | 96.06% | 95.61% |
| True future embedding readout | 98.41% | 98.40% | 98.38% | 98.30% |

The true-future row diagnoses the readout; it is not an available prediction method.
The small state-accuracy gain over persistence would not by itself demonstrate useful
crystallization prediction.

**Advance warnings have measurable skill, with substantial misses and false alarms.**
The following table uses only origins before the first sustained local onset, with
the three most recent physical frames all noncrystalline. A positive event starts
within the specified future interval. Each method/horizon threshold maximizes
validation event F1. Timing statistics condition on correctly warned event windows.

| Method | Horizon | Warning precision | Event recall | F1 | Timing MAE | Timing bias |
|---|---:|---:|---:|---:|---:|---:|
| AR | 3 ps | 37.8% | 45.7% | 0.414 | 0.87 ps | −0.36 ps |
| AR | 6 ps | 45.7% | 52.1% | 0.487 | 1.65 ps | −0.74 ps |
| AR | 9 ps | 49.0% | 51.5% | 0.502 | 2.39 ps | −1.14 ps |
| Direct | 3 ps | 33.2% | 46.6% | 0.387 | 0.83 ps | −0.27 ps |
| Direct | 6 ps | 41.9% | 52.4% | 0.465 | 1.59 ps | −0.42 ps |
| Direct | 9 ps | 48.8% | 50.9% | 0.498 | 2.25 ps | −0.42 ps |

Negative bias means early. AR has the higher event F1 point estimates; direct has
slightly better state accuracy and smaller conditional timing errors. The 9 ps event
F1 source intervals overlap, so this is not evidence of a decisive architecture winner.

At 9 ps, AR has 7,690 correctly warned event windows, 7,240 missed event windows,
8,014 false warnings and 758,556 correctly rejected negative windows. Only 1.91% of
at-risk windows contain onset within 9 ps. Consequently, always predicting no event
would give **98.09% accuracy**, slightly above AR's 98.05%, while detecting zero events.
Accuracy alone is unsuitable here. AR's false-positive rate is 1.05% per negative
window, yet 51.0% of its warnings are false because true imminent events are rare.
These are overlapping windows, not independent alarms or 14,930 separate atoms.

Across correct AR 9 ps warnings, median absolute error is 2.25 ps and the 90th
percentile is 5.25 ps. Of detected event windows, 48.4% are within 1.5 ps. Including
misses, only **24.9% of all positive event windows** are both detected and timed that
closely. Temporal resolution is 0.75 ps; these results do not establish sub-frame timing.

| 9 ps comparator | Event F1 | 95% source interval | Average precision |
|---|---:|---:|---:|
| AR | 0.502 | 0.477–0.525 | 0.447 |
| Direct | 0.498 | 0.463–0.530 | 0.440 |
| Last embedding | 0.196 | 0.172–0.223 | 0.106 |
| History mean | 0.152 | 0.129–0.179 | 0.062 |
| Linear trend | 0.264 | 0.245–0.281 | 0.183 |
| True future embedding readout | 0.844 | 0.807–0.877 | 0.911 |

AR's paired F1 improvement over history mean is +0.350, with a 95% source-bootstrap
interval of +0.317 to +0.381. Its 9 ps precision interval is 44.6–53.6%, recall
49.2–54.0%, and conditional timing MAE 2.29–2.48 ps. Intervals resample 27 whole test
simulations, preserving correlated centers/windows. They condition on the selected
models, readout and validation thresholds, and do not include training-seed uncertainty.

**Fixed-lead predictions expose the difficulty of predicting the actual onset time.**
There is one forecast per eligible local event, beginning exactly the stated time
before onset and ending at onset. These cohorts are event-only and cannot estimate
population precision or false alarms. Each uses the corresponding interval threshold
from the previous analysis.

| Lead / forecast horizon | Eligible events | AR detected | Direct detected | AR detected and within 1.5 ps | AR MAE among detections |
|---|---:|---:|---:|---:|---:|
| 3 ps | 1,213 | 33.8% | 32.3% | 15.2% | 1.46 ps |
| 6 ps | 1,324 | 32.3% | 29.9% | 10.0% | 2.98 ps |
| 9 ps | 1,366 | 25.7% | 25.1% | 4.0% | 4.85 ps |

At 9 ps lead this is 351 detections, 1,015 misses and only 55 events timed within
1.5 ps. Recall's source interval is 21.8–29.8%. Because the true onset is exactly at
the last forecast frame in this assay, detected timing errors can only be early;
4.85 ps is not an unrestricted early/late bias estimate. Thresholded readout and the
persistence rule also limit detection at this boundary: even reading the actual
future embeddings reaches only 55.1% recall in this exact-lead assay. A future study
of timing with room on both sides should place onset inside a longer forecast window.

**The conclusion persists for longer-lived crystal episodes.** Of 1,728 test local
trajectories, 1,448 have a three-frame event and 280 remain censored. Of 1,447 such
onsets with nine-frame follow-up, 1,001 (69.2%) remain crystalline for all nine frames.
Thus the primary event does include short local episodes; it is not irreversible
crystallization. Five-frame and nine-frame rules yield 1,433 and 1,404 event trajectories.
AR 9 ps event F1 is 0.502 / 0.483 / 0.464 for three/five/nine-frame persistence, with
recall 51.5% / 48.9% / 47.7%. Stricter persistence does not turn this into a precise
long-lead event predictor.

At 9 ps, AR event F1 spans 0.466–0.530 across 400–520 K. Recall is 44.7% at 520 K,
versus 51–54% at the other temperatures. Only three test sources and no validation
sources are available at 520 K, so that subset uses thresholds selected at other
temperatures and its comparison has limited precision.

The completed large-training embedding evaluation supports the same interpretation:
AR/direct normalized MSE is 0.209218/0.208638, improving over history mean by
13.25%/13.49%. Their predicted RMS embedding changes are only 67.6%/66.8% of observed
changes. The local examples show smooth forecast margins while the actual environment
changes sharply or flickers. This is consistent with deterministic MSE forecasts
averaging uncertain futures; it does not demonstrate the cause on its own.

The next experiment should explicitly supervise transition risk alongside the latent
trajectory: retain embedding MSE, add per-frame crystal-state classification and a
discrete event-time likelihood with a no-event/right-censored outcome. Select event
checkpoints on validation event skill and calibration, monitor missed-event-aware
timing, and compare at the same false-positive rate. Use train-only transition
sampling with appropriate sampling weights if estimating calibrated event probabilities.
A probabilistic event-time output would describe uncertainty more honestly than a
single crossing of a smoothed mean trajectory. This is a proposed follow-up, not a
training result produced by this analysis.

The [comparison plot](../../output/embedding_forecast/local-crystallization-20260913/plots/prediction-quality.png)
and [three local examples](../../output/embedding_forecast/local-crystallization-20260913/plots/local-trajectories.png)
are standalone figures. Examples are explicitly selected to show the lowest AR timing
error, median detected timing error, and a missed event at 9 ps lead; they are not a
random sample. Exact tables, per-source statistics, thresholds, configs, checkpoint
hashes and paired forecast margins are retained in the
[output directory](../../output/embedding_forecast/local-crystallization-20260913/).
Reproduction commands are in [README.md](README.md). The five focused scientific tests
and six layout/metric-contract checks pass; selected-center periodic PTM verification
passes all 512 comparisons. No simulation or existing training artifact was removed.
