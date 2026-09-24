# Repaired encoder results: what we know

**The optimization repairs worked, and the relaxed-input encoder now shows a
promising local crystallization-ranking signal. The added teacher/distance losses
do not provide a convincing practical advantage.** Average physical forecasting
still changes very little with encoder training. The simplest relaxed-geometry
arm B is the useful candidate to carry forward, with descriptor and untrained
encoder controls retained.

All four fits completed4,096 updates and all evaluations. These are paired
development results from the existing45-source Al cohort:25 fitting,5 tuning and
15 development roots, one encoder/probe seed. No new simulation, training,
checkpoint selection or probability recalibration was performed for this review.

## Local crystallization within12ps

An event is the tracked atom first entering a crystalline PTM1/2/3 state sustained
for three frames. Eligible origins are before that event and have three recent
negative frames. This predicts future local onset, not current crystal-state
recognition or nucleation anywhere in the whole simulation.

The development population contains643 at-risk windows and18 positive windows
from10 sources; the full at-risk population spans15 sources. Equal-source weighting
gives a3.25% event prevalence. Counts below are raw windows, while AP, recall,
precision, false-positive rates and probability errors weight sources equally.

Matched nonlinear hazard predictors, selected only on the5 tuning sources:

| Encoder/control | AP ↑ | Brier ↓ | Detected /18 | Weighted recall | False-positive rate | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: observed geometry | 0.1521 | 0.03325 | 4/18 | 21.5% | 7.59% | 8.69% |
| C: observed + relaxed teacher | 0.1552 | 0.03343 | 4/18 | 21.5% | 7.59% | 8.69% |
| B: relaxed geometry | **0.2675** | **0.02903** | **7/18** | **43.1%** | 6.62% | 17.95% |
| D: relaxed + distance | 0.2668 | 0.02907 | 7/18 | 43.1% | 6.62% | 17.95% |
| Fixed relaxed descriptors | 0.1324 | 0.03084 | 4/18 | 27.8% | 5.44% | 14.67% |
| Temperature only | 0.0586 | 0.03124 | 0/18 | 0% | 0% | undefined |

Thresholds target at most5% false positives on tuning data, not on development
data. The6.62% development rate is therefore reported explicitly. Do not describe
these recalls as measured at an exact5% development false-positive rate.

For B,11/18 event windows are missed. Detected-event timing MAE is3.47ps; only2/18
positive windows are both detected and timed within3ps. D has detected-event
timing MAE3.28ps and3/18 detected within3ps. These tiny counts do not establish a
timing advantage, and the MAE excludes missed events.

At9ps, nonlinear AP is A0.1000, C0.0976, B0.2659 and D0.2673. B detects6/13 event
windows with weighted recall44.5% at6.81% false positives. At3ps, relaxed AP is
about0.707, but there are only3 positives and the false-positive rate is about12%;
that sparse result is not a reliable headline.

## What the uncertainty supports

Paired percentile95% intervals use2,000 whole-source bootstrap draws stratified
by temperature. AP differences below are absolute AP units, not relative percent.

| Nonlinear12ps comparison | AP difference | 95% source interval |
| --- | ---: | ---: |
| B relaxed versus A observed | +0.1155 | [+0.0081, +0.1846] |
| B trained versus untrained relaxed encoder | +0.2351 | [+0.0776, +0.4306] |
| B versus temperature only | +0.2090 | [+0.0565, +0.3971] |
| B versus fixed relaxed descriptors | +0.1352 | [−0.0613, +0.2625] |
| C teacher versus A | +0.0031 | [−0.0007, +0.0104] |
| D distance versus B | −0.0008 | [−0.0028, +0.0008] |

B's AP itself has a wide interval,0.1065–0.4605. Its untrained-encoder nonlinear
control selected the exact constant-prior checkpoint (AP0.0325), which partly
explains the large contrast. The independent linear-readout comparison also
supports a training-associated ranking gain: AP0.1149→0.2087, paired difference
+0.0939 [0.0169,0.1596]. A's nonlinear AP does not improve over its untrained
control,0.1540 versus0.1521 after training.

These intervals condition on one fitted seed and the selected readouts. They do
not include seed or model-selection uncertainty, are post-hoc and unadjusted for
multiple comparisons, and use previously inspected development sources. They
support a candidate for further validation, not a final-test performance claim.
Observed and relaxed inputs also differ through full-cell relaxation; the input
contrast cannot attribute gains solely to removing thermal motion inside8Å.

## Ranking and probability accuracy differ

Higher AP means better event ranking across thresholds. It does not guarantee
well-calibrated probabilities or accurate event-time distributions.

| Predictor |12ps binary log loss ↓ | Joint event-time NLL ↓ |12ps AP ↑ |
| --- | ---: | ---: | ---: |
| B relaxed, linear | **0.13253** | **0.18680** | 0.2087 |
| B relaxed, nonlinear | 0.14398 | 0.21384 | **0.2675** |
| D distance, linear | 0.13240 | 0.18678 | 0.2096 |
| D distance, nonlinear | 0.14450 | 0.21304 | 0.2668 |
| Temperature only, linear | 0.14117 | 0.19372 | 0.0553 |
| Temperature only, nonlinear | 0.14125 | 0.19233 | 0.0586 |

B's nonlinear predictor ranks events better but has worse point estimates of
both log loss and event-time NLL than the temperature-only nonlinear control.
Its Brier score is better. Against temperature, the source intervals for these
probability-error differences span zero for both predictor types. We have not
established calibrated risks. The linear predictor is the more favorable point
estimate for likelihood: it detects6/18 events at5.41% false positives and18.64%
precision. There is no single winner across ranking, likelihood and detection.

The teacher improves the weaker observed linear hazard's NLL by3.77% (source
interval−7.65% to−0.23%) and AP0.0830→0.0970. It remains worse than the untrained
observed linear control on both metrics. Its nonlinear benefit is negligible.
Distance matching leaves both relaxed predictor types essentially unchanged.

## Structural retention and ordinary forecasting

Raw export variation remains99.1–101.3% of initialization across arms, and every
actual training-head tuning block passes the2% retention criterion. On development
data, the saved head's block-balanced geometry MSE improves from0.5572 to0.5245
for A and0.5046 to0.4669 for B; D scores0.4679. The former near-constant decoder
failure is absent.

This does not mean every information-retention measure passes. With fresh matched
nonlinear probes, observed radial MSE still worsens about2.5% for A/C versus their
untrained features, slightly exceeding the2% engineering tolerance. Relaxed radial
MSE improves4.5% for B/D. The earlier status message referred specifically to
the actual training decoder's tuning checks, not every held-out probe.

Future original-MD order8 MSE with matched stronger readouts:

| Exported state |3ps |9ps |12ps |
| --- | ---: | ---: | ---: |
| A observed | 0.56581 | 0.55421 | 0.57472 |
| C observed + teacher | 0.56531 | 0.55438 | 0.57457 |
| B relaxed | 0.53484 | 0.53066 | 0.55839 |
| D relaxed + distance | 0.53517 | 0.53068 | 0.55829 |
| Untrained observed | 0.56436 | 0.55270 | 0.57507 |
| Untrained relaxed | 0.53560 | 0.53095 | 0.55907 |

Training changes these forecast errors by less than0.31% versus untrained exports,
and all corresponding source intervals span zero. C versus A and D versus B
changes are each below0.1%; occasional narrow intervals excluding zero do not
make these practically substantial. Relaxed inputs look better than observed
inputs, but that ordinary-forecast advantage largely exists before training.
The current positive evidence concerns crystallization ranking more than general
future-observable prediction.

## Decision and reproduction

Keep B (plain relaxed-geometry training) as the simplest promising onset candidate.
Keep its linear and nonlinear predictors separate according to whether calibrated
risk or ranking is the priority; neither has established robust calibration here.
Keep fixed relaxed descriptors and untrained encoders as mandatory controls.
There is no evidence here warranting a larger teacher/distance-loss sweep. The
next useful scientific validation would use a larger, independent onset cohort,
with more fitting/tuning events and timing metrics that include misses. No such
run has been launched by this analysis.

The review checked all4 final checkpoint/export hashes, exactly matched event
rows/source IDs, and recomputed all28 hazard readouts'12ps AP, Brier, log loss and
joint NLL. Bootstrap AP was checked against sklearn with tied scores and repeated
source weights; two focused tests passed. Original run tables remain unchanged.

Reproduce from the repository root in conda pointnet-torch214, using a fresh output:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m src.research.structural_state_onset_review \
  --run output/structural_state/repaired-20260923 \
  --output output/structural_state/repaired-review-NEW
```

[Onset intervals](tables/onset_uncertainty.csv) ·
[Paired onset comparisons](tables/paired_onset.csv) ·
[Metric definitions](tables/METRICS.md) · [Audit](technical/audit.json) ·
[Original onset table](../repaired-20260923/tables/onset.csv) ·
[All original physical comparisons](../repaired-20260923/tables/comparisons.csv) ·
[Training-head errors](../repaired-20260923/tables/training_heads.csv) ·
[Scientific protocol](../../../experiments/structural_state_20260923/README.md)
