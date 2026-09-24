# Structural-state screen: results, 23 September 2026

All four encoder fits and their evaluations completed. **This screen does not
establish a better predictive encoder.** The relaxed teacher gives a small,
isolated improvement. Physical-distance supervision improves some structural
readouts, but worsens exported-state forecasts and leaves the original training
decoder nearly constant. The latter is an optimization problem to resolve before
interpreting this arm as a successful representation.

This report covers the four jobs submitted after the encoder-training literature
review, not the older BCR or history-model cohorts. No additional fits were run
for this analysis.

## What was compared

| Arm | Input | Training objective |
| --- | --- | --- |
| A | Observed positions | Predict fixed observed geometric descriptors |
| B | Relaxed positions | Predict fixed relaxed geometric descriptors |
| C | Observed positions | A plus prediction of fixed relaxed geometry |
| D | Relaxed positions | B plus matching embedding distances to physical-descriptor distances |

These are snapshot structural encoders; neither velocities nor histories enter
this screen. Future targets and onset labels are used only by subsequently fitted
evaluation predictors. All encoders start from identical weights: native MACE,
32 channels, 128 exported components, one seed, 2,048 updates. A/B ran on A100;
C/D on RTX PRO 6000, with cuEquivariance and float32/TF32 disabled throughout.
They use the same batch size, optimizer schedule and sampling stream.

The existing paired dataset contains 45 independent Al trajectories: 25 fitting,
five tuning and 15 development sources, with 64 observations per source. The
development sources were inspected in earlier work, so these are **development
results, not a fresh final test**. Patches include all atoms inside 8 Å.

## Physical forecasting

The table reports source-averaged, standardized MSE for eight original-MD physical
observables. Lower is better. Every row uses the same target, source split and
temperature conditions. The readout is ridge plus a tuning-selected nonlinear
residual, which can retain the ridge solution at step zero.

| Representation | 3 ps | 9 ps | 12 ps |
| --- | ---: | ---: | ---: |
| A: observed geometry | 0.56652 | 0.55575 | 0.57467 |
| C: observed + relaxed teacher | 0.56793 | 0.55592 | 0.57260 |
| B: relaxed geometry | 0.54400 | 0.53767 | 0.56319 |
| D: relaxed + physical distance | 0.54879 | 0.54377 | 0.57044 |
| Untrained observed encoder | 0.56713 | 0.55298 | 0.57163 |
| Untrained relaxed encoder | 0.53480 | 0.53803 | 0.55243 |
| Fixed observed descriptors | 0.57545 | 0.56694 | 0.59221 |
| Fixed relaxed descriptors | 0.52753 | 0.53716 | 0.55912 |
| Current-observable persistence | 0.83189 | 0.88203 | 0.89559 |

An untrained encoder here is a frozen random MACE with a trained evaluation
readout, not a model making predictions without any fitting.

The predeclared objective comparisons are:

| Change in exported-state prediction MSE | 3 ps | 9 ps | 12 ps |
| --- | ---: | ---: | ---: |
| C versus A | +0.25% | +0.03% | **−0.36%** |
| Paired 95% source interval | [−0.10%, +0.63%] | [−0.54%, +0.63%] | [−0.54%, −0.22%] |
| D versus B | **+0.88%** | **+1.14%** | **+1.29%** |
| Paired 95% source interval | [+0.31%, +1.49%] | [+0.56%, +1.64%] | [+0.39%, +2.15%] |

C's 12-ps gain occurs in 12/15 development sources, but C still does not beat the
untrained observed encoder's point estimate. D is worse than B at all three
forecast horizons. Its improved structure readouts do not translate into improved
exported-state forecasting.

Readout choice matters. With ridge alone, B improves over its untrained encoder
at all three horizons: 0.54395 versus 0.54718 at 3 ps, 0.53771 versus 0.54214 at
9 ps, and 0.55985 versus 0.56496 at 12 ps. This advantage does not persist with the
stronger readout above. D's intermediate pooled64 features also differ from its
exported128 state: pooled 9-ps MSE is 0.53085 versus B's 0.53354, while pooled
12-ps MSE worsens to 0.56530 versus 0.55754. There is no consistent gain across
readouts, representations and horizons.

The noncrystalline PTM-Other subgroup has 686 observations from all 15 sources;
it also includes interfaces and defects. At 12 ps, exported-state MSE is A
0.71047, C 0.70853, B 0.70165 and D 0.71155. The corresponding untrained observed
and relaxed encoders score 0.70765 and 0.68574. Thus the lack of a clear training
benefit is not solely an artifact of averaging crystalline and other observations.

All these models beat physical persistence. That is evidence of predictability
available to fitted readouts, but the strong random-encoder and fixed-descriptor
controls prevent attributing that benefit to the new encoder objectives.

## Structure retention and the distance-arm failure

C changes most structural errors by about one percent or less relative to A:
current-order MSE improves 1.01%, withheld relaxed l6 improves 0.72%, while
observed angular MSE worsens 1.02%. These are small effects.

D looks more encouraging if only standardized frozen probes are considered.
Against B, relaxed radial MSE falls **42.8%**, withheld angular MSE **4.69%**, and
withheld l6 MSE **6.64%**. Native-space nearest-neighbor discrepancies also improve
4.04% for relaxed angular structure and 4.51% for relaxed l6. These latter checks
use raw Euclidean embedding distances, not channel whitening.

However, the saved checkpoint audit reveals a severe scale problem:

| Diagnostic | B | D |
| --- | ---: | ---: |
| Raw exported feature RMS standard deviation on fitting data | 0.09538 | 0.00005181 |
| Final / initial feature spread | 78.98× | **0.0429×** |
| Saved training-head development geometry MSE | 0.61961 | **0.99532** |
| Constant fitting-mean prediction on the same target | 0.99405 | 0.99405 |

D's exported variation shrinks by about 23 times from initialization. Its capped
training head does no better than a constant prediction. Fresh evaluation probes
standardize the features and can amplify the remaining small differences, so
their structural gains do **not** imply successful reconstruction by the decoder
actually trained with the encoder. The embedding is not exactly constant, and
some neighborhood information survives; this is an amplitude/optimization
failure, not proof that all information disappeared. The detached distance-scale
normalization and its interaction with the bounded physical head need diagnosis
before using this objective again.

There is also a retention problem in the plain objectives: compared with their
own untrained exported features, A's observed radial MSE rises from 0.07303 to
0.11242 (**+53.9%**), and B's relaxed radial MSE from 0.04665 to 0.07268
(**+55.8%**). Both deteriorate on all 15 sources. The corresponding pooled-feature
increases are smaller, +9.5% and +17.5%, indicating that the final export makes
the accessibility problem worse. These results are inconsistent with the proposed
2% retention tolerance. They establish worse decoding under the tested probes,
not irreversible information loss under every possible decoder.

## Sustained crystallization onset

Only **18 positive 12-ps event windows among 643 at-risk development windows**
are available. Source-weighted prevalence is 3.25%. Windows are not independent
event realizations.

| Exported state, linear hazard | Joint event NLL ↓ | 12-ps average precision ↑ | Detected / positive windows |
| --- | ---: | ---: | ---: |
| A | 0.20008 | 0.0723 | 1/18 |
| B | 0.22201 | 0.0579 | 1/18 |
| C | 0.20034 | 0.0750 | 1/18 |
| D | 0.19568 | 0.0257 | 0/18 |
| Temperature-only control | 0.19373 | 0.0553 | 0/18 |

Detection uses thresholds selected for at most 5% false positives on tuning
sources. Actual development false-positive rates for A/B/C/D are
5.02%/4.76%/4.87%/5.61%. There is no separate calibration cohort.

For **all four exported-state MLP hazards**, tuning selects **step zero**: the
near-prior initialization beats subsequent training. D's linear hazard also
selects step zero. Therefore D's lower NLL than B is not evidence of improved
onset information; it largely avoids B's worse fitted predictor. The tiny MLP
risk differences and their differing AP values should not be interpreted as
learned precursor skill.

The fixed relaxed-descriptor MLP has higher AP, 0.1605, and detects 4/18 event
windows, but its joint NLL is 0.22702, worse than the temperature-only MLP's
0.19234. Ranking and likelihood disagree, and the event sample is too small to
declare an onset winner. Earlier studies used different cohorts and protocols;
their AP values are not direct comparators for this table.

## Decision and verification

Do not scale up any of these objectives on the strength of this screen. Retain
the relaxed descriptors and random-encoder readouts as controls. Resolve D's
scale/training-head failure and the plain objectives' retention failure before
treating a larger run as a test of predictive representation learning. C remains
a modest structural-teacher ablation, not a demonstrated forecasting advance.

The audit verified all four final checkpoint/export hashes, finite features,
identical initial state dictionaries, disjoint source roles, all **440** saved
regression readouts' per-source and aggregate errors, and the **20** primary
regression comparisons and bootstrap intervals. Initial exported features differ
by at most 3.0e-8 across the GPU families; this was not a repeated-seed or repeated-
hardware experiment. The 2,000-draw intervals resample whole sources within
temperature, condition on one fitted seed, and are unadjusted for multiple
exploratory comparisons. They do not measure seed uncertainty.

Evidence: [physical readouts](tables/physical.csv),
[neighbors](tables/neighbors.csv), [onset](tables/onset.csv),
[paired comparisons](tables/comparisons.csv),
[frozen metric definitions](tables/METRICS.md),
[checkpoint audit](technical/review-20260923.json),
[audit calculation](technical/review_20260923.py),
[protocol](../../../experiments/structural_state_20260922/README.md).
