# Fixed-endpoint history and persistent-site controls (v1)

Producers: `birth_prediction/temporal.py`, `temporal_inputs.py`,
`temporal_analysis.py`, and the shared retained-data/readout/scoring producers.
This is a separate follow-up; historical predictions and definitions are not replaced.

## Population, labels and fitting

Reuse all 1,475 eight-frame histories at exact 0.75 ps cadence from the original
preappearance release. Preserve source roles, all rows, inverse-event weights,
1:4 source/time-matched cases/controls, and the previously frozen five readout-CV
folds. Fixed test has 15 births in 11 sources; CV evaluates the original training
population (59 births in 49 sources). Neural CV is conditional on encoders whose
pretraining saw these sources. All test-source results are exploratory follow-up
on an already inspected population, not new confirmatory evidence.

Inputs contain geometry-only rich descriptors or frozen geometry-only MACE
states. No labels, IDs, timestamps, age, temperature, velocities, species or
future frames enter the numerical columns. Rows retain their own tracked atom.
The retrospective positive-site and persistent-liquid-control selection remains;
the experiment cannot identify blind prospective incidence. Here time zero is
the first observed local PTM-crystalline atom, not critical size or establishment.

Six arms: rich descriptors, rich/TDA MACE, and VICReg MACE, each with linear and
CatBoost readouts. Frozen encoder checkpoints and preprocessing are unchanged.
There is one fit seed. Reuse train-only linear standardization, the declared C
grid, depth-4 boosting with up to 1,000 trees, validation Logloss selection,
and separate calibration sources from `birth_prediction.md`. No AP-based choice.
The separate constant-prior fit provides the original event-enriched 20% baseline.
These are local frozen probes/descriptor controls, not W&B encoder training.

## Input treatments

Frame indices 0..7 denote −6, −5.25, −4.5, −3.75, −3, −2.25, −1.5, −0.75 ps.
Every fit records its exact observed frames, transformations and input contracts.
The history packet concatenates chronological states, mean, population standard
deviation and last-minus-first change. Temporal summaries are recomputed after
any intervention. The following independently fitted treatments are predeclared:

| Name | Observed frames | Numerical input |
| --- | --- | --- |
| current | 7 | One endpoint descriptor/state vector |
| repeated_current | 7 | Endpoint repeated into eight history slots, with summaries |
| history8 | 0..7 | Eight real frames and summaries |
| shuffled_past | 0..7 | Permute slots 0..6; keep endpoint 7 fixed; recompute summaries |
| history2 | 6,7 | Two real frames and summaries |
| history4 | 4..7 | Four real frames and summaries |
| early4 | 0..3 | Four real frames at an earlier endpoint |
| middle4 | 2..5 | Four real frames at an intermediate endpoint |
| early | 0 | One early descriptor/state vector |
| mean8 | 0..7 | Order-independent mean vector |
| changes8 | 0..7 | Subtract each site's frame-0 vector from every slot, then history packet |
| early_plus_change | 0,7 | Concatenate frame 0 and frame 7 minus frame 0 |
| early_capacity_control | 0 | Concatenate frame 0 and an equally sized zero vector |

Shuffled-past permutations are deterministic from the one declared seed and
matched-set identifier, with the same permutation for all five candidates in
that set. IDs select a random intervention only and are not numerical features.
Shuffling is fitted on both training and held-out inputs, not merely applied
after training. The endpoint stays fixed, so current-frame availability is matched.
Repeated/history/shuffled packets have identical nominal widths and model
families; redundant slots do not imply equal effective statistical complexity.
History length comparisons and fixed-four-frame lead comparisons are separate.

`changes8` removes a constant additive site offset in feature space. Nonlinear
site-dependent fluctuations can remain. It therefore does not by itself eliminate
all persistent site information or prove a mechanism. Likewise, mean8 may denoise
time-varying structure; it is not a pure static-state oracle.

## Predictive scores and uncertainty

Binary NLL, Brier, weighted AP and AUROC use the shared definitions in
`birth_prediction_extension.md`. Both raw and separately calibrated predictions
are reported. The per-fit `remove_frames=0` is a shared fitter bookkeeping field;
the explicit observation name/frames determine this protocol's input and lead.

Matched AUC is the case's mean pairwise win rate against its four controls,
giving ties half credit, averaged with original event weights. Matched probability
gap is case probability minus mean control probability, with the same weights.
Both summaries are constant within each matched set before event-weighted averaging.
They remove comparisons across different sources/observation times.

Intervals use 2,000 whole-source bootstrap draws shared across treatments in each
population. Paired contrasts subtract scores on the same rows in the same draw.
NLL/Brier differences below zero favor the listed treatment; AP/AUROC/matched-AUC
differences above zero favor it. Every source contains full matched sets, so these
source draws retain both classes. Primary comparisons are declared in the recipe,
alongside every treatment versus the constant prior. No multiple-comparison
adjustment or fitting-seed uncertainty is claimed. OOF bootstrap conditions on
the five overlapping fitted readouts and fixed selection/calibration populations.

## Same-model replay across time

Freeze each early-snapshot and current-snapshot readout, its train standardization,
and its calibration. Evaluate it at all eight frames of the same held-out sites.
Replay must reproduce its original endpoint predictions within rtol 1e-6/atol 1e-7.
For CV, each site's model remains its own held-out-fold model across all frames.
Save raw/calibrated arrays and predictor hashes. No refitting or recalibration
uses the replayed endpoints. Export NLL, AP, AUROC, matched AUC and probability
gap versus physical lead, and paired late-minus-early differences. The probability
gap difference is the case-minus-control contrast of within-site score changes.
Model distribution shift under earlier/later inputs remains an interpretation
limit; replay alone cannot prove a causal precursor.

## Structural site persistence and within-site change

Standardize each descriptor/embedding coordinate using original training rows
over all eight frames with event weights. Scaling is for descriptive effect sizes,
not an input to an independently cross-validated readout. Report original training
and fixed-test populations separately. Featurewise case-control gaps are the
case value minus the mean of its four matched controls, averaged with case weights.
Early gap uses frame 0; late gap uses frame 7. Change gap is late minus early,
equivalently a difference of within-site changes between cases and controls.
Intervals use source-resampled complete matched sets. They are exploratory
featurewise intervals without multiplicity correction.

Between-site variation fraction is
`weighted_variance(site_mean_over_8_frames) / (between_variance + mean_within_site_variance)`.
The within-site variance uses population variance across the eight frames; the
between variance is the weighted population variance across sites. Constant
coordinates yield an undefined (blank) fraction. This is a descriptive variance
decomposition, not a noise-corrected ICC or proof of persistent nucleation propensity.
The observation span is only 5.25 ps. These data cannot establish longer-lived
site differences. Every exported CSV freezes these definitions and source hashes.
