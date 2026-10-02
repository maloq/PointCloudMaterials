# Descriptor-family information and model reliance (v1)

This is an exploratory follow-up of the frozen birth-site population. It uses
one snapshot at frame 7 of each existing eight-frame history: 0.75 ps before
first local PTM crystal appearance. Labels still require eventual isolated
establishment. It is not a prospective 3/6 ps incidence assay. All 1,475 rows,
event weights, 1:4 source/time matches, original source roles and five training
source folds are preserved. Original and relaxed rows must agree in every field.
There are 15 births in 11 fixed-test sources and 59 births in 49 CV sources.

## Inputs, fitting and reuse

The four families are actual producer prefixes: geometry, bond_order, cna, tda.
Each of four family-only fits and four leave-one-family-out fits uses the retained
linear or CatBoost fitter. These are 192 new fits: 8 subsets × 2 readouts ×
2 input domains × (one fixed test + five folds). The 24 existing full-feature
snapshot models are referenced without refitting. Their observation, source
partitions, feature producer, fitter, dataset and fitting settings are checked.
Saved predictions are checksum-checked and models are replayed before permutation.

Features are geometry-only descriptors of stored centered 80-atom patches.
The unchanged producer sorts by current radius, clips at 8 Å, and constructs
its own nearest-neighbor subsets. Relaxed inputs preserve stored atom identities
but not necessarily effective descriptor membership. Relaxation minimized each
full periodic current cell with its generating potential and fixed box; its
computational context is broader than the exported patch. Original-MD labels
and eligibility are unchanged. No encoder, velocities, temperature, time,
source identity, species channel, future coordinates or condition covariates
enter the readout. Source/time/pair metadata only establish splits and matches.

Training-only weighted standardization and the original logistic C grid are
retained. CatBoost retains depth 4, up to 1,000 trees and validation-Logloss early
stopping. Selection uses original selection sources and binary NLL. Calibration
uses the separate calibration sources, with the original monotone logit map and
locked false-alarm threshold. One original fitting seed is retained. Descriptor
diagnostics are local, with no encoder training or new W&B runs.

## Scores and contrasts

NLL = weighted mean of `-y log(p) -(1-y) log(1-p)`; Brier = weighted mean
of `(p-y)^2`. Probabilities are clipped to [1e-7,1-1e-7] for scoring. AP and
AUROC use weighted tied-threshold integration; neither selects models. Both
raw and separately calibrated predictions are reported. Prior prevalence is
20% in this event-enriched population, not natural nucleation incidence.

The train-selection-calibration-test table contains resubstitution train error,
selection and calibration errors, and held-out scores for every individual fit.
CV held-out scores pool exactly one held-out-fold prediction for each original
training row; original test sources are not part of CV. Training errors are not
generalization estimates. Calibrated train scores apply a calibration map fitted
on the separate calibration population.

`paired-vs-full` subtracts the same-domain/readout full-feature score from each
refitted subset score. Positive NLL/Brier differences mean worse prediction;
positive AP/AUROC differences mean better ranking. Family-only comparisons
measure the information accessible in that family; leave-one-out comparisons
measure the value remaining after the other families can compensate through
refitting. These are finite-model comparisons, not mutual information estimates.

`relaxed-minus-original` compares the same subset/readout/scope across the paired
input domains, with unchanged rows. Negative NLL/Brier differences favor
relaxation. Matched AUC averages the positive's four pairwise wins (half credit
for ties); matched probability gap subtracts the four controls' mean probability.
Both use original event weights, as in birth_prediction_temporal.md.

Every interval uses 2,000 whole-source draws, with shared draws across compared
subsets and domains. All windows and complete matched sets from a source remain
together. CV intervals condition on the five overlapping fitted models and the
fixed selection/calibration sets. They do not include training-seed uncertainty.

## Matched permutation reliance

Only the full-feature frozen models are permuted. For each of 32 repetitions,
jointly exchange all columns of a group across the five members of each original
case/control match. Donor permutations are shared across models, domains and
groups at the same fold. The four primary families and seven predeclared
secondary groups (l=6 bond order; H0/H1/H2 at 32/80 points) are evaluated.
No refitting or recalibration follows shuffling. Metadata never becomes a feature.

The per-row mean over repetitions of `permuted NLL - original NLL` is averaged
with event weights. Source-bootstrap intervals resample these per-row mean
effects and condition on the saved permutations; permutation Monte Carlo
uncertainty is not included. Per-repeat loss changes and donors are saved.
Positive changes indicate fitted-model reliance. Shuffling preserves source/time
matches but breaks correlations with other descriptor groups: this is not fully
conditional importance and is not a physical or causal intervention. Subgroup
effects overlap, are not additive, and receive no multiplicity correction.

The fixed test has already been inspected. All new comparisons and explanations
are exploratory; no test-driven feature or model promotion is performed.
