# Nested cross-validation for appearance prediction

October3 relaxed-cell recovery changes only execution scheduling and the wall-time
allowance. It retains the original force tolerance, potential, minimizer, labels
and folds; new cell receipts record timeout and producer hashes. The running CV
uses its original frozen fitting/metric code and consumes the verified recovered
input release. Historical exports remain unchanged.

This is a separate exploratory evaluation of the eligible established/failed
appearance cohort, explicitly requested October 2, 2026. It does not overwrite
the previous train/test protocol or fixed-Al64 release. No encoder is fitted.
All original source roles participate in outer CV; previously inspected sources
are not presented as new confirmation data.

## Cohort, observations and folds

Reuse the exact eligibility, labels, 442 descriptors, event weights and matched
liquid controls of `birth_appearance.md`. The retained union is 3,670 rows from
119 sources/ancestors: 95 established events, 18 stronger failed events and 115
other failed events. Stronger failed events are included once, within the 133
eligible failed events. The 28 detected stronger events are a subset of the 184
detected candidates; exclusions remain excluded. No newly searched shooting or
Ta events enter this comparison. Four controls per positive give prevalence 0.20.

Five outer folds group by the recorded preparation ancestor, keeping every source,
event, center, history and matched control together. A deterministic seeded greedy
assignment balances event counts in the three strata, counts of ancestors with
stronger events, rows and total ancestors. No descriptors, temperatures, timestamps
or observed scores enter fold assignment. Five inner folds are independently
constructed within each outer training population using the same grouping and
seed. All models/pools/domains use those exact assignments. Fold support is
exported before fitting; every outer fold must contain stronger failed events.

Each original-pool model is **refitted** on established-event matches from the
outer training ancestors. It cannot reuse the old globally fitted model, which
has seen some outer-held-out sources. Combined models add failed-event matches
from the same outer training ancestors. Both predict exactly the same outer
holdout, including failed events. No pool-specific evaluation exclusion.

## Selection and inputs

Use five inner folds to select logistic C from [0.001,0.01,0.1,1,10] or a common
CatBoost tree count from 1–1,000 by weighted binary NLL, using the existing
`merged_fit.fit`. CatBoost retains depth 4, learning rate 0.04, L2 10 and the
original GPU recipe. Use only the single retained seed. Standardization is
fitted separately on each inner training subset, then on all outer training rows
for the final refit. No outer observations select hyperparameters, scaling,
calibration, early stopping or alarm thresholds. AP is a diagnostic only.

The alarm threshold is the weighted 95th percentile of the selected inner-OOF
negative probabilities; outer alarms use strict greater-than and the threshold
of the model that predicted that row. Threshold selection is training-only.
Inner selected scores are optimistic selection diagnostics, not outer results.

Geometry only: 99 geometry,40 bond-order,45 CNA,258 TDA descriptors from the
original nearest80 atom identities within8Å. Snapshot uses the final crystal-free
frame. History uses eight observations spaced exactly0.75ps over5.25ps,
chronological descriptors plus mean, standard deviation and last-minus-first.
Inputs end0.75ps before the local PTM appearance label. No temperature, simulation
age, explicit time covariates, velocities, species, learned encoder, training
teacher or future input geometry. Original and full-cell-relaxed domains share
row identities and folds. Relaxation is the existing fixed-box generating-MEAM
recipe, not new relaxation work launched by CV. Exact feature arrays are hashed
and bound into each fit context. Historical source roles remain in provenance.

## Scores and uncertainty

The collector requires exactly one outer-held-out probability per retained row
for each treatment. It reports established, stronger failed, other failed, all
failed and combined strata, each including its liquid controls. NLL, Brier,
AP and AUROC reuse `fit.scores` and event weights. Matched AUC (one positive versus
its four controls, half credit for ties) and probability gap reuse
`temporal_analysis.metric_bundle`. Scores are calculated from pooled outer
predictions, not averages of fold AP. Report outer-fold train/test errors
separately; mean_outer_train_nll/ap is the arithmetic mean of five resubstitution
fold scores for that treatment, not a stratum-specific or held-out error.

Resample whole ancestors 2,000 times with paired draws across models to give
percentile95% intervals and paired differences. The current119 sources have119
distinct recorded ancestors. Intervals are conditional on the fitted outer models:
they describe observed population variation, do not rerun fitting, and do not
cover training-seed uncertainty or all dependence caused by overlapping CV
training sets. Five overlapping fold scores are not five independent experiments.
CV increases evaluated stronger events from six to18, but cannot create more
independent events or restore the excluded candidates.

Ten fixed bins report event-weighted predicted probabilities and positive
fractions. Calibration concerns the enriched case/control population, not natural
nucleation risk. Combined-minus-original and history-minus-snapshot contrasts
report NLL/Brier/AP/AUROC/matched-AUC differences with paired ancestor intervals.

The shared fitter's per-fit roles `train`, `training_selection_cv`, `merged_test`
map respectively to outer training, inner selection and outer held-out evaluation.
`outer-split.json` records that mapping and the original historical roles. Final
`fold-train-and-test.csv` uses the explicit mapped names. Raw per-fit `scores.csv`
does not mean the earlier fixed merged test was reused.

All descriptor readouts remain local diagnostic fits without new W&B runs.
Outputs preserve fold assignments, input hashes, selected hyperparameters, models,
probabilities, source/ancestor IDs and exact implementation contracts.
