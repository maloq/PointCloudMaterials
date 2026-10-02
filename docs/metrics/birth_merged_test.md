# Birth readouts with merged evaluation and training-only selection (v1)

On 2026-10-02 the user explicitly requested merging original selection,
calibration and test into one test and rerunning the birth comparisons. This
version changes readout roles only; parent caches, labels, weights, source roles
and historical reports are never rewritten. Original training remains 965
histories (193 positive), 59 births in 49 observed sources. The merged evaluation
has 510 histories (102 positive), 36 births in 28 observed sources. Assigned
trajectories remain 90 training and 60 evaluation, including zero-yield sources.
Sources and melt ancestors cannot cross the two new roles. Cases/controls remain
1:4 matched and event-weighted. The event is first local PTM appearance at a
site retrospectively associated with a later isolated established cluster.
This is event-enriched site discrimination, not prospective incidence.

## Selection and final fitting

Every readout is refit; old models selected/calibrated using former validation
or calibration are not used for merged-test prediction. The five existing source
folds are reused entirely inside original training. Selection and fitting never
receive former selection/calibration/test labels. Original fitting seed and
model families are retained; AP does not affect fitting or selection.

For linear readouts, fit the original C grid in each training fold, with weighted
standardization fitted only on that fold's fitting sources. Select C by pooled
event-weighted held-out-training NLL across five folds. Refit standardization and
logistic regression on all 965 training rows at selected C. CatBoost fits the
original depth-4/Bayesian-bootstrap/GPU recipe to all 1,000 trees in each training
fold. Average per-iteration validation Logloss by held-out-fold event mass, select
one shared tree count, then refit on all original training at that count with
no evaluation set and no early stopping. All other boosting settings are fixed.

No probability-calibration map is fitted. Raw probabilities are used throughout.
The optional decision threshold is the weighted 95th percentile among original
training negatives' out-of-fold probabilities at selected settings; predictions
strictly above it count as alarms. It does not use the merged evaluation set.
Training-CV scores are hyperparameter-selection diagnostics, not unbiased nested
cross-validation performance: those same folds selected C or tree count. Train
resubstitution errors and merged-test errors are reported separately.

## Compared inputs

The full recipe covers 162 distinct numerical treatments per input domain:
18 current-snapshot family comparisons; 76 additional retained temporal controls;
68 additional original truncation/descriptor controls. Exact duplicate inputs
are fitted once. Original and full-cell-relaxed observations yield 324 final
readouts, each with five-fold internal selection and a full-training refit.
The input catalog is frozen in technical/prepared.json. Family-only coverage is
a separate explicit recipe setting. The constant training prior is also scored.

Use the original geometry descriptors and historical frozen rich/TDA and VICReg
MACE features. No new encoder pretraining or simulation is performed. Snapshot,
history, mean, repeated-anchor, shuffled-history, change-only and truncated inputs
use the retained packet producer and exact 0.75 ps source cadence. No temperatures,
ages, time covariates, labels, IDs, velocities or species channels enter numerical
predictor inputs. Geometry uses the original centered 80-atom patch preprocessing;
descriptors re-sort by radius and clip at 8 Å. Relaxation supplies full-current-cell
computational context and keeps original labels. Each fit records encoder and
predictor inputs, checkpoint, relaxation, feature names, physical offsets and
absence of conditions separately.

Historical rich-MACE checkpoint selection used structural validation on former
selection sources. Its merged evaluation is therefore not wholly unseen by
encoder selection. VICReg uses the retained predeclared epoch24. Both encoders
saw original training ancestors, including internal readout-CV folds. All these
sources have already been inspected; refitting readouts does not create a fresh
confirmatory holdout or erase historical study/representation selection.

## Metrics and intervals

NLL, Brier, weighted AP/AUROC, matched AUC and matched probability gap retain the
birth_prediction_temporal definitions. NLL clips probabilities to [1e-7,1-1e-7].
The prior has 20% prevalence, NLL about 0.50040 and Brier 0.16. AP remains a
diagnostic. Reliability tables show weighted mean prediction and event fraction
in fixed bins [0,.1),...,[.9,1]. No calibration fitting uses these bins.

Held-out tables contain pooled merged evaluation and separate former-role strata
so changes in the evaluation population remain visible. Intervals use 2,000
whole-source bootstrap draws, retaining all matched sets and event weights.
All paired comparisons share source draws and row IDs. Differences subtract the
named reference: negative NLL/Brier favors the listed treatment; positive
AP/AUROC/matched-AUC favors it. Intervals condition on the fitted models and
chosen hyperparameters; there is no training-seed uncertainty or multiplicity
correction. Models are not selected or promoted by these evaluation scores.

Full current-snapshot descriptor models also undergo 32 joint-column permutations
within each source/time-matched case/control set. Primary groups are geometry,
bond order, CNA and TDA; secondary groups are l=6 and H0/H1/H2 at 32/80 points.
Donors are shared across models and input domains. The reported effect averages
per-row `permuted NLL - unmodified NLL` over repetitions, then applies event
weights. Source intervals condition on these permutations and omit Monte Carlo
uncertainty. These are correlated-feature model-reliance diagnostics, not causal
effects or fully conditional importance. Save donors and per-repeat loss changes.
