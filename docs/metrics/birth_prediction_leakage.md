# Birth prediction leakage audit and feature reliance

This analysis reuses the frozen preappearance case/control dataset and retained
likelihood-selected rich-descriptor CatBoost and linear models. No model is
refitted or selected, and no input feature is selected for a later fit. The fixed
test covers every removal from zero to seven; readout CV covers removals zero
and seven. The CV descriptor features have no learned pretraining. Neural CV
results discussed alongside them retain their recorded pretraining exposure.

## Observation and replay audits

Inspect the original registered source/melt-lineage roles; verify all matched
sets contain one case and four controls with identical source, event, endpoint,
future-event metadata and row weight. Every input ends before first observed PTM
crystal in the tracked 8-A sphere. Reconstruct every consumed coordinate patch
from its raw tracked atom, exact source timeline and frame; independently inspect
saved full-cell PTM labels over the whole sphere, including atoms outside the
nearest 80. Recompute four evenly spaced descriptor observations per role.
Report exact coordinate and feature duplicates across roles. This does not
certify absence of near duplicates or physical ancestry missing from metadata.

Reversing every metadata array while preserving observation indices must leave
the predictor packet unchanged. Independently reproduce packet construction
from the first retained observation indices only. Frozen CPU model replay must
agree with saved raw and calibrated probabilities to absolute error 2e-6.
This audit cannot remove retrospective sampling: cases are localized using a
future centroid, endpoints use future appearance, and controls use future
survival. These are limitations of the prediction population, distinct from a
label/future-coordinate array accidentally entering a predictor.

## Attribution

The per-frame bank has the producer's actual 442 descriptor names. Packets
concatenate chronological observations, mean, population standard deviation
and last-minus-first. A base descriptor thus appears in n+3 packet fields.
For CatBoost, compute TreeSHAP on held-out rows and verify contribution plus
baseline reproduces each raw logit. Multiply contributions by the fixed
calibration slope. Aggregate absolute contributions across all temporal fields
of each base descriptor, then average with original event-balanced weights.
Linear contribution is `(x - fitting_mean)/fitting_scale * coefficient`, also
multiplied by the fixed calibration slope. Its reference is the fitting mean,
not an independently estimated SHAP background. Attribution `share` divides
each aggregate by the sum of absolute contributions; temporal shares instead
aggregate across base descriptors within each packet field. Raw intercept and
calibration intercept are excluded. At one frame the mean repeats the same
observation, while temporal std/change are zero; duplicate-column attribution
does not imply independent information. No direction or causal mechanism can
be inferred from these absolute values. Larger groups have more opportunities
to accumulate importance.

## Group permutation

Permutation groups are the four complete families and declared physical
subgroups. Jointly replace **every temporal field** of the group's base
descriptors with the same donor row, preserving within-group history and
summary consistency. Use 32 independent seeded permutations. The principal
donors are restricted to each recorded set of one case and four controls,
holding source, endpoint and matching metadata fixed. Supplement with
unrestricted donors within the measured population, and within each outer fold
for CV. Neither operation preserves correlations with unpermuted descriptor
families; these are model-reliance diagnostics, not fully conditional variable
importance or physically realizable trajectories.

For each held-out row, average calibrated binary NLL after replacement minus
its original NLL across permutations. Positive `nll_increase` means the retained
model loses predictive likelihood under that intervention. Normalize original
inverse source/event row weights within the evaluation population. Pool CV
row deltas with their own fitted outer model; never average folds equally.
Report raw base-feature count and n+3-expanded packet-column count.

95% intervals are the 2.5/97.5 percentiles from 2,000 whole-source bootstrap
resamples of these **already averaged, fixed-model** row deltas. They condition
on models, split, calibration and permutation seed; they omit refitting,
training-seed/permutation Monte Carlo uncertainty and multiplicity correction.

## Matched discrimination

Retain event-weighted NLL, Brier, AP and AUROC from the unchanged `fit.scores`
implementation. For each matched set, within-pair AUC is the fraction of four
controls with scores below the case plus half for ties. Top-1 is one if the case
has maximum score, divided by the number tied at that maximum. Average both
with the sum of row weights in that matched set. Constant-score references
are 0.5 and 0.2 respectively. These diagnose discrimination within a common
source and endpoint; they are not prospective event risk, event timing or
source-level nucleation prediction. The deliberately sampled positive fraction
is 0.2, unrelated to its natural prevalence.

Supplementary matched-ranking intervals repeat each pair's AUC/top-1 value over
its five rows, so original row weights recover the matched-set mass. Apply the
same 2,000 whole-source draws to full-history and one-frame values; take percentile
intervals for each score and the paired full-minus-one AUC difference. These also
condition on fitted models and do not refit the CV algorithm. Probability-spread
JSON reports the saved calibration slope/intercept, event-weighted test standard
deviation before/after calibration, and unweighted 0/10/50/90/100 percentiles of
calibrated test probabilities. These are local diagnostics of recorded fits.
