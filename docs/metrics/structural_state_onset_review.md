# Frozen repaired-encoder onset review

This analysis reads completed structural-state v2 predictions; it never refits,
recalibrates, selects checkpoints, or changes the frozen training-run tables.
The target is first sustained local crystallization within12ps, in the original
643 at-risk development windows from15 independent sources. There are18 positive
windows; their source-weighted prevalence is3.2479%. Independent trajectories,
not windows or atom identities, are the bootstrap units.

Each source has equal total weight and its windows divide that weight equally.
AP is sklearn-compatible noninterpolated aggregate average precision, including
complete score ties; it is NOT an average of within-source AP values. Brier is
weighted squared probability error. Binary log loss clips12ps probability to
[1e-7,1-1e-7]. Joint NLL is the sum of survived-bin softplus(logit) terms and the
event-bin softplus(-logit), with no-event examples surviving all five bins ending
at0.75/3/6/9/12ps. Thus binary12ps log loss and joint event-time NLL are distinct.

Use2,000 paired whole-source bootstrap draws, seed20260922, resampling sources
within their original temperature strata. A root drawn m times assigns m/n_root
weight to each of its windows, with total weight normalized across the15 sampled
roots. Recompute AP over the complete weighted cohort in every draw. Zero-positive
AP is undefined and excluded; valid_draws reports the count. All12ps draws here
contain positives. Percentile2.5/97.5 bounds provide95% source intervals.

onset_uncertainty.csv reports frozen-predictor point estimates and intervals.
paired_onset.csv reports candidate minus reference in ABSOLUTE metric units;
positive AP differences are favorable, negative error differences are favorable.
Multiply an AP difference by100 to express percentage points, not percent change.
All comparisons use exactly matched indices, source IDs and event bins.

These are post-hoc exploratory intervals on reused development data, unadjusted
for multiple comparisons. They condition on the one encoder/probe seed and the
already selected tuning checkpoints/thresholds; they exclude training-seed and
model-selection uncertainty. Predictive probability calibration is not established
by a favorable AP, Brier, or one significant interval alone.
