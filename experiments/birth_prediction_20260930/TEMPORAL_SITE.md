# Does history help at a fixed endpoint, or do selected sites remain distinguishable?

Requested 30 September 2026 after reconsidering the weak one-frame fixed-test
evidence. Preserve the previous truncation result; this follow-up changes the
comparison, not its historical interpretation.

Use the existing preappearance cohort and frozen rich/TDA and VICReg encoders,
plus rich TDA/bond-order/CNA/geometry descriptor controls. Fit linear and boosted
readouts using predictive likelihood selection and separate calibration. Keep
all original source roles and the already frozen five readout-CV folds, one seed,
and the same rows for every treatment. Neural CV is pretraining-exposed; independent
encoder transfer is evaluated on the original fixed test. This test has already
been inspected, so the follow-up is exploratory.

## Comparisons

1. At the same −0.75 ps endpoint, compare snapshot, repeated current frames,
   real 2/4/8-frame histories, and separately fitted shuffled-past histories.
   Shuffling keeps the current frame fixed and recomputes summaries.
2. Keep four real frames and shift the endpoint to −3.75/−2.25/−0.75 ps.
   This separates lead from history length without collecting new observations.
3. Compare early snapshot, current snapshot, history mean, baseline-subtracted
   changes, and early structure plus subsequent change. Include a dimension-matched
   early/zero-change control. Real/repeated/shuffled eight-slot packets have the
   same nominal input width; this does not make their effective complexity equal.
4. Freeze early/current snapshot classifiers and replay every held-out site's
   full timeline with the same predictor and calibration. Measure the rise in
   within-source/time case-control separation, not only global AP.
5. Decompose observed structural variation into between-site and within-site
   components. Estimate early/late matched structural gaps and their paired change,
   with whole-source uncertainty. A within-site difference removes an additive
   site offset, not every possible site-dependent dynamical property.

Interpret results together. Useful early separation with little added change
information is consistent with persistent propensity. Rising same-model matched
separation and useful change information support evolving precursors in this
population. Weak predictors and wide intervals establish neither mechanism.
No AP-based model promotion or significance filtering is allowed.

The experiment remains retrospectively localized to future sites, with
persistent-liquid controls. It cannot estimate natural nucleation incidence or
establish blind prospective localization. Its maximum observation span is 5.25 ps.
PTM appearance is distinct from sustained growth, critical size, or the 64-atom
establishment threshold. No new simulations or encoder fitting are included.

There are 13 input treatments × six readouts, plus a constant prior, in each of
the fixed-test and five CV partitions: **474 small fits**. Plotting and same-model
replay do not retrain models. Bootstrap intervals condition on these fitted models.

[Recipe](../../configs/birth_prediction/temporal_site_20260930.json) ·
[Exact definitions](../../docs/metrics/birth_prediction_temporal.md) ·
[Execution](../../docs/birth_prediction.md#fixed-endpoint-and-site-persistence-comparison).
