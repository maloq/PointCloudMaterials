# Completed BCR follow-up: paired analysis

This analysis reads completed predictions and does not fit models. It verifies all
60 probe-pair completion identities, exact paired targets/indices/roots/population
masks, the selected tuning error bound, and reported RMSE against raw predictions.
The three fresh decoders must each have 10,000 updates and an unchanged encoder.
Input SHA-256 hashes and the originating run identity are in `technical/inputs.json`.

`paired_comparisons.csv` reports standardized target RMSE for the reference and
candidate, and `change_percent = 100*(candidate_rmse/reference_rmse - 1)`.
Positive values mean worse error. RMSE pools observations and target dimensions;
it is not an average of per-root RMSEs. Whole roots, including every frame and
center, are resampled together. The 15-root transfer audit resamples three roots
within each of five fixed temperature strata; the six-root melt assay has one
1325 K stratum. Use 10,000 paired bootstrap draws, seed 20260922, and percentile
95% intervals. Counts and sums retain observation weighting even for unequal root
sizes. Actual completed populations have 64 observations/root. `roots_improved`
counts roots with strictly lower candidate mean squared error. Intervals represent
evaluation-root uncertainty conditional on fitted encoders/readouts, not training-
seed or training-root uncertainty. They are exploratory, without multiple-comparison
adjustment; small isolated differences should not be treated as confirmatory claims.

Contrasts: `encoder_training` compares the same representation/readout against
encoder step 0; `export_vs_pooled` compares exported 128D against pooled 64D at the
same encoder step; `residual_vs_ridge` compares the tuning-selected residual readout
against ridge; `relaxed_vs_observed_input` compares relaxed-input and observed-input
readouts of the SAME relaxed targets. Comparing raw observed-target errors against
raw relaxed-target errors does not isolate an input effect because targets/scales
differ. All fitted normalizers and readouts remain those of the original study.

`decoder_comparisons.csv` uses corruption-averaged per-anchor noise MSE and paired
whole-root bootstrap. Here positive `gain_percent = 100*(1 - candidate/reference)`
means improvement, unlike the RMSE change above. The implementation squares back
the RMSE ratio and transforms its interval monotonically. Root pairing retains all
corruptions on an anchor. Comparisons use initial→1,000, initial→10,000 and
1,000→10,000 frozen encoder checkpoints with identical fresh-decoder budgets.
These 10,000-draw intervals may differ slightly from original 1,000-draw exports.

Normalization clarification: the original exported follow-up prose referred to a
weighted sampled Gaussian-noise norm. The actual `bcr.objective.per_environment`
producer divides weighted squared epsilon prediction error by **3*sum(weights)**,
i.e. per-coordinate unit-noise MSE; it does NOT divide by each sampled epsilon norm.
A zero predictor has expected loss 1. This analysis preserves original numerical
results and historical definitions and records this clarification explicitly.

`target_changes.csv` pairs the original per-target/per-degree RMSE exports at steps
0 and 10,000, with the same relative-change formula; these rows have no new interval.
The radial family includes redundant weighted count and density (a constant multiple)
and near-empty radial bins with normalization floor 1e-6. Do not count them as
independent confirmations. Degree-l moment-Gram blocks each contain 36 components.

The overview uses original all-anchor intervention means at encoder step 10,000 and
noise/d0=0.12; strict matched swaps are omitted from that full-population panel.
The separately trained unconditional reference shares evaluation anchors but has a
different training trajectory. A retained-gap percentage, where discussed, is a
descriptive arithmetic ratio, not a causal decomposition of model advantage.


Table export: 2026-09-22T07:49:02.759062+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
