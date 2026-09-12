# Relaxed topology metrics — 2026-09-12

These definitions describe the current standard topology stage, not a retroactive
reinterpretation of retired MACE objectives. Preserve this document with each CSV.
The raw target has 144 coordinates: H0 `[0:16]`, H1 `[16:80]`, H2 `[80:144]`.
Targets describe the configured relaxed neighborhood; potential, minimizer, atom
support and source splits remain part of the run's saved configuration.

| Metric | Calculation and interpretation |
| --- | --- |
| `blocks.Hd.mse` | Mean squared prediction error over samples and coordinates of block d, after undoing the training target transform. Lower is better; units are squared raw descriptor units. |
| `blocks.Hd.scaled_mse` | Block MSE divided by the squared training block scale. The scale is `sqrt(mean(training coordinate variances) + floor²)`, where `floor = floor_fraction * max(training block standard deviations)`. |
| `balanced_mse` | Arithmetic mean of the three scaled block MSEs. H0/H1/H2 receive equal weight despite their 16/64/64 dimensions. Dimensionless; lower is better. |
| `raw_mse` | Mean raw squared error over all 144 coordinates and samples. Unlike balanced MSE, larger blocks contribute more. |
| `blocks.Hd.r2` | `1 - block MSE / mean((target - target coordinate mean)²)` on the evaluated split. Higher is better; negative means worse than the split mean. |
| `blocks.Hd.within_frame_r2` | Same numerator, denominator centered separately within each source/frame context. This tests local variation after removing frame averages. |
| `mean_block_r2`, `mean_within_frame_r2` | Unweighted means of the three respective block scores. They are not R² computed from balanced MSE. |
| `by_temperature`, `by_source`, `by_frame` | Scores on the indicated subsets; source/frame breakdowns recompute their own variance denominators. Temperature entries report balanced MSE. |
| `projector_ridge` | StandardScaler and Ridge fitted on training representations and transformed training targets only, using configured alpha; predictions are transformed back to raw target units before scoring. |
| `prediction` | Trained TDA head if present; otherwise the training-only ridge probe. See `primary_prediction` in the technical result. |
| `training_mean` | Mean raw target from training rows broadcast to evaluation rows. |
| `mse_mean`, `ridge_mse_mean` | Mean of the per-seed test balanced MSEs for the declared model variant. |
| `mse_std`, `ridge_mse_std` | Population standard deviation across the declared seeds (`ddof=0`), not a confidence interval. |
| `relative_mse_reduction` | Average paired row errors over seeds first, then average within each source. Report `1 - mean(candidate source errors)/mean(reference source errors)` with equal source weights. |
| `source_bootstrap_95_percent_interval` | 4,000 paired resamples of whole held-out sources with replacement; 2.5/97.5 percentile bounds of relative error reduction. Configured seed; fitted seeds are averaged before resampling. |

`val/` and `test/` flat training-summary fields use the corresponding split's
prediction or ridge result. The selected checkpoint minimizes the original combined
validation loss; it need not minimize topology error. History interventions replace
or reorder history only at evaluation; the ridge probe still fits real training data.

Intervals are exploratory for previously inspected cohorts. Frames and atoms within
a trajectory are not independent source replicates. Zero variance is not replaced by
an epsilon in this implementation: undefined/nonfinite results must be investigated.
The source manifest, checkpoint hash, target scaling and exact counts are retained in
technical files. Historical values exported again are not recomputed by table export.

Topology comparison CSVs include both bounds of each source-bootstrap interval; the complete comparison JSON is under `technical/metrics.json`.
