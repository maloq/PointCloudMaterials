# Trajectory figure metrics

These figures reuse the completed context-night study's four development-selected
36-epoch-budget family promotions. Actual training may early-stop. No predictor is
fitted and no model is chosen on the displayed test scores.

`forecast_curves.csv` contains equal-source-weighted means across 30 historical
test sources (44,385 overlapping, naturally sampled at-risk windows). Physical
MSE is the original `test_path_scores[:,:,1]`: squared error of the predictive
sample mean, averaged over the 128 physical packet channels after division by
their training-only target scales. Persistence is the corresponding archived
`test_persistence_scores[:,:,1]`. Lead times are 3,6,...,96 ps. Values are not
coordinate displacement errors. Brier at 0.75,1.5,...,96 ps is the squared error
between predicted onset CDF and whether first sustained local onset has occurred.
Mean over horizon is the equally weighted average of these time points.

Figure bands are 2.5/97.5 percentiles of 1,000 paired bootstrap resamples of the
30 source means. They include source uncertainty, not training-seed uncertainty;
windows are not treated as independent replicates. These are historically
examined test roots and exploratory illustrations, not a new confirmatory test.

Illustrative windows are selected as median-error windows within each source,
then the median across eligible sources, excluding already selected sources.
The strata are onset <=12 ps, onset (12,48] ps, no onset <=96 ps, and a missed
<=12 ps onset at the direct model's calibration-only 5% FPR threshold. Selection
uses physical/order/crystallinity path error for representative pictures only.
It does not alter any aggregate metric or select model hyperparameters.

Probabilistic path plots use 64 fresh, fixed-seed CPU samples. The shaded region
is the pointwise 5–95% sample interval, not a confidence interval or simultaneous
trajectory coverage guarantee. No best-of-samples selection. Onset CDFs and
aggregate plots retain the original saved predictions. All rollouts are open
loop. Future physical order is predicted directly alongside the future embedding,
not obtained by decoding that embedding or forecasting atomic positions.

UMAP: training-only per-channel standardization, 30 neighbors, min_dist=0.15,
Euclidean metric, fixed seed. Fit 64 outcome-independent timeline states per each
of 90 training sources, transform 64 per each of 30 test sources. Timeline sampling
covers 0–594 ps and includes post-onset states. Colors are diagnostic labels,
never UMAP fitting targets. Prediction paths transform the 128D predictive mean.
Two-dimensional distances are not physical distances or forecast error metrics.

The spatial illustration uses original periodic coordinates and the exact cached
seven-representative construction. Local radii are converted from normalized
encoder units to angstroms using the recorded Al calibration. Representative
selection is repeated independently at every observation time. Lines/circles in
the diagram show construction, not inferred causal influence or attention weights.


Table export: 2026-09-21T14:32:42.847107+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
