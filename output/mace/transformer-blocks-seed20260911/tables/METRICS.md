# Standard checkpoint analysis metrics — 2026-09-12

Metric CSVs preserve the nested keys from `analysis_metrics.json`. Counts, seeds,
cluster K, dimensions and elapsed times are metadata, not quality scores. Optional
stages contribute only the metrics they actually calculated. Arrays, identities,
configuration and raw machine results remain under `technical/`.

| Metric family | Calculation |
| --- | --- |
| `silhouette_euclidean` | Mean `(b-a)/max(a,b)` on the configured evaluation sample: a is mean same-cluster Euclidean distance; b is the smallest other-cluster mean distance. Higher is better. |
| `silhouette_cosine` | The corresponding silhouette with cosine distance on normalized evaluation features. |
| `calinski_harabasz` | Between-cluster dispersion divided by within-cluster dispersion, scaled by `(N-K)/(K-1)`. Higher is better. |
| `davies_bouldin` | Mean over clusters of the worst pairwise ratio of within-cluster dispersion to centroid separation. Lower is better. |
| `ari_with_gt`, pairwise `ari` | Adjusted Rand agreement; labels are the configured dataset labels or compared partitions. Cluster agreement alone is not proof of physical phases. |
| `nmi_with_gt` | Normalized mutual information using sklearn's arithmetic entropy normalization. |
| PCA explained variance | Sample covariance eigenvalue divided by total sample covariance variance; accumulated ratios sum the selected components. |
| Invariant latent mean/std/min/max | Statistics of the sampled latent entries; std is population std (`ddof=0`). |
| Latent `norm_mean`, `norm_std` | Mean and population std of Euclidean row norms. |
| Cluster counts/proportions | Number/fraction of analyzed samples assigned to each cluster, for the stated frame or population. |

Clustering scores use the saved feature preparation (standardization, optional PCA
and L2 normalization), fit population, sample selection and random seed. They are
not calculated in the displayed UMAP or t-SNE image coordinates. Do not compare
values across different sampling or preprocessing protocols without stating that
change. Spatial/temporal figures retain their stage-specific metadata and original
technical reports. Historical exports attach current definitions but do not claim
the old run was recomputed with current code.

The following section defines every current relaxed-topology score used in the
standard MACE comparisons.
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


Table export: 2026-09-12T07:40:37.596881+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
