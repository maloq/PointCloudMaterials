# Experiment-plan metric tables — 2026-09-12

The collector preserves the metric names requested by the plan. It reads
`final_metrics.json` (including the new `technical/` location), then training/Slurm logs, then checkpoint callback values.
A log value is the last matching occurrence. The current "best" table uses the
collected value of its requested metric; it does not independently search all
training epochs for an extremum. Missing values are displayed as `-`, never zero.

| Field/table | Calculation |
| --- | --- |
| Per-run final/best fields | Collected scalar for that run and stage, following the source order above. Names containing `/` or spaces use `_` in CSV columns. |
| Grouped final/best mean | Arithmetic mean over completed repeats with that metric present. Failed/missing runs do not enter these grouped estimates. |
| Grouped standard deviation | Sample standard deviation (`ddof=1`); zero when fewer than two values are available. |
| `count` | Number of included metric values; distinguish this from successful and scheduled repeat counts. |
| `ci95` fields | Mean ± Student-t critical value times sample std/sqrt(n), using the implementation's finite critical-value lookup and 1.96 outside it. Width is zero for fewer than two repeats; this is not evidence of zero uncertainty. |
| Ungrouped global summary rows | Mean/std of all available collected values, including values from unsuccessful runs. Use grouped tables for completed-repeat comparisons. |
| Display rounding | Values ≥1 use three decimal places; smaller magnitudes use two significant digits. JSON retains original numeric precision. |

Training supervised-cache metrics have the following definitions:

- ARI: sklearn adjusted Rand agreement of ground-truth labels and KMeans assignments.
- NMI: sklearn normalized mutual information (arithmetic entropy normalization).
  For ARI/NMI, KMeans uses the number of observed classes, k-means++ initialization,
  10 initializations, and random state 0.
- `ACC_KMEANS_PLUSPLUS_HUNGARIAN_K*`: maximize the label/cluster contingency-table
  assignment with the Hungarian algorithm, divide matched samples by all samples.
  Use configured K and evaluator seeds. `_STD` across evaluator runs uses `ddof=0`;
  `_BEST` is the largest accuracy. These are evaluator repeats, not training seeds.
- Embedding norm mean/std: Euclidean row norms, population std. Intra-class distance
  means average within-class unordered Euclidean pairs; inter-class means average
  cross-class distances. Optional classification accuracy is cross-validated logistic
  regression as configured in `compute_embedding_quality_metrics`.

For topology metric names (`balanced_mse`, `mean_within_frame_r2`, `projector_ridge_mse`)
use the accompanying topology definitions. Training loss definitions are owned by
the selected objective/config; the collector does not rename one objective as another.
Do not compare losses with different weights, target scalings, splits or supports as
if they measured the same quantity. Historical values exported again are not recomputed.
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
