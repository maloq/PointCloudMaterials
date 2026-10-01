# All-training descriptor comparison

This version replaces interface-adjacent descriptor fits in the active GeoFormer
and frozen MACE viewers with the already fitted `all-{family}-k7` models.
The training population is all 74,880 uniform assay observations from the 90
training sources of fixed Al64 release
`e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`.
It means all phases in the declared sampled training cohort, not every atom in
every training trajectory. No validation, calibration, held-out or static rows
enter the descriptor fit. The primary seed is 17, fixed in advance.

TDA, bond order, CNA and their joint vector each use train-source-balanced means
and standard deviations. The original fit removes columns with standard deviation
at most `1e-8 * max(1, abs(mean))`; standardization is cast to float32 before
division by the square root of the active feature count in each family.
Seven-cluster MiniBatchKMeans centers, column indices, moments and family weights
are reused unchanged. Assignment is nearest center in that space. Reproduced
held-out assignments must exactly equal the original saved primary-seed labels.
PaCMAP is recomputed from this standardized descriptor space for the same
displayed observations; it does not define cluster membership. Its parameters and
version are pinned in `configs/analysis/general_descriptors_20260930.json`.
Neural weights, neural clustering and neural PaCMAP coordinates remain unchanged.

For each neural model/descriptor pair, the contingency has original neural IDs
as rows and descriptor IDs as columns. SciPy's `linear_sum_assignment` maximizes
raw shared membership under a one-to-one permutation. The reference is all
24,960 uniform held-out displayed observations for dynamic Al, and all 684,723
grid centers across six snapshots for static Al. The mapping remains fixed across
frames, source filters and MD slabs. This is a descriptive color assignment,
not a fitted predictor or evidence of physically equivalent states.

Exported metrics:

- `rows`, `matched_rows`, `matched_fraction`: total count, sum of assigned
  contingency cells, and that sum divided by total count.
- Per-pair `intersection`, `union`, `neural_rows`, `descriptor_rows` and
  `neural_fraction`, `descriptor_fraction`, `iou`: intersection divided by
  neural count, descriptor count or union respectively. Empty denominators are
  null, never zero.
- `adjusted_rand_index`: sklearn adjusted Rand agreement, invariant to the
  color permutation.

Browser correspondence uses the currently displayed PaCMAP rows, with the same
fixed permutation. The interface highlight only changes opacity: finite physical
interface distances <=12 Å are emphasized. It never filters observations, refits
clusters, changes correspondence counts or alters MD. The underlying physical
interface definition is preserved from the frozen assay; it is not a learned
cluster label.

Dense descriptor labels, examples and ideal-lattice fits are regenerated for the
new memberships. Up to five examples are uniformly sampled without replacement
per descriptor cluster using the recorded deterministic snapshot/family/cluster
seed. Local-example geometry and PTM fitting reuse the existing producers.
Travel keeps its frozen real observations and neural vectors; only the stored
descriptor IDs change. Its original uniform-plus-cluster-supplement sampling is
recorded in provenance, not represented as a new unbiased sample.

No interface-only metrics are exported in this revision. Earlier interface fits,
projections and metric definitions remain historical artifacts. No significance
claims or atom-level uncertainty intervals are computed. Large crystal/liquid
populations can dominate the pooled metrics; inspect the full contingency and
per-pair IoU.


Table export: 2026-09-30T01:30:45.539594+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/general_descriptor_comparison.json` relative to the analysis root.
