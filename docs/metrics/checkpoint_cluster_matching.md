# Checkpoint explorer: descriptive cluster correspondence

The reference population is exactly the 24,960 original `all_test` PaCMAP
observations from the matched Al64 spatial VICReg study. The `interface20`
population contains 12,690 of these observations and retains its separately
saved PaCMAP layouts. Neither population is resampled per model. Existing
coordinates, cluster assignments and source/frame/atom identities are verified
against saved numerical projections and original assignment files.

For each of 54 neural spaces (three neighbor-alignment coefficients × three
training seeds × epochs 4/12/24 × encoder/projector) and four descriptor families
(TDA, bond order, CNA, family-balanced joint), form the 7 × 7 contingency C with
neural IDs as rows and descriptor IDs as columns. SciPy `linear_sum_assignment`
with `maximize=True` finds the bijection p maximizing sum_n C[n,p(n)]. The
objective is raw shared membership on `all_test`, not normalized overlap.
The recorded solver version determines tie resolution; a tied solution need
not identify a unique physical correspondence.

Descriptor colors stay fixed. Neural cluster n gets descriptor p(n)'s color.
For a given checkpoint, representation and descriptor family, this permutation
stays fixed across populations, filters and full-snapshot MD. Original IDs stay
visible. Mapping colors separately for each checkpoint does not establish a
stable neural cluster identity across training epochs.

Exports record the following on `all_test` and `interface20`, using the same
reference permutation in both. The browser recomputes these summaries only for
the currently displayed source/frame/region subset, without reoptimizing colors:

- Matched fraction: sum_n T[n,p(n)] / sum(T).
- Matched-pair intersection: T[n,p(n)]. Union: row sum + column sum − intersection.
- IoU: intersection / union. Directional fractions divide the intersection by
  neural membership or descriptor membership respectively.
- The heatmap shows counts or P(descriptor ID | neural ID); hover includes both
  directional fractions. Rows are ordered by the assigned descriptor ID.
- Adjusted Rand index: sklearn `adjusted_rand_score` for exported summaries;
  the equivalent contingency formula in the browser. With a = sum choose2(cells),
  b = sum choose2(row sums), c = sum choose2(column sums), q = choose2(total),
  ARI = (a−bc/q)/((b+c)/2−bc/q). Identical trivial partitions are 1. Fewer than two
  displayed centers is undefined. Empty denominators are null/blank, not zero.

These are descriptive correspondences, not independently evaluated predictions
of descriptor labels: the display assignment uses the same reference population.
Large clusters can dominate matched fraction; the matrix and per-pair IoU reveal
splits and merges. There is no model ranking, significance test, atom-level
confidence interval or checkpoint promotion. Color similarity is not physical
ground truth. The static six-snapshot comparison retains its separate reference
population and original frozen definitions.

The default is GeoFormer with spatial-neighbor VICReg, repeat 1 (seed 17), raw
encoder at the final epoch 24. This is a fixed browsing default, not a selected
winner. Epochs 12 and 4 are intermediate and early checkpoints. S0/S0.5/S1 denote
same-center, equal-mixture and spatial-neighbor alignment, with unchanged variance
and covariance terms. Seed repeats are for reproducibility. No encoder training,
clustering or PaCMAP refitting occurs. Full MD coloring uses frozen checkpoints
and saved centroids through the existing dense-MD producer on Slurm; each replay
is bound to the exact original checkpoint hash. Observed inputs are centered,
periodic nearest-80-atom geometry with fixed length normalization, no history,
motion, temperature, absolute time or physical teacher. Frame indices and source
IDs are display metadata. No predictor is run.

Implementation: `checkpoint_explorer.py`, the `summarize` helper in
`cluster_matching.py`, and `cluster_comparison.js` / `.html`, under
`src/research/spatial_vicreg_bias/`. Source, assignment, coordinate and renderer
hashes are saved with the publication and numerical bundle.
