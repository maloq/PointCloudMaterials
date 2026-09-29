# Optimal cluster colors and overlap

This is a descriptive display assignment, not encoder training, cluster fitting
or an independently evaluated classifier. Original neural and descriptor IDs,
memberships and PaCMAP coordinates remain unchanged.

For each of four neural spaces and four descriptor families, sum the saved
all-grid contingency tables over all six static Al snapshots (684,723 centers).
Transpose the original descriptor × neural table to obtain `C[n,d]`, with neural
IDs as rows and descriptor IDs as columns. SciPy `linear_sum_assignment` with
`maximize=True` chooses a bijection `p` maximizing `sum_n C[n,p(n)]`. Raw shared
membership is the objective; no overlap normalization or phase weighting is
used. The SciPy version and source hashes are recorded. Tied optima use the
solver's deterministic ordering and do not establish a unique physical match.

Descriptor colors stay fixed. Neural cluster n receives descriptor p(n)'s color
in both PaCMAP and dense MD. The permutation is fixed across snapshots, filters,
slabs and the all-static/interface20 pages for that neural/descriptor pair.
Changing either feature space selects that pair's stored permutation. Colors
can be switched to original IDs without altering the correspondence statistics.

For any displayed contingency table T using the same ID orientation:

- Matched rows: `sum_n T[n,p(n)]`; matched fraction divides this by total rows.
  This uses the pooled mapping, not a newly optimized filtered mapping.
- Neural fraction: matched intersection / all members of that neural cluster.
- Descriptor fraction: matched intersection / all members of that descriptor cluster.
- Matched-pair IoU: intersection / (neural count + descriptor count − intersection).
  Empty denominators are undefined (JSON null / blank table values), never zero.
- Heatmap cells default to the fraction of each neural cluster assigned to a
  descriptor cluster; a count view is also available. Rows are ordered by their
  assigned descriptor, so the assigned cells lie on the diagonal. The hover
  includes both directional fractions and exact counts.
- Adjusted Rand index is invariant to color permutation. From the contingency,
  let a = sum choose2(cells), b = sum choose2(row sums), c = sum choose2(column
  sums), q = choose2(total). ARI = (a−bc/q)/((b+c)/2−bc/q); the identical trivial
  partition case is 1, while fewer than two observations is undefined.

The exported CSV records pooled and per-snapshot dense-grid overlap diagnostics.
The browser's heatmap, IoU bars, matched fraction and ARI use only the currently
displayed PaCMAP rows after frame/region filters (24,000 all-static or 8,834
interface20 before filtering). These dynamic values are distinct from the
dense reference used to choose colors. The current shared frame slider changes
both the PaCMAP subset and MD snapshot; the independent MD z slab does not change
the projection subset. Neither control refits the mapping. Earlier published
pages had independent frame controls; their numerical exports remain unchanged.

A large matched fraction may be dominated by large crystal/liquid clusters;
the matrix and per-pair IoU expose splits and merges. Shared colors do not
assert physically equivalent states. This static transfer population has no
independent-source prediction claim. Counts are descriptive; no atom-level
confidence interval or significance test is added. Existing metric definitions
and numerical exports remain frozen in their original analysis bundle.

Implementations: `cluster_matching.py` (dense reference and exports) and
`cluster_comparison.js` / `.html` (selected-row diagnostics and display), under
`src/research/spatial_vicreg_bias/`.
