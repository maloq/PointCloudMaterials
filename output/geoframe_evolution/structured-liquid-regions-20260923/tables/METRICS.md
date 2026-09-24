# Dense structured-liquid candidate regions, v1

Uses the frozen GeoFrame reference assay's Ta 2.9 ns and Zr 240 ps snapshots,
full-cell PTM and material-specific fitting-side 75th-percentile qbar6 threshold.
These are structural hypotheses; no temporal or thermodynamic ground truth is
assigned to the term candidate. The original static potential identities are
unknown. No result is silently pooled with another potential's dynamics.

Select context-6 anchor atoms at least two encoder normalization radii from
every accepted FCC/HCP/BCC atom (PTM RMSD <=0.10). Rank by qbar6 and greedily pick
up to six centers, keeping centers more than four radii apart. Selection uses
present geometry only. It deliberately enriches high-order examples and cannot
estimate how frequent precursors are in the whole liquid.

For each selected center analyze **all** atoms within two normalization radii.
The original full snapshot supplies each atom's own 14 nearest neighbors and
their 14 neighbors, so the local order calculation has no patch-boundary halo
truncation. Use equal-weight q4/q6, normalized hat-w4/hat-w6, qbar6 and normalized
q6 bond coherence as in `geoframe_evolution.md`; context 6 requires high qbar6,
low crystalline-neighbor fraction, and exclusion of the five-fold proxy.

Build an undirected graph among context-6 atoms. A bond must appear in **both**
14-neighbor lists and have normalized complex q6 inner product >0.70. Connected
components include singleton candidates. Record atom count, mean qbar6/hat-w6,
minimum distance to an existing PTM crystal and raw best-template counts (FCC,
HCP, BCC, ICO), even when the template RMSD fails acceptance. A component is
potentially truncated if any member has a 14-neighbor entry outside the sampled
sphere. `components_ge5` is only a descriptive size count, not a nucleation
threshold. No critical size or committor is inferred.

CSV fields: selected_regions; maximum_component (largest observed component in
any inspected sphere); total_candidates (sum across the disjoint inspected
spheres). Individual components and atom identities remain in technical JSON/
NPZ. Spatial plots use a central z slab of half-thickness 0.35 normalization
radii. Blue marks ordered-liquid candidates, pink five-fold proxies, grey
unclassified surroundings; drawn bonds satisfy the graph rule. Color is an
annotation of the independent geometry, not an encoder's discovered clusters.


Table export: 2026-09-23T15:04:33.600146+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
