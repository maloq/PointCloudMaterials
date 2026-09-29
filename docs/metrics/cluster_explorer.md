# Linked static structure exploration

This revision reuses frozen cluster assignments and native 128-dimensional encoder
exports. It neither fits an encoder nor selects a checkpoint. The checkpoint,
source positions, cluster/order analysis identities and embedding cache hash are
recorded in `technical/protocol.json`.

Spatial inspection uniformly samples 4,000 saved centers without replacement per
frame, seed 20260926. The same points appear in all spatial panels. The scatter,
atlas and radial analysis use the previously retained sample of up to 64 centers
per frame and cluster; this is not a population-weighted sample. Extra retrieved
centers are inspectable but never added to those summary populations.

Local q4, q6, qbar6, normalized third-order invariants, density and mean neighbor
q6 alignment use the exact definitions in `liquid_structure.bond_order`: the
center and its twelve nearest neighbors each have twelve bonds queried from the
full source. The continuous-order scatter uses mean center-to-neighbor normalized
q6 inner product (signed, dimensionless), against raw best-template PTM RMSD.
PTM uses full-source OVITO matching, only FCC/HCP/BCC/ICO enabled. The RMSD slider
changes acceptance at display time without changing clusters. No raw match is
shown as missing in the scatter, never RMSD zero. PTM Other is not a liquid label.

The atlas samples four neighborhoods per stable rank third of q6 alignment within
each cluster's stratified sample. Ties are ordered by global sample index. Low,
middle and high are relative within-cluster bands, not universal phase thresholds.
Small thirds use all members. Every card retains source frame and exact sample ID.

Radial bins are (0,3.5], (3.5,5], (5,6.5], (6.5,8] Angstrom. The focal atom is
excluded. Every shell contains all full-source atoms in that interval, independent
of the 64-neighbor display crop. Crystal fraction counts FCC/HCP/BCC matches at
RMSD <=0.10; ICO is separate. Orientational coherence is the signed inner product
of each shell atom's normalized q6 vector with the focal vector. Each q6 uses its
own twelve nearest source atoms. First average within each neighborhood and shell,
then give each sampled neighborhood equal weight. Empty shells are undefined and
excluded with their coverage counts retained. The profile line is the mean; bands
are neighborhood 25th–75th percentiles, not uncertainty of the mean. Related frames
and overlapping neighborhoods do not provide independent confidence intervals.

Retrieval uses exact cosine distance, `1 - dot(z/||z||, z'/||z'||)`, on native
128-D exports, before standardization, PCA, UMAP or cluster preprocessing. Every
atlas sample queries all saved centers in the same frame, excluding center
separation <20 Angstrom. Three nearest eligible centers are displayed. Controls
are drawn uniformly from that frame's stratified physical sample with alignment
within 0.05, qbar6 within 0.04 and density within 10% of the query, with the same
20-Angstrom exclusion, also excluding retrieved neighbors. The calipers are fixed;
missing controls are reported. These are coarse physical matches, not evidence
of equal structures or of extra predictive information. Candidate counts, ranks,
cosine distances, separations and identities are exported.

No static view measures future crystallization. Snapshot labels organize data and
never enter an encoder or predictor. Cluster IDs are specific to each clustering.
Sparse bonds are display connections only; there are no shell hulls. SVG is opt-in.
