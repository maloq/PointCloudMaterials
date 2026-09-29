# Rich structural clusters and neural clusters around the interface

This diagnostic fits independent multivariate descriptor partitions, then compares
their atom assignments to retained neural K-means assignments. It is not a scalar
TDA regression, a neural retraining, a checkpoint selector or a future predictor.
The historical readout experiment used 258 TDA, 40 bond-order and 45 CNA columns
before removal of training-constant features; its family averages were prediction
summaries, not descriptor-cluster correspondence. Those exports are preserved.

## Population and inputs

The exact uniform structural assay is reused: 64 fixed atom centers at each of
13 frames per source, 150 original Al source ancestors, 90/15/15/30 frozen roles.
There are 124,800 observations and 24,960 test observations. This is the uniform
structural track, not the temporally selected all64/legacy16 prediction track.
All encoders use identical observations. Atom/source/frame/coordinate and PTM
identity are verified. Existing inference consumes observed periodic relative
coordinates of 80 atoms, with no history, motion, time, temperature or teacher.
The existing neural clusterer consumes the raw encoder or projector vector.
The new descriptor clusterers consume only geometry-derived rich descriptors
of those same 80 atoms: H0/H1/H2 persistence images, smooth Betti curves, lifetime
and birth/death summaries at 32/80 atoms; q_l/qbar_l, w_l and bond coherence;
and CNA signature histograms and moments at fixed/adaptive cutoffs. No spatial
position, distance, PTM class or crystal fraction is a clustering feature.
Physical reference metadata defines diagnostic populations and controls only.

## Interface reference

Reuses frozen PTM (RMSD cutoff 0.1; FCC/HCP/BCC types 1/2/3). On each full periodic
cell, connectivity uses 3.6 Å. A boundary atom is PTM crystalline, belongs to a
connected crystalline component of at least 64 atoms, and lies within 3.6 Å of a
noncrystalline component of at least 64 atoms. The inherited interface_mask
producer supplies disordered connectivity; the additional instantaneous crystal
component filter excludes isolated PTM hits. No future confirmation or trajectory
age enters this definition. This differs explicitly from the established-nucleus
interface prediction target; it is a static operational reference, not truth.

Distance is the periodic distance to the nearest crystal-side boundary atom.
Missing boundaries have infinite distance and a separate population. Crystal-side
atoms are plotted with negative distance, disordered atoms positive; boundary
atoms have zero distance. This is distance to a finite atom layer, not an exact
continuum dividing surface. Defect pockets (noncrystalline components smaller
than 64) are separate from the connected liquid. Interior grain boundaries may
qualify when their disordered components exceed the threshold; no claim of a
unique liquid/solid surface or of specific defect identity is made.

Primary region: within 12 Å on either side. Report the boundary atoms themselves,
crystal and connected-disorder shells 0–3.6, 3.6–8, 8–12, 12–20 and beyond 20 Å,
mixed-support patches, small disordered pockets, and nearby clear-input liquid.
The population masks overlap by design. Never count their sum as a total.

## Independent structural clustering

TDA-only, bond-order-only, CNA-only and their concatenation are clustered in their
original multivariate spaces, never in UMAP. Standardization is fitted only on
training sources, separately for all-phase and interface-within-12-Å fitting.
Equal source mass within each fitting population is used for means, variances
and K-means sample weights. Columns with standard deviation at most
1e-8*max(1, abs(mean)) are removed. Each retained family is divided by the square
root of its active dimension, so the joint metric gives equal expected squared
distance weight to the three standardized families. No PCA truncation,
whitening, target fitting or test-selected metric is used. Correlated descriptor
coordinates can still overweight recurring structure within a family.

MiniBatchKMeans uses K=3/6/7/10, batch 4096, n_init=3, max_iter=200,
reassignment_ratio=0 and independent seeds 17/29/43. Seed 17 is the predeclared
reference, never chosen for the best neural agreement. Pairwise descriptor-fit
ARI on held-out interface atoms quantifies random-initialization stability.
All seed assignments and model parameters are saved. The neural partitions stay
the original GLOBAL fits: interface-focused descriptor fitting does not silently
refit the neural clusters or assert identical fitting populations.

## Correspondence

On identical held-out atom IDs, retain full contingency counts and both row- and
column-normalized heatmaps, ARI, arithmetic-normalized AMI, entropy of each
partition and both conditional entropies in nats. Cluster numbers/colors from
independent fits are arbitrary. Global and each shell/phase population are
reported independently; fewer than 30 observations is explicitly insufficient.
Dominant-cluster counts remain alongside the agreement metrics, since a trivial
one-cluster match is not evidence of distinct interface states.

K=7 interface comparisons at epochs 12/24 and classical controls include 1,000
source bootstrap draws: whole source contingency matrices are resampled and ARI
is recalculated. These intervals condition on the fitted encoders and reference
clusterer. Trajectory figure bands are the min–max of three encoder seeds, not
confidence intervals. All seven saved epochs, raw/projector and K values remain
in the numerical output; no winning epoch or partition is selected.

The matched null shuffles neural assignments 99 times within source, frame,
PTM crystalline/noncrystalline side, floor(distance/3.6 Å), and local 80-atom
crystal-fraction quartile. Report observed MI, mean shuffled MI, their difference,
shuffle central 95% range and fraction of rows in nonsingleton strata. This is a
descriptive spatial/confounder control, not an iid atom-level significance test,
conditional mutual-information estimate or proof of a causal physical state.
Small strata can make the null weak; the movable fraction is essential.
Saved crystal-fraction/diffusion/q6 clusterings receive identical correspondence
assays. They are privileged/reference-dependent controls, not equal-input models.

## Rich profiles and spatial display

Every retained standardized descriptor coordinate is saved as a mean within each
descriptor/neural cluster on held-out interface observations, with counts and
feature names. Plot color is clipped to ±3 training standard deviations; numerical
profiles are not clipped. This allows distinct multivariate signatures to be
examined without reducing them to a family prediction score. Cluster means can
hide multimodality; correspondence matrices and counts must be read with them.
Spatial displays use identical fixed sampled atoms. They are explicitly sparse
64-center xy projections, not interpolated dense maps or estimated surfaces.
Three illustrative sources/frames are chosen by interface sample coverage only,
never by agreement or visual appeal. Inferences use the full held-out population.

All calculations run on Slurm CPUs; diagnostics create no W&B runs. Existing
neural checkpoints and metric definitions remain unchanged. Descriptor agreement
is evidence of reproducible reference structure, not a unique ground truth,
precursor identity, future crystallization prediction or Ta/Zr transfer result.


Table export: 2026-09-29T11:06:11.113135+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/interface_cluster_correspondence.json` relative to the analysis root.
