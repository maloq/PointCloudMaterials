# GATr directional audit, version 1

Frozen Al-only v6 GATr checkpoint 1216, SHA256
`3534c3cd6065d41ca32d6ed3e77534a79429d80ad3e195c0b14beb8ce36e42eb`.
Native selective BF16 inference with FP32 geometric operations; A100 on node07.
No training, smoothing or learned probe. Same 10 held-out Al sources and 40 atom
tracks as the trajectory-stability audit: 801 frames, 0–600 ps, cadence 0.75 ps.
Five training-reference sources supply 420 observations. This is exploratory
analysis of previously inspected test trajectories, not a new blind test.

## Representation

Capture center multivectors after block 1, at the normalized input of block 2's
MLP, and after block 2. Eight channels, 16 PGA coefficients. In the pinned GATr
basis, four triplets transform as 3-vectors under proper rotations around the
center: `plane_normal=(e1,e2,e3)`, `ideal_bivector=(e01,e02,e03)`,
`axial_bivector=(-e23,e13,-e12)`, `point_numerator=(-e023,e013,-e012)`.
Do not interpret arbitrary multivectors as physical points, rotors or crystal
axes. No homogeneous-coordinate division. Only plane_normal is independently
translation invariant; the other triplets use the fixed center as local origin.
We test SO(3), not reflection parity. The final returned multivector is discarded
by the scalar readout. Intermediate location alone does not establish influence
on z; intervention controls separately measure sensitivity at inference precision.

Choose one channel per stage/triplet by largest **training-reference RMS norm**.
Channels of zero reference norm have undefined directions and are excluded from
selected-field summaries, but retained in reference and all-channel tables.
Primary field is block2_mlp_input/point_numerator, selected by this same rule.
No choices depend on temporal or spatial test outcomes. Valid directions require
norm > 0.1 times that channel's training RMS norm. This is a conditioning
threshold, not a calibrated statistical confidence. Both endpoints must pass.
Sensitivity table repeats primary time metrics at fractions 0,0.1,0.3,0.5,1.
All scalar measurement calculations use float64.

Baselines: weighted displacement mean (density dipole) within a cosine taper
5–7 Å, and within native support (15–17 normalized units, about 16.87 Å outer
radius); center excluded. Shape axis is the largest-eigenvalue eigenvector of
the 7 Å weighted displacement second moment about the center. Reject shape axes
if (largest minus middle eigenvalue)/trace < 0.01. Its sign is arbitrary: only
axis-angle or P2 has physical meaning for that baseline.

## Temporal measurements

Unit direction u=v/||v||. Signed turn is acos(clamp(u_t dot u_(t+lag),-1,1))
in degrees; axis turn uses the absolute dot product. P1 is mean dot product;
P2 is mean (3 dot²−1)/2. Independent isotropic directions have mean signed
angle 90°, P1=0, P2=0, mean axis angle 57.3°. `flip90_fraction` is fraction
strictly above 90°; `jump60_fraction` is fraction above 60°. Coverage is valid
pairs/all pairs. Lag summaries use the same endpoint rule at each lag, so their
populations can differ. No angle unwrapping or temporal smoothing.

For cage correction, intersect nearest-80 identity sets at adjacent frames,
remove the central atom, and fit the proper Kabsch rotation xR≈y about the
tracked center (do not recenter the neighbor mean). Compare v_t R to v_(t+1).
`cage_fit_rms_A` is RMS atomic residual after rotation, in Å. This removes
best-fit rigid cage motion; it is approximate when atoms rearrange.
Phase-conditioned pairs retain the same PTM label at both endpoints. PTM
unclassified is not assumed to be liquid. Source-phase counts and valid counts
are exported; absent subsets are undefined, never zero.

`channel_axis_rank1_fraction` averages the largest eigenvalue/trace of the
sum of outer products of valid unit channel directions, separately per
observation; 1 means all channel axes collinear, 1/3 isotropic directions.
Baseline alignment is P2 between simultaneous learned and geometric directions.
`z_increment_spearman` correlates vector-increment norm with z128 increment norm.
`cage_residual_angle_spearman` correlates angular turn with cage fit residual.
Reference channel norms are uncentered RMS. Phase relative RMS divides by the
same fixed training RMS, never a within-phase or within-test fit.

## Spatial measurements and null

At frames 0,128,256,384,512,640,800 in each test source, sample the nearest 128
atoms around each of two previously selected tracked centers, plus 64 seeded
uniform centers. Deduplicate by atom identity, retaining first patch membership.
This gives 70 fixed snapshots, roughly 320 centers each. Every center has its own
native full-support local encoding. Periodic pair separation bins are
[0,4),[4,8),[8,12),[12,18),[18,26),[26,40),[40,80) Å.
These are sampled patches, not an unbiased full-system pair correlation.

Compute P1 and P2 over unordered distinct pairs whose directions are valid.
Null is the **exact expectation** under independent uniform permutations of
valid directions within each snapshot's PTM label groups, conditional on the
valid positions. This preserves label-dependent direction distributions.
For group g of n vectors, s=sum u, Q=sum uu^T: same-group expected dot is
(s·s−n)/(n(n−1)), expected dot²=(tr(Q²)−n)/(n(n−1)). For different groups use
(s_g·s_h)/(n_g n_h) and tr(Q_g Q_h)/(n_g n_h). Convert dot² to P2 and average
over the observed pairs in each distance bin. `excess_p1/p2` subtract this null.
The legacy config field spatial_permutations is unused: exact expectation has
no Monte Carlo error. The null does not remove shared-neighborhood effects.
`overlap80` reports mean intersection size/80 of endpoint nearest-80 identity
sets, including centers. Native support is larger than nearest-80 support;
small overlap80 does not imply disjoint native inputs.

## Aggregation and uncertainty

Temporal summary: average a metric within each source, then equally over ten
sources. Median/p95 summary is the mean of source medians/p95s, not a pooled
quantile. Spatial summary: average each metric equally over that source's
available snapshots in the distance bin, then equally over sources. Export
per-snapshot pair counts. 2,000 source bootstrap draws, stratified by temperature
(two sources each at 400/450/500/510/520 K), with all tracks and frames retained
together. Intervals are percentile 95%, conditional on checkpoint and fixed
training reference. If a source has no defined metric, aggregate is undefined
and the number of available sources is reported. Never treat pairs as independent
replicates. Phase tables contain descriptive source values, plus source-balanced
temporal phase summaries. Primary and block-1 point-triplet spatial phase tables
restrict both endpoints to the same specified PTM phase (unclassified, FCC, HCP),
using the same shuffle expectation and source bootstrap; frames without a pair
in the subset/bin are absent. Small HCP populations remain exploratory.

## Numerical and mechanistic controls

Five random proper rotations on 16 reference inputs check vector covariance
and scalar invariance; repeated inference and final-output erasure controls.
Separate interventions use 60 observations spanning five test sources and three
times, in native BF16 and autocast-disabled float32: erase all token vector
triplets after specified stages, independently randomize atom directions while
preserving radius and scalar support metadata, or scale radii by 1.01 while
preserving support weights. These are out-of-distribution sensitivity probes,
not physically plausible molecular perturbations. Export RMS/max absolute
z128 changes and mean per-state L2 change. A zero difference establishes only
insensitivity for these inputs at the tested numerical precision.
