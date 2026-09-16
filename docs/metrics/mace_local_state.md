# Frozen local-group representations and uncertain states

**Historical protocol, discarded 16 September 2026.** The implementation is no
longer active. Definitions below and original run exports remain as evidence;
see [retained source and results](../discarded_frozen_encoder_maps.md).

The scientific object is a local group of atoms, optionally observed at two
adjacent times spanning 0.75 ps. No forecast, committor, remaining lifetime,
crystallization time or global process coordinate is fitted or selected here.
The MACE backbone and existing VICReg projector remain frozen.

The command dispatcher also exposes separately named `smooth-*` stages for
`mace_local_smooth_v1`. They use their own configuration, implementation and
[metric definitions](mace_local_smooth.md); the affine-map and state-discovery
calculations in this document remain unchanged.

## Group physics and learned distance

Group observables are the weighted mean and standard deviation of the eight
`ORDER_NAMES` in `src/analysis/liquid_structure.py`: q4, q6, normalized w4/w6,
neighbor-averaged qbar6, mean normalized q6 bond coherence, density from the
12th neighbor radius, and smooth coordination (first 12 bonds, scale 3.7 A).
Weights are exactly the encoder's C2 taper: full through 5 A, zero at 7 A.
Every contributing atom and its 12 neighbors have complete 12-bond queries
inside the 18 A halo. These are group statistics, not center-only labels.
The local qbar6 averages complex q6 vectors before their norm; the group mean
then averages that scalar. Nearest-12 identity changes can still affect teachers.

The physical map predicts ten standardized group statistics: means and standard
deviations of q4, q6, qbar6, coherence and density. Training scales are the
within-context standard deviations. A float64 ridge map W is fitted to
within-context residuals; its deployed output uses a fixed training mean and
intercept. Euclidean distance after W is a learned positive-semidefinite physical
distance in frozen feature space, M = W W^T. Its interpretation is conditional on
these chosen teachers. W4, w6, coordination and their spreads, plus instantaneous
and relaxed TDA, are never representation-training targets.

Ridge strength is chosen on validation-source within-context teacher MSE.
Short-time canonical coordinates use current/previous pairs at 0.75 ps,
context-centered covariance, null-direction removal and regularized two-sided
whitening. The current-side canonical coordinates have equal weights; no kinetic
map scaling or implied lifetime is used. Regularization maximizes validation
within-context same-map correlation. Context means are removed only during
fitting/scoring; deployed maps need no whole-frame statistics or time index.
Controls are 16D PCA after train-only feature standardization for inner, dual
and projector features. Coordinates are never fitted in UMAP space.

## Information and geometry tables

The original 30-source partition is unchanged: 18 training, 6 validation and
6 test sources, with 64 centers at each of three contexts per source. This is
source-held-out evaluation of newly fitted maps; it is an existing encoder
development cohort, not a new untouched test set for the whole research program.

`information.csv` uses independent float64 ridge readouts for every representation
and target, selecting each alpha on validation sources. Group targets have one
training standard deviation per target. Each 144D TDA block has one RMS training
standard deviation across its pixels; this handles zero pixels and tiny Gaussian
tails without amplifying them. `normalized_mse` is held-out MSE divided by this
training scale squared. Smaller is better. Per-target `test_r2` uses that source's
target variance; zero variance is undefined. Family summaries average targets,
then give each test source equal weight. Averaged pixel R2 is not a block R2.
Teacher-group accuracy is alignment evidence, not independent physical validation.

`snapshot` uses current features. `trailing_075ps` averages current and previous
mapped features with the same tracked center. Both read out CURRENT observables,
so information lost to averaging is visible. No future observation is used.

`physical_rank_imbalance`: for each query, rank (1=nearest) of its nearest
representation neighbor in group-physics space, averaged and multiplied by 2/N.
Distances/ranks exclude self and are computed separately within each context.
Smaller is better, ideal 2/N; random neighbors are approximately 1. Ties use
average physical ranks. `physical_neighbor_recall` is the intersection of directed
top-8 neighbor lists divided by 8, averaged over queries; higher is better.
These are teacher-neighborhood measures, not independent physical discoveries.

`temporal_change_over_local_variance` is mean squared Euclidean change at the
stated physical lag divided by total within-context TRAIN snapshot variance.
The same denominator is used for the two-frame average. Each source is a separate
row. Lags 0.75/1.5/3/6 ps are measurements, not forecast objectives.
`spatial_change_over_local_variance` uses the actual producer's same-frame
neighbor-center views and the same denominator. `crossing_1e4_over_075ps` compares
mean squared change across the retained 1e-4 A boundary perturbations with natural
0.75 ps changes. This last value pools the existing held-out crossing cohort and
is repeated on source rows; it does not provide a source-specific interval.

## State discovery and uncertainty

HDBSCAN fits in each map's full coordinates, with minimum cluster sizes 40 and
100 and minimum samples 10. No fixed number of states, noise relabeling, or
temporal label smoothing is applied. Models can find no supported states.
`states`/`discovered_states` count density clusters, not demonstrated phases.
`assigned_fraction` counts non-noise labels. `mean_strength` includes zeros for
rejected points and is density membership strength, not calibrated phase certainty.
Soft memberships are retained separately from hard labels; argmax never turns a
rejected point into a definite state. `unassigned_mass` is 1 minus their sum.
`ambiguity` is entropy of memberships conditional on belonging to some state,
divided by log(number of states); for zero/one states it is zero. Interpret it
jointly with unassigned mass. `margin` is the top-two membership difference.
None of these quantities establish thermodynamic state probabilities.

Agreement uses only pairs whose endpoints are both assigned. `same_label` is
their matching fraction. Chance agreement is the dot product of the two endpoint
label-frequency vectors. `adjusted_agreement`=(same-chance)/(1-chance), undefined
when there is only one effective label. `assigned_pair_fraction` always reports
coverage, preventing all-noise or one-state solutions from claiming coherence.
`assigned_ari` compares common assigned points between complete and 80%-source
subsampled catalogs (three repeats), and is undefined unless both have multiple
states. `assigned_overlap` accompanies it. These small sensitivity runs are not
calibrated confidence intervals. Subsampling does not refit the representation.

Static discovery is a separate descriptive protocol: fit only 3,000 PTM Other
centers in Al 166 ps with x<85 A; evaluate PTM Other with x>185 A. The 100 A gap
exceeds twice the encoder's 17 A support. PTM Other is a disordered proxy, not a
proof of liquid. Apply the same catalog unchanged to the other five static frames.
Approximate prediction cannot discover a new state absent from this first-frame
catalog; later rejection is not evidence of an encoder defect.
Static uncertainty is sampled at 1,000 matched physical-label centers per frame;
hard labels and strengths cover every analysis center. Spatial agreement uses
the six nearest analysis centers with both endpoints in the stated region.
`explained_variance` is between-cluster physical variance divided by total
physical variance among assigned sampled points, undefined with fewer than two
states. Always inspect assignment/sample coverage alongside it.
Static catalog sensitivity subsamples 80% of occupied 20 A spatial tiles in the
training slab, three times. Tiles are correlated and are sensitivity units, not
independent physical replicates. The same static trajectory is used throughout.

Sources: [HDBSCAN soft memberships](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html),
[fixed-catalog prediction](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html).
The broader method rationale is in the September 15 local-clustering literature review.


Implementation note (2026-09-15): the shared context graph now retains original
input indices for aligned velocity fields and optionally exposes pre-pooling atom
features. Default structural pooling and all metrics above are unchanged; the
velocity extension is a separate `mace_local_phase_space_v1` protocol. Historical
exported definitions and implementation fingerprints remain untouched.

The shared dispatcher also exposes the separate `motion-*` stages; this does not change this historical protocol's calculations.
