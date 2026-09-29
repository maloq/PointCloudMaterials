# CDV-MACE128 joint snapshot localization

One observed snapshot, 25 shared MACE patches, scalar 128 and vector 16-channel
exports, two vector-message context blocks. All encoder/predictor weights train.
No history, velocity, time/temperature, species or material identity is an input.
The query is an ordinary encoded patch; shared patch heads predict the query's
target, mixed by invariant learned weights. The context covers 12 atoms in
(4,14] and 12 in (14,24] Al-equivalent Angstrom by distance-only farthest coverage.
Each patch uses nearest-80 atoms and radius 8; maximum input reach is 32 A.
Finite covering/candidate changes remain discontinuous and are included in the
full-cell noise/reselection diagnostic. The sampler resolves exact ties by atom
identity; it does not consume crystal labels.

Targets are periodic nearest-atom distances/vectors to the existing >=64 atom,
past-confirmed 1.5-ps PTM crystal lineages. Censor distances at 64 A in likelihood;
only point-error targets are capped. Direction labels require 0<d<64 and nearest
two reference distances differing by more than 1e-5 A. Undefined/ambiguous
directions are masked, with counts retained. No future confirmation is an input.
An atom belonging to this confirmed reference set has d=0 because the nearest
reference atom is itself. This is unsigned distance to the crystal set, not
distance to its boundary or depth inside it. The zero-mass likelihood trains on
such rows; their direction loss is masked, not fitted to an arbitrary zero axis.

Distance density is a 25-component zero-inflated lognormal mixture. Its scalar
parameters, bounds, zero mass and censoring match spatial_distance.model. Each
component has a spherical von Mises-Fisher factor with natural parameter
eta = 8/(1+(d/16)^2) * v/sqrt(dot(v,v)+1e-8). The limiting zero vector is uniform;
no fixed Cartesian axis is substituted. Its normalized log density is
eta.dot(u)-log(4*pi)-log(sinh(norm(eta))/norm(eta)). The conditional direction
likelihood uses component responsibilities conditioned on the true observed
distance. This defines a normalized joint radial/directional density. Direction
is marginalized for zero/censored/ambiguous targets. Report distance marginal NLL
separately, so the distance-only control is comparable.

Training/selection predictive objective is distance NLL plus direction conditional
NLL (except the distance-only arm), plus twice the Bernoulli proximity log losses
at 8/12/20/32 A with weights .05/.15/.4/.4. VCReg is excluded from selection.
Direction error decreases with TRUE distance through concentration; predicting
a farther distance cannot turn off the direction term.

As of 28 September, target population is half fixed-at-risk and half uniform
centers, equal source mass within each half. Each of 256 batch entries is drawn
independently from that population with replacement. No distance labels enter
sampling; no per-batch quotas, rejection, resampling of difficult batches or
inverse sampling weights are used. Every sampled row has unit loss weight.
The existing mixture/source weighting describes which population is sampled;
it is not outcome-dependent balancing. Training mass at 0<d<=8 is 0.1230489158:
expected batch count 31.5005, with binomial empty probability (1-p)^256 ~=2.52e-15.
Mass at 0<d<=20 is 0.2128322565, expected count 54.4851. These are expectations,
not guarantees. Batch distance counts are logged as observations, not constraints.
The former fixed-quota runs retain their captured code/config/metric definitions
under al64-balanced-20260927 and were stopped when this sampling change was requested.
The fresh matched comparison is al64-random-20260928. Sampling is with replacement;
one nominal epoch is ceil(63251/256)=248 updates, not one visit to every row.
Validation is a complete unsampled pass with natural mixture/source weights.
All original fixed evaluation rows and scan query atoms remain unchanged.
The context sampler changed, so historical results are references, not an
isolated direction/VCReg ablation. All three new arms share the new sampler.

VCReg is applied to the actual normalized scalar and vector exports, not a
discarded projector. Training-only fixed scales use 4096 unique patches from
256 sampled training contexts. Scalar covariance uses ordinary centered moments;
vector channel covariance contracts all three components and divides by three.
Each equally weighted sampled query contributes 25 patches to the moments;
their correlation is not interpreted as independent evidence. Variance coefficient
.05, off-diagonal covariance .01, std floor 1, epsilon 1e-4, warmup 512 updates.
This regularizer is invariant under common rotations; independent random
rotations are not used to manufacture training variance. No invariance term,
physical reconstruction, AP loss or AP selection is used.

Distance tables use equal-source scores within each explicitly named population:
marginal NLL, capped-mean RMSE, capped-median MAE, and Brier at 4/8/12/20/32 A.
Directional point estimate is normalized mixture-weighted mean component direction
(using predicted mixture weights, never target-conditioned weights). Undefined
resultants are counted; angular error and <=30 degree accuracy exclude them and
undefined targets. Report eligible, valid-target and defined-prediction counts.
Angular bins are (0,8],(8,16],(16,32],(32,64]; d=64 itself is censored/masked.
Direction NLL in those tables is the conditional spherical NLL on defined rows.

Reliability gives precision, coverage and mean probability at strict >.5/.75/.95
for the same five radii, separately by confirmed-reference visibility. Visibility
is label-side audit data, never a prediction input. Scan alarms require two
successive positions; report misses, conditional median warning distance,
all-path recall >=12/20 A, far-path false alarms and visible-reference counts.
These visibility fields refer to established crystal, not any instantaneous PTM
assignment; the latter was not exported in this release.

Rank uses centered empirical covariance. d95 is the smallest eigenvalue count
covering 95% variance, effective rank exp(entropy of normalized eigenvalues),
participation rank (sum eigenvalues)^2/sum eigenvalues^2. Scalar rank uses native
128-dimensional local exports and mixture-pooled context states. Vector rank
uses the 16x16 contracted channel covariance; it is not a 48-dimensional scalar
rank. Whole dataset here means the entire explicitly named train or fixed-test
population, not all materials. Coordinate orientation can affect ordinary
flattened tensor rank, so it is not the vector statistic reported.

Physical-information diagnostics fit a fixed ridge .01 on standardized training
embeddings and the existing 32 smooth geometry targets (24 radial counts, two
weighted counts, six bond powers), then report training-standardized physical
MSE on fixed held-out test rows with source weighting. They never train MACE or
select checkpoints. Local and context exported states are evaluated separately.

Noise diagnostics perturb each complete periodic cell consistently and reselect
context/neighbors. Three-dimensional RMS noise is fraction .001/.005/.01/.03 of
the mean nearest-neighbor distance across the query's nearest-80 patch; fractions, actual distances and row IDs
are recorded. Response is sqrt(mean squared embedding change / (2 trace(test
embedding covariance))). Temporal diagnostic uses independent snapshot encodings
of the tracked atom one raw native frame apart (verified source timeline .75 ps).
Movement d95/ranks use centered increment covariance; RMS includes mean movement.
BF16 scalar/vector rotation errors are reported separately. There are no time
inputs or temporal smoothness losses. One seed; no significance claim.


## Explicit task-head refactor

The training/model refactor separates typed patch and spatial-context trunks from
task heads and expands training statements. Mathematical objectives, populations,
weights and selectors retain their definitions. Joint/rich-patch initialization
and state names are preserved; distance/control fresh initialization changes
when unused head construction is removed and receives a versioned architecture
identity. Historical continuations use their frozen sources. W&B wall-time stays
local and fixed baselines stay in summary; metric calculations are unchanged.
See [implementation and compatibility evidence](../code_cleanup_implementation.md#training-and-model-follow-up).
