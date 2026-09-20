# GATr information conditional on radial structure, version 1

Frozen Al GATr checkpoint 1216, SHA256
`3534c3cd6065d41ca32d6ed3e77534a79429d80ad3e195c0b14beb8ce36e42eb`.
Native exported z128, no projector or internal-vector replacement. A100/node07,
conda pointnet-torch214. No encoder training. Existing MACE, SOAP and TDA states
are comparators from the same hash-verified trajectory-stability observations.

## Population and provenance

Ten prior encoder-held-out Al MEAM sources, two at each 400/450/500/510/520 K;
four atom identities/source, all 801 frames at 0.75 ps spacing, 32,040 rows.
The parent audit checked encoder training/selection ancestry. These sources were
already explored: this is a descriptive follow-up, not a blind confirmatory test.
Probes train on other members of this held-out cohort, never their own test source.
All source observation, embedding and prediction files have verified SHA256s.
Full-box positions were stored as float16 and the cached local charts are float32.

## Radial controls and counterfactual

The radial vector contains: geometry_packet channels 0:32 (the 7 Å normalized
radial basis), 80 sorted noncentral neighbor radii, 33 equally spaced quantiles
including endpoints of all noncentral radii in native support, five support
statistics (neighbor count, native tapered count, weighted first/second/third
radius moments), and order channels 6:8 (12th-neighbor density and smooth
coordination). Neighbor counts are not density-independent angular features.
Append known temperature and current elapsed time. Do **not** use geometry_packet
32:64: those are inter-neighbor distances and already contain angular information.

R* appends a radial-only GATr z128. For n noncentral radii sorted ascending,
rank k=0..n−1 has z_k=1−2(k+0.5)/n, phi_k=k*pi*(3−sqrt(5)), and the unit direction
(sqrt(1−z_k²) cos(phi_k), sqrt(1−z_k²) sin(phi_k), z_k). Place r_k times this
direction plus the center at zero and apply native observation normalization,
weights and the frozen encoder. This state depends only on the radius multiset
and fixed Al species/scale. The synthetic arrangement is not a physical sample.
Float32 recasting produces ~1e-6 Å maximum radius differences, recorded per source.
Native original-input parity is checked against the stored z on 16 rows/source.

Feature sets: R; R*; R* plus original GATr, angular difference, MACE, SOAP or TDA.
Angular difference is original GATr z minus radial-only GATr z, calculated in
float64. This is not an internal multivector and is not assumed radial-independent:
angular responses may depend on radii. Given the radial-only state, the difference
and original state contain the same information but are differently conditioned
for finite ridge/RFF probes. Future-only controls additionally append the six
current bond-order and 16 angular targets; compare these with and without GATr's
angular difference. Current-order features are never used to predict themselves.

## Outcomes

Structure targets are the six native bond-order channels q4,q6,w4,w6,qbar6 and
mean q6 coherence, and geometry_packet channels 64:80: the pair-weighted Legendre
moments L=1..16 within the 5–7 Å tapered neighborhood. These are rotation-invariant
angular arrangements, not global orientation. Bond order uses the repository's
center-and-neighbors nearest-12 producer. Different descriptor supports remain
part of this comparison. qbar6 and coherence use neighbors' own environments.

Future onset reuses first_sustained_onset and risk_windows from
src/research/forecast_crystallization/local_metrics.py. Crystalline means PTM in
{FCC=1,HCP=2,BCC=3}. Onset is the first frame of eight consecutive crystalline
frames. Eight frame indices span 5.25 ps; the confirmation requires seven future
frames beyond the onset. The sentinel 801 denotes no observed sustained onset.
An at-risk anchor precedes this first onset and is noncrystalline at the current
and previous two frames. Anchors start at frame 2 and end at 665, giving complete
96 ps follow-up plus onset confirmation; the same population is used for all
24/48/96 ps horizons. Label is 1 iff onset-anchor is positive and at most the
horizon in frames. This is first realized local crystallization, not an
iso-configurational probability, cell-wide phase fraction or post-onset persistence.
Eligibility does not use current future-derived structural descriptors as inputs.

## Nested probes

Leave one entire simulation source out (10 outer folds). Fit on the other nine
sources, using every fourth frame (3 ps), with all four atom tracks; score all
eligible test frames. Source membership is checked at both inner and outer splits.
Three GroupKFold inner splits select a separate ridge penalty for each target
from {1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1}. Hyperparameter selection uses only the
outer-training sources. The cohort is not split by rows or overlapping windows.

Every fit weights training sources equally and rows equally within source,
with weights summing to one. Compute input means/variances on those rows in
float64, remove variance <=1e-24 dimensions, standardize retained coordinates,
and divide by sqrt(retained dimension). Exception: TDA's retained coordinates
share the square root of the mean training variance over its 144-coordinate
block. This avoids amplifying nearly absent persistence-image tail bins. An
initial coordinate-standardized TDA diagnostic produced severe held-out
extrapolation and is preserved in technical/initial-sparse-matching; it is not
used for final TDA comparisons. All GATr, radial, MACE and SOAP fits retain the
original preprocessing. Structure targets are centered and
scaled by their own training-source-weighted SD. Future binary targets are
centered with unit scale. The nonlinear map appends 256 cosine random Fourier
features sqrt(2/256)*cos(X Omega+b), with fixed length scales .5,1,2 cycling across
features, normal frequencies divided by length scale and uniform phases 0..2pi.
Seeds are config seed + test-source ID; each inner fit uses the same seed and
fits preprocessing afresh. These are finite approximate nonlinear probes, not
an exhaustive decoder search or a claim of optimal extractable information.

Center the complete feature map by its weighted training mean. Solve
argmin_beta sum_i w_i ||Y_i−Phi_i beta||² + lambda ||beta||², with an unpenalized
intercept handled by centering. GPU float64 symmetric eigensolve evaluates the
penalty path; a significantly negative Gram eigenvalue is fatal. Inner criterion
is source-weighted MSE in inner-training target units, averaged equally across
the three validation splits. Binary outputs are clipped to [0,1] for validation
and test. They are least-squares probability probes, not logistic/calibrated
hazard heads; Brier error is the primary prospective score. No future labels
enter embedding extraction or feature scaling.

## Scores and intervals

For each source, structure loss is mean squared prediction error divided by
the corresponding outer-training target variance. Group scores average target
channels equally: bond_order across six, angular_arrangement across sixteen.
constant_loss uses the outer-training target mean. r2_vs_training_mean is
1−loss/constant_loss, a baseline-relative skill score, not conventional test-mean
R². Future loss is Brier MSE; supplementary source AUROC requires both classes
and average precision requires positives. Missing quantities are blank, never 0.

Conditional gain is 100*(1−mean_sources(candidate loss)/mean_sources(baseline
loss)). Primary baseline is R*, with R→R* and current-order→current-order+GATr
comparisons separately exported. Positive means reduced prediction error.
An additional input-dimension/weighting control duplicates the radial-only
GATr state: R-star plus its own last 128 coordinates again. Compare original
GATr and angular difference with this duplicate, not only with R-star, because
redundant inputs change the ridge prior and random-feature kernel geometry.
For future prediction, a current-order-plus-radial-duplicate control provides
the corresponding comparison after current bond/angular conditioning. These
controls add no angular information and use the same nested selection procedure.
95% percentile intervals use 2,000 paired whole-source draws stratified by
temperature, holding predictions and fits fixed. They describe test-source
variation, not total uncertainty including refitting or representation seeds.

## Natural matched pairs

Enumerate all six unordered pairs of the four centers in each source/frame:
48,060 candidate pairs. Source, temperature, elapsed time and box are identical
within a pair. Accept if RMS difference of sorted first-80 radii AND RMS
difference of full-support 33 quantiles are both <=caliper, and relative density
gap |rho_A−rho_B|/((rho_A+rho_B)/2) <=.02. Primary caliper .05 Å; sensitivity
calipers .025 and .1 Å. Outcomes and embeddings do not select matches.
Input identity and timeline are retained; source-level intervals account for
reuse of centers and overlapping times. This is approximate radial matching,
not equality of every radius. Export coverage and balance by source/caliper.

Score the predicted target contrast (prediction_A−prediction_B) against actual
A−B, in the outer source's training target units. Then compare source losses
and bootstrap exactly as above. Future pairs require both endpoints at risk.
For discordant binary outcomes, concordance is fraction whose predicted
contrast has the correct sign, with exact ties worth .5; export counts and
source values. No outcome-discordant pairs means undefined concordance. If a
source has no matched pairs, it is omitted and available source count is explicit.
Bootstrap temperature strata are then restricted to the available sources.
When every available temperature stratum has only one source, no uncertainty
interval is reported; the bootstrap would otherwise give a misleading zero
width interval. Balance tables include zero-match sources and all candidate pairs.

## Dense spatial extension

Because the four-track sample yielded only three 0.05 Å matches, an explicit
follow-up uses the 70 already-selected spatial snapshots from the geometric
audit: 22,381 centers, two dense 128-atom patches plus 64 uniform centers per
snapshot, deduplicated. Same ten test sources and fixed seven frames/source.
This is an exploratory extension chosen for matching coverage, not an outcome-
selected subset. All unordered same-source/same-frame center pairs are tested
with the same .025/.05/.1 Å and 2% density calipers. Pair losses pool accepted
pairs within source, then weight sources equally. Dense patches can overlap;
only simulation sources are independent replicates.

Reconstruct native periodic neighborhoods and verify original z on 16 rows per
snapshot. Recompute radial controls, geometry_packet targets and SOAP; reuse
verified bond-order/PTM from the source producer. Fit each source's probe on
the original trajectory rows from the other nine sources using exactly the
already-selected hyperparameters and random-feature seed. Spatial labels do
not tune readouts or normalization. Evaluate R-star, plus original GATr, plus
angular difference, and plus SOAP. Source scores and matched contrast gains
use the same equations and source bootstrap. The original longitudinal future
test remains separate; no future outcome is imputed for spatial-only centers.

The displayed example deliberately maximizes absolute q6 difference among
primary-caliper accepted pairs. It illustrates existence and is not a random
sample or an additional statistical test. Aggregate results include all matches.

## Interpretation

A positive gain supports decodable information beyond these specific controls;
it does not prove causal angular influence on crystallization. An absent gain
does not prove absence of information. Richer radial models could reduce gains;
the radialized control is out of distribution; angular-difference scaling may
expose small responses. No comparison here demonstrates that the proposed new
equivariant head helps: that head has not been trained. The old vector-direction
audit measures a different property from information in invariant z128.

## Local structural support (v10)

Current structural MACE/GATr observations use fixed material normalization
`x_model = x_A * 9.192189 / scale_material`, crop to radius <8 before packing,
and quintic C2 weights equal to one through radius 6 and zero at radius 8. There
is no outer halo. MACE uses 5-unit edges, two layers and pooling tapers 0–3,
3–5, 6–8; GATr globally attends only within the cropped sphere and scales its
weighted count by 100. Training, static inference and trajectory inference share
`src/data/structural_pretraining/support.py`. Geometry baselines using the
encoder's support and radial controls now also use that local support. Existing
85-component physical and 80-point instantaneous-TDA targets are unchanged.

The revision is incompatible with previous large-support checkpoints. Historical
exported metric contracts and results retain their original support definitions;
reproduction of those runs requires their frozen code. Current within-domain
VICReg, selection, bond-order and temporal-only curvature metric formulas are
unchanged. Curvature weights are recalibrated at initialization using training
batches under the declared 2%-loss / 10%-encoder-gradient policy. See
[local protocol](../shared_pretraining_local_structure_20260918.md).
