# Paired native-checkpoint conditional information, version 1

This repeats the conditional-information assay with explicitly pinned final
MACE and GATr checkpoints. Its config is
`configs/analysis/conditional_information_local_last.json`. The run freezes
both `last.pt` files at update 622; their separate validation-selected exports
are update 576 and are not substituted. No encoder training occurs. Execution
uses the user-approved H100 on nodesumo01 in conda pointnet-torch214.

## Populations, extraction and ancestry

The original trajectory-stability plan supplies ten previously explored Al
test sources, two each at 400/450/500/510/520 K, four tracked centers/source,
801 frames at 0.75 ps: 32,040 test rows. Five original training-reference
sources contribute 420 sparse observations only for stability normalization.
The original equivariant audit supplies 70 fixed spatial panels, 22,381 centers.
The larger old neighborhood charts and their cached descriptors are retained.
Source/descriptor identities are hash-verified. Model training and selection
lineages are checked against test source ancestries separately for each model.
Probe training uses other test-cohort sources; never the source being scored.

Each model executes in a subprocess importing its own frozen training checkout.
Every checkpoint-recorded implementation file is verified there before inference;
the actual imported model path is also checked. Final weights are extracted by
removing `encoder.` from the training-state keys, without parameter conversion,
and loaded strictly. Material scales come from the verified training manifest.
All new models use fixed x_model=x_A*9.192189/scale_Al, scale_Al=9.121389139452193,
strict radius <8, full weight through 6 and a quintic 6–8 taper. MACE uses native
5-unit edges; each backbone retains native compiled BF16 with FP32 geometry and
export. Repeat, batch-order and proper-rotation tests run on actual observations.
The checkpoint's default z128 forward is used; no projector or bond-order head
is exported. Parent coordinates originate from verified float16 storage; local
offsets and packed positions are float32, probe arithmetic float64.

## Common radial control and feature sets

Retain the old assay's 152 radial/density coordinates: 32 radial basis values,
first 80 noncentral radii, 33 quantiles across old native support, five radial
support statistics, density_r12 and smooth coordination. Append 33 quantiles
and five support statistics for the new radius-8 input: neighbor count, tapered
count, and tapered mean r/r²/r³. No angular target enters these features. Append
known temperature and elapsed time. Thus the scalar radial vector has 192
coordinates including context.

For each original input, sort all noncentral radii inside the new support. At
rank k=0..n-1 place r_k along the deterministic Fibonacci direction with
z_k=1-2(k+0.5)/n and phi_k=k*pi*(3-sqrt(5)), plus the center at zero. The
reconstructed radius error is recorded; support count equality is required.
Both checkpoints encode this same radius-only construction independently.
The common R-star baseline appends both radial-only z128 states, totaling 448
coordinates. It is identical for MACE and GATr comparisons. These counterfactual
geometries are synthetic and not assumed physically distributed.

Feature sets add one original state, its float64 difference from its own radial
state, or another copy of its radial state. Original and difference are
invertible reparameterizations given the radial state but have different finite
ridge priors/kernels. The duplication adds no information and matches added
dimensions, exposing redundant-input regularization effects. Historical v6
GATr/MACE exports, SOAP252 and instantaneous TDA144 are extra trajectory
comparators under the same new common control. Spatial probes use the common
baseline, both current models with their difference/duplication controls, and
SOAP; no old MACE spatial export or spatial TDA is imputed.

Prospective variants first add current bond6 and angular16. Each original state,
its angular difference and its radial duplicate are separately added on top.
Current structural targets never predict themselves in the structural assay.

## Targets and readouts

Targets, eligibility and fitting equations are reused from the prior assay:
six bond-order channels q4,q6,w4,w6,qbar6,q6 coherence, and sixteen weighted
Legendre angular moments. Targets retain their original producer/support.
Future onset is the first of eight consecutive PTM FCC/HCP/BCC frames, with
current and previous two frames noncrystalline and before that first onset.
Anchors 2..665 ensure all 24/48/96 ps horizons plus confirmation are observable.
There are 17,674 eligible rows. Each trajectory supplies one realized future;
this is neither an iso-configurational probability nor a causal outcome.

Ten leave-one-source-out folds; fit every fourth frame of the other nine
sources. Three inner GroupKFold splits select a separate ridge penalty per
target from 1e-6,1e-5,1e-4,1e-3,.01,.1,1. Row weights give each fit source equal
total weight. All means, variances, feature removal (variance <=1e-24) and target
scaling are fitted only on training rows. Retained input coordinates are divided
by their training SD and sqrt(retained dimension). TDA's last 144 coordinates
share sqrt(mean training variance of that block), avoiding rare-tail inflation.
Structure targets use training SD; binary outcomes are centered with unit SD.

Nonlinear probes append 256 cosine RFFs sqrt(2/256)*cos(X Omega+b), with cycling
length scales .5,1,2, Gaussian frequencies divided by scale, uniform phases and
seed=config_seed+test_source_ID. Features are centered by weighted training mean.
Float64 GPU eigensolves minimize sum_i w_i||y_i-Phi_i beta||²+lambda||beta||²;
the intercept is unpenalized through centering. Inner selection minimizes
source-weighted MSE in inner-training target units, averaging three splits.
Binary predictions are clipped to [0,1]. Spatial targets never tune probes:
fit the original trajectory training rows with their existing selected penalties
and RFF seed, then evaluate spatial rows of the held-out source.

## Scores, matching and uncertainty

`source_scores`: MSE divided by outer-training target variance, averaging group
channels equally (bond6 or angular16). `constant_loss` uses the outer-training
target mean; `r2_vs_training_mean` is 1-loss/constant_loss, not test-mean R².
Future loss is Brier error; AUROC requires both classes and AP requires positives.
Undefined metrics are blank. `conditional_gains` compare equal-source mean
losses: improvement_percent=100*(1-mean(candidate)/mean(baseline)). Positive is
better. The same formula compares old and new representations under common
controls. These comparisons do not isolate architecture, supervision, support,
precision or data changes between generations.

Same-source, same-frame pairs require both first-80 radius RMS difference and
old full-support 33-quantile RMS difference <=caliper, and relative density
gap |rhoA-rhoB|/mean(rhoA,rhoB)<=.02. Matching uses the exact previous values,
not the new smaller input support; no embedding, angular target or future
label selects pairs. Calipers .025/.05/.1 Å remain unchanged. The 0.05 Å
longitudinal population has only three pairs; the dense spatial population has
609 pairs across nine sources. At .1 Å spatial coverage is 125,006 pairs across
ten sources. At .025 Å there are no spatial pairs. Exact coverage is asserted.

`matched_scores` evaluates prediction_A-prediction_B versus true_A-true_B in
outer-training target units. Pair losses are averaged within source, then
sources equally. Future pairs need both endpoints at risk. Discordant outcome
concordance uses the correct contrast sign, with ties 0.5. Dense patches and
overlapping pairs are dependent; only source replicates enter uncertainty.

Intervals use 2,000 paired whole-source bootstrap draws within temperature,
holding fitted predictions fixed. They exclude refitting uncertainty and are
not multiple-comparison adjusted. If all represented temperature strata contain
only one source, no interval is reported. Sparse strict spatial matching has
one unavailable source; its remaining singleton temperature stratum is fixed
under bootstrap. Bounds do not establish exhaustive information absence.

## Trajectory stability

Reuse `trajectory_stability.metrics` on time×atom×feature arrays. Each method's
reference mean/covariance is fitted to the same 420 observations from five
training sources. Let trace=sum_j variance_j. Normalized squared jump is
||z(t+1)-z(t)||²/(2*trace). Aggregate RMS is sqrt(mean of ten source means);
its interval uses the paired temperature-stratified source draws. Lag
displacements use the same equation at 1,2,4,8,16,32,64,128,256 frame lags.
Effective rank is trace²/sum(eigenvalue²) on the reference covariance.

Roughness is mean squared second difference divided by mean squared energy of
the two corresponding first increments. Increment cosine and reversal fraction
use nonzero successive increments; reversal means negative cosine. Zero
increments are separately counted. Raw increment MSE, normalized acceleration,
standardized jump and jump p95 retain the prior producer formulas. Summary
roughness, reversal and cosine weight sources equally. SOAP/TDA and historical
v6 exports use identical observed rows and their own reference covariance.

Smoothness alone does not establish useful structure or prediction. Structural
and future tables must be interpreted with the same population, baseline and
probe definition, including absolute forecasting performance against prevalence.


Table export: 2026-09-19T00:05:08.821566+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
