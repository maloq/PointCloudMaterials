# Forecast-encoder diagnostics

The checkpoint is the frozen 256-channel MACE encoder **before** its VICReg
projector, restored by `load_snapshot_encoder`. It has no TDA head and was trained
without TDA supervision. Readout accuracy tests information in this representation,
not an existing direct TDA predictor. All results concern this single checkpoint.

## TDA generalization

Sample 64 centers without replacement from each of 90 existing source/frame
contexts: 18 training, 6 validation and 6 test independent melt lineages, with
three frames per source. Inputs and observed/relaxed 144-component TDA labels
are the exact decoded cache values. Labels are the repository alpha-complex
descriptor (`persistence_image`): H0 has 16 bins; H1 and H2 have 64 each. Relaxed
targets come from the recorded full-cell FIRE quench; observed targets describe
the hot patch. They are distinct scientific targets. The test sources have been
examined in earlier studies; this is exploratory evaluation, not a fresh blind test.

Each readout fits feature standardization and block target scales on training
rows only. Block scales use `fit_targets` with floor fraction 0.05. Ridge alpha
is selected from the configured grid by equal-source validation balanced MSE;
test rows never select it. The three homology blocks receive equal weight:
`mean_d(mean_pixels((prediction-target)^2)/block_scale[d]^2)`.
Raw pixel errors within each block are averaged before averaging the blocks.
Ridge feature transformations and fitting use float64 to avoid an ill-conditioned
float32 solve at weak regularization. The same-source paired comparison table also
reports MACE gains against each declared nonconstant control.

Skill is `1 - source_mean(model_error)/source_mean(training_mean_error)`.
95% intervals resample whole held-out sources with replacement (4,000 draws).
Train/validation intervals in the table use the corresponding split's sources.
All contexts have equal sampled size, so sample and source weighting coincide.

`local_centered_skill` subtracts each source/frame's mean from predictions and
targets separately **only for scoring**, then divides centered prediction error
by centered target energy, using balanced block scaling. It measures local
variation while forgiving a frame-level intercept error. H0/H1/H2 centered R²
are the corresponding unscaled individual-block ratios. `local_uncentered_skill`
keeps full prediction error in the numerator and uses within-context variance
in the denominator, thus also penalizing frame-mean bias.

Controls: training mean; training temperature mean; ridge on the 79 center-neighbor
distances and q4/q6/density/mean nearest-shell distance; MACE ridge fitted to targets
shuffled within training contexts (preserves frame trends); and a global training
label shuffle. Null probes reuse MACE's selected alpha and have one specified RNG
realization. Reduced-data probes use 16 or 32 of the sampled centers per context,
with the full-training feature/target transforms and alpha fixed. They are limited
readout learning curves, not retraining or tuning of the MACE encoder.

## Numerical and geometric changes

Embedding differences use the exact training-only channel scales saved in the
declared completed forecast checkpoint; the normalizer and checkpoint hash are
exported. The reference signal is mean squared standardized change over all
adjacent 0.75 ps pairs in 144 tracked patches (6 held-out sources × 3 contexts ×
8 centers) and 17 frames per path. The table reports both error/signal MSE ratios
and their square roots. Absolute maximum errors retain the original embedding
units. Repeated, reordered, rotated, translated and permuted controls restore
output correspondence before comparison. They use the trained compensated BF16
radial arithmetic. The FP32 radial control keeps trained weights, activations,
normalization and the CUDA backend, replacing only the compensated matrix product.
Jitter has the configured expected three-dimensional RMS displacement per
noncentral atom; its center remains fixed. These are perturbations, not MD steps.

Local coordinate FP16, output embedding FP16, and global coordinate FP16 are
separate interventions. Global storage tests start from retained original float32
positions in one shooting branch. Both reselected and original-member patch
embeddings are measured after an in-memory global float16 round trip. Boxes,
IDs and timelines are unchanged. This isolates coordinate storage, not integrator
precision or irreversible recovery of already-quantized ordinary trajectories.
TDA is recomputed from both versions with the repository descriptor; PTM labels
are independently recomputed. No new trajectory is saved or substituted.

Neighbor retention excludes the common center: intersection of the two 79-ID
sets divided by 79. One-step geometric motion matches all original neighbor atom
IDs and uses center-relative minimum-image displacements in Å. Exact decomposition:
`total = (next_geometry_original_members - initial) +
(next_geometry_reselected_members - next_geometry_original_members)`.
The squared motion and membership terms plus their signed cross term sum to total
change energy; the individual ratios are not disjoint causal percentages.
Controlled boundary substitutions replace the outermost 1/2/4 members by the next
nearest outside atoms in the same frame, with no motion of other atoms. These
finite support interventions are diagnostic, not alternate physical trajectories.

## Time, physical readouts and siblings

At each configured lag, compare instantaneous patches and patches retaining the
initial frame's 80 identities. Global-centered correlation is a norm-weighted
dot-product correlation after subtracting the forecast training mean and dividing
by its scales. Track-centered correlation subtracts each 17-frame path mean;
the short window biases long-lag values and does not establish a decay constant.
TDA increment skill compares changes in the frozen training-only observed-TDA
readout to recomputed observed TDA changes, against zero-change prediction.
Relaxed-readout increment energy is descriptive: no repeated quench labels were
computed at intermediate times.

PTM uses the same 80-atom separated-patch assay as existing physical analyses.
An unrestricted best-template match gives RMSD and margin `0.1-RMSD`. A missing
template produces undefined RMSD/margin (NaN), never a perfect-fit sentinel;
its label is Other. Label changes include all centers. RMSD/margin correlations
and RMS differences use pairs with finite measurements and explicitly count them.
q4/q6 use spherical harmonics of the nearest 12 center bonds; density is
`12/(4*pi*r12^3/3)`. These are continuous structural observables, not phase labels.
Spearman correlations compare change magnitudes, not signed causal responses.

Sixteen shooting siblings share one exact initial configuration (positions, box
and IDs checked). The retained design has 8 momentum groups with 2 thermostat
streams each. Same-momentum pairs isolate thermostat-stream divergence under this
stochastic protocol; different-momentum pairs change unresolved momenta too.
Pairwise squared embedding/TDA distances and PTM disagreement are averaged over
all eligible branch pairs and 16 tracked centers at each physical time. These are
one-parent descriptive results; pairs and centers are not independent replicate
lineages. No population bootstrap is reported for this ensemble.
