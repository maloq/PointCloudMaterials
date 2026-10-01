# Predictive future-statistic baseline, version 1

The recipe is `configs/predictive_baseline/al480_20261001.json`. This is a
value-only simulation-supervised local representation study, not physical
reconstruction pretraining, an active-learning comparison or a trajectory
likelihood model. No AP-based objective, selector, ranking or ensemble weighting.

## Population and observation

The sealed historical Al480 assay has 40 full-cell parents, 20 source trajectories
and 12 independently seeded Langevin continuations per parent. Its 7,661 local
observations share those continuations. All rows and original source roles remain:
4,224/1,152/2,285 rows and 11/3/6 sources for train/selection/historical test.
This is not the Al64 all64/legacy16 window benchmark; Al64 is the frozen controls'
pretraining ancestry. Historical held-out sources have already been studied.

The model sees the center-relative, minimum-image nearest 80 observed atoms,
masked at 8 Angstrom, with 5 Angstrom edges and no halo. Al's fixed material length
multiplier is one. Atom attributes are one constant channel plus the existing
geometric center indicator. Temperature, absolute time, simulation age, species,
material ID, velocity, history and relaxed inputs do not enter any predictor.
Temperature and strata only organize audits. Conditions differ across the pooled
400/450/500 K sources and cells; prediction concerns this local parent population,
not the full-q fixed-condition experiment in the conceptual formulation.

Existing inclusion weights are normalized to total one within each source.
Fitting uses equal source mass, preserving relative population weights inside a
source. All normalization and tuning uses training or selection sources only.
Conditional evaluation groups first normalize weights inside each source/group,
then average sources equally. Atoms and branches do not count as extra sources.

## Fixed target and feature metric

For each branch Y has nine components: crystalline fraction, qbar4 and qbar6 at
3, 6 and 12 ps, in temporal-major order. The tracked center's nearest-80 future
neighborhood is reselected at each horizon and cropped at 8 Angstrom. Fraction
uses existing full-cell PTM FCC/HCP/BCC labels; qbar uses the exact existing
patch descriptor producer. PTM is a hard value-only target; simulator derivatives
are not used. Y is standardized with source-weighted training-shot means and
standard deviations (floor 1e-5).

phi(Y) concatenates normalized Y, its coordinatewise square and 256 Gaussian
random Fourier features sqrt(2/256) cos(Y_normalized omega + phase). The bandwidth
is the median pairwise distance among 1,024 training shots drawn with replacement
using source/population weights. Omega ~ N(0,1)/bandwidth; phases ~ Uniform(0,2pi).
The seed, selected training-shot indices and complete map are frozen for all arms.
No model trains or adapts phi. This finite bank summarizes the distribution of Y;
it does not identify arbitrary trajectory laws.

For each of the three blocks, compute V_block = sum of its training-shot marginal
feature variances. Every coordinate in that block has fixed metric weight
(1/3)/V_block. The stored feature is t = (phi - training_phi_mean)*sqrt(metric).
Thus each centered block contributes 1/3 of the training-shot variation; feature
count alone does not give the RFF block disproportionate weight. The target is
the average of t over all 12 shots, never t applied to the mean future.

## Models and selection

The joint model is native geometry-only MACE, channels 128, two interactions,
ell<=2, correlation 2, existing train-normalized center/pooled export to z128,
and a 128-hidden SiLU MLP predicting 274 centered/scaled feature coordinates.
It starts from random weights. Initial pooling normalization uses 1,024
source-weighted training draws with the target seed. Both encoder and head learn.
There is no auxiliary reconstruction, latent-variance or covariance loss.

Three declared seeds use full batch/microbatch 256, float32, AdamW with encoder
LR 1e-4, head LR 5e-4, weight decay 1e-5 and gradient norm cap 10. An epoch is
ceil(4224/256)=17 full weighted-resampling updates, not an exhaustive pass over
all distinct rows. Objective = mean sum-coordinate squared feature error. It is
equivalent to a fixed-unit-variance Gaussian feature NLL:
0.5*(feature_error + 274*log(2pi)). This is a working feature likelihood, not a
normalized density over physical future trajectories. Lowest source-weighted
selection NLL selects the checkpoint, including the initialized epoch-zero model.
Maximum 200 epochs; stop after 25 consecutive epochs without improvement.
Latest optimizer/model/RNG state and best checkpoint are separate and resumable.

Controls: source-weighted training prior; standardized 442 present geometric
descriptors with ridge; recorded frozen VICReg and Epi 128-exports with ridge.
The ridge intercept is unpenalized; alpha in .001/.01/.1/1/10 minimizes selection
feature error. Descriptor controls and frozen probes stay local. Three joint
encoder fits are the only new online W&B runs.

## Scores, uncertainty and readouts

feature_error = ||prediction - shot_mean||^2 in the frozen metric. Block errors
split that sum into first moments, second moments and RFFs. shot_noise is
sum_coordinate unbiased_sample_variance(t)/12. corrected_feature_error subtracts
shot_noise from feature_error. This equals the average cross-error dot product
over all distinct ordered shot pairs; for a prediction independent of held-out
shots it is unbiased for squared error to the conditional expectation. Keep
negative estimates. Training and selection scores are not reported as unbiased
post-selection estimates.

Comparison averages individual seed errors, not predictions or fitted ensemble
weights. Paired 95% percentile intervals for error minus prior resample whole
held-out sources 2,000 times after averaging fit seeds. Seed SD is population SD
over the three source-averaged fit scores, reported separately. Main test groups
have six sources; temperature groups have only two. Intervals do not capture
selection/target-design uncertainty or justify prospective generalization.

Invert the fixed feature transform to recover physical mean and second moment.
Predicted variance = scale_Y^2*(predicted_E[Y_normalized^2] -
predicted_E[Y_normalized]^2). Do not clip negative predicted variances; report
their fraction. Physical mean MSE compares to the 12-shot physical mean. Variance
MSE compares to unbiased 12-shot sample variance. Those descriptive moment errors
retain label noise. The head is unconstrained and need not represent a valid law.

Freeze each selected z and fit a fresh ridge readout of the same 274-coordinate
target; this assesses linear accessibility separately from the trained head.
Additional local probes predict 18 branch-mean observables excluded from training:
q6 coherence, q6 neighbor dispersion and four TDA measurements at 3/6/12 ps.
Standardize those targets using training branch means; minimize selection squared
error (equivalent fixed Gaussian likelihood). Probe MSE averages 18 coordinates
then sources. These narrow extra questions do not establish universal sufficiency.

After selection, independent Gaussian noise is applied to every observed atom at
sigma .001/.01/.05 Angstrom, then center displacement is subtracted. Neighbors
are not reselected; support/edges are recomputed. noise-response reports weighted
test squared feature-prediction change, not accuracy under a physically simulated
perturbation. The execution gate also checks rotation invariance and a nonzero,
finite encoder gradient; no gate or hardware benchmark creates an online run.

Learning curves, source scores, physical moments, readout selectors and final
comparisons retain this definition and exact implementation hashes. Saved arrays
retain parent and atom IDs. Final metrics update existing training run IDs through
the W&B API without creating evaluation runs.
