# Full-cell atomistic simulator-response training

This separate mechanistic experiment uses complete periodic Al256 perturbed-FCC
configurations and the fixed MACE-MPA-0 medium potential. It is not the local
Al480/MEAM experiment or the fixed Al64 window benchmark. All configurations share
the FCC prototype. Held-out draws test this synthetic configuration distribution,
not independent melts or picosecond crystallization prediction.

## Data, observations and target

Parent indices0:32 train,32:40 select,40:56 test are fixed before simulations.
Each role balances coordinate displacement standard deviations .03/.08/.12/.16 A
by index modulo4. First16 geometries were used in numerical development and are
training only. Selection/test geometries are new. Every branch is newly generated;
no previously examined response labels are used in the new training bank.

Dynamics reuse the audited float64 periodic fixed-potential BAOAB oracle:1fs steps,
450K thermostat and100fs friction time, horizons20/100fs. The physical future
feature is exactly the original `PathFeatures`:16 smooth full-cell radial
mean/variance changes relative to the differentiated initial anchor, scaled by.1,
then128 fixed Fourier features per joint temporal prefix. There are256 target
features. The20fs block observes one time; the100fs block jointly observes20 and
100fs. There is no PTM crystalline-fraction target in this differentiable protocol.

Values use32 independent branches per parent. Train/test responses use8 branches
and2 zero-translation orthonormal coordinate directions. No response labels are
collected for selection configurations. Train AD seeds equal the first8 value
seeds, with independently executed value-only paths verifying numerical equality.
Those are intentionally coupled labels, not independent replicates. Test response
seeds use a disjoint namespace from all32 test value seeds. All parent seed ranges
are disjoint. Store retains full-precision numerical restart states and compact
value/response bundles, never quantized integration states.

The student consumes positions of all256 atoms with minimum-image periodic edges
below5A in the fixed cubic box. Radial/angular features remain differentiable;
only discrete edge/image selection is detached. One constant atom channel and
zero center marker are used. Native MACE128 has two interactions, angular2 and
correlation2, e3nn backend,128-dimensional export and128-hidden nonlinear head.
Pooling concatenates means and population variances of atomwise scalar channels.
Pool normalization uses training geometries only. There is no temperature, time,
species, motion, history, relaxation or scale covariate. The simulator's fixed
box and thermostat define the physical experiment, not predictor covariates.

## Matched fits and selector

Three paired initialization seeds cross values8, responses8, values32. All use
the same train parents, initial weights, sampler order, optimizer and architecture.
The training-feature center and coordinate standard deviations use the32x8 first
value branches; scales are floored at1e-4 and frozen across arms. Let t(q) be the
normalized student output, a the feature center and s its coordinate scale:
m(q)=a+s*t(q). The normalized value target is(mean_b Psi_b-a)/s.

All arms minimize .5*mean((t-target)^2), equivalent to fixed-unit-variance Gaussian
feature NLL up to its constant. Responses8 adds .5*mean(((J_t(q)V-Hbar/s)/r)^2),
weight1. Here r is the RMS of the training-parent branch-mean Hbar/s, floored at
1e-3 and frozen; no validation/test responses fit it. Both the encoder and head
receive gradients from the complete-predictor JVP, with create_graph=True.
Response loss regresses signed directional vectors; no squared-norm correction
or acquisition score is used for its training target.

Each update is the complete32-parent batch, accumulated in microbatches4. This
explicit deviation from256/256 bounds higher-order memory and avoids repeating
the32-parent cohort. AdamW encoder/head rates1e-4/5e-4, weight decay1e-5, gradient
cap10, maximum200 updates, patience40. Neural arithmetic is float32, oracle64.
All fits, including epoch0, select solely by the SAME32-shot selection feature
Gaussian NLL. This is a mechanism/label-budget comparison, not hyperparameter
search. Fixed geometry generation and target specification are unchanged by scores.

## Exported calculations

`learning.csv` reports epoch, mean normalized training value NLL, half response
MSE (zero when absent), selection NLL, selected epoch, pre-clipping gradient norm,
and cumulative optimization/selection time. Response half-MSE uses r above.
NLL adds .5*log(2pi) to .5*coordinate-mean squared error; it is likelihood of the
feature-mean target, not a density model for full atomic futures.

For held-out parent i, let y_b=(Psi_b-a)/s and h_b=H_b/s/r. At every scope,
value_mse=mean_coordinates((t-mean_b y_b)^2). Its corrected version subtracts
mean_coordinates(sample_var_b(y_b))/32. value_nll=.5*(value_mse+log(2pi)).
response_mse=mean_coordinates,directions((J_t V/r-mean_b h_b)^2), corrected by
subtracting mean(sample_var_b(h_b))/8. Variances use ddof1. Corrected quantities
may be negative and are not clipped. Predicted response squared is mean((J_t V/r)^2).
Oracle response squared corrected is mean((mean_b h_b)^2)-mean(sample_var(h_b))/8.
These concern the two declared directions, not the complete768-dimensional Jacobian.
Scopes are20fs(first128),100fs_joint_prefix(last128),full(all256).

`parent-errors.csv` retains arm, seed, parent, displacement stratum and scope.
`summary.csv` averages parent errors and then seeds, not predictions. Delta versus
prior compares with the training8-shot feature mean (normalized zero) and zero
response. `paired-contrasts.csv` is responses8 minus values8 or values32; negative
favors response supervision.95% percentile intervals use2000 configuration draws
resampling four test configurations within each of four displacement strata,
after seed averaging. They omit training-data, design-selection and shared-prototype
uncertainty; they do not imply independent-source physical generalization.

`oracle-cost.csv` records actual branch/screen seconds, value and AD force calls,
and HVP calls by parent. Value8 cost charges exactly8 value-only calls; value32
all32; responses8 exactly8 AD calls, which produce both values and responses.
All arms additionally charge the shared validation32 calls and train/selection
screens. `costs.csv` separates acquisition, cumulative optimization/selection,
their sum and final inference/evaluation. Common numerical-pilot costs, initial
geometry construction, I/O and W&B initialization are outside that sum. The
preflight timing and stage receipts retain those operational contexts. Fixed8/32
shot counts are not an exact equal-wall-time match. No cost-efficiency superiority
claim follows merely from endpoint score differences.

Scientific fits remain online in teshbek/PointCloudMaterials with stable IDs and
optimizer/RNG resume. Final held-out metrics update those IDs without new runs.
Numerical gates and collection do not create W&B runs. Plots show
configuration-bootstrap differences, not predictive information completeness.
