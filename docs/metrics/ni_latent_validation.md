# Ni purpose-matched representation and trajectory validation

Execution revision r2 bounds the workspace of the identical double-precision
4x4 density eigendecompositions by processing at most 1024 matrices per call.
Samples, tolerances, mathematical scores and learned parameters are unchanged.
The revised full forecasts are exported separately; original failures and
historical metric contracts remain preserved.

This family freezes `configs/ni_latent_sim/validation_20261018.json` and the `ni_latent_flow`
validation, diagnostics, kinetics and support producers. Unchanged measurement and trajectory
definitions inherit [ni_latent_flow](ni_latent_flow.md) and
[ni_kinetic_latent](ni_kinetic_latent.md). Historical exports retain their original definitions
and implementation hashes.

## Population, fitting budgets and interpretation

The expanded bank contains 21,600 fixed-central-atom windows from 90 ancestors, including the
original 14,400 windows and 7,200 additional windows from 30 saved MD histories. Its common test
population has 14,880 windows from 62 ancestors; the old reference evaluation retains 10,560
windows from 44 ancestors. Each window contains 64 actual observations at 0.1 ps spacing.
Nearest-80 neighbors are reselected around the same central atom at each observation. Four input
frames span -0.3 through 0 ps; 60 future observations span 0.1 through 6 ps.

The queue declares 42 new fits across three fixed seeds: 30 predictors trained for 32,000
updates and 12 representation-only fits trained for 48 epochs. Predictors compare raw and
scalar-support direct244 states, block flow versus four-frame MDN, and inherited frozen VAE
representations. The 12 native fits compare SelfTICA and bidirectional FMRC on `original16` and
`original28`. They fit no decoder or autonomous predictor. Another 21 original reference
checkpoints receive new method-native diagnostics without retraining.

`old_mixed16` uses 3,840 windows from 16 ancestors, including original, liquid-liquid and
growing-front regimes. The separate data-size contrast uses 80 original-regime windows per
ancestor: 1,280 windows from 16 ancestors versus 2,240 from 28. The larger set adds independent
source/environment coverage; this is not a pure increase in independent samples at a fixed
distribution. All new arms retain the same expanded heldout rows, without outcome filtering.

Only train/test roles are used; original roles and ancestry remain provenance. The original
16-source affine/asinh/affine measurement normalization stays fixed. Shared trajectory increment
scales, high-order thresholds and the measurement-space phase probe use that original mixed
training population. Support-coordinate transforms, new encoders and kinetic readouts use their
declared training subsets. No temperature, absolute simulation time, age or phase labels enter
predictors. Physical horizons organize targets or select lag-specific heads; dimensionless flow
interpolation time is algorithmic. Budgets, seeds, checkpoints, probe recipes and sampler grids
are fixed before test scoring. There is no AP objective, checkpoint selection, early stopping,
calibration fit or new nucleus-birth readout in this family.

These enriched, nonstationary NPT windows do not establish equilibrium, reversibility, detailed
balance, physical rates or a Markovian observed state. Neither a slow coordinate nor local
crystalline entry demonstrates a connected 32/64-atom nucleus, global latent nucleation or
atomistically realizable paths.

## Common forecast, ordering and phase scores

`trajectory`, `retention`, `latent_spectrum` and their applicable `_sources` tables inherit the
shared assessor. The expanded final forecast evaluation includes every test window, with 32
neural paths per origin; these are neural samples sharing one saved MD target, not independent
physical continuations. Each measurement family (BO40, TDA113, density74, velocity17) has weight
1/4, distributed uniformly over its active train-normalized columns.

For paths A,B, distance is sqrt(mean over time of sum_d w_d(A_d-B_d)^2). The energy score is
mean sample-to-observation distance minus half the mean distance between distinct sample pairs.
`path_energy` scores connected paths; `increment_energy` scores successive increments anchored
at the actual initial observation, divided by training RMS increments with floor 0.05;
`increment_correlation_energy` scores time-averaged increment products at 0.1/0.5 ps lags.
Endpoint ensemble-mean MSE, temporal/ensemble power and marginal 90% coverage remain
diagnostics. Scores are evaluated through 1,3,6 ps in the same normalized244 space, not
incomparable latent-coordinate densities.

Controls include persistence, retrospective `encoded_MD`, recursively applied MDN conditional
means and the flow's `zero_noise` representative. Retrospective decoding has future information;
a zero-noise flow path is not a conditional mean. Flow-matching loss is not a tractable NLL and
is never labeled as one.

Continuous order diagnostics retain qbar6, coherence, qbar4 and wbar6, raw MSE, within-ancestor
R2/Spearman, autocorrelation products, occupancy and dwell. Present liquid means actual central
PTM not 1/2/3 and current crystalline neighborhood fraction <=0.05. High-order liquid
additionally requires qbar6 and coherence above the original training-liquid 80th percentiles.
It is an operational diagnostic bin, not an identified phase or a productive precursor.

All new predictor arms use the same train-fitted logistic phase probe in measurement space,
applied to decoded measurements; they do not each fit a different latent phase boundary.
Observed truth remains actual PTM. The probe uses source-balanced sampling up to 60,000 original
training states, C=1 and threshold0.5. Local persistent crystal entry completes three
consecutive crystalline future frames: earliest completion0.3 ps, first-to-last span0.2 ps.
Entry/phase Brier and survival diagnostics inherit this imperfect-probe caveat. Run length
times0.1 ps is sampled occupancy, not continuous-time residence.

## Fixed-checkpoint and sampler diagnostics

`diagnostics` and `diagnostics_sources` use outcome-blind identity ranking: one window per
split/ancestor/regime, with train/test panels kept separate. Fixed dynamics snapshots are
3,200,12,800,32,000 updates. Final block samplers use 10,32,64,128 gridpoints; these report
discretization sensitivity and are never selected on test. Seeds are shared across matched
batches and controls.

Besides the inherited connected-path scores, `marginal_state_energy` averages the energy score of each
future-time marginal through the stated horizon; `marginal_state_coverage90` averages corresponding
interval coverage. MDN `teacher_forced` replaces each four-frame history with actual
observations: only marginal scores are reported, because these samples are not a connected
autonomous path. `shuffled_history` substitutes a different complete history from the same
split/ancestor/regime. `training_trajectory_reference` samples complete saved training futures,
choosing ancestors uniformly and then windows; it neither anchors nor stitches them and is not
conditional MD continuation.

Any nonfinite ensemble member makes the affected origin's predictive score undefined.
`nonfinite_path_fraction` retains all requested paths. An invalid MDN distribution records
`generation_failure_fraction` and an explicit receipt; the affected batch has undefined scores
rather than substituted forecasts. Independent checkpoint/grid/control combinations continue or
resume separately.

## Scalar supports and joint measurement validity

Support arms map bounded channels with logits and nonnegative channels with inverse softplus,
then fit coordinate mean/scale only on their training subset. Boundary handling moves observed
values inward by epsilon=1e-6 of interval width or pinned raw training standard deviation. This
is input preprocessing, not clipping generated samples. The support audit reports per-channel
adjustment counts, maximum raw adjustment and normalized round-trip error.

`validity`/`validity_sources` include generated, deterministic, saved-MD and retrospective
encoded-MD paths. Scalar conditions include bounded bond order and coherence, neighbor-q
population SD <=0.5, nonnegative TDA measurements, radial density, all l=0 density powers,
diagonal higher-order density powers, velocity second moments and squared affine rates. Signed
higher-order cross powers, w/wbar and signed rates remain unconstrained. Scaled scalar excess is
distance outside support divided by max(pinned raw channel SD,1); violation means excess>1e-5 or
a nonfinite coordinate.

`density_psd` checks each symmetric4x4 radial-power Gram matrix for l=0..6; negative minimum
eigenvalue is divided by max(matrix absolute maximum,1). `density_l0_identity` checks
G0=radial_mass outer product, using maximum absolute element error divided by max(outer-product
absolute maximum,1). `velocity_moment` checks each of three shells' second radial moment
>=mean^2, with excess divided by max(abs(second),mean^2,1). Threshold1e-5 applies to each;
nonfinite inputs violate the condition. `*_state_fraction` averages violating states;
`*_path_fraction` counts any violation in a requested path. `any_necessary_constraint_*` takes
the union of scalar and joint conditions.

Existing nonfinite and outside-training-range counters remain separate. `validity_channels`
pools all requested6-ps states per method and reports each channel's violation count/fraction
and maximum scaled excess, including infinite excess. These pooled counters are not
source-bootstrap estimates. No samples are rejected, repaired or averaged only over survivors.
Scalar supports do not enforce the joint identities, integer Betti counts, ordered quantiles or
full atomic realizability; passing these audits is necessary only.

## Method-native kinetics and common predictive readouts

`kinetics_operator` fits a train-only, two-sided linear VAMP basis at lags 0.1,0.5,1,3 ps; test
origin is always frame3. Fixed ranks2,4,8,16,32 are reported without selection. Ridge is0.001
times the largest covariance eigenvalue and relative covariance cutoff is1e-8; unavailable ranks
remain undefined. For train singular values S and heldout moments Cff,Cgg,Cfg, centered VAMP-E
is 2 trace(S Cfg)-trace(S Cff S Cgg). The constant function is omitted. Bootstrap resamples
whole-ancestor moment packets and recomputes this nonlinear score; source-local score rows
cannot simply be averaged to recover the pooled score. Regularization/cutoffs affect the score;
it is not an equilibrium timescale.

`kinetics_selftica` evaluates the actual learned linear head: for A=P phi(x), B=phi(y),
loss=mean_{i!=j}(A_i dot B_j)^2-2 mean_i(A_i dot B_i). The reported statistic uses each ancestor
as an all-pairs batch, then averages ancestors; it is explicitly not the mixed-source minibatch
training loss. `kinetics_field` reports FMRC's sum-over244 squared vector-field error using four
shared Gaussian/interpolation draws per heldout pair at0.5/1 ps, including zero-field risk.
Forward and retrospective backward conditionals are separate; the latter does not reverse
velocities or establish reversibility.

Pairing controls shuffle within ancestor/regime, or additionally match two binary
qbar6/coherence bins using training medians. Singleton groups retain self donors and export
their fraction. Forward matching uses present order; reverse-field matching uses the later
conditioning observation. Positive shuffled-minus-paired loss is conditioning gain; operator
contrasts reverse this sign because larger VAMP-E is better. No test threshold is fitted.

`kinetics_samples` integrates the original FMRC forward head with fixed64-step Heun and32
particles at0.5/1 ps. The initial Gaussian includes path-noise variance0.001^2; endpoints retain
this smoothing. These are future marginals, not connected trajectories. The panel uses up to4
windows per test ancestor/ regime; shuffled donors come from the full heldout pool. Controls are
training empirical future samples and persistence. Energy, mean MSE, variance, coverage and
nonfinite fractions use common normalized measurements. Its scalar-bound counters retain the
older BO/density/velocity subset; they do not constitute the stronger joint forecast-validity
audit above.

`kinetics_probe` fits fixed ridge Gaussian readouts to future244 at0.1,0.5,1,3,6 ps using every
valid training origin>=3. Inputs are present/four-frame latent, current244/history976, and
regional current331/history1324 when applicable. Input standardization and diagonal residual
variance are train-only; ridge is0.001 and variance floor1e-4. Additional controls are a train
Gaussian prior and persistence with train-fitted residual variance. Weighted Gaussian NLL, MSE
and90% coverage are reported for all244, each family and the four order channels. NLL compares
the same target coordinates, not different latent densities. `kinetics_history_gain` is
current-minus-history NLL/MSE; positive gain indicates information accessible to this probe, not
proven non-Markovianity.

All kinetics statistics include all/present-liquid scopes and `_sources` tables. Exact
train/test indices bind identities; controls have separate data/training-cohort hashes and
regional namespaces. Native-only48-epoch fits never invoke their untrained decoder or autonomous
predictor for these assays.

## Aggregation and provenance

Ordinary source-balanced scores average windows within an ancestor, then ancestors equally;
1,000 whole-source bootstrap draws give95% intervals. Matched-seed contrasts average seeds
before source resampling, retaining seed SD and missing-seed counts. Undefined sources are not
silently discarded. `*_by_collection` and `*_collection_paired` separate old60/additional30
source coverage for trajectory, retention and validity. `kinetics_*_individual` and
`diagnostics_individual` concatenate declared arms without selecting winners. Three seeds are
not three times as many independent MD sources.

Metrics bind their frozen contract and implementation hashes. Scientific training uses online
W&B; frozen probes/checkpoint diagnostics stay local. Current-state recoverability, temporal
dependence, conditional forecast quality, local ordering and physical-validity failures answer
different questions and must be reported together rather than reduced to one rank or AP number.
