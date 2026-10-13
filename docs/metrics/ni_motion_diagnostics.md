# Frozen Ni motion diagnostics, version 1

October 11 follow-up exports extend these definitions with predeclared tail
scores and response-predictability readouts below. The first run's exported
definitions and source hashes remain immutable.

These tables audit completed simulators and local frozen Gaussian readouts. They
do not select or train an encoder using test outcomes. The three retained model
arms use seed 20261011 and their original validation-NLL selected checkpoints.
The two physical continuation shots are dependent views of one anchor, not two
independent source observations. Outer Monte Carlo keys are separate replicates.

## Population, observations and time

The source is the sealed 14,400-row Ni collection: 3,840 fitting, 960 selection,
9,600 held-out rows, with source ancestry unchanged. A balanced path panel takes
two hash-ordered rows per existing held-out (ancestor,regime) stratum. Thirteen
held-out ancestors have no growing-front rows in the source collection; they
remain represented in both other regimes. The plan lists these absent strata.
The resulting panel has 214 rows and all 40 held-out ancestors. Regimes are original,
future-liquid and growing-front. This future-conditioned mixture does not
estimate natural population prevalence. The frozen manifest lists actual row
IDs, roles, checkpoint and data hashes, teacher kernel and perturbation settings.

At forecast origin t0, history is t0+[-.3,-.2,-.1,0] ps. Native targets are
t0+tau, tau=.1,.2,...,6 ps. A neural step is 5 fs; the separate physical teacher
integrates at 1 fs and saves every 5 fs through 100 fs. Native NPT observations
refresh nearest80 membership. BAOAB teacher observations follow initial80 atom
identities within evolving full cells at fixed box. Scores never equate these
kernels. All observation standardization uses the existing train-only mean and
scale. Indices 0:128 are frozen geometry, 128:256 velocity secants, 256:280 smooth
radial/angular descriptors. Every raw smooth descriptor is nonnegative.

## Endpoint and path scores

For particles X1,...,XK and one observed vector y of dimension D, energy is
mean_i ||Xi-y||/sqrt(D) - sum_(i!=j)||Xi-Xj||/[2K(K-1)sqrt(D)]. This all-distinct
pair U estimator may be negative. No clipping or disjoint-pair approximation is
used. Paths concatenate standardized coordinates; future increments use only
uncertain future endpoints, excluding t0. Lower energy is better.

Endpoint NLL is minus the log of the equally weighted diagonal Gaussian mixture,
divided by coordinate count. Mean MSE uses the average conditional mean.
Coverage90 uses empirical emitted-sample 5th/95th percentiles. Process variance
is unbiased across-particle conditional-mean variance; emission variance is the
mean predicted conditional diagonal variance. Endpoint scores cover each feature
space at .1/.5/1/3/6 ps. Full paths retain only the 24 smooth descriptors.

Joint scores concatenate two future times (.1,.2), (.5,1), or (3,6) ps. Increment
scores use the corresponding differences. Full-path energy uses all 60 times;
future-increments energy uses their 59 successive differences. Lag-product scores
use the time average of (X(t)-X(t0))*(X(t+lag)-X(t0)), at .1/.5/1 ps lags; these
are second-moment descriptors, not an estimate of a conditional covariance from
one observed trajectory. Excursion is the per-coordinate maximum absolute change
from the initial observation. Occupancy/dwell concatenates fraction of native
times above initial+one training standard deviation and mean active-run duration
in ps. No active run gives zero duration. These are statistical descriptors, not
PTM crystallinity labels or nucleus lifetimes.

Path scores are reported separately for conditional means (a latent-process
diagnostic) and emitted samples (the model's predictive observation law). At each
time the same particle permutation is applied to mean, variance and sample.
Shuffling occurs separately within each tested K prefix; endpoint multisets are
unchanged. Scramble-gap is scrambled minus connected score, paired by row/key.
A positive gap indicates better connected forecasts on that score; a null gap
alone does not establish adequate temporal modeling. The synthetic integration
assay checks sensitivity and endpoint invariance.

Negative-descriptor rate uses inverse-standardized emitted descriptors. Nonfinite
paths fail the stage with a visible receipt; no rows are silently removed.
Persistence repeats the last observation. Its path energy has zero spread term.

## Response audit

One deterministic original teacher anchor per ancestor is re-encoded from the
saved float64 physical states/tangents using the frozen float32 eSEN geometry
map, retaining both shots. No physical simulation or relaxation is repeated.
Target replay RMSE is recorded in each packet; historical targets stay unchanged.

For shot responses r1,r2 (normalized by the unchanged target scale), teacher
mean power is mean[((r1+r2)/2)^2], cross-shot power is mean[r1*r2], unbiased shot
variance is mean[(r1-r2)^2]/2, and two-shot mean noise is mean[(r1-r2)^2]/4.
Raw teacher power omits feature standardization. Zero and persistence response
MSE compare the averaged teacher with zero or the saved initial geometry JVP.
Variance-response RMS uses (phi1-phi2)*(r1-r2), the n=2 unbiased derivative of
shot variance; it is noisy and does not identify a 128-dimensional covariance.

For particle predictions d_i and target r, squared-mean loss is mean[(mean_i d_i-r)^2].
Mean-variance penalty is mean[unbiased_var_i(d_i)]/K. Unbiased-mean loss subtracts
that penalty and can be negative. These values omit the one-half factor in the
value loss and the response loss's batch denominator and outer weight: they are
raw loss components, not a reconstruction of the total training objective.
Only conditional-mean variation enters this penalty, not decoder emission noise.

Q/V secants replace only the present history slot using actual physically
perturbed encoder observations at epsilon=.5,.25,.125; older observations are
unchanged. The denominator is 2*epsilon. All plus/minus predictions use identical
process draws. Epsilon-change MSE compares individual particle secants with the
epsilon=.5 result. This is a student/encoder secant audit, not a finite-difference
validation of the physical force derivative. Two physical directions cannot
establish a general rank/null-space obstruction.

Training normalizer tables draw 64 batches of 16 fitting anchors with replacement
from the entire existing teacher bank. They report timewise response power and
the 1e-5 denominator floor separately for Q/V. Held-out responses are not used
to set normalization, weights, or scientific loss choices.

## History readout

Shift every row's origin by +1 ps. The 14 available observations then cover
[-1.3,0] ps at native .1 ps. History lengths 1/4/8/14 predict the same smooth24
targets at .1/.5/1/3 ps. Compare raw280 with each frozen compact16 representation.
Ordered inputs concatenate history; unordered controls retain present plus past
mean and population standard deviation. Reversed-past controls reverse only
past slots, keeping present and fitted weights fixed. Input width differs across
histories; this is a fixed linear-readout sufficiency assay, not a capacity-matched
nonlinear simulator training experiment.

Gaussian ridge fits use only fitting ancestors. Input scales and intercepts are
fitted there; residual diagonal variance is fitting MSE floored at 1e-4. Ridge
strengths .001/.1/10 apply to the mean covariance (X'X/n) and are selected by
selection Gaussian NLL. Report held-out MSE and per-coordinate Gaussian NLL.

## Aggregation

Path/history tables report equal-row, equal-regime, equal-ancestor and
equal-(ancestor,regime) weighting. Within each named group rows have equal mass;
each group has equal total mass. Fixed design weights are reused in 500
ancestry-bootstrap draws; multiplicities multiply those weights before global
renormalization. Repeated ancestors are never silently deduplicated. Intervals
are percentile 95% intervals, conditional on the chosen checkpoint and outer key.
Response raw tables retain anchor/ancestor identifiers instead of pretending
shots, horizons, or directions are independent sources. Missing values are blank,
not zero. Every export records these definitions and implementation hashes.

## Follow-up: six-ps training and tail measurements

The follow-up pins four completed 6-ps transformer models, values/response crossed
with training seeds 20261013/14, each trained for 4,000 updates at batch256. The
1-ps values-only transformer at seed20261014 supplies the matched horizon control:
same architecture and budget, with both selected checkpoints at update1,000.
Other 1-ps runs with a changed batch schedule are excluded from that comparison.
The old MLP results are historical references, not a pure horizon intervention.
Rows, mean/scale, K prefixes and Monte Carlo keys remain exactly those of the
first audit; training-seed and sampling-key variation remain separate.

At prefixes of1/3/6ps, particle_peak_gt10/100 is the fraction of particles whose
maximum absolute conditional-mean smooth descriptor exceeds10/100 standardized
units. particle_path_mse_gt100 is the fraction whose mean squared difference from
the observed smooth-descriptor path exceeds100 across times and coordinates.
any_particle_path_mse_gt100 is the indicator that a row has at least one such
particle. All paths remain in every predictive score; these indicators do not
filter outputs or alter the predictive law. Thresholds were chosen from the first
audit and frozen before evaluating the new transformer checkpoints.

## Follow-up: response predictability before and after compression

All300 original physical anchors are used:80 fitting (16 ancestors),20 selection
(4 ancestors),200 test (40 ancestors). Each has two query directions Q/V and two
physical shots, whose already-recorded average response is retained. These are
not extra independent states. Predictor inputs are four observed vectors and a
present-slot directional change. Targets are128 geometry-JVP coordinates at
5,10,...,100fs, divided by the same original train-only native feature scales.
The initial geometry JVP is a separate t0 output/control. No native NPT labels
replace the BAOAB response targets, and no future values enter the inputs.

Representations are raw280 observations, the five newly audited compact16
encoders, and the three original-audit compact16 encoders. Raw280 is a fixed
learned/descriptor observation vector, not the full atomic MD state. Directional
changes use the actual saved plus/minus observation pair at epsilon=.5, divided
by2epsilon. Compact directions encode these same pairs and then take the secant.
Only the present history slot changes. No relaxed data, broad spatial replay,
temperature, age, absolute time, species or source ID enter the readouts. Q and V
query families have separately fitted operators; the family chooses the diagnostic
task and does not become an encoder input. Twenty horizons are fixed output slots.

Each representation is scaled using only the fitting anchors, pooling their four
observations; standard deviations are floored at .001. The same scale divides
the directional change, without centering it. Directions then divide by their
fitting root-mean-squared Euclidean norm. This scalar is fixed for all roles.
Let h denote flattened scaled history and d the scaled direction. Linear ridge
uses kernel k((h,d),(h',d'))=d'd'. The state-conditioned readout uses
0.5*(1+exp(-||h-h'||²/(2*dim(h)*b*ell²)))*(d'd'), where b is the median fitting
off-diagonal history distance squared per coordinate and ell is .5 or2. Thus
the readout is nonlinear in observed state, but exactly linear/odd in its query
direction. It is not a linear-readout-only test of a nonlinear encoder.

For each query family and output time, a fixed Gaussian variance is the fitting
mean squared target response over anchors and128 coordinates, with its standard
deviation floored at1e-8. Outputs divide by this standard deviation before fitting.
Solve (K+n*lambda*I)W=Y with lambda in1e-4/1e-3/.01/.1/1/10. Select lambda (and ell
for the state-conditioned readout) by Gaussian NLL on the original selection
ancestors, averaging future slots only; the t0 control never selects the model.
The complete candidate table is retained. This is a fixed-variance conditional
mean likelihood, not a calibrated distribution of physical responses.

Report held-out per-coordinate MSE, zero-predictor MSE and the fixed-variance NLL
at each time. Per-row scores use the existing four weightings and500 ancestor
bootstrap draws. Here each ancestor has five anchors in one physical-anchor
family, so those weighting choices coincide. Baselines are zero, the repeated
exact initial geometry JVP (an explicitly privileged present-response control),
and the repeated present geometry secant from raw280. At t0 the V geometry
response is identically zero; it is not a meaningful relative-error denominator.

The comparison diagnoses accessible predictive information within these declared
readout classes and small fitting cohort. Failure of both readouts cannot prove
that an observable is insufficient; it can also reflect limited data or capacity.
Only two directions per state are available, so this is not a general Jacobian
rank or null-space obstruction test. Frozen readouts stay local without W&B.
