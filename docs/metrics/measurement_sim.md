# Direct-measurement trajectory prediction, version 1

This family belongs to the October 11, 2026 Ni experiment. Historical learned
280-dimensional decoder scores retain their own definitions and are not numerically
interchangeable with these tables.

## Observations and normalization

Each state contains BO40, TDA113, density74 and velocity17 (244 total). BO/TDA
columns retain the original fixed target producer and Ni-to-Al length factor.
Density uses four radial sums and ten radial-pair powers for each angular order
0–6. Velocity has nine radial-shell moments, four instantaneous bond-power rates,
and four affine-flow summaries. Smooth measurements use native angstrom/ps units
and a 5.5 angstrom support. No CNA, learned encoder or learned observation decoder
is present. The native dataset has 14,400 fixed-central-atom sequences, 64 frames
from -0.3 through +6 ps in exact 0.1 ps increments. The first four are inputs.
Ancestors and roles are unchanged (3,840 train / 960 selection / 9,600 test rows).
Selection remains separate; test combines the parent's outer held-out roles.

Means, standard deviations, increment scales and diagonal OU rates use training
rows only. Standard deviation has a 1e-4 floor. Channels at that floor are excluded
from scoring and frozen in the dynamics. Each family has total weight 1/4,
uniform across its remaining channels. Increment RMS has a 0.05 floor in normalized
units. OU correlation uses training adjacent frames, clipped to [0.01, 0.995],
with rate -log(correlation)/0.1 ps. The same OU process without learned residuals
is an evaluation baseline. It is an empirical prior, not a physical simulator.

## Predictive scores (smaller is better)

`nll` is a composite univariate Gaussian-mixture negative log likelihood, averaged
over future observation times and family-weighted channels. Each particle provides
one kernel center, with fixed normalized standard deviation 0.25. The mixture
weight is uniform. This is a marginal score, not a joint path likelihood; bandwidth
is not a learned decoder variance, and it is not added to displayed trajectories.

`path_energy` is the ensemble energy score E[d(X,y)] - E[d(X,X')]/2. Distance is
the square root of the time average of family-weighted squared normalized errors.
The second expectation uses all distinct unordered particle pairs, with no self
pairs, giving the unbiased finite-ensemble estimator. Numerically distance has
a 1e-6 floor. The entire ordered path is one observation in this score.

`increment_energy` applies the same score to consecutive increments, including
the first increment from the exactly observed initial state. Increments are divided
by their training RMS. `correlation_energy` applies it to two per-channel statistics:
the mean product of increments separated by one and five native intervals
(0.1 and 0.5 ps). Products always belong to the same sampled trajectory. This tests
temporal persistence/reversal; it does not identify every possible joint dependence.

The connected objective is nll + path_energy + 0.5 increment_energy +
0.25 correlation_energy. The marginal control uses nll. All arms also have
0.2 times `teacher_half_mse`: half the family-weighted squared difference between
the mean predicted dense path and mean physical two-shot path, averaged over
5,10,...,100 fs. Training uses four neural particles. All arms select checkpoints
by the same selection nll, never held-out scores or AP.

`endpoint_mse` compares the particle mean with the MD endpoint. Persistence is
the observed initial vector, and `mean` is the training mean. `temporal_power`
averages within-path temporal variance over individual particles; `MD_temporal_power`
is the same variance on the saved MD path; `mean_temporal_power` is the variance
of the ensemble-mean path. These are powers, not normalized ratios. The common
initial point is excluded from all three variances. `latent_path_90coverage` is
the weighted fraction of future measurements inside the 5th–95th percentile of
the 32 particle paths, without observation-kernel noise. Neural samples are not
independent physical MD continuations.

Reported trajectory values average within each ancestor, then equally across
ancestors. `lo` and `hi` are percentile intervals from 1,000 bootstrap resamples
of ancestors, separately for each regime. Overlapping local windows do not count
as independent source histories. Added liquid/front regimes were selected by
future behavior; these scores are not unconditional population frequencies.

## Simulator sensitivity

The teacher bank reuses 300 original Ni EAM/FS anchors: 80 train, 20 selection,
200 test across the same 16/4/40 ancestors. It stores two fixed-box BAOAB physical
shots and position/velocity tangents at each of twenty 5-fs horizons. Physical
integration/tangents were float64, forces float32. New measurement values/JVPs
are evaluated in float32. This is separate from the native NPT long trajectories.
Physical teachers track the initial 80 atom identities; native observations
reselect the nearest 80 around the same central atom.

Output target response is the analytic directional derivative of the 91 smooth
channels, averaged over the two physical shots. BO/TDA still receive value losses;
hard topology/rank changes are not mislabeled as analytic output derivatives.
Student response is a centered secant in the original Q/V perturbation direction,
using all 244 actual perturbed input measurements at the present time while the
preceding three observations remain fixed. Perturbation amplitude is epsilon=0.25
times the archived physical direction (whose Q/V units/scales are retained in the
teacher archive). Plus/minus predictions share random draws. Training averages
four neural paths, evaluation 32. The input map includes hard BO/TDA measurements,
so this secant is an approximation, not a claim of differentiability everywhere.

`response_loss` averages normalized squared errors separately over density74,
velocity-shell9, bond-rate4 and flow4 groups, twenty horizons and both directions.
Each fixed denominator is training-bank mean squared target response for that
group/horizon/direction, floored at 1e-5. The response treatment adds this loss
with weight one. Train and selection response losses use this same denominator;
zero does not necessarily score exactly one when the floor is active.

Held-out `relative_mse` instead divides ancestor-balanced squared error by the
ancestor-balanced target power, floored at 1e-5, for each group/time/direction.
`zero_relative_mse` uses target power as numerator; `persistence_relative_mse`
uses the error of transporting the instantaneous response unchanged. These are
explicit reference scores, not necessarily exactly one. `response_rms_ratio` is
sqrt(predicted power / target power). `response_cosine` is cross power divided by
the geometric mean of predicted and target powers (1e-20 denominator floor).
`half_step_relative_mse` repeats the student secant at epsilon/2;
`half_step_difference` measures its squared difference from epsilon, divided by
the same target-power denominator. Raw target power is reported to expose small
or effectively zero physical response.

`paired_sensitivity` compares the response treatment with its same-seed connected
control, zero and persistence. Values are (response error - control error) divided
by target power; negative favors response training. Confidence intervals use
1,000 paired ancestor resamples, recomputing numerator and denominator. No test
result selects a checkpoint or changes a training hyperparameter.

## Other columns

`update` is an optimizer update; `horizon_ps` in learning tables is that batch's
chosen 1/3/6-ps unroll. `gradient_norm` is the norm before clipping at ten.
`selected_update` refers to the marginal selection likelihood. Training histories
use batch statistics; selection histories use 128 fixed selection rows and eight
particles. Zero `response_loss` in controls means the term was disabled, not
perfect sensitivity. Blank selection fields mean no evaluation at that update.
Group/source counts and regime labels are cohort metadata, not network inputs.
