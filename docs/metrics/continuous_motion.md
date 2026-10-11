# Continuous motion from sparse Al observations

This family covers the matched fixed-step, continuous and derivative-supervised
latent dynamics campaign. It measures future prediction, not crystal labels,
clustering accuracy or reconstructed atomic trajectories.

## Observations, splits and independence

The pinned all0.1ps inventory supplies88 verified Lee2003-MEAM source views with
saved Q,V, descending from84 independent Al64 ancestors. Every view supplies two
outcome-blind initial times and all64 fixed centers. Exact duplicate input/future
records are removed, retaining ancestry and original roles. Coarser histories,
velocity-free million-atom data, relaxed data and four unknown-potential views
are outside this matched protocol. Source/row counts are not independent counts.

The original Al64 training ancestors are divided deterministically into fitting
and inner selection. Original selection/calibration/test ancestors are pooled
only for evaluation under the user's existing instruction. No test row enters
normalization, fitting, early selection or teacher EMA. Ancestor→view→row uniform
sampling limits descendant multiplicity in training. Inner selection uses one
fixed deduplicated draw from that hierarchy; its mean NLL selects checkpoints.

## Targets and likelihood

`response_geometry.features.observed` produces24 rotation-invariant statistics:
12 radial Gaussian shell means and12 angular power statistics. Each is computed
on the future neighborhood of the same center atom, with nearest80 atoms selected
at each saved future. Targets at exact0.1,0.5,3,12ps are actual source frames,
never interpolated histories. They are normalized by a single24-vector mean and
standard deviation over fitting rows and horizons, with a1e-4 scale floor.

The head gives a24-dimensional diagonal Gaussian at each latent SDE particle.
The reported joint mixture NLL is
`-log(mean_particle Normal24(y; mu, diag(exp(logvar)))) / 24`.
Training uses4 particles; selection and evaluation use8. Log variances are
bounded by `-5 + 8 sigmoid(raw)`. The likelihood is a distribution of these
statistics, not the full law of all future coordinates. Fixed Monte Carlo seeds
make evaluation reproducible; results can retain finite-particle error.

`mean_mse` is mean squared normalized feature error of the particle-mean forecast.
`persistence_mse` uses present descriptors unchanged as the prediction.
`marginal_nll` fits separate per-horizon diagonal Gaussian moments on fitting
targets only, without encoder inputs. `coverage90` is the fraction of feature
values with mixture marginal CDF between0.05 and0.95; nominal coverage is0.90.
These controls and calibration scores do not select checkpoints.

All64 and legacy16 retain the same fixed rows for every encoder. Scores first
average within ancestor, then average ancestors equally. NLL intervals are
2.5/97.5 percentiles of1000 bootstrap resamples of ancestor means. They describe
ancestor sampling uncertainty conditional on the trained fit; separate seeds
are reported separately, not treated as new physical samples.

## Short physics and learning metrics

Each initial time has one fixed outcome-blind center. The full70304-atom cell
continues for100fs with FastAl,1fs BAOAB, fixed initial box and1000fs friction,
native V and source thermostat T. Integration states, momenta, tangents, noise
and saved observables are float64; force arithmetic is float32. Initial archive
quantization remains part of the input provenance. Two independent thermostat
shots share their noise with the corresponding position/velocity tangent.

There are two local directions, initially supported on the visible80 atoms:
0.01A position RMS and0.1A/ps velocity RMS (RMS norm per atom, zero net shift).
Analytic force Hessian-vector products propagate these input sensitivities.
The rest of the complete periodic cell evolves normally. Only100fs enters
training; actual10,20,50fs continuations are withheld.

All three treatments receive the same short conditional-mean latent endpoint
loss: `0.5 mean((prediction - EMA_E(Q_future).mean(shots))**2)`.
The derivative treatment additionally uses the sum over two directions of
`0.25 mean((JVP_prediction - mean_shot JVP_teacher)**2) /
max(mean(mean_shot JVP_teacher**2), 1e-6)`.
The target encoder parameters have stop-gradient and EMA rate0.01. The teacher
JVP includes the complete simulator tangent then the target encoder Jacobian.
Student latent SDE noise is independent of physical BAOAB noise: the target is
a mean sensitivity, not a same-path equality across those two stochastic laws.
Two physical shots provide a noisy estimate; the relative loss is not claimed
to be an unbiased population sensitivity loss.

Total training loss is future NLL +0.1 short endpoint loss +0.1 relative tangent
loss (derivative arm only) +0.02 noncollapse penalty. The latter is
`mean(relu(1-sqrt(var(s)+1e-4))) +0.01 mean(offdiag(cov(s))**2)`.
`gradient_norm` is the pre-clipping norm; gradients clip at10. No AP, crystal,
species, temperature, age or coordinate-reconstruction targets enter this fit.

## Sub-cadence diagnostics

`dense-meam` compares forecasts with two independently generated held-out-source
shots. Feature-mean squared error subtracts the unbiased across-shot sample
variance divided by2. Negative estimates are valid finite-sample fluctuations.
`ballistic_mean_mse` uses Q+Vt of the same local atoms, with the same correction.
The fixed-step baseline uses explicitly labeled linear interpolation of its
predicted0/100fs feature moments at intermediate queries; these are never labels.

`drift_step_refinement_mse` compares noise-disabled0.05 and0.025ps integration
of the same trained model. It isolates drift discretization and is not a
stochastic solver convergence claim. Query endpoints can shorten a step.

Native0.01ps source data supply additional actual10,20,50,90fs observations.
Their original training ancestry is explicit: these are withheld times, not
independent test sources. Fresh fixed-box BAOAB and saved original NPT futures
remain separate evaluation laws; their scores are never pooled.

## Explicit128x2 execution profile

The additional A40/A100/L40S workers use microbatch128 and two accumulated
halves per effective batch256. A detached structural pass computes the global
256-sample noncollapse value and its derivative with respect to all structural
embeddings. Each recomputed half injects its slice of that derivative. Computing
two independent128-sample covariance losses would be a different objective and
is not used. The reported penalty is the original full-batch scalar, never the
linear surrogate used to propagate its cached derivative.

Each update draws one latent-noise seed. Both halves generate the reference
256-sample noise tensor at each solver step and consume their own slice. Thus
sample identities, simulator-record draws, gradient clipping, optimizer and EMA
frequency, and the latent Monte Carlo experiment remain matched. Selection and
evaluation keep the reference query blocks and noise; their encoder passes can
be split internally. Floating-point reduction order and GPU architecture can
still introduce small numerical differences.

The scientific fit identity refers to the canonical campaign. This profile is
a separately frozen, recorded execution implementation. Its source hash and
manual gradient-equivalence verification are recorded in
`technical/small-gpu-v1/implementation.json`; actual microbatch, host, device,
start update and producer identity appear per segment in each fit's
`technical/executions/` and checkpoint metadata. Canonical config microbatch256
is the reference/default, not a claim about every execution segment. One
optimizer checkpoint and fit lease are shared across devices, so workers never
train duplicate copies concurrently. Profile-produced metric tables have their
own implementation snapshot in the fit's recorded `metrics_root`. The combined
publication retains the original aggregation contract and links individual fit
receipts. No source rows or target definitions change.
