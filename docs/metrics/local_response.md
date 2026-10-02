# Local80 MLIP response pilot

Every model sees the same90/15/30 train/selection/test queries, one predetermined
fixed64 center per original source, with inherited roles and no calibration data.
Frame64,320,656 and center index17*source-rank modulo64 are assigned without
outcome inspection. This new shooting-query assay is not a replacement or subset
score of the all64 crystallization-window benchmark. Subsequent matched
crystallization evaluation must include that benchmark's complete all64 rows.

Parents are decoded from verified float16 MEAM histories. The new teacher is
MACE-MPA-0 medium at450K,1fs BAOAB,100fs friction and20/100fs observation horizons.
Parent quantization is explicit; differentiable perturbations/integration remain
float32, never re-quantized. Native restart/query tensors retain precision; no
trajectory positions are exported by this collector. Teacher conditions are
never neural inputs. All atoms in an open spherical environment move; initial
directions perturb only the nearest80 atoms, keep the center fixed, and have
zero summed displacement and orthonormal columns. Exterior tangents start at
zero but evolve through force Hessians. No hard neighbor reselection occurs in
the initial perturbation. Fresh velocities and thermostat noise are generated in
original70304-atom order and gathered by original row, ensuring matched random
streams for nested environments and paired finite differences.

The future observable at the tracked center contains eight Gaussian radial
densities (centers2..5A,width0.35A) and four orientational powers l=1..4. Weights
are (1-r)^4(1+4r) for r=distance/5.5A<1 and zero outside. Each angular power is
the component mean square of the weighted mean spherical harmonics, using e3nn
component normalization. All environment atoms can contribute inside this
smooth support. Changes relative to each path's own differentiable initial
observable are divided by0.1. Each temporal prefix has128 fixed cosine Fourier
features, standard-normal frequencies divided by sqrt(prefix dimension), uniform
phases, amplitude sqrt(2/128). Target dimensions0:128 describe20fs;128:256 describe
the joint20+100fs prefix, not a100fs marginal. The complete target has256 dimensions.

Training uses8 value shots,8 value+response shots, or32 value shots. Training
responses reuse their forward values with one independently executed value audit.
Test32 value and8 response streams are disjoint. Selection always uses32 values.
Each feature is centered/scaled using only the first8 training shots across all
training parents (population standard deviation, floor1e-4). Responses divide by
the same feature scales; a single further scale is the RMS of their training
parent means, floored at1e-3. The ordinary objective is half normalized squared
error, equivalent to a fixed unit-variance Gaussian NLL apart from its constant.
Response treatment adds half squared complete-predictor JVP error divided by
the response RMS, weight1. Encoder and nonlinear head both receive gradients.
Checkpoint selection is shared validation value NLL, never AP or response error.

The values8_time control trains until synchronized optimization+validation time
reaches the same-seed response fit's measured time (or its explicit6000-epoch cap).
It disables patience stopping but retains validation-NLL checkpoint selection.
This is an optimization-time control, not equal total acquisition cost or equal
selection multiplicity. Tables state whether the measured budget was reached.
Scientific fits stay online in W&B; gates and saved-checkpoint evaluation are local.

Held-out value/response MSE uses the independently sampled target means. Corrected
MSE subtracts sample variance (ddof1) divided by shot count, averaged across the
same coordinates. It can be negative. Reported value NLL is0.5*(uncorrected MSE+
log(2*pi)); it is a feature-mean Gaussian score, not a calibrated atomistic
trajectory likelihood. Response MSE includes the common training-response RMS.
Three-seed errors are averaged within source before2000 paired bootstrap draws
of the30 test sources, with replacement. Intervals are percentile95% and exclude
training-source/seed uncertainty. Contrast is response minus its comparator;
negative means lower error. Prior is zero normalized value and zero response.
Every checkpoint must export all135 identical parent/source IDs; no row dropping.

Environment radius is selected only on four training queries and two common-noise
branches each, comparing18/24/30A candidates to every larger radius through36A.
Each prefix must satisfy RFF value RMS error<=1e-4 and response relative Frobenius
error<=5%. These are sensitivity tolerances, not confidence bounds or an exact
full-cell guarantee. No candidate passing means no training. Float32 versus
float64 path checks require relative error<=1% and maximum absolute<=1e-4;
AD/FD minimum error across declared0.03/0.01A steps must be<=2%. Independent ASE
versus chunked GPU graph forces/HVPs require relative error<=0.1%. Student
input JVPs, response-only encoder gradients and translation/permutation
invariance are checked separately. All gate failures are preserved.

Acquisition costs are actual complete batch times with reuse/audits included.
They are costs of a shared bank, not hypothetical per-arm simulation charges.
Timing excludes initial model load. Gate peak GiB is peak torch allocated GPU
memory, including resident models. Training_seconds includes synchronized
optimization and validation; checkpoint/logging I/O is excluded. Wall-time fields
are stored locally and in final W&B summaries, not duplicate history metrics.

Partial-observation interpretation: fixed-initial-environment derivatives are
interventional targets. They need not equal derivatives of the observational
conditional mean given only80 atoms. This experiment tests useful regularization;
short-time results do not establish picosecond crystallization prediction.
