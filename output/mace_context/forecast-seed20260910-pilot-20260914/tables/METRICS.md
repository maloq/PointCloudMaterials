# MACE context and center-readout pilot

This family compares raw 256-channel scalar encoder embeddings. `mean80` uses
the original 80-node graph and mean readout; `halo_mean80` computes complete
two-hop context but retains hard 80-node pooling; `halo_inner` pools with unit
weight through 5 Angstrom and a C2 quintic taper to zero at 7 Angstrom;
`halo_center` returns the tracked center node. All use the original native 5
Angstrom edge cutoff and two interactions. Candidate coordinates extend to 18
Angstrom, including an explicitly checked training augmentation margin.
The quintic is evaluated in float64 as `(1-t)^3 (1+3t+6t^2)`, then stored
in float32. Its factored form prevents negative roundoff weights near the cutoff.

The cohort contains 64 original diagnostic anchors per context, 90 contexts,
5,760 rows, and the original 18/6/6 independent-source train/validation/test
split. Raw offsets are regenerated as float32 from the retained trajectories;
rounding their nearest-80 offsets to float16 must exactly reproduce each original
VICReg view. This avoids an additional offset storage round trip. Absolute
trajectory quantization remains present. This previously examined cohort is
exploratory, not a new untouched confirmatory test set.

`hot_test_mse` and `relaxed_test_mse` are equal-weight averages of H0, H1 and H2
descriptor MSE, divided by training-only block scales from `fit_targets`.
Descriptor widths are 16/64/64. Ridge uses float64 SVD, training-only feature
standardization and centering, with alpha selected solely on validation sources
over 1e-14 to 1000. The initial frozen export used 1e-9 to 1000; the grid was
extended downward when the trained center's validation optimum hit that lower
boundary. This adaptation uses validation scores only. Report train/validation/test errors, mean-label baseline,
grid-edge selection, and validation-tuned shuffled-training-label controls.
These are reconstruction errors, not classification accuracy. The labels still
describe hard nearest-80 supports even when encoder support is smooth.
`test_within_context_reduction` removes each test frame's mean separately from
predictions and labels before computing balanced error reduction versus zero
residuals. This diagnostic measures local variation after removing frame-wide
differences. Its use of test-frame means is only an evaluation decomposition;
it does not fit or calibrate the deployable readout on test labels.

Paired comparisons resample all six test sources with replacement 4,000 times.
Report relative MSE reduction `1 - candidate/reference` and percentile 95%
intervals. One encoder initialization is used; intervals do not estimate
training-seed uncertainty.

Continuous structural readouts use the original probe geometry's last four
columns: q4, q6, nearest-12 shell density, and mean nearest-12 distance. They
are computed from a smaller physical neighborhood than the TDA rank-80 boundary.
A shared ridge penalty minimizes mean validation MSE divided by training target
variances. Each reported mean-baseline reduction is `1 - MSE / mean_baseline_MSE`
on the declared split; the baseline predicts the training mean. Temporal
structural increment reductions use the retained temporal observables' columns
2–5 and compare with zero-increment persistence, separately for each observable.

Temporal data reuse 144 tracked atoms/context combinations over 17 frames spaced
0.75 ps. Lags are 0.75/1.5/3/6/9/12 ps. `relative_latent_mse` divides the mean
squared embedding increment by the scalar mean of training feature variances.
TDA increment error applies the same fitted hot-label readout to both times.
`tda_increment_reduction` is one minus balanced increment-prediction MSE divided
by the zero-increment (persistence) MSE. Pair overlap is not independent sampling.

Controlled crossings use four preselected tracked centers per test context (72
clouds). The 80th and 81st atoms are placed on their original radial directions
at the mean of their initial radii plus/minus epsilon; the two radii are swapped
on the other side. All methods see the same two physical configurations. The
baseline reselects nearest 80 on each side. Epsilon is 0.1/0.01/0.001/0.0001
Angstrom. `boundary_fraction` is the mean squared embedding difference at the
smallest epsilon divided by that method's measured 0.75 ps increment energy.
The denominator is explicitly representation-specific; it is not a forecast
error. The complete curve distinguishes a finite membership jump from continuous
sensitivity to atom motion. Purely numerical effects can dominate small values.
The optional `labels` stage recomputes the repository's exact 144D persistence
image on both hard-80 clouds in every identical crossing. It exports balanced
TDA squared differences and their ratio to observed 0.75 ps TDA increment
energy, using the same training-only hot-target block scales. This distinguishes
an encoder membership artifact from a discontinuity also present in its target.

`feature_rank` is the participation ratio of centered training feature covariance,
computed as `(sum(singular_values**2))**2/sum(singular_values**4)` without per-channel
standardization. Training feature variance is also retained to reveal collapse.

Training is a distinct eight-epoch warm-start pilot from the forecast checkpoint,
batch 256, constant AdamW learning rate 1e-4, original weight decay and clipping.
It uses original spatial/temporal VICReg loss, projector and augmentation settings.
Identical full-candidate augmentation draws are used across methods before the
baseline crop. No TDA labels enter encoder training. Complete-batch projector
BatchNorm and VICReg statistics precede exact encoder gradient replay in
microbatches of eight. Lowest validation VICReg loss selects the checkpoint;
test sources are first evaluated after selection. GPU verification compares
pruned/full node features, invariances and replay gradients against retained
autograd graphs with matching microbatch shapes.

Scores and per-row paired errors live in `technical/`; exported CSVs retain this
definition and hashes of all context computation and scoring implementations.


Table export: 2026-09-14T18:36:07.155195+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
