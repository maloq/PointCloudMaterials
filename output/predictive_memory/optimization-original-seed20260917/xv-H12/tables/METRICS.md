# Predictive memory: partial-observation pilot v1

This protocol uses no crystallization, PTM, onset, basin, nucleus, or topology
labels for encoder training or selection. Scores refer to continuous physical
packets on a fixed tracked-center support (full weight through 5 A, C2 taper to
zero at 7 A). Observation support includes every message-passing atom and ends
at 17 A, with the same C2 taper from 15 A. IDs only establish correspondence.

The packet has 128 coordinates: radial Gaussian basis (32), pair-distance basis
(32), weighted Legendre angular correlations of orders 1--16 (16), relative
speed basis (16), signed radial velocity basis (16), and 16 geometric/motion
moments. Exact endpoints, widths, normalizations and affine-fit ridge are in
`src/data/predictive_memory/targets.py`. Each coordinate is centered and divided
by its training-only population standard deviation, floored at 1e-4. Training
present and future packets jointly determine this common transform. Validation
and test data never determine scales.

Future offsets are 0.75, 3, 12, 48, and 96 ps. One mixture component spans the
entire 640-coordinate path. Each Gaussian has learned diagonal plus rank-two
covariance; diagonal standard deviations are softplus(raw)+0.05. Four components
are used in the pilot. `joint_nll` is negative log probability of the whole
standardized path divided by 640, in nats per coordinate per lag. It is a proper
predictive score under this fixed transform; values can be negative. It is not
an entropy or mutual-information estimate. No individual-lag mixture choices
are made. Selection uses the lowest validation joint NLL on the middle anchor
of each validation source, after the same prescribed optimizer budget.

`future_mse` measures the mixture mean's squared error averaged across all
coordinates and lags. Block and lag variants restrict that average. `present_mse`
is current packet reconstruction error from the exported state. `persistence_mse`
repeats the observed current physical packet at all future offsets. This is a
physical-target baseline, with access to the full current packet, including
velocities; it is not a positions-only observation-matched baseline. All MSEs
are in the common training-standardized coordinate system. The training loss is
joint NLL + `training.present_weight` times present MSE (0.05 in the original
pilot, 0.05 versus 1.0 in the optimization follow-up); no slowness, bending,
whitening or teacher penalty. Each configuration declares its update budget;
checkpoint selection still uses validation NLL only. The optimization follow-up
uses 12,000 updates, with validation every 250, and retains validation NLL,
present MSE and future MSE in `technical/validation.jsonl`.

Reported means first average the three anchors of each source, then weight
sources equally. `ci95` is a percentile interval from 500 whole-source bootstrap
resamples, using the run seed. Paired history gain is snapshot NLL minus history
NLL on exactly matched source/center/anchor tuples; positive means history helps.
Repeated-anchor gain is repeated-anchor NLL minus real-history NLL. A positive
point estimate with an interval crossing zero is unresolved. These are
exploratory, single-training-seed intervals: training-seed uncertainty is absent.

Comparison can select `--modalities xv` for the four-fit velocity-input replicate;
the default still requires all eight x/xv fits. Each selected modality must include
H=0,12,48 and its separately trained H=48 repeated-anchor control. Replicate
reports retain their own seed and source intervals; these are not pooled into an
independence claim or a combined training-seed confidence interval.

All samples use inherited exploratory sources; the three anchors are adjacent
(299.25, 300, 300.75 ps), not independent realizations. A single center is the
first ID of each inherited sorted four-center selection, not a fresh uniform
draw. Full-box float16 quantization can contribute to apparent memory gains;
no matched full-precision memory comparison is available in this pilot.

Runtime fields are wall seconds including validation/export after dataset load,
peak allocated GPU GiB (2^30 bytes), parameter count including allocated but
unused control pathways, trained updates, and selected checkpoint update.

## State-use and linear-readout diagnostics

These diagnostics freeze completed encoders and heads. For each model, replace
the state with its mean over training windows while keeping the actual temperature
condition and trained head fixed. `mean_state_*` reports this intervention's NLL,
future MSE and present MSE. `mean_state_*_increase` subtracts the original score,
so a positive value means removing sample-specific state information hurts.
Report validation and exploratory test separately with the same paired source
bootstrap. A nonpositive increase is evidence that this intervention does not
harm this fitted predictor; it is not proof of an exactly constant function,
a retrained condition-only likelihood, an entropy estimate, or state sufficiency.

Linear diagnostic readouts predict the full standardized physical path from
temperature only, current 128-coordinate physical packet plus temperature, or
exported state plus temperature. A fourth readout reconstructs the present
packet from the state alone. The packet input includes relative velocities; it
is observation-matched to xv, and only an information reference for x. These
readouts neither alter nor replace the native encoder. No past-packet readout
or full-history sufficiency test is implemented in this diagnostic.

Input feature means and population standard deviations (floored at 1e-8), output
intercepts and coefficients use training windows only. Ridge solves sum-squared
error + alpha times squared coefficient norm, with an unpenalized intercept,
using a float64 SVD. Alpha is selected from 0.001,0.01,0.1,1,10,100,1000,10000 by
MSE on the middle anchor of each validation source; ties take the first value.
Future-readout selection averages all 640 coordinates; present-readout selection
averages 128. Test values never select normalization, coefficients or alpha.
Readout predictions use the same per-block, per-lag and overall MSE definitions
as mixture means, but have no likelihood score. `future_mse_gain` subtracts
the candidate readout MSE from temperature-only readout MSE on paired test rows;
positive favors the candidate. Report source intervals separately for each
training seed. Saved coefficients, predictions, intervention scores, input
artifact hashes and release checksum accompany the exported summary.


Table export: 2026-09-17T05:42:15.863338+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
