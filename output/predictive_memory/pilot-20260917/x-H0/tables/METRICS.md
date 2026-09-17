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
joint NLL + 0.05 present MSE; no slowness, bending, whitening or teacher penalty.

Reported means first average the three anchors of each source, then weight
sources equally. `ci95` is a percentile interval from 500 whole-source bootstrap
resamples, using the run seed. Paired history gain is snapshot NLL minus history
NLL on exactly matched source/center/anchor tuples; positive means history helps.
Repeated-anchor gain is repeated-anchor NLL minus real-history NLL. A positive
point estimate with an interval crossing zero is unresolved. These are
exploratory, single-training-seed intervals: training-seed uncertainty is absent.

All samples use inherited exploratory sources; the three anchors are adjacent
(299.25, 300, 300.75 ps), not independent realizations. A single center is the
first ID of each inherited sorted four-center selection, not a fresh uniform
draw. Full-box float16 quantization can contribute to apparent memory gains;
no matched full-precision memory comparison is available in this pilot.

Runtime fields are wall seconds including validation/export after dataset load,
peak allocated GPU GiB (2^30 bytes), parameter count including allocated but
unused control pathways, trained updates, and selected checkpoint update.


Table export: 2026-09-16T22:30:07.003430+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
