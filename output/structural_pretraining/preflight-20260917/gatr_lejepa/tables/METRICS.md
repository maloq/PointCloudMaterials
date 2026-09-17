# Structural neighbor pretraining metrics, version 1

This protocol trains on centered, material-cutoff-normalized atomic observations.
Targets are calculated in physical Angstrom before normalization. It is distinct
from the older 128-channel geometry/velocity predictive-memory packet.

Geometry has 85 channels: radial RBF32, pair-distance RBF32, Legendre16, and five
smooth-count/radial moments. Instantaneous TDA has 144 channels: H0 curve16 and
H1/H2 images64 each, with the existing physical nearest-80 and 3.5 A death cutoff.
There are no relaxed-topology or velocity targets.

Training-only means and population standard deviations are fitted to the frozen
release's valid target observations; standard deviations are floored at 1e-4.
Geometry statistics use current/spatial/future endpoints, never the unlabeled
past frames. TDA statistics use only valid instantaneous labels. Selection rows
and all descendants of held-out lineages are excluded from both fits.

For each observation, block error is mean squared error over the block's
standardized channels. Physical error averages its four blocks equally; topology
error averages its three blocks equally. Each view predicts its own targets.
Training averages physical error across both endpoints; topology averages over
valid target rows. The task anchor is physical + 0.25 * instantaneous TDA.

Selection exports use current observations only, with three causal inputs for
GATr JEPA and one for VICReg. Errors are averaged within each independent source
first, then equally across the fifteen selection sources. Selection score is
physical + 0.25 * instantaneous TDA. This selects a checkpoint; it is not a test
score or a source-bootstrap interval. State_std_mean is the mean per-coordinate
population standard deviation over these selection states, not a whitened or
within-material kinetic quality score. Stored row indices identify the release.

VICReg is (25 I + 25 V + C)/51, where I is paired-projector squared distance,
V averages relu(1-sqrt(sample_variance+1e-4)), and C is off-diagonal sample
covariance squared sum divided by projector width. V/C average over endpoints.
The full statistical batch contains 128 distinct anchor records from a single
material/potential family. This excludes between-material means from its
variance floor, but does not make correlated atomic observations independent.

Temporal JEPA uses 0.95 * next-projector MSE + 0.05 * SIGReg. The predictor sees
the anchor's three-frame causal state and actual elapsed ps; the next snapshot
is separately encoded as its target. Static batches instead have a spatial
partner, no future likelihood, and SIGReg alone. SIGReg is the pinned authors'
sliced Epps–Pulley implementation, 256 directions, 17 trapezoidal points on
[0,3] with symmetry weighting, averaged over endpoints. Its internal step buffer
is preserved in checkpoints. Representation weight is 0.1 in the total loss.
Latent_persistence compares the two current projected states within that model;
raw latent errors cannot rank encoders with different learned target spaces.

Exact full-batch gradients are obtained by a detached encoder pass, one joint
head/regularizer backward, and encoder recomputation with cached state gradients
and replayed RNG. Independent microbatch regularizers are not averaged.

Physical/TDA MSE is lower-is-better. Update counts, input waiting, GPU names,
label counts and training timing are operational diagnostics, not independent
research replications or hardware-benchmark results. One seed measures no
training-seed uncertainty. The three requested runs change architecture and,
for JEPA, input history and objective; there is no physical-only ablation here.


Table export: 2026-09-17T21:24:53.694430+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
