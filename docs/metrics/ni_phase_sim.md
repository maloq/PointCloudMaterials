# Dense Ni phase-space prediction

All new fits use train/test only. The final fixed update is evaluated; no test
selection, calibration, normalization, early stopping or AP objective. Tables
identify their exact implementation hashes. Native source ancestry remains the
independent sampling unit. Model repetitions are training seeds, not new MD.

## Training (`learning.csv`)

Coordinates are relative to the fixed central atom, divided by5Å. Velocities
are relative to that atom, divided by10Å/ps. Predictions start exactly at the
observed state and decode changes from the initial learned code. A step is5fs.
Each update samples one of1/4/10/20-step autonomous rollouts and uniformly
samples a valid start inside each recorded100fs continuation. All intermediate
states are supervised, with no insertion of true states inside the rollout.

`path` is MSE of decoded Q,V at every predicted step, divided separately by
train-only ballistic-residual RMS at that lag. Floors in normalized units are
0.002 for Q and0.02 for V. The scale averages all eligible starts of all train
paths. `kinematic` is MSE of the normalized position increment minus the
trapezoidal velocity integral, divided by0.002. `reconstruction` is joint
encoder/decoder state MSE on the same dynamic observations; it is not separate
static pretraining. `latent_value` matches one neural step from an encoded true
state to the encoding of the next true state.

`physical_response` matches exact neural JVPs of the autonomous decoded path to
the simulator's Q/V tangents. One of the two directions is sampled per update.
The propagated direction at the sampled start belongs to the original physical
perturbation, and is not renormalized. Each Q/V output/query/lag is divided by
its train teacher mean squared response, floored at1e-8 in normalized units.
`latent_response` matches J_G(phi(H))[J_phi(H)u] to J_phi(H_next)u_next on the
first step of that sampled rollout; starts cover every5fs transition. Its
denominator is detached target latent response power, floored at1e-6. Both
networks receive gradients. It is an auxiliary consistency objective; shrinking
unneeded latent channels is possible, so decoded physical response is the
primary sensitivity evaluation. Loss weights and all budgets are frozen in
the recipe. Missing objectives in a control are logged as zero by design.

## Prediction (`prediction.csv`)

All test inputs are evaluated from the original0fs state, free-running through
100fs. Tables report5/20/50/100fs. Each source contributes the mean over its
anchors; reported values average those source means equally. MSE averages XYZ
components and atom IDs, separately for position (Å²) and velocity (Å²/ps²).
`all79` excludes the central atom whose relative Q,V are identically zero.
`inner16` is the first16 noncentral IDs selected at the initial time; IDs are
never reselected. This diagnoses missing environmental support near the crop
boundary, without filtering any observation.

`response_Q_q`, etc., are decoded tangent MSEs for the named initial perturbation
and output, with exactly the train normalizers used in fitting. They are not
automatically ratios to one: compare with the explicitly reported zero-response
control on the same test rows. Ballistic Q=Q0+tV0 and V=V0, and persistence,
include their analytical tangents. Perturbations are initially0.01Å Q RMS or
0.1Å/ps V RMS across the physical selected80, mean removed; centering changes
the resulting local relative direction RMS.

`minimum_pair_distance_A` is the minimum decoded distance between distinct IDs.
`overlap_fraction` is the fraction of local patches with any pair below1Å.
These are stability diagnostics, not energy conservation or nucleus labels.

## Measurements and latent transport

`measurements.csv` evaluates the decoded physical atomic state directly using
existing density74 and velocity17 descriptors at5/20/50/100fs. Group-normalized
MSE uses training initial-state standard deviations with floor1e-5. Original
BO40 and TDA113 are evaluated at100fs using the historical fixed Ni length
normalization. BO/TDA channels with train standard deviation<=1e-8 are excluded
from this standardized metric; no row is excluded. All aggregation is source
balanced. These diagnostics are not training losses.

`latent_transport.csv` reports per-anchor/per-step Q-direction latent tangent
MSE and target power separately at all20 true transitions. The numerical JVP
audit compares neural exact derivatives to a centered secant at amplitude0.1;
it is not physical-model skill. Mean latent coordinate standard deviation is
a collapse diagnostic, not a calibrated information score.

`paired.csv` compares models within each initialization on identical test
anchors. Differences are left minus right. Percentile95% intervals resample
independent source ancestors1,000 times, after within-source averaging. They
describe source uncertainty conditional on that fitted seed. The direct model
has fewer parameters; the matched latent derivative ablations support cleaner
attribution than the capacity-unmatched direct/latent comparison.

Four plotted examples are the first four frozen test rows, never chosen by
performance. Atomic paths are deterministic physical/neural paths, not samples
of the NPT future distribution. The experiment does not evaluate nucleation AP.
