# Local coordinate/velocity MACE metrics

Protocol: `mace_local_phase_space_v1`. These are **current local state** quantities,
not predictions of a future phase, crystallization time, or process progress.
All input velocities have units Angstrom/ps. All candidate positions are physical
Angstrom offsets from the tracked center, with complete 18 A candidate halos.

## Representation

`structure` (256 channels) is the smooth-inner MACE representation, standardized
using training features. `activity` (32 channels) is even under velocity reversal.
`flow` (16 channels) is odd. Motion uses velocities relative to the smoothly
pooled group velocity and pairwise velocity differences. Uniform translations,
rotations/reflections, permutations, and velocity boosts preserve the output.
Only atoms inside 7 A contribute to pooling, with unit weights inside 5 A and a
compact C2 quintic taper between 5 and 7 A. Both native 5 A message-passing layers
retain complete neighborhoods. Motion edge messages also taper to zero at 5 A.
Motion blocks are exactly zero for zero relative velocities. The coordinates-only
control has zero motion blocks and predicts activity from structure alone.

## Data separation and weights

The original independently melted sources retain their producer optimization /
model-selection / final-validation assignments as train / val / test. All older
shared-prepared-liquid trajectories and shooting descendants remain together in
training, even when old branch protocols used different split names. There are
30 independently prepared validation sources and 30 independent test sources.
No randomly split frames or overlapping sibling lineages enter those holdouts.

Training includes every retained record once per epoch, with inverse sample-count
weights within original source runs (`balance_group`). Thus hundreds of siblings
do not collectively outweigh equally weighted independently prepared sources.
Weights have mean one across the complete training epoch. Normalization uses the
same source-balanced weights, including current and previous local observations.
Test metrics average current/previous observations within each independent source
and then average sources equally. The 95% percentile interval resamples the 30
independent test preparations 2,000 times; it is not an atom-wise confidence interval.

## Physical normalized mean squared errors (lower is better)

`bond_order`: 16 smooth weighted means and standard deviations of q4, q6, w4, w6,
neighbor-averaged q6, q6 coherence, 12-neighbor density and smooth coordination.
Definitions come directly from `mace_local_state.physics.group_observables`.
Each column is centered/scaled using its training mean/standard deviation.

`instantaneous_TDA_H0`, `_H1`, `_H2`: persistence images of the **current** 80
nearest atoms, dimensions 16/64/64. Alpha-complex squared filtration radii become
Angstrom radii, finite deaths above 3.5 A are excluded, and the fixed Gaussian
images use the definitions in `liquid_structure.persistence_image`. Each column
is centered by its training mean. Within each homology block, the common scale
is the square root of the mean training column variance. Errors average columns
within that block. These are not relaxed/quench labels and are not numerically
interchangeable with older 400-dimensional normalized TDA targets.

`motion_even`: six instantaneous group observables: mean relative speed squared
(A²/ps²), fourth speed moment (A⁴/ps⁴), squared divergence (1/ps²), squared
deviatoric strain rate (1/ps²), squared rotation rate (1/ps²), and mean squared
non-affine velocity residual (A²/ps²). The smoothly weighted local least-squares
velocity gradient solves `(rᵀWr + 0.001 I) A = rᵀWu`, with centered coordinates
and bulk-subtracted velocities; the isotropic ridge is 0.001 A². Strain uses
the squared Frobenius norm of the symmetric traceless part; rotation uses that
of the antisymmetric part. Residual velocity is `u-rA`.

`motion_odd`: divergence, radial flux, and the third moment of radial relative
velocity. Radial velocity is `(r·u)/sqrt(|r|² + 0.25 A²)`. Flux is its weighted
mean. These reverse sign with all velocities. Even targets use training mean/std;
odd targets use exactly zero mean and training RMS to preserve reversal parity.
Errors average standardized squared residuals. A value near one corresponds to
the training-mean baseline at the training distribution, not necessarily the test
baseline under a distribution shift.

## Training and checkpoint selection

Training loss = mean of the four structural errors + mean of the nine standardized
motion-target errors + 0.2 structural teacher-retention MSE + 0.05 excess temporal
change. The teacher is the selected dual-physics MACE checkpoint, evaluated on
the newly prepared inputs. Its feature scales have a floor at 5% of the median
training channel standard deviation. Retention compares all 256 standardized
structural channels. Excess change penalizes only the positive part of the new
structural squared increment minus the teacher's increment, at measured physical
lags <=0.8 ps. Velocities and activity are not temporally forced to remain fixed.
The whole MACE backbone and new heads train together. This continuation does not
add a VICReg objective or forecast target. Prior VICReg information enters through
checkpoint initialization and feature retention.

Each variant selects its checkpoint by the smallest validation mean over the six
reported error families, including epoch zero. Test data never select checkpoints,
fit normalizations, determine weights, or tune thresholds. Random model/head seeds,
training order, structural supervision and budgets are matched across variants.
Status `completed_epochs` counts all finished training epochs; `selected_epoch`
identifies the best validation checkpoint and can be earlier.

## Stability and interventions

For each named block, `mean_squared_increment` is the mean squared Euclidean
difference for the same tracked center between current and previous frames.
`training_mean_squared_independent_distance` is twice the summed population
variance of a fixed random sample of up to 512 training anchors.
`normalized_temporal_change` divides the former by the latter. A zero reference
variance gives null (undefined), never zero. Test lags are retained in the cache;
these independently melted test trajectories have 0.75 ps spacing. Blocks have
different roles and should be compared separately rather than mixing motion
fluctuations into one structural-stability number.

Velocity interventions hold coordinates and targets fixed while removing all
velocities (`zero`), permuting velocities among atoms within each complete input
halo (`shuffle`), or applying one float16 storage round trip to velocities.
`reverse` flips all velocities and the expected signs of the three odd targets.
Structural embedding relative change is the Frobenius difference divided by the
unaltered structural Frobenius norm. Motion embedding changes average squared
Euclidean differences; physical intervention errors use the same training target
normalizations. Shuffling preserves the patch velocity distribution but destroys
position/velocity alignment. These are controlled input interventions, not extra
training or causal claims about the molecular dynamics.

The GPU verification report separately checks symmetries block by block and
compares direct backpropagation to microbatch gradient replay. CPU tests also
check atom identity in the graph, expansion/rotation/translation observables,
and paired position/velocity dump conversion. Small invariant errors arise from
float32 arithmetic; exact time parity and zero-motion properties are architectural.
