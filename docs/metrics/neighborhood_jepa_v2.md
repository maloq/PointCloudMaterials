# Neighborhood JEPA v2: native Al, geometry-anchored causal prediction

Population: 8,995 sampled training anchors from 36 independent native Al lineages;
480 development anchors from 15 other lineages. All use Lee 2003 MEAM and exactly
0.75 ps between frames. These are previously used development sources, not a new
test set. Temperature is a causal predictor input. Encoder exports are independent
128-dimensional signed invariant states and typed l=1,2,4,6 tensors, four radial
channels each. Raw invariant exports enter all data heads identically in train,
evaluation and deployment. No future-dependent normalization is performed.

Physical85 error averages standardized feature errors within each of its declared
blocks and then equally over blocks. TDA144 uses the inherited instantaneous TDA
block definition. Normalizers use the native-Al training subset only. Development
aggregation averages observations within source and then sources equally.
`selection_score = physical + 0.25*tda` is reconstruction, not evidence of dynamics.

Each angular channel is directly supervised against a fixed C2-supported solid
harmonic moment. Radii are 3.5,5,6.5,8 in the existing normalized coordinates. At
radius R, use component-normalized Y_l(x/R), without unit-vector normalization,
with quintic taper from .75R to R, divided by 1+sum(taper). All m components share
one train-only RMS scale per degree/radial channel, floored at 1e-3. Correct zero
cubic l=1/l=2 channels are allowed. No learned inverse angular decoder exists.
`geometry` averages scaled squared component error across radial and degree blocks.

Loss = current/next Physical85 + .25 TDA + .1 fixed moments +
.25 fixed future Physical85/.25 TDA (B–E only) + .1 invariant/equivariant latent
prediction (C–E as declared) + .1 per-sample SIGReg discrepancy. Query-family
means receive fixed weights of one; adding neighbors never dilutes the future
center weight. No past retrodiction or previous-state context is used.
SIGReg is the pinned Epps–Pulley statistic divided by the actual count of
independent current anchors, projected to 64D. It is deliberately not the raw
N-scaled statistic. There is one domain, mixture weight one. All target gradients
are retained. Conditional predictions and equivariant components are not Gaussianized.

Future metrics decode the current-conditioned predicted center at +.75 ps; A's
untrained future head is flagged diagnostic. Baselines are observed geometry
persistence, train-fitted per-temperature means, ridge current-geometry forecasts
(penalty .01 times training rows), and scalar per-temperature mean reversion.
These are separate from current/next observed-snapshot reconstruction.

Per-source arrays retain physical targets/predictions, temperatures, atom IDs,
frames, lineages, invariant/projected covariance spectra and error columns.
Effective rank is trace(C)^2/trace(C^2). One seed is exploratory; no automatic
promotion or `learned_dynamics` flag is defined. Five arms share initialization,
1,280 updates, batch 256 and 327,680 anchor draws; required views are 2/2/8/8/14.

The September 20 performance release preserves sampled batch order while grouping
disk reads by shard. Selection encodes only the current center used by these
metrics, and caches immutable tracked-center phase labels. Equivariant current
states are also retained for downstream horizon diagnostics. Metric formulas,
source weighting, evaluation cadence and selection criteria are unchanged.
