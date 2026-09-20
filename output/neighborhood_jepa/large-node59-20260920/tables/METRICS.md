# Neighborhood JEPA v2: native Al, geometry-anchored causal prediction

Population: 32,768 sampled training anchors from 90 independent native Al lineages;
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
promotion or `learned_dynamics` flag is defined. These two scale runs use arm E (14 views per anchor), width64 and global anchor batches 1,024 / 2,048 on node53 / node59. One global objective and SIGReg are differentiated before exact two-GPU encoder gradient replay. Both retain target gradients. Separate preflight timing fixes the complete update budget before training, targeting three hours. Epoch equivalents mean anchor draws / 32,768, not exhaustive shuffled passes. Per-run resolved configuration records the chosen count. These are equal-wall-time exploratory scale runs, not an isolated batch-size or hardware comparison.


Table export: 2026-09-20T16:47:12.207157+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
