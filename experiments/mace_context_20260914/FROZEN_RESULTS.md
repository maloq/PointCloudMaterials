# Frozen forecast checkpoint: context and center readouts

Both requested readouts remove the finite neighbor-membership jump in the tested
crossings. Their squared embedding difference decreases approximately 100-fold
for each 10-fold reduction in displacement. The original nearest-80 graph and
mean readout approaches a nonzero jump instead.

The smooth inner readout improves prediction of relaxed topology but sacrifices
some information about the instantaneous hard-80 point cloud. The center readout
better captures several immediate local structural changes and is weaker for
patch-wide topology.

| Frozen readout | Hot TDA test MSE | Relaxed TDA test MSE | Crossing / natural 0.75 ps embedding energy |
|---|---:|---:|---:|
| Original 80-node mean | 0.017148 | 0.035753 | 0.03668 |
| Complete context, hard-80 mean | 0.042502 | 0.031095 | 0.006452 |
| Complete context, smooth inner mean | 0.041357 | 0.030925 | 1.43e-9 |
| Complete context, tracked center | 0.077556 | 0.049863 | 1.49e-10 |

The crossing values use epsilon 0.0001 Angstrom and 72 matched test patches.
Each denominator is that representation's own natural temporal increment energy;
the complete displacement curves, not their smallest value alone, demonstrate
continuity. Additional message context alone reduces the normalized jump by
82.4%, while hard membership in pooling leaves a residual jump.

Recomputing the exact repository TDA descriptor on both sides confirms that the
hard-80 **label also jumps**: its squared difference approaches 2.51% of ordinary
0.75 ps TDA increment energy. Thus some discontinuity is built into the current
target definition. This does not explain the entire reconstruction tradeoff,
but it makes perfectly smooth embeddings and exact hard-80 label recovery
conflicting requirements at these crossings.

Smooth inner pooling reduces relaxed-TDA MSE by **13.50%** versus the original
encoder, with a paired six-source bootstrap 95% interval of **10.60–16.38%**.
It improves every test source. Its hot-TDA error is **2.41 times** the original.
The center's relaxed-TDA error increases 39.47% and hot-TDA error is 4.52 times
the original. These are balanced descriptor MSEs, not classification accuracies.

The retained information generalizes. For smooth inner pooling, relaxed-TDA
train/validation/test MSE is 0.026683/0.029339/0.030925, compared with test MSE
0.858473 after shuffling training labels. After removing frame-wide differences,
the fixed readout explains 87.33% of within-frame relaxed-TDA variation, versus
85.91% for the original. This supports real held-out descriptor information,
without claiming that VICReg learned all of it rather than retaining information
already available from the architecture and MLIP initialization.

Physical changes remain represented, with different spatial emphasis. At 0.75
ps, hot-TDA increment error reduction versus persistence is 82.55% for the
original, 61.71% for smooth inner pooling, and 28.39% for the center. For the
center, local q6 increment error reduction improves from the original's 11.32%
to 20.00%; nearest-shell density improves from 26.66% to 63.52%, and mean first-
shell distance from 82.19% to 97.25%. Smooth inner pooling is weaker on these
immediate local changes. Smoothing does not uniformly improve every physical
observable.

All comparisons use 5,760 original diagnostic anchors and independent 18/6/6
source splits. Readout scaling and coefficients fit training sources; ridge
penalties use validation sources. This is an exploratory cohort already examined
in earlier diagnostics. The frozen comparison precedes matched eight-epoch
VICReg continuation of the original mean and both requested variants. It does
not establish downstream forecasting improvement.

The graph computation agrees with an unpruned full-context calculation at about
1e-13 relative squared error. Rotation, point-order and extra-halo controls pass.
Encoder gradient replay agrees within 0.007% relative L2 error with retained
autograd graphs. The 18 Angstrom candidate sphere is safely below half the
smallest source box length (106.21 Angstrom).

![Frozen comparison](../../output/mace_context/forecast-seed20260910-pilot-20260914/plots/frozen-context-comparison.png)

Pinned [scores](../../output/mace_context/forecast-seed20260910-pilot-20260914/tables/frozen-comparison.csv),
[machine summary](../../output/mace_context/forecast-seed20260910-pilot-20260914/technical/frozen-summary.json),
and [metric definitions](../../docs/metrics/mace_context.md).
The pinned table used ridge penalties 1e-9 to 1000; its exact exported definition
is [retained here](../../output/mace_context/forecast-seed20260910-pilot-20260914/tables/FROZEN_METRICS.md).
The current comparison extends that grid to 1e-14 after the trained center's
validation optimum reached the initial lower boundary.
The [current comparison](../../output/mace_context/forecast-seed20260910-pilot-20260914/README.md)
also includes trained variants as their evaluation completes.
