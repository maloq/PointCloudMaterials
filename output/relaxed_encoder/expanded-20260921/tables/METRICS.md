# Order-preserving neighborhood JEPA regularization study

Population and base reconstruction, fixed future, and equivariant geometry metrics
are as defined in `neighborhood_jepa_large_v2.md`: 32,768 training anchors from 90
native Al Lee-MEAM lineages, 480 development anchors from 15 separate lineages,
0.75 ps spacing. No new simulations or test-source training. Each update samples
512 independent current anchors, using the same 14 observed snapshot views as E.
768 updates mean 12 sampled epoch equivalents, not shuffled complete passes.
Every regularizer receives only current-center exports, not correlated neighbors.

The expanded relaxed study may set `regularizer_scope=temperature`: evaluate the
same SIGReg or VICReg penalty separately on current anchors at each observed
temperature, then average equally across temperatures represented in the batch.
Every group must have at least two anchors. Between-temperature mean differences
cannot satisfy the conditional variance term. This uses known temperature, not
phase labels; no test inputs are used. The recipe records weights and scope.

Alongside raw covariance participation rank, `invariant_correlation_effective_rank`
and its projected analogue compute the same participation ratio after centering
and dividing each channel by its development sample standard deviation (floor
1e-8). `*_std_quantiles` report channel standard deviations at0/10/50/90/100%.
Correlation rank distinguishes scale anisotropy from redundant channels but is
not itself a selection objective or proof of predictive information. The within-
temperature/noncrystalline rank metrics retain their original definitions.

All arms retain physical85, instantaneous TDA144, fixed angular moment anchors,
conditional present/future neighbor predictions, and fixed future physical/TDA
prediction. Both sides of latent prediction receive gradients. VICReg here means
its variance/covariance regularizer; conditional JEPA prediction provides the
alignment objective. We do not force future states to equal the current state.

Eight new order targets are q4, q6, w4, w6, averaged q6, mean bond q6 coherence,
12th-neighbor density, and smooth coordination. They use the exact repository
`liquid_structure.bond_order` producer, cutoff 3.5 Angstrom, with center plus its
12 nearest atoms and 12 nearest bonds per atom. Neighbors are restricted to the
encoder-observed crop: this can differ from the full raw-neighborhood baseline.
Current/next order values are standardized by training-only pooled moments;
standard deviations are floored at 1e-4. `order` is the equal-component standardized
MSE, averaged within source and then over development sources. Training adds .25
times order MSE, except the explicitly declared no-order control. All order targets
are decoded from the actual exported invariant state, never the regularizer projector.

SIGReg is the existing Epps-Pulley statistic / independent batch count (256 slices,
17 quadrature points). VICReg is `25*mean(relu(1-sqrt(diag(C)+1e-4))) +
sum(offdiag(C)^2)/D`, with sample covariance denominator N-1. Report its two
components and total. Regularizer multipliers and projector/export treatments are
predeclared in `regularization/specs.py`; identity-projector arms act on the export.

EpiJEPA-inspired score: fixed random width16 MACE invariant features, orthogonal
128-to-64 projection, seed9173, no learned weights/labels. Center and standardize
reservoir columns per batch, divide by sqrt(64), and solve ridge rho=3 by augmented
FP64 QR. Center projected learned features and divide by global RMS with epsilon
1e-4 to control amplitude. With W=(H'H+3I)^-1 H'Z, score S=.5 log2 det(I+30 W'W).
Cholesky on the equivalent smaller reservoir-side matrix computes the score.
Loss is -S/S0, where S0 is the mean score over four fixed initial training batches.
The reference image CNN and image-view alignment are replaced by geometric MACE
and causal neighbor prediction: this is an adaptation, not a reproduction of the
[EpiJEPA blog experiment](https://the-puzzler.github.io/blog/epijepa/).
The upstream source and commit are pinned in the run's technical/references.

The inherited encoder pooling readout already uses per-observation LayerNorm,
not BatchNorm. Train and evaluation exports remain batch-independent. Export
LayerNorm is per observation; raw-export arms replace the outermost one with
identity, retaining internal normalization. Warm transfer strictly loads all parent
encoder and prediction weights, resets projector/order head and optimizer.

Selection caches immutable phase labels and encodes only the current center;
training reads are grouped by shard and restored to the sampled order. These
performance changes preserve all objectives, aggregation and evaluation cadence.
Checkpoint selection remains source-equal `physical + .25*tda` in every arm.
Three continuations choose the warm, order-anchored winner separately within
SIGReg/VICReg/Epi using development `physical + .25*tda + .25*order +
.25*future_physical`, and add 1,536 updates at half LR. Crystallization test scores
never drive promotion. Comparisons use one training seed and are exploratory.

Effective rank is covariance participation ratio trace(C)^2/trace(C^2), including
within-temperature and noncrystalline subgroups; it is not the entropy rank in
the blog. A high rank alone is not evidence of retained useful information.
Frozen linear/MLP crystallization readouts follow `neighborhood_crystallization_v2.md`,
including matched geometry-only, geometry+motion+order, and condition-only baselines.


Table export: 2026-09-21T16:49:43.510231+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
