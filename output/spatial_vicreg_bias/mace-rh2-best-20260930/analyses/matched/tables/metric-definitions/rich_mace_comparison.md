# Frozen current rich-MACE comparison

The latest normalized residual-head MACE (RH2) is selected by the training run's
recorded minimum Al selection family-balanced descriptor Gaussian NLL. The frozen
checkpoint is epoch 14, optimizer update 1806, selection NLL 1.0348901381502593.
Selection uses the recorded training validation population, never AP or these
PaCMAP/MD test observations. This analysis launches no neural training. The exact
checkpoint, training producer and inference dependency hashes are recorded.

The encoder exports its 256-D scalar state with frozen scalar centering/scaling.
The descriptor prediction head is not an encoder input. Inputs are one centered
nearest-80 coordinate patch, fixed material length normalization (factor 1 for Al),
radius 8, cutoff 5 and one constant atom channel. No temperature, age, time,
species/material ID, history, motion or teacher is fed at inference. The model
was trained on 442 local descriptors plus scalar/vector variance/covariance and
scalar-mean regularization; descriptor agreement is therefore not an independent
discovery test. The six static snapshots are relaxed; training patches are raw.
Inference runs from the frozen source tree with recorded BF16 autocast and FP32
exports, chunks 256. The FP32 normalized residual decoder is loaded strictly but
inference uses only the encoder state.

Seven-cluster MiniBatchKMeans uses only the same 74,880 uniform training-source
observations from fixed Al64 release
`e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`, all64 track.
Seed 20260929, batch 4096, n_init 3, max_iter 200, reassignment_ratio 0; raw state,
no additional fitted whitening. Dense/sampled labels are audited for identical
atom/source/frame identities. Held-out/static observations do not fit centroids.

Descriptor fits, raw descriptors and their PaCMAP coordinates are reused from the
all-training reference: TDA, bond order, CNA and the family-balanced joint vector.
Their fit uses 74,880 uniform observations from 90 training sources, primary seed
17; see [the descriptor contract](general_descriptor_comparison.md). No descriptor
features, descriptor clusters or descriptor PaCMAP are recomputed here.

For each descriptor family and dataset, contingency rows are original neural IDs
and columns original descriptor IDs. A one-to-one SciPy assignment maximizes
raw shared membership on 24,960 held-out displayed observations, or all 684,723
dense grid centers pooled across the six static frames. This color mapping is
fixed across frames, filters and slabs. Shared colors do not establish equivalent
physical states.

Exports include the matching reference, displayed population (24,960/24,000) and
each dense snapshot. `rows`, `matched_rows`, `matched_fraction` report sample
count, assigned contingency mass and its fraction. Per-pair `intersection`,
`union`, `neural_rows`, `descriptor_rows`, `iou`, `neural_fraction` and
`descriptor_fraction` use the two original membership sets and fixed mapping.
Empty denominators are null. `adjusted_rand_index` is sklearn's chance-adjusted
Rand score. Browser correspondence uses currently displayed rows. Counts are
atoms, not independent trajectories; no confidence interval or significance claim
is computed.

Only full-population neural 3D PaCMAP is fitted, with pinned parameters from the
reference recipe. PaCMAP does not define clustering. Thirteen held-out full MD
frames and six static frames retain all prior centers. The interface toggle only
emphasizes finite distances <=12 Å in PaCMAP, keeping all observations and scores.
There are no separate interface fits, pages or metric subsets in this revision.

Five deterministic uniform examples per cluster, sparse geometry, PTM ideal
lattice overlays and both observed-atom travel paths reuse the existing display
producers. Travel uses the new 256-D vectors on the same frozen observed centers.
Paths never modify structures or imply time trajectories. Previous MACE analysis
artifacts are removed only after the new viewer and source identities are verified;
the original scientific training run is outside that replacement scope.


Table export: 2026-09-30T11:24:06.266182+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/rich_mace_comparison.json` relative to the analysis root.
