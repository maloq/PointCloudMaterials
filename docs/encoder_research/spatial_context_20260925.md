# Spatial context: previous implementation and next design

Reviewed 25 September 2026. This distinguishes **context used by a predictor** from **context that changes the exported encoder state**. The latest capacity experiments use a single local snapshot; they do not yet have the symmetric multi-patch context described below.

## What the symmetric implementation did

The September 21 structured-context study built a cuboctahedral query stencil:

* One focal center.
* Twelve query locations at 10 Å.
* Twelve more at 20 Å, in the same directions.

The twelve directions are permutations of `(0, ±1, ±1)/sqrt(2)`. A periodic nearest-atom assignment maps the 24 noncentral queries to distinct real atoms, with maximum query error 4 Å. It uses a Hungarian one-to-one assignment, rather than allowing multiple queries to pick the same atom. The query points are symmetric; the selected atoms generally are not. The stencil is tied to the box frame: rotating coordinates and the stencil together preserves the geometric features, but resampling rotated coordinates against an unchanged stencil is not continuously rotation invariant.

Each selected atom centers a local patch, encoded into a frozen 128-dimensional feature. The original observed study used current and historical observations. The later **relaxed-reuse** experiment used three available observations within a maximum 72 ps history, with actual offsets and no interpolation. For relaxed patches, selected observed atom identities survive the quench; the nearest-80 observed candidate identity set is also retained. These are local readouts from full-cell relaxation, not independently relaxed isolated clusters.

Across times, a slot refers to the same query location relative to the focal center. It is not a guarantee that the same surrounding atom occupies that slot throughout history.

A width-128 context head alternates two spatial-attention blocks with two causal temporal-attention blocks. Spatial attention operates over 25 slots within a frame; temporal attention operates over observations of each slot. It pools the final current-frame context with total weight 1 for each of center, inner shell and outer shell. A residual carries the initial current-center token into the final output.

**This trained a contextual forecaster on frozen local encodings. It did not make the local 128-dimensional encoder exports context dependent.**

Source: [stencil and assignment](../../src/research/structured_context/geometry.py), [attention](../../src/research/structured_context/model.py), [reuse input producer](../../src/research/structured_context/reuse_data.py), [reuse protocol](../../experiments/structured_relaxed_reuse_20260922/README.md).

## How relative geometry entered

Let `q_i` be the nominal query position and `r_i` the actual representative atom's center-relative position. The attention head received four scalar token features:

1. `|q_i| / 20`.
2. `|r_i| / 25`.
3. `|(r_i - q_i) / 4|²`.
4. `((r_i - q_i) / 4) · (q_i / 20)`.

Each attention pair also received a learned bias from:

* Actual squared pair distance, `|(r_i-r_j)/25|²`.
* Nominal squared pair distance, `|(q_i-q_j)/20|²`.
* The dot product between assignment offsets, `((r_i-q_i)/4)·((r_j-q_j)/4)`.
* Historical time difference divided by 48 ps, and its square.

Thus it could distinguish spatial arrangements of representative points, rather than merely average a bag of embeddings. All-pairs distances can encode angular layout implicitly. However, the local embeddings were scalar invariants: rotating an individual local neighborhood can leave its token unchanged. Pair distances between token centers cannot recover the local orientation information already discarded. The model did not explicitly couple the crystal orientation inside one patch to the orientation inside another.

The full predictor also had additional inputs. In relaxed-reuse, these were current geometric descriptors, two observed-history descriptor differences, four outer-shell descriptors, and the historical seven-component condition vector: five temperature indicators plus normalized simulation age and its square. Older interfaces allocated an additional unused descriptor-difference block and unused new-encoder block. The actual nonzero inputs are visible in `ReusePaths.raw_information` and `ReusePaths.observed`.

These are **historically condition-aware results**. Future designs must omit temperature, simulation age and explicit time covariates under the current policy. Observation timestamps may organize causal inputs/labels; they must not be silently reintroduced as learned features. Historical results must keep their original input record.

## What the newer hierarchical implementation changed

The September 24 spatial-hierarchy encoder uses three **nested balls around the same focal atom**, not 25 independently encoded patches. Each region supplies a 49-dimensional geometric token:

* Six smooth radial masses.
* Twenty-one cross-radial Gram entries for degree-2 harmonic moments.
* Twenty-one for degree-4 moments.
* One smooth count.

Radius sets were `[4,6,8]`, `[8,10,12]` and `[8,12,16]` Å for the local/near/wide comparisons. The early-context version injects attention-derived scalar updates and gates into the local atom features before subsequent MACE operations. Therefore context can change the eventual exported state. A late-context arm provides the corresponding comparison.

This is inexpensive broad context, but each token contracts its moments independently. It captures local ordering strength and cross-radial alignment inside a region; it does not fully preserve orientation relationships between separate region tokens and the detailed local features. Source: [regional tokens and feedback](../../src/research/spatial_hierarchy/model.py).

## Proposed next encoder

Use a fine local MACE graph plus a sparse, broader context graph **inside one trainable encoder**, with context feedback before final pooling. Preserve scalar and selected vector/tensor channels until local/context information has interacted.

For context nodes `i,j`, use actual periodic displacement `r_ij`, radial weights and spherical harmonics `Y_l(r_ij)`. Tensor-product messages can then couple local orientation features to the direction and orientation of surrounding regions. Finish with an invariant focal readout. The output remains a single state suitable for the existing hazard head, without temperature/time covariates.

Two useful levels of implementation ambition:

| Design | Why try it | Cost / limitation |
| --- | --- | --- |
| Extend current nested-region tokens with equivariant moments and cross-region/local contractions | Reuses existing smooth multiscale geometry; retains relative orientation evidence discarded by independent Gram features | Coarse moments may hide multiple distinct structures; normalization must preserve scale/count information. |
| Shared atom backbone, sparse contextual nodes, joint fine/coarse message passing | Learned surrounding structure and reusable atom computations across focal centers | Larger architectural change; current center-conditioned, patch-tapered backbone cannot simply share features across patches. |

Prefer smooth radial support and contributions inside attention normalization. The old nearest-representative assignment can jump when atoms swap slots; averaging those already-selected tokens does not remove that boundary discontinuity. A smooth neighborhood aggregation onto context nodes would address the assignment itself, at extra compute cost.

Relevant primary literature:

* [PaiNN](https://arxiv.org/abs/2102.03150): retains equivariant directional channels instead of relying solely on invariant atom states.
* [Long-Short-Range Message Passing](https://arxiv.org/abs/2304.13542): combines local atoms and coarser groups to capture nonlocal information efficiently.
* [Ewald-based Long-Range Message Passing](https://arxiv.org/abs/2303.04791): reciprocal-space messages provide another route to global context. This is a larger observation/computation change and is not the first choice for a bounded local onset task.

These papers motivate mechanisms; they do not establish which will improve our Al AP3/AP6. Compare local-only, invariant coarse context, equivariant coarse context and shared-backbone context with matched sources, supervision and output dimension. Log support/halo, context construction, motion, relaxation and training-only teachers separately. Measure AP3 first, AP6 second, plus 0.75 ps temporal stability, normalized input-noise response and assignment/boundary sensitivity. See [runtime trade-offs](performance_refactor_20260925.md).
