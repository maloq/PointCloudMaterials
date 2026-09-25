# A focal atomic state embedded within its surroundings

Question: does a representation of the **same central 8 Å neighborhood** become
more predictive when it is conditioned on a hierarchy of surrounding regions?
The output remains one 128-dimensional state for the central neighborhood, with
the same central physical reconstruction and local-onset targets. The hierarchy
is spatial; there is no nested-prefix embedding loss or changing target region.

The motivation is local/global exchange, related in spirit to
[Hierarchical Molecular Graph Self-Supervised Learning (HiMol), 2023](https://doi.org/10.1038/s42004-023-00825-5),
which uses a graph hierarchy for bidirectional local/global information flow.
That work concerns molecular graphs and motifs. Here the regions are smooth
three-dimensional neighborhoods in a metal; this is a new hypothesis to test,
not a reproduction or an established crystallization result.

## How surroundings enter the encoder

1. Construct the original 8 Å atom graph, with exactly the original coordinates.
2. Describe three nested surrounding regions using smooth radial and harmonic
   moments of the **current observed coordinates**. A token has six radial
   channels, 21 cross-radial l2 contractions, 21 l4 contractions and a smooth count.
   Fit-only normalization and trainable MLPs turn these into three region tokens.
3. At each native MACE block, aggregate local atom scalars upward into the region
   tokens. Atom queries attend to those tokens, which then update scalar features
   and gate vector/tensor channels before the next spatial computation. The gates
   are invariant, so the equivariant tensor channels preserve their rotation law.
4. Pool the central atom and the original local region. A learned context residual
   adjusts this focal summary, followed by the usual 128-channel readout.

The coarse context path uses a fixed, rotation-invariant geometry basis followed
by trainable updates. It does not run MACE on every atom out to 16 Å, import frozen
pretrained context embeddings, use PTM labels as inputs, or access future frames.
It loses some outer geometric detail; testing richer learned outer-atom features
is a later experiment if this cheaper hierarchy helps.

The existing 4/6/8 Å tensor-pooling experiment rearranges information inside one
8 Å graph. This study adds actual observations beyond that graph and lets them
change the local spatial computation before the final pooling.

## Four matched one-seed fits

| Arm | Region radii | Where context enters | Question |
|---|---|---|---|
| L-local | 4, 6, 8 Å | Between spatial blocks and at pooling | Control with the same hierarchy machinery but only local information |
| W-wide-late | 8, 12, 16 Å | After local spatial message passing | Is a larger descriptor appended at the end sufficient? |
| H-hierarchy12 | 8, 10, 12 Å | Between spatial blocks and at pooling | Does moderate surrounding context help? |
| H-hierarchy16 | 8, 12, 16 Å | Between spatial blocks and at pooling | Does larger context help; does early interaction beat late fusion? |

Every arm has the same declared parameter shapes, shared initialization seed and
the same focal MACE graph. The late arm leaves the atom-update modules unused, so
equal allocated parameter counts do not imply equal effective capacity or FLOPs.
All radii use a quintic smooth boundary and fixed volume-based count normalization.
The inner region always remains the target; a larger surrounding crystal region
is not relabeled as a local event.

Use the same 2,880 observations and 25/5/15 independent-root fitting/tuning/reused
development splits as the robust-onset screen. Train 2,048 updates with present
geometry/order, 3/9/12 ps future physical increments, event likelihood and full-risk
Smooth-AP ranking. No relaxed teacher or noise augmentation in this comparison.
Training settings, physical-retention selection gates and probe recipes are held
fixed. Only the hierarchy/context policy changes. No new simulation is needed.

## Checks and interpretation

- Verify the inner coordinates against the original parent cache before training.
  Register raw source manifest and consumed frame/box/atom identity checksums.
  The actual training graph is reused unchanged. Independent wider extraction
  reproduces its coordinates within 3.82e-6 Å (periodic-wrapping arithmetic);
  rebuilt inference is also checked with the clean replay diagnostic.
- Test rotation/permutation invariance, smooth support, gradients into the context
  branch, and an explicit intervention showing that context reaches the input to
  the second spatial block.
- Evaluate source-weighted central-onset AP, Brier, recall/FPR, frozen probes and
  central present/future retention. Bootstrap differences against L-local.
- Retain exact 0.75 ps stability and dataset/movement spectral dimensions. Reuse
  the original dense chart's ~16.87 Å candidate support, making a new 16 Å input
  cache rather than silently using the previous 10 Å native export.
- Evaluate input noise relative to local twelve-neighbor spacing, perturbing the
  full observed region and rebuilding both the local graph and context summaries.
- After training, hold every coordinate inside 8 Å fixed and replace only the
  8–16 Å surroundings with those from another development root at the same
  temperature. Report normalized embedding response and clean replay error.
  This is an artificial intervention, not an achievable molecular trajectory;
  sensitivity alone is not success. Predictive improvement must justify it.

There are only 30 fitting and 18 development onset windows, with one seed and
reused development sources. This is a mechanism screen, not proof of a generally
best encoder. The dense chart is descriptive and may overlap fitting roots; it
does not select checkpoints. [Inherited metric definitions](../../docs/metrics/robust_onset.md)
and [hierarchy-specific definitions](../../docs/metrics/spatial_hierarchy.md).

Configuration: [screen_20260924.json](../../configs/spatial_hierarchy/screen_20260924.json).
Reproduction and Slurm operations: [guide](../../docs/spatial_hierarchy.md).
Results: `output/encoder_research/spatial-hierarchy-20260924/RESULTS.md`.
