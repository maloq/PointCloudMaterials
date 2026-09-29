# Spatial VICReg mechanism study: completed findings

All nine GeoFormer encoders completed 24 full passes (108,552 updates each),
with all 63 checkpoint assays, both raw/projector exports, four K values, six
classical controls and all 18 primary-endpoint W&B evaluation updates complete.
Initial weights match across treatments within each seed. Primary population:
24,960 observations from 30 fixed held-out Al source ancestors; 12,266 have no
PTM-detected crystal anywhere in their 80-atom input. Training used all phases,
geometry only and no future labels. These are historically examined test sources.

[Summary figure and frozen numerical export](../../output/spatial_vicreg_bias/matched-al64-20260929/analyses/completed-review-v1/README.md)
· [Protocol](README.md) · [Execution](../../docs/spatial_vicreg_bias.md).

## Answer to the research question

Neighbor alignment changes the spatial geometry of the representation, but it
is not necessary for the disappearance of meaningful global liquid clusters.
The strongest observed failure is that global clustering loses liquid resolution
while the continuous encoder still supports useful liquid-structure readouts.
It is incorrect to infer complete information erasure from the collapsed-looking
liquid cluster alone. The results also do not establish distinct precursor states.

At epoch 24, arithmetic means across the three paired seeds:

| Pair-alignment treatment | Normalized neighbor squared distance, raw z | Crystal-free TDA error reduction from continuous z | Crystal-free TDA error reduction from K=7 membership | Crystal-free geometry error reduction from continuous z |
| --- | ---: | ---: | ---: | ---: |
| Same-center only, alpha=0 | 0.459 | 15.24% | 0.0000% | 25.45% |
| Half neighbor alignment, alpha=0.5 | 0.128 | 13.29% | 0.0004% | 27.98% |
| Neighbor alignment, alpha=1 | 0.155 | 12.41% | 0.0001% | 29.69% |

Error reduction is relative to the training-source mean predictor on the same
held-out population, with the original target standardization and equal source
weighting. It is not classification accuracy, AP, conventional test-mean R² or
a crystallization prediction score. Pair distances are normalized by training
variance; their reduction describes similarity relative to overall embedding
spread, not an absolute physical length or a literal diffusion coefficient.

## What the interventions establish

**1. Neighbor pairing adds spatial contraction.** Turning on neighbor alignment
reduces normalized neighboring-embedding distances by roughly 66–72% relative to
same-center training. Both neighbor treatments do so in all three paired seeds.
The response is not monotonic: alpha=0.5 contracts more than alpha=1 on this metric.
The projector shows an even stronger contrast: approximately 0.849, 0.168 and
0.154. This supports a learned smoothing effect under the matched-view protocol.

**2. Global liquid-cluster resolution disappears even without neighbor pairing.**
At epoch 24, every raw encoder places 99.96–100% of the 12,266 strict-clear liquid
observations into a single global K=7 cluster. All seven clusters are occupied
in the full test population, and mixed noncrystalline-center inputs occupy
multiple clusters. This is a loss of resolution within liquid, not total
representation collapse. K=3/6 show the same limitation; K=10 sometimes divides
liquid and recovers about 2.4–2.6% TDA skill, but inconsistently across seeds and
treatments. Therefore the conclusion is about this global clustering procedure,
not every possible clustering of these embeddings.

**3. Continuous information improves while cluster information disappears.**
For same-center training, raw-z liquid TDA readout skill increases from 9.77%
at epoch 4 to 15.24% at epoch 24, while global K=7 membership skill falls from
2.30% to zero. The continuous improvement is +5.47 percentage points, with a
paired-source bootstrap interval of +4.22 to +6.53 points. The epoch4 comparison
is descriptive; the predeclared primary endpoints remain 12 and 24.
This directly challenges the interpretation that worsening cluster appearance
necessarily means the encoder has lost the measured liquid information.

**4. Alignment redistributes accessible structural information.** At epoch 24,
same-center raw z beats full-neighbor raw z on liquid TDA readout by 2.83 points
[1.79, 3.65], while full-neighbor training beats same-center on geometric-feature
readout by 4.24 points [3.99, 4.49]. The geometry family excludes the density
identity targets. Liquid bond-order skills are similar (about 5%); CNA is about
14.7–15.7%. There is no universally best treatment across these measurements.
The TDA ordering persists on the original all64 and legacy16 tracks, although
absolute scores differ with their populations. Intervals resample source IDs,
paired across treatments, after averaging seeds; they condition on these three
trained seeds and are not multiplicity-adjusted significance declarations.

**5. Many broad cluster–descriptor associations are reproducible by a simple
crystal-fraction field.** Across all phases, K=7 clusters of the input crystal
fraction predict the bond-order family with 61.5% error reduction, compared
with 64.5–66.6% for learned raw-z clusters. Corresponding TDA values are 6.97%
versus 7.41–7.75%. This privileged classical control uses PTM labels; diffusion
controls also reach outside the encoder support. They are not equal-input
learned competitors. They show why broad descriptor agreement and colored
intermediate bands alone do not prove discovery of a distinct state. Some
learned-cluster gains beyond the fitted controls remain positive; finite ridge
readouts can also gain from nonlinear feature expansion, so that is not a
conditional-mutual-information result.

**6. Strong interface broadening is not established.** Raw-z 10–90% phase-profile
widths average about 6.41, 6.66 and 6.72 Å for alpha=0/0.5/1. These are small
changes in a 2 Å-binned pooled distance-contrast profile, without a width
uncertainty estimate. Projector widths are nonmonotonic with alpha. We should
not claim a large or universal smoothing-induced shell thickness. The scalar
phase coordinate is defined from bulk reference means, and the distance contrast
is not a reconstructed signed interface distance.

## Encoder versus projector

For full-neighbor training, the effective rank within crystal-free liquid is
about 8.47 in raw z but only 1.08 in the projector. Raw z also predicts the tested
liquid descriptors better than the projector: e.g. TDA 12.41% versus 10.01%,
geometry 29.69% versus 17.46%. Low rank and retained scalar readout signal can
coexist; neither number alone proves or disproves meaningful liquid structure.
The trained loss space and the exported representation must remain distinguished.

## What remains unresolved and the next discriminating check

The evidence favors a combination of learned spatial contraction, phase-dominated
representation geometry, and coarse global partitioning. It does not establish
that neighboring-view alignment alone causes the liquid-cluster failure, that
all interfacial structure is artificial, or that the model detects precursors.
No new future prediction, new-nucleus test or Ta/Zr transfer was performed here.
The pure-liquid annotations were evaluation strata, never training exclusions.

Before changing the training loss, the most direct follow-up is to keep these
encoders frozen, fit a liquid-only clusterer on training-source liquid inputs,
and evaluate its descriptor predictions on the same fixed held-out sources.
Compare native z with projector y and a declared train-fitted metric adjustment.
If resolution returns, the global metric/partition was a major bottleneck; if it
does not, useful information may remain continuous rather than form distinct
liquid clusters. This follow-up has not been launched by this results review.
