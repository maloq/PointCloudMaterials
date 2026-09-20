# Structural neighbor pretraining metrics, version 1

This protocol trains on centered, material-cutoff-normalized atomic observations.
Targets are calculated in physical Angstrom before normalization. It is distinct
from the older 128-channel geometry/velocity predictive-memory packet.

Geometry has 85 channels: radial RBF32, pair-distance RBF32, Legendre16, and five
smooth-count/radial moments. Instantaneous TDA has 144 channels: H0 curve16 and
H1/H2 images64 each, with the existing physical nearest-80 and 3.5 A death cutoff.
There are no relaxed-topology or velocity targets.

Training-only means and population standard deviations are fitted to the frozen
release's valid target observations; standard deviations are floored at 1e-4.
Geometry statistics use current/spatial/future endpoints, never the unlabeled
past frames. TDA statistics use only valid instantaneous labels. Selection rows
and all descendants of held-out lineages are excluded from both fits.

For each observation, block error is mean squared error over the block's
standardized channels. Physical error averages its four blocks equally; topology
error averages its three blocks equally. Each view predicts its own targets.
Training averages physical error across both endpoints; topology averages over
valid target rows. The task anchor is physical + 0.25 * instantaneous TDA.

Selection exports use current observations only, with three causal inputs for
GATr JEPA and one for VICReg. Errors are averaged within each independent source
first, then equally across the fifteen selection sources. Selection score is
physical + 0.25 * instantaneous TDA. This selects a checkpoint; it is not a test
score or a source-bootstrap interval. State_std_mean is the mean per-coordinate
population standard deviation over these selection states, not a whitened or
within-material kinetic quality score. Stored row indices identify the release.

VICReg is (25 I + 25 V + C)/51, where I is paired-projector squared distance,
V averages relu(1-sqrt(sample_variance+1e-4)), and C is off-diagonal sample
covariance squared sum divided by projector width. V/C average over endpoints.
The logged `vicreg` total is 25 I + 25 V + C, using the reference VICReg module's
total-loss logging convention. This protocol retains sample variance (N-1);
the older reference uses population variance (N), so their numerical totals
need not match on identical inputs. `representation` remains this total divided
by 51; `vicreg_weighted`
is its actual contribution to the combined loss, 0.1 * vicreg / 51. These two
additional diagnostics apply only to VICReg and do not change its gradients
or training coefficient.
The full statistical batch contains 128 distinct anchor records from a single
material/potential family. This excludes between-material means from its
variance floor, but does not make correlated atomic observations independent.

Temporal JEPA uses 0.95 * next-projector MSE + 0.05 * SIGReg. The predictor sees
the anchor's three-frame causal state and actual elapsed ps; the next snapshot
is separately encoded as its target. Static batches instead have a spatial
partner, no future likelihood, and SIGReg alone. SIGReg is the pinned authors'
sliced Epps–Pulley implementation, 256 directions, 17 trapezoidal points on
[0,3] with symmetry weighting, averaged over endpoints. Its internal step buffer
is preserved in checkpoints. Representation weight is 0.1 in the total loss.
Latent_persistence compares the two current projected states within that model;
raw latent errors cannot rank encoders with different learned target spaces.

Exact full-batch gradients are obtained by a detached encoder pass, one joint
head/regularizer backward, and encoder recomputation with cached state gradients
and replayed RNG. Independent microbatch regularizers are not averaged.
In the BF16 shared-pretraining restart, VICReg explicitly disables autocast and
uses FP32 projector values for its moments, covariance and total. The scalar
readout/head normalization and float64 health diagnostics are defined in the
[shared protocol](shared_pretraining.md); the task loss weights stay unchanged.

Physical/TDA MSE is lower-is-better. Update counts, input waiting, GPU names,
label counts and training timing are operational diagnostics, not independent
research replications or hardware-benchmark results. One seed measures no
training-seed uncertainty. The three requested runs change architecture and,
for JEPA, input history and objective; there is no physical-only ablation here.

Architecture revision `structural_v4_geometry_fp32_2x` uses the enlarged snapshot
encoders, explicit history allocation, protected FP32 geometric/readout paths and
compensated BF16 GATr scalar products documented in
[the architecture record](../shared_pretraining_geometry_fp32_2x.md). The physical,
TDA and representation formulas and coefficients above are unchanged. Its
`structural_neighbors_v2` training identity includes the precision adapter source
and CUDA kernel versions. Earlier exports retain their original definitions.


The shared trainer's v5 correlation auxiliary and additional variance diagnostics
are defined in [shared pretraining](shared_pretraining.md#compiled-repair-protocol-v5).
The standalone structural trainer retains correlation weight zero unless its
Objective is explicitly constructed with another weight. The original VICReg
formula and target-block reductions are unchanged.

The current v6 structural model conditions its training heads across observations;
validated head checkpoints require training-only moment refresh. The Al-only
shared trainer and subset normalization are defined in
[the v6 shared protocol](shared_pretraining.md#al-only-conditioned-head-protocol-v6).
Historical exports retain their frozen definitions and source.

## Full instantaneous-TDA expansion (September 18)

`structural_neighbors_full_tda_v1` retains the parent 250,000 training anchors
and 480 native-Al selection anchors, then adds 37,500 shooting anchors. New
shooting draws balance the 17 eligible training lineages and cover all 456
eligible trajectories, using source/frame combinations absent from the parent.
The total is 287,500 training anchors across Al/Mg/Ti/Ta/Zr. Every supervised
anchor, spatial partner and next-frame endpoint has instantaneous TDA144 from
the same physical nearest-80 producer. Earlier two history frames are context,
not supervised endpoints. The old approximately 25% training-label mask is
replaced by complete endpoint coverage. Parent arrays/labels remain immutable.
Fixed material scales and held-out ancestry assignments are inherited; target
means/stds are refitted on all expanded training endpoints. Objective weights,
descriptor definition and block reductions remain unchanged. Selection remains
native Al only, not a test of generalization to other metals.

The v7 shared GATr triplet protocol can filter static shards before opening
arrays and refits the same target moments on included dynamic training endpoints.
The cache's context-only previous frame is available as an independent snapshot;
its placeholder descriptor is never a supervision target. Existing structural
protocols keep their declared populations. See [v7 definitions](shared_pretraining.md#mixed-material-dynamic-triplets-v7).

The optional central bond-order targets and equivariant MACE cache interface are
used only by the mixed v9 shared-pretraining protocol, documented in
[shared_pretraining.md](shared_pretraining.md). Existing structural calculations
and the default 128-dimensional encoder output are unchanged.

## Local structural support (v10)

Current structural MACE/GATr observations use fixed material normalization
`x_model = x_A * 9.192189 / scale_material`, crop to radius <8 before packing,
and quintic C2 weights equal to one through radius 6 and zero at radius 8. There
is no outer halo. MACE uses 5-unit edges, two layers and pooling tapers 0–3,
3–5, 6–8; GATr globally attends only within the cropped sphere and scales its
weighted count by 100. Training, static inference and trajectory inference share
`src/data/structural_pretraining/support.py`. Geometry baselines using the
encoder's support and radial controls now also use that local support. Existing
85-component physical and 80-point instantaneous-TDA targets are unchanged.

The revision is incompatible with previous large-support checkpoints. Historical
exported metric contracts and results retain their original support definitions;
reproduction of those runs requires their frozen code. Current within-domain
VICReg, selection, bond-order and temporal-only curvature metric formulas are
unchanged. Curvature weights are recalibrated at initialization using training
batches under the declared 2%-loss / 10%-encoder-gradient policy. See
[local protocol](../shared_pretraining_local_structure_20260918.md).

## Local GATr bond supervision (v11)

`shared_pretraining_local_gatr_bond_v11` adds the same q4m/q6m targets and
`12 * mean_m(error^2)` per-order loss as local MACE, with mean over orders and
loss coefficient 0.1. Only current/partner snapshots are supervised; the past
snapshot remains curvature context. Its 704 training-only tensor features are
formed from the final learned atom multivectors' four vector sectors across
eight channels: separately replace each vector v by `v/sqrt(sum(v^2)+1e-4)`,
compute real component-normalized solid harmonics l=4,6, and average atoms using
the encoder's local support weights. Concatenate 32x4e then 32x6e; an equivariant
linear readout predicts 1x4e+1x6e. All these operations use FP32 under BF16 AMP.
No raw target vectors or invariant z enter this head. Powers are taken before
pooling, so inversion-symmetric local order need not vanish. Export remains
128-dimensional; no additional encoder pass is required. Selection remains
physical+0.25*TDA, with bond errors and magnitudes reported separately. Training
is from scratch for five epochs with the small curvature coefficient recalibrated
on training batches, including the bond loss in its base-gradient comparison.

The 19 September dynamic expansion samples additional unused source/frame pairs
within training-only strata and ancestral lineages. Short-trajectory capacity
is redistributed within its lineage. Parent targets and material scales are
unchanged; every new supervised endpoint receives the original instantaneous
nearest-80 TDA144. Static shards may be inherited for provenance but are excluded
from this training subset. See the expanded dual-GPU MACE operational recipe.
