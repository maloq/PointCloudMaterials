# Encoder families and how we trained them

[Handbook](README.md) · [Results](results.md) · [All study records](studies.md)

An **encoder** maps the declared observations to an exported state `z`. A
**projector** transforms that state for a representation loss. A **training head**
decodes a supervised target during encoder training. A **frozen probe** is fitted
after fixing the encoder. A **forecaster** predicts future physical quantities or
latent trajectories. Results on these different outputs are not interchangeable.

“Native MACE” here means our task-trained MACE atom-feature pathway, retaining
scalar/vector/tensor channels and trained pooling. It does not mean a frozen
pretrained MACE descriptor, a potential's energy output, or one immutable
architecture used by every study. Channel counts, normalization, support, history
and initialization belong to the specific configuration. cuEquivariance changes
execution of equivariant operations; enabling it does not add an encoder objective.

## 1. GeoFrame and GeoFrame V2

**Computation.** Local point clouds are grouped into patches, canonical frames are
selected, and patch information is processed by a geometry-aware transformer and
pooled. V2 enriches pair geometry with relative frames, displacement, radial and
shape information, and optional chirality/parity handling. Hard neighborhood,
ordering and frame choices can change discontinuously. Rotation invariance at
ordinary inputs does not imply continuity at those choices.

**Training variants we actually retained.** Static spatial VICReg/VISReg, with
FactorVAE-style factorization in three historical selected models; later
spatiotemporal VICReg and VISReg fine-tunes; scratch task-supervised and predictive
controls. FactorVAE is an objective component here, not the encoder architecture.
The restored gallery contains four analyses of **three** trained models:
multiscale VICReg epoch059, V2 VICReg epoch034, and V2 VISReg epoch159; the fourth is
a reanalysis of epoch034. The later temporal fine-tunes explicitly omit FactorVAE.

In our [FactorVAE loss implementation](../../src/training_methods/contrastive_learning/vicreg_module.py),
a discriminator distinguishes joint projected embeddings from embeddings with
independently shuffled coordinates. The encoder/projector receives an adversarial
total-correlation penalty; discriminator and representation updates alternate.
Grouped coordinates, spectral normalization and latent noise are run-specific
settings. This is not a point-cloud variational autoencoder with an atom decoder,
and factorization of projected coordinates does not establish factorization or
physical completeness of the raw encoder output.

VICReg combines view agreement with variance and covariance terms. In the temporal
fine-tune, each anchor is paired with a nearby spatial center and the same atom
at a later observed frame; losses act on projected states. Deployment still takes
one snapshot. Feeding temporal pairs during training does not by itself create a
history-dependent encoder. The reported 80-point fine-tunes differ from the older
160-point temporal-stability assay.

**What happened.** Static Al separation and normalized projector drift improved
for the VICReg fine-tune, while raw encoder drift/rank gave a more mixed picture.
A controlled interpolation audit isolated finite jumps at canonical-frame
switches, including float32 Ta. Holding or transporting frames removed almost all
of the selected frame-boundary jump; grouping changes remained another mechanism.
The motion-transport experiment was an intervention, not a validated replacement
streaming encoder. GeoFrame V3 remained a proposal in the cited design record.

The [35-pass reproduction](../../output/geoframe_evolution/epoch34-review-20260923/RESULTS.md)
now provides a fixed-objective trajectory: encoder liquid-order fidelity improves
while projector fidelity falls in Al/Ta/Zr. Fault readout improves but liquid
subtypes remain mixed in K=7. The archived epoch034 VICReg and epoch159 VISReg
models are different objectives, not an early/late pair. This does not isolate
FactorVAE's causal role; that needs a matched ablation.

**Evidence and implementation:** [restored FactorVAE galleries](../../output/factor_vae_archive/index.html),
[GeoFrame V2](../../src/models/encoders/geo_frame_transformer_v2.py),
[grouping/frame producer](../../src/models/encoders/ri_mae_encoder.py),
[completed spatial/temporal comparison](../geoframe_spatiotemporal_vicreg_20260905.md).
Archive continuity and early sweeps are linked in the [study index](studies.md).

## 2. Shooting, predictive-atlas and continuous-geometry controls

The early shooting work asked whether local representations distinguish the
**distribution of futures** from a common starting position, using repeated
velocity branches. Frozen GeoFrame, history-based predictors, multiscale context,
future-law retrieval and limited last-block encoder fine-tuning were distinct
interventions. Much of this work fitted downstream models; it must not all be
counted as encoder training. Descendants share ancestry and belong in one split.
See the historical [predictive-atlas account](../predictive_atlas_current_progress_20260903.md)
and the STORE collections indexed under `shooting_atlas`.

The September5 task-supervised comparison trained **reference MACE, SchNet-style
continuous filters, smooth-density MLP and GeoFrame** from scratch. Fixed SOAP,
density-PCA, TDA and coarse-order controls received learned readouts, not new
atomic encoders. Current-geometry warm-up preceded future shooting targets;
three learning-rate candidates and three final seeds were used. Nine
representations produced45 trials/27 selected models. These are not27 distinct
backbone designs.

The subsequent temporal-hypothesis campaign tested static/temporal agreement,
EMA future-latent prediction, motion and sensitivity losses, data amount, larger
halos and wider MACE. TDA/SOAP/order were held out of encoder training in that
campaign and used for frozen evaluation. Predictive density and MACE improved
some downstream scores; density was a strong control. Screen and confirmation
budgets differed. References: [task-trained results](../predictive_encoder_training_20260905.md),
[smooth-density implementation](../../src/models/encoders/smooth_density.py),
[reference MACE](../../src/models/encoders/mace_encoder.py).

## 3. Pretrained scalar MACE, VICReg and topology denoising

**Computation.** Earlier models start from MACE-MP-0b2 small weights and export
pooled scalar node features, commonly two128-channel blocks. A later projector
is a different representation. This exported scalar pathway is not the later
interleaved vector-history architecture.

**Sequence of trials.** The archive retains pretrained spatial/temporal runs,
strict80-point support,0.1ps pairs/cosine schedules, objective removals, topology
continuations, balanced/joint-property objectives, target-complete support,
plain80, normalized original VICReg, VICReg+TDA, hot/relaxed variants and full-data
Al/Mg/Ta/Ti training. Throughput/BF16 experiments are execution variants, not new
scientific encoder families. Old commands require their frozen code/configuration;
they are not automatically supported by today's trainer.

**Denoising and temporal variants must be distinguished.**

| Variant | What is learned; where history enters |
| --- | --- |
| Frozen MACE denoising | MACE stays frozen; heads recover relaxed topology from scalar features |
| Single-frame VICReg | MACE fine-tunes on view agreement; optional PCA32 or balanced144D TDA head |
| Five-frame mean | Trainable frame encoders, followed by an average |
| Pooled temporal transformer | Attention after each frame has already been pooled |
| Temporal residual | Current pooled embedding plus a learned history correction |
| Atom-temporal fusion | Attention over tracked atoms' scalar MACE features before spatial pooling |
| Atom-anchor control | Same fusion capacity with repeated current features |

Atom-temporal fusion is earlier than group pooling but still later than that
frame's MACE spatial computation. It is not the later spatial/temporal block
interleaving. The September9 temporal-transformer pilot was stopped after16 epochs:
regularization dominated gradients and frozen topology probing deteriorated.
The September10–11 matched24-fit study found useful but modest history denoising;
five-frame mean was a strong baseline and trained-head checkpoint selection
obscured representation quality. Relaxed-TDA reconstruction is not atomic-coordinate
reconstruction and does not establish crystallization prediction.

References: [pretrained pathway](../../src/models/encoders/pretrained_mace.py),
[temporal models](../../src/models/encoders/mace_temporal.py),
[frozen denoising](../../src/models/encoders/mace_denoising.py),
[24-fit protocol/results](../../experiments/mace_vicreg_relaxed_20260910/RESULTS_20260911.md).

## 4. Complete-context, center/inner and velocity encoders

The context work separates the **computational halo** needed for message paths
from the **weighted inner region** whose state is summarized. Smooth cutoffs
reduce artificial membership jumps. Tracked-center features preserve different
information from a smooth inner average. Concatenation can expose this
complementarity; using two separately trained backbones is a different capacity
comparison from two readouts of one backbone.

`dual_ssl` learns physical heads from detached features while retaining the
representation objective. `dual_physics` also sends physical reconstruction
gradients into the shared backbone. They reconstruct instantaneous/relaxed TDA
and local order/density. Increment readouts receiving **both** endpoints diagnose
change information; they are not forecasts.

The coordinate/velocity model fine-tunes the backbone and exports **256 structural
+32 time-even activity +16 time-odd flow channels**. Geometry-conditioned relative
velocity messages and parity constructions enforce boost/reversal behavior. It
uses current physical targets and a fixed teacher, without future supervision.
The separate motion channels do not mean motion changes the main structural
state in the same way as the later causal model. Both velocity and coordinates-only
controls completed. A native training-data-amount study is also retained.

References: [context](../../src/models/encoders/mace_context.py),
[recovery protocol](../../experiments/mace_context_recovery_20260914/README.md),
[velocity protocol](../../experiments/mace_velocity_20260915/README.md),
[data-amount study](../../experiments/mace_data_amount_20260916/README.md).

## 5. Discarded maps on frozen encoder features

PCA, learned physical distances, temporal canonical maps, direct slowness,
curvature, direction and motion-subspace objectives transformed already frozen
states. These were explicit representation experiments, but did not train atom
message passing. Smoothness often improved while instantaneous topology worsened.
The initial direct-smoothness sweep failed its information gate; the consecutive
motion study completed44 fits and none passed the information gate or joint0.10
jump requirement. Capacity fits without a final established evaluation remain
incomplete evidence. Preserve these negative controls; do not call them native
encoder improvements. [Discarded scope](../discarded_frozen_encoder_maps.md).

## 6. Native causal MACE and predictive memory

**Causal MACE** trains geometry, relative velocities and atom history jointly,
with scalar/vector/rank-two features, causal temporal messages between spatial
blocks, smooth support and final multiscale pooling. A later spatial block sees
history-informed atom states. Parameters belong to the actual exported encoder.

The A–E progression tested present reconstruction; added multi-horizon future
physical targets; explicit velocity; real history; then extra slowness training.
The repeated-anchor model is a separately trained capacity control. E received
additional updates, so E−D does not isolate the penalty. Gaussian heads,
stronger frozen probes, raw-history add-backs and width/budget comparisons were
separate evaluations. All use causal inputs; “causal” is a time-order statement,
not proof of causal identification or a sufficient Markov state.

**Predictive memory** changes the scientific protocol: a17Å total observation
radius,0/12/48ps histories, fixed128-coordinate physical targets at five lags up
to96ps, and a four-component path distribution with low-rank covariance. Joint
path NLL plus present reconstruction trains the encoder. The old causal study's
169-coordinate/block-weighted MSE is not this score. Initial weak reconstruction
allowed heads to make little use of sample-specific state; stronger retention
helped one completed seed, but that objective's replication was not completed.

[Architecture](../mace_causal.md), [memory protocol](../../experiments/predictive_memory_20260917/README.md),
[consolidated results](../../output/predictive_memory/research-summary-20260917/RESULTS.md),
[final stopped follow-up](../../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md).

## 7. Local-predictability MACE/GATr and shared structural pretraining

The local-predictability study first established descriptor forecasting and
observability controls, then trained raw-atom MACE and axial GATr states for
physical targets or hazards. Current-state recognition worked well; native onset
and future prediction still lagged strong fixed packets at the tested budget.
Raw future-input oracles measure label accessibility, not forecasts.

**Shared structural pretraining** moves representation learning away from onset
labels. MACE uses equivariant many-body messages; GATr uses geometric-algebra
streams and attention. Snapshot spatial/temporal VICReg variants and history JEPA/
SIGReg variants use fixed physical and instantaneous-TDA anchors. Exported z128,
projected states and typed equivariant streams must be identified separately.

Revisions include the original five-metal release, unstable12-epoch large-LR
runs, calibrated-head/precision repairs, Al-only v6, mixed temporal-backtracking,
local-support/bond-order supervision, mixed MACE and expanded-data MACE. These
change support, material sampling, initialization and heads; “MACE versus GATr”
without the revision/checkpoint is underspecified. The failed runs exhibited
saturated heads/collapse. Later local update622 MACE retained appreciable angular
information beyond radial controls, while matched GATr remained largely radial
and smoother. More smoothness did not establish improved onset prediction.

[Local-predictability report](../../output/local_predictability/research-summary-20260917/RESULTS.md),
[shared-pretraining records](../../experiments/shared_pretraining_20260918/README.md),
[local622 diagnostic](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md),
[training implementation](../../src/training_methods/shared_pretraining/).

## 8. Neighborhood JEPA and regularization variants

An independently evaluated snapshot MACE encoder supplies invariant and
**typed equivariant** features. Predictors query neighboring centers and times;
physical/TDA anchors prevent the learned target from being the sole objective.
Training can use temporal information while deployment remains snapshot-only.
Not every JEPA variant here uses a frozen EMA teacher: the multihorizon recipe
explicitly uses the same jointly trained encoder with gradients on both sides.

V1 combined multi-material neighbor/time prediction and regularization. V2 fixed
causal normalization, equivariant scale confounds and sample-count dependence of
SIGReg, and isolated current/future-neighbor contributions on native Al. Later
large-batch/capacity trials and a16-arm regularizer/projector/order-anchor sweep
compared SIGReg, VICReg variance/covariance and EpiJEPA-inspired geometry. Three
multihorizon arms added3/6/9ps future-center latent/physical prediction. These
names describe implementations inspired by the methods, not exact reproductions
of every published algorithm. Low rank and weak frozen onset gains motivated
more explicit structural-retention tests.

[V1](../../experiments/neighborhood_jepa_20260920/README.md),
[V2](../../experiments/neighborhood_jepa_v2_20260920/README.md),
[regularizers](../../experiments/neighborhood_jepa_regularization_20260920/README.md),
[multihorizon](../../experiments/neighborhood_jepa_multihorizon_20260920/README.md).

## 9. Relaxed inputs/targets

Hot→hot, hot→cold and cold→cold comparisons separate observed geometry, relaxed
input and relaxed targets. Expanded runs add temperature-conditional SIGReg or
VICReg and compare original parent checkpoints. Full-cell minimization uses a
particular potential; its output is a different observation and requires
preprocessing at deployment. It can convey context beyond the final local crop.
Its benefit cannot be attributed solely to local thermal-noise removal.

The expanded and larger fixed-grid onset assays favor relaxed observations in
several comparisons; explicit cold-geometry descriptors remain strong and exceed
the tested encoders in the larger assay. Raising embedding rank is not a monotonic
improvement in physical neighbors or onset skill. Hot→cold continuation stopped
before completion is not a completed arm.

[Expanded study](../../experiments/relaxed_encoder_expanded_20260921/README.md),
[larger test](../../output/relaxed_encoder/large-test-20260921/RESULTS.md),
[geometry audit](../../output/representation_audit/liquid-geometry-20260922/RESULTS.md).

## 10. Bottleneck-conditioned reconstruction (BCR)

The clean MACE encoder exports an invariant128-vector. A decoder also sees noisy
atom geometry and the noise scale, and uses that code to reconstruct/denoise.
The code is a restricted conditioning route, but the decoder has a rich geometric
input. This is an **architectural information bottleneck**, not automatically a
variational mutual-information penalty. Low denoising loss alone does not prove
that detailed structure is stored in the exported code.

The implementation/overfit checks, independent-root G1 molten-Al pilot, and
conditioning/relaxed-transfer audit are different stages. Controls intervene on
correct, swapped, random, zero and optimized-constant codes, and refit decoders
on frozen initial/intermediate/final encoders. The final code improved a fresh
decoder's noise MSE by3.07% at the largest audited noise, but radial decoding
worsened. An optimized constant almost reproduced the original decoder. BCR
therefore taught some useful conditioning under this test without preserving
all the desired structural information. It has no established onset advantage
from the all-liquid G1 cohort.

[Protocol](../../experiments/bcr_v1_20260921/README.md),
[G1](../../experiments/bcr_g1_20260921/README.md),
[follow-up findings](../../experiments/bcr_followup_20260922/RESULTS.md).

## 11. Fixed structural-state reconstruction, repairs and factorial

The simpler screen trains snapshot MACE directly against **fixed radial17 and
l2/l4 Gram36+36 geometry targets** from its exported state. A observes MD geometry;
B observes relaxed geometry; C adds relaxed targets to observed input; D adds
physical-distance supervision to relaxed input. Neither positions-only inference
nor this initial training uses an observed temporal history. There is no BCR
noisy-geometry decoder bypass and no explicit information-bottleneck/KL penalty.
Finite-dimensional pooling remains an architectural compression.

V1's distance scaling allowed severe amplitude shrinkage, hidden by freshly
rescaled probes. V2 repairs that with fixed fitting statistics, pooled64 plus a
learned64 export, calibrated bounded linear heads, fixed initial distance scales,
a warm-up and actual-head/spread audits. The learned64 is a function of pooled64;
128 exported slots are not128 independent information channels. All four repaired
fits completed4096 updates on25 fitting/5 tuning/15 reused development roots.
Relaxed onset ranking was promising in one seed; ordinary future-order MSE barely
changed, and teacher/distance additions gave little benefit.

The completed **two-seed distance/future factorial** then gave every arm current
original-MD order supervision and added distance,9ps future residuals or both.
The residual is relative to a fixed training-fitted **linear** present-state
baseline, not all information available in the present. Twelve-ps future order,
withheld geometry and onset are evaluation tasks. None passed the predeclared
mechanism criteria; distance gave a small probability-score improvement and future
residual supervision did not produce a replicated benefit. All second-seed MLP
hazards selected the constant-risk step0 checkpoint. This is an important
qualification of the earlier favorable one-seed onset screen.

[V1](../../experiments/structural_state_20260922/README.md),
[repaired protocol](../../experiments/structural_state_20260923/README.md),
[repaired review](../../output/structural_state/repaired-review-20260923/README.md),
[completed factorial](../../output/structural_state/future-metric-20260923/RESULTS.md).

## Downstream models and other registered architectures

Direct/AR latent forecasts, mixtures, diffusion paths, structured spatial/temporal
context, hazard heads and geometry add-backs usually operate on **frozen** encoders.
Compare encoder learning only when the experiment explicitly fine-tunes or
reinitializes the backbone. Different downstream heads are not independent
encoder replications. The [forecast records](../../experiments/README.md) and
catalogue family `frozen_forecast` retain those results because they tell us what
the representation makes accessible.

The code also registers PointNet, DGCNN, VN, EGNN, NequIP, RI-MAE and MLP options.
Registration alone is not evidence of a completed experiment. The catalogue
includes older `output/experiments` tables as `legacy_controls`; the main handbook
starts with GeoFrame as requested. Exact initialization/checkpoint dependencies
are in each run's provenance, not inferred from a class name.
