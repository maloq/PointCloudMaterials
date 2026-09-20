# Shared structural / causal pretraining and frozen probes, version 3

The September 18 VICReg restart uses batch 1,024, 12 epochs (2,930 updates),
peak LR 0.002, 293 warmup updates and cosine decay to 0.00002. It starts fresh
MACE and GATr models, one seed, on the same broad release. BF16 autocast is used
for eligible encoder/head operations; master parameters, optimizer state,
VICReg sample-variance/covariance statistics and losses remain FP32. Backend
operations not eligible for autocast retain their supported precision.
Gradient caching uses the same autocast mode in both encoder passes.

Scalar readout inputs and hidden preactivations use per-observation LayerNorm
without learned affine parameters, as does the exported state. This is not
batch whitening and uses no held-out or training-population statistics. Physical,
topology and projector heads use the same normalized interfaces. The future
head is normalized and starts with a zero last layer, predicting its training
mean initially; no causal fits are launched in this restart.

Selection state, projector and physical/TDA prediction spread is calculated as
mean coordinate population standard deviation using float64 reductions. Fits
stop if any is nonfinite or at most the configured 1e-6 threshold. Selection
is checked every 64 updates. By update 640, a structural fit's best present
score must beat its matching material/potential/static-group training-mean
baseline. Means use exactly the original target producer's training endpoints
(current/spatial/future for dynamics, current/spatial for static geometry) and
valid instantaneous TDA labels. Selection sources affect evaluation only.
`gain_over_training_mean` is baseline minus present physical + 0.25*TDA error;
positive means improvement. This gate detects failed learning, not scientific
validation across all materials: selection is still the fifteen native Al sources.
Group-specific training curves separate material, potential and static/dynamic
populations; the aggregate gradient norm remains a backbone/head parameter norm.

Historical version-2 settings and the unchanged physical metric definitions
follow. Frozen historical exports retain their original descriptions and hashes.

Structural targets and VICReg/SIGReg definitions are those of
[structural pretraining v1](structural_pretraining.md). Each view reconstructs
its own geometry85 and instantaneous TDA144; no velocities or relaxed targets.
The schedule uses the configured statistical batch: 512 anchor pairs in the
local campaign and 1,024 in its H200 batch-size comparison, with 12 equivalent
epochs of anchor draws, linear warmup for 10% of updates to peak 0.02 and cosine decay to
0.0002. Microbatching uses exact full-batch gradient caching, including uneven
last chunks. Epoch equivalents count anchor draws divided by training records;
partner/history frames do not increase the epoch count.
Doubling batch size halves optimizer-update counts at equal anchor exposure;
regularizer formulas and coefficients stay fixed, including the upstream
Epps–Pulley sample-count factor in SIGReg. Probe training remains at batch
1,024 in both campaigns. This changes the training comparison, not the physical
or topology error definitions below.

VICReg training logs `train/vicreg` = 25 I + 25 V + C and
`train/vicreg_weighted` = 0.1 * train/vicreg / 51. The latter is its contribution
to the combined training loss; `train/representation` retains the normalized
total, train/vicreg / 51. These diagnostics do not change objective scaling and
are absent for JEPA.

Causal targets use the original 150-source Al cohort and 38,400 fixed windows:
90/15/15/30 train/selection/calibration/test sources. Inputs are the current
snapshot or three frames at -1.5, -0.75 and 0 ps. Future geometry/TDA at 0.75,
3 and 9 ps never enters encoder inputs. Current, next and future targets are
calculated in physical Angstrom with the existing instantaneous TDA producer.
Present heads retain the structural release normalization. Forecast mean and
population standard deviation (floor 1e-4) fit only native training sources,
across current and three future target observations. Future losses average
four physical blocks and three topology blocks equally, and three horizons
equally: future = physical_MSE + 0.25*TDA_MSE. Total causal loss adds this future
term to the structural objective, retaining present reconstruction and the
parent representation regularizer. Every fourth update replays the broad
structural data without future supervision. Twelve causal epochs count only
native anchor draws; broad replay draws are reported separately.

Checkpoint selection uses current physical + 0.25*TDA for structural training.
Causal selection adds future physical + 0.25*TDA, averaged over three horizons.
Rows are averaged within each selection source, then equally over sources.
Structural selection has 480 rows; causal selection takes a seeded fixed 64
rows from each of fifteen selection sources. These scores are for selection,
not held-out test results. State_std_mean is the average coordinate population
standard deviation on the selection states, not a whitened distance metric.

Frozen analyses use all 38,400 identical rows for each backbone, separately
extracting selected structural and causal states. Target standardization uses
the causal release's train-only forecast mean/std for every model. Ridge
regularization is chosen on selection sources. The nonlinear probe is a
128→256→256→916 residual on the selected ridge, with a zero last layer and the
ridge itself as its initial candidate. Its 1,000 updates use the declared
warmup/cosine schedule, and selection labels choose its checkpoint. Target
blocks and four horizons receive the same weights as above. Neither test nor
calibration targets affect fitting or scaling.

Baselines are per-channel/per-horizon training means, a temperature-only ridge,
and persistence of each observed current physical target. Direct trained heads
are evaluated separately: structural heads only have current outputs; causal
heads also have future outputs. Missing structural future heads are not scored.
Raw latent MSE is not a cross-backbone quality metric.

Test metrics average per-row standardized MSE over channels within each of
radial32, pair32, angular16, moments5, H0-16, H1-64 and H2-64. Errors average
within source first, then equally over independent sources; physical is the
mean of its four blocks and topology the mean of its three. Results are shown
for all test rows, current noncrystalline rows (PTM class outside 1/2/3), and
separate temperatures. Noncrystalline filtering occurs only at evaluation.

Positive gain_over_persistence means lower candidate error: baseline minus
candidate standardized MSE, separately by horizon and block. The confidence
interval is the 2.5/97.5 percentile of 4,000 paired whole-source bootstrap
resamples; it does not quantify training-seed uncertainty. Test source counts
and available rows accompany each population. All models use seed 20260919.
Learning rate, input wait and gradient norms are operational diagnostics, not
independent research observations. Per-update duration (`seconds`) and peak
allocated/reserved GiB remain in local `technical/updates.jsonl`; they are not
sent as custom W&B training metrics. Duration is host elapsed time around the
cached optimizer update, excluding input waiting, validation and checkpoint
writing. W&B provides its own system monitoring. Peak VRAM varies with
observation size; a 40 GiB allocator budget bounds cached memory.

## Geometry-protected architecture revision (v4)

`shared_pretraining_v4` records `architecture_revision` explicitly. The snapshot
encoders contain 147,139 MACE or 803,216 GATr parameters; history GATr creates its
temporal blocks only when requested. With outer BF16 autocast, MACE's invariant
radial MLPs use BF16; GATr scalar-to-scalar maps use three compensated BF16
products with FP32 outputs and accumulations. Geometry, joint geometric attention,
residual streams, pooling, readouts, projector, present/future physical heads and
latent predictor remain FP32. Loss formulas, target normalizers, source balancing
and health thresholds do not change. See the
[architecture and arithmetic record](../shared_pretraining_geometry_fp32_2x.md).
This is a new model/implementation identity, not an exact resume of a v3 fit.


## Compiled repair protocol (v5)

Fresh `vicreg_compiled_repair` recipes retain the enlarged v4 encoders, the
original physical + 0.25 instantaneous-TDA + 0.1 VICReg/51 objective, and add
`0.1 * physical_correlation_loss`. For each of the 85 physical coordinates,
center decoder predictions and raw physical targets over both views of the
full homogeneous training batch. Compute population variances and covariance;
correlation is covariance / sqrt((prediction_variance + 1e-8) *
(target_variance + 1e-8)). Average `1 - correlation` over coordinates with raw
target variance > 1e-6. Return zero when all targets are constant. This auxiliary
uses the existing release's raw physical units for its variance eligibility;
it is not a unit-independent threshold or a new selection metric. The MSE
anchors still fix physical amplitudes. Selection scores remain unchanged and
exclude this auxiliary term, so a lower training total does not imply better
physical reconstruction. The weight is recorded in run identity.

`projector_std` averages actual unbiased coordinate standard deviations over
the two views, without VICReg's epsilon. `projector_participation_ratio` is the
mean of trace(C)^2 / sum(C^2) for each view, with denominator floor 1e-30; zero
covariance reports zero. This is variance concentration, not algebraic rank.
`variance_active_fraction` averages the fraction of coordinates whose
VICReg sqrt(variance + 1e-4) is below one. `state_std`,
`physical_prediction_std` and `tda_prediction_std` report mean unbiased
coordinate standard deviation over the concatenated views. These training
statistics are within the sampled material/potential/static group. Selection
spread statistics retain their separate float64 definitions above.

Compilation preserves checkpoint key names, enables dynamic shapes, disables
GATr's Python einsum cache, preserves precision casts and sets compiled backward
autocast to off, matching the two backward passes outside autocast. Recompile
limit exhaustion raises an error instead of silently switching to eager.
`technical/compilation.json` records compiled graphs and graph breaks; partial
library graph breaks are permitted. Geometry stays FP32; GATr scalar BF16 uses
fused high/low decomposition with FP32 accumulation and deterministic split-K
weight gradients. Compiler and kernel choices change execution, not loss
coefficients. Old runs retain their frozen implementations and definitions.

## Al-only conditioned-head protocol (v6)

`materials=["Al"]` filters shards before opening arrays and before training or
selection row construction. It recomputes target mean/std from the included
training endpoints using `prepare.finalize`'s population and 1e-4 std floor.
The subset identity records its parent release and material filter. No excluded
material contributes target moments. The v5 objective coefficients and block
errors are unchanged, but old broad-normalized numbers are not directly
comparable. The v6 source-held-out selection set is native Al MEAM only.

Head input BatchNorm (eps 1e-6) conditions across observations in the full
statistical batch; projector hidden BatchNorm uses eps 1e-5. Physical/TDA heads
retain hidden row LayerNorm. Before selection and best-checkpoint export,
running moments are refreshed from a deterministic proportional sample of
`head_calibration_rows` training anchors at current encoder weights. Mean and
unbiased variance are reduced in float64, then stored in the FP32 running
buffers. The hidden projector moments are fitted after its refreshed input
normalizer. This uses neither target labels nor held-out observations and
changes no learned weights. Inference uses fixed buffers; the encoder remains
independent of other observations. Training BN statistics are never reused
from encoder microbatches, because heads are evaluated once per full update.

`encoder_lr` is the scheduled head `lr` times the configured
`encoder_lr_multiplier`. Both follow the same warmup/cosine shape. The Al
recipes use 0.0002 / 0.002 peaks and 12 * 125000 / 1024 rounded up to 1,465
updates. Group logging includes spatial/temporal view type. New health checks
require the best score to beat the training-group-mean baseline by 5% after
256 updates; three consecutive scores above min(0.95 * baseline,
1.5 * best-before-those-three) stop the fit. These are operational gates,
not uncertainty estimates or new quality metrics.

Compilation is full-graph in v6. GATr's tensor-mask attention uses native SDPA;
both einsum path optimizers are disabled to preserve symbolic atom counts.
Contiguous scalar slices and runtime Triton strides support shape changes
through backward. The metric equations and physical precision boundaries are
unchanged. See [execution and scope](../shared_pretraining_al_stability_20260918.md).

## Broad full-TDA structural continuation

The September 18 continuation initializes GATr from the validated Al-only best
checkpoint (parent step 1216) and fits three epoch equivalents on the expanded
287,500-anchor Al/Mg/Ti/Ta/Zr release described in
[structural metrics](structural_pretraining.md#full-instantaneous-tda-expansion-september-18).
All supervised endpoints must have finite TDA labels; missing labels fail before
training. Encoder and projector parameters transfer exactly. Physical/TDA final
linear layers are transformed for the newly fitted target moments: weights
multiply by old_std/new_std; bias becomes
`(old_std * old_bias + old_mean - new_mean) / new_std`. This preserves the initial
physical-unit prediction before the training-only head-moment refresh. Optimizer
and warmup/cosine schedule restart, with 843 updates at batch 1024. Peak head LR
is 0.002 and encoder LR 0.0002. The objective is unchanged, including VICReg and
the physical correlation anchor. Batch statistics remain within material,
potential and static/dynamic group. No cross-material variance objective is added.

Step-zero selection is evaluated after training-only head calibration and can
remain the best checkpoint if fitting fails to improve it. This makes the parent
and updated candidates comparable under the same new target units and evaluation
procedure. Old Al-normalized scores are not numerically comparable without
reevaluation. Only the native Al selection cohort is evaluated here; other-metal
training losses do not establish other-metal held-out generalization.

## Mixed-material dynamic triplets (v7)

`batch_mode=mixed_triplets` is fresh snapshot GATr VICReg training. Static shards
are filtered before any input arrays or target statistics are loaded. All target
mean/std moments are refitted from included training endpoints, retaining the
original block definitions and 1e-4 std floor. The current recipe has 254,520
anchors across four materials and five material/potential domains. Batch 2,048
samples all five domains: proportional weights with minimum 128, clamping
undersized groups before largest-remainder rounding. Observations are shuffled
across domains, without replacement per update. An epoch is a draw equivalent,
not a complete shuffled pass. Twelve epochs give 1,492 updates.

Auxiliary head input moments are computed within domain over both paired views,
not pooled across materials or encoder microbatches. Physical/TDA input norms
have per-domain learned scale/bias, eps 1e-6 and shared downstream layers.
The projector uses per-domain input normalization (no affine), then a shared
128-to-256 layer, per-domain hidden normalization (affine, eps 1e-5), ReLU and
shared 256-to-64 output. Every normalization operation uses population variance
in training and calibration. Before evaluation, 128 fixed training-only anchors
per domain provide independent anchor/spatial/future snapshots (384 states per
domain); float64 calibration moments are stored as FP32. Hidden moments are
fitted after input moments. Evaluation uses these fixed buffers; exported
snapshot encoders have no group-dependent normalization or batch dependence.

VICReg is evaluated separately within each domain's paired projected states,
then averaged with weight n_domain / batch_size. `vicreg` is the weighted raw
25*invariance + 25*variance + covariance; `representation` is that value / 51;
`vicreg_weighted` is 0.1 * representation. All existing variance/rank diagnostics
from VICReg have the same weighted-within-domain reduction. Physical correlation
is likewise evaluated within domain over the two views before weighting.
Physical and TDA MSEs retain global included-training target standardization,
block averaging and observation weighting. `groups/MATERIAL/POTENTIAL/*` logs
per-domain physical, instantaneous_tda, raw vicreg, pair count and current-view
mean unbiased state std. Global `state_std` and prediction std values include
between-material variation and must not be interpreted as within-domain spread.

Each frame is encoded separately. Temporal-pair order is current, next, previous;
spatial-pair order is current, spatial, previous, next. Each entry is B snapshots.
Only the first two entries have physical/TDA and VICReg supervision. Earlier
cached frames have placeholder descriptors and are strictly excluded from these
losses. Persistent center identity is checked across previous/current/next.
`h_previous` and `h_next` are actual positive time differences in ps.

`backtracking` = mean over anchors of the squared Euclidean norm of
`2*(h_previous*(z_next-z_current) + h_next*(z_previous-z_current)) /
(h_previous+h_next)`. This equals the requested second difference for equal
spacing and vanishes for constant velocity at irregular times. The channel
reduction is a sum, not a mean. There is no variance normalization, inverse-dt²,
stop-gradient, predictor, or temporal encoder. `backtracking_weighted` is the
configured weight (0.001) times this value. `backtracking_loss_fraction` divides
it by the detached total with denominator floor 1e-12.

Total = physical + 0.25*instantaneous_tda + 0.1*representation +
0.1*physical_correlation_loss + 0.001*backtracking. Selection remains physical +
0.25*TDA and excludes all representation/correlation/curvature terms. Present
selection is native-Al source-held-out only; other-metal training curves are not
held-out evidence. Head and encoder peak LRs are 0.002 and 0.0002, respectively.
The source-held-out group-mean baseline is fitted from training endpoints only.
Past frozen run definitions remain unchanged. See the
[execution recipe](../shared_pretraining_mixed_triplets_20260918.md).

## Compact W&B projection (new submissions)

The `compact_v1` logger changes presentation only. It does not change the
per-update rows in `technical/updates.jsonl`, selection rows, CSV exports,
objective calculations, gradients or health gates. Historical runs keep their
frozen logger. The following names supersede earlier W&B-specific aliases for
new submissions; raw technical metric names remain as defined above.

All `loss/*` fields are weighted contributions: `total` = original `loss`,
`physical` = original physical MSE, `tda` = 0.25 * instantaneous TDA MSE,
`vicreg` = original `vicreg_weighted` (0.1 * raw VICReg / 51),
`physical_correlation` = its original weighted term, and `backtracking` = its
original weighted term. Only active objectives appear. JEPA substitutes
`loss/jepa` = 0.1 * representation and shows raw `jepa/next_latent_mse` and
`jepa/sigreg`. Causal fits additionally show `loss/future`. Raw VICReg is logged
once as `vicreg/total` = 25 I + 25 V + C, alongside the unweighted
`vicreg/invariance`, `vicreg/variance`, `vicreg/covariance` components.

Training scalars are arithmetic means over every update since the previous
emission, including mixed spatial/temporal draws. All updates have the same
statistical batch size in a run. An absent causal future objective on replay
counts as zero, using the full interval denominator, so displayed weighted
components sum to displayed total. Diagnostics use the count of updates where
they are defined. No additional temporal smoothing is applied. Emissions occur
at update 1, each configured logging interval, before validation, and at clean
shutdown; partial intervals are retained. The learning-rate curve is the last
head LR, not an average. Epoch equivalent is anchor exposures / training rows;
causal replay updates do not increase native epoch exposure. Both epoch and
training-step axes are hidden from automatic plots.

`health/projector_std` retains the existing actual standard-deviation diagnostic;
`health/projector_effective_rank` is the existing covariance participation ratio,
not algebraic rank. In mixed training these are weighted within-domain values.
The global mixed-material state std is not plotted because it includes material
separation. Detailed per-domain values are retained as latest-value summary
metadata and as complete raw JSONL histories, without duplicate chart series.

`validation/score` is the unchanged checkpoint-selection score;
`validation/present_physical` and `validation/present_tda` are unchanged,
unweighted source-balanced MSEs. Causal fits add `validation/future_physical`
and `validation/future_tda`, each the arithmetic mean of its three existing
per-horizon scores. Validation points are not averaged across evaluations.
The present training-group-mean baseline, selection counts and best score/step
are summary metadata, not constant or duplicate curves. Detailed block/horizon
errors and spread values remain in local exports. No custom timing, memory,
boolean, batch-count, gradient-norm or per-view duplicate series are sent.
Built-in W&B system monitoring is unaffected. The current mixed VICReg recipe
has 16 custom plotted series; see [the layout](../shared_pretraining_logging.md).

## Temporal-only calibrated backtracking (v8)

The encoder architecture, target moments, within-domain VICReg and selection
metrics remain v7. Spatial updates now contain exactly two B-sized entries:
anchor then spatial partner. They do not load past/next frames and report
backtracking = backtracking_weighted = backtracking_loss_fraction = 0. Temporal
updates keep current/next/previous, the same positive timestamp checks and
second-difference formula. Only the first two entries are supervised.

The coefficient is fixed after a training-only gradient calibration at the
transition checkpoint. Three full temporal batches use the unchanged sampler.
For each, L_base is the complete physical/TDA/VICReg/correlation loss without
backtracking, C is the raw backtracking penalty, and g_base, g_C are their
separately accumulated encoder parameter gradients before clipping. Head
parameters are excluded from these gradient norms. Choose
`min(median((0.02/0.98)*L_base/C), min(0.10*norm(g_base)/norm(g_C)))`, rounding
down to two significant digits. This targets a 2% scalar loss fraction on
calibration temporal updates, subject to the gradient-ratio constraint. The
reference fractions, norms, gradient cosines, sampled-index hashes and parent
checkpoint hash are retained in the calibration JSON. No held-out data or
optimizer updates are used in calibration. The cap is a calibration criterion,
not a dynamic training cap, and there are no extra calibration passes during
production training. The fixed coefficient applies only to temporal updates;
there is no implicit probability correction or additional time normalization.

An explicit v7-to-v8 continuation validates unchanged data, architecture, target
buffers, update budget and scientific settings other than the coefficient.
Model/optimizer/RNG state and the global update index continue, while selection
history starts with a new baseline at the parent step. Epoch equivalents and
cosine warmup/decay retain the original total exposure budget. The new run has
its own identity, checkpoint hash ancestry and W&B history; it is not an exact
resume of the old objective. See the
[execution record](../shared_pretraining_temporal_backtracking_20260918.md).

## Mixed MACE with equivariant bond order (v9)

`shared_pretraining_mixed_mace_bond_v9` uses the same dynamic-only, mixed-domain
snapshot sampling and within-domain VICReg as v8. It trains from scratch. The
bond-order auxiliary head consumes the tracked center's learned MACE l=2 atom
features before invariant pooling. Learned equivariant tensor products form
l=4 and l=6 outputs. It has no access to coordinates, target harmonics or the
128-channel invariant state. The exported encoder remains a snapshot-to-128
invariant encoder; auxiliary weights are retained in full training checkpoints.

Targets are central q4m/q6m averages over the 12 nearest noncoincident supported
atoms of each supervised snapshot, using the real e3nn `component` spherical
harmonics. Uniform coordinate normalization does not change them. The neighbor
definition matches `analysis.liquid_structure.bond_order` for its central atom;
that producer uses complex integral-normalized harmonics. The real targets are
sqrt(4*pi) times an orthogonal basis conversion. Therefore the ordinary scalar
bond order is `Q_l = sqrt(mean_m(q_lm**2))`. Hard nearest-neighbor selection can
change membership at ties; no claim of a smooth bond-order target is made.

For each observation and l, `bond_order_errors = 12*mean_m((prediction-target)^2)`.
The fixed factor 12 makes the zero predictor's expected error one for independent
isotropic random bonds. It is not an empirical or material-specific whitening.
`bond_order` is the mean across both l values and supervised observations;
`bond_order_q4`, `bond_order_q6` are its per-order terms. The additional training
contribution is `bond_order_weighted = 0.1*bond_order`, shown once in W&B as
`loss/bond_order`. Past context is not bond-supervised. FP32 tensor contractions
are enforced under BF16 autocast, and no per-m normalization is permitted.

Validation reports source-balanced `bond_order`, `bond_order_blocks.q4/q6`,
`bond_order_zero_baseline`, and the unscaled scalar magnitude MSEs
`bond_order_magnitude_mse.Q4/Q6`. Per-observation predictions, targets and errors
are exported with the selected candidate. W&B adds only `validation/bond_order`.
Checkpoint selection remains physical + 0.25*TDA, preserving the GATr comparison;
bond-order skill is an additional diagnostic, not evidence that orientation is
encoded in the invariant exported state. Native-Al validation remains the only
held-out material assay in this recipe.

Backtracking uses only the first 128 invariant features and only temporal
updates. Its fixed coefficient is calibrated at fresh MACE initialization using
the v8 training-only scalar/encoder-gradient policy; L_base includes bond order.
Calibration and throughput measurements are a separate disposable preflight,
never a production training stage. Five epoch equivalents mean
ceil(5*254520/2048)=622 independently sampled updates, not exhaustive shuffled
passes over the release. The 10% warmup and cosine schedule span these 622 updates.

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


Table export: 2026-09-18T22:57:49.973506+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
