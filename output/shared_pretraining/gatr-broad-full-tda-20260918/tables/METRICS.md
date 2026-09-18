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


Table export: 2026-09-18T16:30:40.349715+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
