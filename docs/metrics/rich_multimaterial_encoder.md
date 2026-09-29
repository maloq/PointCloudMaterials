# Multimaterial local rich-descriptor learning

MM-RD-MACE256-D3-C3-L3-Z256-RH2 predicts 442 **local patch** descriptors through one
256-dimensional exported embedding. It does not predict the earlier 3,536
25-patch context summaries. Comparisons must preserve this distinction.

Encoder: width 256, three MACE interactions, correlation 3, max angular order 3,
one constant atom channel, cuEq,
nearest 80 candidates, smooth radius 8 support and cutoff 5, no halo. Coordinates
use the existing fixed training-material length normalization with Al as reference.
There is no species, material ID, scale scalar, temperature, time, velocity,
history, phase label or surrounding-patch context input. The normalized residual
decoder described below receives only the exported patch embedding. The scalar embedding is the actual
CapacityEncoder output; vector channels enter regularization only.

Targets reuse `patch_descriptors` without changing its definitions: 99 geometry,
40 bond-order, 45 CNA and 258 TDA features, including alpha-complex H0/H1/H2.
The support filter and all descriptor cutoffs operate in the same normalized
coordinates consumed by the encoder. They describe this neighborhood; they are
not atomwise reconstruction targets and do not use unobserved surrounding atoms.

The eligible raw dynamic pool contains 13,423,868 neighborhoods: Al 7,537,284,
Mg 566,244, Ti 728,000 and Ta 4,592,340. Static/inherent configurations and paired
relaxed views are excluded. Original train ancestries and fixed material scales
are retained. This pool includes all phases; there is no crystal/liquid eligibility
filter. More atoms/frames from a parent are correlated observations.

The fitting subset is a fixed uniform draw without replacement from that pool,
using seed+71, selected once by measured compute cost, never outcomes or validation
scores. All selected rows appear once per epoch, shuffled in 64-shard blocks,
for the declared number of complete epochs (60 in the active correlation3/angular3 recipe). Actual row IDs, material counts, source count and checksum are
frozen in `technical/training-pool-row-ids.npy` and `batch-plan.json`. Material
proportions are preserved in expectation, not by fitted outcome weights.

Target means and SDs are recomputed on exactly this fitting subset. Columns with
SD <1e-4 are fixed at the fitting mean, excluded from the loss, and retained in
the exported metrics. The task likelihood is
`0.5 * mean_families(mean_active_features(standardized squared error))
+ 0.5*log(2*pi)`, with equal mass for geometry, bond_order, cna and tda.
VCReg uses differentiable global-batch scalar and vector covariances; variance
weight .05, covariance weight .01, standard-deviation floor 1, epsilon 1e-4.
The RH2 repair adds `0.01 * mean_channels(global_batch_mean(z)^2)` to discourage
scalar offsets that centered covariance cannot detect. Both regularizers ramp
over five epochs; VCReg and mean-penalty terms are logged separately. Spatial encoder, descriptor decoder and
vector projection train jointly. AdamW weight decay 1e-5; gradient clipping 5.
The active correlation3/angular3 recipe uses the requested common peak LR .01 for both
encoder and head, with five-epoch linear warmup and cosine decay to 1e-5.
The proposed lower-LR R2 queue (encoder/head 1e-4/2e-4) was cancelled before
scientific fitting; its local learning diagnostic is not a trained-model result.
The Al failure did not isolate LR from batch size/update count and does not
justify automatically reducing multimaterial LR. Earlier frozen definitions
and diagnostic artifacts retain their actual settings.

Numerical sizing is a separate, local-only job. The active recipe requires
global batch 8,192: 8,192 on one GPU or 4,096 each on two GPUs. It checks compiled
forward/backward, finite gradients in all three interactions, allocated and
reserved memory, and CUDA device free memory, including non-PyTorch allocations.
PyTorch's allocator is limited to 94% of VRAM; require at least 4 GiB device-free
headroom and allocated memory below 94% minus 4 GiB. Activation checkpointing
is enabled with 1,024-patch chunks, following the user-approved execution choice
after the no-checkpoint memory failure. The exact whole-batch VCReg is retained.
Whole-graph Torch compilation is disabled; native cuEquivariance kernels remain
enabled, with persistent NVRTC kernel caching outside the repository. Numerical checks must not silently reduce the declared global batch. cuEq `cudaMallocAsync`
`cudaErrorMemoryAllocation` and PyTorch OOM both bound the search; other runtime
errors propagate. Three consecutive forward/backward steps verify the declared per-GPU batch with
finite gradients in every interaction. RH2 then performs a separate 12-update
local learning check at peak LR 0.01 on fresh 256-row training-pool draws, with
256-patch chunks. It records descriptor-only gradients in every interaction and
prediction variation. All diagnostic weights are discarded; the scientific run
initializes afresh. No batch-size search or online diagnostic run is created.
Out-of-memory errors require an explicit execution-plan
change rather than automatically shrinking the batch.


The fitting population is now an explicit recipe count: 1,056,768 patches.
It was selected before the original training and is unchanged by the epoch
extension or GPU handoff. Preparation uses the deterministic seed+71 sample IDs
and recomputes target statistics on precisely those rows; no timing-dependent
subset selection remains. The original sizing artifacts retain their historical
measurements.

The active recipe specifies 60 epochs and 7,740 optimizer updates. The user explicitly
replaced the proposed 190-epoch extension with at most 60 epochs. Warmup remains five epochs; cosine reaches 1e-5 at update 7,740 (the
last executed zero-based scheduler index is 7,739). No compute-hour counter
or cumulative elapsed-time stop condition participates in fitting.

Selection is the original 192,960 Al structural observations from 15 held-out
selection sources. RH2 evaluates all these rows every epoch (the prior frozen
run evaluated every five epochs);
choose the minimum task Gaussian NLL, excluding regularization from selection.
Keep the final checkpoint as well. Fixed Al64 calibration/test sample IDs and
raw hot observations are preserved exactly: 16,848 / 45,291 rows. Their event
labels are not loaded by this workflow. External materials remain train-only;
this does not establish held-out cross-material generalization.

Final exports include selected-checkpoint predictions, 256-D states and row IDs.
Report each feature's standardized MSE, descriptor-unit RMSE and R² using the
evaluated population variance (undefined if <=1e-10). Descriptor units involving
length are Al-equivalent normalized units, not native Å for every material.
Family MSE is the mean over active columns; skill is `1-model_MSE/mean_predictor_MSE`.
The mean predictor always uses the actual fitting-subset mean. A fixed sample of
up to 8,192 fitted rows per material gives training diagnostics, labeled
`train_audit_*`; these are not held-out material scores. Selection, calibration
and test remain separate. W&B is online for scientific training only; final
metrics update that training run through its existing ID.

## Mean-relative error for subsequent runs

Validation logs and final exports additionally report
`relative_mse_to_training_mean = standardized_mse / training_mean_mse`.
For each target, the constant predictor is its frozen fitting-subset mean,
which is zero after the existing training-only standardization. Its MSE on the
evaluated rows is therefore the mean squared standardized target. Neither target
means nor scales are refitted on selection, calibration, test or training-audit
populations. Baseline MSE on held-out populations need not equal one.

Report the numerator, denominator and ratio for all active targets together and
for each of `geometry`, `bond_order`, `cna` and `tda`. Each family gives equal
weight to its active columns. The overall score gives equal weight to the four
families, matching the training likelihood; it is the ratio of the two weighted
MSEs, not an average of feature/family ratios. Squared-error sums are accumulated
over all evaluated rows and distributed ranks before division, so batch sizes
and rank partitions do not change the definition. A ratio of 1 matches the mean
predictor, 0.5 means half its MSE, 0 is perfect, and values above 1 are worse.

W&B validation history uses `validation/relative_mse_to_training_mean` and
`validation/{family}_relative_mse_to_training_mean`, alongside model MSEs.
Fixed `validation/*training_mean_mse` denominators are summary-only; local
validation records still retain them. Custom wall-time `seconds` stays local,
since W&B already records runtime. Historical time/baseline series are hidden
on resume and receive no new points. This logging change does not alter metric
calculations or training. Final training-run summaries use the same suffixes under
`evaluation/{population}/`. `scores.csv` includes an overall `family=all` row;
`features.csv` also includes the baseline MSE, ratio and skill for every target,
including inactive targets marked `trained=false`. Ratios and skill are undefined
(JSON null / blank CSV, never zero) when baseline MSE is <=1e-10. Nonfinite MSEs
are errors. R² remains distinct: it uses the evaluated population's variance,
whereas this ratio always compares against the training-mean constant.

These additional diagnostics do not change the likelihood, VCReg, checkpoint
selector or validation cadence. They apply to runs frozen after this change;
the running experiment's code and historical metric definitions remain intact.

## RH2 head and learning diagnostics

The earlier Linear→SiLU→Linear head saturated at negative inputs: epoch-5
predictions were constant across 256 diagnostic patches and descriptor gradients
into the three MACE interactions were approximately 1e-14. Its full selection
MSE was 1.00125 times the training-mean baseline. The old artifacts are preserved.

RH2 starts from scratch on the identical fitting IDs/target transform. The head
uses non-affine LayerNorm on the 256-D exported state, a 256→512 stem, and two
residual blocks. Each block is LayerNorm→Linear(512,1024)→LayerNorm→SiLU→
Linear(1024,512), added with scale 1/sqrt(2). LayerNorm→Linear(512,442) produces
the nonlinear prediction; a parallel direct Linear(256,442) path consumes the
normalized state. Output and residual-ending weights initialize at .01 times
the usual linear initialization, with zero biases. The head computes in FP32;
MACE remains BF16/cuEq with the user-approved activation checkpointing. Only the
exported state is decoded. LayerNorm is per sample and introduces no batch or
condition inputs. Normalizing immediately before SiLU prevents a shared negative
offset from saturating all its units. The direct path does not use SiLU.

Every eighth update (and first/epoch-end), report mean-relative training errors
overall and per family, weighted mean per-target prediction SD, RMS scalar
embedding mean, minimum scalar SD from VCReg, and descriptor-only gradient RMS
at the exported embedding. The last is d(sum of per-patch descriptor NLL)/dz,
averaged in squared norm over global-batch rows and scalar channels; it excludes
regularization and removes the 1/batch factor. It is a head-to-encoder signal,
not a parameter-gradient norm. Existing `gradient_norm` remains the combined
parameter norm before clipping. Statistics aggregate ranks before division.
Training ratios use the same batch for model and training-mean baseline errors;
they are diagnostics, not held-out scores. Prediction SD averages active columns
with the same family-balanced weights as the loss.

Three consecutive logged observations with prediction SD <1e-7 AND descriptor
embedding gradient RMS <1e-10 checkpoint and fail loudly, instead of continuing
a disconnected constant predictor. The check is an execution failure detector,
not a scientific checkpoint selector. Selection remains descriptor NLL alone.


## Immediate H100 execution and continuation

The recipe fixes global batch 8,192, permits one or two GPUs, and reuses
the exact previously selected fitting subset. Every selected
row, target transform, epoch permutation and optimizer-update LR schedule stays
fixed after a move. One GPU consumes the whole global batch; two GPUs take
alternating disjoint rows, 4,096/GPU for full batches. Task-loss scaling and global
VCReg covariance preserve the global objective. Hardware and reduction-order
rounding can change floating-point results; bitwise equality is not claimed.

A checkpoint records model, AdamW state, epoch/batch cursor, RNG and elapsed
wall seconds for provenance. Completion depends only on the requested epoch
count. Before Slurm ends an allocation, the worker checkpoints and continuation
resumes the remaining updates. No scheduler restart or data expansion occurs.
Each execution records the GPU count and starting update.

The queued continuation requests a handoff through an atomic control file. The
current worker saves at an optimizer boundary and exits. An exclusive filesystem
lock covers preparation, training and export; continuation cannot overwrite a
live worker's checkpoints. A completed H100 run cancels its pending continuation.
Al context fitting remains stopped.

The new correlation3/angular3 run starts from scratch with its own scientific
identity and W&B run. The prior correlation2/angular2 checkpoint at update 598 is
preserved. The prepared 190-epoch continuation was never resumed. The new run
reuses its exact fitting sample IDs and target transform, with frozen checksums;
new architecture, LR and batch settings are explicitly recorded in the recipe.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.


## Explicit task-head refactor

The training/model refactor separates typed patch and spatial-context trunks from
task heads and expands training statements. Mathematical objectives, populations,
weights and selectors retain their definitions. Joint/rich-patch initialization
and state names are preserved; distance/control fresh initialization changes
when unused head construction is removed and receives a versioned architecture
identity. Historical continuations use their frozen sources. W&B wall-time stays
local and fixed baselines stay in summary; metric calculations are unchanged.
See [implementation and compatibility evidence](../code_cleanup_implementation.md#training-and-model-follow-up).
