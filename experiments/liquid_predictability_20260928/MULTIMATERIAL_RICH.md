# Raw multimaterial local-descriptor capacity experiment

## RH2 review and literature, 30 September

Training was stopped by the user at update 5,486 (42 completed epochs plus
68/129 batches). All numerical statements here refer to recorded validation
or training diagnostics; the selected checkpoint's detached final evaluation
is a separate output. There is one training seed.

| Validation checkpoint | Error / fitting-mean baseline | Interpretation |
|---|---:|---|
| Epoch 1 | 0.375217 | Initial useful descriptor learning |
| Epoch 3 | 3.330859 | First large optimization disruption |
| Epoch 5 | 3.756747 | Further disruption near peak LR |
| Epoch 14, selected by minimum NLL | **0.262784** | Best observed held-out Al reconstruction |
| Epoch 15 | 11.102880 | Large disruption just after GPU handoff |
| Epoch 39 | 0.277409 | Subsequent recovery close to the best |
| Epoch 42, last full validation | 0.361737 | Still worse than the selected checkpoint |

At the selected checkpoint, geometry/bond-order/CNA/TDA ratios are
0.324658 / 0.257397 / 0.212221 / 0.255050. Overall error is 73.72% below the
fitting-mean baseline. This demonstrates substantial information about these
442 present-time descriptors in the 256-D code on held-out Al sources. It
does not establish within-liquid precursor skill, forecast sufficiency,
linear accessibility, robust embedding distances or held-out Mg/Ti/Ta skill.
The head and mean anchoring were changed together: their individual effects
are not identified by this repair.

Recovering from a spike did not make the run consistently productive: no later
validation checkpoint improved on epoch 14. This is not evidence that the
architecture needs more parameters or that the task contains no information.
Training loss also spiked; ordinary validation-only overfitting is insufficient
to explain the trace. At the logged updates 1,807–3,096, 30% of observed gradient
norms exceeded the clipping threshold 5; the largest was 2,568.37. These are
pre-clipping norms at the logging cadence, not a count over every optimizer step.
The two-GPU handoff and the epoch-15 disruption coincide; causation is unresolved.

### A concrete data-loader finding

An exact replay of the frozen `NativeStructuralDataset.epoch` producer for all
5,486 consumed batches preserves the fitting row IDs, seed, zero-based epoch
indices, batch8192 and 64-shard block shuffle. It loads no coordinates and fits
no model. Al occupies 7.50–97.84% of a batch; Ta occupies 0–88.51%; Mg has median
batch fraction zero despite being 4.20% of fitting rows. The largest single
lineage supplies a median 36.61% and a maximum 88.51% of a batch. Uniform row
inclusion over complete epochs therefore does not imply a well-mixed batch.

The full replay and summaries are retained at
`technical/training-review-20260930/batch-composition.json` in the RH2 run.
Material fractions are count/batch-size; largest-lineage fraction is the
maximum lineage count/batch-size. Correlations between each material fraction
and logged task NLL are small (absolute Pearson correlation below 0.033), so
this audit alone does not attribute spikes to material composition. It does
identify changing batch distributions for the task and covariance regularizer.

The selected 1,056,768 positions `(N,80,3)` and targets `(N,442)` together occupy
about 2.685 GiB in float32, excluding graphs/activations. Pack this fixed subset
once in IDS and globally permute row IDs each epoch. A host-resident/pinned or
GPU-resident packed loader can retain the original population weights while
reducing file I/O and batch-composition swings. GPU residency requires a measured
activation-memory margin. Do not silently replace row-proportional sampling
with equal material weights: that would define another scientific treatment.

### What the literature suggests testing

1. **Verify the handoff, then compare controlled LR continuations.** Replay one
   checkpoint and identical global rows on one and two GPUs; compare predictions,
   task/regularization losses, gradients and one AdamW update, allowing declared
   floating-point tolerances. Use local numerical diagnostics, with no W&B
   diagnostic run or automated test suite. Then branch from the selected
   epoch-14 optimizer/model state with common LR 0.0003, 0.001 and 0.003 for a
   short, matched update budget. Keep VCReg, batch, rows and sampler fixed for
   this comparison; evaluate validation NLL and clipping/update-size statistics.
   These values are proposed engineering settings, not paper-derived optima.
   [Cohen et al., adaptive edge of stability](https://arxiv.org/abs/2207.14484)
   report stability-boundary behavior for large-batch Adam. [Kalra and
   Barkeshli, warmup](https://arxiv.org/abs/2406.09405) connect warmup to improved
   conditioning and tolerable target LR. Neither paper diagnoses this MACE run
   or implies that recovery proves the chosen LR is efficient.

2. **Measure batch utility and independently compare global mixing.** At 8192
   there are only 129 updates per epoch. Compare packed, globally shuffled
   batches against the frozen block sampler, then compare batch2048/4096/8192
   using both matched examples and matched elapsed training time; report update
   counts explicitly. [McCandlish et al.](https://arxiv.org/abs/1812.06162) show
   that gradient noise scale predicts diminishing returns from batch growth
   across their studied tasks. That motivates a measurement here, not a claim
   that 8192 is intrinsically too large. Avoid equating GPU utilization with
   scientific improvement per unit compute.

3. **Keep VCReg, but compare where and how it acts.** The current scalar and
   vector covariance penalties operate directly on exported channels. The
   [official VICReg implementation](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)
   applies its objective after a projector. [Mialon et al.](https://arxiv.org/abs/2209.14905)
   study how projector properties affect independence, and also warn that a
   learned projector trained only to satisfy VCReg can become degenerate without
   another anchoring task. A relevant treatment is therefore a *task-connected*
   projector whose outputs feed descriptor heads and receive VCReg, while the
   exported 256-D code retains a modest variance safeguard. Do not add a free
   auxiliary head that can satisfy regularization without retaining useful
   information. Monitor code rank and simple frozen readouts; this is an ablation,
   not an established improvement for atomic structures.

4. **Measure gradient competition before making the decoder larger.** Four
   families receive equal scalar loss weight, but their gradients need not have
   equal magnitude or compatible directions. Log per-family and VCReg gradients
   on shared encoder parameters and their pairwise cosines on fixed diagnostic
   batches. Compare the current decoder with a shared task projector plus four
   modest family heads at a declared parameter budget. If imbalance is present,
   [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) is a motivated
   training-weight ablation. Keep the original equal-family validation NLL as
   the selector, and preserve every target's published score. Do not let adaptive
   weights redefine which held-out errors count as success.

5. **Audit discontinuous teachers and material/phase shortcuts.** The producer
   uses a hard radius-8 point selection, hard CNA bond cutoffs and hard pair/angle
   histogram bins. TDA already has smoothed images/Betti curves and retains all
   finite intervals; do not mistakenly reintroduce an old death-radius cutoff.
   [Adams et al.](https://jmlr.org/papers/v18/16-337.html) establish stability for
   persistence images under their assumptions, not an arbitrary entire
   preprocessing chain. Perturbations should be normalized by a local length
   scale, then compare teacher changes with embedding changes. Smooth histograms
   and boundary tapering are separate teacher variants; keep original hard
   descriptors as readouts for comparison. Report within-liquid, source-level
   and training-material diagnostics so phase/material separation cannot conceal
   poor local information. No material/phase metadata enters the encoder.

The first practical priorities are the numerical handoff check, a packed
globally mixed loader and a short LR continuation comparison. Subsequent
objective/head treatments should change one factor at a time. No new scientific
training runs were submitted as part of stopping and evaluating this run.

### Completed selected-checkpoint evaluation

The detached evaluation completed successfully in 7.09 minutes on the L40S,
including loading/verifying frozen data. Its selection ratio 0.262783608 agrees
with the training value 0.262784428 within 8.3e-7; inference used smaller chunks.
All held-out results below use the originally selected epoch-14 checkpoint.

| Population | Patches | Overall error / fitting-mean baseline |
|---|---:|---:|
| Al selection | 192,960 | 0.262784 |
| Al calibration | 16,848 | 0.410716 |
| Al test | 45,291 | **0.406666** |
| Fitted Al audit | 8,192 | 0.261536 |
| Fitted Mg audit | 8,192 | 0.439719 |
| Fitted Ta audit | 8,192 | 0.347365 |
| Fitted Ti audit | 8,192 | 0.411871 |

| Test descriptor family | Error / fitting-mean baseline |
|---|---:|
| Geometry | 0.435048 |
| Bond order | 0.487274 |
| CNA | 0.308578 |
| TDA | 0.397338 |

The test reduction is **59.33%**, below the selection reduction of 73.72%.
Selection consists of structural selection observations; calibration/test use
the fixed Al64 sample contract. Population composition differs, so this gap
alone does not establish ordinary memorization. Other-material training audits
do not establish that one metal generalizes better than another.

Although the head has 442 outputs, 289 features exceeded the frozen fitting SD
threshold and actually participated in the task loss: geometry 91/99,
bond order 40/40, CNA 45/45 and TDA 113/258. The other 153 outputs were fixed at
the training mean. Of the active targets, nine have test MSE above the fitting-mean
baseline; sixteen have negative R² where test variance defines R². These counts
refer to different baselines. Some large relative errors concern almost absent
features with tiny absolute error; inspect both columns before changing targets.

A separate descriptive audit reused
`src/research/trajectory_stability/spectrum.py:spectrum` on the exported states.
It centers raw 256-D codes and gives every exported row equal weight (not source
weight). On test data, participation rank is **12.91**, entropy rank **17.10**,
d95 **18**, and d99 **31**; numerical rank at a 1e-10 relative eigenvalue
threshold is 256. Selection participation rank is 5.50 and d95 is 16.
These are covariance-based linear spectral dimensions, not nonlinear intrinsic
dimensions or counts of independently predictable physical variables. They do
not establish collapse: the model retains nonzero variation in all channels,
but most variation is concentrated in a small subspace. The audited codes also
do not show near-zero per-row channel SD (test minimum 0.637), so the selected
head is not routinely dividing by almost-zero input variance.

The spectrum JSON records population, producer hash, embedding hash, full
eigenvalues and weighting at
`technical/training-review-20260930/embedding-spectrum.json`. A useful subsequent
diagnostic is training-fitted PCA plus matched frozen readouts at 16/32/64/256
dimensions, keeping source splits intact. Low spectral rank alone is not a reason
to force a larger rank or reduce backbone capacity.

The covariance penalty in this implementation divides summed off-diagonal
squares by `D*(D-1)`; the official VICReg code divides by `D`. Therefore equal
numeric covariance coefficients do not mean equal regularization strength:
the scalar D=256 penalty is smaller by a factor 255 for the same covariance
matrix and coefficient. This is a declared loss convention, not proof of a bug.
Any projector/regularization comparison must account for that scaling rather
than copying published coefficients directly.

### Detached evaluation scope

The original NLL-selected epoch-14 checkpoint is evaluated with its frozen
model, data and metric producer. Outputs cover all 192,960 Al selection patches,
16,848 calibration patches, 45,291 test patches and 8,192 fitted patches for each
of Al/Mg/Ti/Ta. The last four are training audits, not held-out material tests.
Exports include the 256-D codes, predictions, fixed sample IDs, per-feature
MSE/RMSE/R² and family/overall mean-relative errors. No test result changes the
checkpoint choice. Final scores update the original online run through its
recorded ID; evaluation itself remains local and detached.

## Normalized residual head repair, 29 September

The correlation3/angular3 run lost descriptor gradients when its prediction
head's SiLU inputs became strongly negative around update256. Full epoch-5
selection MSE was 1.00125 times the training-mean baseline; this is an optimization
failure, not evidence that the representation task is unlearnable.

RH2 asks whether normalizing the head and keeping a direct linear path preserves
descriptor learning through the requested LR schedule. It adds scalar mean
anchoring to VCReg, which is otherwise insensitive to embedding offsets. The
normalized head has two residual blocks (width512, expansion2) and computes in
FP32. MACE architecture, fixed sample IDs, transforms, seed, batch8192 and the
common maximum LR .01 remain unchanged. Fit afresh for 60 epochs, selecting by
descriptor NLL; evaluate every epoch. Because the head and mean regularizer both
change, this is a repair, not an isolated causal ablation of either component.

Report mean-relative MSE overall/per family, descriptor-gradient flow into the
state and prediction variation. Keep the stopped run's true metrics and weights.
[RH2 recipe](../../configs/liquid_predictability/rich_multimaterial_residual_20260929.json).

## Original protocol and subsequent revisions

Question: can MACE256/L3/Z256 retain rich local geometry, bond order, CNA and TDA
information across a broader range of raw materials and local configurations?

Train from scratch on a fixed compute-budget subset of the 13.42-million-patch
raw dynamic Al/Mg/Ti/Ta pool. Uniform sampling does not use crystal labels or
descriptor values. Keep 60 complete passes and select the subset size from
measured two-GPU throughput so fitting fits a ten-hour reservation with headroom.
LR warms to .004 over five epochs, then decays by cosine. Keep scalar/vector VCReg.

One exported 256-D local embedding predicts all 442 patch descriptors. There is
no 25-patch context predictor. This changes both task and population relative to
the 183,596-context Al experiment; their raw MSE values are not a matched capacity
comparison. Descriptor information may transfer to later context prediction,
but no improvement in distance-to-crystal prediction is presumed.

Primary reports: per-family standardized MSE and mean-baseline skill, per-feature
RMSE/R², Al selection/calibration/test performance and separately labeled fitted
material audits. Validation likelihood selects checkpoints every five epochs.
External materials have no independent held-out split in this release; their
training audits cannot establish generalization. One seed, fixed Al64 source
roles, no relaxation, no time/temperature/species inputs.

[Recipe](../../configs/liquid_predictability/rich_multimaterial_20260929.json) ·
[Metrics](../../docs/metrics/rich_multimaterial_encoder.md) ·
[Execution](../../docs/rich_multimaterial_encoder.md).

## Repair protocol

The first multimaterial sizing job failed before fitting. The replacement retains
this scientific task, all source roles, 60 full epochs and the ten-hour two-GPU
budget. It caps global batch at 4,096, uses encoder/head peak LR 1e-4/2e-4,
five-epoch warmup/cosine and unchanged VCReg. Batch and fitting subset are frozen
by numerical memory/throughput measurements, with no selection on test results.
The local fixed-draw learning check is engineering evidence only.
[Replacement recipe](../../configs/liquid_predictability/rich_multimaterial_repair_20260929.json).

The R2 lower-LR queue was cancelled before fitting. The active
[LR004 recipe](../../configs/liquid_predictability/rich_multimaterial_lr004_20260929.json)
restores the user-requested common maximum LR .004 for encoder and head, retaining
the memory repair and all scientific data/objective/budget definitions. The Al
failure did not isolate a learning-rate effect from batch/update-count changes.
