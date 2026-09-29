# Raw multimaterial local-descriptor capacity experiment

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
