# RD-MACE256 descriptor learning

User-requested 256-channel, three-interaction MACE with a 256-D scalar patch
embedding and 256-D exported context state. Geometry-only, one constant atom
channel, nearest 80 candidates, radius 8 Å, edge cutoff 5 Å, no halo. Three
interactions stay within the consumed patch; they do not extend its input support.
Two shared vector-context message blocks combine 25 patches. The only decoder
input is the 256-D invariant context state; a 512-unit hidden layer predicts all
3,536 geometry, bond-order, CNA and TDA summaries. No descriptors are model inputs.
No history, velocity, temperature, time, material ID or relaxed geometry enters.

The sealed raw Al64 rich-descriptor cohort is unchanged: 183,596 training,
41,418 selection, 34,117 calibration, 66,839 test observations. Source roles and
row weights are unchanged. Geometry and target transforms use training rows only.
This is label-free descriptor fitting on an existing phase-conditioned cohort,
not a new unconditional sample of liquid states.

Targets are standardized using their population-weighted training mean and SD.
SD below 1e-4 marks an unresolved/constant target: fix its output to its training
mean, omit it from the objective, and report it in the per-feature table. All
3,536 outputs remain present. The Gaussian task loss is
0.5 * mean_over_families(mean_over_varying_features(standardized_error²))
+ 0.5*log(2*pi). This is a fixed-unit-variance likelihood with equal family mass.

Each of 60 epochs visits every training row once, in a reproducible permutation,
including the final partial batch. With N rows and normalized row weights w_i,
the batch objective is mean(N*w_i*loss_i). Distributed partitions have disjoint
rows; their scaled summed losses and averaged gradients equal the global batch
loss even for the final partial batch. VCReg uses the differentiable global
scalar/vector covariance of the uniformly shuffled batch's patch features.
Thus its empirical sampling differs from the earlier weighted replacement
sampler; the target likelihood retains the same population weighting.

VCReg: .05 variance, .01 covariance, std floor 1, epsilon 1e-4; ramp over five
epochs. AdamW weight decay 1e-5, gradient clipping at norm 5. LR warms linearly
to 0.004 over five full epochs, then follows cosine decay to 1e-5 at the final
optimizer update. Larger measured batch sizes change the number of updates,
not the 60 passes or 11,015,760 training-context visits. Best checkpoint uses
weighted validation task NLL only. Also retain the epoch-60 checkpoint.

Numerical batch measurement is local, separate from scientific training. Search
in increments of 256 contexts per GPU after complete data upload, compiled
forward/backward and AdamW-state allocation. Allow 85% of device memory minus
2 GiB, and repeat the selected batch on different training rows. The remaining
memory covers distributed communication and geometry variability. Freeze the
batch/world size and reject smaller-VRAM resume hardware. This is the largest
measured batch within this reserve, not a claim about optimal statistical batch.

Final evaluation reuses the liquid-control export formulas: per-feature native
RMSE, standardized MSE, R²=1-MSE/held-out variance (undefined at zero variance),
family MSE and improvement over the training-mean predictor. Train/selection/
calibration/test rows remain separate; no held-out optimization. Native feature
units and column definitions remain the original rich-descriptor definitions.
Save predictions, exact row IDs and 256-D states. Test summaries update the
original online W&B run through its stable ID. Debug measurement stays local.


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
