# Spatial context on the fixed Al64 population

Question: do directional spatial-context predictors improve predictive information
over a symmetric invariant predictor when evaluated on the fixed 64-center cohort?

This repeats the [Al16 comparison](../equivariant_context_20260925/README.md) with
fresh supervised encoders and the sealed Al64 all-center population. Source roles
remain 90 train, 15 selection, 15 calibration and 30 historical test sources.
There are 126,545 windows: 43,523 / 20,883 / 16,848 / 45,291 in those roles.
Test sources were examined previously; this is not a new untouched test set.

Two shared native MACE encoders (observed and same-time relaxed coordinates)
and four frozen-encoder context predictors per domain make ten scientific fits.
The predictors are symmetric invariant, vector messages, tensor attention and
harmonic hierarchy. Each fit uses seed 20260924, 4,096 optimizer updates and
batch/microbatch 512. The encoder has width/export 128 and 634,496 parameters.
The fixed update budget is unchanged; the larger population changes effective
passes over training data. Structural pretraining is not added to this comparison.

Encoder inputs are current relative coordinates in an 8 A patch, up to 80 atoms,
5 A edges and two message-passing layers, without a halo. Every patch uses the
same encoder. Predictors receive 25 patch representations and relative geometry
at stencil radii 0/10/20 A, with up to 4 A query displacement. The maximum
geometric reach is 32 A. The invariant variant uses scalar features and invariant
geometry; the directional variants add the declared equivariant fields. They
combine shared patch predictions with a learned gate, without a focal bypass.
There is no history, velocity, temperature, simulation age, absolute-time input
or training-only teacher. Relaxation uses converged same-time full-cell quenches.

Training and checkpoint selection use source-weighted first-event/survival NLL.
AP at 3/6 ps is diagnostic only. Report NLL, AP, raw/calibrated Brier and binary
log loss, plus recall and actual FPR at a calibration-set alarm threshold. Fit
normalization on training sources only and probability calibration on calibration
sources only. Whole-source bootstrap uncertainty does not cover training seeds.

Al64 and historical Al16 headline metrics use different row populations. Direct
historical comparisons require restricting new predictions by immutable sample ID
to the legacy16 subset; do not attribute full-population differences to a model
change. New prediction files retain those sample IDs.

[Recipe](../../configs/equivariant_context/al64_20260925/comparison.json),
[dataset contract](../../docs/datasets/fixed_al64.md),
[execution and W&B fields](../../docs/equivariant_context.md),
[metric definitions](../../docs/metrics/equivariant_context.md).
