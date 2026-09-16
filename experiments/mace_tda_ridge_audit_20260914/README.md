# Retesting single-frame VICReg versus TDA supervision

Question: do the previously reported balanced ridge errors 0.032914 (VICReg)
and 0.032813 (VICReg plus balanced TDA) survive fresh inference and independent
calculation, and was topology supervision actually enabled only in the intended
encoder-training runs?

Use all six original selected single-frame checkpoints: seeds 20260910, 20260911
and 20260912 for `anchor_vicreg` and `anchor_blocks`. Preserve original 18/6/6
training/validation/test source splits, all 23,040 neighborhoods, the 144-dimensional
relaxed topology targets, training-only scaling, and ridge alpha 1.

The old comparison evaluated the 128-dimensional **projector** representation.
Forecasting instead uses the 256-dimensional **encoder** representation. The audit
re-encodes both from the physical input coordinates for every checkpoint and fits
both readouts. It neither retrains encoders nor changes checkpoint selection.

The additional checks cover checkpoint hashes and exact loaded weights, original
source functions, target and coordinate provenance, disjoint source splits,
training-only target transforms, independent float64 ridge normal equations,
independent H0/H1/H2-balanced scoring, and shuffled-training-target controls.
The original training step is rerun with unchanged weights/input/RNG but shuffled
topology targets: a disabled TDA objective must be unaffected; enabled TDA must
affect the loss and gradients, with gradients reaching encoder, projector and head.

Paired confidence intervals resample six test simulations after averaging matched
errors across three seeds. This cohort has already been inspected; comparisons
are exploratory and cannot establish equivalence or forecast superiority.

Run with conda `pointnet`:

```bash
python -m src.research.mace_tda_ridge_audit.run --config configs/analysis/mace_tda_ridge_audit.json
```

Use a fresh output for a new full run. `--stage summarize` reproduces tables and
the paired-seed plot from this run's retained errors, without inference.
`--stage precision` reapplies the historical analysis's `high` matrix precision
to the freshly extracted encoder features and checks both ridge and trained-head
scores against the preserved prediction arrays.

## Completed audit

All six checkpoints were re-encoded and independently scored. The historical
projector ridge means reproduce: **0.032914 versus 0.032813**. The fresh float64
ridge calculation at highest inference precision gives **0.032915 versus 0.032813**.
The 256D encoder comparison, relevant to forecast inputs, gives **0.034454 versus
0.034450**. Both paired source-bootstrap intervals include zero improvement.

There is no observed accidental TDA supervision in the VICReg-only objective:
autograd finds no dependence on the topology target. Enabled TDA produces
nonzero encoder/projector/head gradients and target permutations change the
parameter gradients substantially. The two methods have different saved weights.
Both post-training ridge probes intentionally fit training TDA labels.

Source/target/scaling audits and the independent solver support the original
metric calculation. Shuffled-target controls score approximately 0.84–0.85 versus
0.03–0.035 with correctly paired labels. The shared checkpoint loader did have a
non-default-GPU placement defect exposed by this audit; it was fixed and regression
tested. Fresh scores after exact weight verification reproduce the old values,
so that defect does not explain their near equality. See the
[device-loading record](../../docs/mace_checkpoint_loading.md).

The evidence supports a very small ridge difference for these retained checkpoints,
not general equivalence of TDA and VICReg or optimality for crystallization forecasts.

[Recipe](../../configs/analysis/mace_tda_ridge_audit.json),
[implementation](../../src/research/mace_tda_ridge_audit/run.py),
[results](../../output/mace_tda_ridge_audit/single-frame-20260914/README.md), and
[metric definitions](../../docs/metrics/mace_tda_ridge_audit.md).

## Where topology information comes from

The initialization extension compares the original frozen MLIP MACE with three
fresh random MACE initializations and the audited VICReg / VICReg+TDA models. All
use the same 256D raw encoder and supervised training-only ridge, so dimensionality,
readout supervision and data splits are matched. Random models use the native
constructor, sharing architectural geometry and neighbor normalization but copying
no learned tensor; unused atomic energies and energy scale/shift are neutralized.
Accelerated inference is cross-checked against native e3nn on real inputs.

The primary ridge penalty remains alpha 1. A secondary grid from 1e-9 to 1000
selects alpha only on validation data for every model, guarding against different
feature conditioning. Neither protocol changes encoder weights. Three random
seeds measure initialization variation; MLIP remains one fixed checkpoint. Source
bootstraps do not establish performance on unexamined simulation distributions.

```bash
python -m src.research.mace_tda_ridge_audit.run --config configs/analysis/mace_tda_initialization.json --stage initialization
```

[Initialization results](../../output/mace_tda_ridge_audit/initialization-matched-20260914/README.md).

Completed: [initialization findings](INITIALIZATION_20260914.md). Validation-tuned
ridge gives random MACE 0.036271, original MLIP 0.035854, VICReg 0.033114 and
VICReg+TDA 0.033084. Much of the apparent MLIP advantage at fixed alpha 1 comes
from different regularization sensitivity; topology recoverability is already
strong in random geometric features with a supervised readout.

## Direct comparison without a readout

The user explicitly requested no trained readout either. The direct protocol uses
only the original MLIP and the same three random encoders, preserving their raw
256D outputs. It fits no linear map, scaling, projection or other parameters.
Euclidean and cosine embedding distances are each compared with Euclidean
distances in the raw H0/H1/H2 blocks separately. We report rank correlation and
overlap of the ten nearest neighbors, with fixed k=10 and no parameter search.

All 4,608 held-out structures form the global pool. A second analysis restricts
each comparison to one simulation frame (256 structures; 18 frames), controlling
source/time/temperature differences. Correlation uses 100,000 fixed distinct
global pairs or all pairs within each frame; neighborhood search always uses the
whole eligible pool. A shared row-permutation control accompanies each metric.
We do not use pairwise p-values or call these scores prediction accuracy.

```bash
python -m src.research.mace_tda_ridge_audit.run --config configs/analysis/mace_tda_direct.json --stage direct
```

[Direct results](../../output/mace_tda_ridge_audit/direct-20260914/README.md), with
exact target construction and metric definitions in the exported METRICS.md.

Completed: [direct findings and metric explanation](DIRECT_20260914.md).
Within-frame cosine-distance agreement with topology is 0.350 for random MACE
and 0.398 for original MLIP; top-10 neighbor overlap is 8.67% and 9.29%, versus
3.92% chance. Global correlations are much higher (0.817 and 0.847), motivating
the local comparison. No readout or scaling is fitted in these measurements.
