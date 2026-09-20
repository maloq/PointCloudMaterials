# Does causal geometry-anchored neighbor prediction improve a snapshot state?

The reviewed v1 could numerically use future features in training normalization,
reduce equivariant prediction loss by rescaling learned tensors, and change SIGReg
strength by changing statistical batch size. V2 removes those confounds without
rewriting old results. Reconstruction improvement alone does not establish dynamics.

Primary population: native Al / Lee MEAM only, 0.75 ps saved-frame cadence, 36
training and 15 development roots. Existing multi-material and legacy runs are
external comparisons, not matched causal ablations. One seed follows the user's
standing faster-results preference; three-seed confirmation is deferred.

| Arm | Reconstruction + SIGReg | Fixed future geometry | Current neighbors | Future neighbors + center |
|---|---|---|---|---|
| A | yes | no | no | no |
| B | yes | yes | no | no |
| C | yes | yes | yes | no |
| D | yes | yes | no | yes |
| E | yes | yes | yes | yes |

All arms use identical current/next independent reconstruction targets, initialization,
batch 256, encoder/head peak LR .0002/.002, 10% warmup cosine, 1,280 updates,
327,680 anchor draws, selective BF16 and compiled MACE. Angular outputs receive
fixed smooth moments with saved physical scales. SIGReg is explicitly per-sample
characteristic-function discrepancy, weight .1, after the invariant projector.
No slowness, retrodiction or history context. Query families have equal fixed mass.
No automatic long-run promotion. Capacity-changing tensor-product predictors,
direct-export SIGReg and history controls remain later isolated ablations.

Development reports physical/TDA retention, per-degree geometric scale/error,
projected/raw spectra and future physical errors against persistence, condition
means, geometry ridge and mean reversion. Scores from legacy mixed-material models
are not directly compared as if they used the same target normalization.

The independent downstream axis is matched frozen-feature crystallization prediction
from one current local structure. All completed v1 models, the three old VICReg
MACE/GATr references and A–E use the same linear/MLP hazard budgets and original
150-source split roles. The test cohort is historically used, not newly untouched.
Test metrics never select training arms. No robust one-seed ranking is claimed.
A smaller NLL/AP change must be interpreted with source uncertainty and misses.

Reproduce using `python -m src.training_methods.neighborhood_jepa.v2.data --config
configs/neighborhood_jepa/v2_native_al_20260920.json`, tests in
`tests/test_neighborhood_jepa_v2.py`, and the existing queue's v2 module.
[Metric definitions](../../docs/metrics/neighborhood_jepa_v2.md),
[crystallization definitions](../../docs/metrics/neighborhood_crystallization_v2.md),
[execution record](../../docs/neighborhood_jepa_v2.md).

## Three-hour scale runs on node53 and node59

Two additional one-seed arm-E runs retain the corrected causal interface, fixed
angular moments, physical/TDA anchors and sample-normalized SIGReg. Both expand
MACE width from 32 to 64 (212,499 to 657,299 encoder parameters), while keeping the exported 128 invariant and 120
angular channels. Training uses 32,768 native Al anchors across 90 independent
training lineages (versus 8,995 / 36); selection remains the same 480 anchors /
15 lineages. Data are existing Lee 2003 MEAM trajectories; no new MD is generated.

| Node | Paired devices | Global anchor batch | Training target |
| --- | --- | --- | --- |
| node53 | 2 H100 NVL | 1,024 | approximately 3 hours |
| node59 | 2 RTX PRO 6000 | 2,048 | approximately 3 hours |

Each anchor requires 14 observations. Graph microbatches of 128 are distributed
across both GPUs, followed by one global loss and exact gradient replay. SIGReg
uses the complete anchor batch, not independent device-local statistics. BF16,
cuEquivariance and torch.compile are enabled. Encoder/head peak learning rates
remain 0.0002 / 0.002 with 10% warmup and cosine decay. No batch-based LR scaling.

A separate disposable timing preflight fixes the update count before training;
the scientific LR schedule never adapts to timing. The selected budget must be
at least 40 sampled epoch equivalents. Sampling is uniform without replacement
within each update; an epoch equivalent is 32,768 anchor draws, not a guaranteed
complete shuffled pass. Exact updates/epochs and measured parameter counts are
recorded in each run's `technical/preflight.json` and `resolved-config.json`.

These runs jointly change data, capacity and optimization budget. They test
whether scaling helps the corrected protocol; they cannot isolate each cause.
Node comparisons also differ in anchor batch and achieved update budget.
Selection still uses present physical/TDA development error. After each fit,
frozen linear and nonlinear crystallization probes use the established matched
150-source assay, including classification, calibration, timing and spatial
metrics. Existing development/test reuse remains exploratory.

Recipes: `configs/neighborhood_jepa/large_20260920/`. Results:
`output/neighborhood_jepa/large-node53-20260920/` and
`output/neighborhood_jepa/large-node59-20260920/`.

## Scale-run findings (20 September 2026)

[Completed scale-run analysis](../../output/neighborhood_jepa/large-20260920/RESULTS.md): both large models improve crystallization AP over small corrected arm E, but neither exceeds the older VICReg MACE or hand-crafted geometry baseline. Node53 reconstructs better but has slightly weaker AP/NLL than node59. Both have concentrated invariant covariance and next-frame physical error worse than geometry ridge. One seed; no causal attribution to width, data or batch separately. Node53 reached its three-hour training limit; both frozen assays are complete.


Baseline input clarification (verified against the frozen producer): the historically named `geometry-baseline` is a **geometry + motion + bond-order descriptor baseline**, not positions-only. Its 136 descriptor inputs comprise 85 geometric packet components, 43 velocity-derived packet components, and 8 order/density/coordination components. The packet uses relative positions/velocities within 7 Angstrom with taper from 5 to 7 Angstrom; bond-order statistics use the center and its 12 nearest neighbors and their bonds. Seven temperature/time conditions are appended to both descriptor and encoder probes. The snapshot MACE/GATr encoders do not receive velocities. Therefore, this comparison is useful as a richer physical-input reference, but does not establish that hand-crafted geometry alone outperforms a learned representation on matched inputs. A positions-only descriptor ablation (93 descriptors plus conditions) has not been run in this comparison. No TDA, PTM class labels or future observations enter these descriptor features. Historical artifact names and scores are unchanged.
