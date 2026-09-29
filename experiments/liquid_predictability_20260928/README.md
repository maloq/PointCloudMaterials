# Is external-crystal distance encoded in liquid geometry?

- [Multimaterial local-descriptor capacity under a ten-hour fit budget](MULTIMATERIAL_RICH.md).

[RD-MACE256 capacity and duration experiment](RICH_CAPACITY.md): 60 full raw-data passes to learn all rich descriptors.

[Sensitivity, rich-feature learning and paired relaxation extension](CONTROLS.md) records the 2026-09-28 user-requested queue.


Question: does crystal-free observed liquid structure improve prediction of the
distance to an existing external crystal on unseen trajectories, and is failure
specific to the representation, the training sample size, or the observation task?

This study responds to [LCD overfitting](../crystal_interface_20260928/LIQUID_TRAINING_DRIFT.md).
Preserve the fixed Al64 source roles and sealed dense observations. Use the same
183,596 clear fitting contexts from 88 eligible sources, with 41,418 selection,
34,117 calibration and 66,839 test contexts. Their original 150 source assignments
remain unchanged. No new MD simulation, material, history or explicit conditions.
The target is current spatial distance, not future crystallization onset.

The visible positive control uses original visible-query rows retained in the
cache; new visible proposals were not retained by the earlier expansion. Its
recorded proposal-denominator weights define a separate diagnostic population,
not an estimate over every possible visible query in the expanded proposal set.

| Arm | Input/model | Purpose |
|---|---|---|
| prior | Flexible 25-component distance distribution, no observation | Strong unconditional reference |
| descriptor_linear | Linear distribution-parameter head on 224 physical/context descriptors | Accessible structural signal |
| descriptor_mlp25/50/full | Descriptor MLP, nested 25/50/100% training sources | Nonlinear descriptor signal and data learning curve |
| frozen_mace | Frozen original snapshot encoder; train same vector-context readout | Information retained by the existing representation |
| joint_mace | Same snapshot parent; jointly trained vector context and encoder + VCReg | Matched full-data adaptation |
| joint_scratch25/50/full | Randomly initialized MACE, nested source fractions + VCReg | Source-naive encoder learning curve |
| visible_mace | Liquid query with crystal visible in context + VCReg | Separate easier positive control |
| visible_prior | No-input distribution on the visible population | Positive-control reference |

Twelve fits total: five online scientific encoder fits and seven local controls/
readouts. One seed, batch and microbatch 512, one GPU per MACE run. Twelve nominal
blocks of 256 optimizer updates each (3072 updates, 1,572,864 replacement draws).
Keep the update budget identical across source fractions; these are not exhaustive
epochs. MACE uses width/export 128, vector channels 16, two spatial interactions
per patch, two vector-context blocks, identical encoder at all 25 patches and no
special center bypass. Radius 8 Å, nearest 80 candidates, maximum support 32 Å.

Every model selects on full-selection **distance NLL from the first block**.
Joint encoders retain VCReg (.05 variance/.01 covariance) with its existing warmup.
Direction and early-proximity losses are removed from this information assay;
20/32/48-Å Brier scores and calibration remain diagnostics. Every block checkpoint
is retained. The current LCD run remains a historical separate objective.

Before interpreting model scores, inspect within-snapshot physical profiles and
source-bootstrap correlations versus distance. Reconstruct the exact clearance
between consumed input atoms and established crystal; neither clearance nor labels
are inputs. Compare clear predictions on exactly matched held-out rows using NLL,
RMSE, Brier scores and calibration, including clearance/distance subgroups.

Predeclare 2% RMSE reduction as the smallest practically interesting benefit for
this assay. Positive likelihood gain can still demonstrate small predictive
information when RMSE barely changes. Bounds below that threshold concern tested
fitted predictors only; unsuccessful models cannot establish zero physical
information. Source bootstrap does not include training-seed uncertainty. The
existing benchmark has already informed development; stronger confirmation would
require independent held-out sources without resplitting this release.

[Configuration](../../configs/liquid_predictability/al64_20260928.json) ·
[Metric definitions](../../docs/metrics/liquid_predictability.md) ·
[Execution and output](../../docs/liquid_predictability.md).

Reproduction: `python -m src.research.liquid_predictability.queue launch --config configs/liquid_predictability/al64_20260928.json`.
Preparation and numerical verification precede launch as documented in the workflow.

This design uses [proper scoring rules](https://doi.org/10.1198/016214506000001437)
and [predefined practical-equivalence margins](https://doi.org/10.1177/2515245918770963),
with paired independent-source uncertainty rather than treating atoms as replicates.
