# TDA diagnostic at an unfinished training checkpoint

The diagnostic copies one atomically published `last.pt` checkpoint and records
its exact update, checkpoint hash, original model identity and the matching
validation record. Capture is a progress query, not checkpoint optimization.
The model and its jointly trained nonlinear head are reconstructed from the
training run's frozen code and configuration. No fitting or new online run occurs.
Training continues independently. Epochs are actual updates divided by the
declared full-dataset updates per epoch; planned epochs are recorded separately.

The unchanged panel contains 4,096 Al validation observations from 15 ancestors.
Inference uses batch256, the original BF16 encoder and FP32 head, and both original
0.01-A jitter views with seed `training_seed + batch_start`. Individual squared
errors are averaged over rows and views; predictions are not averaged first.
FP64 diagnostic reductions reconstruct logged `descriptor_half_mse` to within
absolute tolerance0.0002. Predictions and all panel/checkpoint hashes are retained.

The [fixed-final TDA definitions](spatial_tda_difficulty.md) define the 18 blocks
and active outputs. MSE is in the training-standardized descriptor coordinates.
Only nonzero-weight outputs participate in block/family metrics. `r2` is one minus
summed MSE divided by summed validation variance across active block coordinates.
Undefined feature R2 (variance <=1e-12) is blank. `mean_baseline_skill` instead
uses mean squared standardized targets as denominator (constant training mean).
One is perfect, zero matches the named baseline, and negative is worse.

Model `tda_weighted_mse` gives equal weight to the 18 TDA blocks, with equal
weight to active coordinates inside each block. `tda_r2` and
`tda_mean_baseline_skill` use that same weighting in their error and baseline sums.
`descriptor_half_mse` is 0.5 times the full original weighted MSE, including bond
order. `bond_order_mse` averages the 40 bond-order coordinates.

An unfinished checkpoint versus final12-epoch references is an explicitly
unequal-training-budget progress comparison. It cannot establish final model
superiority. These are same-panel validation diagnostics from single runs, with
no nucleation claim or assertion of independent row-level statistical replication.
