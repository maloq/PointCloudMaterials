# Line-JEPA Metrics

This file explains the metrics logged by `LineJEPAModule` and `LineJEPALoss`.

Metric names are logged with the stage prefix added by Lightning, for example
`train/line_jepa_pred`, `val/line_jepa_pred`, or `test/line_jepa_pred`.
The generic `loss` metric is also logged as both `{stage}/loss` and
`loss/{stage}`.

## Notation

For each prediction task:

- `target_features`: embedding of the target line position, from the configured target encoder.
- `context_features`: embeddings of the other line positions used as context.
- `context_mean = mean(context_features, dim=1)`.
- `prediction`: predictor output.
- `cos(a, b)`: mean cosine similarity over prediction rows.

If `line_jepa_prediction_target: target`:

```text
prediction_target = target_features
```

If `line_jepa_prediction_target: residual`:

```text
residual = target_features - context_mean
prediction_target = residual
pred_residual = prediction
z_hat = context_mean + pred_residual
```

All per-task diagnostics below are averaged over the prediction rows in the
batch. If a batch predicts multiple target positions per structure, the
prediction row count is `batch_size * number_of_prediction_tasks`.

## Loss Metrics

`line_jepa`

Weighted Line-JEPA objective added to the total training loss:

```text
line_jepa_weight * (
    line_jepa_prediction_coeff * line_jepa_pred
  + line_jepa_sigreg_coeff * line_jepa_sigreg
  + line_jepa_std_coeff * line_jepa_std
  + line_jepa_cov_coeff * line_jepa_cov
)
```

Lower is better. This is logged only when Line-JEPA is active for the current
epoch.

`line_jepa_pred`

Unweighted prediction loss between `prediction` and `prediction_target`.
The loss is either MSE or Smooth L1, depending on `line_jepa_prediction_loss`.
It is averaged over embedding dimensions and prediction rows. If hard
prediction weighting is enabled, the per-row losses are multiplied by the
normalized hard/easy weights before averaging.

Lower is better.

`line_jepa_sigreg`

Unweighted SIGReg loss on pooled regularizer embeddings. It compares random
one-dimensional projected embedding distributions against a standard Gaussian
characteristic function.

Lower is better. Logged only when `line_jepa_sigreg_coeff > 0`.

`line_jepa_std`

Unweighted VICReg-style standard-deviation loss on pooled projected
regularizer embeddings:

```text
mean(relu(line_jepa_std_target - std_per_dimension))
```

Lower is better. Zero means every projected dimension has at least the target
standard deviation. Logged only when `line_jepa_std_coeff > 0`.

`line_jepa_cov`

Unweighted VICReg-style covariance loss on pooled projected regularizer
embeddings. It penalizes squared off-diagonal covariance terms.

Lower is better. Logged only when `line_jepa_cov_coeff > 0`.

`loss`

Total weighted objective for the whole module. In pure Line-JEPA runs this is
the sum of active Line-JEPA losses. In hybrid runs it can also include VICReg
or SwAV losses.

Lower is better, but compare it only within the same config because coefficients
change its scale.

## Target-View Similarity Metrics

These are logged only when `line_jepa_target_view_sim_coeff > 0`.

`line_jepa_target_view_sim`

Raw MSE between two projected embeddings of the same target line position:

- the target position inside the normal encoded line views;
- an extra independently augmented target-only view.

Lower means the target representation is more stable across those two views.

`line_jepa_target_view_sim_weighted`

Weighted target-view similarity loss added to the total objective:

```text
line_jepa_target_view_sim_coeff * line_jepa_target_view_sim
```

Lower is better. This is the value that contributes to `loss`.

`line_jepa_target_view_cos`

Cosine similarity between the same two projected target-view embeddings used by
`line_jepa_target_view_sim`.

Closer to `1` means the two target views point in the same embedding direction.

## Prediction Cosine Metrics

`line_jepa_pred_target_cos`

Logged in target-prediction mode. It is:

```text
cos(prediction, target_features)
```

Closer to `1` means the predictor output points in the same direction as the
target embedding.

`line_jepa_pred_residual_cos`

Logged in residual-prediction mode. It is:

```text
cos(prediction, target_features - context_mean)
```

This measures residual direction quality, not full target reconstruction.
Closer to `1` means the predicted residual points in the same direction as the
true residual.

`line_jepa_context_mean_target_cos`

Cosine similarity between the context-only baseline and the target:

```text
cos(context_mean, target_features)
```

This is a baseline diagnostic. If it is already high, the target embedding is
similar to the surrounding context before the predictor adds anything.

## Residual-Mode Reconstruction Metrics

These are logged only when `line_jepa_prediction_target: residual`.

`line_jepa_residual_norm`

Mean true residual magnitude:

```text
residual_norm = norm(target_features - context_mean)
```

This is the size of the correction needed to move from the context baseline to
the target.

`line_jepa_pred_residual_error`

Mean absolute residual prediction error:

```text
pred_error = norm(pred_residual - residual)
```

Lower is better. Zero means the predicted residual exactly matches the true
residual.

`line_jepa_relative_residual_error`

Residual prediction error relative to residual size:

```text
relative_residual_error = pred_error / residual_norm.clamp_min(1e-6)
```

Lower is better. Around `1` means the predictor error is about as large as the
whole context-to-target residual. Values above `1` mean the predicted residual
is worse than predicting no residual, by this norm metric.

`line_jepa_baseline_residual_error`

Context-only baseline error:

```text
baseline_error = residual_norm
```

This is the target reconstruction error if the model predicted zero residual,
so `z_hat = context_mean`. It intentionally has the same value as
`line_jepa_residual_norm`, but the name makes the improvement formula explicit.

`line_jepa_residual_error_improvement`

Absolute improvement over the context-only baseline:

```text
improvement = baseline_error - pred_error
```

Positive is better. Zero means no improvement over using `context_mean` alone.
Negative means the predicted residual made the target reconstruction worse than
the context-only baseline.

`line_jepa_relative_residual_error_improvement`

Improvement normalized by baseline error:

```text
relative_improvement = improvement / baseline_error.clamp_min(1e-6)
```

Higher is better. `1` is perfect residual prediction, `0` means no improvement
over context alone, and negative values mean worse than context alone.

`line_jepa_recon_target_from_residual_cos`

Cosine similarity between the residual-reconstructed target and the true target:

```text
z_hat = context_mean + pred_residual
cos(z_hat, target_features)
```

Closer to `1` means the full reconstructed target embedding points in the same
direction as the true target embedding.

## Hard Prediction Weighting Metrics

These are logged only when hard prediction weighting is active, meaning
`line_jepa_prediction_hard_weight_low != line_jepa_prediction_hard_weight_high`.

The novelty score is:

```text
novelty = norm(target_features - context_mean)
```

`line_jepa_prediction_novelty_mean`

Mean novelty over all prediction rows. Larger means the target positions are
farther from their context mean in embedding space.

`line_jepa_prediction_novelty_hard_mean`

Mean novelty over the selected hard rows. Hard rows are the top
`line_jepa_prediction_hard_top_fraction` by novelty.

`line_jepa_prediction_novelty_easy_mean`

Mean novelty over the rows not selected as hard. If every row is hard, this is
logged as `0`.

`line_jepa_prediction_hard_threshold`

Smallest novelty among the selected hard rows. Rows with novelty at or above
this threshold are in the hard set, subject to top-k tie behavior.

## Quick Reading Guide

For residual-mode runs, the most useful target-recovery metrics are usually:

- `line_jepa_pred_residual_error`: absolute residual error.
- `line_jepa_relative_residual_error`: residual error normalized by target difficulty.
- `line_jepa_relative_residual_error_improvement`: improvement over context alone.
- `line_jepa_recon_target_from_residual_cos`: final reconstructed-target direction.

For representation-collapse monitoring, watch:

- `line_jepa_std`: should move toward zero when the std regularizer is active.
- `line_jepa_cov`: should move downward when the covariance regularizer is active.
- `line_jepa_target_view_cos`: should usually increase if target-view similarity is active.
