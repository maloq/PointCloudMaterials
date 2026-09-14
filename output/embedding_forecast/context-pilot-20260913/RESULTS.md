# Observed-history comparison

Matched 9 ps forecasts; identical sources, anchors, targets, normalization and per-fit update budgets.

| Model | History / ps | Path MSE | +9 ps MSE | Gain vs trained anchor |
| --- | ---: | ---: | ---: | ---: |
| autoregressive_gru | 0 | 0.225769 | 0.243112 | 0.00% |
| mean_residual_gru | 0 | 0.228534 | 0.246494 | 0.00% |
| autoregressive_gru | 1.5 | 0.214840 | 0.232438 | 4.84% |
| mean_residual_gru | 1.5 | 0.216167 | 0.234218 | 5.41% |
| autoregressive_gru | 3 | 0.211791 | 0.229203 | 6.19% |
| mean_residual_gru | 3 | 0.212719 | 0.230293 | 6.92% |
| autoregressive_gru | 6 | 0.209411 | 0.226392 | 7.25% |
| mean_residual_gru | 6 | 0.210087 | 0.227122 | 8.07% |
| autoregressive_gru | 12 | 0.207835 | 0.224253 | 7.94% |
| mean_residual_gru | 12 | 0.208616 | 0.225183 | 8.72% |
| autoregressive_gru | 24 | 0.207421 | 0.223363 | 8.13% |
| mean_residual_gru | 24 | 0.208161 | 0.224322 | 8.91% |

[Full metrics](tables/context-quality.csv) · [Definitions](tables/METRICS.md) · [Context plot](plots/context-quality.png) · [Horizon plot](plots/horizon-errors.png)

Two fitted seeds and 27 previously examined test sources: exploratory evidence. Anchor gains include averaging/denoising benefits; compare the history-mean baseline and reversal intervention before attributing gains to temporal ordering. Compact-model results do not establish the optimal history for the larger production models.
