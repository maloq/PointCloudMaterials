# Embedding forecast metrics — 2026-09-12

Embeddings are standardized using training windows only and the configured scale
floor. Histories end at the anchor. Targets are either separate `(0,3]`, `(3,6]`,
`(6,9]` ps means (or the configured horizons) or the complete future path. They are
not cumulative means. Autoregressive evaluation always rolls out predictions.

| Metric | Calculation |
| --- | --- |
| `mse` | Mean squared error over target times/bins and embedding dimensions for each window, in standardized coordinates. |
| `raw_mse` | The same squared residuals multiplied by the squared training scale before averaging. Units: squared encoder-output units. |
| `mse_by_step`, `bin_mse` | MSE over coordinates at each future step or separate bin. For path predictions, bin error compares averages of the predicted and observed path inside each bin. |
| `increment_mse` | MSE of consecutive predicted versus observed path increments, both starting from the actual anchor. |
| `persistence_mse` | Target error of repeating the final observed embedding. |
| `history_mean_mse` | Target error of repeating the mean observed history embedding. |
| `linear_trend_mse` | Target error of extrapolating from the anchor with a least-squares slope fitted across observed history steps. A single observed frame has slope zero. |
| `train_mean_mse` | Target error of a zero prediction in standardized coordinates, i.e. the training embedding mean. |
| `sample_mean` | Mean across all windows; sources with more windows contribute more. |
| `source_mean`, `source_mean_mse` | Mean within each source, then an equally weighted source average. Collection additionally averages the declared seeds. |
| `per_temperature` | Window-weighted mean within a temperature; this differs from source-weighted metrics. |
| `seed_std` | Population standard deviation (`ddof=0`) of source-mean MSE across fitted seeds; undefined for one seed. |
| `gain` | `1 - candidate mean error/reference mean error`, using equally weighted source errors. Higher is better; 0 is parity. |
| `ci95` | 2,000 paired resamples of whole sources with replacement; 2.5/97.5 percentiles of gain. Evaluation uses seed 20260911; collection averages fitted-seed errors before resampling. |
| `change_amplitude_ratio` | Square root of the sum of predicted change energies divided by observed change energies, measured relative to the anchor. 1 means matched aggregate change amplitude. |
| `change_cosine_weighted` | Sum of predicted/observed change dot products divided by the sum of their products of norms; this is not an unweighted mean of window cosines. |
| `nll` | Negative joint Gaussian log probability of the flattened target path, divided by its number of coordinates. |
| `coverage90` | Fraction of coordinates within mean ± 1.6448536269514722 marginal standard deviations. This is marginal coverage, not simultaneous path coverage. |
| `interval90_width` | Average full marginal interval width, `2 * 1.6448536269514722 * std`, in standardized coordinates. |
| `marginal_crps` | Mean Gaussian CRPS: `std * [r*(2*Phi(r)-1) + 2*phi(r) - 1/sqrt(pi)]`, where r is the standardized residual. |
| `energy_score` | Mean norm of 16 sampled joint-path residuals minus half the mean distance of 8 independent sample pairs, divided by `sqrt(number of target coordinates)`. Lower is better. |

`delta_energy`, `predicted_delta_energy`, `delta_dot` and `delta_norm_product` are the
per-window components of the change metrics above. `validation_selection_score`
uses the configured `criterion`; `selected_epoch` is zero-based. Counts and horizon
indices are metadata, not model-quality scores.

Null/blank denotes a zero reference error, zero physical-change denominator, fewer
than two sources for an interval, or an undefined seed spread. No epsilon is inserted.
Intervals condition on fitted seeds and do not claim independent atom/frame samples.
Keep `test_errors.npz`: it retains identities and paired errors required by collection.
Keep the small example arrays for inspecting forecast paths. Historical values
exported again are not recomputed by table export.
