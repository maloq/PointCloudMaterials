# Embedding forecast metrics — 2026-09-12

Embeddings are standardized using unique training embeddings only and the configured
scale floor. Histories end at the anchor. Targets are either separate `(0,3]`, `(3,6]`,
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

Storage precision: forecast caches default to float32; explicit `data.storage_dtype: float16` rounds frozen embeddings once before storage. Windows decode to float32 and scaling uses the stored training embeddings. Targets and reported errors therefore refer to those rounded embeddings. Record the dtype and conversion provenance when comparing runs; metric formulas and splits are unchanged.

Execution update (2026-09-13): the optional resident loader keeps the stored
embeddings on the compute device and expands gathered windows to float32 there.
It preserves the original CPU sampler, windows, identities, scaling and formulas;
there is no additional quantization or mixed-precision model arithmetic. Example
arrays explicitly return to CPU for export. `train_s` and `validation_s` are wall
seconds per epoch phase; progress batch counts are execution metadata. A reviewed
implementation transition records old/new hashes when continuing a checkpoint
under this execution backend. Historical exported definitions remain unchanged.

## Spatial context and trajectory mixtures (2026-09-13 extension)

New direct GRU variants may add an observed-frame spatial token: the mean of eight
nearest other cached-center embeddings minus the center embedding, plus their mean
and outer distances divided by 10 angstrom. Selection uses periodic positions at
that same observed frame. Cached centers are sparse (1,024 of 70,304 atoms); this is
surrounding cached-center context, not a dense first-neighbor coordination shell.
The GPU pools stored float16 embeddings in float32 and stores the resulting mean
in float16. The center-embedding training scaler is used for both center and mean.
History interventions in these variants affect central history only; spatial history
remains observed. Existing nonspatial metric calculations are unchanged.

`trajectory_mixture` represents K diagonal-Gaussian distributions over the flattened
complete future path. A single component identity applies to every future frame and
embedding channel. Component log densities sum over time and channels before
`logsumexp(log mixture weight + component log density)`. NLL is the negative joint
log likelihood divided by the number of future coordinates, after mixture reduction.
K=1 is the matched heteroscedastic Gaussian control. A positive per-coordinate
standard-deviation floor prevents zero-variance singularities.

Point metrics and the three separate bin MSEs use the mixture's probability-weighted
mean path. Mixture CRPS is the exact univariate Gaussian-mixture CRPS averaged over
future coordinates: sum_k pi_k A(y-mu_k,sigma_k) minus one half of sum_kl pi_k pi_l
A(mu_k-mu_l,sqrt(sigma_k^2+sigma_l^2)), where A(d,s)=2s phi(d/s)+d(2Phi(d/s)-1).
`coverage90` for mixtures is the fraction of true coordinates whose mixture CDF lies
in [0.05,0.95], equivalent to central marginal 90% interval coverage. It is not a
simultaneous whole-path coverage claim. Energy score uses 16 independent whole-path
samples, mean distance to truth minus half the mean distance between eight disjoint
sample pairs, divided by the square root of the number of future coordinates.
Each sample draws one mixture component for its complete path, then independent
conditional coordinate noise. Best-of-K error is not used as a proper score.

Mixture entropy and exp(entropy) measure predicted component-weight diversity per
origin. `component_weight` and `component_responsibility` retain average prior weights
and full-target posterior responsibilities per component. Low effective count or
unused components are scientific findings, not reasons to omit a run. Mixture
checkpoints use source-mean validation NLL; deterministic checkpoints use MSE.
When a configuration reuses normalization from a checkpoint, its cache identity and
checkpoint SHA-256 are recorded. This reuses unique-training-embedding statistics;
no validation/test moments enter normalization. Existing exported definitions and
source snapshots retain their original identities.

Storage portability (2026-09-13): cache locations resolve through the machine profile. Path relocation does not change metric definitions, data checksums or scientific resume settings. Historical exported definitions remain unchanged.

The optional execution setting `validation_residency: host` keeps validation inputs
and pooled neighbors in host RAM while training data stay on the compute device.
The same windows, float16 neighbor storage, normalization, full validation set and
source-weighted selection criterion apply. Validation batches are copied to the
model device. This avoids retaining both spatial splits in GPU memory and preserves
the configured training batch and optimizer updates. Test evaluation remains resident
on the compute device after the training loaders are released.
