# Forecasting local-structure embedding trajectories — September 11, 2026

Question: does the **ordered history of local-structure embeddings** predict
future local structure beyond a trained anchor or mean-history control? The
preferred task predicts every future frame through 9 ps. The second task predicts
three **separate temporal means over (0,3], (3,6], (6,9] ps**, as requested.

Implementation: [`src/training_methods/embedding_forecast/`](../../src/training_methods/embedding_forecast/).
This is a separate frozen-embedding forecasting workflow. It does not retrain the
geometric encoder or change the original VICReg/relaxation objectives.

## What the latest results imply

The [September 11 MACE report](../mace_vicreg_relaxed_20260910/RESULTS_20260911.md)
found five-frame mean pooling improved balanced relaxed-topology ridge error by
9.69% over a single frame, with a six-source bootstrap interval [1.19%,15.43%].
Pooled attention did not clearly improve on the mean. Real versus reversed past
changed errors much less than replacing history by repeated anchors. These
observations support multiple measurements, but do not yet establish useful
order-sensitive dynamics. They concern relaxed topology, not future embeddings.

In 15/18 balanced-TDA fits, the combined validation objective retained a different
epoch from the lowest logged TDA error. We therefore freeze the target space and
select forecasts on **source-mean validation MSE**, or joint NLL for probabilistic
models. Both the selected checkpoint and the last optimizer checkpoint remain.

The initial target encoder is the original-VICReg **single-frame MACE, seed
20260910**, whose retained checkpoint is specified in each configuration. The
report found its ridge score (0.032914 averaged over seeds) comparable to adding
balanced TDA (0.032813). It provides an efficient fixed snapshot representation.
Using the five-frame mean encoder as the target would instead forecast overlapping
history summaries and artificially smooth short-horizon changes. That is a
different target protocol and is deliberately outside this comparison.

## Data and precise targets

Reuse the 30 independent Al MEAM melt sources already selected by
[`training_fire.json`](../mace_al_denoising_20260910/training_fire.json): 18 training,
6 validation and 6 test sources, at 400/450/510 K. The source selection and its
campaign producer establish independent preparation seeds and whole-source splits.
All atoms, times and derived windows from one melt stay in one split. Test sources
have already been inspected in previous studies; results remain exploratory.

For a fixed center atom i, z_i(t) is the frozen 256-channel snapshot embedding
before the VICReg projector. At every frame, query that center's instantaneous
80 nearest atoms under periodic boundaries. **Only the center identity is fixed**;
neighbor membership may change. Past neighborhoods never use future-selected atom
membership. Keep the trained 9.192189 Å normalization and encoder arithmetic.
No source identity, temperature, future coordinates, targets or augmentation-view
axis enters the forecasting network.

The measured cadence is 0.75 ps. The primary input is nine embeddings at
[-6,-5.25,-4.5,-3.75,-3,-2.25,-1.5,-0.75,0] ps. The full target is twelve embeddings
at [0.75,1.5,...,9] ps. Each temporal-mean target averages four future embeddings:

| Target | Included offsets, ps |
| --- | --- |
| First mean | 0.75, 1.5, 2.25, 3 |
| Second mean | 3.75, 4.5, 5.25, 6 |
| Third mean | 6.75, 7.5, 8.25, 9 |

There is no t=0 contribution and no cumulative averaging. These are time means
along one observed future, not empirical averages across independent shooting
branches. Squared-error regression estimates their conditional expectation.

Preparation encodes contiguous segments around the existing anchors 40/240/640,
without relaxation or new simulation. Primary preparation uses 64 centers/source,
25 frames/segment, and five forecast anchors/segment spaced 0.75 ps apart: 17,280
training, 5,760 validation and 5,760 test windows. These overlapping windows are
not independent replicates; uncertainty is estimated at the source level.
The embedding arrays occupy about 141 MiB, plus small identity/timeline metadata.

The cache stores each segment once as float32 `(center,time,256)` `.npy` shards.
The loader opens them as memory maps and gathers a whole batch per shard, with
arithmetic window indices, persistent spawned workers and pinned GPU transfers.
It does not copy every overlapping window or rerun MACE each epoch. Embeddings
stay float32 to preserve the small dynamical signal. Existing verified float16
positions, float32 boxes, exact IDs/timesteps and simulation precision are unchanged.
Cache provenance includes encoder/config/producer/source-manifest checksums and
checksums of all generated embedding, frame and identity arrays. Completed shards
are verified on preparation restart; a changed protocol requires a new cache.

Fit mean and per-channel scale on unique **training** embeddings only. The scale
floor is 5% of the RMS training-channel standard deviation, recorded explicitly
to avoid magnifying near-constant channels. No unit-sphere normalization removes
embedding magnitude. The same transformation defines both inputs and targets.

## Models and objectives

The initial models predict a correction to the last observed embedding. The
pilot-informed primary models instead predict a correction to the **observed
history mean**, making that strong baseline available in every latent channel
without learning to reconstruct it through a width-128 bottleneck. History tokens
contain z, its backward first difference and the actual relative time in ps
(divided by 9 ps for the network). There is no centered finite difference or
velocity estimate from quantized coordinates. A shared time-conditioned residual
MLP in the direct models decodes all requested future times together. Their
forecast-frame coupling comes from the common history state and decoder. The
autoregressive extension below instead feeds each prediction into the next step.

| Variant | Role |
| --- | --- |
| `path_mean_residual` | Direct reference: two-layer GRU, width 128, predicts a correction to the history mean at all 12 future times |
| `path_mean_residual_control` | Same mean residual baseline and network, but history tokens contain only the history mean; tests order-sensitive information beyond a learned mean-based forecast |
| `bins_mean_residual`, `bins_mean_residual_control` | Corresponding direct-bin model and learned mean-only control |
| `path_gru` | Original anchor-residual GRU, retained as an explicit baseline |
| `path_anchor` | Same GRU/decoder capacity; every input frame repeats the anchor |
| `path_mean` | Same capacity; every input frame repeats the history mean, with the true anchor residual skip |
| `path_mlp` | Dense history encoder, a simple order-aware baseline |
| `path_transformer` | Two-layer, four-head attention over observed history only |
| `path_gaussian` | GRU predicting a joint low-rank Gaussian over the entire 12×256-dimensional path |
| `bins_gru`, `bins_anchor`, `bins_mean`, `bins_mlp` | Corresponding direct three-bin predictions |
| `path_ar_rollout` | GRU encoder plus recurrent residual decoder, trained through its own 12-step predictions |
| `path_ar_mean_control`, `path_ar_anchor_control` | Same autoregressive capacity with mean-only or repeated-anchor input |
| `path_ar_rollout_mse` | Autoregressive rollout trained with frame MSE alone |
| `path_ar_teacher_forced` | Same architecture trained on observed future prefixes with one-step MSE; evaluated on its own rollout |

The dense encoder-decoder baseline is motivated by the simplicity of
[TiDE](https://arxiv.org/abs/2304.08424), rather than assuming attention is necessary.
This implementation is a small residual forecaster, not a reproduction of TiDE.

For deterministic paths, let y and p be the standardized target and forecast:

`L = MSE(p,y) + 0.25 MSE(bin_mean(p),bin_mean(y)) + 0.1 MSE(diff([z0,p]),diff([z0,y]))`.

Each MSE averages its own frame/bin and channel axes. The last term matches
observed increments; it does not penalize the forecast merely for moving.
The first term and checkpoint selection remain full-path MSE. For direct means,
use equal-weight MSE over the three bins, with no invented within-bin increments.
Loss weights are experiment settings. Follow-up ablations should remove the bin
and increment terms separately on validation data before expanding model size.

The probabilistic variant uses **joint Gaussian NLL divided by 12×256**, without
additional MSE terms. Covariance is `diag(softplus(s)+0.03)^2 + U Uᵀ`, with rank
four U spanning time and embedding channels. It can draw correlated whole paths;
a set of independent per-frame error bars cannot do this. Factor weights start
small and nonzero, since exactly zero factors have zero covariance gradient.
This is a unimodal conditional distribution, not evidence that crystallization
futures are Gaussian or that a sampled path is a physically realizable structure.
If held-out futures are clearly multimodal, a subsequent conditional path
diffusion model is a reasonable next experiment; probabilistic diffusion
forecasting has precedent in [TimeGrad](https://arxiv.org/abs/2101.12072). It is not
implemented or required for this first controlled comparison.

### Autoregressive prediction

`architecture: autoregressive_gru` reuses the two-layer history GRU and the
same prepared embedding windows. It adds a GRUCell decoder whose inputs are the
previous embedding, the fixed encoded history context and the next physical time.
The decoder predicts a residual increment and adds it to the previous embedding.
For the pilot, the initial embedding is the observed history mean, motivated by
the completed direct-model comparison. `autoregressive.initial_state: anchor`
is also supported as an explicit alternative; it is not an implicit fallback.

With history context c, decoder state h and forecast p:

`h_k = GRUCell([p_(k-1), c, time_k], h_(k-1))`

`p_k = p_(k-1) + residual_MLP(h_k, time_k)`.

Set h_0=c and p_0 to the declared observed baseline. There are twelve sequential
steps at 0.75 ps. **Predictions are never detached in rollout training**, so a
late-horizon loss can change the earlier predictions and recurrent state.
The four frames in each requested bin can still be averaged for comparison with
direct means. The autoregressive architecture accepts deterministic full-path
targets only; a correlated Gaussian over the direct output is a different method.

Training strategy is explicit in each variant's `autoregressive.training`:

- `rollout`: use p_(k-1) at every future step during both fitting and evaluation.
  The main variant uses the same frame/bin/increment loss as direct paths.
- `teacher_forcing`: only during fitting, replace p_(k-1) by the actual preceding
  future embedding y_(k-1). The first prediction still uses observed history;
  y_k is never an input to prediction k. This protocol uses one-step MSE with
  bin and increment loss weights zero. `path_ar_rollout_mse` supplies the matched
  frame-MSE-only comparison, separating conditioning strategy from auxiliary losses.

Every validation epoch, checkpoint selection and final test **runs the complete
predicted rollout without any future observations**, including for the
teacher-forced fit. The public inference API is `model(history)`; teacher forcing
has a separate training-only method that raises during evaluation. Inspect the
saved per-step error curve, especially +0.75 versus +9 ps, rather than interpreting
teacher-forced training MSE as a deployed forecast score.

The AR controls have identical parameter counts. The AR decoder adds parameters
and sequential work relative to the direct model; the direct-versus-AR comparison
is therefore a comparison of these implemented methods, not a capacity-matched
architecture isolation. Counts and elapsed times are retained with every fit.

## Experiments, selection and analysis

1. Run the small real-data pilot first. It uses the same 30 source splits and
   three contexts, eight centers/source, one forecast anchor/context, one seed:
   432/144/144 train/validation/test windows. Test both objectives and their
   trained anchor controls, plus the joint Gaussian. This verifies the pipeline
   and rough signal; it cannot settle the architecture comparison.
2. Run the configured matrix with seeds 20260911/20260912/20260913, matched
   optimizer, batch size, update budget and validation patience. Prioritize the
   full-trajectory models. AdamW LR 0.001, weight decay 0.0001, 60 epochs maximum,
   cosine decay to 5% of initial LR, gradient norm clip 5, patience 12.
   The matrix now includes the original 14 variants and five autoregressive
   additions; `--variant` allows prioritizing the useful comparisons.
3. Run the 3 ps history ablation for the two mean-residual, two anchor-residual
   GRU methods and the autoregressive rollout on the **same cache,
   center IDs, anchor times and futures**. `anchor_history_ps: 6` fixes the common
   anchor grid; reducing `history_ps` does not create extra earlier anchors.
4. Only after consistent gains, expand independent sources/centers or compare
   representations. A changed encoder needs its own cache and its own baselines;
   raw MSE from two different latent coordinate systems is not a fair ranking.

Monitor training loss components, gradient norm, LR, validation MSE by future
frame and bin, and source-mean selection score every epoch in `training.jsonl`.
There is no test-set checkpoint selection or hidden weights-only restart.

Final analysis records:

- Standardized MSE per future frame/bin, raw embedding MSE, and both sample-mean
  and equal-source-mean aggregates; separate scores by temperature and source.
- Skill `1 − model_MSE/reference_MSE` against persistence, the observed history
  mean, past-only least-squares linear extrapolation and the training mean.
  Compare against separately **trained** anchor and mean controls to isolate
  temporal information from learning a denoiser or adding network capacity.
- Full-path predictions averaged into the same three bins and compared directly
  with the direct-bin method; this is the fair cross-objective comparison.
- Increment error, predicted/true change RMS ratio and norm-weighted change
  cosine. Large apparent level accuracy with near-zero predicted changes is a
  regression-to-static-state warning, not a successful dynamics forecast.
- Repeated-anchor and reverse-past interventions, keeping the anchor fixed.
  These measure sensitivity under distribution shift; the trained controls
  provide the stronger information comparison.
- For the Gaussian: joint NLL, marginal CRPS, 90% marginal coverage and width,
  and whole-path energy score using 16 reproducibly sampled trajectories.
  Marginal 90% coverage is not simultaneous 90% coverage of the entire path.
- Paired source-bootstrap intervals after averaging errors across model seeds;
  seed spread is also reported. Six sources provide limited uncertainty
  resolution. Undefined ratios/intervals are explicit nulls, never epsilon fixes.

Acceptance: reproducible positive skill over the trained anchor and mean
controls, including late-bin performance, across independent sources/seeds.
Ordered-history benefit additionally needs a real-versus-reversed-past effect
and improvement over the trained mean control. For distributional forecasts,
require NLL/CRPS/energy and coverage–width checks together. A falling composite
loss, smooth mean trajectory or good embedding reconstruction alone does not
establish phase-transition predictability. Physical-descriptor probes on observed
future structures are a subsequent validation, not labels already established here.

## Reproduction and artifacts

Run from the repository root in `pointnet`:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot.json --stage all
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_mean_residual.json --stage train
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_autoregressive.json --stage train
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/pilot_comparison.json --stage collect

conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/main.json --stage prepare
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/main.json --stage train --variant path_mean_residual --seed 20260911
# Omit --variant and --seed to execute the configured matrix serially.
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/main.json --stage collect
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config experiments/embedding_forecast_20260911/history3.json --stage train
```

`--stage evaluate` reconstructs a selected model from `best.pt` and reuses its
saved scaler, target definition and verified cache. `--device cpu|cuda:0` changes
compute placement. Fresh training refuses existing run directories; choose a
new configured output for a new attempt. Preparation resumes verified complete
shards with the identical cache protocol; interrupted partial shards are rebuilt.
Explicit `--resume` restores the last model, optimizer, scheduler, fitted scaler,
sampling and CPU/CUDA RNG states; the scientific configuration and implementation
must match. `--epochs-per-invocation N` pauses at a completed epoch boundary for
a queued continuation. A paused chunk does not evaluate test data.
`--stage queue --queue-config PLAN.json` freezes execution source/configuration
and submits shared preparation, independent Slurm training chains, and final
collection. See the [submitted enlarged runs](SCALE_RUN.md) for the 125-source,
32-epoch direct and autoregressive comparison with augmented histories, now one
Slurm job per model.

Primary cache: `/home/ids/vmorozov/training-cache/embedding-forecast-20260911`.
Outputs: `output/embedding_forecast_20260911/{pilot,main,history3}`. Per fit:
`best.pt`, `last.pt`, `training.jsonl`, `test_metrics.json`, paired-row
`test_errors.npz`, small `test_examples.npz`, `history_interventions.json`,
`forecast_scores.png`, configuration/data summaries and completion status.
The test examples are in original encoder units. Test errors are in the saved
training-standardized units unless labelled raw. No full duplicated prediction
cache is retained. Matrix collection writes `comparison.json` and refuses
incomplete runs or mismatched held-out rows.

The new `src/training_methods/embedding_forecast/` package and
[`test_embedding_forecast.py`](../../tests/test_embedding_forecast.py) are maintained
implementation and regression coverage. The reports and JSON configurations
are versioned experiment records, including the pilot
follow-up and combined comparison. Generated caches, logs, checkpoints, plots and
pilot scores are disposable run artifacts under the declared output locations.
See [RESULTS.md](RESULTS.md) for the initial pilot and
[AUTOREGRESSIVE_RESULTS.md](AUTOREGRESSIVE_RESULTS.md) for the five-fit follow-up,
which favors AR rollout over the direct reference in this single-seed test.
The larger matrix and
matched-history-length ablation are configured but have not been launched.
