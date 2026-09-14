# History, spatial context and full-path mixtures

The maintained forecast metrics are defined in `forecast.md`; the local physical
assay follows `forecast_crystallization.md`. This comparison exports newly evaluated
runs. Historical tables and source snapshots retain their original definitions.

## Matched embedding forecasts

Every model predicts 12 future embeddings (0.75–9 ps). Training and standard test
windows have 6 ps spacing and common origins that permit 12 ps of past observation.
A 6 ps model receives only its last nine frames; a 12 ps model receives 17 frames.
Training uses every cached center in the 74 training sources, with 24 validation and
27 test sources. The 256-channel normalization comes from unique training embeddings;
collection requires bitwise equal mean/scale and identical test source, atom and
origin arrays. Reported embedding scores average within source, then across sources.

Spatial context is the mean embedding of the eight nearest OTHER cached centers,
reselected using periodic coordinates at each observed frame. These centers are a
sparse sample of the simulation, not the eight nearest physical atoms. The mean and
outer neighbor distances are additional inputs in angstrom divided by 10 angstrom.
Pooling is float32, then stored in float16 on the GPU. No future positions or future
neighbor embeddings enter a forecast. Augmentation alters the central history only:
Gaussian noise 0.01 in standardized units and past-frame dropout 0.15; the anchor is
retained. The observed spatial history is unchanged under central-history interventions.

Probabilistic forecasts use one latent component for the entire 12-by-256 path,
with a diagonal conditional Gaussian in each component. The training objective is
joint mixture negative log likelihood summed over time/channels before log-sum-exp,
divided by 3072. No independent component selection at each future time is allowed.
K=1 controls for the Gaussian likelihood; K=4 tests additional trajectory modes.
Validation chooses the lowest source-mean MSE for deterministic models or joint NLL
for probabilistic models. Compare MSE for the mixture-weighted mean, marginal CRPS,
central marginal 90% coverage, full-path energy score, gate entropy/effective component
count, and posterior responsibilities; definitions and sampling are in `forecast.md`.
Low mixture occupancy is evidence that the modes collapsed, even if NLL improved.

## Local crystallization and timing

Physical truth is reused from the independent local assay: PTM FCC/HCP/BCC of the
exact tracked center, not the first global nucleus. There are 64 fixed labeled centers
per source. Primary onset is the first frame of three consecutive crystal frames;
a nine-frame persistence sensitivity is also reported. Confirmation frames after
onset are used only to define physical truth. Never-onset trajectories remain negatives.
The previously fitted train-only ridge crystal readout is frozen for all runs.

All local forecasts use the same every-frame origins 16 through 780 (765 origins),
leaving enough future truth for the nine-frame sensitivity. Results use the same 24
validation and 27 test sources, irrespective of model history. The risk set is origins
before first onset with the last three physical observed frames all noncrystal.

Two readouts are reported for probabilistic models:

- `mean_margin`: the frozen linear crystal readout of the mixture-weighted mean path.
- `frame_crystal_probability`: the exact per-frame probability of a nonnegative linear
  readout under the mixture. Component projected variance is the sum of embedding
  variances times squared readout weights; component Gaussian CDFs are gate weighted.
  This is a probability under the embedding model/readout, not a calibrated physical
  sustained-onset probability.

At horizon H, the event score is the maximum of the first H/0.75 frame scores. A
threshold maximizes F1 on validation risk windows separately for each fitted seed,
readout, horizon and persistence definition; highest threshold wins exact ties.
Test labels do not select thresholds or checkpoints. Predicted time is the first
forecast frame crossing that threshold. The forecast crossing itself is not required
to persist: this preserves the previous assay and compares both readouts identically.

Transition tables report precision, recall, F1, false-positive rate, AUROC and average
precision over pooled eligible windows. State tables classify PTM state at exactly H.
Timing MAE/bias and conditional timing accuracy use true positives only; missed events
remain in event recall and `timed_within_1_5_ps_recall`, whose denominator is all actual
events in the risk windows. Fixed-lead tables use one eligible origin per local event,
exactly H ps before onset. Their timing errors can only be early or zero because the
true onset is at the final predicted frame. They do not measure late-time error.
Blank metrics are undefined, never zero. These repeated-window metrics overweight
long pre-onset histories by design; fixed-lead recall supplies an event-level view.

## Uncertainty and pairing

Each individual run bootstraps 27 independent test sources 2,000 times, keeping all
centers and overlapping windows together. For paired event F1, first average the
per-source confusion counts across the same configured seeds, then bootstrap the same source
indices for candidate and control. This is F1 of seed-averaged confusion counts, not
mean per-seed F1 and not a prediction ensemble. Paired embedding differences first
average each source's MSE across seeds, then bootstrap the paired source differences.
Intervals are 2.5/97.5 percentiles. Seed replication and physical sources are distinct;
these intervals describe source uncertainty conditional on the configured fitted seeds.
Bars show mean per-seed scores and dots show individual seeds; tables retain full values.
Previously inspected test sources make this an exploratory experiment.

Spatial fits keep validation embeddings and neighbor pools in host memory, copying
each batch to the GPU. Training batch 8192, data, optimizer updates and validation
selection are unchanged. Nonspatial fits retain both splits on the GPU. This is a
memory-residency difference, not a change to the experimental objective.
The optimized runtime stages the complete validation split on the GPU only during
validation and releases it before training resumes. It copies the same prepooled
tensors without changing their precision, validation windows, scores or selection.
Earlier completed fits retain their original runtime and exported definitions.

## Short-history extension

A configuration-driven comparison also tests 0 (anchor only), 1.5, 3 and 6 ps of
history for deterministic and spatial K=4 forecasts against their 12 ps counterparts.
At 0 ps the model receives one central observation and, for spatial models, the
same-frame neighbor pool. The clean anchor is never augmented; intermediate-frame
augmentation therefore has no effect in that condition. All lengths retain the
same 12 ps common-origin grid, future targets, model widths/depths, batch, epochs,
source split and training normalization. Parameter counts are unchanged within each
model family. The fixed physical risk set is identical, including its three-frame
noncrystal eligibility criterion; those physical labels are not model inputs.

The extended plan explicitly lists comparison pairs and retained `reference_runs`.
References point to existing selected checkpoints and their `local_directory` assay
artifacts. They are verified and read without retraining, recopying or rewriting old
scores. New fits and extended tables have a separate output root. Per-run thresholds
and metrics, seed averaging and the source-bootstrap formulas above are unchanged.
The original six-condition plan retains its original comparison pairs.

## Interim completed-fit cohorts

An explicit interim plan may contain one matched seed per condition or a subset of
conditions with both seeds completed. Every compared condition must have exactly the
same seed set. With one seed, averaging across seeds is the identity operation;
the source bootstrap measures source uncertainty conditional on that one fitted model,
and supplies no estimate of seed variability. Cohort membership, excluded unfinished
fits and the seed count belong in the report and cohort metadata. Tables from cohorts
with different seed sets are not pooled. Metric formulas, source pairing, physical
labels, validation-selected thresholds and checkpoint checks remain unchanged.

Presentation update: the generated report embeds its existing transition plot and
links the full-size image directly. Bars remain the mean per-seed score and dots
remain individual fitted seeds; metric calculations and exported values are unchanged.

## Structure and embedding trajectory visualization

The `trajectories` workflow reuses completed checkpoints, the frozen physical assay,
and exact cached source/center/frame identities. It does not change cohort scores.
Four diagnostic test examples use the spatial K=4 model's saved 9 ps, three-frame
validation threshold: correct within 1.5 ps, early by more than 3 ps, missed, and
false alarm within 9 ps. Positive cases have true onset 3–6 ps after the origin to
leave visible post-onset structure. In manifest source order, choose the first
center/origin of the first still-missing category, using a distinct source per case.
These outcome-selected examples are not an unbiased estimate of predictive accuracy.

Point clouds follow the actual encoder producer: the same center atom and its
instantaneous 80 nearest periodic atoms (including the center), reselected at every
frame. Coordinates are minimum-image offsets in angstrom, without temporal rotation
alignment; neighboring identities can change. They are measured simulation frames,
not structures decoded from predicted embeddings. Only center PTM labels are shown;
neighbor colors denote membership of the origin neighborhood, not neighbor phases.

All embeddings are standardized by the selected models' identical training mean and
scale. The revised visualization fits one two-dimensional UMAP to the retained equal
sample from all 74 training sources (4,736 embeddings). Parameters are 30 neighbors,
minimum distance 0.15, Euclidean input metric, and fixed fit/transform seeds. True
and predicted query embeddings are deduplicated and transformed together in a single
batch; overlapping windows reuse exactly the same transformed coordinates. The mean
embedding is transformed directly: UMAP of a mean is not the mean of UMAP coordinates.
Axes are nonlinear display coordinates without physical units or explained-variance
percentages. Previous PCA figures, raw extraction and frozen definitions are retained.
See the official [UMAP transform protocol](https://umap-learn.readthedocs.io/en/latest/transform.html).

The complete-path overview separates spatial occupancy from time. Every frame appears
as a point without connecting lines in the UMAP map; colors indicate measured PTM
state. Time panels retain faint raw points and summarize nonoverlapping eight-frame
blocks by their median UMAP coordinates at mean sample time (6 ps bin width at this
cadence). The final partial block is retained. These medians are visual summaries,
not new embeddings or smoothed training inputs. Physical-state strips retain all frames.

The structured feature heatmap shows each feature's additive contribution to the
change in crystal-classifier score relative to its mean observed 17-frame history.
Writing the frozen score in forecast-standardized coordinates as s(z) = w·z + b,
the cell for feature j and time t is w_j [z_j(t) − mean_history(z_j)]. The full sum
over 256 features equals s(z(t)) − mean_history(s(z)); the displayed top 20 features
alone need not sum to it. Choose these 20 by decreasing |w_j| times the feature's
sample standard deviation on the retained training sample. The order is fixed for
all cases. Positive values push the score toward crystal; negative values push it
away. This is an exact additive decomposition of the learned linear score, not
physical causality or feature independence. Separate total-score curves use all 256
features. Full-channel future heatmaps show standardized feature changes from the
same history baseline, with features ordered by average-linkage clustering of
training Pearson-correlation distances and optimal leaf ordering. Color scales are
shared and unclipped. Neither ranking nor grouping uses test futures.

The crystal-classifier score (formerly crystal margin) is the crystal ridge score
minus the noncrystal ridge score. It is neither a physical order parameter nor a
probability. Model P(positive score), formerly readout probability, is the analytic
probability of a nonnegative linear score under the predicted embedding mixture at
each future frame: sum_k pi_k Phi((w·mu_kt+b)/sqrt(sum_j w_j² sigma_ktj²)). It is not
a calibrated probability of physical PTM crystallinity or of sustained onset. The
validation-selected warning threshold and retrospective PTM onset labels are unchanged.
Plots give measured state, score and probability separate axes with these meanings.
`observed-paths.csv` now records UMAP1/2, time relative to the origin, tracked atom,
measured binary PTM state and the frozen classifier score in the new result root.

The K=4 paths are the conditional component means and the gate-weighted mean. Each
sample uses the maintained sampler: choose one component for all future frames,
then draw its diagonal Gaussian residuals independently across time and channels.
Samples are neither smoothed nor selected for closeness to the actual future.
Component indices are model-specific and do not establish distinct physical phases.
Selected-window CPU predictions are checked against saved GPU assay readouts; spatial
pooling retains the assay's float32 reduction and cache-dtype storage. Exact arrays,
input/checkpoint hashes and the implementation contract accompany exported figures.

## Broader spatial context

The context-scale extension compares 32, 128 and 512 nearest OTHER cached centers
against the retained 8-center control, at 3 and 12 ps history with matched seeds.
The 256-dimensional neighbor mean and mean/outer neighbor distances remain the
same model inputs; width, layers, mixture components, augmentation, windows,
normalization, optimizer, 12-epoch budget and checkpoint selection are held fixed.
These are sparse cached centers, not immediate physical-atom neighbors. Each
radius is measured from the actual periodic geometry and exported in the spatial
cache's per-source records. `spatial_neighbors` is exported as comparison metadata.

An explicit `prepool_embeddings` cache option computes same-frame float32 neighbor
means once on CPU, stores the result in the embedding cache dtype, and reuses those
exact stored values in training, validation, test and local physical inference.
This avoids repeated broad-neighborhood gathers and does not add future information.
All pooled arrays, neighbor indices and distances have retained hashes. Gathers
are bounded to 2^24 embedding elements; the original K=8 / D=256 chunk stays 8192
center-frame pairs. The frozen 8-center reference used the original float32
CPU/GPU reduction followed by float16 storage; reduction-device rounding is not
claimed bitwise equivalent. Runtime checks require the fitted neighbor count to
match its cache, and the physical plan must name each run's exact spatial cache.
Separate per-context paired reports preserve completed cohorts if later fits stop.
Scientific metrics, validation thresholds and source-bootstrap formulas are unchanged.

## Completed-cohort summary

The collector's `--stage summarize` reads completed spatial-size, history and original
comparison artifacts into a fresh report without changing checkpoints or thresholds.
Summary CSV rows are arithmetic means of the two retained per-seed metrics. They
are not ensemble predictions. Plots of paired F1 differences reuse the collector's
seed-averaged-count source-bootstrap intervals verbatim; these unadjusted intervals
are distinct from differences of mean per-seed F1. All input JSON, selected test
metrics, training logs and source-statistics archives are fingerprinted in the report.

Physical spatial scales are medians across test-source median outer distances and
means across test-source mean neighbor distances, as recorded by cache preparation.
Temperature tables first average the saved seven-column source statistics across
seeds, then sum within each test temperature. Columns are TP, FP, FN, TN, absolute
timing-error sum, signed timing-error sum and count within 1.5 ps. F1 is
2 TP / (2 TP + FP + FN); recall and timed recall divide their respective detected
counts by TP + FN. This keeps the original globally selected validation threshold
for every seed, horizon and persistence rule. Temperature strata are descriptive,
not independently tuned evaluations or new bootstrap intervals.

Component-weight bars rank weights separately within each fitted seed: component
identities are not aligned across seeds. Coverage plots show marginal embedding
coverage, not calibration of physical onset probabilities. Training curves show
source-mean validation NLL at the recorded epochs. Fixed-lead and repeated-window
recall keep separate panels because their event denominators differ. The summary
does not reinterpret conditional timing MAE as an error over missed events.

## Individual-neighbor spatial attention

The attention variant replaces the precomputed neighbor mean with a learned network
over every selected neighbor at each observed frame. Existing nearest-neighbor
indices and frozen embeddings are reused; a float32 cache supplies periodic center
positions and box lengths. Minimum-image relative vectors use only history frames.
Each neighbor's standardized embedding difference passes through a two-layer SiLU
MLP; radial-basis distance features have a separate MLP. Four attention heads score
individual keys against a query from the target's temporal token, with a learned
radial bias. Two residual attention blocks refine that token. Values are learned
before aggregation. Head-weighted direction vectors produce a head-by-head Gram
matrix, retaining directional arrangement in rotation/reflection-invariant scalars.
Mean/maximum distance remain additional inputs. The temporal GRU and whole-path
mixture likelihood are unchanged. This is a custom experiment, not a replication
of [Point Transformer](https://arxiv.org/abs/2012.09164).

The final spatial block exports per-window attention entropy, the mean of exp(entropy)
over observed frames/heads (effective neighbors), mean maximum attention weight, and
attention-weighted mean neighbor distance in angstrom. Evaluation averages within
sources then across sources. These weights are not causal importance. The configured
attention branch can use CUDA bfloat16 autocast; the temporal model, forecast outputs
and likelihood calculation remain float32.

Explicit microbatches accumulate gradients from slices of the same sampled batch.
Each mean loss is weighted by the slice's sample count divided by the actual batch
count, including a partial final batch. Normalization and central augmentation occur
once on the complete batch; clipping and the optimizer update occur once after
accumulation. Effective batch size and updates per epoch are retained. Floating-point
reduction order can differ. Validation may swap training tensors to host RAM, use
the GPU for validation, then restore training tensors. Windows and selection stay fixed.
