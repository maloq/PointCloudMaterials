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
per-source confusion counts across the same two seeds, then bootstrap the same source
indices for candidate and control. This is F1 of seed-averaged confusion counts, not
mean per-seed F1 and not a prediction ensemble. Paired embedding differences first
average each source's MSE across seeds, then bootstrap the paired source differences.
Intervals are 2.5/97.5 percentiles. Seed replication and physical sources are distinct;
these intervals describe source uncertainty conditional on the two fitted seeds.
Bars show mean per-seed scores and dots show individual seeds; tables retain full values.
Previously inspected test sources make this an exploratory experiment.

Spatial fits keep validation embeddings and neighbor pools in host memory, copying
each batch to the GPU. Training batch 8192, data, optimizer updates and validation
selection are unchanged. Nonspatial fits retain both splits on the GPU. This is a
memory-residency difference, not a change to the experimental objective.

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
