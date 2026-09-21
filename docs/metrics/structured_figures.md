# Structured forecast visualization and fixed-event offsets

The source is the completed symmetric MACE/GATr eight-fit experiment. Archived
CDFs, path scores, split identities and selected checkpoints remain unchanged.
All aggregate physical/Brier curves use the original predictions and equal-source
weighting as in `trajectory_figures.md` and `structured_context.md`. Standardized
physical128 scores have the same physical targets across backbones; latent scores
and independent UMAP coordinate systems are not cross-backbone quality measures.

## Fixed event/control cohort

An event is a distinct `(source, tracked center, first sustained onset)` from
the held-out sources, not an overlapping forecast window. Keep only events for
which all requested offsets (3, 6, 9, 12, 18, 24, 36, 48 ps) have an archived
at-risk forecast. Round each origin DOWN to the existing 3 ps grid. Actual lead
is in `[nominal, nominal+2.25]` ps, with the same residual at all offsets for
one event. Store requested/actual leads, source, center, onset and origin frames.

Select one control from the same source, using a seed determined by source and
event center: another tracked center with no first onset by the case onset and
an available at-risk prediction at all the same origins. A control can crystallize
later and may be reused for several cases. Selection never uses predictor scores.
The same events, controls and origins are used for every offset and all eight fits.
This conditions on full offset availability and excludes events without controls;
counts and excluded reasons are exported. It is not a deployment-prevalence sample.

Each source receives weight `1/S`, shared equally by its included events. Split
each event's weight equally between case and control. **Matched AP** is the pooled
weighted average precision using `CDF(actual lead)` as score: probability of onset
by the reference event time. Positive prevalence is fixed at 0.5, so chance AP is
0.5. Do not compare its absolute value to the previous natural-population 12 ps AP.
The horizon changes with lead; AP is not an average of undefined positive-only APs.

## Timing and probabilities

For each positive event's forecast, `mass[k] = CDF[k] - CDF[k-1]`, with zero at
time zero and times `(k+1)*0.75 ps`, up to 96 ps. Restricted-mean predicted delay:

`sum_k mass[k] * time[k] + (1 - CDF[127]) * 96`.

This matches the existing left-survival-sum restricted mean. Timing MAE is the
source-balanced mean absolute difference from actual lead over **all matched
events**, including missed alarms. It does not truncate the distribution at the
known event time or discard non-detections. Remaining survival mass is capped at
96 ps, not interpreted as a known physical onset at 96 ps.

Also export conditional-96-ps timing MAE: `sum(mass*time)/CDF[127]`. It can hide
low event probability, so it accompanies restricted timing and mean event mass.
Zero CDF[127] makes this conditional statistic undefined and raises an explicit
error. Export average case/control probability by reference onset and event mass
by 96 ps. There is no threshold fitting on the event-aligned cohort.

Use 1,000 paired source-bootstrap replicates, with the same source draws for every
model/offset/metric. All cases and any reused controls from each source stay together.
Recompute weighted pooled AP within each draw; do not average per-source AP.
Percentile 95% bands capture source uncertainty, not one-seed optimization uncertainty.

## Illustrations and representation maps

Four ordinary trajectory examples are selected using median-error windows within
sources, then source medians, under MACE direct; use identical cases for GATr.
Three event-aligned examples use only source and onset-time quantiles, never model
scores. Fresh 64-draw fixed-seed inference produces illustrative sample paths;
deterministic paths and CDFs are replay-checked against saved evaluation values.
Checkpoint, cached feature, producer and prediction hashes are retained.

UMAP and per-channel scaling fit only on 64 uniformly sampled states per training
source (5,760); transform 64 per test source (1,920). Same states across backbones,
separate maps. All frame selection is outcome-independent. Physical labels are
color only; UMAP distances cannot establish predictive quality. Spatial diagrams
reconstruct actual periodic coordinates and replay the cached 25-query assignment.
Figures are PNG; full definitions and captions are separate from the pictures.
