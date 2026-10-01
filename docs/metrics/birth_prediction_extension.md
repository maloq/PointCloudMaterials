# Birth-history truncation and frozen-readout cross-validation (v1)

Producers: `src/research/birth_prediction/{extension,analysis,fit,features}.py`.
The original data release and historical predictions remain immutable. The
parent's metric definition is [birth_prediction.md](birth_prediction.md).

## Observation and populations

Use exactly the existing 1,475 event-centered histories: 295 birth cases and
1,180 matched liquid controls, with 20% positive prevalence in every source.
The original 90/15/15/30 source roles, labels, case/control matching and event
weights remain unchanged. No new data, encoder training or feature export is
required. Timestamp, temperature and all other explicit conditions are absent
from model inputs. Export encoder and predictor input records separately.

Remove zero through seven newest observations from the original eight-frame
history, keeping its start fixed. Remaining observations are 8/7/6/5/4/3/2/1;
spans are 5.25/4.50/3.75/3.00/2.25/1.50/0.75/0.00 ps; leads to first observed
PTM crystal are 0.75/1.50/2.25/3.00/3.75/4.50/5.25/6.00 ps. This changes both
history length and lead. Establishment is a different event; these are not
establishment-within-3/6-ps probabilities. At one frame, temporal standard
deviation and last-minus-first are zero; mean repeats that frame's features.
No later observations contribute to any feature.

The primary fixed-test comparison references the existing 44 fits at removals
0–3 and adds 44 fits at removals 4–7. Every model uses the same test rows:
40 positive and 160 negative histories from 15 births in 11 observed sources.
Original per-fit artifacts and exported definitions are preserved.

## Cross-validation

Five outer folds partition the **original training sources only**. Keep all
sources with the same recorded melt lineage together. Balance eligible birth
counts greedily, with seeded random tie order; retain zero-yield sources in the
registered fold design. There are 49 observed training sources, 59 births and
965 histories. Original selection sources choose C/tree iteration count, and
original calibration sources fit the probability map/false-alarm threshold,
independently of every outer evaluation fold. They remain fixed across folds.
Original test rows are excluded from CV scoring and retained CV prediction
files. All outer-fold training, selection, calibration and evaluation source
sets must be disjoint. Every eligible training row receives exactly one
out-of-fold prediction per treatment. There is one fitting seed and one fixed
fold-assignment seed; no fitted prediction ensemble is formed.

The native encoders remain frozen and were pretrained on the original training
sources, including outer evaluation sources. This is **readout cross-validation
conditional on the retained representations**, not an independent end-to-end
encoder-generalization test. Descriptor controls have no learned feature
pretraining. The separate fixed original test remains the primary comparison
for transfer to sources unseen by encoder fitting. CV and fixed-test populations
are reported and plotted separately; neither replaces the other.

Each fit uses the parent's binary predictive likelihood, train-only linear
standardization, selection NLL and separate monotone calibration. AP never
selects a model, checkpoint, regularizer, threshold or history endpoint.

## Metrics and uncertainty

Event-balanced row weight is unchanged: inverse rows associated with each
source/event. Normalize these weights within the measured population. Report
raw and calibrated binary NLL and Brier, sklearn-equivalent weighted AP/AUROC,
and calibrated recall and achieved false-positive rate at the threshold fixed
on calibration negatives. CV uses each row's own outer-model threshold.

NLL clips probabilities to [1e-7, 1-1e-7], then averages
`-y*log(p)-(1-y)*log(1-p)`. Brier averages `(p-y)^2`. AP sorts predictions in
descending order, aggregates tied scores and sums precision times each recall
increment. AUROC trapezoidally integrates TPR against FPR, including origin.
These curve formulas equal the parent's sklearn definitions, including ties
and omitted zero-weight rows. Batch source resamples reuse one sorted curve;
this changes computational cost, not the estimator.

95% intervals are 2.5/97.5 percentiles of 2,000 whole-source resamples, with the
same source draws shared across all models/endpoints within a population.
Drop a draw only when it has no positive or no negative weight and report the
valid draw count. Source-paired NLL contrasts compare each treatment with the
prior at that endpoint and with its own full-history fit. Differences below
zero favor the listed treatment. Intervals condition on fitted readouts, frozen
features, selection/calibration sources, one fitting seed and one fold design;
they do not refit within the bootstrap, quantify training-seed uncertainty or
correct for multiple comparisons. Bootstrap of pooled OOF predictions is a
conditional, descriptive source-uncertainty estimate, not a full sampling
distribution of the cross-validation training algorithm.

OOF scores pool all eligible training rows with their original event weights;
do not average fold scores with unequal sample/event masses. Also retain every
fold's score and its sample standard deviation across five folds. This describes
variation across refits and evaluation populations; overlapping fits are not
independent replicates, so **do not use SD/sqrt(5) as a confidence interval**.

Reliability tables use ten fixed probability bins `[0,.1), ... ,[.9,1]`, weighted
mean prediction and observed positive fraction. Empty bins are absent. Bin-rate
intervals use the same source resamples, excluding draws without bin support;
record rows, sources and supported draws. These are classification probabilities
for the deliberately enriched case/control population, not absolute nucleation
risk. Every numerical CSV freezes definitions and implementation hashes.
