# Completed crystallization-transfer comparison

`fits.csv` collates the completed initial and scaling queues. Original metric
values are copied unchanged from each fit's `metrics.json`; `horizons.csv` keeps
all six horizons, classification, conditional timing with misses and sampled
spatial diagnostics. Their definitions are in `crystallization_transfer.md` and
in each original campaign's frozen `tables/METRICS.md`. The initial queue includes
50 trained fits plus one unfitted no-transition baseline; scaling adds 52 fits.
A total of 103 completed evaluations therefore means 102 trained models.

`comparisons.csv` and `paired_metrics.csv` compare event-time negative
log-likelihood. For every trained model, the source mean of the per-window hazard
loss is calculated from saved test logits and event labels. The unweighted mean
of the 30 source means must reproduce the original test NLL. Paired comparisons
require exactly identical saved indices, source IDs, event labels and row metadata.
Difference means A minus B, so negative values favor A. Relative change is 100
multiplied by the mean difference divided by B's mean NLL.

Intervals are percentile 95% intervals from 5,000 paired bootstrap draws of whole
test sources, resampling within temperature strata with seed 20260919. Each draw
uses the same resampled sources for A and B. They measure source uncertainty
conditional on the one training seed and these trained models. They do not
include training-seed or training-subset uncertainty, or correct for multiple
comparisons. They are exploratory comparisons of the declared queue, not new
model selection using the test set. Each model's checkpoint was selected using
selection sources only. The best overall recipe mentioned in a report is ranked
by selection NLL, not by test metrics.

The scaling plot shows test-NLL point estimates for frozen scalar/tensor models.
Radius and training-source sweeps share the three-epoch full-data update budget;
the duration sweep has independently trained 1/3/6-epoch schedules. No curve joins
the initial with-replacement queue to the shuffled-epoch queue. Sparse spatial
context samples, overlapping windows and repeated event monitoring retain the
limitations documented in the original protocol. Input file hashes accompany
this read-only aggregation in `technical/input-hashes.json`.
