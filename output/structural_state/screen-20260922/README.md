# Fixed structural targets after BCR

Queue collection: **complete**; 4/4 encoder/evaluation pipelines complete.

[Results and interpretation, 23 September 2026](RESULTS.md): no consistent
predictive improvement; a small relaxed-teacher effect and a distance-arm
scale/training-head failure require separate interpretation.

One seed; 25 fitting, five tuning and 15 reused development roots. Native MACE with cuEquivariance, fixed 2,048-update exports and matched full-radius observations. No final-test claim.

- [Physical readouts](tables/physical.csv): training and withheld target families, including future physical order.
- [Embedding neighbors](tables/neighbors.csv): current-temperature/PTM-matched neighbors in original feature space.
- [Onset predictions](tables/onset.csv): natural at-risk population, NLL, AP, Brier, false alarms and timing with misses.
- [Paired comparisons](tables/comparisons.csv): negative changes mean smaller error; intervals resample whole sources within temperature.
- [Metric definitions](tables/METRICS.md).

Angular and l6 moments, cached current-order measurements, future order, and onset labels were withheld from encoder losses. Current coarse PTM labels only define matched sampling strata. The sparse four-frame cohort is a mechanism screen; event counts and intervals must accompany interpretation.
