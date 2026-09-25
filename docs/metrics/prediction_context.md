# Historical prediction-context audit

One row per `(collection, model, readout)` in the September24 horizon review
and completed larger supervised AP3/AP6 experiment. AP3 and AP6 are copied
unchanged from the cited original CSVs; they are not newly fitted or recalculated.
Consult each original table's frozen METRICS.md for the population, weighting,
selection and uncertainty definitions. These populations are not interchangeable.

The audit separates encoder inputs, prediction-head side inputs, geometry support,
history, velocities, relaxation preprocessing and training-only teachers. Numbers
of input columns are checked against saved probe weights where available. Evidence
paths and checksums are retained in technical/contexts.json. Blank values mean
unmeasured/not applicable, never zero or a verified absence. `history_ps=0` means
a snapshot observation; `predictor_time=true` means explicit simulation-age
covariates, which are distinct from historical frames and forecast horizons.

This export is a dated audit of275 comparisons, not a claim that every archived
encoder study has been checked. No original result or frozen implementation is
modified. Do not infer comparable embedding-only prediction from equal AP labels:
some predictors also receive97 current physical descriptors.
