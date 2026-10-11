# Do simulator responses produce useful clustering geometry?

Use unrelaxed training and unrelaxed held-out inputs only. The user selected
many-to-one K7 cluster agreement as the main structural diagnostic on October3.
A linear or nonlinear phase classifier does not substitute for this measurement.

Questions:

1. Does the current response loss improve native clustering relative to a matched
value-only model? Does training-fitted covariance normalization or clustering
predicted future means improve correspondence?
2. Is90-geometry coverage limiting? Compare90 and720 starting geometries with
matched architecture, optimizer updates and actual batch size.
3. Are short-time changes insufficient? Separate richer changes from absolute
future endpoints plus changes, keeping the same20/100fs paths.
4. Does matching latent pair distances to simulator future-distribution distances
improve clustering geometry while retaining predictive information?

Frozen study: six existing checkpoints (two objectives ×three seeds), three
clustering spaces and three fixed clustering seeds. Fit preprocessing and clusters
on the same74,880 unrelaxed training rows; evaluate every24,960 held-out row.
Expanded study: eight arms ×three seeds,400 updates, batch256, microbatch8.
Legacy90/720 value/response; rich-delta720 response; rich-joint720 value/response;
rich-joint720 response plus noise-corrected future-distribution distance loss.

Many-to-one agreement is descriptive purity at fixed K7; it is accompanied by
occupancy, effective cluster count, adjusted Rand agreement and seed stability.
No cluster/crystal labels enter training, sampling, checkpoint selection or fitted
ensemble weights. Selection remains validation future-feature likelihood. Target
definitions differ between feature ablations, so their NLLs are not directly ranked.
Retain all source roles, no temperature/time covariates, no relaxed inputs.

The substantive downstream research question remains detected-nucleus birth
prediction, not merely general crystallization. These runs address the requested
representation/clustering mechanism; they do not themselves establish birth skill.

[Recipes and reproduction commands](../../docs/simulations/response_geometry_20261003.md) ·
[Clustering metric definitions](../../docs/metrics/response_geometry_clusters.md) ·
[Training metric definitions](../../docs/metrics/response_geometry_training.md).
