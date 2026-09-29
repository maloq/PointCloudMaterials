# Six exact 0.10-ps MD observations: external ancestry holdouts

CD-MACE128-D6-010ps uses offsets -0.5/-0.4/-0.3/-0.2/-0.1/0 ps at every source.
Only raw saved observations are used, with exact atom identities and constant
physical spacing. No interpolation, repeated-frame padding, time/cadence input,
material ID, species, velocities, temperature or surrounding-patch embedding.
Geometry has the established fixed material length normalization.

The existing 27 external trajectories form five ancestry groups. Roles are
predeclared by whole group, without examining labels or scores:

- Train: independent million-atom Al melt/source family, Mg archive, Ti archive;
  15 trajectories and 3,374,496 windows (Al 2,290,000; Mg 377,496; Ti 707,000).
- Selection: Al archive family, six trajectories and 377,496 windows.
- Test: Ta archive family, six trajectories and 3,061,560 windows.

The fixed native Al64 split is not changed and is not this temporal benchmark.
Both molten/source Al continuations stay together; all related branches of each
archive stay together. Validation and test each contain just ONE ancestry family.
Ta is also an unseen material. These scores describe a limited transfer assay,
not a many-independent-source generalization estimate or a matched comparison
with the nominal 0.75-ps Al test. No confidence intervals are claimed.

Initialize the encoder from the recorded native-Al-only scratch O-NLL encoder;
it has not trained on external geometry. Initialize temporal/distance heads
fresh. The old multimaterial distance checkpoint is forbidden for this assay
because it already saw the held-out external families. This initialization is
supervised on native Al onset, not self-supervised and not physical reconstruction.

The six-frame architecture, accumulation, proper likelihood, optimizer and
12-epoch selection rule follow `distance_encoder_dense_history`. Train loss has
equal material mass. Selection/test metrics give equal trajectory mass, not
independent-ancestry uncertainty. Both the real-history and repeated-current
control are trained end to end with identical populations and seed 20260926.
Distance units are Al-equivalent Angstrom, not raw material Angstrom.

`validation.csv` reports the proper combined objective (censored distance NLL
plus twice the fixed-weight CDF Bernoulli losses at 8/12/20/32), distance NLL,
early log loss, capped-mean RMSE and Brier at 20/32. Checkpoint selection begins
at epoch 12 and minimizes the combined objective. AP is never optimized.

After fitting, `distance.csv` reports selection and test rows/source counts,
NLL censored at 64, censored fraction, capped-mean RMSE, capped-median MAE and CDF
Brier scores at 4/8/12/20/32. Equal source weights normalize to one. Infinite
observed distance is right-censored and capped at 64 for point errors.
`confidence-reliability.csv` uses strict predicted-probability thresholds
>.5/.75/.95 for each distance radius. Coverage is the selected source-balanced
mass; mean probability and observed precision renormalize that mass. Empty
subsets are blank. These are single-point confidence checks, not spatial warning
distances. There is no external spatial-scan/visibility assay in this release.
Raw predictions retain source/material indices with explicit lookup records.
Evaluation updates the existing online training run; it creates no W&B run.

Implementation revision (2026-09-27): the shared trainer also supports full-history checkpoint initialization and explicit material subsets for the separate material-adaptation protocol. Historical calculations and frozen exports are unchanged.
