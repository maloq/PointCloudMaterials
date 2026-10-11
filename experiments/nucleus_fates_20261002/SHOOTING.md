# Shooting extension: availability of new births and failed embryos

Question: how many additional isolated new establishments and disappearing small
crystal episodes are observable in the existing Al shooting archive?

Run the same full-cell PTM/connectivity/overlap rules on an exact 0.30 ps grid.
Use physical-time persistence thresholds, recording their actual realization on
that grid. Trace parent-present embryos separately; preserve their growth/fate
records without calling their later establishment a new birth. Count each nested
original/continued future once, and report events, futures, parent configurations
and ancestral source runs separately.

Primary outputs: newly observed isolated establishments; isolated ≥8-atom failed
episodes with and without the sustained-size restriction; post-establishment
growth, dissolution, merge and censoring counts. Report by campaign/protocol.
These retrospective, correlated availability counts are not nucleation rates.

Recipe: `configs/analysis/shooting_births_20261002.json`.
[Execution](../../docs/shooting_births.md) ·
[Definitions](../../docs/metrics/shooting_births.md).
