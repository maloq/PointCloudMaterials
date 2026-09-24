# Improvements for convenient, reliable research

[Handbook](README.md) · [Current analysis tools](analysis.md)

The immediate navigation improvement is implemented: one family/evaluation guide,
a curated results table, an offline searchable catalogue and a SQLite evidence
store with original source references. It reads the existing research layout;
old experiments and their metric definitions remain intact. The next improvements
should reduce ambiguity and manual assembly, rather than add another independent
reporting system.

| Priority | Improvement | Why it matters here | Concrete acceptance criterion |
| --- | --- | --- | --- |
| 1 | A machine-readable study/run/evaluation identity | One checkpoint has many names/analyses; different grids and supports currently look comparable | Every export records study ID, run ID, checkpoint hash, feature stage, dataset/root-split/row hashes, input/target domain, physical lags, seed, readout and selection rule |
| 1 | One evaluation receipt and result card after each completed stage | Training completion, export completion and evaluation completion repeatedly diverged | Card links protocol, selected/final/initial checkpoint, frozen definitions, tables, predictions and plots; reports failed/missing stages explicitly; no success inferred from submitted job |
| 1 | Matching and baseline gates before headline generation | Stale checkpoints, step0 hazards, shrinkage and mismatched observations produced misleading apparent gains | Automated checks verify rows/targets, constant/persistence/initial controls, native-head retention, feature spread and readout selected step; baseline selection visibly labelled |
| 1 | Shared reporting primitives, separate scientific protocols | Source-weighted AP, timing and uncertainty are easy to aggregate incorrectly | Tested paired-root bootstrap and AP/tie/null behavior reused where definitions match; each assay retains its own bins, origins, support and censoring |
| 1 | Immutable H200 evaluation bundles | User summaries cannot reproduce paired differences or verify cohorts | Export predictions+labels+row/root IDs, scalers, per-source errors, model/config hashes and metric contract; verify bundle locally without retraining |
| 2 | Enforce metric-contract integrity | Current `check_metric_docs()` loads contracts; it does not compare the declared hashes with source files despite the audit command's wording | A dedicated audit fails on absent families/docs, stale hashes and changed calculations; new exports capture actual implementation; historical contracts remain unchanged |
| 2 | Automatic catalogue refresh after completed reports | Current navigation snapshots become stale until rebuilt | Report stage registers its output and updates the catalogue without scanning training caches or triggering benchmarks; incomplete bundles remain pending |
| 2 | A common compact figure pack | Many attractive UMAPs are hard to compare and omit scientific controls | Every encoder report includes information/neighbor errors, raw spread, declared-lag jumps, physical forecasting, PR+calibration, timing+misses and source/seed effects when applicable |
| 2 | A compare-by-protocol interface | A universal best-score list would reward metric/cohort changes | UI permits ranking only after matching target, population, row IDs, scaling, horizon and readout; external comparisons require a visible qualifier |
| 2 | Checkpoint registry and self-contained inference export | Some loaders still need an older pretrained checkpoint/config just to construct a newer model | Export package includes architecture revision, weights, fitted normalization, species/support/precision settings and a native input/output verification fixture |
| 3 | Fresh confirmation sources and balanced seed replication | Repeated development inspection and single-seed fitting dominate uncertainty | Freeze final sources before model promotion; separate source and seed variation; size evaluation by independent roots and onsets, not atom/window count |
| 3 | Incremental, content-addressed catalogue imports | A broad historical refresh rereads many unchanged small files | Cache imported records by content identity; preserve revisions and aliases; report actual coverage, not just a successful exit |

The metric-contract audit gap above is a recommendation, not a claimed repair in
this documentation task. Changing global historical export behavior deserves a
separate reviewed change. The new catalogue records its own hashes and validates
quoted evidence, but does not retroactively validate or recompute old metrics.

For day-to-day work, make the smallest useful result card first: the question,
what changed, the exact comparison population, one paired effect with uncertainty,
what failed, and a link to the full table. Then add projections and galleries.
A negative result with clear controls is more reusable than a large gallery whose
checkpoint or cohort is uncertain.

Avoid using “latest,” GPU name or folder date as model identity. Maintain a short
human alias for convenience, but resolve it to an immutable checkpoint and export
contract. Separate statistical batch size from computational microbatch size;
record optimizer updates and sampled exposure when evaluating acceleration.
Benchmarking should remain an explicit engineering activity, separate from
scientific training and performance claims.

For the next encoder study, the most useful common reporting panel is: actual
head versus frozen probes; observed versus relaxed target domains; all-state
versus within-liquid error; raw distance neighbors; future physical skill beyond
present baselines; onset ranking **and** probability/timing quality; and source-
paired effects shown separately per training seed. This follows the failures and
positive controls already documented, rather than assuming that more dimensions,
stronger regularization or longer history must improve the state.

## Implemented in the September23 snapshot screen

The [new queue](../encoder_screen.md) pins checkpoint/producer/reference identities,
deduplicates encoder tensors, retains cached embeddings and predictions, separates
plotting from GPU work, and records incomplete/failed stages explicitly. Its CPU
metric benchmark measured 24.8 s → 0.46 s with one thread instead of 16, about 54× for
one 4096-anchor frame metric stage. KMeans assignments matched exactly; maximum
scalar-score drift was 0.00091 and a few logistic classifications changed because
of numerical reductions. This is not a 54× end-to-end claim. Caching the immutable
physical references and the shared current-physics baseline also removes repeated
work. The global metric-contract audit gap above remains unmodified.
