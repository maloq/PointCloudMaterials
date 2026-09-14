# Source refactor: first review slice

Scope: Phases 0–2 of the supplied plan, in the current namespace. Starting commit
`1afd24c7e8e6aa9ddb12dfaea60fee7117df5e41`, branch
`codex/research-retention-20260913`; the working tree was clean. The supplied review
referenced older commit `e54ec960779d6c18246d39259a5b99878a4f0a80`.
New and edited code follows PEP 8; unrelated source is not reformatted.

## Producer and consumer audit

| Producer / path | Contract and consumers | Disposition |
| --- | --- | --- |
| `data_load.PointCloudDataset` → static datamodule | Float32 `points` per sample `[N, 3]`, optional center `coords`; source-specific labels/radii and cache shard ordering. Seeded random split, validation reused for test, optional sample caps. `train_model`, descriptor baselines, analysis. | Maintained; dataset, sampling, normalization and cache implementation unchanged. |
| `data_load.SyntheticPointCloudDataset` → synthetic datamodule | Generated-data ingestion, sorted environment directories, configured class selection and geometric augmentation; seeded random split and retained training indices. SSL may disable dataset augmentation. | Maintained; preserve the existing config-dependent protocol. |
| `TemporalLAMMPSDumpDataset` → temporal datamodule | `points` `[B,T,N,3]`, local atom IDs, center positions/IDs, integer frame indices/timesteps, anchor and source identity. Batched `__getitems__` returns a dictionary consumed by the identity collator. Random split of window anchors, dense window-major/center-minor layout. | Maintained; reader, radius, sampler, split and collator untouched. |
| `SpatiotemporalViewDataset` → view datamodule | Prepared spatial/temporal views and targets consumed by VICReg; producer-specific preparation and splitting remain in `spatiotemporal_views.py`. | Maintained; selector uses its concrete module. |
| `RelaxedHistoryDataset` → history datamodule | Three float32 views, `[H,80,3]` or anchor `[80,3]`; fixed atom identities within each history, radius normalization, transformed TDA targets, row/context/source/frame/temperature fields. Manifest source splits. | Maintained through `vicreg_mace_relaxed.yaml`; already supported by trainer, newly supported through compatibility constructor. |
| Forecast / shooting | Forecast causal per-frame neighbor reselection, whole-lineage splits, train-only scaling and checkpointed sampler RNG; shooting parent/branch identities and outcome timelines. | Separate scientific protocols; no implementation or hash changes. |

The maintained Hydra contrastive command calls `train_entrypoint.train`, resolves
`VICRegModule` or `TemporalSSLModule`, then calls `trainer.train_model`. The trainer
seeds, constructs data, constructs the model, configures callbacks and calls
Lightning fit/test. Objectives remain in the existing modules; output/provenance
and post-training analysis remain in `train_entrypoint` and `experiment_runner`.
The method-level `data_module_class` override precedes data-kind access. Repository
search found no current declaration, but the public extension behavior is retained.

Audit searched Python, YAML, JSON and Markdown references across source, tests,
configs, scripts and retained experiments, including import strings, `_target_`,
module commands, `.impl` and `data_module_class`. Data kinds normalize whitespace
and case; no additional data-kind alias is implemented. Active recipes include
static, spatiotemporal binary and relaxed histories. Synthetic and temporal LAMMPS
remain supported; this change does not restore retired configs. Training method
aliases (`visreg`, `contrastive`, `temporal_vicreg`, `temporal_ssl`) are unrelated
and unchanged.

Checkpoint evaluation's `build_datamodule` currently treats every non-synthetic
kind as static. Analysis explicitly requests coordinates and has lazy static
loading. These are distinct policies, not duplicate ordinary training dispatch;
neither is changed. The evaluation default merits a separate correctness review.
The temporal sampler rejects `Subset` (including the datamodule's capped subset);
that pre-existing incompatibility is characterized, not silently repaired.

## Ownership and removal decisions

| Old owner / layer | New owner / action | Compatibility |
| --- | --- | --- |
| Ordinary switch in `trainer.py`, switch plus Lightning wrapper in `data_modules/registry.py` | One plain `create_datamodule` in the existing registry module | Return the concrete instance; no forwarding lifecycle or `.impl`. |
| `PointCloudDataModule(cfg)` | Explicit alias to the selection function | Named consumer: descriptor baseline command. Keep until that caller and historical serialized/command references are separately audited. Not a class/subclass API. |
| Trainer and supervised-cache `.impl` reach-through | Read the concrete datamodule | Remove only wrapper-specific branches, keeping existing absent-dataset behavior. |
| Descriptor baseline `.impl` reach-through | Read concrete train/val/test datasets | Existing useful missing-dataset error retained. |
| `data_module.py`, concrete class aliases | Retained | Analysis and checkpoint evaluation use these import paths; broader historical alias retirement is deferred. |
| Static/temporal preparation, simulation engines, method packages | No moves | Phase 3+ is a separate slice; unknown reachability is not evidence of dead code. |

No command path is retired and no cluster job is inspected, submitted or modified.
Submitted-job and external historical deserialization needs remain unresolved for
future retirement work. Data, checkpoints, source snapshots and resume configs are
untouched. This pass makes no new exact-continuation certification. Forecast's
protected sources and gate are unchanged; use retained frozen sources for old runs
whose provenance requires the old checkout.

## Validation evidence

Baseline full suite command (CPU, `pointnet`):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run -n pointnet python -m pytest tests -q
```

Before production edits, the focused suite passed **20 tests** in 16.71 seconds:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run -n pointnet python -m pytest \
  tests/test_datamodule_selection.py \
  tests/test_static_sample_cache_fast_path.py \
  tests/test_temporal_lammps_binary.py -q
```

Tests use tiny temporary cache shards and a six-frame trajectory. Assertions cover
fixed static split indices, cache values, cross-shard/non-monotonic/duplicate reads,
actual collators, repeated fit/validate/test setup, empty datamodule checkpoint
state, incomplete batches and distributed sampler ordering. Construction stubs
cover all five trainer kinds and override precedence; they are not numerical
validation. Existing cache tests also exercise worker loading.

The untouched-source full suite passed **485 tests**, with **8 skips** and
359 warnings, in 387.51 seconds. The separately extended metric-stage and
relaxed-history suite passed **8 tests** in 8.06 seconds, including history
loss/gradient and source-split checks. No baseline failures occurred.
Post-change results follow after integration.

## Consolidation results

Production commit `a1d54ea` removes the second ordinary switch, the forwarding
Lightning datamodule, four forwarding lifecycle methods, and wrapper-specific
reach-through in the trainer, descriptor baseline and supervised metric cache.
`create_datamodule` is exported from both current data-module import paths.
The compatibility constructor remains a direct alias for the descriptor caller.
The production slice changes six source files; no data or method implementation
moves, equations, checkpoint attributes, loader options or metric contracts change.

The combined focused suite passed **35 tests** in 16.84 seconds after the refactor.
This is the three-file characterization command above plus
`tests/test_metric_stage_controls.py` and `tests/test_vicreg_relaxed_histories.py`.
Seven additional selector tests check concrete instance types/attributes, the
compatibility constructor, unknown-kind errors, and override precedence even when
ordinary data configuration is absent.

A disposable Lightning comparison loads the original registry source directly
from the pinned commit and compares it with the new factory. One deterministic
CPU Adam step over the same temporary static-cache fixture produced **bit-exact
model parameters, optimizer state, loss and batch order** (`loss=5.239713668823242`).
This validates Lightning data plumbing; it is not a new certification of old
research checkpoint inference, warm starts or exact training continuation.
Existing history tests exercise the actual VICReg loss and gradients.

### Small loading diagnostic

Seven repetitions use identical two-shard float32 fixtures (seven two-point
clouds), seed 42, batch size 2, no workers, CPU, and the same 300-row validation
request order. Cache opening/setup creates fresh dataset objects; warm iteration
uses the existing cache and actual datamodule loader. Every checksum is exactly
75300.0 before and after. No new cache materialization path is introduced.

| Measurement | Baseline | Refactored |
| --- | --- | --- |
| Cache opening/setup median (range), ms | 1.699 (1.586–3.001) | 1.769 (1.650–2.738) |
| Warm samples/s median (range) | 9,858 (8,814–9,981) | 10,259 (9,289–10,386) |
| Process peak host RSS, MiB | 839.16 | 840.03 |

Ranges overlap; these small measurements do not establish a speedup. RSS includes
framework imports, and the small difference is not evidence of a dataset-memory
regression. This diagnostic measures cache opening, **not cold scientific cache
preparation from real source trajectories**. Full-size cold preparation, GPU
memory/performance and real-checkpoint comparisons were not run; unchanged
producer implementations should still receive that validation before Phase 3.

Raw logs and disposable diagnostic code are retained locally under
`output/maintenance/src-refactor-20260914/technical/`. From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  conda run -n pointnet python \
  output/maintenance/src-refactor-20260914/technical/loader_benchmark.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  conda run -n pointnet python \
  output/maintenance/src-refactor-20260914/technical/lightning_parity.py
```

Run the loading diagnostic on the pinned source for baseline numbers; its fixture
helper is the retained pre-existing static-cache test helper. Diagnostic code is
not a maintained command or a second production implementation.

### Full integration

The full CPU command, with `-rs` added to report skip reasons, passed **510 tests**,
with **8 skips**, 363 warnings and no failures, in 368.09 seconds. Compared with
485 baseline passes, this adds 25 passing characterization/regression cases.
The eight skips are CUDA-only: two forecast device-gather cases, one context-mixture
transfer/training case, two spatial-attention cases, one fused-MACE backend case,
and two MACE BF16 cases. They do not count as GPU numerical validation.
No unavailable external potential caused a baseline or integration failure.

`git diff --check` passes. All added Python lines fit 79 columns; the final two
style-only edits shorten a docstring and wrap a test dictionary entry. The full
suite covers the unchanged metric contracts and forecast continuation rejection
gates. Real retained-checkpoint inference, warm-start and exact-resume comparisons
remain unperformed; the tiny one-step check must not be substituted for them.

Next review slice: extract shared static source resolution and cutoff operations
only after baselining real lazy-analysis/cache behavior and full-size preparation.
Keep simulation/method moves and forecast protected-code changes out of that slice.

## Phase 3a: shared static source operations

Starting commit: `f9d490b1146596dd615cf9a9a59481abfff3a68c` (clean checkout).
The previous 510-pass/8-CUDA-skip integration is the full-suite baseline for this
unchanged source. This slice extracts shared operations only; it does not move
whole datasets, SOAP, simulation engines, training methods or forecast code.

| Existing operation | Consumers | New owner |
| --- | --- | --- |
| `PointCloudDataset._resolve_sources` | Eager static loading and lazy analysis | `src.data.static_sources.resolve_sources` |
| `_load_points` | Static sampling, source cutoff estimation, lazy analysis | `src.data.static_sources.load_points` |
| `PointCloudDataset._resolve_auto_cutoff_config` | Static, synthetic, temporal LAMMPS and temporal-real analysis | `src.data.static_sources.resolve_auto_cutoff_config` |
| `PointCloudDataset._estimate_source_cutoff_radius` | Static, synthetic and lazy analysis | `src.data.static_sources.estimate_source_cutoff_radius` |
| `_ShardValueSequence` | Static-cache and lazy-analysis source/radius metadata | `src.data.static_sources.ShardValueSequence` |

The shared source module owns source descriptors, validated point ingestion,
cutoff settings/calculation and compact source-shard metadata. `prepare_data`
remains authoritative for OFF decoding and existing point-sampling algorithms.
Dataset-specific cache preparation and multiprocessing stay in `data_load.py`;
lazy representative reconstruction keeps its own lifecycle and point selection.
The static cutoff estimator remains non-periodic; temporal trajectory cutoff
calculation continues to use its distinct producer-specific implementation.

Repository Python/config/command/document searches found no external callers of
these old private names beyond the consumers above. All known consumers migrate
together; no private forwarding methods are added. Concrete dataset classes and
the `data_load` module path remain unchanged, including their pickling names.
No command or submitted-job path is retired. Whole-checkout historical source
snapshots and exact-resume requirements remain as described above.

`SoapCoordDataset` is a Parquet feature/coordinate reader, unrelated to atom
sampling. The only discovered caller is the old `data_load.py` embedded demo,
which passes an unsupported `num_coord_dims` argument. Keep it pending a dedicated
historical-consumer review; lack of maintained imports does not prove it disposable.

### Baseline characterization and scientific caveat

Temporary fixtures exercise two sources with duplicate names, distinct radii,
source caps, truncated inference rows, negative/duplicate/non-monotonic indexing,
missing raw input at lazy construction, delayed missing-file errors, cached
representative reuse, invalid cache ordering, NPY/OFF ingestion and a known pooled
cutoff quantile. Source-cutoff estimation must not modify NumPy's global RNG.

A tied-distance lattice exposed an existing eager/lazy point-array difference.
The characterization therefore records the lazy output from the pinned baseline
separately; it does not assume equality to eager sampling or change either
algorithm. Scalar/batched neighbor-query ordering may explain this; its scientific impact
requires separate review and is not resolved by this extraction.

Real-data baseline diagnostics use the complete 166 ps aluminum source, 160 points,
seed 42, automatic cutoff settings from the active static recipe, zero overlap,
two dropped edge layers and no sample cap. Each of three fresh temporary caches
contains 13,824 neighborhoods. This is full preparation of one real frame, not
the complete six-frame active analysis recipe. Cache directories are new; OS page
cache is not flushed. Existing data, sample caches and inference outputs are not
modified. Raw diagnostics live in
`output/maintenance/static-source-refactor-20260914/technical/`.

Before production edits the focused baseline passed **35 tests** in 26.33 seconds:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run -n pointnet python -m pytest \
  tests/test_analysis_fast_paths.py tests/test_static_sample_cache_fast_path.py \
  tests/test_temporal_lammps_binary.py tests/test_atomistic_generator.py -q
```

The first draft assertion that eager and lazy lattice point arrays were identical
failed against baseline; the corrected characterization preserves the separate
lazy result. No existing focused test failed.

### Static extraction results

Characterization commit: `8ab6311`; production extraction: `6b4660a`.
`LazyStaticAnalysisDataset` no longer imports `PointCloudDataset` or private helpers
from `data_load`. All five operations have one implementation in
`src/data/static_sources.py`; their old private definitions are removed. Static,
synthetic and both temporal configuration consumers call the same public functions.
The extracted implementations were compared by Python AST against `f9d490b`,
allowing only public-name substitutions and docstring changes: all five matched.
Numerical operations, error messages, scalar types and required fields are retained.
The compact sequence retains its original `typing.Sequence` base.

The focused command above plus `tests/test_analysis_storage.py` passed **40 tests**
in 36.23 seconds. Static-cache bulk loading now also explicitly exercises spawned
workers, verifying current dataset/sequence serialization and duplicate requests.
This does not certify loading historical pickles containing the old private
sequence class; those must use their original frozen source. Dataset class import
paths themselves are unchanged. No speculative compatibility shim was added.

### Real-data comparison

Every baseline/refactored repetition agrees exactly on:

- 13,824 samples and automatic radius **9.186229173717608**;
- the complete sample-cache fingerprint;
- SHA-256 of the full float32 point array and coordinate array;
- SHA-256 of four lazy representative requests, including a repeated index.

The initial three-repetition comparison ran alongside focused tests. Preparation
medians were 3.418 s before and 3.587 s after; first lazy-load medians were 1.311 s
and 1.384 s. Because this indicated a possible slowdown, the comparison was repeated
sequentially without concurrent tests. The baseline ran from a temporary detached
checkout of `f9d490b`, while both processes used the same working directory, input,
configuration, seeds and batch order. That clean temporary checkout was removed
after the diagnostic completed; the baseline remains available in Git.

Isolated three-repetition comparison, median (range):

| Measurement | Baseline | Refactored |
| --- | --- | --- |
| New sample-cache preparation, s | 3.431 (3.322–3.435) | 3.291 (3.258–3.294) |
| Warm iteration, samples/s | 109,727 (107,792–136,689) | 136,710 (128,464–137,018) |
| Lazy metadata construction, ms | 2.655 (2.652–3.435) | 3.313 (2.709–3.754) |
| First four representatives, s | 1.295 (1.291–1.325) | 1.305 (1.285–1.332) |
| Repeated four representatives, ms | 0.678 (0.658–0.746) | 0.721 (0.707–0.770) |
| Whole-process peak RSS, MiB | 1,089.79 | 1,090.98 |

The initial slowdown did not persist. These small repeated measurements show
system variability, not a demonstrated speedup or a reliable sub-millisecond
regression. The unchanged numerical bodies and full-array identity support the
behavioral comparison; RSS includes framework imports and both dataset paths.
No GPU memory/performance or complete six-frame production run was measured.

The disposable diagnostic is
`output/maintenance/static-source-refactor-20260914/technical/compare_static.py`.
It uses the existing `PointCloudDataset` arguments and lazy-analysis class;
no maintained command or parallel dataset implementation was introduced:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  conda run -n pointnet python \
  output/maintenance/static-source-refactor-20260914/technical/compare_static.py \
  output/maintenance/static-source-refactor-20260914/technical/RESULT.json
```

For baseline reproduction, set `PYTHONPATH` to a separate checkout of `f9d490b`
while retaining the main repository working directory. The JSON files and logs
retain all repetitions and hashes. Temporary caches are created independently;
this command does not rebuild an existing research cache.

### Static slice integration

The full CPU suite (`python -m pytest tests -q -rs` with the environment above)
passed **514 tests**, with **8 CUDA-only skips**, 361 warnings and no failures,
in 387.01 seconds. This adds four passing tests to the 510-pass baseline.
The skipped checks are the same forecast device-gather/transfer, spatial-attention,
fused-MACE and BF16 GPU cases listed in the preceding integration record.
`git diff --check` passes; the new source module fits PEP 8's 79-column limit.
The full suite also validates the unchanged current metric contracts and forecast
continuation gates. No real retained-checkpoint inference/warm-start/exact-resume
certification is claimed by this data-operation extraction.

Remaining ownership work is deliberately separate: dataset-class/SOAP separation,
additional temporal and simulation ownership changes, and any scientific repair
of the eager/lazy discrepancy. No generators, archived protocols, private source
snapshots or existing data products were deleted in this slice.
