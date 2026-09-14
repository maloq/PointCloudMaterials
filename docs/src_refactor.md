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

## Completion pass (starting at d119fd2)

The user authorized completing the remaining refactor and removing the temporal
sample-cap functionality. Temporal datasets/window sampling remain supported;
`TemporalLAMMPSDataModule` no longer reads or applies the shared `max_samples`
setting. Static/synthetic caps remain unchanged. A positive shared value is now
ignored for temporal data, preserving the complete dense window/center layout.

The correctness commit also prevents source-name suffix collisions, routes
checkpoint evaluation through concrete datamodule selection, and takes lazy
radius metadata directly from the prepared cache producer. The last change avoids
re-estimating an already-recorded radius and preserves delayed raw point loading.
It does not change the separate eager/lazy neighborhood-order protocols. The
focused data/evaluation suite passed 45 tests in 25.39 seconds.

Slurm was inspected read-only before moves. Active jobs 991371_3/4/5 invoke
`src.simulation.campaigns.independent_meam_high_temperature`; 991395 invokes
`scripts/run_lammps_campaign.py elemental`. Those launcher paths and arguments
are retained. The external SBATCH files were inspected but not edited. Other
allocations were interactive/pending bash jobs. No jobs are submitted or changed.

### Final ownership

The completion pass uses separate correctness, ownership and formatting commits.
Numerical loops remain explicit; no universal dataset, trainer, registration
framework or new runtime dependency was introduced.

| Previous owner | Current implementation | Boundary preserved |
| --- | --- | --- |
| `data_utils/data_load.py`, `prepare_data.py` | `data/{static,synthetic,soap,sampling,static_sources}.py` | Static sampling/cache, generated-data consumption and SOAP coordinates are separate. |
| `data_utils/data_modules/registry.py`, ordinary trainer switch | `data/loaders.py`, concrete `data/data_modules/` | One selector, model override first, unchanged concrete lifecycle and custom temporal sampler. |
| `data_utils/temporal_lammps_{dataset,binary}.py` | `data/temporal.py`, `data/trajectories/lammps.py` | Dense windows, atom IDs, timelines, binary validation and split order. |
| `data_utils/shooting{,_binary,_binary_dataset}.py` | `data/shooting.py`, `data/trajectories/shooting.py`, `data/shooting_binary_dataset.py` | Shooting lineage, frame identities and validated outcomes remain distinct. |
| `data_utils/conversion/`, `shooting_text_conversion.py` | `data/conversion/` | Precision checks, checksums, locks, atomic completion and verified deletion gates. |
| `data_utils/{mace_history,mace_relaxed}.py` | `data/{histories,relaxed}.py` | Fixed-neighbor history construction and paired relaxed clouds have multiple consumers. |
| `data_utils/{relaxed_histories,spatiotemporal_views,temporal_binary_context_dataset}.py` | `data/{relaxed_histories,spatiotemporal,temporal_context}.py` | Concrete example definitions and existing sampler/collator behavior. |
| Static context, TDA preparation, temporal inspection | `data/{atomic_context,topology_views,inspect_temporal}.py` | Existing commands and source-specific preparation. |
| `data_utils/{mace_denoising,mace_existing}.py` | `training_methods/mace_denoising/{data,existing_data}.py` | Denoising target preparation belongs with its training protocol. |
| `data_utils/pretrained_mace{,_gpu}.py`, MACE preflight/queue | `training_methods/pretrained_mace/{data,resident,preflight,queue}.py` | Resident iteration remains separate; queue and module commands retain their arguments. |
| Flat MACE/predictive training files | `training_methods/{mace_denoising,mace_temporal,pretrained_mace,predictive_structure}/train.py` | Existing PyTorch loops, attributes, objectives and checkpoint keys. |
| Base SSL, reused VICReg/SwAV losses, supervised metrics, optimizer and MACE mechanics | `training_methods/shared/` | Shared code has actual cross-method consumers and no imports from a concrete SSL method. |
| `data_utils/synthetic/atomistic/`, synthetic visualization | `simulation/atomistic/`, `simulation/visualization.py` | Coherent engine implementations; campaign commands remain in their established locations. |
| Temporal hypothesis data/train/evaluation | `research/temporal_hypotheses_12h/{data,train,evaluate}.py` | The recorded hypothesis protocol stays together. Its reused atomic JSON writer belongs in `experiment_runner/artifacts.py`. |

The two-method Lightning registry now returns a concrete class and its default
post-training analysis policy. Explicit-name/config precedence, `vicreg`, `visreg`,
`contrastive`, `temporal_vicreg` and `temporal_ssl` aliases are characterized. There
is no mutable registration API or import-string specification. Independent MACE,
forecast and temporal-VAMP loops do not use this selector.

Reusable encoders, large analysis features and the spatiotemporal stability-probe
workflow were reviewed without cosmetic splitting. `temporal_vamp` remains its
separate work package; its changes are shared-data imports and the existing
Lightning selector consumer. `data_utils/topology_targets.py` stays in place
because it is a shared, metric-hashed numerical implementation. Moving it adds no
needed ownership benefit here. Its current and historical metric hashes remain
unchanged.

### Compatibility retained and removed

These are explicit imports/commands, with one implementation each:

- `data_utils/data_load.py`: `PointCloudDataset`, `SyntheticPointCloudDataset` and
  `SoapCoordDataset` object paths. `prepare_data.py` retains the pre-refactor
  sampling/read functions used by restored source recipes. Remove these only when
  those saved-object/read APIs and partial source restorations are retired.
- Old temporal/shooting/history/datamodule modules retain their named classes for
  deserialization. `data_utils/data_module.py` retains the established public
  constructors, including `PointCloudDataModule`; it constructs no extra object.
- `data_utils/shooting_binary.ShootingBinaryTrajectory` remains the import used by
  protected forecast data/spatial/attention files and two metric-hashed research
  files. `data_utils/temporal_lammps_binary.TemporalLAMMPSBinaryTrajectory` and
  `data_utils/spatiotemporal_views.{periodic_tree,local_views}` are also used by
  frozen forecast-related producers. Remove these only with a separately reviewed
  source transition; do not edit protected sources to remove an import bridge.
- Historical `BaseSSLModule`, `VICRegLoss`, `EvalBatchStatsBatchNorm1d`, `SwAVLoss`
  and `NormInvariantHead` class paths remain explicit saved-object imports.
  Method-package exports preserve saved-object classes such as `Learner` and
  `Predictor`. Functions are imported from each concrete `train` module; package
  exports do not shadow that submodule with a function named `train`. Class
  forwarders can be removed when retained serialized objects no longer need them.
- The atomistic CLI modules under `data_utils/synthetic/` retain the commands in
  `docs/atomistic_generator.md`, the optimized homogeneous campaign recipes and
  campaign subprocess arguments. The implementations are in `simulation/`.
  Retire a command only after migrating its recorded recipes and submitted jobs.
- `data_utils.synthetic.atomistic.calculator.VerletSkinMACECalculator` is the
  canonical public class identity in existing potential qualification reports.
  The moved class retains that `__module__` identity and the old module explicitly
  imports it. Producer provenance still hashes the actual
  `simulation/atomistic/calculator.py`; a test checks both the public identity and
  the implementing method's source file. This does not certify any source-hash
  transition or rewrite a qualification report.
- `training_methods.pretrained_mace` keeps its `python -m` command via
  `__main__.py`; the historical queue, TDA, paired-relaxation and pretrained-data preparation
  commands have exact `main` forwarders. No external launcher or submitted job was edited.

Removed layers include the `.impl` Lightning wrapper and duplicate data switch,
the dynamic two-method registry/specification, dynamic package attribute exports,
unused function-only forwarding files for MACE logging/objectives, optimizer
helpers, supervised-cache helpers and configuration warnings, and the broken
standalone `data_load.py` demonstration. No data products, checkpoints, source
snapshots, manifests, archived protocols or simulation restarts were removed.

Moving source-hashed preparation code intentionally changes its producer
identity. Existing cache/resume validation must reject uncertified producer
changes; create a new cache/run or use the retained source snapshot. Historical
manifests and exact-continuation allowlists were not rewritten.

### Additional stale references

The maintained SOAP descriptor called a helper deleted in `256f006`. Its
replacement constructs DScribe SOAP directly with the exact historical helper's
arguments, including compression, sparse output and float64 precision. A tiny Al
fixture matches values generated with `256f006^` (`rtol=1e-12`, `atol=1e-14`).
Analysis now explicitly tells `temporal_motif_field` checkpoint users to use the
recorded source snapshot instead of importing an already-deleted model. The
retired training method is not reactivated.

The previously observed eager/lazy tied-neighbor ordering discrepancy remains a
separate scientific question. This refactor does not assert that these two
sampling protocols are interchangeable.

### Completion validation

Formatting is a separate commit (`2c30900`). The formatter ran in an isolated
`/tmp` environment, without changing project/conda dependencies. All **116**
selected files were checked against `cfeaea7`; **105** changed. Their Python ASTs
match after normalizing only import grouping, parentheses around deletion targets
and docstring whitespace. Black's own equivalence checks also ran. The 79-column
formatting target improves the previously compressed training loops; inherited
long literals/docstrings, deliberate imports after backend/path setup and a few
legacy identifier spellings remain. This is not a claim of repository-wide
pycodestyle compliance. The disposable `formatting.json` records that boundary.

Focused checks in `pointnet` include:

| Area | Result |
| --- | --- |
| Sample-cap/selection/source/radius fixes | 45 passed |
| Static/synthetic/SOAP dataset separation | 39 passed |
| Shared SSL/MACE mechanics | 32 passed |
| Method package moves | Denoising 8; temporal 11; pretrained 7; predictive 3 passed |
| Explicit Lightning selection | 6 baseline cases passed; 11 selection/post-analysis checks passed afterward |
| Trajectory/conversion/forecast integration | 70 passed, 2 skipped initially; six metric-hash failures led to retaining original protected research imports |
| Forecast recheck with original metric hashes | 35 passed, 2 skipped |
| Concrete loader ownership | 35 passed |
| MACE/history preparation | 34 passed |
| Temporal hypothesis ownership/shared writer | 18 passed |
| Simulation move | 141 passed initially; a migrated test stub's historical class identity was restored; generator/integrity recheck: 24 passed |
| Temporal example ownership | 34 passed initially; two dispatch stubs were migrated to the actual new owner; selector recheck: 26 passed |
| Forwarder cleanup | 16 passed |
| SOAP/retired motif/analysis fast paths | 12 passed |
| Final context/TDA preparation move | 8 passed; historical TDA command help also succeeded |

These overlapping focused counts are not added together. Integration results
below are the authoritative total. Failed collection attempts from mistyped test
paths or running the `pytest` executable without the repository import root are
not test coverage; successful runs use `python -m pytest`.

A seeded CPU VICReg diagnostic was captured before shared-training moves and
repeated after formatting. It compares the real method's spatiotemporal loss,
all gradients, AdamW state, model state keys/weights and inference output with
`rtol=0, atol=0`. Every comparison passed, including strict loading of the captured
checkpoint and optimizer state. This is a tiny before/after weight-loading and
optimizer-step check, not certification of historical production exact resume.

Reproduce with the retained diagnostic (the saved baseline was produced before
commit `492a003`):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  conda run -n pointnet python \
  output/maintenance/refactor-completion-20260914/technical/ssl_parity.py \
  output/maintenance/refactor-completion-20260914/technical/ssl-baseline.pt compare
```

Protected `training_methods/embedding_forecast` sources are byte-identical to
`d119fd2`. Every current metric implementation hash remains unchanged. Existing
negative continuation tests still reject changed data identity and uncertified
source transitions. Historical exports, potential qualification reports and
resume allowlists were not rewritten.

The full CPU integration command passed **530 tests**, with **8 CUDA-only skips**,
361 warnings and no failures in **370.69 seconds**:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  conda run -n pointnet python -m pytest tests -q -rs
```

The skips are actual forecast device gather/transfer, spatial attention, fused
MACE and BF16 GPU checks. A final method-package export regression was added after
that suite had collected: its targeted method suite passed **26 tests**, including
that new case. Thus the integration total above does not include the additional
export test or count overlapping targeted tests twice. Final historical MACE
preparation forwarders also passed their `--help` checks.

Fourteen command checks succeeded: the conversion dispatcher and temporal
converter, elemental campaign, temporal inspection, active independent-MEAM
campaign module, pretrained-MACE training/queue, predictive-structure training,
old/new TDA commands, atomistic generator/homogeneous campaign, and historical
paired-relaxation/pretrained-data preparation commands. These checks execute
argument parsing/imports only, not simulations or training. Final AST import and
internal module-string audits found no unresolved repository modules.

Validation logs and disposable scripts are retained in
`output/maintenance/refactor-completion-20260914/technical/`.

### Completion real-source comparison

After tests and other checks stopped, the same existing static diagnostic ran
sequentially against a clean detached `d119fd2` checkout and final implementation
`7492429`. Each used three repetitions, fresh temporary caches, the same full
`datasets/Al/inherent_configurations_off/166ps.npy` source, zero workers, seeds,
configuration and batch order. The clean temporary checkout was removed after
validation; the baseline commit remains in Git. No research cache was rebuilt.

All six runs agree exactly on **13,824 samples**, radius
**9.186229173717608**, complete cache fingerprint, full float32 point/coordinate
array hashes and lazy representative array hashes (including a duplicate request).

| Measurement | Baseline median (range) | Final median (range) |
| --- | --- | --- |
| Fresh cache preparation, s | 3.283 (3.235–3.292) | 3.286 (3.237–3.291) |
| Warm iteration, samples/s | 135,532 (134,522–135,816) | 137,575 (134,998–137,820) |
| Lazy metadata construction, ms | 2.675 (2.654–3.207) | 2.596 (2.538–3.039) |
| First four representatives, s | 1.300 (1.298–1.315) | 0.510 (0.500–0.511) |
| Repeated four representatives, ms | 0.673 (0.650–0.811) | 0.672 (0.651–0.747) |
| Whole-process peak RSS, MiB | 1,089.63 | 1,089.92 |

Preparation, warm iteration and memory show no material regression in this
comparison. The first lazy requests improve because the producer-recorded radius
is reused rather than estimated again; their actual point arrays remain exact.
This is a local three-repetition measurement, not a universal throughput claim.
Raw results are `static-baseline.json`, `static-final.json` and
`static-comparison.json` in the completion diagnostic directory.

Checks not performed: real retained-production checkpoint inference/warm-start
and complete historical exact-resume certification; GPU/fused-MACE/BF16 numerical
and memory checks; full real temporal/shooting training or production-potential
simulation campaigns. The CPU tests cover tiny deterministic examples, producer
validation, checkpoint/continuation gates and simulation integrity. None of the
unrun checks is represented as passing or as permission to resume with changed
source hashes. The structural refactor is complete within these explicit
compatibility and scientific boundaries.

### Post-queue cleanup checklist

This is a deferred cleanup note, not a deletion plan approved for execution.
An empty Slurm queue alone does not retire historical commands, saved-object
imports, source-hash contracts or required restart data.

**Remove the two temporary submitted-job launchers once their dependencies end:**

| File to remove | Replacement for future runs | Removal condition |
| --- | --- | --- |
| `scripts/run_lammps_independent_meam_source_campaign.py` | `python scripts/run_lammps_campaign.py independent-meam-source ...` | Both independent-source campaigns and their controller/retry chains have finished; no submitted job, local controller or retained live recipe invokes this file. |
| `scripts/run_lammps_independent_meam_510_520K_sources.py` | `python -m src.simulation.campaigns.independent_meam_high_temperature ...` | Same condition, including remaining high-temperature array tasks and final summarization/publication. |

The September 14 read-only check still found array tasks `991371_3/4/5`
(`al_520K_finish`) and job `991395` (`al_1m_450K`) running. Their inspected
`run.sbatch` files invoke the maintained high-temperature module and elemental
campaign dispatcher respectively, rather than these two temporary wrappers.
The array launcher also performs final summarization and publication after all
six workers finish. Retain those maintained implementation/command paths.
Allocations `992069` and `991772` were running `bash`, and `991900` was pending;
their allocation names do not establish whether child training queues are done.
Recheck the live state when cleanup is actually performed; these IDs are a dated
observation, not a permanent list of all dependencies.

**Review these additional candidates after queues finish; migrate their remaining
consumers before removing them:**

| Candidate | Required follow-up before removal |
| --- | --- |
| CLI files matching `src/data_utils/synthetic/atomistic_*.py` | Move/reuse their argument parsing under simulation ownership and migrate recorded commands. `simulation/atomistic/{homogeneous_campaign,transition_campaign}.py` still construct old module commands; `scripts/run_optimized_al_homogeneous_campaign.sh` also invokes the old homogeneous command. Merely waiting for queues is insufficient. |
| `src/training_methods/pretrained_mace_queue.py` | Migrate recipes to `python -m src.training_methods.pretrained_mace.queue`; verify no queued controller still launches the old module. Keep the actual `pretrained_mace/queue.py` implementation. |
| `src/data_utils/spatiotemporal_tda.py` | Migrate remaining historical preparation recipes to `python -m src.data.topology_views`. |
| `src/data_utils/mace_relaxed.py` | Migrate remaining paired-cache recipes to `python -m src.data.relaxed`. |
| Command forwarding in `src/data_utils/pretrained_mace.py` | Migrate preparation commands to `python -m src.training_methods.pretrained_mace.data`. Its `Quadruplets` saved-object import has a separate lifetime; do not delete the whole file just because the command is unused. |

**Do not remove solely because queues have finished:**

- The maintained campaign modules, `scripts/run_lammps_campaign.py`, training
  method packages, their `__main__.py` commands, or actual queue implementations.
- Dataset/datamodule/history/SSL class import bridges and method-package class
  exports required by retained saved objects. Check actual retained consumers
  before narrowing them, as listed in “Compatibility retained and removed”.
- Forecast reader/sampling bridges, metric-hashed numerical sources, or the
  historical qualified-calculator class identity. These require their own
  compatibility review; queue completion does not authorize a source transition.
- Checkpoints, exact-resume state, immutable source/manifests/configurations,
  paired test data, simulation restarts, or archived scientific records.

Before deleting the eligible wrappers, inspect submitted jobs **and** controller
processes, pending dependencies/retries, local training queues and handoff plans.
Verify final summaries/publication, including stopped failures and restart state,
and publish required SCRATCH artifacts to STORE. Preserve external launcher files
and logs as execution provenance; do not edit them or treat them as repository
cleanup targets. Recheck imports, command strings, tests, configs, current docs
and restored recipes. Update `scripts/README.md` and this checklist in the removal
commit, then run relevant command/import tests. Any later artifact cleanup must
use the documented storage/clean preview and verified archive procedure; this
source cleanup does not authorize deleting outputs.
