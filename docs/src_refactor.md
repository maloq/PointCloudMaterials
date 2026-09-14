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
