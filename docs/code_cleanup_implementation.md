# Shared execution and artifact methods

The pre-cleanup workspace is preserved in commit `ad7b31a2` on
`codex/code-cleanup`. This implementation follows the recurring mechanics in the
[cleanup review](code_cleanup_review_20260929.md); the review remains a broader
backlog for analysis, simulation and historical research modules.

## Implemented methods and consumers

| Shared owner | Methods | Migrated consumers |
| --- | --- | --- |
| `src/experiment_runner/execution.py` | `ExecutionBundle.freeze`, `SlurmQueue.render/submit/submission`, `recorded_stage` | Liquid descriptor, sensitivity-control and rich multimaterial queues; stage receipts in the first two |
| `src/experiment_runner/preparation.py` | `PreparationShard.verified/offset/progress/complete` | Liquid descriptor sources and rich multimaterial preparation shards |
| `src/experiment_runner/array_exports.py` | `ArrayExport.write` with completion checks | Control prediction/state exports and frozen liquid-predictability features |
| `src/experiment_runner/checkpoints.py` | `TrainingState.capture/read/restore/save` | Liquid-predictability and sensitivity-control training |
| `src/experiment_runner/metric_docs.py` | `write_metric_rows`, shared table ownership and hash binding | Four liquid-predictability table writers; existing scalar exporter uses the same binding |
| `src/experiment_runner/wandb_tracking.py` | `start_online_run`, `online_training`, `update_recorded_summary` | Supervised-onset tracking, Lightning initialization, shared MACE logging and frozen shared-pretraining diagnostics |
| `src/experiment_runner/artifacts.py` | `file_hash`, `implementation_hashes` | Streaming artifact checksums, explicit shared-helper provenance |

Scientific callers still declare dependency graphs, resources, row populations,
models, objectives, selectors, normalization and RNG policies. The two training
loops retain their different checkpoint fields: control training records Torch
and CUDA RNG state; liquid-predictability retains its existing sampler-only RNG
record. Their multi-action sampling, optimizer and checkpoint statements are now
split into readable steps. No model-head or objective unification is introduced.

The original command provenance tracker in
`src/experiment_runner/tracking.py` remains available unchanged. W&B lifecycle
methods have a separate module to keep that interface intact.

## Correctness changes

- Checkpoint discovery reads an explicit selected-checkpoint receipt or the
  retained Lightning callback selection. It rejects missing or ambiguous
  selections and unreadable artifacts. A moved historical run resolves the
  recorded filename inside that run only. A valid zero best score remains zero.
- Dependent stages fail when their recorded checkpoint is missing. New Lightning
  runs record their declared monitored selection for subsequent consumers.
- Row CSVs now receive atomic publication and per-table hashes/definition
  bindings. Empty tables require declared columns. Scalar export bytes retain
  their previous format.
- Scientific tracking requires online mode and verifies the requested run ID,
  entity and project. Lightning reuses an existing receipt ID; a new run derives
  its stable ID from its resolved output directory. Its project remains
  `teshbek/PointCloudMaterials`, matching the original initializer even where
  historical configs contain a different `project_name` field.
- Frozen shared-pretraining diagnostics update their recorded parent training
  IDs through the W&B API. They create no new W&B run. Tracking failures propagate
  and leave a local failure receipt.
- Partial array exports remain `.building.npy`; publication requires all rows
  and the expected shapes. Preparation validates recorded identity, checksums
  and resume bounds.

Metric descriptions and `docs/metrics/contracts.json` include the affected
implementation dependencies. Formulas, rows, weights, fitting populations and
scientific selectors remain unchanged. Existing exports keep their frozen
definitions; changed implementation hashes require a new analysis revision.
Existing fits and caches must continue through their recorded source snapshots,
because their strict identity checks intentionally reject a changed producer.
Historical inputs, AP-trained artifacts and discontinued treatments retain their
original records.

The top-level environment instructions now use `pointnet-torch214`. The config
index labels historical AP-selected proposals consistently with the current
research objective.

## Verification

Verification uses `pointnet-torch214` and disposable local diagnostics. The receipt
is at `output/maintenance/code-cleanup/technical/verification.json` (ignored).
No test directory, test suite or test dependencies were added.

- Eighteen CPU/GPU/dependency cases across the three migrated queues generated
  byte-identical Slurm scripts and identical submission arguments to the actual
  nested job methods from `ad7b31a2`. Scheduler calls were substituted locally.
- The 20-row retained visible-prior score table reproduced the baseline CSV
  writer's exact bytes. The scalar writer also matched the original exporter;
  the new per-table checksum matched the exported file.
- Exporting 4,096 retained control prediction/state rows preserved dtype, values
  and order. An incomplete export failed before final filenames were published.
- Retained visible-prior and control-MACE checkpoint payloads survived save/read
  with every field and tensor unchanged. Their saved sampler continuations
  produced identical draws.
- One completed rich multimaterial shard passed its recorded task, identity and
  file checksums. Valid partial progress resumed; out-of-bounds progress failed.
- A retained Lightning run resolved its recorded selected epoch. Local evidence
  confirmed zero-score preservation and rejection of a missing recorded target.
- Local W&B substitutes confirmed stable resumable IDs, API-only diagnostic
  updates, rejection of offline/disabled training and failure exit codes.
- All 123 metric families passed contract validation.
- All 1,045 Python files under `src/` and `scripts/` parsed; 14 changed/helper
  modules imported and all three queue CLIs accepted `--help`. Import checks used
  the queues' existing `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting, required by
  the installed e3nn constants loader.

No scientific training, GPU inference, live Slurm submission or W&B network call
was run for this cleanup. Artifact comparisons validate the shared I/O mechanics;
they do not establish a fresh training trajectory or GPU numerical equivalence.
The large analysis/simulation modules and further legacy call sites remain
documented follow-up work in the review.

## Training and model follow-up

The next refactor starts from `254c3991` and introduces
`src/research/crystal_vector/trunk.py`:

| Component | Responsibility |
| --- | --- |
| `TypedPatchTrunk` | Geometry-only MACE, typed scalar/vector export, chunk padding and normalization |
| `PatchLayout` | Actual exported scalar/vector widths and cache packing/unpacking |
| `SpatialContextTrunk` | Query-relative geometry and vector-message context operations |
| `JointCrystalVector` | Distance-mixture and direction heads |
| `DistanceMACE` | Distance-mixture heads only |
| `ControlMACE` | Exported context state and control/descriptor readout |
| `RichPatchMACE` | Single-patch state and normalized residual descriptor readout |

The three liquid-predictability models now live in that package's `models.py`;
training modules import them. No model constructs task heads to delete them, and
the rich patch model no longer borrows a method from an unrelated predictor.
Distance-mixture construction/calculation is shared without nesting modules or
changing historical joint checkpoint keys. Cached fields use the trunk's layout
instead of repeated `128 + 16*3` arithmetic. New input records take widths and
depth from the executed model.

Dense statements were expanded in six related training modules and their models.
Formatting preserved their pre-format AST, including optimizer/gradient-reduction
order, counters, checkpoint state and selectors. The descriptor MLP file is also
AST-identical to `254c3991`. Distinct sampling, objectives, distributed execution
and RNG protocols stay explicit. Wall-time remains in local records and is
excluded from W&B history, following the current logging policy. Concurrent
rich-multimaterial summary-only baseline logging was preserved.

### Initialization and checkpoint contract

`JointCrystalVector` and `RichPatchMACE` retain their module names, registration
order and seeded initialization. `ControlMACE` retains its checkpoint tensor
names/shapes. Removing discarded head construction changes its fresh head
initialization; it is now recorded as `control_mace_v2`.

`DistanceMACE` is now `distance_mace_v2`: its state excludes
`direction_channels.weight`, `direction_offset.weight` and
`direction_offset.bias`. Their old parameters were frozen and unused by the
distance predictor. This also changes fresh combiner initialization. Historical
distance checkpoints continue with their recorded source; the production loader
does not silently discard unexpected keys. A diagnostic explicitly projected
only these three unused tensors to compare the remaining model. No checkpoint
was rewritten and no live run was resumed or amended by this refactor.

Changed source hashes and architecture identities require fresh run/export
revisions. The unchanged joint/rich state format does not bypass provenance checks
for training continuation. Capacity ablations retain their recorded widths.

### Follow-up verification

Local receipts are under `output/maintenance/training-refactor/technical/`.
No test suite or new automated test files were added.

- A small two-interaction e3nn CPU diagnostic compared the joint model with its
  actual class from `254c3991`: initialization, RNG state, complete patch/context/
  head forward (including tail-chunk padding), and gradients were bitwise equal.
- Retained native control-MACE and joint-distance weights gave bitwise equal
  context/readout outputs and used-parameter gradients with identical cached
  patch fields. Control loaded strictly with no removed tensors; distance removed
  exactly the three documented unused tensors for the diagnostic only.
- The retained RH2 rich-patch checkpoint loaded strictly. Its initialization,
  RNG state and readout outputs matched the previous class exactly.
- Packing/unpacking exported scalar/vector fields retained exact values.
- An eight-update local descriptor-MLP diagnostic reproduced predictions,
  optimizer/model/RNG checkpoint state and the validation-selected epoch exactly.
- All 1,046 Python sources parsed, 12 affected modules imported, four CLI help
  paths loaded and all 123 metric families validated. Across the nine formatted
  files, statement semicolons decreased from 391 to zero.

These checks cover CPU computation and retained weights. Native cuEq GPU
inference, compiled execution and multi-GPU training were not run. In particular,
the removed-head models do not claim equal fresh initialization under equal seeds.

### Architecture retirement proposal

No complete experiment family is deleted in this pass. The following candidates
are grounded in the current policy and inspected producers, rather than scores:

| Candidate | Proposed action | Dependency/preservation requirement |
| --- | --- | --- |
| Physical-reconstruction `StructuralModel` training and its observed/relaxed/teacher variants (`src/research/structural_state/model.py`, `configs/structural_state/`) | Retire the live reconstruction treatments and dependent training recipes; this objective was explicitly discontinued | Keep `GeometryEncoder`, shared error/calibration utilities and physical-information diagnostics; preserve historical source/checkpoints/results |
| AP-tuned robust-onset and spatial-hierarchy experiment paths (`src/research/robust_onset/train.py`, `configs/robust_onset/`, `configs/spatial_hierarchy/`) | Retire ranking replay, AP-selection recipes and their experiment-specific predictor/trainer paths | Robust-onset metrics and geometry are shared; `spatial_hierarchy.Model` inherits `robust_onset.Model`. Resolve these imports together and verify retained frozen sources before deleting classes |
| Temperature/time-conditioned JEPA predictors (`NeighborhoodModel`, v2 `Model`, multihorizon `Model`) | Retire their existing conditional training paths from the active interface; future condition-free protocols would need their own declared implementation | The producer passes lag features and temperature into prediction. Encoder/export classes remain used by Epi, analysis adapters and frozen checkpoints, so the package cannot be deleted wholesale |
| Replacement-only JEPA wrapper constructors (`v2/model.py`, `regularization/model.py`) | Consolidate construction so the final encoder is built once, then remove redundant wrapper implementations | Export normalization/projector treatments are scientific controls. Keep their identities and outputs; preserve historical parameter layouts and initialize future versions explicitly |

Before actual deletion, inventory retained source snapshots and artifact readers,
update configs/command indices and separate shared utilities from the retired
trainer. Git history alone is not a substitute for the producer source recorded
by a frozen run. Current likelihood-trained MACE, matched capacity/initialization
ablations, raw-versus-relaxed comparisons, descriptor/prior controls and separate
history-cadence protocols remain supported scientific distinctions.
