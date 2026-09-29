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
