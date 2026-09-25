# Preparing and running the equivariant-context comparison

## Fixed Al64 repeat and online metrics

Launched detached on 2026-09-25 at approximately 13:37 CEST as Slurm step
**1008517.8** on node59: two tasks, distinct GPUs, one GPU per scientific fit.
[Launch receipt](/work/PERSO/vmorozov/analysis/equivariant_context/al64-b512-20260925/technical/launch.json),
[observed lane](/work/PERSO/vmorozov/analysis/equivariant_context/al64-b512-20260925/technical/lane-hot.json),
[relaxed lane](/work/PERSO/vmorozov/analysis/equivariant_context/al64-b512-20260925/technical/lane-cold.json).
Validation before launch: 81 relevant tests passed; all eight real-data predictor
checks passed at batch 512; both production encoder preflights passed. Tests and
preflights created no online runs. Frozen workers use the captured source snapshot.

The new recipe is `configs/equivariant_context/al64_20260925/comparison.json`.
Use it with the same `prepare`, `check`, `launch` and `collect` commands below.
It pins the sealed Al64 dataset identity and all64 sample population, trains
fresh observed/relaxed encoders, and preserves the previous ten-fit update budget.
See the [protocol](../experiments/equivariant_context_al64_20260925/README.md).
Results are in `${storage:analysis}/equivariant_context/al64-b512-20260925`;
large feature arrays are in `${storage:cache}/equivariant-context/al64-b512-20260925`.
Context normalization streams one source shard at a time into GPU memory, using
float64 statistics fitted only to training sources. Host memory does not grow
with a second complete normalized copy of the cohort.

W&B group: `equivariant-context-al64-20260925`, project
`teshbek/PointCloudMaterials`. Names identify cohort, Observed/Relaxed input and
Shared MACE encoder / predictor variant. All runs retain stable resumable IDs.
Initial encoder runs:
[Observed](https://wandb.ai/teshbek/PointCloudMaterials/runs/dc66d8f38a396d414e2c),
[Relaxed](https://wandb.ai/teshbek/PointCloudMaterials/runs/121474b52cc44dc18af8).

| Fields | Meaning |
| --- | --- |
| `train/event_nll`, `train/gradient_norm`, `train/*learning_rate` | Sampled training updates, recorded every 32 updates |
| `validation/event_nll` | Natural source-weighted validation likelihood, every 256 updates; the sole selector |
| `validation/average_precision_3ps`, `validation/average_precision_6ps` | Raw-risk ranking diagnostics, never selectors |
| `validation/brier_score_*`, `validation/binary_log_loss_*` | Raw probability quality at 3/6 ps |
| `checkpoint/selected_update`, `checkpoint/validation_event_nll` | Selected checkpoint and its likelihood |
| `test/selected_event_nll`, `test/average_precision_*` | Final selected-checkpoint test scores |
| `test/brier_score_raw_*`, `test/brier_score_calibrated_*` | Probability error before/after calibration |
| `test/binary_log_loss_raw_*`, `test/binary_log_loss_calibrated_*` | Binary proper score before/after calibration |
| `test/recall_at_calibration_fpr05_*`, `test/false_positive_rate_*` | Fixed calibration-threshold alarm performance |
| `data/*`, `model/*` | Dataset identity, role counts and parameter counts, stored once |

Final test scores are scalar summary columns, not repeated training curves.
No test, debug or hardware benchmark creates a W&B run. Local JSONL, prediction
files and frozen metric definitions remain the source of truth. Existing finished
fits can expose already computed results in their original W&B runs with:

```bash
python -m src.research.equivariant_context.queue sync-tracking --config configs/equivariant_context/comparison_20260925.json
```

This uses the W&B public API to update summaries and display names, creates no
runs and does not reconstruct missing historical validation curves. Its local
receipt is `technical/wandb-summary-sync.json`.

## Historical Al16 run

Status on 2026-09-25: all ten Al16 fits completed, batch/microbatch 512.
Ran detached as Slurm step **1008517.7**, two one-GPU tasks.
[Launch receipt](/work/PERSO/vmorozov/analysis/equivariant_context/node59-b512-v2-20260925/technical/launch.json),
[observed lane](/work/PERSO/vmorozov/analysis/equivariant_context/node59-b512-v2-20260925/technical/lane-hot.json),
[relaxed lane](/work/PERSO/vmorozov/analysis/equivariant_context/node59-b512-v2-20260925/technical/lane-cold.json).
See the [scientific protocol](../experiments/equivariant_context_20260925/README.md).
All ten existing W&B runs received named final-score summary columns; the
backfill receipt is `technical/wandb-summary-sync.json` beside the launch receipt.
Implementation is in `src/research/equivariant_context/`; no standalone copied
training scripts or new simulations are required.

Activate conda `pointnet-torch214` and run from the repository root:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.equivariant_context.queue prepare --config configs/equivariant_context/comparison_20260925.json
python -m src.research.equivariant_context.queue check --config configs/equivariant_context/comparison_20260925.json
```

`prepare` audits the existing source manifests, 3,006 relaxed cell receipts,
source roles and potential ancestry. It writes inventory and dependency plans.
`check` runs random-weight real-data forward/backward
checks at batch 512. It never fits a scientific model, submits a job or starts
an online W&B run. Encoder width and cuEquivariance stay at their production settings for
the real-data checks.

For the current two-GPU allocation (1008517 on node59):

```bash
python -m src.research.equivariant_context.queue launch --config configs/equivariant_context/comparison_20260925.json
```

This detaches one Slurm step with two tasks and `--gpus-per-task=1`, binding
the tasks to different GPU UUIDs. One task handles each domain sequentially.
Each scientific run uses one GPU; the entire queue stays on node59. It uses the
remaining allocation time and checkpoints unfinished fixed-budget work.
`technical/launch.json` and `lane-hot.json` / `lane-cold.json` record its state.
The two base encoders require their existing supervised preflight receipts.

For a future campaign using fresh Slurm jobs instead of an existing allocation:

```bash
python -m src.research.equivariant_context.queue submit --config configs/equivariant_context/comparison_20260925.json
```

Submission requires matching checks, source/config/data hashes and the prepared
inventory. It freezes code, configs, tests and metric definitions. It submits
two independent GPU preparation jobs, one per coordinate domain. Each trains
its shared local encoder, then extracts frozen scalar/tensor features. Four
GPU predictor jobs depend on successful completion of their corresponding
preparation job. A failed dependency cancels its waiting children.

The current partition is RTX6000PRO on node59, one GPU, eight CPUs and 32 GB host
memory per task. The current allocation supplies two GPUs and 64 GB in total.
For the alternative fresh-job submission route, limits are eight hours per preparation job and two hours per
predictor job: **32 GPU-hours of maximum requested allocation**, not an elapsed
time estimate. The scheduler may run independent fits concurrently. Extraction
has not been timed over the full cohort. Predictors use resident GPU feature
arrays; learned features are cached only after the shared encoder is frozen.
The encoder uses cuEquivariance fused ir_mul operations and the existing compiled
spatial training path. W&B online is mandatory for the ten scientific fits.

All paths resolve through `machine.local.yaml`. Results are under
`${storage:analysis}/equivariant_context/node59-b512-v2-20260925`, features under
`${storage:cache}/equivariant-context/node59-b512-v2-20260925`. The top-level
`technical/plan.json`, `inventory.json`, `checks.json` and later
`submissions.json` provide the preparation/submission receipts. Each predictor
exports readable `tables/metrics.csv` and frozen `tables/METRICS.md`, while
checkpoints, normalization, predictions, logs and W&B receipts are in `technical/`.

After jobs complete, collect paired comparisons:

```bash
python -m src.research.equivariant_context.queue collect --config configs/equivariant_context/comparison_20260925.json
```

Collection reports partial completion honestly and never starts training. Final
per-run metrics are already written by each predictor worker. Collection writes
`tables/comparison.csv` with paired whole-source uncertainty against the invariant
control and `technical/comparison.json` with the full nested records.

For a deadline interruption, inspect `technical/<stage>-state.json` and the Slurm
log. Re-submit that stage's saved script from `technical/slurm/` after the old job
has ended; keep its frozen code/config. Encoder and predictor optimizer states
resume; extraction verifies and skips complete source shards. An interrupted
partial source is recomputed. A fixed update budget must finish before dependent
extraction or final scoring; a partial fit is never silently labeled complete.
Re-submit canceled dependent scripts with an explicit dependency on the new
preparation job. Do not repeat `submit`, which rejects duplicate campaigns.

Fresh source/config changes require a fresh output/cache identity and checks.
Do not edit submitted snapshots or reuse old AP-trained checkpoints. To use an
H200 on another server, transfer the frozen source/config package and registered
raw/relaxed input collections, map paths locally, run preparation/checks there,
then use the worker entry point inside the server's GPU allocation. No special
H200 model or altered batch is required; current Slurm partition names are local.

The first Slurm step (1008517.6) was stopped after concurrent identity-file writes
collided at startup. The fix makes workers read the existing frozen identity
without rewriting it. The v2 queue preserves the shared base-encoder outputs
under `node59-b512-20260925/base-hot` and `base-cold`, including the resumable
W&B run identities; context results/cache use `node59-b512-v2-20260925`.

### Runtime implementation, 2026-09-25

Predictor caches are loaded by variant: scalar-only fits do not upload or gather
unused equivariant fields. The shard checksum still covers the complete artifact.
Hierarchy blocks combine linear scale weights before aggregating fields, avoiding
the large receiver × sender × channel × component intermediate. Patch extraction
pads atom arrays to `80 * chunk`; there are no dummy edges or extra physical atoms.
Each new checkpoint/completion receipt records predictor fields, including the
fixed nominal geometry. The scientific objective and architecture are unchanged.
Use fresh output/cache identities for new code. Running jobs use frozen source
snapshots; do not resume those outputs with changed workspace code.

### Batched feature extraction

New workspace runs resolve an `extraction` configuration with:

```json
{"chunk": 256, "workers": 2, "prefetch": 2, "compile": true}
```

`chunk` defaults to the run's recorded microbatch (512 in the existing explicit
ablation); the other defaults are shown above. Overrides belong in the scientific
run config and are captured in the frozen identity and preparation plan. Use a
fresh output/cache identity for changed code; existing workers retain their
original snapshots.

`features.py` supplies a reusable typed-feature exporter. Its compiled forward
returns scalar z and equivariant center/pooled l=1,2 fields. Degree-4/6 coordinate
bond fields are evaluated over all atoms together and reduced by patch, preserving
center exclusion, both radial envelopes, channel/component order and `n_ref`.
Chunk outputs stay on the GPU until one packed device-to-host copy per frame.
The typed forward is compiled once per exporter, with possible specializations
for short chunks. No frozen features are reused across encoder checkpoints.
Training compilation is unchanged. A trained-checkpoint comparison includes
repeated-reference controls: the original GPU exporter itself has small
run-to-run variation amplified by its fitted normalization. Compare averages
as well as single runs; do not attribute that variation to compilation alone.

Two CPU workers prepare upcoming frames, including ancestry checks, periodic
neighbor selection, graph edges and pinned transfer buffers. The ordered queue
holds at most two pending futures plus the consumed frame. CUDA work remains on
the consumer thread. Pinned transfers are submitted nonblocking on that thread's
stream; a separate transfer stream or GPU graph capture is not used. Source
shards remain the restart unit: an incomplete source is recomputed, while a
completed source is checksum-verified and skipped. Worker errors propagate and
queued work is canceled when the consumer exits.

`technical/extraction-<domain>-timings.jsonl` records per-frame preparation time,
time waiting for the CPU producer, and consumer upload/geometry submission,
encoder submission, bond/packing submission, final download/wait and total time.
Submission timings are CPU wall times, **not isolated GPU kernel times**; the
final download includes waiting for preceding asynchronous GPU work. Preparation
may overlap consumption, so adding preparation and consumption times does not
give elapsed campaign time. Initial kernel setup is included when it occurs.
