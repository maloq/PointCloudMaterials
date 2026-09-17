# H100 + H200: one-seed, 16-hour predictability queue

## H200 reported export resumption, 17 September

User-supplied remote update, received after the consolidated report: all H200
training is complete. Exports had stopped at an open-file limit; the remote
worker raised the limit, verified that all 150 sources open, and resumed exports
and evaluation detached without retraining. The reported remaining estimate was
2–3 hours at that update. This is a remote status report, not a locally verified
process status or a new completion-time estimate.

Validation values are recorded in the
[scientific report addendum](../output/local_predictability/research-summary-20260917/RESULTS.md#6-latest-h200-physical-study-training-complete-validation-nearly-tied).
Final paired test results remain pending. No local job was restarted for this
documentation update.

## Detached continuation and diagnostics, 17 September

The original H100 process stopped at snapshot evaluation with `EMFILE`: 150
sources retain seven memory-mapped arrays each, exceeding the inherited 1,024
open-file soft limit. All parent and snapshot training checkpoints were saved.
The new worker raises its own soft limit to 8,192 (hard limit 131,072), without
changing loader, training or checkpoint identities. An actual load of all 150
sources and CUDA checkpoint tests passed. The old failed execution is preserved.

The detached queue now assigns:

1. H100: finish snapshot evaluation, then train/evaluate history12.
2. RTX6000: train/evaluate repeat12 from the same parent (4,096 updates).
3. RTX6000, after its continuation: raw-atom current-state diagnostic (8,192 updates).
4. RTX6000, after the raw diagnostic: frozen linear/MLP onset readouts for the
   completed snapshot and repeat12 encoders (four fits). Only the history12
   readouts wait for the H100 continuation (two fits).

Core stage outputs stay in `output/local_predictability/h100-native-onset-20260917`;
worker records are `technical/execution-h100-split/execution/run_record.json` and
`technical/execution-rtx6000-split/execution/run_record.json`.
New outputs are `output/local_predictability/rtx-raw-observability-20260917` and
`output/local_predictability/rtx-native-readouts-20260917`. Their tracked execution
specs declare dependencies; failed prerequisites block dependent experiments.
All runs use seed 20260919. RTX training stops by 18:02:01 UTC, with the final hour
reserved before allocation 996727 expires. Scientific interpretation is deferred.

The remaining RTX raw-state fit and frozen-state extraction use cuEquivariance.
Their refreshed tracked executions are `technical/execution-cueq/`; the earlier
dependency waiters were explicitly stopped before their commands began. The raw
fit maps the seeded initial weights before creating a fresh AdamW optimizer.
Each frozen encoder is checked against the original checkpoint on retained inputs
before CuEq extraction. Original checkpoints and optimizer states stay intact;
the active H100/RTX core fits finish on their existing backend.

After raw-state training/export completed, the idle whole-queue readout waiter
was replaced by `technical/execution-ready-first/`. It starts the ready snapshot
and repeat12 variants immediately; `variant_dependencies.history12` retains the
H100 prerequisite. Tracked failure/deadline checks still apply at that boundary.
The six-fit budget, output locations, checkpoint identities and seed are unchanged.

Queued diagnostics also use three CPU input workers and one-batch lookahead on a
separate CUDA copy stream. Each CPU worker owns its trajectory handles and a share
of the configured frame-cache budget; one dispatcher owns the bounded GPU cache.
Pinned transfers use events and allocator stream tracking. Sampling lookahead
restores sampler state immediately, so checkpoints record only trained draws.
Inputs, source weights, targets, batch eight and update budgets are unchanged.
Training logs expose cumulative `input_wait_seconds` for host waits on prepared
batches; this excludes waits enqueued on the GPU stream. This addresses cold input
stalls without silently increasing the statistical batch or adding new fits.
An actual CuEq snapshot-onset workload with identical 20 training batches took
12.56 s sequentially and 8.87 s with parallel lookahead (1.42x); the maximum final
parameter difference was 2.4e-7. Serial ran first on a shared GPU, so this is a
provisional pipeline measurement. The validation artifacts are under
`output/predictive_memory/cueq-validation-20260917/technical/`. Resume each active
run from its tracked source/config snapshot to preserve its exact implementation.

The [scientific definitions](../experiments/local_predictability_20260917/OBSERVABILITY.md)
distinguish the all-state raw task from onset prediction and frozen-state decoding.

Additional RTX6000 execution: allocation 996727 on node58 was idle after the
descriptor queue completed. The packet observability queue uses
`configs/local_predictability/rtx6000_observability.json`, seed 20260919, and
output `output/local_predictability/rtx-observability-20260917`. It trains until
18:02:01 UTC at the latest, reserving an hour before allocation expiry. The
maintained tracking wrapper retains logs, source/config snapshots and completion
status; scientific analysis is deferred at the user's request. See
[protocol](../experiments/local_predictability_20260917/OBSERVABILITY.md).

**Planning only; no new experiments have been launched by preparing this handoff.**
Use seed 20260919 for every stochastic fit. The budget is a shared 16-hour wall-clock
window with two concurrently available devices, up to 32 GPU-hours. First complete
report target: hour 12; finish sooner when possible. A shorter remaining allocation
always wins. The user superseded the initial 72-hour H200-only idea and multi-seed
comparisons. Old stopped training/simulation jobs stay stopped.

- [Scientific protocol](../experiments/local_predictability_20260917/README.md)
- [Configuration](../configs/local_predictability/two_gpu_16h.json)
- [Materialized planning queue](handoffs/local_predictability_16h/technical/queue.json)
- [Portable inherited source manifest](handoffs/local_predictability_16h/technical/source_manifest.json)
- Receiving tasks: [local H100](handoffs/local_predictability_16h/TASK_H100.md),
  [remote H200](handoffs/local_predictability_16h/TASK_H200.md).

## Device assignments and deadlines

| Elapsed time | Local H100 / local CPUs | Remote H200 / remote CPUs |
| --- | --- | --- |
| 0–2 h | Reaggregate old positive control; source/fold audit; freeze center IDs | Audit uploaded sources; implement/test batched simple heads and nested history gates |
| 2–4 h | Core-center labels, coverage, descriptors and observability audit | Window loader, bounded input/graph cache, small-set fit and workload timing |
| 4–6 h | Descriptor hazard models and physical ridge references | Start physical parent + triplet after gates and shared update budget freeze |
| 6–10 h | Supervised onset parent + triplet | Finish physical triplet, frozen-state readouts and interventions |
| 10–12 h | Paired report, calibration, source intervals | Return artifacts and code; publish completed comparison immediately |
| 12–15 h | Core preparation/evaluation contingency only | Core preparation/evaluation contingency only |
| 15–16 h | Checkpoint, export and preserve artifacts | Checkpoint, export and preserve artifacts |

These are planning envelopes, **not measured completion estimates**. Assay
construction and broader batching are real implementation work. If the core data
release or verified trainer is not ready by hour 4, report the blockage and revise
the feasible remainder before launching a matched group. Publish the audited
baseline/coverage result even if no complete native group fits the remaining time.
Do not reserve GPUs idly for deferred optional work.

At planning inspection the local allocation was job 995957 on node53, H100 NVL
95,830 MiB, with an end of 2026-09-18 03:59:30 UTC. This is a historical observation,
not a launcher identity: query the current controller on execution and stop at
`min(shared start + 16 h, actual allocation end)`. Reserve the last hour. No remote
host/allocation identity was supplied, so the remote task must obtain its own real
device and deadline. Do not fake Slurm variables on an unmanaged H200 server.

## What is runnable now, and what needs implementation

The only new command implemented for this plan validates the specification and
writes job identities. It does not allocate hardware, train, evaluate or detach:

```bash
conda run -n pointnet python -m src.research.local_predictability.plan \
  --config configs/local_predictability/two_gpu_16h.json \
  --output /tmp/local-predictability-queue.json
```

It refuses to overwrite an existing output. The JSON has **29 model job entries**:
16 descriptor hazard configurations, five physical ridge configurations, two
snapshot parents and six native continuations. Linear/ridge jobs contain small
validation-selected regularization grids; this is not a claim of exactly 29
individual optimizer invocations. Assay audits and readout diagnostics have their
own preparation/evaluation budget. Auxiliary center/shell jobs require validated
features. Optional extensions are disabled.

Reuse existing producers rather than copying old runners:

- `src/data/predictive_memory/{observations,targets,windows}.py`: audited atomic
  observations and 128-component physical targets. Add explicit broad-window
  preparation; the old cache's one center / three anchors is insufficient.
- `src/training_methods/predictive_memory/{model,train,diagnose}.py`: retain native
  tensor computation and diagnostic ideas, but add the simpler heads, full
  temporal-residual gates and batched training. The existing trainer requires
  batch size one; editing only a config does not implement this queue.
- `src/research/forecast_crystallization/local_{data,predict,analyze,metrics}.py`:
  reuse the producer/label logic, with a separately versioned corrected assay.
- `src/analysis/liquid_structure.py`: validate center-order feature semantics.
  Older packed causal code is a reference, not a drop-in cache/model adapter.

H100 owns cohort/assay/descriptors/metrics. H200 owns native loader/model/training
changes. Exchange patches and exact hashes before H100 native fitting. Freeze a
source snapshot per worker; never change implementation underneath live fits.
Use tests for source exclusion, periodic IDs, causal input, confirmation/censoring,
small-set fit, batching output/gradient parity and nested gates. Register the new
metric family and calculations before exporting metric CSVs. Keep the
family-scoped metric-doc fix: a hardware benchmark definition must not block an
unrelated training export. Do not run the general hardware benchmark as part of
training; use only short actual-workload timing needed to choose a feasible K.

Existing legacy reaggregation is available through
`python -m src.research.forecast_crystallization.local_analyze --config CONFIG`.
Copy retained technical inputs into a **new** output, update the copied exact
`experiments/forecast_crystallization_20260913/technical/local_config.json`, and
keep its old folds/weights. Saved prediction/observation/score/label arrays suffice
for this check. Full re-inference instead needs the old encoder, forecasters and
embedding cache; audit that dependency before promising it on H200.

## Use VRAM for observations, not another width sweep

Keep model width 16. Retain compact immutable targets/descriptor tables on device
where useful; mmap raw arrays and build frame graphs on demand into a bounded
cache shared by overlapping windows. The full 16-center descriptor timeline is
about 0.92 GiB in float32 before metadata, much smaller than precomputed atomic
history graphs. Do not materialize a graph for every overlapping history window
or cache learned intermediate atom features during encoder training.

Use packed independent windows and gradient accumulation to preserve effective
batch eight. Select physical microbatch separately on each GPU using a tiny
training-only workload, preserving identical optimizer updates and frame cadence.
Cap residency at roughly 80% of available VRAM and measure peak backward memory.
Validation/inference can use larger batches. Start FP32; any faster numerical path
must pass output/gradient checks before the implementation freeze. Existing
H100/H200 timings from other cache contracts are not throughput estimates here.

Choose a shared K from 1024/2048/4096 after measured p90 training, validation and
export costs on both workers. Include all four native stages per objective and
final scoring in the reservation. If even the smallest complete group cannot fit,
publish fitting diagnostics rather than launch a knowingly incomplete comparison.
Use `experiment_registry.py run --spec` for execution records once real commands
exist, and the existing allocation controller only with actual explicit modules,
arguments, identities and dependencies. The planning JSON is not that controller's
schema. Detach only a validated, deadline-aware executable queue at a later launch.

## Transfer and exchange

The H200 inventory already reports 171.78 GiB of Al histories. **Do not send this
collection again.** Resolve the 150 portable paths against that server's local
catalog and verify manifests, arrays, identities, velocities and timelines. The
three required dataset IDs are:

- `al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`
- `al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`
- `al_520K_remaining24_20260913T204114Z`

The handoff manifest records inherited producer hashes; creating it does not
verify remote files. Transfer only explicitly missing sources after the audit;
never silently fall back to 125 sources. The older causal/memory caches provide
reference artifacts, not the new expanded observation sample. Static structures,
synthetic clouds, Ti data and hardware benchmark results are not core dependencies.

Send the same repository revision to both machines, including any uncommitted
patches needed for this task. Required new files are `configs/local_predictability/`,
`src/research/local_predictability/`, `experiments/local_predictability_20260917/`,
this document and `docs/handoffs/local_predictability_16h/`. Preserve receiving
`machine.local.yaml` and resolve dataset locations there. Include the current
`src/experiment_runner/metric_docs.py` family-scoping fix and its regression tests
if that change is not yet on the receiving branch. Read the source/target revision
before applying a patch; do not overwrite unrelated receiving-server work.

At the data freeze H100 sends a compact release: lineage/fold/center/anchor IDs,
per-row eligibility/censor masks, physical target arrays and train-only scales,
assay labels for **external evaluation only**, provenance hashes, and metric
contracts. H200 reconstructs raw histories from its existing trajectories and
verifies matching targets. Its physical trainer must not consume assay labels.
The H200 code patch must arrive before local native fits; sharing code can occur
before completion of the data release.

Return selected checkpoints and exact-resume states, predictions/row IDs, source
metrics, configuration/implementation hashes, runtime records and any incomplete
job reasons. Use fresh `output/local_predictability/RUN/{tables,plots,technical}`
paths. Every metric CSV needs `tables/METRICS.md` via the existing exporter.
Neither an output directory nor an interrupted checkpoint counts as a completed
scientific comparison. Stop after the core report; no automatic extra seeds,
H48 fits or new simulations.
