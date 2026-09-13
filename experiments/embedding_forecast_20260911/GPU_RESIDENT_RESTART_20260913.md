# GPU-resident forecast continuation — September 13, 2026

Question: can the enlarged direct and autoregressive full-trajectory forecasts
use the existing H100 allocations efficiently without changing the scientific
protocol or discarding completed epochs?

## Active execution

| Method | Existing allocation / replacement step | Node | Completed epochs retained | Total epochs |
| --- | --- | --- | --- | --- |
| Autoregressive rollout | 990769 / 990769.0 | nodesumo01 | 21 | 32 |
| Direct future trajectory | 990770 / 990770.4 | node53 | 5 | 32 |

Both detached replacements were launched at 10:58 UTC on September 13. They
restore the last complete epoch, including optimizer, scheduler, fitted scaler,
CPU/CUDA and sampler RNG, best score and patience. The unfinished old epoch is
replayed. Exact best/last checkpoint bytes and the previous logs/configuration
are retained in each handoff directory before interruption.

Scientific configuration is the exact September 12 recovery configuration:
125 sources (74 train, 24 validation, 27 test), 1,024 centers/source,
59,181,056 training windows, nine observed embeddings over 6 ps and twelve
predicted embeddings through 9 ps. Batch size remains 32,768, model width 512
with four GRU layers, and history noise/frame dropout remain enabled. Losses,
splits, frame cadence, normalization, learning-rate schedule and 32-epoch budget
are preserved. See [SCALE_RUN.md](SCALE_RUN.md) and
[RECOVERY_20260912.md](RECOVERY_20260912.md) for the scientific/storage protocol.

## Implementation and measurements

`resident.py` loads each stored embedding once into GPU memory, gathers windows
there, and expands the stored float16 values to float32 before normalization.
The existing CPU RandomSampler retains the exact shuffled window order and
checkpointed RNG state. CPU metadata retain exact source, atom and frame IDs.
No repeated full-window CPU-to-GPU transfer or multiprocessing batch prefetch is
needed. Train/validation storage totals 38.329 GiB; final test evaluation releases
those loaders and loads only the test split.

The full-data H100 preflight loaded both splits in 17.1 s, matched an entire
shuffled 32,768-window batch exactly across sources, completed 30 discarded
updates from the saved AR optimizer state, and evaluated 98,304 validation
windows. Warmed median AR update: 0.384 s; peak allocated memory: 77.401 GiB;
peak reserved memory: 82.707 GiB. These are preflight
measurements, not completed optimized epoch times or new scientific scores.
The prior complete epochs took roughly 29 minutes for AR and 25.6 minutes for
direct prediction. Production progress logs and new `train_s`/`validation_s`
fields permit measuring the actual improvement after the first resumed epoch.

At 11:01–11:02 UTC, the production AR fit had advanced through at least 300
updates in epoch 22, with roughly 0.41 s per warmed update and 96% sampled GPU
utilization. Direct had advanced through at least 500 updates in epoch 6, with
roughly 0.25 s per warmed update and 99% utilization. GPU memory use was 85,491
and 76,873 MiB, respectively. The exact progress observation is saved in
`technical/verified_progress.json`; both allocations remain running.

Validation: 40 forecast/layout tests plus three live-subprocess handoff tests
passed. Coverage includes CPU/CUDA FP16/FP32 batch parity, sample identity/RNG,
augmented checkpoint continuation across loaders, explicit implementation hash
transitions and holding/restoring the original launcher. Metric formulas are
unchanged; CPU conversion of example exports supports resident CUDA batches.
New metric tables export reviewed definitions and implementation hashes.

## Reproduction and records

Maintained implementation: `src/training_methods/embedding_forecast/`.
Runtime options: [technical/gpu_resident_runtime.json](technical/gpu_resident_runtime.json).
Reviewed exact source transition:
[technical/gpu_resident_transition_20260913.json](technical/gpu_resident_transition_20260913.json).
Execution artifacts:
[`output/embedding_forecast/gpu-resident-restart-20260913/`](../../output/embedding_forecast/gpu-resident-restart-20260913/).
Its `technical/` contains immutable source plus metric documents, source hashes,
scientific/runtime configurations, `handoff_plan.json`, detached launch commands,
per-method run specs/logs, retained checkpoints and collection submission.

A normal explicit continuation of this implementation uses:

```bash
conda run --no-capture-output -n pointnet python -m src.training_methods.embedding_forecast \
  --config output/embedding_forecast/gpu-resident-restart-20260913/technical/config.json \
  --stage train --variant path_ar_large_aug --seed 20260911 --resume \
  --runtime-config experiments/embedding_forecast_20260911/technical/gpu_resident_runtime.json \
  --resume-transition experiments/embedding_forecast_20260911/technical/gpu_resident_transition_20260913.json
```

Use the frozen-source command in each `run_spec.json` to reproduce this exact
implementation. Do not start another writer while these fits are active.
The actual in-allocation restart uses the maintained module
`python -m src.training_methods.embedding_forecast.handoff --plan PLAN.json --variant NAME`
inside an overlapping `srun --jobid` step. Exact commands are in
`technical/launches.json`. The plan contains top-level scientific `config`,
reviewed `transition`, and `runs[variant]` with `job_id`, `node`, `seed`, `output`,
`artifacts`, `original_record`, and the replacement `spec`.

The immutable old batch launcher is temporarily stopped to keep its allocation
alive while its scientific child is replaced. It is released when the replacement
finishes. The intentionally interrupted old child causes an expected failure
entry in the original batch job; that entry is not the optimized fit's completion
record. Read each replacement `execution/run_record.json`, `handoff.json`, and
the forecast's `status.json` to distinguish success from failure.
Collection job **991021** replaces pending **990771**, uses
`afterany:990769:990770`, and requires both replacement tracked commands to have
succeeded as well as complete, matched forecast artifacts. It fails explicitly
if either optimized run fails or an allocation expires; it cannot report partial
training as a completed comparison. No new GPU allocations were submitted.

The scientific outputs retain their exact-resume directories under
`output/embedding_forecast_20260911/scale/runs/`. Their new progress/runtime files
are stored alongside the legacy checkpoints; final tables/plots use the readable
layout. The added modules and tests are maintained implementation; this report
and runtime/transition JSON files are versioned experiment records. Generated
launchers, logs and preflight results are run artifacts; retained checkpoints
remain protected exact-resume state.
