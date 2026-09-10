# MACE throughput improvements — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Research question: how much faster can the same target-complete MACE training
objective run after increasing graph chunks, keeping prepared data on GPU,
reusing geometry between gradient-cache passes, and compiling/testing faster
learned matrix operations?

The user authorized improvements 1–4 from the
[performance proposal](../mace_target_complete_20260908/PERFORMANCE_PLAN.md),
a training restart and a measured speed comparison. This experiment changes
execution, not the scientific objective or encoder architecture.

## Configuration and reproduction

- [Benchmark configuration](benchmark.json): frozen pre-change source snapshots,
  matched initial weights, batches, augmentations, effective batch 1,536,
  learning rates, warmup, timed updates and numerical acceptance thresholds.
- [Training](training.json), [queue](plan.json), [static analysis](static.yaml),
  [tracked detached launcher](run_spec.json).
- [Longer confirmation benchmark](confirmation.json): repeats the baseline,
  geometry reuse and full-FP32 compilation for 40 measured updates per variant.
- Output: [`output/mace_throughput_20260908`](../../output/mace_throughput_20260908/).
- Optimizer and compilation caches: `/tmp/vmorozov_mace_throughput_20260908`.
  Prepared source data are read-only. GPU packing adds no persistent data cache.

From the repository root with the `pointnet` environment:

```bash
python -m src.analysis.mace_performance --config experiments/mace_throughput_20260908/benchmark.json
python -m src.analysis.mace_performance --config experiments/mace_throughput_20260908/confirmation.json
python scripts/experiment_registry.py run --spec experiments/mace_throughput_20260908/run_spec.json
```

The benchmark updates only this experiment's restart configuration with the
fastest accepted settings. It records rejected variants and their tracebacks;
an error in the baseline or absence of a passing GPU-data/geometry configuration
stops the workflow. The training queue independently runs the existing full
gradient preflight, then training, frozen probes and encoder-only static analysis.
Use a new tracked output for another launch.

## Implementation

`src/data_utils/pretrained_mace_gpu.py` packs the original 80-atom float16
coordinates, fixed PCA/scaled float32 targets, conditions and material indices
on CUDA. Batch selection preserves the original shard/row identities and sampling
order. Coordinates convert to float32 after selection. Validation targets use the
same fixed training scaler. GPU memory use is recorded explicitly.

The encoder separates `build_geometry` from `forward_from_geometry`. Geometry
contains edge indices, full-precision support weights, radial basis, spherical
harmonics and central-readout indices. It is cached for one optimizer update and
reused when replaying dL/dz. Original and perturbed views have separate geometry.
The checkpoint's radial embedding is verified to have no trainable parameters.
Learned node/message features are never reused across optimizer updates.

`src/training_methods/mace_performance.py` preserves chunk order across original
and augmented views, including partial chunks at their boundary. Both ordinary
backpropagation and per-task gradient diagnostics use this same replay mapping.
The exact whole-batch VICReg objective is retained.

Faster matrix arithmetic is limited to the learned encoder computation and its
backward pass. Geometry and loss/covariance calculations use highest FP32 matrix
precision. The previous setting is restored even on exceptions. Selective
`torch.compile` targets the two radial-weight MLPs with `fullgraph=True`, dynamic
edge counts and CUDA graph capture disabled. Compilation failures are reported;
there is no suppressed compiler error or silent eager fallback in those regions.
Compiled modules keep the same state-dictionary keys and parameter count.
The initial compilation trial required disabling AOTAutograd buffer donation
to accept `retain_graph`. The independent restart preflight then found that the
installed compiled radial MLP backward still produced incorrect parameter
gradients on repeated calls. Per-task diagnostics/PCGrad therefore explicitly
execute their encoder replay under `torch.compiler.set_stance('force_eager')`.
Ordinary training steps remain compiled. This is an explicit execution policy,
not an exception-triggered fallback. Both ordinary and per-task gradients are
now compared against the original reference before benchmark timing.
Independent benchmark variants reset Dynamo's in-memory guard cache so one
variant cannot exhaust another's recompilation budget. Compiler failures remain
loud; resetting benchmark state is not an eager fallback.
Encoder export records the selected execution settings for static inference.

## Restart and comparison protocol

The preceding detached run was explicitly stopped. Its completed first-epoch
weights at update 704 were preserved as `interrupted_best.pt` and
`interrupted_last.pt` under the preceding run's repository output. Its original
optimizer checkpoints and W&B history remain available.

The new run restarts from the same original all-objective warm start and 0.1 I
context projection, with a fresh optimizer and the same six-epoch per-step
warmup/cosine schedule. This is an explicit fresh restart, not an optimizer resume.
There are 4,224 updates, batch 1,536, 80 atoms, and Al/Mg/Ta 0.1 ps temporal pairs.
Peak learning rates remain 3e-4 encoder and 3e-3 heads. The single exported
embedding and all spatial/temporal/TDA/forecast/nuisance objectives are unchanged.
W&B is online. Node53 allocation 984861 has a safety cutoff at 2026-09-09 05:25:06
Europe/Paris. Static-Al analysis remains scheduled after training.

Each benchmark variant starts from identical model weights. The baseline uses
snapshots of the actual pre-change encoder and trainer. Each timing phase uses
a fresh fused AdamW optimizer at the requested peak learning rates. The measured
step includes acquiring its batch, all forward passes, loss and gradient replay,
gradient clipping, optimizer update and CUDA completion. CPU prefetch is used for
the host-data variants. Resident packing and first checked step/compilation time
are recorded separately. The 24-update block includes one expensive task-gradient
diagnostic; full training runs that diagnostic only once per 100 updates.

Before timing, every candidate is compared with the original full-batch loss and
parameter gradient, plus encoder outputs and rotation behavior on mixed-material
clouds. Fixed acceptance thresholds are relative gradient L2 error <=0.005 and
embedding/rotation error MSE <=1% of the measured response to a 0.005 Å coordinate
perturbation. Thus numerical RMS must be below 10% of that nuisance response.
These checks establish numerical tolerance, not unchanged long-run scientific
results. The final standard analysis remains necessary.

## Findings and file roles

[Measured comparison](../../output/mace_throughput_20260908/BENCHMARK.md) and
[raw timings and numerical checks](../../output/mace_throughput_20260908/benchmark.json)
are produced by the benchmark. No speedup is claimed before measurement.

The first sweep measured 3.364 s/update for the baseline, 2.822 with chunks of
1,536, 2.498 with GPU-resident data, and 2.489 with geometry reuse. The latter's
incremental difference is small. TF32 failed the predeclared embedding tolerance:
its error was 8.4% of nuisance-response MSE, about 29% in RMS, versus a 10% RMS
limit. It is excluded. The initial compiled-FP32 trial exposed the retained-graph
buffer-donation incompatibility; the confirmation reruns it after that fix.

[Confirmation results](../../output/mace_throughput_20260908/confirmation/BENCHMARK.md)
and their raw JSON preserve the separate, longer measurement protocol.

The first 40-update compiled confirmation passed ordinary gradient checks, but
the independent preflight exposed incorrect repeated-backward task gradients.
Its timings are preserved under `confirmation_before_diagnostic_fix/` and are
superseded. The queue failed before any new training updates. Its complete logs
and provenance are in `failed_preflight_20260908_163114/`.

The corrected 40-update confirmation repeats the timing with explicit eager task
replay and checks both gradient paths. It measured **3.200 → 2.422 s/update,
1.32× throughput and 24.3% less time**. Ordinary relative gradient L2 error is
2.81e-6; diagnostic error is 3.93e-6. Numerical embedding error is 9.04e-9 of the
nuisance-response MSE. Peak allocated memory is 60.2 GiB; GPU-resident data use
0.679 GiB. TF32 remains disabled. The corrected detached queue was launched on
node53 at 16:44:54 Europe/Paris, with full preflight before the first update.
See the [final comparison](../../output/mace_throughput_20260908/FINAL_BENCHMARK.md)
for the distinction between the initial sweep, confirmation and actual training.

New shared GPU-data, replay and benchmark modules under `src/` are maintained
implementations. This directory contains versioned experiment records.
The pre-change source snapshots, logs and generated reports under `output/` are
disposable benchmark/provenance artifacts. Eight focused tests pass, including
chunk-boundary gradient replay, precision restoration and exact eager task
replay when the model is otherwise compiled; existing support and objective
tests remain.

The corrected queue passed full preflight and reached update 30/4224 at 16:47:49 Paris with finite loss and online W&B. [Live verification](../../output/mace_throughput_20260908/live_verification.json) records process identity, execution settings and the early timing/ETA. [W&B run](https://wandb.ai/teshbek/PointCloudMaterials/runs/7eee2176).
