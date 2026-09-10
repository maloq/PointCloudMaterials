# Selective BF16 MACE throughput — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Research question: can BF16 accelerate the existing 80-atom MACE objective
without masking small structural changes or corrupting cached task gradients?

Configuration: [benchmark.json](benchmark.json), with an isolated
[candidate configuration](candidate.json). Output:
[`output/mace_bf16_20260908`](../../output/mace_bf16_20260908/).

Reproduction with `pointnet`, from the repository root:

```bash
python -m src.analysis.mace_performance --config experiments/mace_bf16_20260908/benchmark.json
```

The effective VICReg batch stays 1,536 quadruplets. This first screen uses chunks
of 256 clouds because the existing trainer is paused in memory on the same H100.
It retains optimizer, scheduler and random state. A disposable detached wrapper
resumes that exact process in a `finally` block, including benchmark failure or
a 15-minute timeout. Its launch, process identity, logs and pause duration are
recorded in the output directory. The pause counts toward the trainer's existing
wall-clock budget. There is enough allocation and training-budget headroom for
this bounded pause. No production configuration is overwritten by the benchmark.

The baseline is the optimized compiled FP32 path with the same reduced chunk,
GPU data and geometry reuse. Compare ordinary BF16 radial matrices, second-only
interaction autocast, both-interaction autocast, and compensated BF16 radial
matrices. Geometry, nonlinearities in the radial-only variants, feature pooling,
losses and master parameters remain FP32. Every candidate must pass the existing
gradient and perturbation-scale numerical gates before timing; failures are
explicitly recorded. A chunk-256 result does not establish speedup over the
production chunk-1536 run. A full-memory confirmation would be required to claim
that or change the production configuration.

Compensated multiplication splits each FP32 operand into a BF16 high part and
BF16 residual, evaluates high×high and the two cross products with FP32 outputs,
and omits low×low. The custom backward uses the same compensated multiplication
on FP32 gradients, avoiding BF16 gradient output rounding. This is an application
of an established numerical method, not a new scientific encoder:
[Henry et al. (2019)](https://arxiv.org/abs/1904.06376).

The CUDA matrix test checks forward and both operand gradients against FP32;
the full-model benchmark separately checks both ordinary and per-task gradients,
embeddings, rotation behavior and a 0.005 Å perturbation response.

The first screen rejected ordinary BF16: radial-only embedding error was 5.52×
the 0.005 Å perturbation-response MSE; second-interaction-only was 2.27×, and
both interactions 17.76×. These are fixed-weight numerical errors, not measured
outcomes of long BF16 training. The trainer resumed after a 154-second pause.

The initial compiled compensated trial unexpectedly behaved like ordinary BF16.
Inspection of the generated Triton kernel found that Inductor eliminated the
downcast/upcast used to calculate each residual, making that residual zero.
The compensated mode now explicitly enables `emulate_precision_casts=True` on
its compiled radial MLPs. Eager and compiled matrix tests both pass, checking
forward and both backward operands to relative error <2e-5.

[Corrected benchmark](corrected.json) uses 512-cloud chunks, 24 measured updates
per variant, and the actual running experiment's preserved update-200 checkpoint
in `trained_reference.pt`. It compares the current compiled FP32 path against
corrected compensated BF16, with matched initial weights and effective batch.
Its [candidate config](trained_candidate.json) is separate from the live run.
The independently recorded pause remains bounded to 15 minutes with automatic
resume. Results: [corrected report](../../output/mace_bf16_20260908/corrected/BENCHMARK.md).

Reproduce the correction with:

```bash
python -m src.analysis.mace_performance --config experiments/mace_bf16_20260908/corrected.json
```

The corrected comparison passed: **3.162 → 2.617 seconds/update, 1.21×
throughput and 17.2% less update time** at chunk 512. The compensated encoder's
relative parameter-gradient error is 0.000679; its numerical embedding error is
7.61e-6 of nuisance-response MSE, or 0.276% of its RMS response. Ordinary and
per-task gradient checks both passed. The second pause lasted 211 seconds and
the same FP32 trainer resumed, with optimizer and schedule preserved.

The production-size confirmation was queued after the complete
training/probe/static-analysis controller released the GPU. It uses
[full_batch.json](full_batch.json) (40 timed updates, chunk 1536),
[full_candidate.json](full_candidate.json), [plan.json](plan.json), and
[run_spec.json](run_spec.json). The existing queue runs only the benchmark
preparation command with an explicitly empty training-run list. It does not
start another scientific training or modify the active run. Output:
[`full_batch/`](../../output/mace_bf16_20260908/full_batch/).

```bash
python scripts/experiment_registry.py run --spec experiments/mace_bf16_20260908/run_spec.json
```

The 1.21× figure compares identical chunks of 512 and must not be multiplied
by the preceding optimization result or described as the live run's speedup.
Long-run scientific quality under BF16 training has not been measured.

File roles: `src/models/encoders/mace_bf16.py` is shared precision implementation;
`tests/test_mace_bf16.py` is its numerical regression test. This directory contains
experiment records. The pause wrapper, process snapshots, logs and benchmark
results in `output/` are disposable diagnostics and provenance. The maintained
benchmark command is reused and documented in `scripts/README.md`.

Final screen report: [RESULTS.md](../../output/mace_bf16_20260908/RESULTS.md). Ten focused tests passed. The FP32 trainer was verified advancing at update 380 after both pauses.

The user subsequently requested starting BF16 training immediately. The waiting
confirmation and FP32 controller were explicitly stopped. The actual continuation
is [mace_bf16_training_20260908](../mace_bf16_training_20260908/README.md), restoring
update 500, Adam moments and the original schedule/batch position. The unexecuted
production-size benchmark configurations remain reproducible records.
