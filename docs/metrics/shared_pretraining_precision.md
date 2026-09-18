# Shared pretraining precision timings, version 1

This is a separate operational measurement of the actual cached training update,
not a training objective or a scientific accuracy score. FP32 disables TF32;
BF16 uses autocast with FP32 weights, optimizer state and VICReg statistics.
Each process initializes the same architecture and seed and uses the same
batch of `batch_size` largest-support anchors in the selected material/potential group
(1,024 in the original comparison; 64 in the v4 initial preflight),
with identical microbatch sizes and view identities. AdamW uses fixed LR 0.0003 and its default weight decay 0.01 in every
benchmark mode. Live training uses weight decay 0.0001 and the configured
warmup/cosine LR; the benchmark measures a matched update computation, not
the live optimization trajectory.

Two complete updates warm up CUDA kernels and optimizer state. Five following
updates are timed individually with `time.monotonic()` and CUDA synchronization
at both measurement boundaries. CPU graph construction/input preparation is
excluded; pinned-host transfer, both encoder passes, full-batch heads/losses,
backward and optimizer step are included. The same batch is reused each time.

`median_seconds` and `mean_seconds` describe these five durations; lower is
faster. `speedup` = FP32 median / BF16 median; above one favors BF16.
`time_reduction_percent` = 100 * (1 - BF16 median / FP32 median).
Peak allocated and reserved memory are PyTorch allocator maxima after the
warmup reset, divided by 2^30 (GiB). They exclude non-PyTorch CUDA memory;
retained allocations and allocator reservation from warmup can remain present.
Memory reduction uses peak allocated GiB, not reserved memory.

These measurements describe one repeated batch and device per architecture,
not end-to-end training throughput, training-seed uncertainty or precision
equivalence of final model quality. Raw update durations and exact row indices
are preserved beside the comparison, with producer/config/data identities.

The v4 geometry-protected models use selective scalar BF16 arithmetic, including
GATr compensated BF16 products with FP32 accumulation; `precision=bf16` no longer
means blanket backbone autocast. `architecture_revision`, encoder parameter
count and implementation hashes identify the measured code. Optional `--compile`
compiles only the encoder with precision casts preserved and allows upstream
graph breaks. `compiled` records that choice. `warmup_seconds` includes compilation
and the complete warmup updates; it is not isolated compiler time. Training enables the same compilation path through the explicit
`compile_encoder` configuration field; benchmarking remains a separate command.


The v5 comparison uses effective batch 1,024 on both devices. MACE uses
microbatch 64; GATr uses 256. The reference is eager FP32 with TF32 off, using
identical enlarged encoders, objectives, seed, observations and update count.
The selected path is dynamic compiled selective BF16, including the fused
GATr compensated GEMM. Speedup is eager FP32 median / selected median. Additional
compiled FP32 and compiled three-GEMM GATr runs isolate compiler and custom
kernel effects. All backward passes compile with autocast off. The reported
compiler counters and initialization-plus-warmup duration are separate from
steady-state measured updates. Compile time must be amortized and input
preparation remains excluded. This comparison cannot establish epoch speedup.

The v6 Al-only repair removes GATr attention graph breaks, disables both einsum
path optimizers, and uses shape-polymorphic scalar slices/Triton strides. Its
conditioned heads and separate encoder LR are documented in
[the shared v6 protocol](shared_pretraining.md#al-only-conditioned-head-protocol-v6).
The September 18 v5 timing table remains historical; its timings are not new
measurements of v6. Training does not perform a hardware benchmark.

The broad full-TDA continuation retains those precision boundaries. Parent
initialization rebases decoder outputs to the new target normalization; it does
not change GEMM arithmetic or provide a new speed measurement. No profiling
stage is part of the dependent preparation/training queue.

Mixed-triplet v7 GATr reuses these encoder precision boundaries. Grouped head
normalization, VICReg and the timestamp-corrected second difference run in FP32;
calibration reductions use float64 and store FP32 buffers. This protocol has
three or four snapshot views per anchor, so its step timing is not a matched
comparison against the earlier two-view training workload. See the
[objective definition](shared_pretraining.md#mixed-material-dynamic-triplets-v7).

New shared-runtime submissions use compact W&B projection and interval-mean
training charts. Raw timing, peak-memory and full per-update values stay in the
technical JSONL; profiling equations and precision kernels are unchanged. This
logging change is not a new speed measurement. See
[logging semantics](shared_pretraining.md#compact-wb-projection-new-submissions).

The v8 continuation reduces spatial updates from four encoded snapshots to two,
while temporal updates retain three. It calibrates the fixed backtracking
coefficient using separate FP32 encoder-gradient norms under the same selective
BF16 forward arithmetic, outside production training. No kernel precision or
hardware-benchmark equation changes. Workload reductions alone are not measured
precision speedups; see [v8](shared_pretraining.md#temporal-only-calibrated-backtracking-v8).

Mixed MACE v9 additionally caches central atom tensors for an FP32 equivariant
bond-order head. Its separate run-local preflight records actual mixed updates
and exposed input-wait time, with compilation warmup identified separately.
The older homogeneous `profile` command does not measure this mixed objective.
Production training does not invoke that profiler or any hardware benchmark.
