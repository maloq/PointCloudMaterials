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


Table export: 2026-09-18T10:08:38.064250+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
