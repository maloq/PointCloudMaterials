# Shared pretraining precision timings, version 1

This is a separate operational measurement of the actual cached training update,
not a training objective or a scientific accuracy score. FP32 disables TF32;
BF16 uses autocast with FP32 weights, optimizer state and VICReg statistics.
Each process initializes the same architecture and seed and uses the same
batch of 1,024 largest-support anchors in the selected material/potential group,
with identical microbatch sizes and view identities. AdamW uses LR 0.0003.

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


Table export: 2026-09-18T07:59:38.485727+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
