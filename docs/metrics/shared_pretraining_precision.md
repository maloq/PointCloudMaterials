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

## Local structural support (v10)

Current structural MACE/GATr observations use fixed material normalization
`x_model = x_A * 9.192189 / scale_material`, crop to radius <8 before packing,
and quintic C2 weights equal to one through radius 6 and zero at radius 8. There
is no outer halo. MACE uses 5-unit edges, two layers and pooling tapers 0–3,
3–5, 6–8; GATr globally attends only within the cropped sphere and scales its
weighted count by 100. Training, static inference and trajectory inference share
`src/data/structural_pretraining/support.py`. Geometry baselines using the
encoder's support and radial controls now also use that local support. Existing
85-component physical and 80-point instantaneous-TDA targets are unchanged.

The revision is incompatible with previous large-support checkpoints. Historical
exported metric contracts and results retain their original support definitions;
reproduction of those runs requires their frozen code. Current within-domain
VICReg, selection, bond-order and temporal-only curvature metric formulas are
unchanged. Curvature weights are recalibrated at initialization using training
batches under the declared 2%-loss / 10%-encoder-gradient policy. See
[local protocol](../shared_pretraining_local_structure_20260918.md).

## Local GATr bond supervision (v11)

`shared_pretraining_local_gatr_bond_v11` adds the same q4m/q6m targets and
`12 * mean_m(error^2)` per-order loss as local MACE, with mean over orders and
loss coefficient 0.1. Only current/partner snapshots are supervised; the past
snapshot remains curvature context. Its 704 training-only tensor features are
formed from the final learned atom multivectors' four vector sectors across
eight channels: separately replace each vector v by `v/sqrt(sum(v^2)+1e-4)`,
compute real component-normalized solid harmonics l=4,6, and average atoms using
the encoder's local support weights. Concatenate 32x4e then 32x6e; an equivariant
linear readout predicts 1x4e+1x6e. All these operations use FP32 under BF16 AMP.
No raw target vectors or invariant z enter this head. Powers are taken before
pooling, so inversion-symmetric local order need not vanish. Export remains
128-dimensional; no additional encoder pass is required. Selection remains
physical+0.25*TDA, with bond errors and magnitudes reported separately. Training
is from scratch for five epochs with the small curvature coefficient recalibrated
on training batches, including the bond loss in its base-gradient comparison.

The 19 September two-GPU structural MACE executor preserves the global objective
and sums encoder gradients before clipping. Each device uses the same selective
BF16/FP32 boundaries. Device-scaling preflight is recorded separately from fits;
production training does not run precision or hardware benchmarks.

For the process-prefetch two-GPU executor, throughput is measured in a
separate preflight on matched sampled updates, including exposed input wait and
update computation after startup. Global B and precision remain fixed. CPU worker
startup and CUDA compilation are excluded from warmed measurements and reported
separately when measured. These timings do not include periodic validation or
checkpointing; they are not whole-run completion-time measurements.
