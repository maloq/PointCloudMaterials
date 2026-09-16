# Causal MACE runtime benchmark

Uses hash-verified physical cache examples and the actual D model/task objective.
At each tensor width and effective batch, reference, packed host and packed device
start with identical parameters and examples. FP32/TF32-disabled throughout.
Before timing, exported states must agree within atol 2e-5, rtol 2e-4; the complete
parameter gradient must have relative L2 error <= 1e-3 and identical absent-gradient
patterns. This is numerical agreement, not bitwise reproducibility.

Training timing includes source tensors/labels, forward, full physical loss,
backward, clipping and AdamW step. It excludes initialization, cache verification,
CPU preflight and one-time residency. Each trial follows declared warmup, times
several updates with perf_counter and synchronizes CUDA at its end. Median trial
wall time divided by updates gives step_seconds; effective batch divided by
step_seconds gives examples_per_second. Raw trials are retained. Peak allocated
and reserved GiB are PyTorch process peaks during timed training after warmup;
they include resident data, gradients and optimizer, but not other GPU processes.

Evaluation examples/sec includes actual encoder/heads, input transfer and output
CPU copies on the declared validation prefix; it excludes physical metric tables.
No model selection is performed. GPU process snapshots before/after disclose
contention but cannot prove exclusivity between snapshots. Speeds measured with
other jobs running are provisional, not isolated hardware peak throughput.
