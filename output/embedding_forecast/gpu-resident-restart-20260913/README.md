# GPU-resident forecast continuation

Autoregressive **990769.0** on **nodesumo01** and direct **990770.4** on **node53**
resume the existing 32-epoch fits from 21 and 5 completed epochs, respectively.

- [Autoregressive log](technical/path_ar_large_aug/command.log) and [handoff](technical/path_ar_large_aug/handoff.json).
- [Direct log](technical/path_direct_large_aug/command.log) and [handoff](technical/path_direct_large_aug/handoff.json).
- [Exact detached launch commands](technical/launches.json) and [collection job 991021](technical/collect/submission.json).
- [Full-cache preflight](technical/full_cache_preflight.json).
- [Protocol and completion semantics](../../../experiments/embedding_forecast_20260911/GPU_RESIDENT_RESTART_20260913.md).

The original batch commands were deliberately interrupted. Their final Slurm
exit codes do not describe the replacement fits. Collection waits for both
allocations and checks both replacement execution records and completed forecast
artifacts. Training is still in progress; no new final scores are claimed.
