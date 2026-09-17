# Same-H100 backbone screen and matched onset repeats

The physical errors, fitting gates and workload measurements retain the
[v2 definitions](local_predictability_backbone_v2.md). This comparison reads their
immutable exports and does not change the target normalization, populations or
prediction loss. All current fits use seed 20260919.

`h100_speed.csv` compares the same eight windows, observation shapes, FP32
precision, effective batch eight, execution microbatch eight, and GPU model.
Data/configuration/implementation identities must also match. Update duration is
the arithmetic mean of 20 synchronized forward/backward/clipping/AdamW timings
after two warmups. Windows/second is eight divided by that mean. GATr training
speedup is MACE mean duration divided by GATr mean duration; values above one
favor GATr. Validation speedup is the ratio of the two total validation durations
on those eight windows. Cold preparation and allocated peak GiB are separate.
Warm repeated-batch timing does not measure full-epoch input throughput, and the
short validation timing is not a large-validation benchmark. Report the actual
kernel trace, not an inferred FlashAttention dispatch.

`physical_snapshot.csv` contains the standardized present/future MSE and each
future horizon from completed 2,048-update fits, with the selection-chosen step
and exact split population. Row indices, source/center/anchor and standardized
targets must match before a comparison is exported. These are one-seed point
estimates, not training-seed uncertainty intervals.

The onset repetition matches the completed native MACE v1 experiment: 4,096
snapshot-parent updates and 4,096 additional updates for each of snapshot,
12 ps history, and repeated-current-frame control. Each architecture has its own
parent. Optimizers and source samplers restart for each continuation. The new
GATr run verifies cohort, release, labels, conditions, selection rows, seed,
budget and the terminal sampler state against all four completed MACE stages.
All comparisons use the same eligible at-risk population and seven conditions.

`onset_VARIANT.csv` copies the existing prospective assay calculations for both
architectures: six horizons, source-weighted joint event NLL, window log loss,
Brier score, average precision, calibrated threshold, recall, false-positive
rate and precision. See [native onset](local_predictability_native_onset.md) and
[assay definitions](local_predictability.md). The saved per-metric confidence
intervals are the original source bootstrap intervals, conditional on this one
seed; they are not confidence intervals for the architecture difference.
Dense alarm timing is not inferred from the sparse native grid.

The MACE onset reference was already fitted using the frozen v1/e3nn trainer.
Its scientific objective and example budget match the GATr repetition, but its
historical elapsed time is not used for acceleration claims. Speed is measured
separately using fresh cuEquivariance MACE and GATr on the same H100. No additional
model, seed, objective or budget is selected from test outcomes.
