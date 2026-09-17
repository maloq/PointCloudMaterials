# Predictive-memory experiments: stop update

**17 September 2026, 13:47 CEST onward.** The user requested that current runs stop.
The local training queue is inactive, both simulation workers have exited, and
no experiment was restarted. The H100 allocation remains available but has no
training process. This updates the [earlier consolidated report](../research-summary-20260917/RESULTS.md);
its historical results remain unchanged.

## New result: second-seed 12 ps history under the original loss

One additional fit completed its full 12,000-update budget and evaluation after
the previous report's capture: width 16, seed 20260918, positions and velocities,
12 ps history, present-reconstruction weight 0.05. Its final table export failed
at 12:07 CEST because a hardware-benchmark file did not match its metric-contract
hash. That mismatch is now resolved in the checkout. This was an export failure,
not a numerical failure during training.

The saved latest checkpoint is at update 12,000; validation selected update 4,000.
Before finalizing the export, we checked the resolved configuration, checkpoint
budget and release identity, unchanged predictive-memory metric implementation,
all source/center/anchor pairings against the snapshot control, and all **45**
saved metric estimates with their source-bootstrap intervals. Everything matched.
Export recovery used CPU only and performed no optimizer updates. The original
failure log and status are preserved in the [recovery audit](../optimization-original-seed20260918/xv-H12/technical/export-recovery.json).

| Model, original loss, seed 20260918 | Selected update | Test joint-path NLL ↓ | Test future MSE ↓ | Test present MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Current positions and velocities | 4,000 | 0.92506 | 0.93388 | 0.73871 |
| Real 12 ps history | 4,000 | 0.92750 | 0.92347 | 0.75455 |

These use the existing partial-observation protocol: 30 test sources, three
neighboring anchors per source, five physical future lags through 96 ps, and
training-only target normalization. NLL scores the joint predictive distribution;
MSE scores its mean. [Metric definitions](tables/METRICS.md).

Paired gain means snapshot score minus history score; positive favors history:

| Score | Gain | 95% whole-source bootstrap interval |
| --- | ---: | --- |
| Joint-path NLL | −0.00244 | [−0.02285, +0.01879] |
| Future physical MSE | +0.01041 | [+0.00301, +0.01896] |
| Present reconstruction MSE | −0.01584 | [−0.08569, +0.03088] |

**History reduces mean future error by about 1.1%, but the primary NLL comparison
is unresolved.** The intervals use the existing 500-resample source estimator
and do not measure training-seed uncertainty. H48 and repeated-current-frame
controls were not run for this longer-budget seed, so this is explicitly a
partial comparison, not a completed four-model cohort. The
[paired metric export](../optimization-original-seed20260918/partial-comparison-stopped/tables/partial-comparison.csv)
and its frozen definitions retain the exact values.

## Updated completion and interpretation

The optimization follow-up ends with **10 of 16 fits complete**, six not started:

| Configuration | Seed 20260917 | Seed 20260918 |
| --- | --- | --- |
| Original loss, weight 0.05 | Snapshot, H12, H48, repeated frame complete | Snapshot and H12 complete; H48/repeat not started |
| Stronger present loss, weight 1.0 | All four complete | All four not started |

Across the two studies and reported H200 work, the count is now **72 completed
encoder training runs/stages and 128 learned probes**. These are work counts,
not independent scientific replications. [Updated fit table](tables/memory-fits.csv).

The main conclusion is unchanged. The stronger-present-loss first seed still
favors history on NLL, but **its second-seed replication was not run**. The newly
completed result uses the original loss and cannot supply that replication.
The width comparison, direct physical-feature baseline advantage, and limits
on sufficiency/onset claims remain as stated in the full report.

## Simulation stop and preservation

One more training source completed since the earlier snapshot. The campaign
ends with **3 of 12 sources complete**, two interrupted, seven not started:

- Sources 000–002: completed 500 K training trajectories, 2,561 frames each,
  paired float32/float16 observations, native restart files and verified STORE
  publication.
- Source 003: interrupted during melt preparation; its two melt restart files
  and other partial artifacts are retained in SCRATCH and in a verified STORE
  archive.
- Source 004: interrupted during measurement; its melt/final-melt and available
  measurement restart files, partial trajectory and logs are also retained in
  SCRATCH and a verified STORE archive. Its sealed-test physical outcomes were
  not inspected.
- Sources 005–011: never started. Prepared inputs and immutable campaign and
  launch records remain available.

Stop signals were sent to Slurm workers **995981_0 and 995981_1**. Their LAMMPS
steps returned exit 143, triggering preservation and worker exit. Their native
status says `failed`; the separate stop receipt records that this was a user
requested interruption. Both archives report completed hash verification. No
partial trajectory is counted as a completed source or added to training data.

See [captured source status](tables/data-production-status.csv),
[stop and archive receipts](technical/user-stop.json), and the
[simulation record](../../../docs/simulations/predictive_memory_precision_20260917.md).
The three finished sources are all 500 K training sources; the planned
temperature/split comparison is incomplete and supports no new model result yet.

## Captured evidence

The [updated evidence snapshot](technical/evidence.json) retains all previously
completed cohorts plus the recovered fit and new source. The
[input hashes](technical/inputs.json) and [metric contract](technical/metric-contract.json)
identify its producers. The separate partial comparison preserves its own
prediction/checkpoint hashes and metric definitions. No training or simulation
will restart automatically from this update.
