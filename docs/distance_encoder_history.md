# Trainable MD-history crystal-distance encoder

Separate Al/Ta adaptation of the completed six-frame encoder is documented in
[material fine-tuning](distance_encoder_material_finetune.md), including the
limited shared-ancestry Ta shooting evaluation.

## Two six-observation queues

Run the existing `python -m src.research.distance_encoder.history_queue submit
--config CONFIG` command in `pointnet-torch214` with either recipe:

- `configs/distance_encoder/md_dense6_075nominal_20260926.json`: all-data run,
  nominal 0.75 ps; native Al uses 0.75 ps and external trajectories exactly 0.70 ps.
- `configs/distance_encoder/md_dense6_010ps_20260926.json`: exact 0.10 ps throughout,
  with independent external ancestry roles and a native-Al-only initialization.

There is no interpolation. Actual offsets and the explicitly approved 0.70-ps
approximation are recorded in source plans and prediction-context receipts.
Each queue trains real history and its repeated-current control for 12 epochs.
Global batch is 1024: two H100s, microbatch 256, two accumulation steps. W&B stays
online in `teshbek/PointCloudMaterials`, group `cd-mace128-md-cadence-20260926`.

One twelve-worker CPU preparation job per experiment extracts a shared float32
geometry bank. A dependent sealing job verifies every coordinate receipt. GPU
jobs depend on the seals and request 16 hours; caches are on IDS outside the repo.
The all-data job additionally prepares the fixed Al spatial-evaluation geometry.
The 0.10-ps queue evaluates external holdout distance/confidence, without claiming
spatial path warning metrics. It updates each existing scientific W&B run.

Roots are `${storage:analysis}/distance_encoder/md-dense6-075nominal-20260926`
and `${storage:analysis}/distance_encoder/md-dense6-010ps-20260926`.
Inspect `technical/launch.json`, frozen `technical/code/`, queue-state, per-arm
state and Slurm logs. Geometry locations are explicit in each recipe. Resume
frozen producers and preserve identities; never change a running configuration.

The abandoned mixed-cadence D6 proposal never submitted GPU training. Its
preparation was cancelled and CPU writes also hit the IDS quota. Completed native
coordinates and compatible evaluation observations are reused only after SHA256
and source-contract checks, recorded in `technical/*reuse.json`. Unsubmitted
Al-only 0.75-ps candidate receipts remain historical preparation, not a trained fit.

To keep new geometry on IDS, inactive 2026-09-11 full-forecast and 2026-09-19
transfer caches are archived under `${storage:archive}/training-cache-archive-20260926/`,
with per-file SHA256 receipts and original paths preserved as links. Checkpoints,
results and active cache leases are preserved.

[Scientific protocol](../experiments/distance_encoder_20260926/DENSE_HISTORY.md) ·
[Native spatial definitions](metrics/distance_encoder_dense_history.md) ·
[External holdout definitions](metrics/distance_encoder_dense_external.md)

## Three observations over six ps

Use conda `pointnet-torch214` and submit:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.distance_encoder.history_queue submit --config configs/distance_encoder/md_history6_20260926.json
```

The launcher freezes code, metric contracts and both arm recipes. It submits
an independent eight-core CPU preparation job and a two-H100 12-hour GPU queue.
The GPU job fine-tunes real history and the repeated-current control sequentially,
then evaluates both and replays the original snapshot checkpoint. Work is detached.
Training and CPU preparation can overlap. CPU failures are recorded and propagated.
Scientific fits/evaluations remain online in `teshbek/PointCloudMaterials`, group
`cd-mace128-md-history6-20260926`; local checks create no W&B runs.

Run root: `${storage:analysis}/distance_encoder/md-history6-20260926`.
Inspect `technical/launch.json`, `technical/queue-state.json`, `technical/slurm-JOB.log`,
and each arm's `technical/state.json`. Checkpoints include optimizer state and
deterministic permutation position. Resume the recorded frozen worker; do not
resubmit against an existing launch or change its code/config in place. An incomplete
time-limited fit remains resumable and does not enter final evaluation.

Training reads the existing sealed structural geometry and distance labels;
there is no new simulation or learned-feature cache. Coordinates occupy one
resident bank per GPU and integer indices select the exact tracked-center histories.
The first two label frames of each source are excluded identically in both arms.
Each arm records exact eligible counts, material weights and input provenance.

Evaluation geometry lives outside the repository at
`${storage:cache}/distance-encoder/md-history6-evaluation-20260926`. It reconstructs
the local neighborhoods at -6/-3/0 ps for every fixed held-out row and existing
scan-position atom. It verifies raw IDs/timelines, PTM/component receipts, and
agreement of current geometry/labels with previous spatial evaluation. Geometry
receipts are reusable by both arms; no trained embeddings are cached.

Results are in `{baseline,real,repeated_current}/front/analyses/front-v1/`, with
distance/alarms/path/reliability CSVs, `tables/METRICS.md`, raw predictions and
frozen implementation hashes. Encoder and temporal-predictor input records are
saved separately from target/audit metadata. The trained spatial encoder still
exports 128 dimensions per frame; the temporal predictor also produces a 128-D
state internally, but is not interchangeable with the per-frame export.

[Scientific protocol](../experiments/distance_encoder_20260926/HISTORY.md) ·
[Definitions](metrics/distance_encoder_history.md)

The audited eligible release contains **11,375,472 training sequences**
(Al 7,229,416; Mg 377,496; Ti 707,000; Ta 3,061,560) and
**190,080 Al selection sequences**. There are 117 train sources and 15 selection
sources; external branch counts do not imply independent ancestry.

Submitted 26 September 2026: CPU preparation **1009761**, GPU queue **1009762**.
At submission the CPU job runs on nodecpu04; the H100 request is pending on
`QOSMaxGRESPerUser`, so training starts when existing allocations release GPU quota.
No existing fit or user allocation was stopped. The CPU preparation is independent
and already processing held-out sources.

## Submitted six-observation runs (26 September 2026)

| Experiment | CPU geometry | Seal | Separate spatial preparation | GPU queue |
| --- | --- | --- | --- | --- |
| Nominal 0.75 ps (actual 0.75/0.70) | 1009790_0 | 1009791 | 1009789 | 1009792 |
| Exact 0.10 ps | 1009793_0 | 1009794 | Not applicable | 1009795 |

Both GPU jobs are detached with successful-preparation dependencies. Each runs
real history then repeated-current for 12 epochs, followed by its declared
evaluation. At handoff CPU preparation was running and GPUs waited on dependencies.
