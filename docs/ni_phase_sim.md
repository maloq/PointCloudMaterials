# Dense Ni coordinate/velocity learning

## Measurement-supervision workflow

`python -m src.research.ni_phase_sim.measurement_queue launch --config
configs/ni_latent_sim/phase_measurements_20261014.json` starts the twelve-model
follow-up described in [its protocol](../experiments/ni_phase_measurements_20261014/README.md).
The `collect` stage computes measurements and smooth JVPs from sealed teachers;
`prepare` pins data, parent checkpoints, source and metric contracts. They are
complete before launch. The separate preflight recipe is local only.

Launch starts one detached worker on the current allocation and requests five
single-GPU workers with RTX6000PRO/A100/A40/L40S as alternatives. All workers
share one absolute 12-hour deadline. A shorter local allocation can submit one
handoff within that same deadline. Stage subprocesses have a hard timeout as a
backstop to checkpoint-aware stopping. Only the launched stage's process group
can be terminated. Other chat jobs and simulation campaigns are untouched.

Stages are train, evaluate, births and diagnose, claimed with shared file locks.
Each has two bounded attempts and visible failure receipts. Checkpoints retain
optimizer and sampling state. Evaluation failure does not block independent
fits. No scheduled chat task is needed. Scientific W&B IDs use `phasemeas-`;
frozen diagnostic heads and CatBoost probes remain local. A shared feature pool
retains six generated caches and protects active leases.

Watch `technical/submission.json`, `lane-*.json`, `worker-start-*.json`, and each
`analyses/<arm>/technical/progress.json`. Frozen diagnostics write
`diagnostics/seed-*/technical/progress.json`. The submission receipt records the
absolute deadline; completion requires all twelve fits, their evaluations and
three diagnostic bundles. Partial checkpoints and failure receipts remain
available when the deadline expires.

Entry point: `python -m src.research.ni_phase_sim STAGE --config CONFIG`.
Recipe: `configs/ni_latent_sim/phase_space_20261013.json`.
Scientific protocol: [phase-space prediction](../experiments/ni_phase_sim_20261013/README.md).

`prepare` pins the previously prepared diverse320 training anchors and310 pooled
test anchors, checks duplicate identity and ancestry, and preserves original
source roles. It does not wait for the separate density-rate campaign.
`benchmark` is a local two-training-state integration/performance diagnostic;
it creates no W&B run. Its outputs use the preflight `dense-20261013-v1` run.
Scientific work has the separate `dense-20261013-v2` identity.

`launch` freezes source/configuration/metric definitions and submits one12-hour
two-GPU job to alternative RTX6000PRO/A100/A40/L40S partitions. The job owns a
detached two-worker queue; no chat review or automation is required. Existing
node jobs are left alone. No automatic extension exceeds this campaign's limit.
`team`, `worker`, `train`, `evaluate` and `compare` are its maintained stages.

Both workers collect missing training teachers. Once training is ready, one
starts fitting while the other finishes test teachers. A completed fit has an
independent evaluation stage. First direct/latent controls complete before the
derivative arms. Failed work has two bounded attempts, resumes the same last
checkpoint, and leaves an explicit receipt; other independent fits continue.
Training checkpoints every50 updates, online W&B under teshbek/PointCloudMaterials
with stable `phase-...` IDs. Scientific tables are durable before W&B evaluation
summary updates. The queue stops near the Slurm deadline and preserves partial
checkpoints instead of silently changing the declared training budget.

Simulation response observables go through SCRATCH to checksummed STORE;
float64 cropped Q,V and tangents are retained before quantization. Source
manifests and fixed atom IDs are verified by the existing native readers.
These are local response packets, not new quantized full-history archives.
The existing shared Ni EAM/FS CUDA backend is unchanged.

Watch `technical/submission.json`, `launch-<job>.json`, `lane-*.json`, each
`analyses/<arm>/technical/progress.json`, and checkpoint/attempt receipts.
Results include `tables/METRICS.md`, frozen hashes, paired comparisons and
`plots/trajectories/` beneath each analysis. The preflight benchmark used A40;
actual elapsed time depends on the allocated GPU and full cohort I/O.

## October13 launch

The single two-GPU request1036591 was pending because free compatible GPUs were
on different nodes. Its receipt-owned pending reservation was replaced with two
cooperating one-GPU workers using the same immutable source/configuration:
**1036592 on node29/A40** and **1036593 on node32/A40**. They have a shared
absolute12-hour deadline and use the existing `worker` CLI stage with an
explicit `--lane` and `--deadline`. The active receipt is
`technical/submission-workers.json`; the original reservation receipt remains.
No other campaign job was stopped or changed. The A40 batch256/microbatch256
benchmark used up to17.3GiB and1.7s for a20-step derivative-heavy update.

Actual training and checkpoint progress verified: `direct-s20261023` reached update350, with an online W&B receipt. All320 training teachers are complete; held-out acquisition continues independently. The shared deadline is 2026-10-13T14:08:07.350147+02:00.

Completion checked October13: all12 fits reached8,000 updates, all12 evaluations and W&B summaries completed, and all630 teachers are present. Workers exited without pausing at10:45:57 and10:46:48 CEST, before the14:08 cap. See the scientific protocol for results; historical launch/progress receipts above remain unchanged.

## Nucleus readout follow-up

`python -m src.research.ni_phase_sim.births launch --config
configs/ni_latent_sim/phase_births_20261013.json` freezes a separate evaluation
bundle and starts a detached worker in the current GPU allocation. It reuses
the old sixty-source birth cache, adds the completed thirty-source catalogue,
then extracts measurement controls and frozen phase-model features. Four CPU
probe workers use two threads each; GPU inference uses the available node GPU.
There is no new neural training or physical simulation.

The run is `output/ni_phase_births/90sources-20261013-v1/`. Execution receipts
are in `technical/{launch,progress,cohort,complete}.json`, with `worker.log` and
an immutable `source/` bundle. Every independent probe saves its model,
predictions and completion receipt before moving on. Two bounded attempts per
probe preserve failure receipts and allow other fits to proceed. Rerunning the
frozen `run` command resumes completed data/features/fits. The shared feature
pool retains at most six model caches; models, predictions and metrics are
durable. Existing W&B training IDs receive final evaluation summaries; local
probes create no new online runs. `summarize` regenerates the comparison from
retained probabilities under the frozen metric contract.

The October 13 follow-up completed all 216 probes in 330 seconds. There were no
failed fits or W&B summary failures. Input coverage is 19,800 assay rows from
19,652 unique states, with no missing exact leads. Test events: 81 default
64-atom roots/38 ancestors and 135 32-atom roots/42 ancestors. See the result
README for the measured improvements and limits.

## History-attention workflow

`src.research.ni_phase_sim.history` supports prepare, collect, benchmark, launch,
worker, train, evaluate, births and compare. The recipe is
`configs/ni_latent_sim/phase_history_20261013.json`. The separate preflight recipe
keeps local checks outside scientific training. Collection recovers exact four
frame histories; completion records hashes and checks current Q,V against the
parent caches. The history cache is derived data, not independent simulation.

Launch freezes source/configuration/metric definitions, starts a detached worker
on the available node GPU and requests three additional single-GPU workers with
RTX6000PRO/A100/A40/L40S as alternatives. Workers claim independent stages via
shared locks; one prepares measurement controls while other workers train.
Every arm saves resumable checkpoints every50 updates and has at most two
attempts per stage, with visible failure receipts. Independent fits continue
after a failure. Each lane has a12-hour window and may submit one continuation
from the same frozen bundle if unfinished at its deadline. Scheduled chat work
is not required. Only this queue's jobs are controlled by this workflow.

Run: `output/ni_phase_history/attention-20261013-v1/`. Live ownership and progress
are under `technical/{submission,lane-*,worker-start-*}.json`; per-model logs,
checkpoints and progress are in `analyses/<arm>/technical/`. Physical plots and
tables are beside those; nucleus probes are grouped under `births/analyses/`.
The six-cache retention pool covers disposable latent features only. Online
W&B runs use stable resumable IDs; local probes update their associated runs.

October13 launch: local worker in allocation1037290 plus jobs1037567 (node60,
RTX6000PRO),1037568 (node59,RTX6000PRO),1037569 (node50,L40S). All three requests
started. The full-history derivative, full-history values and classic derivative
fits wrote resumable checkpoints and online W&B receipts. Measurement controls
completed; the local lane then began the classic values fit. The queue has21
models and automatically evaluates each fixed final checkpoint. Launch checks
and exact current-input comparability receipts are in the run's technical folder.
