# Overnight Ni history/context queue, October19

Scientific recipe: [protocol](../experiments/ni_ordering_flow_20261019/README.md),
[configuration](../configs/ni_latent_sim/ordering_20261019.json).

The detached nine-hour queue has21fixed-budget fits and independently resumable
source preparation, full-path, rare-entry, response and birth evaluations. Source
and config are copied before launch. Per-stage locks, two bounded attempts,
checkpoint resumption, online resumable W&B and visible failure receipts are
independent of chat wakeups. A failed task does not block unrelated baseline fits.
Allocation continuations retain the original global deadline and fixed budgets.

GPU0 on node53 currently rejects CUDA initialization (error999); GPU1 passed.
The queue requests five alternative RTX6000PRO/A100/A40/L40S workers in addition
to GPU1. No unrelated job is changed. Read actual launch receipts for allocation
and startup status; configured workers are not evidence of running jobs.

Output: `output/ni_ordering_flow/history-context-20261019-v1`, linked to STORE.
The source-level observed-history cache is on IDS; native futures remain sealed.
Each source checkpoints every32requested frames. Original regional caches and
all overlapping native measurements are reused. Preparation remeasures births
with nearest80 reselected at every observation, preserving central IDs and labels.

Inspect `technical/submission.json`, `lane-*.json`, `worker-start-*.json` and
`analyses/*/technical/progress.json`. `complete.json` means fixed-budget training
or preparation, while evaluation, rare, response and births each have separate
completion markers. `*-failed.json` records bounded failures. The comparison
publishes available results without treating missing seeds as successful fits.

## Verified launch

Launched October19 at00:30:49 Paris; global deadline09:30:49. GPU1 of allocation1046338 runs locally. Jobs1047123–1047127 are running on RTX PRO6000(node60), A100(node02), and A40(node32/node33/node29). All three baseline fits have online W&B receipts and resumable training progress; first passed5000updates. Six source packets completed at the first follow-up, with no failed stages. Real-data model/derivative backpropagation, checkpoint resume, four-origin forecast/rare scoring and plot export passed locally without debug W&B runs. See `technical/launch-verification.json`; these are execution checks, not new predictive results.

Separate physical-pilot worker1047132 obtained an A100 on node04; worker1047133 is queued behind the per-user GPU limit. The unavailable local GPU0 still occupies an allocation slot. No user allocation is modified to bypass that limit. Latest launch observations are in `technical/latest-launch-status.json`.
