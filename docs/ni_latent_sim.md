# First compressed Ni simulator experiments

## Autonomous overnight queue, October 11

The earlier larger overnight study was only a planning document. Its optional
07:56 CEST review triggered but failed because `codex exec resume` could not
acquire the active desktop thread's writer lock. No larger campaign was
submitted by that review. The failed timer and its stderr are preserved at
`/home/infres/vmorozov/.codex/scheduled-followups/ni-motion-review-20261011-0756`.

The replacement **actually launched** at13:39:57 CEST on October11, on node60
allocation1034050, using both GPUs. Recipe:
`configs/ni_latent_sim/autonomous_overnight_20261011.json`.
[Scientific protocol](../experiments/ni_latent_sim_20261010/AUTONOMOUS_OVERNIGHT.md).
It contains eight4,000-update transformer fits crossing1/6ps training,
values/physical derivatives and two seeds, using the sealed14,400 local
sequences. The planned36,608-row fitting collection is not included.

Launch command in conda pointnet-torch214:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.ni_latent_sim.workflow submit --config configs/ni_latent_sim/autonomous_overnight_20261011.json
```

Results are under
`/store/PERSO/vmorozov/experiments/ni_latent_sim/autonomous-overnight-20261011-v1`.
Plan identity: `8f8190fd36612fd1edf93244c84d2b99101ce85f5ed39cb6326ded27ec9cda76`.
`technical/launch.json` owns detached timeout supervisors4145375/4145377,
which started independent process sessions and were adopted by PID1 after the
submitter exited. The frozen bundle is `technical/code`. Both online W&B runs
started and wrote real training checkpoints. The committed deadline is
October12 at03:39:46 CEST, about14hours after launch and before this allocation
expires. This is a budget, not a guarantee that the queue occupies every minute.

The local long-horizon preflight completed full batch256/four-particle forward
and backward passes at1 and6ps, including dense physical response gradients,
with finite losses/gradients and no optimizer updates or diagnostic W&B run.
The6ps pass used21.36GiB peak allocated GPU memory. Its8.18seconds were measured
concurrently with scientific training on GPU0, so they are not a dedicated-GPU
throughput estimate. The receipt is `technical/long-horizon-preflight.json`.

The existing resilience worker checkpoints model, optimizer, data/noise RNG
and progress every100updates and saves the likelihood-selected best checkpoint
every200updates when improved. Each stage has two recorded attempts. The
supervisor can recover an unexpectedly exited worker; retries resume the same
run ID/checkpoint. Failed stages remain visible, other independent fits proceed,
and available checkpoints can still be evaluated. Incremental tables and the
progress report do not depend on completion of every fit. No chat process is
part of this dependency chain. A node/allocation failure still ends local
processes; checkpoints and receipts persist in STORE for continuation.

The optional review was re-created for19:39:57 CEST, six hours after launch,
with a fresh ephemeral CLI session instead of resuming the active chat. A real
read-only connectivity call completed with `CHECK_OK` while this desktop chat
was active. Its detached timer, prompt, two-attempt retry policy and receipts
are in
`/home/infres/vmorozov/.codex/scheduled-followups/ni-motion-review-20261011-1940`.
This is a local timer, not a native desktop scheduled task. It requires the node
and login to remain available. Its only role is optional diagnosis and steering;
it cannot gate training and is instructed to preserve healthy running fits.
The review writes `technical/six-hour-review.md` in the campaign.

Transformer extension: `configs/ni_latent_sim/transformer_path_response_20261011.json`
uses two causal width64/four-head blocks, paired whole-path response seeds, and
exact reuse of the completed dense bank. Each worker waits for its corresponding
MLP lane before touching its GPU. Results are under
`ni_latent_sim/transformer-path-response-20261011-v1`; final paired tables and
plots are in `analyses/architecture-comparison-v1`. No new simulation or dataset
is created. The shared six-entry feature cache, original roles, online W&B and
absolute8.5-hour/allocation deadline remain in force.
[Scientific protocol](../experiments/ni_latent_sim_20261010/TRANSFORMER.md).

Transformer preflight on idle GPU1 checked causal masking by perturbing the last
token (exactly zero effect on preceding token outputs), identical dense/native
100-fs endpoints, and full batch256 forward/backward through200steps with all
physical loss terms. Warm costs were0.790/0.786seconds, peak allocated memory
3.95GB; derivative-only transition gradient norm was0.0001047 and finite.
No optimizer update or online diagnostic run occurred. The original300 target
packets are reused through a pinned producer plan; a new model identity does
not create duplicate physical observations.

Dense-bank preparation initially stopped at299/300 encoded targets because
one componentwise cross-GPU tolerance rejected a near-zero velocity component.
All300 physical paths were already saved. The retained
`technical/cross-device-replay-audit.json` establishes exact initial Q/V and
directions, maximum normwise relative Q/V/state/tangent discrepancy1.90e-6,
position maximum absolute1.08e-6Angstrom and velocity5.89e-5Angstrom/ps.
The repaired check compares position and velocity tensor norms separately,
threshold1e-4, while preserving exact initial-state/direction checks. Original
failure receipts and frozen code remain under the original run; continuation
uses `technical/code-v2` and retains the original08:37:31CEST deadline.
A stale previous completion marker caused lane1 to read the last packet early;
that receipt is retained too, and lane1 restarted after all300 were sealed.
New dense completion markers identify the execution attempt to prevent reuse
across restarts. The MLP recipe, tensors, split, losses and update count did not
change. Both MLP lanes subsequently reached scientific training.

The October11 dense-path extension uses
`configs/ni_latent_sim/dense_path_response_20261011.json` and the same workflow.
Both GPU lanes dynamically claim individual physical anchors, each preparing
twenty5-fs value/response targets from a real100-fs continuation. Completed
cropped float64 paths are checksummed from SCRATCH into STORE under
`simulations/ni-dense-path-response-20261011-v1`. Encoded derivative packets use
IDS `ni-latent-sim/dense-path-response-20261011-v1`; the existing14,400-row long
collection is reused without copying or changing it. Fits write
`ni_latent_sim/dense-path-response-20261011-v1` beneath training storage.
Both workers share the original six-entry leased birth-feature cache.
Six scientific fits use5-fs learned steps and3200updates each. The deadline is
8.5hours from submission or five minutes before allocation expiry. Failure or
budget-stop receipts stop a waiting peer. [Scientific comparison](../experiments/ni_latent_sim_20261010/DENSE_PATH.md).

The implementation probe used actual source data with no optimizer updates or
online diagnostic runs. Dense outputs at100fs exactly matched the same model's
native-interval endpoint; history slots retained their intended0.1ps spacing.
Batch256 likelihood forward/backward through200fine steps took0.080seconds
after warmup on RTX PRO6000, with finite gradients. All three physical loss
variants produced finite nonzero transition gradients. The first real physical
path produced20x128 values and20x128x2 derivatives in3.42seconds of target
encoding. Cross-device replay maximum absolute discrepancies were1.40e-5 in
the combined Q,V state and5.92e-7 in its tangent. Per-anchor replay comparisons
allow rtol2e-4/atol2e-5 for the original float32-force hardware difference;
perturbation directions remain exactly equal. Detailed receipts are in the
new run's `technical/*-probe.json` files.

Recipe: `configs/ni_latent_sim/compact_memory_response_20261010.json`.
Entry point: `python -m src.research.ni_latent_sim.workflow submit --config`
followed by that recipe path, in conda pointnet-torch214.

The two local GPUs prepare disjoint native0.1ps sequence packets, join once,
then run matched value and response lanes. Each lane fits16/64-dimensional
codes,1/4-frame memory, and two seeds,3200 updates per fit. Both reuse the exact
unrelaxed eSEN spatial checkpoint and physical response bank. The broad spatial
collection is not loaded. The user's attached October10 discussion motivates
these hypotheses; cited methods in that discussion are not independently
claimed as implemented here.

Results: `${storage:training_storage}/ni_latent_sim/compact-memory-response-20261010-v1`.
Dataset: `ni-latent-sequences-20261010`; new physical diagnostics:
`ni-latent-lifting-20261010`. Plans pin the parent plan hash, original source
manifests, cadence, roles, encoder checkpoint and all metric producers.
Execution is frozen under `technical/code`, with local PIDs, allocation ID,
GPU mapping and absolute8.5-hour deadline in `technical/launch.json`.

The main scientific scope is local stochastic dynamics and compression, not
full-cell atomistic generation. A first empirical decoder selects physical
training microstates and checks their independent MD continuations. A learned
atomistic decoder, equal-cost extra-perturbed-value teacher arm, collective
long-horizon derivatives, and generated-event rates remain later experiments.

[Metric definitions and input contract](metrics/ni_latent_sim.md).

October10 launch: allocation1032567 on node60, two RTX PRO6000 Blackwell GPUs.
Absolute deadline11:32:30 CEST (8.5hours from the original launch), with checkpoint
checks and an external process timeout. `technical/launch.json` records current
PIDs3902421/3902423 and frozen `technical/code-v2`. The initial code bundle and
receipt remain preserved; the implementation update added bounded feature-cache
retention, held-out response diagnostics and final scientific plots before fits
started, without extending the budget. At03:05 both GPUs were actively preparing
the native sequences; training starts after the two-lane preparation barrier.

The local real-data implementation probe encoded16x64 observations and completed
full batch256 forward/backward checks for all four dimension/history settings,
plus a physical-response-loss backward pass, with finite gradients and zero
scientific optimizer updates. First-window preparation took10.1seconds; the
simple two-lane extrapolation for300 anchors is25minutes, excluding birth-history
preparation and filesystem variability. This is a benchmark, not a completion claim.

October10 completion: all16 fits reached3200 updates; both workers completed
their evaluations and physical lifting diagnostics. Final completion was
03:37:47 CEST,35.14minutes after launch, well before the11:32:30 deadline.
The workers exited normally. Both allocation GPUs were idle at the15:50 check.
The completed scientific findings, including unstable6ps rollouts and failed
held-out sensitivity learning, are recorded in
[the experiment report](../experiments/ni_latent_sim_20261010/README.md).

The follow-up `configs/ni_latent_sim/derivative_stability_20261010.json` reuses
the exact sealed feature/teacher collection. It crosses physical-response
matching, learned-map Jacobian regularization, and1/6ps training horizons,
with two seeds and3200 updates per cell. No new full-cell simulations are needed.
Results go to `ni_latent_sim/derivative-stability-20261010-v1` beneath training
storage. The existing workflow submits two detached local lanes, records their
ownership/PIDs and freezes the code. It caps execution at2.5hours or five minutes
before the allocation ends, whichever is earlier. See the
[scientific protocol](../experiments/ni_latent_sim_20261010/DERIVATIVE_STABILITY.md).

Stability launch: October10 at15:59:31 CEST, allocation1032567/node60, GPUs0/1,
timeout supervisors3993416/3993418. Plan identity
`cb76ec2a6770850447339a3348c376703991061935e8990d8df2db3d881fad8a`.
The launch receipt fixes the deadline at18:29:22 CEST. The real-input numerical
probe compared the exact learned-map JVP with a centered finite difference
(relative discrepancy0.000107), checked active-penalty transition gradients,
and completed batch256 forward/backward at both horizons with finite gradients.
Steady benchmark costs were approximately0.009/0.033seconds per1ps update and
0.045/0.067seconds per6ps update without/with the Jacobian term, excluding
physical-teacher loss, optimizer, validation, tracking and readouts. These are
hardware measurements, not reported scientific results.

Stability completion: both workers completed normally by16:22:12 CEST on
October10 (22.68minutes). All16x3200-update fits and associated evaluations are
complete, including the aggregate tables and `plots/stability.png`. The next
allocation observed at19:45 is1033621; these completed runs belong to the earlier
1032567 allocation. No new training was launched during the results review.
[Findings and comparison limits](../experiments/ni_latent_sim_20261010/DERIVATIVE_STABILITY.md#completed-results-october-10).

The requested threefold collection uses `configs/ni_latent_sim/regimes3x_20261010.json`
and registry ID `ni-latent-sequences-regimes3x-20261010`. Its recipe and result
revision are `regimes3x-20261010-v2`; the unlaunchedv1 directory preserves the
smaller candidate-search preflight. New encoded inputs live under the IDS cache
`ni-latent-sim/regimes3x-20261010-v2`, with results in training storage under
`ni_latent_sim/regimes3x-20261010-v2`. The existing two-lane workflow classifies
sources, selects exactly9,600 new rows, encodes them, joins the original4,800,
then runs the16-arm training and evaluation factorial. Births and short physical
teachers are checksum-bound links to the unchanged base banks. No new independent
MD trajectories are generated. The deadline is8.5hours or five minutes before
the current allocation expires. [Scientific sampling definition](../experiments/ni_latent_sim_20261010/REGIMES3X.md).

The real-data preflight checked a persistent-liquid sequence centered on atom14445
and a growing-front sequence centered on atom979 in `ni-T1000-melt09`. All64
frames retained the same center identity at exactly zero relative position.
The front example's local crystalline fraction rose from0.525 to0.825;128 total
observations encoded to finite280-dimensional vectors. It performed no neural
optimizer updates or online diagnostic tracking. Its receipt is
`technical/fixed-center-audit.json` in the new run.

Threefold launch: October10 at20:46:02 CEST on node60 allocation1033621, with
GPU lanes0/1 and timeout supervisors4035047/4035049. The frozen bundle is
`technical/code`; plan identity is
`d854b29cdbd2a60382002bb092c7789a1d877395ae28f78b5e1833ba97f99c6c`.
The deadline is October11 at05:15:53 CEST. Both source-classification lanes start
with CPU label/cache work; GPU encoding and the scientific fits follow
automatically after the selection barrier. Only receipts from this launch
authorize managing these workers.

Dense-path launch: October11 at00:07:41 CEST, node60 allocation1033621, GPUs0/1, timeout supervisors4073262/4073264. Plan identity `1887abd73e484ae97d6857157fe3eb657d51ee28b3ab53ce3272ef691a6cc38a`. The frozen bundle is `technical/code`; the absolute deadline is 2026-10-11 08:37:31 CEST. Both workers prepare dense physical teachers before the six scientific fits and their evaluations. These launch receipts establish current-chat ownership.

Transformer queue launch: October11 01:34:15 CEST, allocation1033621/node60, two lane supervisors4081207/4081209. Frozen bundle `technical/code`, plan identity `b8c18088c1f37c2b672e109ca54052cb88d855b9d74f05abccf86f4fb454c6e9`. Both receipts confirm `waiting_for_previous_gpu_lane`; workers start independently once their MLP predecessor completes. Absolute deadline 10:04:06 CEST. The original MLP repair uses supervisors4080410/4080663 and its earlier08:37:31 deadline.


October11 resilience update (current-chat request): historical endpoint replay is
an advisory cross-device numerical diagnostic, never a teacher acceptance gate.
A failed or unavailable replay audit is recorded while independently valid new
paths remain usable. The existing audit found exact initial states/directions
and at most1.90e-6 normwise discrepancy in the endpoint Q/V response fields.

Dense-path campaigns now use a detached supervisor per GPU lane. It can adopt
only a PID identified in that campaign's current-chat launch receipt, leaving a
healthy worker untouched. Unexpected worker exits trigger at most two process
recoveries before the original absolute deadline. Stage receipts give training,
rollout scoring, birth prediction and physical-path scoring independent bounded
retries. Training resumes the same optimizer/RNG checkpoint and W&B ID; it does
not reduce the prescribed updates, cohort or precision. Birth feature export
can run independently of rollout scoring. A failed evaluation or plot does not
cancel another fit. Packet readiness is verified directly rather than inferred
from stale completion flags. Nonfinite model states and changed data/checkpoint
identities remain errors in the affected stage; rows are never silently dropped.

A fit that exhausts its retries can still have its best available validation-NLL
checkpoint evaluated, with completed/requested/selected updates recorded as
partial training. Full completion requires all requested stages. An incremental
`analyses/progress-v1/README.md` and `technical/live-status.json` expose available
results and failure receipts. Dashboard failures are logged without interrupting
computation. Exports produced after a code recovery use a separate metric
revision when the original numerical contract differs; existing tables and
frozen definitions are preserved. Supervision and recovery receipts are under
`technical/supervision/`; the original launches and deadlines remain unchanged.

Resilience deployment: October11 at01:46 CEST, same allocation1033621. MLP
supervisor timeout PIDs4085899/4085901 adopt healthy workers4080410/4080663;
transformer supervisor timeout PIDs4085937/4085939 adopt queued workers4081207/
4081209. Each campaign preserves its original launch and records the addition
in `technical/resilience-launch.json`, with frozen `technical/code-resilient-v1`.
All four supervision receipts were verified in `watching` state, with zero
restarts. Both GPUs remained active; three MLP models had all four evaluation
tables and a fourth had finished training. These are observations at deployment,
not a claim that the campaigns are complete. The previously rejected real path
20cc229669394e788bb4 was replay-audited successfully in diagnostic-only mode.
