# Initial-velocity response pilot,4 October2026

Current new production uses the
[CUDA MEAM 0.5 ps recipe](../../configs/simulation/response_velocity_cuda500_float32_20261004.json):
100/500 fs targets, the full original 90/15/30 source roles, four fixed sampled
velocities per geometry, and four model arms times three training seeds.
The user explicitly reauthorized runs longer than 2–3 hours. The new compiled
CUDA force/Hessian-vector interface replaces the force backend while retaining
the fixed-velocity BAOAB protocol. Force arithmetic, states, tangents, noise,
periodic boxes and observables use float32, explicitly requested by the user.
The unchanged backend retains only its scalar energy reduction in float64.
Only prepared cells are reused; earlier precision banks remain separate. Completed engine validation
and implementation hashes are pinned; no additional MEAM checks are scheduled.

Two independent single-GPU Blackwell jobs **1020884** and **1020885** allow
use of GPUs on different nodes. Their receipts are in `technical/launch.json`.
A detached helper in allocation **1020718** starts
immediately. Shared locks, archive publication, online training, all64 evaluation
and resumable checkpoints retain their prior contracts. Each queued job has
24 hours and at most two automatic resumptions. Output root:
`output/response_velocity/cuda-meam-500fs-float32-20261004-v2/`.

On 4 October at 18:15 CEST, the user offered both free GPUs in existing
node60 allocation **1020716**. Two detached workers joined the same frozen
float32 campaign through step **1020716.2**, one GPU per task, with shared
fit locks. Collection was already sealed (540/540 queries); these workers
train and evaluate remaining model/seed combinations. Their launch receipt is
`technical/helper-node60-launch.json` and their logs are
`technical/worker-node60-{0,1}.log`. The interactive allocation itself is not
owned by this response campaign and must not be cancelled by its workers.

The mixed-precision v1 attempt was stopped on the user's float32 instruction:
only its own jobs **1020874**, **1020875** and helper PID **4031143** were
stopped. Its frozen recipe, completed batches and stop receipt are retained;
its partial targets are not included in the float32 bank.

```bash
python -m src.research.response_velocity.queue launch --config configs/simulation/response_velocity_cuda500_float32_20261004.json
```

The earlier explicitly reduced
[three-hour recipe](../../configs/simulation/response_velocity_three_hour_20261004.json)
continues on its recorded two-GPU job **1020829**, with its original deadline.
Its local helper was stopped to give the current node's GPU to the CUDA run.
The user stopped the original large response-training run. Only jobs owned by this chat
are subject to that instruction; simulation and other-chat jobs are excluded.

The small pilot retains full periodic MEAM and all three physical horizons,
but uses 12/4/4 original train/selection/test ancestors, two velocity fields,
two shots per bank, one training seed and 200 updates (batch24/microbatch8).
All four arms and the unchanged all64 clustering rows are retained. It is a
feasibility pilot with limited statistical power, not the full comparison.
Existing matching two-shot batches retain their original producer records.
All workers share a three-hour deadline from launch, with no automatic requeue.
Use the same launch command below with the three-hour recipe path.
Current results are under
`output/response_velocity/initial-velocity-20261004-three-hour-v1/`.
The longer campaign and its frozen configuration are retained as stopped history.

[Recipe](../../configs/simulation/response_velocity_20261004.json) ·
[scientific protocol](../../experiments/response_velocity_20261004/README.md) ·
[metrics](../metrics/response_velocity.md).

Reuses the original135 unrelaxed local80 observations and fixed source roles.
Four velocity fields are newly sampled and saved per geometry. Initial velocities
remain fixed within each conditional shooting bank. The validated GPU MEAM
implementation supplies float64 forces and four tangent directions; horizons
are0.1/0.5/1ps along each single1ps trajectory. No coordinate histories, new
quenches or changed parent trajectories. No claim of native NPT continuation.

All scientific implementation is in `src/research/response_velocity/`.
Use conda `pointnet-torch214` and the standard trusted-e3nn environment flag.

```bash
python -m src.research.response_velocity.queue launch --config configs/simulation/response_velocity_20261004.json
```

Launch freezes source and metrics, authenticates online W&B, requests two GPUs
and starts a detached helper on the current allocation. Workers share query and
fit locks. The completed student derivative/symmetry checks are reused. On
4 October the user explicitly requested proceeding without further MEAM checks;
the execution receipt records the skipped checks and previously completed evidence.
Production uses full native periodic cells. Main workers may requeue
twice at allocation expiry; the interactive helper stops without requeuing its shell.

Each completed batch is saved on SCRATCH then checksum-published to STORE before
proceeding. Fixed input positions/velocities and response features are preserved;
no new trajectory position arrays are exported. Partial batches are recomputed
on resume. Data sealing waits for every query, including helper-owned work.
After sealing, four model arms times three seeds train with batch256/microbatch8.
They are followed automatically by all64 static clustering with explicitly
sampled velocity inputs. Predictive results report each horizon separately.
Comparison tables and completion receipts publish only existing results.

Result root uses `output/response_velocity/initial-velocity-20261004-periodic-v4/`;
simulation registry entry is `response-velocity-20261004`. Native MD velocities,
history models and downstream nuclei-birth fits remain separate future work.

Launched4 October2026: two-GPU job1020753 and detached node58 helper in
allocation1020718. Revisionv2 corrects interrupted-gate resume handling; the
completed24A numerical queries and derivative checks retain their original
producer identity and are reused by checksum. No scientific training was run
under the earlier preflight revision.

Periodic revision v3: the 24/36 Å and 36/48 Å comparisons failed at 0.5/1 ps;
v2 stopped before production and pending coordinator 1020753 was cancelled.
Full native periodic cells replace the open environments. The completed periodic
checks remain recorded; further MEAM validation was stopped at the user's request.

Revision v4 fixes empty force-free payload handling for held-out derivative-only
shots. It reuses the exact prepared parents, student check, and decision to skip
remaining MEAM checks. The MEAM implementation and scientific protocol are
unchanged. Coordinator 1020766 and its v3 helper were stopped before training;
their frozen artifacts remain available. Current jobs and worker progress are
recorded in the v4 result root under `technical/`.

Current launch: two-GPU coordinator **1020769**, with detached helper PID **4004544**
in allocation **1020718** on node58. Training and evaluation remain queued behind
collection; the launch alone is not a completion claim.

## Future-outcome similarity

The float32 0.5 ps campaign completed all twelve fits and all64 clustering
evaluations on 4 October. The additional retrieval diagnostic uses only its
sealed test response bank and validation-selected checkpoints. Recipe:
`configs/analysis/response_future_similarity_20261004.json`.
Output: the campaign's `analyses/future-similarity-v1/`.
Metric definitions: [response future similarity](../metrics/response_future_similarity.md).

Run `python -m src.research.response_future_similarity prepare --config configs/analysis/response_future_similarity_20261004.json` in the live repository.
For extraction, use the analysis producer's absolute file path with stage
`extract --config ANALYSIS/technical/config.json --lane 0` or `--lane 1`.
Set `PYTHONPATH` and working directory to the campaign's `technical/code`,
`PCM_PROJECT_ROOT` to the repository, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`,
and one allocated `CUDA_VISIBLE_DEVICES` per lane. The extraction imports
the exact frozen model, checks checkpoint hashes and reproduces saved
predictions before saving the small held-out representations.

After both extraction receipts are complete, run `analyze` with the same
configuration from the live repository, then `tracking` to append diagnostic
summaries to the existing scientific W&B run IDs. No evaluation runs are
created. The two extraction processes were launched in the user-offered
node60 allocation1020716; their PIDs and commands are saved in
`analyses/future-similarity-v1/technical/extraction-launch.json`.
