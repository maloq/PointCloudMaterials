# Operating the context information-recovery experiment

The scientific recipe is
[`configs/analysis/mace_context_recovery.json`](../configs/analysis/mace_context_recovery.json).
The original context cache and completed feature extractions are read-only inputs.
New scores, histories, checkpoints and logs are stored under the configured WORK
analysis root `mace_context_recovery/forecast-seed20260910-20260914`; compact results
and live training statuses are copied to the corresponding local `output/` run.

`recovery-readouts` completes the cached feature experiment. `recovery-verify`
fits common initial physical heads and checks combined-versus-separate readouts,
rotation/permutation invariance and direct-versus-replayed gradients on GPU.
`recovery-train` requires that verification and an explicit variant. Training
uses conda `pointnet` and the existing node57 allocation 992489. The two explicit
allocation plans are [GPU0](../configs/mace-context/recovery-gpu0.json) and
[GPU1](../configs/mace-context/recovery-gpu1.json).

A detached CPU Slurm controller starts the two GPU steps inside that allocation,
waits for both and then runs `recovery-summarize`. This does not allocate new GPUs
or cancel the user's allocation. GPU0 runs `dual_ssl`; GPU1 runs `dual_physics`.
The controller job ID and exact submitted script are recorded in the run's
`technical/detached-submission.json` and `technical/detached-controller.sh`.
Do not edit the submitted script, plans or training implementation while active.

Monitor `technical/train-dual_ssl/status.json` and
`technical/train-dual_physics/status.json` in the local output. Detailed command
logs are external under `technical/lanes/gpu0/technical/train-dual-ssl.log` and
`technical/lanes/gpu1/technical/train-dual-physics.log`. The controller log is named
`technical/controller-JOBID.log`. The source/config snapshots are retained in
the research execution records under `technical/executions/`.

Each training directory keeps `best.pt`, `last.pt`, `history.json`, and its exact
config and input hashes. Checkpoint protocol `mace_context_recovery_joint_v1`
contains model/head/optimizer state, fixed feature and target transforms, RNG
states and the last completed epoch. It is not a Lightning checkpoint. Restore
the original model first, then strictly load `model_state`; request combined
features through `context_features(..., return_center=True)` on a `halo_inner`
graph. Column order is 256 smooth-inner channels then 256 tracked-center channels.
The trained physical heads take standardized combined features. Ordinary ridge
readouts refit their own training-only standardization after encoder selection.

Training checks the configured allocation deadline before each new epoch and
retains completed-epoch state if insufficient time remains. An interrupted run
requires inspection and an explicit continuation attempt; commands do not silently
overwrite or restart an existing training directory. Evaluation extracts selected
embeddings, full boundary curves and temporal readouts; the controller collects
the completed results automatically. Forecaster retraining is a separate workflow.
