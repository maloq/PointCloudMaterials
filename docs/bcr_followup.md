# Eight-hour BCR follow-up queue

The [scientific protocol](../experiments/bcr_followup_20260922/README.md) prioritizes
frozen-code diagnostics and archived observed/relaxed structural comparisons before
three matched fresh-decoder fits. No encoder retraining, new simulation, hardware
benchmark, or final-test selection runs inside this queue.

Use conda `pointnet-torch214`. Active recipe:
`configs/bcr/followup_20260922/study.json`. Results:
`output/bcr/conditioning-audit-20260922/`; paired full-radius cache:
`${storage:cache}/bcr/relaxed-audit-20260922`. Machine routing remains in ignored
`machine.local.yaml`. The derived cache is registered in `configs/datasets.json`.

```bash
conda activate pointnet-torch214
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.bcr_followup.queue preflight --config configs/bcr/followup_20260922/study.json
python -m src.research.bcr_followup.queue submit --config configs/bcr/followup_20260922/study.json
```

Preflight runs BCR unit tests, replays real checkpoint features/corruptions, exercises
three disposable production-batch decoder updates, verifies the encoder is unchanged,
and extracts one paired full-cell neighborhood set. This bounded correctness check
is separate from training. Source selection is frozen at preflight, before submit.
Submission requires its passing receipt bound to code/config/data identities.

Submission requests one RTX6000PRO GPU, six CPUs, 40 GiB RAM and eight hours through
Slurm. It freezes executable source/config/metric definitions under `technical/code`;
subsequent workspace edits cannot change the running job. Job ID and script are in
`technical/launch.json` and `technical/audit.sbatch`; logs use `technical/audit-JOB.log`.
No terminal needs to remain open. Do not submit a duplicate launch while one exists.

The worker reserves seven minutes at the allocation end for checkpoint/report work.
Decoder and constant-code optimizers, streams and RNG states are saved; completed
feature/probe pairs are reused. Interrupted small residual probes rerun from their
fixed seed. Tables are exported after each stage with `tables/METRICS.md` and frozen
implementation hashes. `technical/queue-status.json` distinguishes running, complete,
checkpointed at time limit, and failed. Failures retain context and are not silently
skipped. Stop-on-error applies to bad ancestry, checksums, precision records or support.

To resume, use the saved `technical/audit.sbatch` with a new allocation after checking
that the recorded job has stopped (`sbatch PATH/technical/audit.sbatch`). The immutable
worker resumes completed stages; it does not need a new preflight or altered receipt.
The actual job's wall deadline bounds each invocation. Keep subsequent Slurm job IDs
with the original launch record. `queue report --config CONFIG` regenerates readable
tables from completed records without fitting. Do not modify a run's configuration
or implementation in place; choose a new output directory for a revised protocol.

Expected workload is four checkpoint exports, 60 ridge/residual probe pairs, two
frozen-decoder intervention banks, and three 10,000-update fresh-decoder fits. Earlier
RTX6000PRO frozen-encoder timings were about 0.37 s/update (about three hours for
30,000); exports, constant fits, readouts and archive checks use the remaining budget.
This is a planning estimate, not a guaranteed finish time. At the time limit, an
unfinished stage is clearly marked and can resume; no automatic lower-budget result
is substituted for a completed matched comparison.

For the 22 September launch, all other GPUs were occupied. Execution therefore
starts immediately as detached Slurm step `1003651.15` on node58, with about
6 h 28 min left in that allocation. Job `1003781` was reduced to a 91-minute
continuation dependent on the existing allocation ending. The combined allocation
budget from launch is below eight hours (28,759 s); checkpoint reserves further
reduce training time. `technical/launch.json` records both identities and the exact
budget. `technical/audit-current-1003651.log` is the active log. The detached wrapper
cancels the continuation if the current worker completes or fails; a time-limited,
checkpointed worker permits continuation. Do not cancel the interactive allocation
to end a completed experiment; only the dedicated experiment step/job is managed.
