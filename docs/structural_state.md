# Running the fixed structural-state screen

## Repaired queue on the current node, 23 September

Use `configs/structural_state/repaired_20260923.json` with the current code;
[scientific changes](../experiments/structural_state_20260923/README.md).
The historical recipe below requires its original frozen code. The new queue
reuses the already registered structural-state-screen-20260922 cache and verifies
its checksums; it does not prepare or register a second copy of the data.

In conda pointnet-torch214, from the repository root:

    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m src.research.structural_state.queue preflight --config configs/structural_state/repaired_20260923.json
    python -m src.research.structural_state.queue launch-local --config configs/structural_state/repaired_20260923.json

`launch-local` validates the current Slurm allocation deadline, checks an exact
preflight identity, freezes code and starts detached worker processes using only
the visible allocated GPUs. GPU0 runs A then C; GPU1 runs B then D. Each arm trains
and evaluates before the next arm on that GPU. The current allocation is1005857
on node61 (two RTX PRO6000 GPUs, expiry24 September05:15:41 local time). No new
Slurm allocation is submitted. Closing the terminal does not stop the processes;
ending the Slurm allocation does.

Output: `output/structural_state/repaired-20260923`. Worker PIDs, GPU assignments,
commands and code paths are in `technical/launch.json`; logs are
`technical/local-gpu0.log` and `technical/local-gpu1.log`. Arm status files record
training/evaluation/complete/failed/checkpointed. Table collection is serialized
after each arm, including partial deadline checkpoints. The final primary state
is always update4096. Do not edit frozen code or reuse old checkpoints.

To resume after a stopped allocation, run the recorded `serial --config ...
--arms ...` command from its frozen checkout under a new allocation, with the
recorded environment and one GPU per process. Completed stages are reused and
partial encoder fits restore optimizer, fitting calibration and RNG state.
An interrupted tiny evaluation probe restarts from its fixed seed. Inspect the
failure reason before resuming a failed, rather than timed-out, run.

## Historical 22 September execution

Use conda pointnet-torch214. The scientific protocol is in
[the study](../experiments/structural_state_20260922/README.md), and the recipe is
[screen_20260922.json](../configs/structural_state/screen_20260922.json).

The preparation stage verifies existing paired-patch and original-MD assay hashes,
then caches untruncated graphs and fixed targets under
${storage:cache}/structural-state/screen-20260922. The derived collection is
registered as structural-state-screen-20260922. No molecular dynamics runs here.

From the repository root:

    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python -m src.research.structural_state.queue prepare --config configs/structural_state/screen_20260922.json
    python -m src.research.structural_state.queue preflight --config configs/structural_state/screen_20260922.json
    python -m src.research.structural_state.queue submit --config configs/structural_state/screen_20260922.json

Preflight tests native/cached outputs and gradients, symmetries, full/microbatch
losses, checkpoint resume, fitting-only normalizers and source/Slurm isolation.
It also performs three disposable production-size updates for every arm and
checks the real onset and neighbor evaluation paths. It is a separate correctness
command, never an automatic benchmark/training stage.

Submission freezes source/configs/metric definitions under the run's technical/code
directory. Four independent Slurm GPU jobs each train one arm and then evaluate
its frozen exports. All GPU jobs exclude node58 as requested; they use the
configured GPU partition list and one GPU, four CPUs and 48 GiB host memory each.
Each job has an eight-hour wall limit. A one-hour CPU collector depends on all
four jobs ending, including failed or checkpointed jobs, so partial results remain
visible instead of leaving an impossible afterok dependency pending indefinitely.

The run root is output/structural_state/screen-20260922. Inspect:

- technical/launch.json for jobs, immutable scripts and frozen code.
- technical/ARM-status.json and ARM-JOB.log for running/checkpointed/failed stages.
- technical/fits/ARM/last.pt for optimizer, sampler and RNG state.
- technical/fits/ARM/initial.pt, best.pt and features.npz for retained references.
- technical/evaluation/ARM for probe weights, predictions and calibration details.
- tables/ for documented physical, neighbor, onset and paired-comparison CSVs.

Geometry is immutable in this protocol, so parameter-free radial features and
spherical harmonics are cached once on the GPU. Graph batches use index gathers;
no trajectory IO, neighbor searches or descriptor calculation occurs per update.
Both native tensor products and contractions use cuEquivariance. Arithmetic is
float32 with TF32 disabled. Different scheduled hardware is recorded per fit;
one seed does not measure GPU-dependent optimization variability.

The worker reserves seven minutes before Slurm expiry (five minutes from the
allocation helper plus two minutes for a checkpoint). Training saves optimizer,
stream, relational scale and RNG state. Completed probes are reused; a partial
small probe repeats from its fixed seed. Failures stop the job and preserve a
traceback. No reduced scientific budget is silently substituted.

After an expired job has stopped, resume its exact immutable ARM.sbatch with
sbatch; do not rerun submit, which rejects duplicate campaign launches. Record
the new job identity and rerun the saved collector after all resumed jobs end.
Change neither config nor code inside a frozen run. The CPU collector can also
be invoked from that frozen checkout with queue collect --config CONFIG.

## 22 September submission

A100 preflight job1004308 completed successfully on node04 after the RTX correctness check. Scientific fits: A1004317, B1004318, C1004319, D1004320; dependent CPU collector1004321. Eligible partitions are A100,H100,RTX6000PRO, excluding node58. All run detached from an immutable code snapshot. The dataset registry was refreshed into the run’s technical/dataset-registry directory to avoid concurrent writers to the shared registry.

## Distance/future factorial, 23 September

The [two-seed scientific protocol](../experiments/structural_state_future_20260923/README.md)
reuses the verified cache and existing queue. Use conda pointnet-torch214 and the
environment exports above. Run separate correctness checks with
`queue preflight --config configs/structural_state/future_metric_seed20260923.json`
on visible GPU0, and the seed20260924 recipe on visible GPU1. Then launch:

    CUDA_VISIBLE_DEVICES=0 python -m src.research.structural_state.queue launch-local --config configs/structural_state/future_metric_seed20260923.json
    CUDA_VISIBLE_DEVICES=1 python -m src.research.structural_state.queue launch-local --config configs/structural_state/future_metric_seed20260924.json

Both use the current node61 allocation1005857, one detached serial four-arm queue
per GPU. Do not repeat a launch that already has technical/launch.json. Each seed
has its own immutable code snapshot, fit/evaluation receipts and local-gpu0.log
(the name is local to its single-visible-GPU queue). Existing resume instructions
apply using the recorded frozen serial command and GPU assignment.

Seed outputs: output/structural_state/future-metric-seed20260923 and
future-metric-seed20260924. The combined output is
output/structural_state/future-metric-20260923. After each arm, the worker exports
seed tables and updates the combined report, which includes only fully evaluated
seeds and explicitly records pending seeds. The same collector can be called as:

    python -m src.research.structural_state.factorial_report --config configs/structural_state/future_metric_campaign_20260923.json

New encoder checkpoints are technical/fits/ARM/{initial,last,best,step-1024,
step-2048,step-4096}.pt in each seed output. Auxiliary future-target baseline
coefficients/scalers are retained in future-targets.json and checkpoints.
Combined tables include paired onset uncertainty, physical/neighbor errors,
predeclared mechanism rules and source comparisons; plots/onset-factorial.png is
created after the first complete seed. The dataset registry availability snapshot
is in the combined output's technical/dataset-registry. No cache is modified.

### Factorial collector recovery, 23 September

The initial cross-seed collector rejected one initial-export value beyond its
pointwise CUDA tolerance after seed20260923 completed. Because the serial worker
called that collector after each arm, seed20260924 stopped after E; F had not
started. All seven completed fits and evaluations were valid and preserved.
F was launched through `queue worker --arm F-distance-future` from its original
frozen checkout, on GPU1 in allocation1005857. Its command/PID are recorded in
technical/recovery-launch.json and its output in recovery-F-distance-future.log.

The collector now verifies bitwise initial encoder state, separately audits
normalization/head prediction roundoff, and compares exports in aggregate and
maximum absolute units. Four repeated native forwards of the identical checkpoint
confirmed the numerical scale; the receipt is in the combined run's
technical/cuda-initial-repeat-audit.json. Training code, checkpoints, predictions
and scientific success criteria were not changed. The original partial export's
metric definitions are preserved under technical/initial-report-version.
After F finishes, run the original frozen seed's `queue collect`, followed by the
current corrected factorial_report command above. Do not rerun the old serial
worker: its unchanged collector still contains the original assertion.
