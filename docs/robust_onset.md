# Robust onset encoder queue

Scientific rationale, literature and arm definitions are in the
[protocol](../experiments/robust_onset_20260924/README.md).
Use conda `pointnet-torch214`. No new MD or minimization is performed.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.robust_onset.queue preflight --config configs/robust_onset/screen_20260924.json
python -m src.research.robust_onset.queue submit --config configs/robust_onset/screen_20260924.json
```

Preflight checks production batches and encoder gradients for all eight arms,
both readouts, the noisy input banks and full-cohort ranking replay. It is a
separate correctness stage; no hardware benchmark runs inside training.
Submission requires passing preflight for the exact source/config/data identity.
The submission snapshots source, configs, metric definitions and tests before
launching a Slurm array. Later workspace edits cannot change the queued jobs.

Eight independent array tasks, at most two GPUs concurrently, four hours per
task, one seed. Eligible partitions: RTX6000PRO, H100, A100, L40S. Maximum reserved
training/evaluation budget is 32 GPU-hours; with two slots continuously available,
at most 16 hours of task time, plus scheduler waiting. Jobs exit early on completion.
This is a ceiling, not a measured ETA. Each task trains then runs physical/event,
noise and dense trajectory evaluations. CPU collection runs after the array,
including after failures, and explicitly lists incomplete arms.

The run's `technical/launch.json` records actual job IDs. Per-arm status and Slurm
logs live under `technical/`, fits under `technical/fits/`, and evaluation arrays
under `technical/evaluations/`. Tables and definitions are under `tables/`.
Outputs live on WORK, augmentation graphs on IDS; paths resolve through
`machine.local.yaml`. No H200 endpoint is assumed available.

The same frozen worker can resume from `last.pt` with explicit `--arm NAME` inside
a new allocation; RNG, optimizer, initial calibration, selected best state and
source identity are retained. A timeout checkpoints and exits with incomplete
status; it does not silently lower the update budget or label a partial fit complete.
Do not resubmit from edited workspace code into the original output. Use the
frozen `technical/code` tree or a fresh experiment directory.

Collect without resubmission:

```bash
python -m src.research.robust_onset.queue collect --config configs/robust_onset/screen_20260924.json
```

The AP uncertainty intervals are whole-root sampling intervals. One-seed runs do
not estimate seed uncertainty. Historical development sources are reused, so any
winner needs confirmation on a larger separate cohort.
