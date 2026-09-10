# Variant C, TDA from epoch 1, 24 epochs — 2026-09-09

## Recovery — September 9, 15:18 Paris

The initial queue failed during data preparation, before any training update.
Al 175 ps frame 24 reached the 10,000-iteration limit at fmax 0.13385 eV/Å.
The first 17 completed paired frames are retained. The failed frame and prior
status/configuration files are archived under `recovery_20260909_1518/` in the
first run output; original registry execution records remain intact.

The current configuration increases the CG budget to 100,000 iterations,
500,000 force evaluations and a 3,600-second per-frame timeout. The generating
potential, fixed cell and 0.01 eV/Å convergence requirement are unchanged.
Previously converged frames remain valid and are reused. No unconverged frame
is accepted. Both queues are detached on current H100 allocation 986459,
ending September 10 at 06:00:55 Paris, with a 05:30:55 safety deadline.
The 12-epoch/TDA6 workflow runs first, followed by the fresh 24-epoch/TDA1
workflow after its training and analysis finish. Online W&B initializes at
training start; no training result is claimed during data preparation.

The original launch specifications below record the first attempt. Recovery
uses the maintained runner and this explicit specification:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_thermal80_tda1_24ep_20260909/recovery_run_spec.json
```

Live queue status stays in the original output root. Recovery launch and
execution metadata are stored separately in `recovery_20260909_1518/`.
These added specifications and notes are experiment records; generated retry
logs and verification files are output artifacts. No new runner was introduced.

Question: does longer training with relaxed-topology supervision from the first
update improve thermal consistency, topology and spatial structure together?

This is a fresh start from the same small MACE MLIP weights as the
[first variant C run](../mace_thermal80_20260909/README.md), using its identical
paired cache. Full-cell relaxation and TDA preparation are reused. Both TDA
activation and total training budget change, so this is not an isolated ablation
of TDA onset. Physical protocol, source limitations and split caveats are those
documented in the first run.

Training uses 26,624 anchors per epoch (Al 16,384, Mg 8,192, Ta 2,048),
1,024 validation anchors, and the same six hot/relaxed training views of 80 atoms.
Batch size 1,536, microbatch 512, 18 updates/epoch: 432 updates over 24 epochs,
638,976 anchor exposures and 3,833,856 training-view exposures.

Fixed spatial/temporal VICReg and hot/relaxed consistency remain unchanged.
All six views predict relaxed TDA from epoch 1, including the first update.
Encoder/head peak LR remains 1e-4/1e-3, with per-update cosine decay across
24 epochs, one-epoch encoder warmup and half-epoch head warmup. Both optimizers
start at 5% of peak LR. Select lowest validation loss among epochs 1–24.
Online W&B: `PointCloudMaterials`, run ID `therm1e24s09`.

## Reproduction and scheduling

From the repository root in conda environment `pointnet`:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_thermal80_tda1_24ep_20260909/run_spec.json
```

The maintained queue waits for the first C controller to complete training,
frozen probes and static analysis, and exit. The dependency records its PID
and process start identity. It then runs the real-MACE preflight with the
configured epoch-1 boundary, trains afresh, and runs frozen probes. Standard
encoder-only static Al analysis follows if allocation time permits.

Allocation 984861 on node53 ends September 9 at 05:55:06 Paris. This queue
reserves five minutes, giving a 05:50:06 deadline. Training and probes require
an estimated 5,500 seconds remaining (latest estimated start 04:18:26, after
preflight). If that budget is unavailable, the queue records
`pending_allocation_time` and does not launch a shortened training.
Analysis additionally requires 1,800 seconds remaining; otherwise its status is
`analysis_pending_allocation_time`. Estimates are not completion guarantees.

Outputs: [queue status](../../output/mace_thermal80_tda1_24ep_20260909/status.json),
`output/mace_thermal80_tda1_24ep_20260909/runs/thermal/`, and
[tests](../../output/mace_thermal80_tda1_24ep_20260909/tests.log).
Results are pending. The first run owns the shared data cache; retain it until
both workflows finish.

## Code ownership and checks

This directory contains experiment records and configurations, with no new
runner. Maintained trainer metadata and GPU preflight now follow the configured
TDA epoch rather than hardcoded epoch-six text/checks. The existing activation
test covers epochs 1 and 6, including optimizer state, head updates and cosine
decay to the final LR. GPU preflight runs serially after the predecessor.
Generated logs, launch metadata and test results are disposable output artifacts.

Launched detached at 01:40 Paris; queue controller PID 3661957.
Seven focused checks passed; controller session, dependency identity and 16-CPU
affinity verified. [Launch verification](../../output/mace_thermal80_tda1_24ep_20260909/launch_verification.json).

Recovery verified at 15:21 Paris: the failed Al frame converged after 11,723
iterations in 158 seconds, fmax 0.0066689 eV/Å. Paired extraction and verified
float16 conversion completed; preparation advanced to frame 40. Eleven focused
training/numerical tests passed. Both detached queues remain active. GPU
preflight and actual training are still pending completion of the paired cache.
[Recovery evidence](../../output/mace_thermal80_20260909/recovery_20260909_1518/recovery_verified.json).
