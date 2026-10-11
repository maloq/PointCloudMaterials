# Saved-state nucleus recovery

Recipe: `configs/simulation/nucleus_recovery_20261004.json`.
Implementation: `src/research/nucleus_recovery.py` and
`src/research/nucleus_recovery_md.py`. Uses the existing validated FastAl CUDA
MEAM and native NPT adapter; this workflow does not introduce a potential gate.

Run in conda `pointnet-torch214`:

```bash
python -m src.research.nucleus_recovery prepare --config configs/simulation/nucleus_recovery_20261004.json
python -m src.research.nucleus_recovery launch --config configs/simulation/nucleus_recovery_20261004.json
python -m src.research.nucleus_recovery summarize --config configs/simulation/nucleus_recovery_20261004.json
```

Launch freezes code, potential identities and the plan, then detaches two
workers onto visible GPUs 0 and 1. Workers claim per-continuation file leases;
completed branches are immutable. Exact observations checkpoint Q,p and torch
CUDA RNG state for BAOAB, and native binary restart plus Q,V for NPT. Both
workers pause before the current allocation deadline with 15 minutes reserved
for publication. Resume with the **frozen** module/config on a new allocation;
do not replace its code. Launch receipts identify owned PIDs; no other jobs
or the containing allocation are stopped or modified.

Primary coverage is all 15 isolated births in the existing original test
cohort (11 sources). Starts are 0.75, 3, 6 and 12 ps before establishment and
0.75 ps before the earliest eligible local appearance. Each ends 3 ps after
the original persistence confirmation. Original settings NPT is compared with
four independent future-noise BAOAB continuations at source temperature and
450 K. Identical sourceT=450 K cases share their four executions. Temperature
is a physical thermostat parameter only; there is no encoder or predictor.

Two retained float32 shooting events add offsets 0.3, 3 and 6 ps, four matched
noise seeds, and original/rounded-Q/rounded-V/rounded-QV states. They retain
their native 0.3 ps observation grid. This gives 81 saved initial states and
671 continuations. Near-establishment starts receive priority, then all longer
offsets; order is fixed without using replay outcomes.

New MD lives under SCRATCH `simulations/nucleus_recovery-saved-state-20261004`.
Initial states, precise checkpoints, completed trajectory conversions and
analysis records archive under STORE `simulations/nucleus_recovery-saved-state-20261004`.
Research tables, plots, logs and ownership receipts live under
`experiments/nucleus_recovery/saved-state-20261004`. No W&B run is created for
this frozen diagnostic. [Metric definitions](../metrics/nucleus_recovery.md).

This is one declared Al sub-study, not completion of the shared Al pool.
Other original births, dense 0.1/0.01 ps descendants, other shooting campaigns,
million-atom data and relaxed views remain outside this recovery queue.
The wider Al birth catalogue and MD simulations are separate jobs.

October 5 resume: the original allocation ended at 604/671 published branches.
One additional fully integrated and analyzed branch was recovered by verifying
and publishing its existing float16 conversion (605/671); its earlier export
failure remains in `stage-export-failure.json`. The user authorized node60
allocation 1022218. `technical/resume-20261005.json` records the owned detached
GPU 0 worker, followed by its handoff to the control campaign. Its new frozen
snapshot differs only in NFS source-retirement handling; simulation and metric
hashes remain unchanged. The full-state/bath-reset ablation is still proposed,
not part of this resumed queue.

The transient shared-scratch permission failure at 14:09 interrupted the resumed
worker. Original access and checkpoints returned by 14:12. The replacement
node60 worker resumes the same unfinished task from its checkpoint; its ownership
receipt is in the control bundle's
`technical/resume-after-scratch-20261005/launch.json`. Completed archives and
historical failure records are preserved, with no change to the scientific plan.
