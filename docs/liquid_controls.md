# Running the liquid sensitivity and relaxed-input queue

Use conda `pointnet-torch214`:

```bash
python -m src.research.liquid_predictability.control_queue submit --config configs/liquid_predictability/controls_relaxed_20260928.json
```

The submission freezes code, metric definitions, recipes, archived-cell inventory
and the original row intersection. CPU tasks prepare synthetic targets and paired
relaxed descriptors independently. Eight CPU lanes extract existing archived
quenches; no MD or minimization is launched. A dependent seal audits shared row
identity and both label definitions before any paired fit.

One GPU performs boosting, then a separate GPU numerical verification job gates
two MACE lanes. Each lane uses one GPU and isolated fit subprocesses. At most two
GPUs are occupied by this queue at a time. Four CPU baseline suites use at most
two concurrent tasks. Final evaluation/report jobs depend on all fits succeeding.
MACE fits have 12 blocks ×256 updates, batch/microbatch 256, cuEquivariance, BF16,
compiled spatial blocks and activation checkpointing. Geometry resides on GPU;
shared immutable descriptor banks are memory mapped and reused.

Ten MACE fits: seven synthetic sensitivity controls; full raw rich-feature
prediction; paired raw and relaxed rich-feature prediction. All scientific MACE
fits are online in teshbek/PointCloudMaterials with stable IDs. Boosting, mean,
ridge, linear probability and descriptor-MLP controls remain local. Numerical
verification never opens W&B. Training failures retain tracebacks and checkpoints;
do not silently disable tracking.

Seven synthetic datasets each have a prior and depth-4 all-feature GPU booster.
Four paired input×label protocols each repeat the existing ten descriptor arms
plus training mean, ridge and linear-probability baselines: 52 paired fits.
The existing raw-full boosting result is preserved; raw-paired fits are new.

Cache: `${storage:cache}/liquid-predictability/controls-relaxed-20260928`.
Results: `${storage:analysis}/liquid_predictability/controls-relaxed-20260928`.
`technical/launch.json` records job IDs; `technical/code/config.json` is the exact
resume recipe. Preparation checkpoints per archived source/frame, MACE every 64
updates. Resume the recorded failed stage from the frozen code directory.
Checkpointed-but-incomplete fitting exits unsuccessfully so reports never claim
an unfinished fit is complete. Historical run files are not changed.

Final report: `analyses/controls-v1/README.md`; source-paired tables, full feature
fidelity and generated-label oracle results sit beside frozen metric contracts.
`analyses/paired-coverage-v1` preserves all exclusions before the common cohort.
See [scientific design](../experiments/liquid_predictability_20260928/CONTROLS.md)
and [metric definitions](metrics/liquid_controls.md).

## Full-coverage extension

`python -m src.research.liquid_predictability.control_relaxation submit --config configs/simulation/liquid_full_relaxation_20260928.json`
freezes all required original source/frame cells, reuses completed quenches, and
submits missing fixed-cell FIRE tasks. Completion gates automatic submission of
`configs/liquid_predictability/controls_full_relaxed_20260928.json`: four full
paired boosting/baseline suites and two matched feature-learning MACE fits.
The seven synthetic assays already use all raw data and are not duplicated.
See [relaxation production](simulations/liquid_full_relaxation_20260928.md).

## Submitted 2026-09-28

- Existing-archive preparation: `1013261`; synthetic datasets: `1013260`
  (completed). Dependent seal/baselines/boosting/numerical-check/MACE/report:
  `1013262` through `1013267`.
- Missing full-cell relaxations: `1013277`, four allocations hosting sixteen
  independent MPI workers. Full-study submission dependency: `1013281`.
- The initial full-study launcher `1013278` was replaced before execution to
  preserve the physical project's storage/catalog resolution when submitting
  from frozen code. Relaxation workers retain their original code snapshot;
  `technical/study-code-v2` freezes the corrected downstream launcher.
- Local batch-256 forward/backward verification passed for the synthetic MACE
  and all-3536-feature MACE heads, with finite losses/gradients. Its receipt is
  `technical/early-numerical-check.json` in the existing-archive study. Relaxed
  numerical verification remains an explicit dependency before training.

Job IDs are execution records, not results. Consult each `technical/launch.json`
and Slurm state for live progress; final cohort sizes follow the shared relaxed
eligibility audit, not the pre-relaxation availability count.
