# Liquid predictability study execution

Recipe: `configs/liquid_predictability/al64_20260928.json`. Implementation:
`src/research/liquid_predictability/`; [scientific protocol](../experiments/liquid_predictability_20260928/README.md).
Use conda `pointnet-torch214`. No automated test suite is added.

Outputs resolve to `${storage:analysis}/liquid_predictability/al64-20260928`.
Physical-descriptor and clearance preparation lives outside the repo at
`${storage:cache}/liquid-predictability/al64-20260928`. The original sealed geometry
is read in place. Frozen encoder features use the lease-protected six-entry LRU
at `${storage:cache}/context-features`; checkpoints and predictions are not caches.

Run the `preflight` stage with `--arm joint_mace` locally before launch. It performs
one real full-batch forward/backward without an optimizer or W&B run. Save its
configuration hash, finite-gradient result and batch in `technical/preflight.json`.
Physical preparation can be checked with `prepare --source SOURCE_ID`; it replays
sealed targets and actual crystal visibility on that source.

`launch` freezes source, metric contracts and configuration. It submits a CPU array
(twenty sources per task, up to eight tasks), then sealing, then physical profiles
and local distribution/descriptor fits. Two detached GPU lanes wait for sealing
and the previous LCD controller to exit, avoiding concurrent GPU ownership. Each
GPU lane processes its declared arms sequentially in isolated subprocesses.
Slurm continuation tasks run after the current allocation ends and are cancelled
when their local lanes complete. Failures leave explicit state/tracebacks.

Stages: `prepare`, `seal`, `profiles`, `cpu`, `worker --lane 0|1`, `fit --arm NAME`,
`compare`, `preflight --arm NAME`. Always resume using `technical/code/config.json`
and the frozen working directory, not a changed repository implementation.

Joint encoder fits use online W&B with stable IDs, readable model names, distance
NLL, VCReg, pre-clipping gradient norm, both learning rates, validation RMSE and
20/32/48-Å Brier scores. Final metrics attach to that same run. Distribution fits,
descriptor controls, the frozen readout and preparation remain local.

Each arm saves `technical/best.pt`, exact-resume `last.pt`, all epoch checkpoints,
training/validation JSONL, prediction-context records and identity receipts.
Evaluation writes `analyses/predictability-v1/{tables,technical}`. The study writes
`analyses/physical-profiles-v1` and, after all arms finish, `analyses/comparison-v1`.
The latter includes paired source intervals and the predeclared practical-effect
assessment. Metric CSVs carry frozen definitions and hashes.

The `complete.json` training receipt is distinct from evaluation completion.
Resuming a completed scientific fit exports missing metrics through the existing
W&B API identity, without opening another training run. Source subsets are
deterministic and their exact IDs are recorded; no held-out role is resplit.
