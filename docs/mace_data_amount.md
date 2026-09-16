# Native encoder data-amount study

Use conda `pointnet` and the existing `src.research.mace_velocity` command with
`data-prepare`, `data-smoke`, or `data-study`; recipe
`configs/analysis/mace_data_amount.json`. Source preparation reuses verified
sequence labels and rebuilds actual position/velocity clouds. It never supplies
old frozen embeddings to training. All data-size fits update MACE parameters.

Preparation cache: `${storage:cache}/mace-native-data-amount-20260916`.
Run: `${storage:archive}/analysis/mace_data_amount/independent-sources-20260916`.
STORE is used because WORK is currently at quota. Preserve cache `plan.json`,
`complete.json`, per-source identities, and run config/normalization/checkpoints.

The main run must execute on allocated H100 node53, job991900. A small detached
CPU-only Slurm controller holds the `srun --jobid=991900 --overlap --exact` client,
so it survives the node51 interactive allocation ending. The controller requests
no new GPU (`--gres=none` on P100, where CPU-only capacity was immediately
available; the CPU partition had a later predicted start). Retain the exact submission script, job ID, log and allocation/step
identity in `technical/`. The runtime checks H100, node and job identity. Budget
is limited to 6,900 seconds; benchmark rejects an overlong planned sweep before
launching fits. A time-budget failure preserves the current exact optimizer state
and is a partial run, not a successful completed study. No implicit resume is
implemented; investigate and define any continuation explicitly.

Read `technical/status.json` for progress and `technical/partial-results.json`
for completed-fit counts. `tables/quality.csv` contains all/low-order train,
validation and development-test metrics. `tables/source_quality.csv` retains
source-level quantities, and `tables/METRICS.md` freezes their definitions and
implementation hashes. `plots/learning_curves.png`/`.pdf` update after each fit.
Each `technical/nNNN-seedSEED/` contains `last.pt` (matched-update primary result),
`best.pt` (validation-selected), history, status and evaluated feature arrays.
Auxiliary heads and direction bases are saved for evaluation, but the exported
representation is the native MACE/velocity encoder output. Checkpoints also retain
normalization, exact optimizer/RNG state, selection and implementation identity.

Load a resulting native checkpoint with
`src.research.mace_velocity.data_amount_model.load_native_encoder(path, device)`;
then use the existing `mace_velocity.inference.embed_local_groups` helper on
aligned position/velocity halos. Loading needs the retained original foundation
model file and the checkpoint, with no training cache or prior adapted encoder.

Smoke outputs live under the main run's `technical/smoke/`; their four updates per
fit test execution only. They are not scientific learning-curve results.

See [scientific protocol](../experiments/mace_data_amount_20260916/README.md),
[exact metrics](metrics/mace_data_amount.md), and the
[glossary](research_glossary.md#independent-source-learning-curve).
