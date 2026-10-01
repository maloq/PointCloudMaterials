# Packed-data TDA / VCReg study

## Full-data direct-VCReg combination

The 30 September follow-up combines the structured heads and block normalization
with original-strength VCReg on exported embeddings. Its recipe is
[`rich_tda_block_direct_full_20260930.json`](../configs/liquid_predictability/rich_tda_block_direct_full_20260930.json).
It starts from scratch on the exact 1,056,768 packed RH2 patches, with batch 8,192,
20 epochs, width 256/depth 3/correlation 3/angular 3/export 256 and peak LR .01. No
topological-distance term is added. It preserves the label-free validation selector,
fixed Al64 test, geometry-only inputs and online W&B.

This new combination was not screened in the seven pilots; it is a hypothesis
suggested by their separate findings. Both regularization placement and covariance
normalization differ from the automatic winner. The already-running automatic
winner (job 1015284, projector/per-channel VCReg with block-scaled reconstruction)
continues as a full-data reference. No prior run/checkpoint is overwritten.

Submitted as independent one-GPU Slurm job **1015877** on 30 September, with a
24-hour allocation request; initial status is pending GPU resources. Online W&B
starts when the training worker starts. The full-batch numerical check had finite
loss/gradients through all three interaction layers and used 35.58 GiB peak
allocated GPU memory. Its diagnostic weights were discarded. The launch and
verification receipts are in the new run's `technical/` directory.

After local numerical verification of the actual recipe, submit with:

```bash
python -m src.research.liquid_predictability.rich_objective_study submit-run --config configs/liquid_predictability/rich_tda_block_direct_full_20260930.json
```

`submit-run` freezes code/config and launches one GPU through Slurm. The new output
is `${storage:training_storage}/liquid_predictability/mm-tda-block-direct-full-20260930`.
`train-run --config FROZEN_CONFIG` resumes the exact full-data run if interrupted.
All objective and metric formulas remain the [existing study definitions](metrics/rich_tda_objectives.md).

Use conda `pointnet-torch214`. The [recipe](../configs/liquid_predictability/rich_tda_pilots_20260930.json)
reuses the stopped RH2 sample IDs and raw descriptors. Packing goes to IDS via
`loader.packed_cache`; outputs, checkpoints and logs go to STORE through
`${storage:training_storage}`. No new simulation or descriptor computation.

```bash
python -m src.research.liquid_predictability.rich_objective_study prepare --config configs/liquid_predictability/rich_tda_pilots_20260930.json
python -m src.research.liquid_predictability.rich_objective_study submit --config configs/liquid_predictability/rich_tda_pilots_20260930.json
```

Submission requires a matching `technical/numerical-check.json` from local
verification of the new objectives and declared full batch, without W&B. This is
separate from scientific fitting, not a hardware benchmark in every training job.
Prepared arrays retain checksums and original row/target identities. They occupy
about 2.7GiB for the full set plus 0.27GiB for the pilot; fitting loads one into host
RAM and prefetches pinned batches. A global permutation replaces shard-block I/O.

The queue freezes source/config/metric definitions. Seven single-GPU pilot jobs
run online in the existing W&B project. Promotion depends on successful pilots,
selects their lowest common validation NLL, writes `technical/promotion.json`,
and starts a fresh full-data model. The final run exports fixed Al64 calibration/
test predictions and updates its W&B summary. Pilot jobs never evaluate test.
Local tables use metric family `rich_tda_objectives` with implementation hashes.

`--first-external` reserves arm 0 for a detached step in an existing allocation;
the other six are queued one at a time, so at most two fits run concurrently.
Run that step from `technical/code` using its frozen `config.json` and
`pilot --arm 0`. Promotion still verifies all seven completion markers; an external
failure cannot silently promote a partial comparison. Without this flag, the
array includes all seven with at most two concurrent workers.

Training locks exclude duplicate workers. Resume a checkpoint with the identical
frozen `pilot --arm INDEX` command. Epoch order, AdamW state and cosine cursor
resume; Slurm expiry does not shorten the declared schedule. Inspect
`technical/launch.json`, each `pilots/ARM/technical/state.json`, training/validation
JSONL and W&B `technical/wandb/fit/run.json`. Scientific definitions and limitations
are [here](metrics/rich_tda_objectives.md).

## Submitted 30 September 2026

Seven pilots use 105,677 fitting patches (59,487 Al, 4,385 Mg, 36,041 Ta,
5,764 Ti) from all 117 original training sources. The selected full-data fit uses
exactly 1,056,768 patches. Pilot and full schedules are 20 epochs each.

- Arm 0 runs detached on node60 allocation 1015174, tmux socket `pcm-rich-tda`,
  session `first-pilot` supervised from node39.
- Array 1015283 runs arms 1–6. Its initial throttle is 1 while arm 0 occupies the
  existing GPU; after arm 0 succeeds, the wrapper raises the throttle to 2.
- Job 1015284 depends on successful completion of the array, verifies all seven
  pilot completions, selects the best validation NLL, then trains/evaluates the
  full-data recipe from scratch.

Outputs and receipts are under
`${storage:training_storage}/liquid_predictability/rich-tda-vcreg-pilots-20260930`.
Initial online runs:
[embedding-pair](https://wandb.ai/teshbek/PointCloudMaterials/runs/1e472348aa1a8125ca1f)
and [projector-pair](https://wandb.ai/teshbek/PointCloudMaterials/runs/d9a1885b29c09ffb54a5).

Local verification used no online runs: exact row coverage and resume order,
finite structured-head gradients, TDL agreement with a direct tied-pair calculation
(maximum gradient difference 8.94e-8), exact scalar/vector covariance multiplier
ratios 255/31, and two complete batch 8192 updates through the largest TDA variant.
Peak allocated memory was 35.56 GiB on RTX PRO 6000; the second update took 17.38 s after
compilation. First-epoch Al batch fractions were 55.5–56.8%. These are engineering
checks, not pilot or validation performance results.
