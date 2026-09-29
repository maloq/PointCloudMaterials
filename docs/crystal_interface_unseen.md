# Dense non-visible-interface experiment

**The GPU fit was superseded before training.** Use the
[strict liquid-only distance workflow](crystal_liquid_distance.md). The CPU expansion
continues to completion and supplies that corrected fit; this document preserves
the earlier protocol and launch record.

Use conda `pointnet-torch214`. `configs/crystal_vector/interface_unseen_20260928.json`
defines one VCReg treatment on two GPUs, global batch/microbatch 512.

```bash
python -m src.research.crystal_vector.interface_queue launch \
  --config configs/crystal_vector/interface_unseen_20260928.json --gpu 0
```

Launch inside an allocation with GPUs 0 and 1. It freezes code and metric contracts,
queues parallel CPU expansion/sealing, then starts one detached controller that
waits for successful sealing. Training starts via two local distributed processes;
one online W&B identity records the scientific fit. Associated evaluation follows.
A two-GPU continuation job resumes the captured state after allocation expiry;
the controller cancels it when work completes.

Cache: `${storage:cache}/crystal-interface/al64-dense-clear-20260928`.
Output: `${storage:analysis}/crystal_interface/al64-unseen-dense-vcreg-20260928`.
`technical/launch.json` records preparation/continuation jobs and the controller;
`technical/variant-0.json` records waiting, training, evaluation or failure.
Per-source candidate counts record which uniformly proposed queries were retained.
Original geometry and labels are preserved at the beginning of each source shard.

The 2026-09-28 launch uses allocation **1012728** on **node58**, both RTX PRO 6000
GPUs. CPU preparation array **1012822** has 15 tasks, at most eight concurrently;
sealing is **1012823**, and the two-GPU continuation is **1012824**. The first
controller stopped because `sacct` could not reach the accounting daemon. The
restarted controller queries the live `scontrol` dependency state; the original
failure and operational code change are preserved in `technical/`. Preparation
was unaffected. W&B starts when the scientific training begins, after sealing.

The CPU overfitting audit reads the six completed runs without retraining:

```bash
python -m src.research.crystal_vector.overfit \
  --output '${storage:analysis}/crystal_interface/review-20260928/analyses/overfitting-v1'
```

It writes metric tables, a learning-curve figure and input/source audit receipts
locally, with frozen definitions. No diagnostic W&B runs are created.

[Completed audit interpretation](../experiments/crystal_interface_20260928/OVERFITTING.md)
and [scientific protocol](../experiments/crystal_interface_20260928/UNSEEN.md).

## Feature dominance audit

The [feature-specific report](../experiments/crystal_interface_20260928/FEATURE_DOMINANCE.md)
examines the completed VCReg interface checkpoint, separately from the new adaptation.
Run the CPU probes and frozen-predictor GPU interventions as local diagnostics:

```bash
python -m src.research.crystal_vector.feature_audit probes \
  --config configs/analysis/crystal_feature_dominance_20260928.json
python -m src.research.crystal_vector.feature_audit interventions \
  --config configs/analysis/crystal_feature_dominance_20260928.json --device cuda:0
python -m src.research.crystal_vector.feature_figures \
  --root '${storage:analysis}/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3'
```

Use the recorded conda environment, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, and a
modest BLAS/OpenMP thread count. Both numerical stages verify the frozen checkpoint
and saved predictions; neither starts a W&B run. Figures only read saved CSVs.
Use a new named analysis revision if changing any calculation or diagnostic design.
