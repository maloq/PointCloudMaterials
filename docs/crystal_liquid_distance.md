# Liquid-only external-crystal distance run

Use `pointnet-torch214`, on node58 with two allocated RTX PRO 6000 GPUs. The recipe
is `configs/crystal_vector/liquid_distance_20260928.json`. Its global batch and
microbatch are 512, split 256 per GPU. One online W&B run records joint training.

```bash
python -m src.research.crystal_vector.interface_queue launch \
  --config configs/crystal_vector/liquid_distance_20260928.json --gpu 0
```

The launch requires the local full-batch preflight receipt, freezes source/config
and metric definitions, reuses the declared prepared-data identity, and starts a
detached controller. `prepared_data` references the existing CPU preparation array
1012822 and sealing job 1012823. It does not regenerate this geometry. These jobs
continue after the earlier waiting controller was stopped; its GPU backup 1012824
was canceled before any training.

Cache: `${storage:cache}/crystal-interface/al64-dense-clear-20260928`.
Output: `${storage:analysis}/crystal_liquid_distance/al64-dense-vcreg-20260928`.
`technical/launch.json` records the new controller/continuation and reused CPU jobs;
`technical/variant-0.json` records the active stage. The selected model's
`technical/observation-filter.json` records the strict fitting/selection populations.
A two-GPU continuation resumes if allocation time expires; completion cancels it.

Launch on 2026-09-28: allocation **1012728**, node **node58**, controller PID
**2047573**, continuation job **1012891**. The local representative-batch check used
the sealed original cohort with the strict predicate and passed a two-GPU backward
step (global 512, finite gradients, no optimizer update). The complete expanded
cohort is checked again when the training loader opens the sealed manifest.

The expanded cohort is sealed: 1,133,196 cached contexts, 24.38 GiB of coordinate
arrays. The live loader reproduced every audited strict population count. Online
scientific run: [LCD-MACE128-VC](https://wandb.ai/teshbek/PointCloudMaterials/runs/32f3b44a24e49a2113b2).

The scientific protocol is [liquid distance](../experiments/crystal_interface_20260928/LIQUID_DISTANCE.md).
The original interface-only exclusion artifacts retain their original config and
are marked superseded before training. Data and previous results are preserved.
