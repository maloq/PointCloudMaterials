# Separate Al and Ta fine-tuning

Update, 27 September evening: Al completed twelve epochs and evaluation. Ta was
stopped at the user's request; epoch 11.1304 is preserved and evaluated in a
separate interim bundle, without further training. [Scientific results](../experiments/distance_encoder_20260926/MATERIAL_RESULTS.md).
The explicit command `material_evaluate --config CONFIG --interim-checkpoint PATH`
keeps partial-checkpoint exports separate from the unchanged final selector.

Submitted 27 September inside the user's node61 allocation **1010610**:
Al uses GPU 0; Ta uses GPU 1 after CPU array **1010631** and seal **1010632**.
The Al scientific W&B run is [b1a3ac3c6f88fcd4fce9](https://wandb.ai/teshbek/PointCloudMaterials/runs/b1a3ac3c6f88fcd4fce9).
Ta receives its stable online training ID when preparation completes.

The completed CD-MACE128-D6-075nominal model initializes two independent children,
CD-MACE128-D6-Al-FT and CD-MACE128-D6-Ta-FT. Both train the complete encoder and
temporal head for twelve epochs. [Scientific protocol and metrics](metrics/distance_encoder_material_finetune.md).

Use conda `pointnet-torch214`. Recipes are
`configs/distance_encoder/material_al_20260927.json` and
`configs/distance_encoder/material_ta_20260927.json`.

```bash
python -m src.research.distance_encoder.material_queue submit --config configs/distance_encoder/material_al_20260927.json --allocation ALLOCATION_ID --gpu 0
python -m src.research.distance_encoder.material_queue submit --config configs/distance_encoder/material_ta_20260927.json --allocation ALLOCATION_ID --gpu 1
```

Each command detaches a process pinned to one GPU inside the current two-GPU
Slurm interactive step. Both processes share the existing host memory/CPU allocation. The allocation must remain
alive. No user allocation is stopped. Code/configs/metric contracts are frozen
under each run's `technical/code`; stdout is `technical/worker.log`, progress is
`technical/queue-state.json`, and scientific training has an online W&B receipt.
The trainer checkpoints before its time limit and refuses to evaluate an
incomplete fit. Resume the frozen worker/config; do not submit a second identity.

Al reuses existing sealed geometry and fixed spatial paths. Ta submits a four-task
CPU array to prepare the four completed parent00 shooting branches and a dependent
seal. Existing PTM/periodic component-lineage code produces causal distance labels;
no new simulation is run. This is a within-known-parent test because all older Ta
data already entered parent training. No older training trajectory becomes test.
The GPU worker waits for sealed data and reports failed preparation explicitly.

New geometry stays on IDS at `${storage:cache}/distance-encoder/ta-shooting-material-eval-20260927-v2`.
PTM/reference provenance stays on WORK at `${storage:analysis}/crystallization_origin/ta-material-eval-20260927-v2`.
Raw shooting arrays remain on STORE. Completed model roots are
`${storage:analysis}/distance_encoder/material-al-20260927-v2` and
`${storage:analysis}/distance_encoder/material-ta-20260927-v2`.

Al paired results are in `analyses/material-comparison-v1/tables/`; Ta paired
results are in `distance-holdout/analyses/material-v1/tables/`. Every numerical
bundle includes frozen metric definitions and calculation hashes. Al reports
distance and front warning; Ta reports point distance and confidence reliability.

The initial launch stopped before training: the input verifier incorrectly compared semantic array hashes with NPY-file hashes, and nested exclusive steps waited behind the interactive step. Those receipts are preserved; v2 uses the trajectory producer checksum verifier and explicit per-process GPU assignment.
