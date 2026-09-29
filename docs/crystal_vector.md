# CDV-MACE128 execution

**Current recipe:** [independent random sampling, 28 September](../configs/crystal_vector/al64_20260928.json).
Results are `${storage:analysis}/crystal_vector/al64-random-20260928`.
The former quota-based lanes below were stopped and their continuation jobs
canceled; their partial checkpoints and frozen scientific definitions remain.
The new three-arm comparison starts from the same original parent, with fresh
optimizers and run identities. See the [sampling revision](../experiments/crystal_vector_20260928/README.md).

The revised lanes are detached on node61 inside allocation 1010610:
[VCReg primary](https://wandb.ai/teshbek/PointCloudMaterials/runs/8b10b5efda78a4882335)
and [directional control](https://wandb.ai/teshbek/PointCloudMaterials/runs/75059c28c8a4b35d79fd).
Distance-only follows in lane 0; continuation jobs are 1012036 and 1012037.
The random-sampling check averaged 31.54 centers at 0<d<=8 A across 2,000
batches (range 15–53, expected mean 31.5005); no quota/rejection was applied.
The new full-batch backward check had finite gradients. Former durable
checkpoints are preserved at updates 1920 (VCReg) and 768 (directional control).

The sealed existing geometry is reused. Run `launch` with the new recipe;
do not regenerate it with the changed live training source. If geometry ever
needs rebuilding, use a fresh dataset root or the original captured producer.

```bash
python -m src.research.crystal_vector.queue launch --config configs/crystal_vector/al64_20260928.json
```

The following launch record describes the superseded 27 September run.

Launched detached on node61 on 27 September 2026 inside allocation **1010610**,
one RTX PRO 6000 Blackwell GPU per lane. Primary W&B:
[distance + direction + VCReg](https://wandb.ai/teshbek/PointCloudMaterials/runs/bc543351bdc438b27a75);
matched control: [distance + direction](https://wandb.ai/teshbek/PointCloudMaterials/runs/2f04088dabfaae99f9d8).
Distance-only follows in lane 0. Automatic continuation job IDs are **1011978**
and **1011979**, dependent on the current allocation ending. They are canceled
when their lane completes. See the durable
[launch receipt](/work/PERSO/vmorozov/analysis/crystal_vector/al64-balanced-20260927/technical/launch.json).

The full-batch local check used 256 contexts, passed finite backward gradients
through both MACE and vector export, and used about 7.5 GiB peak allocated VRAM.
Warm compiled updates measured about 0.66 seconds before concurrent training.
These checks created no W&B runs. Float32 rotation checks gave vector relative
error 2e-6 and direction maximum component error 2.9e-5; the distance-only
covering selected identical atom IDs after rotation. All 126,545 original fixed
rows are retained exactly, plus 28,512 uniform fitting/selection rows and 22,872
scan rows. The model has **826,293** trainable parameters.

Use conda `pointnet-torch214`. Implementation lives in
`src/research/crystal_vector/`; the [scientific protocol](../experiments/crystal_vector_20260927/README.md)
and [metric definitions](metrics/crystal_vector.md) declare the target and selector.

```bash
# Historical producer/recipe; resume only with its captured source when explicitly wanted.
python -m src.research.crystal_vector.queue prepare --config configs/crystal_vector/al64_20260927.json
```

The launch freezes source/config/metric contracts and starts two detached GPU
lanes inside the user's current two-GPU allocation. Lane 0 trains the primary
VCReg model and distance-only control; lane 1 trains the directional control.
Associated full evaluation follows training in each lane. Scientific fits alone
create online W&B runs; evaluation updates their existing IDs. Local correctness
checks create no online runs and no automated test suite.

Coordinate caches reside on IDS, outside the repository:
`${storage:cache}/crystal-vector/al64-covering-20260927`. No encoder feature cache
is reused across optimization steps. Coordinate banks are deduplicated within
each source/frame; overlapping query patches are encoded once per minibatch.
Context batches are 256; MACE processes chunks with activation checkpointing.

Results are `${storage:analysis}/crystal_vector/al64-balanced-20260927`.
`technical/launch.json` records processes and continuation jobs; lane logs and
states are beside it. Each variant has `technical/{best,last}.pt`, original
prediction-context/sampling records, training/validation JSONL and W&B receipts.
Numerical outputs live in `analyses/localization-v1/{tables,technical}`.

Checkpointing includes optimizer and sampler RNG at intra-epoch update boundaries.
Before allocation expiry, a worker saves and exits. Dependent one-GPU Slurm jobs
resume the same frozen lane after the current allocation ends. Successful lanes
cancel their unused continuation jobs. Completed training is skipped on resume;
no checkpoint is evaluated as final before its full training budget completes.
Do not edit frozen source/configs. A failed scientific check is visible in the
lane state and log, never replaced by a silent fallback/offline run.
