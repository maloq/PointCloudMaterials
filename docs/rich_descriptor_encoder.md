# RD-MACE256-L3-Z256

Requested 29 September 2026: increase MACE capacity and duration for the same rich
descriptor task. Recipe:
[rich_mace256_20260929.json](../configs/liquid_predictability/rich_mace256_20260929.json).

The user selected all 183,596 **unrelaxed** training contexts and the original
3,536 targets. This is roughly twelve times the paired-archive training cohort,
and is identical to the earlier full-raw feature-learning cohort. Each context
uses 25 shared-weight patches; each patch contains 80 candidate atoms. Frozen
validation/calibration/test sources are unchanged. No descriptor recomputation,
simulation or relaxation is needed for this run.

The patch MACE has width 256, three spatial interactions and a 256-D scalar
export. Two vector-context blocks of width 256 and 32 vector channels produce
one 256-D invariant state. Its 256→512→3536 decoder receives only that state.
VCReg is retained. Sixty exact shuffled data passes replace the older 256-update
training blocks; the large batch is measured before submission. LR warms to
0.004 over five epochs and decays with cosine to 1e-5. Selection is validation
descriptor likelihood. See [metric definitions](metrics/rich_descriptor_encoder.md).

Use `pointnet-torch214`. Implementation remains in `src/`:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.research.liquid_predictability.rich_encoder probe --config configs/liquid_predictability/rich_mace256_20260929.json
python -m src.research.liquid_predictability.rich_encoder submit --config configs/liquid_predictability/rich_mace256_20260929.json --allocation ALLOCATION
```

The environment flag is required by this environment's installed e3nn constants
loader; submitted worker scripts set it explicitly. The expandable CUDA allocator
avoids fragmentation across large, variable-geometry batches; keep its environment
setting for both profiling and submission.

`probe` uses one visible 96-GB GPU, real training inputs, forward/backward and
optimizer allocation, without W&B. It saves `technical/batch-plan.json`; training
cannot silently change it. The two-GPU fit keeps global differentiable VCReg.
BF16, cuEquivariance, compiled spatial blocks, activation checkpointing, GPU
resident observations/targets and per-batch deduplication are retained.

Submission freezes source and metric contracts. With `--allocation`, training
starts detached inside that current two-GPU allocation and a Slurm continuation
waits for its end. Without it, Slurm starts the run. Each 24-hour RTX6000PRO job
checkpoints before its deadline and requeues itself until 60 epochs and evaluation
finish. Scientific exceptions fail loudly and do not trigger blind requeue.
Checkpoint every 64 updates and every epoch; keep best/last and each fifth epoch.
Resume uses the same batch, two GPUs, source rows and W&B ID.

Output: `${storage:analysis}/liquid_predictability/rd-mace256-l3-z256-20260929`.
Dataset caches remain outside the repository and are reused read-only.
`technical/launch.json`, `state.json`, `training.jsonl`, `validation.jsonl` and
`wandb/fit/run.json` expose execution and progress. Scientific results go in
`analyses/prediction-v1`, with all feature metrics and exact embedding row IDs.

## Launch on 29 September

The boundary measurement selected **12,800 contexts per GPU**, **25,600 global**,
on two RTX PRO 6000 Blackwell 96-GB GPUs. Peak allocated memory was 83.52 GB
(77.79 GiB); the second geometry draw also passed. The next tested batch, 13,056,
failed allocation. Eight updates per exact epoch give **480 updates** and
11,015,760 context visits across 60 epochs. The encoder has **3,819,328** parameters;
the complete encoder/context/descriptor model has **6,515,024**.

Detached local fitting starts in node58 allocation `1012728`; Slurm continuation
`1013467` waits for that allocation's end and requests two RTX6000PRO GPUs.
The memory-probe log and full measurements are retained under `technical/`.

The first distributed optimizer update was finite and checkpointed. The second
encountered allocator fragmentation with the default allocator. The run resumed
from that checkpoint with `expandable_segments:True`, retaining its batch and
W&B ID. `technical/allocator-restart.json` preserves this execution change;
the continuation script carries the same setting.
Updates 2 and 3 then completed with finite losses and gradients, with no batch
reduction. Online tracking:
[RD-MACE256-L3-Z256](https://wandb.ai/teshbek/PointCloudMaterials/runs/510c406ff50096b08bfb).
