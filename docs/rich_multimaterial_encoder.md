# Multimaterial MACE256 descriptor run: correlation 3, angular order 3

## RH2 repair: normalized residual descriptor head

The prior SiLU head saturated; that run was checkpointed at update 1,196 and its
two-GPU continuation 1014187 was cancelled. Keep its weights, logs and diagnostic
report. It is not a successful descriptor encoder. The replacement is
**MM-RD-MACE256-D3-C3-L3-Z256-RH2**, using
[the RH2 recipe](../configs/liquid_predictability/rich_multimaterial_residual_20260929.json).

It starts afresh on the identical 1,056,768 fitting patches and target transform.
MACE remains width256/depth3/correlation3/angular3 with 256-D export, global batch
8,192, checkpointed chunks1,024, common peak LR .01, five-epoch warmup/cosine,
and 60 epochs. The new FP32 head uses a normalized direct linear path plus two
normalized residual blocks of width512/expansion2. Scalar/vector VCReg remains;
an additional .01 scalar-mean penalty prevents unchecked embedding offsets.

Validation now runs every epoch, with mean-relative errors overall and per family.
Training also logs prediction variation and the descriptor-only gradient entering
the embedding. A persistent constant-predictor/no-task-gradient condition saves
the checkpoint and fails loudly. See [definitions](metrics/rich_multimaterial_encoder.md#rh2-head-and-learning-diagnostics).

The separate local check verifies batch memory and 12 updates at peak LR; its
weights are discarded and it creates no W&B run. The scientific fit is online.
The outer launcher is a detached Slurm step in the current H100 allocation.
The launch supervisor stays alive until its worker exits, so the Slurm step does
not finish immediately after spawning training. A two-GPU continuation uses the
existing exclusive lease and optimizer-boundary checkpoint handoff.

Output: `${storage:analysis}/liquid_predictability/mm-rd-mace256-c3-l3-residual-head-20260929`.
Submitted on 29 September: H100 allocation **1013895** on node53.
WORK filled during the first update; its 513-GiB quota prevented receipt/log
writes, including cleanup, and the initial continuation 1014590 was cancelled.
The step-1 checkpoint and all run artifacts were checksum-verified on STORE.
Active physical output is now
`${storage:training_storage}/liquid_predictability/mm-rd-mace256-c3-l3-residual-head-20260929`.
A per-run `machine.local.yaml` alias resolves the unchanged frozen output setting
there. No dataset or scientific identity changed. Some already verified duplicate
files remain on WORK because quota also prevented their removal; the active
files and migration receipt are on STORE.

Resume uses tmux socket `pcm-training`, session `mmrd-rh2` on node58, supervising
a Slurm step on node53. Replacement two-GPU continuation: **1014597**.
Scientific tracking:
[RH2 W&B run](https://wandb.ai/teshbek/PointCloudMaterials/runs/139e0665675ff7eb87c8).
Total parameters: 8,819,124 (6,230,592 encoder; 2,572,148 descriptor head;
16,384 vector projection). Three full-batch numerical passes retained finite
gradients and fit with a measured 69.66-GiB PyTorch allocation peak. The separate
12-update, batch256 peak-LR stress check retained task gradients in all three
interactions, but had large transient losses when jumping directly to .01;
it does not establish stable scientific fitting. The actual run uses five full
epochs of LR warmup and independently tracked validation.
Use conda `pointnet-torch214` and the existing `rich_multimaterial_queue start`
entry point with the RH2 config and current allocation. `technical/start.log`,
`local-worker.log`, `state.json`, `launch.json`, and `wandb/fit/run.json` record
progress. Historical execution recipes below retain their original settings.

## Current execution: 60 epochs, batch 8,192

**Launch status:** configured, not training. A native cuEquivariance forward with
activation checkpointing disabled retained 55.34 GiB at 1,024 patches and
82.97 GiB at 1,536 patches; it exhausted the 93.09-GiB H100 while adding the
next 256-patch chunk. See `technical/no-checkpoint-memory.json`. This was a
forward-memory check, not a successful full-batch backward or scientific fit.
The full-graph Torch compilation diagnostic was stopped during unusually slow
Inductor lowering; no compiled-path memory figure is claimed. The user subsequently chose checkpointing with larger chunks on one/two GPUs.
The active chunk is 1,024 patches; the declared global batch stays 8,192.
Native cuEquivariance kernels remain enabled. Whole-graph `torch.compile` is
disabled for this launch after the slow Inductor diagnostic; this execution
choice is explicit in the recipe. Native kernel compilation is cached in IDS.


The [active recipe](../configs/liquid_predictability/rich_multimaterial_c3_l3_20260929.json)
uses width 256, three message-passing interactions, correlation order **3** and
maximum angular order **3**. Hidden irreps are
`256x0e + 256x1o + 256x2e + 256x3o`; the exported scalar embedding remains 256-D.
Its head predicts the same 442 descriptors. Geometry-only inputs and material
normalization are unchanged.

Train a **new randomly initialized model** for **60 complete epochs** with global
batch **8,192**, maximum LR **0.01**, five-epoch linear warmup and cosine decay to
**1e-5 on the last update**. There is no compute-hour counter or time-based training
completion rule. Activation checkpointing is **on, in 1,024-patch chunks** (user-approved). The execution check measures
actual memory before launch; it cannot silently lower the declared global batch.

Use precisely the previous **1,056,768 raw patches**: 593,617 Al, 44,369 Mg,
57,284 Ti and 361,498 Ta, from 117 training sources. Sample IDs and target-transform
checksums must match the earlier run. No descriptors are recalculated. There are
129 optimizer updates per epoch and **7,740 total updates**. The changed angular
representations/product order require new weights and a distinct W&B run; the
previous checkpoint at update 598 is retained.

The prepared H100 workflow supports a detached worker in the allocation and two-GPU
continuation using the same global batch and optimizer/LR cursor; the new run has not been submitted. Slurm allocation
expiry checkpoints progress; it does not complete the study or shorten cosine.

The `start` workflow verifies the fixed batch, freezes/verifies the exact
sample IDs and transforms, starts training detached and queues the two-GPU
continuation. A failed numerical check stops before scientific fitting.
For a fresh output directory, use conda `pointnet-torch214`:

```bash
python -m src.research.liquid_predictability.rich_multimaterial_queue start --config configs/liquid_predictability/rich_multimaterial_c3_l3_20260929.json --allocation ALLOCATION
```

Output: `${storage:analysis}/liquid_predictability/mm-rd-mace256-d3-c3-l3-z256-20260929`.
Inspect `technical/launch.json`, `worker-lease.json`, `executions.jsonl`,
`state.json` and `local-worker.log` for current process and Slurm job IDs.

The earlier width256/correlation2/angular2 run remains stopped at update 598.
Its proposed 190-epoch extension was prepared but never resumed; the current
user request supersedes that proposal. The original 60-epoch checkpoint and
prior code/logs remain in its output directory. Al context fitting stays stopped.

Earlier submission details follow for provenance.

## Historical queue: restore requested maximum LR 0.004

The lower-LR R2 jobs 1013920/1013921 were cancelled while still pending; neither
began scientific fitting. Reducing LR based on the Al failure confounded learning
rate with batch/update count and was not supported for the multimaterial task.
The active [LR004 recipe](../configs/liquid_predictability/rich_multimaterial_lr004_20260929.json)
restores peak **0.004 for both encoder and head**, five-epoch warmup and cosine
decay to 1e-5. It retains the memory fix, global batch cap 4,096, 60 epochs, the
same raw descriptor cache and a ten-hour fit budget on two GPUs. No Al restart.

Output: `${storage:analysis}/liquid_predictability/mm-rd-mace256-l3-z256-lr004-20260929`.
Use the same preflight/submit commands below with `rich_multimaterial_lr004_20260929.json`.
The R2 details below are historical and did not produce a scientific fit.

Replacement jobs: **1013933** (two-GPU sizing) → **1013934**
(two-GPU scientific fit, ten-hour limit). The local numerical check retained the
2,048/GPU batch, finite gradients and memory reserve. Its separate fixed-draw
32-update diagnostic jumped immediately to peak LR, without the scientific
warmup: NLL 1.3923 → 1.3526, with transient excursions up to 70.55. This is not
evidence of stable scientific training or validation skill. The actual fit uses
the full five-epoch warmup; no automatic LR reduction is applied.


## Repair and restart, 29 September

Only the multimaterial fit is being restarted. The Al context run remains stopped.
The original multimaterial batch search failed on cuEquivariance's CUDA workspace
allocation; it produced no scientific training checkpoint. Its pending jobs
1013503/1013504 were cancelled, with original artifacts preserved.

Use [the repair recipe](../configs/liquid_predictability/rich_multimaterial_repair_20260929.json).
It reuses the complete 13,423,868-patch raw descriptor pool; preparation is skipped
only after verifying its sealed release identity, plan/statistics hashes, ancestry
and coordinate normalization. The pool is not the final training subset.

The numerical search now handles cuEq OOM explicitly, reserves CUDA workspace
memory and caps batches at 2,048 patches/GPU (4,096 global). Encoder/head peak
learning rates are 1e-4/2e-4, with five-epoch warmup and cosine decay; VCReg,
width256/depth3/embedding256, all 442 targets and 60 complete epochs are retained.
A fixed-draw learning diagnostic stays local. The scientific fit remains online
in W&B with a new identity. All configuration changes are recorded in the recipe.

**Ten hours means two GPUs together, at most 20 GPU-hours for the fit.** Queue
waiting and separate numerical timing are outside that limit. The fitting subset
is selected from measured two-GPU throughput with 20% headroom. A warmup update
removes compilation from the per-step throughput estimate. Final data counts are
in `technical/batch-plan.json`, once sizing completes.

```bash
python -m src.research.liquid_predictability.rich_multimaterial_queue preflight --config configs/liquid_predictability/rich_multimaterial_repair_20260929.json
python -m src.research.liquid_predictability.rich_multimaterial_queue submit --config configs/liquid_predictability/rich_multimaterial_repair_20260929.json
```

New output: `${storage:analysis}/liquid_predictability/mm-rd-mace256-l3-z256-repair-20260929`.
Submission receipt: **1013920** performs two-GPU subset sizing; **1013921**
then runs scientific training with a ten-hour limit, two RTX6000PRO GPUs and
online W&B. At submission both were pending (priority/dependency). The completed
local H100 check selected 2,048 patches per GPU (4,096 global); three consecutive
steps used 4.46 GiB peak PyTorch allocations and retained about 87.6 GiB device
free memory. A 32-update fixed-training-draw diagnostic reduced descriptor NLL
from 1.39228 to 1.14156. This demonstrates local optimization only, not validation
skill. No diagnostic W&B run was created. Al training was not restarted.

Original protocol and submission history follow below.

User request, 29 September 2026: extend the rich-descriptor experiment to large
raw multimaterial data, retaining width 256, three interactions, embedding 256,
60 full epochs, maximum LR .004, warmup/cosine and the largest measured practical
batch. Size the **training subset** to fit about ten hours on two GPUs.

Recipe: [rich_multimaterial_20260929.json](../configs/liquid_predictability/rich_multimaterial_20260929.json).
Scientific definitions: [rich_multimaterial_encoder](metrics/rich_multimaterial_encoder.md).

This experiment trains one shared local encoder, without a context predictor.
One input is the existing nearest-80 patch, cropped at radius 8 after the
established fixed material normalization. Its 256-D state predicts 442 geometry,
bond-order, CNA and TDA descriptors of that same patch. The earlier Al run combines
25 patches and predicts 3,536 context summaries; its numerical scores are a
different task. No species or explicit condition is passed to either model.

The available source pool is the existing dynamic part of the expanded native
structural release:

| Material | Eligible raw patches |
| --- | ---: |
| Al | 7,537,284 |
| Mg | 566,244 |
| Ti | 728,000 |
| Ta | 4,592,340 |
| Total | 13,423,868 |

Static/inherent inputs are excluded, including the available Zr data. This keeps
the new run unrelaxed. No MD or minimization is submitted. Parent-source roles
and ancestry exclusions are unchanged. Additional 192,960 Al structural selection
rows are separate; fixed Al64 calibration/test retain their exact sample IDs.

CPU jobs cache normalized coordinates and the 442 descriptor targets in IDS.
This permits different compute-budget subsets without repeating descriptor work.
The data loader uses bounded memory maps and pinned prefetch rather than keeping
the whole corpus in VRAM. Encoder computation uses BF16, cuEq, compiled typed
spatial blocks, checkpointed patch chunks and two-GPU global VCReg.

## Submission and progress

Use conda `pointnet-torch214`:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.liquid_predictability.rich_multimaterial_queue preflight --config configs/liquid_predictability/rich_multimaterial_20260929.json
python -m src.research.liquid_predictability.rich_multimaterial_queue submit --config configs/liquid_predictability/rich_multimaterial_20260929.json
```

The local preflight checks real descriptors across roles/materials and the
resumable producer. It creates no W&B run. Submission freezes source, configuration
and metric contracts, then creates a dependency chain:

1. CPU descriptor array: 8 tasks, 8 workers each, verified resumable shards.
2. CPU sealing: all shard checksums and training-pool target statistics.
3. One-GPU numerical batch sizing; expandable allocator and consecutive updates.
4. Two-GPU numerical timing and fitting-subset selection.
5. Two-GPU scientific training, online W&B, **10-hour Slurm limit**.

The exact fitting count is determined by the timing job, not guessed from the
25-patch Al experiment. It is frozen before scientific training. Its estimate
includes validation, subset transforms and final exports, with 20% time headroom.
Data preparation, profiling and queue waiting are separate from the fit budget.
Every fifth epoch evaluates the complete structural validation population;
selection uses descriptor NLL. Retain best, last and every fifth-epoch checkpoint.
The worker does not silently extend the ten-hour allocation if the estimate is
too optimistic; it preserves the checkpoint and reports the incomplete state.

Output: `${storage:analysis}/liquid_predictability/mm-rd-mace256-l3-z256-20260929`.
Cache: `${storage:cache}/rich-descriptors/multimaterial-raw-20260929`.
Execution is under `technical/`: `launch.json`, `batch-probe-progress.json`,
`batch-candidate.json`, `batch-plan.json`, `training-pool-row-ids.npy`, `state.json`,
`training.jsonl`, `validation.jsonl`, and `wandb/fit/run.json`.
Final tables/states/predictions go in `analyses/descriptor-v1`.

Runs frozen after the 29 September mean-baseline logging update additionally log
`validation/relative_mse_to_training_mean` overall and with each descriptor-family
prefix (`geometry`, `bond_order`, `cna`, `tda`). This is model MSE divided by the
error of the frozen training-mean constant: 1 matches the baseline, 0.5 halves
its error. The overall score weights descriptor families equally. Final summaries
and CSVs include the same comparisons, with individual targets in `features.csv`.
See the [metric definition](metrics/rich_multimaterial_encoder.md#mean-relative-error-for-subsequent-runs).
The already frozen running experiment is not modified by this logging update.

## Submitted 29 September 2026

| Stage | Slurm job |
| --- | --- |
| CPU descriptor array | 1013493 (eight tasks) |
| Seal dataset | 1013494 |
| GPU batch measurement | 1013502 |
| Two-GPU timing and subset selection | 1013503 |
| Two-GPU training, ten-hour reservation | 1013504 |

Preparation began on nodecpu04/nodecpu05. The initial sixteen-task submission
reached the per-user job-count limit; its own array was cancelled and verified
partial shards were retained. The smaller array and all dependencies were then
accepted. CPU source is frozen under `technical/code`; the GPU sizing/training
snapshot is `technical/gpu-code-v2`. Submission receipts preserve both revisions.
The original Al context fit later stopped after a CUDA workspace OOM and is not restarted by this queue.

Kernel binaries are cached outside the repository using `CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR` under `${storage:cache}/compiled/cueq`, supported by the [NVIDIA release notes](https://github.com/NVIDIA/cuEquivariance/blob/main/CHANGELOG.md). This avoids repeating completed native kernel compilation on subsequent launches.
