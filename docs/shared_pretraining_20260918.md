# Shared pretraining campaign: execution and continuation

The [scientific protocol](../experiments/shared_pretraining_20260918/README.md)
defines three fresh 12-epoch structural fits, three initialized causal fits and
three paired structural/causal frozen evaluations. Peak learning rate is 0.02,
with 10% linear warmup and cosine decay to 0.0002. Batch size is 512 anchor pairs.

The separate [H200 handoff](h200_shared_pretraining_task_20260918.md) repeats all
three pipelines with batch 1,024 and the same 12-epoch exposure and peak LR. Its
new configs, output directories and W&B IDs are independent of this submitted
local campaign; it runs serially using an explicit deadline without Slurm IDs.

## Data and memory

Registry IDs are `structural-neighbors-250k-20260917` (five-metal structural
release) and `shared-causal-38400-20260918` (Al causal/frozen-evaluation release).
Use [the registry](../DATASETS.md) for paths and generating-potential provenance.
No simulation was launched. The causal release was derived from existing
positions and the frozen native cohort; all 38,400 row identities and source
roles passed an explicit audit.

The physical microbatches are MACE 176, GATr snapshot 416 and GATr history 48.
A 40 GiB CUDA allocator budget and `PYTORCH_ALLOC_CONF=expandable_segments:True`
limit fragmentation and leave room on 48 GB L40S GPUs. Measured preflight peaks
vary with material, input size and device; the requested approximate 40 GB is
not a constant tensor allocation. Full statistical batches stay at 512.
Preflight timing/memory measurement is a separate diagnostic command, not an
extra benchmark inside scientific training.

## Maintained commands

From the repository root in conda `pointnet`:

```bash
python -m src.training_methods.shared_pretraining.data --config configs/shared_pretraining/data.json --workers 6
python -m src.training_methods.shared_pretraining.queue submit --plan configs/shared_pretraining/campaign.json
```

Preparation resumes immutable source/frame shards and preserves all source
manifests and target hashes. The submitted campaign freezes executable Python,
configs and metric contracts under its `technical/code/`, with machine storage
settings and links to the existing data/output roots. Subsequent workspace edits
do not alter the queued jobs. Do not edit this snapshot while its jobs remain
submitted. `technical/code-files.json` records its Python hashes.

The queue starts MACE–VICReg inside H100 allocation 995957 and GATr–VICReg
inside RTX6000PRO allocation 997207. Each local worker can proceed to causal
training and analysis while its allocation has time. **New JEPA uses a fresh
Slurm GPU allocation**, as requested. The old JEPA pilot was stopped at its
update-832 checkpoint; its exact-resume checkpoint and original source snapshot
remain in the pilot run directory.

Submitted jobs request one GPU from `H100,RTX6000PRO,L40S`, eight CPUs and 96 GB
host memory. Each structural chain has up to three 16-hour slots. It resumes
unfinished training or exits promptly if the local/earlier slot already finished;
these slots do not request additional epochs. Later slots depend on termination
of the preceding allocation. The last slot must finish the budget before the
16-hour causal and 8-hour analysis jobs become eligible through `afterok`.
Actual allocation depends on cluster availability. This permits 12-epoch JEPA
to span allocations without extending or overwriting the earlier pilot.

`technical/submissions.json` records exact job IDs, dependencies, scripts, local
PIDs and frozen config paths. Current state is available through `squeue` and
individual `technical/status.json` files. The trainer checkpoints before its
Slurm deadline and responds to TERM/USR1 by checkpointing at an update boundary.
Fatal scientific failures block later stages rather than silently changing
batch size, precision or objectives. Allocation continuation restores model,
optimizer, absolute schedule position, RNG and SIGReg buffers; switching GPU
models can change floating-point rounding, so it is not a bitwise cross-device
reproducibility claim.

## Tracking and artifacts

Submitted September 18 at 00:57 Paris time. All three structural runs were
verified advancing with checkpoints and online W&B after launch.

| Variant | Active start | Structural continuation jobs | Causal job | Analysis job |
| --- | --- | --- | --- | --- |
| MACE–VICReg | Existing H100 995957 | 997557 → 997558 → 997559 | 997560 | 997561 |
| GATr–VICReg | Existing RTX6000PRO 997207 | 997562 → 997563 → 997564 | 997565 | 997566 |
| GATr–JEPA | **New H100 997567**, nodesumo01 | 997568 → 997569 | 997570 | 997571 |

The two existing-allocation workers also advance their own causal/analysis
stages when time permits. Submitted stage jobs detect completed work and exit
without repeating it. See the [submission receipt](../output/shared_pretraining/campaign-20260918/technical/submissions.json)
and [launch check](../output/shared_pretraining/campaign-20260918/technical/launch-health.json).

Online W&B project: [teshbek / PointCloudMaterials](https://wandb.ai/teshbek/PointCloudMaterials).
Campaign group: `shared-12ep-20260918`; diagnostic runs use a separate preflight
group. Each phase has a stable W&B ID, recorded with its URL in
`technical/wandb_run.json`. The run resumes that ID across allocation changes.
Logs include learning rate, epoch equivalents, each loss, gradient norm,
allocated/reserved memory, timing and selection metrics. Raw trajectories and
checkpoint weights are not uploaded as artifacts.

Each stage has its own `output/shared_pretraining/<variant>-<stage>-20260918/`:

- `technical/last.pt`: model/optimizer/schedule/RNG resume state.
- `technical/best.pt` and `encoder.pt`: selected full model and exported encoder.
- `technical/updates.jsonl` and `validation.jsonl`: complete local training traces.
- `tables/selection.csv` and `tables/METRICS.md`: defined selection metrics.
- Analysis: source-cached embeddings, ridge/nonlinear readouts, paired predictions,
  test metrics with source intervals, and `plots/frozen_prediction.png`.

Training resumes automatically from `last.pt` after checking scientific
identity. Microbatch size is operational; statistical batch, losses, schedule,
seed, data and parent identity are fixed. The six scientific training phases
and three analysis stages remain distinct in W&B and on disk.
