# Expanded MACE crystallization transfer queue

Recipe: `configs/crystallization_transfer/mace_20260919.json`.
Scientific protocol: [comparison](../experiments/crystallization_transfer_20260919/README.md).

Use conda `pointnet-torch214`:

```bash
python -m src.research.crystallization_transfer.queue submit --config configs/crystallization_transfer/mace_20260919.json
```

Submission freezes source, configuration and metric contracts. Four detached
workers occupy the already allocated H100 (997799), two node61 RTX PRO 6000
GPUs (999611, one independent lane each), and node58 RTX PRO 6000 (999702).
No new allocation or remote H200 transfer is submitted. Per-lane preparation
uses two CPU processes; already prepared graphs and tensor features are reused.
All caches belong in IDS via the machine-local cache root. Original trajectories
and PTM labels are immutable. Each source has completion hashes. The ~40 GiB
raw graph cache plus native tensor features is shared across fits.

After the shared release is complete, workers lock-claim independent fit variants.
The shorter H100 allocation prioritizes frozen comparisons. Other lanes start
fine-tuning/scratch comparisons, then take remaining tasks. Fits save selected
and latest checkpoints, preserve their update budget across allocation resumes,
and stop training with evaluation reserve before the allocation deadline.
Input preparation uses persistent CPU processes with four updates prefetched;
frozen fits reuse native tensor features and never rerun MACE. Prediction heads
use the same source-balanced batches, so fine-tuning and scratch receive equal
sample budgets. There is no hardware benchmark inside training.

Status and logs: `output/crystallization_transfer/mace-expanded-20260919/technical/`:
`lane-N.json`, `lane-N.log`, `queue.json`, `submissions.json`, and `runs/NAME/status.json`.
Readable metrics are in the top-level `tables/`; machine predictions/checkpoints
are per fit in `technical/runs/`. The queue performs predeclared evaluation but
does not select new experiments based on test results. No notification monitor
is left running after handoff.

## Detached scaling continuation

```bash
python -m src.research.crystallization_transfer.queue submit --config configs/crystallization_transfer/mace_scaling_20260919.json
```

This separate, frozen 52-fit extension reuses the original cache identity and
pinned parent checkpoint. It writes to
`output/crystallization_transfer/mace-scaling-20260919/`. Each detached lane waits
for its corresponding original worker to finish and exit before using its GPU.
Original jobs and their frozen code are unchanged. A failed predecessor blocks
that lane explicitly. No extra GPU allocation is requested; the existing H100
and three RTX PRO 6000 GPUs take tasks with shared file locks. H100 prioritizes
frozen fits. Six-epoch fits are last. Workers checkpoint before allocation expiry;
all queued fits are not guaranteed to finish before their allocations expire.

Per-fit `training-population.json` and `training-indices.npy` record the exact
nested subset, full-pass lengths and total optimizer budget. `status.json` reports
planned updates during training. Data-size comparisons use the update budget of
three epochs of all training data; duration comparisons use 1/3/6 actual epochs.
Small fractions are selected only from training sources. Full evaluation remains
identical, with unchanged alarm calibration and source-bootstrap metrics.

## Corrected adaptive-attention queue

```bash
python -m src.research.crystallization_transfer.queue submit --config configs/crystallization_transfer/mace_adaptive_20260919.json
```

This fresh queue reuses the original source cache and parent; it does not resume
weights from the compromised fine-tuning runs. It uses node61 allocation 999611
(two RTX PRO 6000 GPUs) and node58 allocation 999702 (one RTX PRO 6000). The H100
allocation expired. The earlier queues are complete, so this recipe bypasses
predecessor waiting. Initial workers use the existing allocations; the dependent
continuation slots described below extend the queue budget.

The frozen source snapshot, 52-screen queue, per-fit status and predictions live
in `output/crystallization_transfer/mace-adaptive-20260919/technical/`. A locked
`promotions.json` adds ten 12/24-epoch fits after every screen completes; ranking
reads only selection NLL. Per-fit `validation.jsonl` records normalization scale,
geometry sensitivity at a fixed condition, feature spread and selection NLL.
`normalization-indices.npy` contains the exact training-only calibration rows.

Direct backward retains microbatch activation graphs across the complete batch
on these 96 GB devices. The explicit `encoder_backward: replay` option can reduce
activation memory at the cost of a second encoder forward. CPU preparation stays
in persistent processes with prefetched batches. Detached workers share fit
locks and save exact optimizer checkpoints before allocation expiry. Pending
fits may outlast the allocations; the immutable queue and checkpoints remain
available for a continuation. The scientific protocol is
[ADAPTIVE.md](../experiments/crystallization_transfer_20260919/ADAPTIVE.md).

The measured native step time exceeds the remaining initial allocation budget.
Two one-GPU, 16-hour continuation slots are therefore submitted after node61's
allocation 999611 ends, using `RTX6000PRO,H100`. They replace its two lanes rather
than increasing current GPU concurrency. They retain Slurm's GPU assignment,
load the same frozen code/config and resume locked unfinished fits before taking
new work. Both exit when the queue is complete; 16 hours is a cap per slot.

```bash
python -m src.research.crystallization_transfer.queue continue --config configs/crystallization_transfer/mace_adaptive_continuations_20260919.json
```

Submission records and exact batch scripts are in `technical/continuations.json`
and `technical/continuation-N.sbatch`. These operational continuation settings do
not alter the scientific plan or the running workers' source snapshot.
