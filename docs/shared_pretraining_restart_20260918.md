# VICReg restart: batch 1,024, peak LR 0.002, BF16

The user corrected the previous peak-LR typo and requested fresh MACE and GATr
VICReg training, doubled batches and BF16. Both use the existing 250k broad
release, seed 20260919, snapshot inputs, spatial/temporal neighbor views,
physical + 0.25 instantaneous-TDA anchors and the unchanged VICReg coefficient.
Twelve epochs require 2,930 updates. Warmup lasts 293 updates, followed by cosine
decay from 0.002 to 0.00002. Failed September 18 weights are not resumed.

The active recipes are in
[`configs/shared_pretraining/restart_b1024_lr002_bf16`](../configs/shared_pretraining/restart_b1024_lr002_bf16).
MACE uses allocation 997799 on nodesumo01 (H100), GATr allocation 997864 on
node58 (RTX PRO 6000). Effective batch size is 1,024 anchor pairs, with full-batch
VICReg statistics; encoder microbatches are 176 and 416 respectively and the
allocator limit remains 40 GiB. Detached workers checkpoint every 16 updates
and validate every 64. They stop and save before allocation expiry.

Launch once from `pointnet`:

```bash
python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/restart_b1024_lr002_bf16/campaign.json
```

The launcher freezes source/configs and writes detached process receipts in
`output/shared_pretraining/restart-b1024-lr002-bf16-20260918/technical/submissions.json`.
Per-run status, exact-resume state and online W&B URLs are under each variant's
`technical/`. This launch includes neither JEPA nor dependent causal fits.

Corrections include normalized scalar readout inputs/hidden preactivations,
bounded exported-state scale, normalized physical/TDA/projector heads,
float64 selection/probe reductions, per-group loss curves, and explicit checks
for finite-but-collapsed representations and decoders. A train-only group-mean
baseline must be beaten by update 640. See the exact
[metric definitions](metrics/shared_pretraining.md). This is a learning guard,
not a guarantee of predictive usefulness. The later causal head now has a
normalized interface and zero initial output; its training is not scheduled.

BF16 autocast runs eligible encoder/head kernels while weights, AdamW moments,
VICReg statistics and losses remain FP32. Both gradient-cache passes use the
same precision. BF16 uses no FP16 gradient scaler. Custom W&B timing/memory
series stay removed; raw/weighted VICReg totals and system monitoring remain.

## Requested precision comparison

The separate profile command compares FP32 (TF32 off) and BF16 on identical
real-data indices, batch 1,024, fixed microbatch size and initial seed. Each
fresh process does two untimed warmup updates and five synchronized measured
updates. It uses the largest-support group to check the 40 GiB limit. CPU data
preparation is excluded; device transfer, forward/backward, gradient caching
and optimizer step are included. This is a repeated-batch compute comparison,
not end-to-end epoch throughput or an accuracy-equivalence claim.

```bash
python -m src.training_methods.shared_pretraining.profile \
  --config configs/shared_pretraining/restart_b1024_lr002_bf16/mace_vicreg_structural.json \
  --microbatch 176 --precision bf16 --output /absolute/path/to/measurement.json
```

Repeat with `float32`, and use the GATr config/microbatch 416 on RTX6000.
Raw timings, device identity, indices, loss values and peak memory are retained
in the campaign's `technical/*-{float32,bf16}.json`. Median FP32 update time
divided by median BF16 time defines speedup. Training itself does not benchmark.

Measured medians on identical Ta/unknown-static batches:

| Model / device | FP32 update | BF16 update | Speedup | Peak allocated GiB, FP32 → BF16 |
| --- | ---: | ---: | ---: | ---: |
| MACE / H100 | 3.451 s | 3.216 s | 1.07x | 34.95 → 18.57 |
| GATr / RTX6000 | 5.072 s | 3.194 s | 1.59x | 35.70 → 21.28 |

See [the retained comparison](../output/shared_pretraining/restart-b1024-lr002-bf16-20260918/PRECISION.md)
for definitions, raw measurements and hashes. These timings exclude CPU preparation.

## Launch receipt

Both detached workers started on September 18 at approximately 07:58 UTC,
with separate frozen source and fresh run directories. Online runs:

- [MACE / H100](https://wandb.ai/teshbek/PointCloudMaterials/runs/mace-v3-1024-bf16-0918).
- [GATr / RTX6000](https://wandb.ai/teshbek/PointCloudMaterials/runs/gatr-v3-1024-bf16-0918).

Before launch, the CPU suite passed 16 tests; both BF16 cached-gradient tests
passed on RTX6000, and both architectures completed BF16 updates on H100.
The memory/timing preflight completed seven full real-data updates per
precision and architecture. These checks validate execution, gradient caching
and numerical bookkeeping; they do not establish final representation quality.
