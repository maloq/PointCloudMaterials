# Two-GPU MACE execution optimization

This changes execution of the existing `mace-expanded-dual-20260919` fit. It
preserves the 1,018,080-observation dynamic release, the global batch of 2,048,
seed, five-epoch-equivalent schedule, BF16/FP32 boundaries, objective coefficients
and selected-checkpoint rule. The completed smaller-data reference is unaffected.

## Input pipeline

Four persistent CPU processes prepare six updates ahead using the exact same
step-derived sampling as `mixed.prepare`. Each worker holds the parent's row and
group index, lazily maps immutable shard arrays, and keeps a bounded observation/
graph cache. Workers do not initialize CUDA or refit normalization. Only the main
training process pins completed batches and sends them to GPUs. Ordered consumption
keeps labels, temporal triples, domain quotas and resume sample order unchanged.
Worker errors propagate to the trainer. Lookahead is bounded and cancelled/drained
on shutdown. A resumed job reconstructs lookahead from the checkpoint step.

This removes Python contention between CPU geometry preparation and GPU launches.
It does not precompute the entire million-anchor graph corpus or change TDA targets.

## GPU execution and numerical audit

The deployed encoder executor is byte-identical to the original frozen two-GPU
implementation, with microbatch 512. This upgrade changes CPU preparation and
prefetching only. Global VICReg statistics, tensor supervision, gradient replay,
GPU synchronization and optimizer mathematics remain unchanged.

Event-based scheduling, coalesced transfers and microbatch 1,024 were investigated
but are not deployed. A stricter audit found that the old compiled BF16/cuEquivariance
executor itself produces variable encoder gradients on a fixed trained checkpoint
and fixed input batch: relative L2 differences were 0.223 for an old/old repeat,
0.231 for old/candidate, and 0.219 for candidate/candidate. Parameters were held
fixed with zero learning rate. Losses changed by only approximately 1e-6. Most of
the absolute gradient difference was in the invariant pooling/compression layers.
These are measurements on one checkpoint and one spatial batch; the underlying
numerical cause and effect on learning are not established. The evidence is in
`technical/gradient_audit.json` and retained gradient tensors. Small losses alone
are insufficient to certify gradient reproducibility here.

For this speedup, the GPU implementation is preserved and input parity is checked
bitwise instead. There is no unverified numerical change in the production handoff.

## Safe continuation and evidence

The run is checkpointed before switching. Its previous checkpoint, identity and
status are preserved. An explicit execution-transition receipt pins the exact old
and replacement identities and checkpoint hash. The trainer rejects altered data,
models, library versions, sampling, schedule or unreviewed source files. Loading
continues the existing optimizer, scheduler, RNG, validation history and W&B run.
It does not reset training or extend the five-epoch budget.

Source: `src/training_methods/shared_pretraining/input_pipeline.py`,
`parallel_mace.py` and `resume.py`. Real-data compiled parity and a matched runtime
comparison are retained under
`output/shared_pretraining/mace-dual-optimized-checks-20260919/technical/`.
Comparisons run separately from production; there is no benchmark stage inside
training. Metric definitions and execution semantics are in
[the shared training contract](metrics/shared_pretraining.md).

## Measured result and production handoff

The matched compiled BF16 comparison used the same checkpoint (update 1,456),
global batch 2,048, microbatch 512 and sampled updates 1,000–1,031 on node61's
two RTX PRO 6000 Blackwell GPUs. The first 16 updates of each variant were
discarded. The remaining 16 contained the same spatial/temporal sequence.

| Input preparation | Mean update plus input wait | Mean exposed input wait |
| --- | ---: | ---: |
| Previous threaded preparation | 1.5772 s | 0.26335 s |
| Four processes, six-batch lookahead | 0.8208 s | 0.01012 s |

Measured throughput improved 1.922×, a 48.0% reduction in update time. This is
one short paired comparison, not a whole-epoch speed estimate; startup,
validation, checkpointing and W&B logging are excluded. CPU regression tests
also ran during part of the baseline window, so this is an operational estimate
rather than an isolated CPU benchmark. The production continuation supplies a
further check of actual input wait. Full-batch spatial and temporal inputs were
bitwise identical, including targets and sampled indices; 39 targeted CPU tests
passed. The unchanged GPU executor is not claimed to have deterministic gradients.

Production was gracefully checkpointed at update **1,611 / 2,486**, with its
checkpoint, selected model, identity and metric definitions preserved under
`technical/execution_history/pre-optimization-step1611/`. The five-epoch budget
continues using `configs/shared_pretraining/mace_optimized_dual/campaign.json`.
The exact continuation receipt and frozen executable code live in
`output/shared_pretraining/mace-dual-optimized-campaign-20260919/technical/`.
Training keeps the [existing W&B run](https://wandb.ai/teshbek/PointCloudMaterials/runs/mace-4x-dual-0919).

The detached continuation in allocation 999611 reached update 1,664 and completed
its next validation: present selection 0.12520 versus the unchanged training-mean
baseline 0.51039. Its last 32 measured production updates averaged 0.8495 s of
update computation and 0.01490 s of exposed input wait. These use different rows
from the matched comparison above, but confirm the reduced starvation during
actual training. The measurements are retained in `technical/production-handoff.json`
under the optimization-check run. No monitoring task is scheduled after handoff.
