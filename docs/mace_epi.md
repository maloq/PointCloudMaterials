# Running paired MACE + Epi

Use conda `pointnet-torch214` and `configs/mace_epi/campaign.json`. The
[scientific protocol](../experiments/mace_paired_epi_20260923/README.md) defines
three treatments, two seeds and24 complete passes per fit.

```bash
python -m pytest tests/test_mace_epi.py -q
python -m src.research.mace_epi.queue submit --config configs/mace_epi/campaign.json
```

Submission copies the existing native numerical and8×spatial input caches,
freezes source/config/metric definitions, and submits a separate production-batch
GPU correctness gate. Two Slurm workers depend on its successful completion;
each has one exact-resume continuation. Failed scientific tasks retain explicit
receipts and are not silently retried. Completed queues exit immediately.
Current allocations and frozen parameter-search jobs are untouched.

The large run is on the configured analysis storage under
`encoder_research/mace-vicreg-epi-20260923/`, avoiding the full home quota.
`technical/launch.json` records job IDs and frozen commands. Each fit saves
`technical/fits/NAME/epoch-000.pt`, `epoch-004.pt`, `epoch-012.pt`, `epoch-024.pt`
and exact optimizer/RNG resume state in `last.pt`. Training history is
`training.jsonl`; progress is `status.json`. The full native assay and liquid
supplements are in `technical/evaluations/` and `technical/supplements/`.
Readable native results are `index.html`, `plots/`, `tables/summary.csv`.

To resume on another allocated GPU, use the active orchestration command in
`technical/orchestration-v3/launch.json`, with its original `PCM_PROJECT_ROOT`.
The wrapper invokes the original frozen scientific producer/config.
Per-fit locks prevent duplicate fits. Preserve frozen code and configuration;
do not relaunch the active recipe over an existing run. Training stops before
the allocation deadline and restores the exact next permutation/batch.

Pip's disposable download cache was cleared during setup to recover1.76GB;
installed environments and research artifacts were preserved. Compiler caches
for these jobs use node-local temporary storage. Dataset registry refresh
initially hit the home quota and completed after clearing the download cache;
the native data and reservoir manifests are also checked directly by this workflow.

## Submission on23September2026

GPU preflight:1006454 (node60, RTX6000PRO). Training workers:1006455 and1006457;
exact-resume continuations:1006456 and1006458. Training depends on successful
preflight. Seven CPU correctness tests passed before submission.

GPU preflight passed all three objectives atB512 on the RTX PRO6000, including
exact initial encoder identity and strict native-export reload. Worker1006455
started the first Epi fit on node60 and reached update32 (0.5 complete passes)
with finite losses and gradients. Worker1006457 is queued for resources.

## Cached orchestration release

Report setup was repeatedly resolving the entire model catalogue. The new
`cached_queue.py` loads it once per process and returns independent copies,
including in child stages. The original frozen source and experiment identity
remain untouched. Eight CPU tests pass, including mutation isolation for cached
configurations.

The first fit checkpointed exactly at update266 (4.15625 passes), with optimizer
and RNG state preserved. Original workers1006455–1006458 were superseded by
workers1006472/1006474 and continuations1006473/1006475. The active receipt is
`technical/orchestration-v2/launch.json`; the original launch points to it.
The existing successful GPU gate is verified by configuration hash at each worker
start. Initial report generation was interrupted after submission to replace its
repeated catalogue resolution; this did not cancel the submitted fits.

## Shutdown recovery on24September2026

The queue audit found that both Epi fits had completed all1536 updates/24 epochs,
but their training subprocesses had not exited. On node61, job1006474 was waiting
for its multiprocessing resource tracker with0% GPU use, despite its completion
receipt being over eight hours old. Job1006473 could never start because its
predecessor1006472 had failed; that predecessor's final Slurm accounting record
was unavailable during the audit.

The orchestration wrapper now flushes output and exits after the synchronous
stage returns, preserving normal, failed and checkpointed exit codes. Eleven
tests passed, including subprocess tests with deliberately blocked finalizers.
The completed fit also replayed its completion check and exited successfully on
node61. All saved milestone checkpoints were verified for identity and finite
model tensors; the final checkpoint hashes and24-epoch counters matched their
completion receipts.

The stalled job1006474 and pending jobs1006473/1006475 were superseded by detached
workers1007451/1007452, each capped at12 hours. The original frozen scientific
producer, config and checkpoints remain unchanged. Completed fits are skipped
and proceed to evaluation. The recovery receipt and wrapper hash are in
`technical/orchestration-v3/launch.json`; previous launch receipts point to it.
