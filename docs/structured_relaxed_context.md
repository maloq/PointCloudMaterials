# Relaxed symmetric-context workflow

The active route reuses completed archived cells; it never launches relaxation.
Use conda `pointnet-torch214` and
`configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json`:

```bash
python -m src.research.structured_context.reuse freeze --config configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json
python -m src.research.structured_context.reuse precision --config configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json
python -m src.research.structured_context.reuse verify --config configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json
python -m src.research.structured_context.reuse submit --config configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json --allocation ALLOCATION_ID
```

Run precision/verification inside a GPU allocation. Submission freezes executable
code and starts a detached Slurm step inside the specified existing allocation.
It extracts features once, then submits three additional GPU fit workers and
continues fitting itself. No new GPU is reserved waiting for data preparation.
The eight fits share locks and resume exact checkpoints. Inspect `technical/launches.json`,
`fit-launches.json`, `lane-*.json`, `runs/*/status.json`, and `RESULTS.md`.

Continue from the immutable code snapshot with the same module's `worker` command;
`--dispatch` enables idempotent submission of any missing fitting workers. The
original bash allocation is never cancelled by this workflow. Preflight uses a
separate four-source cache and run directory and never supplies scientific scores.

Archived cells preserve source/frame/potential and conversion receipts; array
checksums are verified before feature extraction. Original precise dumps are not
required. Archive quantization is measured on separate saved precise benchmark
cells. The scientific inventory is frozen, even if other data producers continue.

[Active research protocol](../experiments/structured_relaxed_reuse_20260922/README.md).
The earlier fresh-relaxation preparation proposal in `structured_context.relaxed`
is superseded. Its bounded job1003685 was cancelled while pending; it produced no
new cells. Its execution checks remain under `symmetric-relaxed-mace-20260921`.
