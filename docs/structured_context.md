# Symmetric-context workflow

Use conda `pointnet-torch214`. From the repository root:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.research.structured_context.queue verify --config configs/crystallization_transfer/symmetric_mace_gatr_20260921.json
python -m src.research.structured_context.queue verify-pipeline --config configs/crystallization_transfer/symmetric_mace_gatr_20260921.json
python -m src.research.structured_context.queue submit --config configs/crystallization_transfer/symmetric_mace_gatr_20260921.json
```

Submit once from the configured current allocation. Submission freezes source,
configuration and metric documentation, detaches one worker into that allocation,
and requests four additional single-GPU workers. It leaves unrelated jobs alone.
Workers first share source extraction, then claim the eight forecast fits. Locks
prevent duplicate work. CPU frame preparation overlaps frozen GPU inference.
MACE uses its compiled BF16-protected producer; the requested historical GATr uses
FP32 with TF32 disabled to reproduce its original analysis. Training uses resident
timelines rather than rereading atomic coordinates on every update.

The cache lives in the registered IDS training-cache collection
`crystallization-structured-20260921`. Source progress is resumable at flushed frame
boundaries; completion includes checksums and replay against existing MACE centers.
The campaign freezes checkpoint provenance and the exact historical GATr producer.

Inspect `technical/lane-*.json`, per-source `progress.json`, Slurm logs and each
`technical/runs/FIT/status.json`. `RESULTS.md` is regenerated after each completed
fit; metric CSVs are under `tables/`. Checkpointed work is picked up by a surviving
worker. If all allocations expire, run the frozen module's `worker` stage from
`technical/code` in a new allocation with a unique `--lane` name. A failed source
or fit stops loudly and must be investigated before retrying.

No hardware benchmark is part of training. Preflight records replay precision,
finite gradients and free rollouts separately from scientific fit results.

Scientific specification: [experiment](../experiments/structured_context_20260921/README.md).

## Completed-model figures

`python -m src.research.structured_context.figures --config
configs/analysis/structured_context_figures.json` reuses saved predictions and
performs small inference-only replays on the configured device. It does not train.
Independent UMAPs are fitted on training-source features only. Output includes a
PNG gallery, separate captions, fixed-event offset metrics and source provenance.
`--stage render` reuses prepared arrays; historical irregular-context figures stay
in their original run. [Event/column definitions](metrics/structured_figures.md).
