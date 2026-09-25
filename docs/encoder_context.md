# Encoder training and context comparison

Recipe: `configs/encoder_context/al64_20260925/campaign.json`.
Implementation: `src/research/encoder_context/` using the existing supervised and
context trainers. [Scientific protocol](../experiments/encoder_context_epochs_20260925/README.md).

Use conda `pointnet-torch214` from the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TORCHINDUCTOR_COMPILE_THREADS=4
python -m src.research.encoder_context.queue prepare --config configs/encoder_context/al64_20260925/campaign.json
python -m src.research.encoder_context.queue check --config configs/encoder_context/al64_20260925/campaign.json
python -m src.research.encoder_context.queue launch --config configs/encoder_context/al64_20260925/campaign.json
```

Preparation freezes recipes/source identities and audits current-frame context
ancestry. Checks exercise real-data gradients without online runs. Launch
freezes code and starts two detached one-GPU Slurm tasks in the declared current
allocation, with checkpointed sequential stages. No scientific run is left
untracked: W&B online group `encoder-context-al64-epochs-20260925` in
`teshbek/PointCloudMaterials`. Training curves include epoch, objective, gradients,
learning rates, validation proper scores and AP diagnostics where labels apply.

Results: `${storage:analysis}/encoder_context/al64-epochs-20260925`.
Caches: `${storage:scratch}/training-cache/encoder-context/al64-epochs-20260925`
(outside the repository). Existing sealed inputs remain on IDS.
Structural positions fit directly in VRAM. Fixed-80 spatial graphs are constructed
on the GPU per batch; CPU/GPU parity checks cover outputs and gradients against
the compact graph producer. Context features are exported once per encoder and
shared by both predictors. One source shard is normalized at a time to bound host
RAM, with training-only feature statistics. Dense observed evaluation uses all
test centers/frames and is prepared once for the four observed encoders.

The lane states and per-stage logs/checkpoints live under `technical/`. A failed
stage stops its lane and records the traceback. Resume its `worker` command with
the frozen launch config and the same stage/method/domain, inside a valid GPU
allocation. Do not resume using edited workspace code. Completed stages are
verified and skipped; a timeout never marks an incomplete epoch budget complete.
The last lane collects tables/plots after both lanes complete. To collect manually:

```bash
python -m src.research.encoder_context.queue report --config configs/encoder_context/al64_20260925/campaign.json
```

Full evaluation tables are per encoder (`base-hot` / `base-cold`), while each
context predictor keeps its full predictions and separate legacy16 scores.
`RESULTS.md`, `tables/comparison.csv`, metric definitions and `plots/` summarize
the complete study. No context predictor is trained on an old encoder merely
because its feature cache already exists.

Storage policy (user update): no quota checks, archive scans or earlier-run
recovery dependencies in this queue. New cache directories are created directly
on SCRATCH. The independent historical-cache archive was stopped with its
originals retained; that operation is not a prerequisite for scientific work.
The current pretraining jobs keep their frozen code and online run IDs. A
machine-local path mapping directs their campaign's later cache writes to the
same external SCRATCH directory without restarting training.

The active run's `technical/continuation-receipt.json` records its detached
continuation. It waits only for the running structural fits and dense evaluation
inputs, then starts the eight encoder/predictor pipelines. It uses the frozen
bootstrap code and the external-cache mapping recorded in
`technical/cache-location.json`; no earlier-run export or storage archive blocks
it. Historical interrupted Al64 exports remain separately marked incomplete.
