# Structural-path crystallization queue

Environment: conda `pointnet-torch214`. Recipe:
`configs/crystallization_transfer/mace_paths_20260919.json`.

```bash
python -m src.research.crystallization_paths.queue submit \
  --config configs/crystallization_transfer/mace_paths_20260919.json
```

The ten-fit companion queue submits two one-GPU, eight-CPU, 64GB RAM, 16-hour Slurm
slots on RTX6000PRO/H100/L40S. It does not alter or interrupt active transfer queues.
Submission freezes executable code, the validation-selected reference settings,
source identities and parent checkpoint. Workers lock-claim missing future-center
extractions, then fits. All original observations remain in their immutable cache.

`output/crystallization_transfer/mace-paths-20260919/technical` retains plan,
queue, submission IDs, worker states, fit statuses, checkpoints and raw predictions.
The future-center cache is registered as `crystallization-paths-mace-20260919`.
Existing cache supplies center embeddings through498ps; only missing frames through
594ps need inference. No simulation is generated. A cached-anchor replay checks the
compiled BF16/cuEquivariance producer against the existing features at each source.

Once prepared, deduplicated source timelines stay on the GPU. Windows gather observed
and future indices independently on device; no per-update raw-graph loading or MACE
forward is needed. The small forecast heads and their loss calculations use FP32.
This isolates forecasting from mixed-precision encoder changes.

Fits save exact optimizer/model/RNG resumes every epoch and on deadline/signals;
training is resumed with the same frozen code. Evaluation uses the selected checkpoint
and restarts if interrupted. Failed tasks stop the worker with a traceback. Another
worker never silently substitutes failed settings. Inspect remaining statuses before
submitting further allocation slots; a 16-hour request is a cap, not a runtime forecast.

Manual continuation in an allocation, preserving Slurm's CUDA_VISIBLE_DEVICES:

```bash
cd output/crystallization_transfer/mace-paths-20260919/technical/code
python -m src.research.crystallization_paths.queue worker \
  --config configs/crystallization_transfer/mace_paths_20260919.json --lane 2
```

Tables include frozen metric definitions and hashes. Dense CDFs and per-window path
errors permit paired later analysis; a small outcome-independent sample of generated
paths is retained in each fit's `sample-trajectories.npz`. No hardware benchmark is
run inside training.

## Targeted follow-up

The [diagnosis and protocol](../experiments/crystallization_transfer_20260919/PATH_REFINEMENT.md) motivate 30 one-seed screens capped at 12 epochs, then five validation-selected fits capped at 36 epochs. Early stopping can finish sooner. Submit two independent 16-hour GPU slots using:

```bash
python -m src.research.crystallization_paths.queue submit \
  --config configs/crystallization_transfer/mace_path_refinement_20260919.json
```

The completed future cache is reused after checking its plan identity, parent checkpoint and checksums. Missing reused artifacts fail rather than trigger extraction. Timelines stay resident across fits; history ablations change observation offsets. Resume checkpoints include optimizer/RNG, early-stopping state and EMA when enabled. Promotions wait for all screens, constrain physical prediction error, then rank event Brier on selection sources. Test results never select promotions. Native fine-tuning/scratch experiments continue independently.

Reproduce selection-only diagnostic replays with:

```bash
python -m src.research.crystallization_paths.diagnose \
  --original output/crystallization_transfer/mace-paths-20260919 \
  --output output/crystallization_transfer/path-diagnosis-20260919
```
