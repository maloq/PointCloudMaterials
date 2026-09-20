# Equivariant neighborhood JEPA execution

Use conda `pointnet-torch214`. The current campaign runs two independent MACE fits
on node53's two H100 NVLs in allocation 1000616, with approximately 23 hours
allocated at launch. It does not request other nodes or run GATr.

```bash
python -m src.training_methods.neighborhood_jepa.prepare \
  --config configs/neighborhood_jepa/data_20260920.json
python -m src.training_methods.neighborhood_jepa.queue submit \
  --config configs/neighborhood_jepa/mace_20260920.json
```

Preparation was completed and checked before submission. It reuses registered raw
trajectories and existing center targets. The new derived collection is registered
as `neighborhood-jepa-tracked-six-65536-20260920`; its plan pins the parent manifest,
source identities, lineages, atom IDs and physical times. Shards have checksums.
Positions/edges in this training cache are not simulation restarts.

Submission freezes source/config/metric definitions, and a detached Slurm step
receives **both allocated GPUs**. The coordinator assigns one visible GPU to each
worker. File locks prevent duplicate fits; a failed fit stops that lane with its
traceback and blocks promotion. No worker substitutes settings after a failure.
Current `output/neighborhood_jepa/mace-20260920/technical` contains launch/worker
states, W&B IDs, specifications, checkpoint/optimizer/RNG state and predictions.
Tables retain frozen metric definitions and implementation hashes.

A bounded lazy mmap cache avoids exhausting file descriptors. Spawned workers
reopen arrays lazily rather than serializing parent memmaps; tensor sharing uses
file-system handles to avoid one open descriptor per prefetched tensor. Four persistent
spawned CPU workers prefetch packed batches per GPU. MACE runs compiled with
cuEquivariance. Gradient replay preserves the global grouped objective without
holding all 21-view encoder activation graphs simultaneously. There is no hardware
benchmark inside production training. Validation does synchronous deterministic
training-moment calibration and evaluates every fixed selection anchor.

The deadline reserves time for exact optimizer/RNG checkpoints. `last.pt` supports
resuming from the immutable campaign code with its same frozen configuration;
`best.pt` includes the predictor and normalization; `encoder.pt` exports the
snapshot encoder and its tensor layout. To continue after interruption, use the
frozen code's `queue worker --config ... --lane ...` inside a valid one-GPU
allocation, preserving its CUDA_VISIBLE_DEVICES. Do not resubmit the original
`submit` command or silently change its schedule/cache.

W&B group: `neighborhood-jepa-20260920` in `teshbek/PointCloudMaterials`.
Dashboards contain weighted loss contributions and fixed physical/TDA selection
metrics; native W&B GPU monitoring supplies hardware plots. Current scientific
protocol: [neighborhood JEPA](../experiments/neighborhood_jepa_20260920/README.md).
