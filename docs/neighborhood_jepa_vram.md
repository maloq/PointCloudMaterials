# Neighborhood JEPA execution on GPUs with more memory

`src/training_methods/neighborhood_jepa/execution.py` implements three independent
execution controls for the regularization and multi-horizon MACE protocols:

1. Encoder microbatch size is independent of the statistical anchor batch. At
   B512, fourteen views still mean 7,168 observations and seventeen views mean
   8,704. Changing the encoder microbatch does not accumulate approximate
   regularizer statistics over chunks.
2. `retain_chunks` saves the autograd graphs for the first specified number of
   encoder chunks. After computing the full-batch objective once, its export
   gradients backpropagate through those retained graphs. Other chunks replay
   their encoder forwards. Keeping all chunks eliminates replay entirely.
3. `gpu_cache` uploads packed positions, support weights, species, edges, graph
   indices and scales once per update. Replay reuses those device tensors.
   Packing constructs only fields consumed by the neighborhood encoder; padded
   coordinates and dummy decoder targets are omitted. The input cache lasts one
   update and never stores learned representations or substitutes old targets.

Prediction and teacher branches remain jointly trainable. Causal input rules,
physical anchors, losses, data splits, optimizer schedules, seed and checkpoint
selection are unchanged. BF16 scalar operations and protected FP32 geometry are
unchanged. Different chunk sizes can introduce ordinary floating-point reduction
differences; they are not claimed to be bitwise identical.

The optional `execution_profiles` recipe field lists explicit offline-tested
tiers with `min_vram_GiB`, `microbatch`, `retain_chunks` and `gpu_cache`. The runtime
chooses the highest eligible memory tier without running a benchmark. Resolved
hardware and profile are recorded in each fit's `executions/` receipts. Profiles
are part of the immutable configuration; changing execution tier on another GPU
preserves optimizer, objective, RNG and update state when resuming that recipe.
No OOM-catching fallback changes a scientific run silently.

The compiled inference forward is explicitly initialized before retaining any
training graphs, on fresh starts and resumes. A full B512 check exposed a BF16
startup-order discrepancy that the small tests did not catch. Initializing the
inference path first (as the original replay algorithm does) restored agreement;
the priming helper preserves parameters and CPU/CUDA RNG state. This is compiler
initialization, not a training-time hardware benchmark.

The runtime also compares canonical manifest digests on resume. Its previous
Python-object comparison rejected unchanged layout metadata because JSON loads
tuples as lists. Model, optimizer, objective, step and RNG restoration remain
strict; genuine configuration or source changes still reject the checkpoint.

Run hardware measurements separately, for example:

```bash
conda activate pointnet-torch214
python -m src.training_methods.neighborhood_jepa.profile_execution \
  --config configs/neighborhood_jepa/multihorizon_20260920/study.json \
  --microbatch 512 --retain-chunks 4 --gpu-cache --steps 3 \
  --output output/maintenance/neighborhood-vram-20260920/technical/retained512x4.json
```

The `--reference --microbatch 128` case uses the original full replay and input
packing. The profiler uses actual B512 training batches, compiled BF16 width64
MACE, AdamW updates and the VICReg multi-horizon objective. Compilation/warmup,
host wait and steady update time are reported separately. First-batch gradients
are saved to compare numerical agreement before updates. There is no W&B
benchmark logging or benchmark call inside training.

Correctness tests: `tests/test_neighborhood_execution.py` verifies exact packed
input and target equivalence, anchor/view order, censoring masks, GPU storage
reuse, forward-count savings, objective state and joint gradients across SIGReg,
VICReg and Epi regularization, including compiled BF16. Existing causality and
rotation tests remain applicable because the encoder math is unchanged.

Deployment uses a new frozen source release for unstarted jobs. Historical
snapshots and started fits retain their original producer. This combines with
the separate shard-grouping and selection-view improvements described in
[the queue workflow](neighborhood_jepa_regularization.md).

## Selected memory tiers

The active recipes keep B512 and select these execution settings by reported
device memory (GiB):

| Minimum device memory | Encoder microbatch | Retained chunks | Raw GPU input cache |
| --- | ---: | ---: | --- |
| 80 GiB | 128 | 16 | Enabled |
| 40 GiB | 128 | 6 | Enabled |

Larger encoder microbatches of 256 and 512 are implemented and tested, but 128
was faster on the measured RTX PRO 6000 workload. Thus extra memory is spent on
retained activations by default. The high-memory setting keeps 2,048 of 8,704
observation graphs for backward; the statistical batch remains 512 anchors.
The tiers are conservative on larger devices such as H200; their speed advantage
has only been measured on RTX PRO 6000, not assumed for other hardware.

Measurements and deployment receipts:
[memory benchmark](../output/maintenance/neighborhood-vram-20260920/README.md).
