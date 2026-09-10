# Plain 80-atom MACE — 2026-09-09

Question: can a simpler, consistent encoder/target protocol recover useful spatial
structure while learning short-time invariance and liquid topology?

The model starts fresh from MACE-MP-0b2 small MLIP weights. It runs the ordinary
finite MACE graph on exactly 80 atoms and mean-pools both scalar node-feature
blocks into 256 coordinates. Every atom participates. The pretrained native
5 Å edge cutoff remains; there is no extra outer-radius taper, central pruning,
or learned context residual. No teacher, projector, forecast, or nuisance head.

Three training views: anchor, an independently centered spatial neighbor (one
of the six nearest atoms), and the same atom at the short stored lag. Temporal
invariance uses only verified 0.1 ps pairs; Al shooting 0.3 ps pairs are excluded
from that term. Their anchor/spatial views still train the encoder.

The fixed loss is `25 * mean(spatial MSE, eligible temporal MSE) + 25 * variance
penalty + covariance penalty`, with one shared average variance/covariance budget
across the three views. No per-element centering or weights. Beginning in epoch
6, add TDA MSE with fixed coefficient 1. The TDA head has no gradients, optimizer
state, weight decay, or nonzero LR during epochs 1–5.

TDA is recomputed on the same stored 80-atom float16 coordinates, decoded to
float32: full alpha-complex H0/H1/H2 persistence images, normalized by 79.
A 32-component PCA/scaler is fitted on uniform training rows only. Feature scaling
is fixed from fresh MLIP features on uniform training anchors. The fourth stored
view is used only for frozen post-training prediction probes.

| Material | Training anchors per epoch | Validation anchors |
|---|---:|---:|
| Al | 245,760 | 8,192 |
| Mg | 32,768 | 1,024 |
| Ta | 8,192 | 256 |

Each epoch visits all 286,720 stored anchors once, including the last partial
batch. No element balancing or oversampling. Batch 1,536, microbatch 512:
187 optimizer updates per epoch, 2,244 over 12 epochs. That is 3,440,640 anchor
exposures and 10,321,920 view exposures. These are exposures, not independent
states; exact deduplicated three-view counts are written to `data_summary.json`.
Validation uses all 9,472 stored anchors. Al dominates this natural mixture;
per-material validation metrics remain visible without affecting loss weights.

AdamW: encoder peak LR 1e-4, head peak LR 1e-3, weight decay 1e-5, gradient norm
clip 5. Per-update cosine decay to 1e-6; encoder warms up for one epoch, TDA head
warms up for half an epoch beginning at epoch six. Select the lowest validation
loss among epochs 6–12, so TDA-off losses cannot win checkpoint selection.
Compensated BF16 radial matrices, FP32 geometry/losses, cuEquivariance, compiled
radial MLPs, GPU-resident data, and exact gradient replay are retained.

## Reproduction

Activate conda `pointnet`, then from the repository root:

```bash
python -m src.training_methods.pretrained_mace --config experiments/mace_plain80_20260909/training.json --stage prepare
python -m src.training_methods.pretrained_mace --config experiments/mace_plain80_20260909/training.json --stage preflight
python scripts/experiment_registry.py run --spec experiments/mace_plain80_20260909/run_spec.json
```

The registry runs the allocation-aware queue: training, frozen topology/forecast
probes, then the existing full static-analysis pipeline on the encoder alone.
Online W&B is required. `training.json` contains the scientific settings;
`plan.json` contains the allocation/time budget. Fresh training refuses to
overwrite a previously initialized run directory.

## Checks and results

[Output/status](../../output/mace_plain80_20260909/) ·
[GPU preflight](../../output/mace_plain80_20260909/preflight.json) ·
[Tests](../../output/mace_plain80_20260909/tests.log)

Preflight checks the native MACE forward, rotation/permutation invariance,
contribution of atoms 66–80, exact cached TDA recomputation, direct-versus-cached
gradients before/after TDA activation, and finite real-batch peak-LR updates.
Final checks: 16 tests passed, real-MACE GPU preflight passed, and the existing
static-analysis checkpoint loader reproduced the encoder features. Measured
roughly 4 seconds/update and 21 GiB allocated GPU memory. The encoder has
8,219,792 trainable parameters; the TDA head adds 74,016. Training was launched
detached at 00:34 Paris, with online [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/plain809);
[launch verification](../../output/mace_plain80_20260909/launch_verification.json)
confirms actual optimizer steps, head LR zero, and the 2,244-update schedule.
Initial loss is dominated by covariance of correlated MLIP channels; the total
is not a normalized accuracy score. Inspect its fixed components and held-out
per-material rank/coherence/TDA scores, including the change at epoch six.

Static Al uses the requested existing analysis configuration and original grid.
Several static frames are ancestors of training continuations; this analysis is
descriptive, not an independent generalization test. Ta validation uses later
times/disjoint IDs of one source. No PTM/HCP label is treated as definitive truth.

## Code ownership and retirement

Maintained code is indexed in [scripts/README.md](../../scripts/README.md).
This directory contains experiment records/configs only. Generated diagnostics,
logs, checkpoints and results stay under the output directory. There is one
MACE trainer, dataset reader/target producer, and encoder implementation.
Retired loss-balancing, PCGrad, topology-attraction/ranking, nuisance/prediction,
resume/selection, smooth-support and diagnostic code was deleted; no legacy
trainer remains. Completed experiment results and their recorded source/config
snapshots remain available for historical reproduction.

The 2026-09-09 cleanup removed 22 retired source/test/diagnostic files
(2,194 lines), in addition to shortening the existing encoder/data/training
modules. The pre-cleanup research implementation is preserved in the immutable
[source snapshot](../../output/mace_plain80_20260909/runs/plain/tracking/20260908T222343.302282Z/source.tar.gz),
not as a live legacy trainer.
