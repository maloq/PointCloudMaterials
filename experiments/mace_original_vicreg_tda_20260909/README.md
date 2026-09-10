# Original VICReg + TDA on normalized MACE — 2026-09-09

Question: does topology supervision improve the same 128D structural coordinates
learned by the successful original VICReg pipeline?

This is a fresh matched run from the same small MLIP checkpoint, after the
pure-VICReg run and its static analysis finish. It keeps normalized 80-point
Al/Mg/Ta views, the fixed internal species channel, original projector, spatial
and 0.1 ps temporal VICReg, AdamW, augmentation, seed and schedule. There is no
element input, sampling rebalance, loss balancing, teacher or forecasting loss.

Configuration: [vicreg_pretrained_mace_geometry_tda.yaml](../../configs/vicreg_pretrained_mace_geometry_tda.yaml).
24 epochs, batch 512, 576 updates/epoch, 13,824 updates total. There are 294,912
training triplets and 18,432 validation triplets, each containing three 80-point
views. TDA is active **from epoch 1** with a fixed coefficient of **1**:

`total_loss = original_spatial_temporal_VICReg + TDA_MSE`.

The TDA head is `128 → 256 → SiLU → 32` (41,248 parameters), attached to the
same projected representation exported for analysis. It reuses the three
projector outputs already computed by VICReg, so it adds no BatchNorm updates
and no extra MACE forward pass. Creating the head preserves the baseline's
random-generator state for encoder/projector initialization and sampling.
It uses the same optimizer and warmup/cosine schedule as the rest of the model.

Targets reuse the existing full alpha-complex descriptor: H0 death-radius
curve and H1/H2 persistence images, totaling 144 values. All 80 supplied atoms
participate; there is no 65-atom crop or radial taper. Coordinates are decoded
from the existing normalized cache and multiplied by the same shared Al
reference length used inside MACE. Material identity and per-material physical
units are not reintroduced. Targets describe the unaugmented cached point set;
input jitter/mirroring remain the original training augmentations.

As in the previous TDA protocol, a 32-component PCA with whitening is fitted
only on 12,288 uniformly selected training triplets (36,864 views). Validation
data never fit the target scaler. The PCA mean, components, scales, explained
variance, checksums and row correspondence are retained in the target cache:
`/home/ids/vmorozov/experiments/mace_original_vicreg_tda_20260909/targets`.
There are 940,032 target view occurrences across training and validation.

W&B reports `train/tda_mse`, `val/tda_mse` and the matching
`tda_mean_baseline_mse`, which is the error of always predicting the training
mean. A useful head must beat that baseline on validation. The original VICReg
components remain separately logged; the TDA head adds no inference outputs.
The best checkpoint is selected using the combined validation objective.

The maintained registry executes two explicit specs. CPU preparation can finish
while the current GPU run continues. The new GPU run waits for the **outer
execution record** of the predecessor, which covers both its training and its
standard static analysis, and also waits for successful target preparation.
Failed/dead predecessors abort the queue rather than silently launching the run.
The dependency deadline is September 10 at 01:30 Paris, leaving 4.5 hours before
the allocation ends at 06:00:55; the normal expected start is much earlier.

```bash
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_original_vicreg_tda_20260909/prepare_spec.json
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_original_vicreg_tda_20260909/run_spec.json \
  --wait-for-dependencies-until 2026-09-10T01:30:00+02:00
```

Both commands are launched detached on node53. W&B is online when training
starts. The original training entry point then runs `configs/analysis/static.yaml`
on the new best checkpoint. All checkpoints, logs, and analysis results stay
under `output/mace_original_vicreg_tda_20260909/` in this repo; static figures are
under `train/analysis/`. Queue status is `queue_status.json`; final execution
state is `execution/run_record.json`.

Verification exercises the actual original Lightning module with a small CPU
encoder: exact unchanged VICReg loss, additive TDA loss, finite gradients and
updates in backbone/projector/TDA head, one projector call per view, and strict
static-inference checkpoint loading without the target cache. Target tests
check the last/outer atom, normalization units, split isolation and row/view
alignment. Queue tests check predecessor analysis completion, failures, dead
processes and deadlines. Logs are in the output directory.

Files here are versioned experiment records. The optional head and cache loader
are maintained training/data implementation. `src/data_utils/spatiotemporal_tda.py`
is the maintained producer for this distinct normalized three-view cache; it
reuses the existing persistence descriptor. Registry dependency waiting extends
the existing maintained command. Targets, status, logs and analysis exports are
generated artifacts, not extra training runners.

Launch verification: preparation completed in 381.85 seconds, all 940,032 target
views are cached, and PCA32 retains 99.9979% of image variance. The constant-mean
validation baseline is 0.98059. All 21 targeted tests passed. At 19:16 Paris the
detached controller was alive, waiting only for the predecessor's combined
training/analysis record. See the [output record](../../output/mace_original_vicreg_tda_20260909/README.md).

## Restart after predecessor analysis failure

The original queue stopped at 19:34 because baseline analysis hit Dynamo's
recompilation limit; the TDA training had not started. The recovery uses
`recovery_spec.json`, which waits for the baseline's new analysis-only execution
record and reuses the already completed TDA cache. Its training command, config,
seed and original `train/` output location are unchanged. Execution and queue
records for this attempt are under `output/mace_original_vicreg_tda_20260909/recovery/`.

```bash
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_original_vicreg_tda_20260909/recovery_spec.json \
  --wait-for-dependencies-until 2026-09-10T01:30:00+02:00
```

The shared real-GPU preflight also supports this configuration:

```bash
PYTHONPATH=. conda run -n pointnet python experiments/mace_original_vicreg_20260909/preflight.py \
  --config configs/vicreg_pretrained_mace_geometry_tda.yaml \
  --output output/mace_original_vicreg_tda_20260909/recovery/preflight
```

Analysis now uses eager radial layers to avoid exhausting the training compiler
cache. Training keeps its existing compilation and BF16 settings. The trainer
also saves the actual final checkpoint before analysis, even if the best epoch
occurred earlier.
