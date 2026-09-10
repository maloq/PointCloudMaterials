# Pretrained MACE in the original VICReg pipeline — 2026-09-09

Question: does the pretrained small MACE backbone learn useful structural
coordinates when the original GeoFrame input normalization, projector and
Lightning VICReg training method are restored?

The [preceding audit](../../output/mace_vicreg_audit_20260909/REPORT.md) identified
strong material offsets in physical-coordinate, species-conditioned MLIP features.
This experiment uses `VICRegModule` and the registered `PretrainedMACEGeometry`
encoder. The public encoder accepts only `(batch, 80, 3)` coordinates. The existing
spatiotemporal loader returns anchor, spatial-neighbor and same-atom temporal
views, already divided by the original source cutoff radii:
Al 9.192189 Å, Mg 10.169428 Å and Ta 9.388275 Å.

Inside the backbone, every cloud is multiplied by the **same** 9.192189 Å
reference length, and every node uses the same pretrained Al channel. Actual
material identity and source radius do not enter the encoder. This shared unit
conversion puts dimensionless coordinates in the MLIP's distance range without
undoing normalization across materials. The pretrained 5 Å interaction graph,
two scalar blocks and full 80-atom mean pooling are retained. The fixed channel
is an internal part of the pretrained weights, not an element-type input.

Configuration: [vicreg_pretrained_mace_geometry.yaml](../../configs/vicreg_pretrained_mace_geometry.yaml).
The MACE backbone has 8,219,792 trainable parameters; the original 128D MLP
projector adds 66,048. No fitted feature scaler or additional objective is used.
VICReg uses the repository's exact population-variance convention, covariance
normalization and pair averaging, with coefficients 25/25/1. The public
representation is the original pipeline's 128D projector output; the raw
backbone output has 256 dimensions. Evaluation uses BatchNorm running statistics.

Data: the existing normalized Al/Mg/Ta cache at
`/home/ids/vmorozov/experiments/geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905/views`.
Only recorded lag-step 1 pairs are selected, corresponding to 0.1 ps for all
three materials. Each view has exactly 80 points. There are **294,912 training
triplets** (98,304 per material) and **18,432 validation triplets** (6,144 per
material), with the original split preserved. Counts describe cached triplets,
not distinct atoms. The existing cache has equal material counts; training uses
ordinary shuffled batches, without a weighted sampler or element balancing.
Each epoch encodes 884,736 view occurrences; views may recur across triplets.

Training: fresh MLIP initialization, 24 epochs, batch 512, 576 optimizer updates
per epoch, 13,824 updates total. AdamW peak LR 1e-3, weight decay 0.04, gradient
clip 1, three warmup epochs from 1e-4 followed by cosine to 1e-6. This uses the
original shared **epoch-based** scheduler. Original view jitter/mirroring and
projector are unchanged. Selective compensated radial BF16 is retained inside
MACE; geometry, tensor products, VICReg statistics and the Lightning precision
remain FP32. There is no gradient replay and no TDA loss in this diagnostic run.

The maintained original entry point runs training and then the standard
`configs/analysis/static.yaml` analysis on its best validation checkpoint.
Results stay in `output/mace_original_vicreg_20260909/train/analysis` in the repo.
The analysis uses the checkpoint's usual 128D representation and the standard
six-frame static Al input configuration. No alternative MACE analysis adapter
or custom clustering workflow is involved.

```bash
PYTHONPATH=. conda run -n pointnet python experiments/mace_original_vicreg_20260909/preflight.py
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_original_vicreg_20260909/run_spec.json
```

The preflight verifies the coordinate-only contract, fixed-channel equivalence,
the actual pair loss, and eight real Lightning optimizer steps. It reports
finite nonzero backbone/projector gradients, loss 20.2483 → 19.4945,
0.430 seconds per warmed step, and 59.32 GiB peak allocated GPU memory.
The eight-step validation loss is not a trained-quality comparison: the
projector's BatchNorm statistics are still warming up. Results are saved in
`output/mace_original_vicreg_20260909/preflight/result.json`. An earlier
evaluation-mode diagnostic is retained separately and is not the training test.

Files here are versioned experiment records and reproduction diagnostics.
The encoder registration and temporal-lag selection in `src/` are maintained
implementation. Logs, checkpoints, execution records and analysis exports are
generated artifacts under the run output directory.

Launched detached on node53 in allocation 986459 at 17:58 Paris time.
[Online W&B run](https://wandb.ai/teshbek/PointCloudMaterials/runs/louqqyow).
At the startup check, W&B reported optimizer step 99, training loss 15.223 and
LR 1e-4 in warmup. These are startup observations, not final results. Eleven
targeted tests passed. Training is estimated to take about 1 hour 45 minutes,
followed by analysis; see the live W&B page and generated execution records.

The automatic analysis default in `src/analysis/config.py` was corrected from
the nonexistent `checkpoint_analysis.yaml` to the existing `static.yaml`.

## Analysis recovery — September 9, 20:52 Paris

All 24 epochs finished at 19:33. Analysis failed on its first inference batch
because the compiled BF16 radial layers hit Dynamo's shared recompilation limit
after the training phase. The selected best checkpoint is zero-based epoch 13,
step 8,064. It remains the correct checkpoint for the configured best-validation
analysis. The original `last.ckpt` also points to that epoch because this
Lightning version did not refresh it when validation worsened. Future runs now
explicitly save the actual final optimizer state after `fit`, before analysis.

Analysis config construction now disables radial compilation for this encoder,
while retaining compensated BF16 arithmetic and the trained weights. The
checkpoint's compiled/eager projected embeddings agree to 7.2e-6 maximum
absolute error on the verification sample; batch sizes 1/2/5/7/16/31 also agree.
Reproduce this check with:

```bash
PYTHONPATH=. conda run -n pointnet python experiments/mace_original_vicreg_20260909/verify_analysis_inference.py
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_original_vicreg_20260909/analysis_recovery_spec.json
```

The recovery spec uses the existing standard analysis command and saved best
checkpoint. It writes to the original `train/analysis/` location, with execution
records and verification under `output/mace_original_vicreg_20260909/analysis_recovery/`.
The queued TDA run now depends on this recovered analysis completing successfully.
Recovery YAML/spec and verification code are experiment records; outputs are
generated diagnostics. No training epochs are replayed.
