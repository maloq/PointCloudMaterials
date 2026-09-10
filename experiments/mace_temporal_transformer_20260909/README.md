# Temporal MACE transformer — September 9, 2026

Stopped at the user's request after 16 completed epochs; best checkpoint is
epoch 15. Frozen analysis and [failure diagnosis](DIAGNOSIS.md) are complete.
Regularization dominated gradients, and temporal-feature probe MSE worsened
from 0.958 initially to 1.123 after training; frozen MACE five-frame mean gives
0.934. No replacement training has been launched.

Question: can joint MACE and temporal attention recover relaxed local topology
from five causal observations more accurately than an instantaneous descriptor?

Configuration: [training.json](training.json). Reproduction, in `pointnet`:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_temporal_transformer_20260909/run_spec.json
```

The maintained MACE family command dispatches the explicit `temporal80`
protocol to `src/training_methods/mace_temporal.py`. `--stage all` prepares
histories, fits training-only scaling, runs a real-GPU gradient preflight,
trains, then performs temporal analysis. Individual stages use the same config:

```bash
python -m src.training_methods.pretrained_mace --config experiments/mace_temporal_transformer_20260909/training.json --stage analysis
```

For a new reproduction, set a fresh output/cache, W&B ID and an available
allocation's explicit deadline in the configuration. No optimizer restart
semantics are implied. Current allocation: H100 986459 on node53, with a
September 10 05:30:55 Paris safety deadline.

## Data and model

Reuse all 12 converged paired-cache shards from
`output/mace_thermal80_20260909/data/manifest.json`: 26,624 training anchors
(Al 16,384, Mg 8,192, Ta 2,048) and 1,024 validation anchors. Each observation
is a five-frame history at offsets −0.4, −0.3, −0.2, −0.1, 0 ps. Follow the
anchor's recorded hot-selected 80 identities, centered on the same atom, using
each frame's periodic box. Final history frames must match the original hot
cache exactly. Relaxed targets remain those of the converged full periodic
fixed-cell minimization with the original generating EAM potential.

The history producer stores float16 offsets, exact integer frame/identity arrays,
quantization error and checksums. It preserves source artifacts. Only anchor
histories enter training. Spatial, short-time and future partner histories are
prepared for post-training measurements. The Ta source is shared across splits,
with different centers and verified non-overlapping observed time windows.

Architecture: shared small pretrained MACE, 256→128 frame projection, learned
physical-time encoding, two 4-head transformer blocks with feed-forward width
512, and the normalized final anchor token. No temporal mean pooling or
per-frame attraction. MACE remains material-aware (Al/Mg/Ta); this differs from
the separate normalized geometry-only original-VICReg experiments.

## Training and selection

Fresh end-to-end training for 24 epochs; batch 1,536, history microbatch 64,
18 steps/epoch, 432 updates, 638,976 anchor exposures and 3,194,880 frame
exposures. Exact gradient replay computes full-batch covariance while retaining
only one history microbatch's backbone activations. Transformer dropout is zero.

Loss: relaxed-anchor TDA MSE from epoch one + 25 × variance floor + covariance
penalty on the combined embedding. There is no spatial, temporal or hot/relaxed
attraction loss. Target PCA uses all training anchors (144→32); feature scaling
uses 2,048 training histories. Encoder/head peak LR 1e-4/1e-3, AdamW decay 1e-5,
gradient clipping 5, warmup/cosine to 1e-6. Select minimum validation TDA MSE.
MACE radial arithmetic uses the previously qualified compensated BF16 backend;
geometry and temporal transformer remain float32.

## Analysis and limitations

Analysis uses real held-out windows: trained-head relaxed TDA MSE and R²,
initial/final frozen ridge probes, a previous single-frame thermal80 reference,
within-material effective rank and spatial/temporal distance ratios. Repeated
anchor and reversed-past interventions test dependence on history without
retraining. Saved plots show training, latent PCA, topology predictions and
sampled validation-Al spatial clusters. These are descriptive clusters, not
identified phases or a full-cell static classification.

Al/Mg validation sources are held out within their campaigns; Ta is not an
independent source. Validation also selects the checkpoint, so it is not an
untouched test. The earlier single-frame reference had a different objective
and exposure budget; no controlled architecture superiority is claimed.

Outputs: [`output/mace_temporal_transformer_20260909`](../../output/mace_temporal_transformer_20260909/).
Status: [status.json](../../output/mace_temporal_transformer_20260909/status.json).
Findings will be in [analysis/RESULTS.md](../../output/mace_temporal_transformer_20260909/analysis/RESULTS.md).

File roles: this directory is an experiment record. The history producer,
trainer and analysis under `src/` and their tests are maintained implementation.
Generated histories, checkpoints, logs, figures and provenance live under the
output directory. No new runner or simulation is introduced.

## Launch verification

Launched at 23:07 Paris on September 9; tracked controller PID 3801699.
All 18 focused tests passed. The GPU preflight measured maximum replay-gradient
error 1.72e-5 and nonzero coordinate gradients through all five frames.
All 27,648 histories' anchors match their paired hot clouds exactly; maximum
additional local float16 quantization error is 0.00390625 Å. Training was verified
at update 15/432, about three seconds/update and 22 GB GPU memory.
[W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/mact5s09).
These are launch/preflight observations; completed findings belong in the report.
