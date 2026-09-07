# GFv2 spatial and temporal VICReg — September 5, 2026

## Post-training audit

The 60-epoch run completed. Its best validation checkpoint is zero-based epoch 3.
An audit found that `last.ckpt` also contains epoch 3: this Lightning version only
updated it when a top-k checkpoint was saved. The epoch-59 weights are unavailable.
The runner now explicitly saves every latest epoch and the final model; a
regression test verifies latest-checkpoint saving despite worsening validation.
The old run's code snapshot and checkpoints are preserved.

Completed Al/Mg/Ta post-training analyses, controlled baseline comparisons and
interpretation are in
`/home/ids/vmorozov/experiments/geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905/post_training/RESULTS.md`.
Projector temporal drift improved, but raw encoder effective rank declined and
static Ta clustering separation worsened. The result supports further downstream
testing, not an unconditional replacement of the original encoder.

This experiment fine-tunes the GFv2 encoder and VICReg projector used in the
current temporal analyses. Initialization is the epoch-34 checkpoint under
`output/detached/vicreg_geoframe_v2_factor_sn_grouped_scratch_20260831_160541/`.
All 168 encoder/projector state tensors are restored. Optimizer and epoch counters
start fresh; the old FactorVAE discriminator is excluded from this VICReg run.

The ordinary static VICReg path is unchanged. Enabling `vicreg_temporal_view`
with `data.kind: spatiotemporal_binary` consumes three explicit point clouds:

1. The 80 nearest atoms around an anchor atom at time t.
2. The 80 nearest atoms around a randomly selected member of that anchor's eight
   nearest other atoms, also at time t. Neighborhoods are rebuilt around this
   spatial center using the full periodic frame.
3. The 80 nearest atoms around the **same anchor atom ID** at t + 0.1 or 0.5 ps.
   Atom membership may change between frames; the central identity does not.

The encoder processes all three views together. The projector processes each
view once, sharing the projected anchor between the two losses:

`loss = (VICReg(anchor, spatial) + w * VICReg(anchor, temporal)) / (1 + w)`

The run uses w = 1 and the existing VICReg coefficients 25 / 25 / 1. Each pair
retains the variance and covariance penalties. Training uses the baseline mirror
and jitter augmentations; validation uses the fixed prepared views without them.
Non-finite three-view losses raise an error. Encoder temporal MSE, relative MSE,
and latent standard deviation are recorded in addition to projector losses.

## Data and split

Only the six Al and six Mg continuations in `datasets/zr_al_mg_initial_6x24ps`
and the single Ta branch in `datasets/ta_initial_1x24ps` are used. Zr is excluded.
The overnight 70,304-atom Al source campaign is a separate dataset and is not
included. Periodic offsets use each frame's current NPT box and the original
static GFv2 per-material normalization radii: Al 9.192189 Å, Mg 10.169428 Å,
Ta 9.388275 Å.

Training anchors span 2–17.5 ps, every 0.5 ps; validation anchors span
20–23.5 ps, every 0.5 ps. Including the largest temporal lag, training ends at
18 ps and validation covers 20–24 ps. The first 2 ps are omitted to reduce the
influence of initialization relaxation. Training and validation also use disjoint
central-atom ID pools. This remains validation within the same trajectories,
not a test on independent simulations; Ta has only one starting structure.

There are 196,608 training and 12,288 validation triplets **per material**:
589,824 training and 36,864 validation triplets total. Ta receives six times as
many sampled centers per branch so its single branch has equal material weight.
Temporal lag counts are exactly balanced within each anchor. Cached pair records
retain anchor ID, spatial-center ID, anchor frame and temporal lag for auditing.

Al/Mg supplied positions are global-coordinate float16 arrays, with reported
maximum quantization error up to 0.125 Å. Local caches are float32, which does
not restore the precision lost in the inputs. Frames are decoded and wrapped
before periodic neighbor searches, including values rounded to the box boundary.
Ta uses the verified float32 trajectory. A future higher-precision Al/Mg dataset
would provide a cleaner test of small thermal motions.

## Run and verification

Configuration: `configs/vicreg_geoframe_v2_spatiotemporal_almgta_20260905.yaml`.
The run uses the pointnet environment and H100 on the current node, nodesumo01, batch size 2048,
60 epochs, learning rate 1e-4 with three warmup epochs and cosine decay,
BF16 mixed precision, and eight data-loader workers. W&B logging is offline.

Artifacts are under
`/home/ids/vmorozov/experiments/geoframe_v2_spatiotemporal_Al_Mg_Ta_20260905/`:

- `views/manifest.json`: exact sources, preparation parameters and sample counts.
- `preflight.json`: actual GPU gradient-step and source-pair verification.
- `training_runner.json` and `training.log`: detached process and training log.
- `training/status.json` and `training/epoch_metrics.jsonl`: progress and metrics.
- `training/last.ckpt` and the best validation checkpoint: model/optimizer state.
- `training/baseline_stability.json`: original model on a fixed probe of 768
  validation samples per material, in float32 without training augmentations.
- `training/best_checkpoint_stability.json`: the same probe after training,
  separately measuring 0.1 and 0.5 ps and encoder/projector representations.

The fixed probe covers all input branches. Lower temporal MSE alone is not
sufficient evidence of improvement: inspect relative MSE and latent standard
deviation together to distinguish stability from collapse. Independent structural
and downstream evaluation is still needed before replacing the current checkpoint.

Unit checks cover periodic crossings, float16 boundary wrapping, both loss
gradient paths and non-finite loss rejection. Real cached spatial/temporal views
were compared exactly with reconstruction from each material's source trajectory.
The targeted run passed 12 tests; one pre-existing mirror-configuration test failed
because it references the absent `configs/vicreg_geo_frame_multi.yaml`.

## Corrected VICReg / VISReg comparison, September 5

Launched detached on node53 (H100 NVL) at 14:07 Paris time. Queue PID: 223026.
Run root: `/home/ids/vmorozov/experiments/geoframe_v2_vicreg_vs_visreg_20260905`.
The queue trains VICReg, then VISReg, then runs the matched comparison automatically.
Both runs use the same original GFv2 initialization, balanced Al/Mg/Ta triplet cache,
60 epochs, batch size 2048, learning rate 1e-4, and seed 123.
Validation now uses a fixed permutation mixing materials in every batch.
VISReg matches the existing GFv2 configuration: lambda 0.4, 4096 projections,
scale/shape/center coefficients 1/0.5/0.1. Both objectives average spatial and temporal pair losses equally.

The corrected callback saves `last.ckpt` every epoch, an additional snapshot every ten epochs,
and `final.ckpt` before loading the best checkpoint. The checkpoint audit verifies final step retention.
Six focused tests pass, including worsened-validation checkpoint retention and temporal/spatial gradients
for both objectives. Full-batch GPU smoke tests restored all 168 model tensors and produced finite gradients.

Execution uses a frozen `source/` copy because concurrent repository reorganization moved the entry points.
The queue is `experiments/spatiotemporal_20260905/run_spatiotemporal_objective_comparison.py` inside that copy.
Inspect `queue_status.json`, `vicreg/status.json`, and `visreg/status.json` for progress.
The report will be `comparison/RESULTS.md`, with `metrics.json`, plots, and saved embeddings.
It compares baseline and best/final checkpoints for both losses on identical temporal pairs and static
neighborhoods, reporting raw/relative temporal MSE, effective rank, and static clustering for encoder and projector.
The different objective scales must not be compared directly; same-source validation and Al/Mg coordinate
quantization remain limitations. This is a single-seed comparison, not a tuned multi-seed benchmark.

### Larger-batch restart after GPU benchmarking (14:23 Paris)

At the user's request, stopped the batch-2048 queue while preserving its checkpoints.
New detached queue PID 231775:
`/home/ids/vmorozov/experiments/geoframe_v2_vicreg_vs_visreg_largebatch_20260905`.
VICReg then VISReg each restart from the original GFv2 weights with training batch 8192,
validation batch 2048, and unchanged 60-epoch/1e-4 learning-rate schedule.
There are now 72 updates per epoch, 4,320 in total; dataset exposure is unchanged but optimizer
update count is four times lower. This is a new matched experiment, not an equivalent resume.

A real-data GPU benchmark of 20 optimizer steps after compilation/warmup measured:

| Batch | VICReg triplets/s | VISReg triplets/s | Peak allocated GB (max of objectives) |
|---|---:|---:|---:|
| 2048 | 41,214 | 39,607 | 9.8 |
| 8192 | 46,814 | 42,927 | 39.9 |
| 16384 | 46,675 | 42,097 | 79.8 |

Batch 8192 was fastest; batch 16384 fit but did not improve throughput. Fused AdamW changed
step timing by less than 1%, so it was not enabled. BF16, TF32, compiled encoder, pinned-memory
asynchronous transfer, and persistent workers were already in use. Loader-only fetching was
faster than GPU processing. CUDA-graph reduce-overhead mode remains disabled because repository
validation previously found call-order-dependent GFv2 embeddings on this stack.

All benchmark sizes completed with finite gradients. Six focused tests passed; the new loaders
were verified to produce 72 training and 18 validation batches per epoch. Full benchmark measurements,
launch metadata, frozen source hashes, logs, and the eventual automatic comparison are under the new root.

## Completed analysis saved in the repository (September 5, 15:39 Paris)

The batch-8192 VICReg and VISReg runs both completed 60 epochs / 4,320 updates.
Both selected epoch 49 as their own-objective validation minimum. The original
automatic comparison failed because a relocated analysis module was absent from
the frozen source. It was rerun successfully from the current repository code.

All new analysis outputs, including copies of the best/final/last checkpoints, are
in `output/geoframe_v2_spatiotemporal_analysis_20260905/` on `/home/infres`.
See the [consolidated report](../output/geoframe_v2_spatiotemporal_analysis_20260905/RESULTS.md).
Actual checkpoint tensors were verified: last and final are identical and contain
step 4,320 for each objective.

Full static Al analysis evaluated the original model and best/final checkpoints
for both objectives on all 772,953 neighborhoods in the six-snapshot regular-grid
cache. Full inference, full clustering fitting, and identical sampled coordinates
were verified. This retains the standard boundary exclusions; it is not a per-atom
scan. Silhouette uses the pipeline's 3,000-row diagnostic subset.

| Checkpoint | Full-Al cosine silhouette | Davies–Bouldin |
|---|---:|---:|
| Original GFv2 | 0.7095 | 0.7106 |
| VICReg best | 0.7646 | 0.7955 |
| VICReg final | 0.7571 | 0.8039 |
| VISReg best | 0.6137 | 1.0454 |
| VISReg final | 0.5911 | 1.0149 |

VICReg best is preferred among the fine-tunes, particularly for projector output.
Its normalized encoder drift decreases about 76% / 72% / 76% on Al / Mg / Ta,
but raw encoder drift increases and effective rank decreases. The projector shows
clearer smoothing: absolute temporal MSE decreases on all three materials. Within
source branches, VICReg improves relative drift in all 13, but the Al 166 ps gain
is only about 5%. This is not a uniform solution to bare-encoder temporal stability.
The original model retains a better Davies–Bouldin score than VICReg, and no
independent-trajectory or multi-seed claim is made.

The output contains 253 PNGs, 30 interactive HTML files, exact pipeline configs,
embeddings, labels, structural representatives, and machine-readable metrics
(about 3.0 GB). Experiment-specific branch diagnostics and full-analysis
consolidation helpers are documented in
[the experiment record](../experiments/spatiotemporal_20260905/README.md).

The proposed next experiment—causal smoothing, fixed-target future prediction,
and an audit of discontinuous geometry choices—is recorded in
[Smooth and predictive GeoFrame states](geoframe_smooth_predictive_state_20260905.md).
This is a research proposal; it does not describe another completed training run.
