# Full normalized MACE VICReg with Ta/Ti — September 10, 2026

Question: how does the corrected MACE + VICReg recipe behave after adding the
completed Ta/Ti simulations to the original Al/Mg/Ta data? The user requested
this full run detached on the existing H100 allocation. Production uses the
original `train_contrastive` entry point, `VICRegModule` and Lightning trainer.

## Recipe

[vicreg_mace_full.yaml](../../configs/vicreg_mace_full.yaml) inherits the
[working original MACE configuration](../../configs/vicreg_pretrained_mace_geometry.yaml).
All 8,219,792 used MACE parameters train from MACE-MP-0b2-small initialization.
The encoder takes 80 normalized points, with a fixed internal Al channel, shared
9.192189 Å reference length and native 5 Å interaction cutoff. Material identity
is not an encoder input. Original spatial/temporal VICReg uses coefficients
25/25/1, a 128D MLP projector and BatchNorm. Batch 512, AdamW peak LR 0.001,
weight decay 0.04, clip 1, three warmup epochs and cosine to 1e-6, **24 epochs**.
W&B is online. The full data-expansion run uses the original VICReg objective;
the [MEAM matrix](../mace_vicreg_relaxed_20260910/README.md) separately tests relaxed
TDA and history fusion in this same trainer.

The full H100 run retains the original compensated radial BF16 and radial
compilation, with one full 512-cloud backbone call and no activation checkpointing.
The history comparison uses activation checkpointing to fit five observations;
the full single-frame run has room for the original execution settings. Standard
post-training analysis disables radial compilation as in the corrected reference.
There is no gradient accumulation; projector and VICReg statistics use batch 512.

## Dataset and normalization

[data.json](data.json) fixes every source, manifest checksum, cutoff, anchor
frame, ID split and sampling count. Original Al/Mg/Ta views are reused through
verified links. Added sources: five Ta trajectories from 2.7/2.8/2.9/3.0/3.60 ns
parents, and six Ti trajectories from 0/8/32/40/56/192 ps parents. The
`Ti_early_slurm` exports have the same six array checksums as `Ti`; they are
recorded as duplicates and counted once.

Source cutoffs: Al 9.192189 Å, Mg 10.169428 Å, Ta 9.388275 Å,
**Ti 9.2477642791 Å**. Ti's cutoff was fitted once using the existing estimator
on a training-time frame: 160-neighbor distance, quantile 0.995, 4,000 centers,
seed 42, factor 1.02, periodic boundaries, 7.6 Å boundary margin. Full provenance
is in `data.json`. Coordinates are divided by the material's fixed source
radius, then the encoder applies the shared reference length above.

All selected temporal pairs have lag 0.1 ps. New Ta train anchors span
2–17.5 ps, validation 20–23.5 ps; Ti train anchors span 2–172.5 ps, validation
200–235 ps. The original center-row split is preserved: training `row % 5 != 0`,
validation `row % 5 == 0`. Each added trajectory contributes 16,384 training
and 1,024 validation triplets. Local arrays use float16 storage and float32
decoding; preparation records checksums.

| Material | Training | Validation |
|---|---:|---:|
| Al | 98,304 | 6,144 |
| Mg | 98,304 | 6,144 |
| Ta, original plus new | 180,224 | 11,264 |
| Ti | 98,304 | 6,144 |
| Total triplets | 475,136 | 29,696 |

All original triplets remain, so Ta has a larger sampling share after expansion.
Training uses the original ordinary shuffle: 928 updates/epoch, **22,272 total**.
Validation is disjoint in time and center IDs within trajectories. Ti branches
share one source lineage and Ta archived snapshots are related; these validation
scores do not establish independent-source generalization. The MEAM comparison
has separate source splits for that question.

## Execution and reproduction

Slurm CPU controller **988568**, `nodecpu03`, owns the launch processes and has
four cores/16 GiB for preparation. GPU training uses existing H100 allocation
**988041**, `node53`, ending September 11 at 15:40 CEST. Existing L40S allocation
988064 continues the three pure-VICReg MEAM seeds. The CPU batch job is
independent of the IDE and L40S session.

The controller prepares data, then starts full training and the standard static
Al analysis on the best validation checkpoint. In parallel it preserves all
three completed H100 anchor-PCA runs, stops their old launcher and collector,
and launches the remaining 18 comparisons with clients on the CPU batch node.
The MEAM collector then waits for successful completion of all assigned runs.
Failures retain logs and produce a nonzero controller exit status.

At 17:58 CEST the H100 launcher migration completed. Slurm reports
`SrunHost=nodecpu03` for continuation step 988041.18, confirming independence
from the expiring L40S host. Full-data preparation is active; the full training
command starts automatically after the completed cache manifest is published.

Output: `output/mace_vicreg_full_20260910/`. `preparation/` tracks construction;
`views/manifest.json` records data/checksums; `controller/` tracks full training;
`train/` contains checkpoints/W&B metadata; `train/analysis/` contains the standard
static analysis. Generated launch/migration scripts and logs are in `slurm/`.

```bash
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_full_20260910/run_prepare.json
conda run --no-capture-output -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_vicreg_full_20260910/run_spec.json
```

Tracked specs require fresh execution output directories. To verify/reuse the
prepared cache directly, use the existing producer command:

```bash
conda run --no-capture-output -n pointnet python \
  experiments/spatiotemporal_20260905/prepare_spatiotemporal_vicreg_views.py \
  --config experiments/mace_vicreg_full_20260910/data.json
```

The original neighborhood method moved into `src/data_utils/spatiotemporal_views.py`;
the existing command now accepts explicit expansion settings. A regression test
checks scaling, anchor/spatial/future identities, float16 storage and ID splits.
All 46 selected data/encoder/VICReg/analysis/dependency tests pass. Findings
are pending; launch success is not a completed training result.

The existing original-MACE preflight also passed eight discarded Lightning
updates and validation on the H100 while another comparison was training.
It used the unchanged original Al/Mg/Ta cache to test the full-run execution
settings before expanded-cache completion: peak allocated memory 59.304 GiB,
median warm step 0.934 seconds under concurrent GPU load, and loss
19.9308 → 19.2324 with finite nonzero backbone/projector gradients. These are
execution checks, not expanded-data training results. Reproduce with
`experiments/mace_original_vicreg_20260909/preflight.py --config
output/mace_vicreg_full_20260910/preflight/resolved_reference_data.yaml --output
output/mace_vicreg_full_20260910/preflight`; the report is `result.json` there.

File roles: this directory contains experiment records, the YAML is a training
configuration, shared producer changes/tests are maintained code. Launch scripts,
diagnostics, caches, logs and results under `output/` are generated run artifacts.
No training command or training loop was added.


## Integrated topology analysis — September 10, 23:37 CEST

The 24-epoch fit and its existing static-Al analysis completed. The same retained
best checkpoint (epoch 9, zero-based) is also included in the
[standard analysis batch](../mace_vicreg_relaxed_20260910/analysis_completed.yaml),
which adds the pipeline's source-held-out MEAM topology stage. Results are under
`output/mace_vicreg_full_20260910/train/analysis_standard/`; the original full
rendering remains in `analysis/`. The new topology stage has completed with test
projector-ridge balanced MSE **0.03414314**. This checkpoint has no trained TDA
head, so its topology result is a training-only linear probe. Main static
figures for the new batch are still being generated. See the
[integration record](../mace_vicreg_relaxed_20260910/README.md) for metrics,
cohort limits, fast rendering settings and detached controller 988958.

## September 11 storage and visualization update

The full model's UMAP, t-SNE, spatial views and metrics are now in the short
[full-model gallery](../../output/mace/full/index.html). Detailed analysis
artifacts are on IDS in `/home/ids/vmorozov/analysis/mace/artifacts/full/`.
The completed 24-epoch fit and its selected epoch-9 checkpoint are retained;
its redundant `last.ckpt` was removed after successful analysis.

The reusable view cache is now physically at
`/home/ids/vmorozov/training-cache/mace-full/`. Its old `views/` path is a
compatibility symlink, created only after full copy/checksum verification. Current
training reads IDS directly. Keep [data.json](data.json)'s historical output path:
the producer includes that exact configuration in cache identity, and the symlink
allows the reproduction command above to verify/reuse it without rebuilding.
Original Al/Mg/Ta links and all new Ta/Ti data are preserved.

Current MACE configs remove inference caches after successful analysis and avoid
duplicate figure sets, paper-format exports and validation prediction arrays.
See the [September 11 record](../mace_vicreg_relaxed_20260910/README.md#umap-flat-reports-and-storage--september-11)
for cleanup provenance and current detached analysis job **989060**. Earlier
controller descriptions above are dated execution history.
