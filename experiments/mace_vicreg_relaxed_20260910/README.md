# Relaxed MEAM topology in the original VICReg trainer — September 10, 2026

**Completed findings:** [September 11 analysis of all 24 MEAM fits](RESULTS_20260911.md).
History improves linear-probe topology accuracy modestly; checkpoint selection
substantially distorts the trained-head results. All twelve overnight fits finished.

Latest execution: [September 11 overnight H100 comparison](OVERNIGHT_20260911.md),
nine priority fits plus three residual-fusion fits if allocation time permits.
Uses existing VICReg training and standard analysis, with flat results in
[`output/mace/`](../../output/mace/index.html).

## Question and baseline

Does the corrected, trainable MACE + VICReg recipe learn relaxed Al topology,
and does learned history fusion improve over a single frame or mean pooling?
The user explicitly selected the corrected MACE recipe and requested that new
experiments use the existing VICReg codebase. Production training runs through
`src.training_methods.contrastive_learning.train_contrastive` and `VICRegModule`.
There is no separate optimizer loop or replacement VICReg implementation.

The reference is the [original normalized MACE recipe](../mace_original_vicreg_20260909/README.md)
and its [matched TDA extension](../mace_original_vicreg_tda_20260909/README.md).
Their saved full static-Al cosine silhouettes are 0.6809 and 0.6897; these are
structural sanity checks, not proof of relaxed-topology accuracy. The user chose
this MACE recipe after being shown the stronger GeoFrame silhouette as well.
Every new run starts from the same pretrained MACE-MP-0b2-small weights. All
8,219,792 used backbone parameters are trainable; prior denoising weights are
not loaded. This is a fresh comparison using the working training recipe.

## Unchanged training settings

The [configuration](../../configs/vicreg_mace_relaxed.yaml) inherits
`vicreg_pretrained_mace_geometry_tda.yaml`:

- Normalize each Al cloud by the original 9.192189 Å source radius, then use
  the same shared reference length inside MACE and one fixed internal Al channel.
  MACE keeps native 5 Å pair edges and all 80 supplied atoms. There is no fitted
  input-feature standardizer.
- Original spatial and same-center temporal VICReg pairs; shared anchor
  projector output, exact original variance/covariance conventions and 25/25/1
  coefficients. The two pair losses have equal weight.
- Original 128D MLP projector, BatchNorm behavior, jitter and mirroring. The
  exported representation is the projector output; TDA is attached to it.
- AdamW LR 0.001, weight decay 0.04, gradient clipping 1, batch 512, no gradient
  accumulation, three warmup epochs, cosine decay to 1e-6, and **24 epochs**.
  Each epoch has 27 updates; each run has 648 updates. Checkpoint selection uses
  the original combined validation loss, not a new selection rule.
- Online W&B through the existing trainer; each run records `wandb_run.json`.

Activation checkpointing chunks MACE into 128-cloud calls while retaining one
full batch of 512 for the projector and VICReg statistics. This is standard
PyTorch activation checkpointing inside the encoder. No frozen feature cache or
custom gradient replay is used. Radial compilation is explicitly disabled after
both GPU preflights hit the shared Dynamo recompilation limit on validation.
Compensated BF16 arithmetic remains enabled as in the working recipe; geometry,
projector, targets and losses remain FP32. These are execution changes to fit
histories and retain the original statistical batch size.

## Data and explicit protocol differences

The source is the completed [uniform MEAM FIRE dataset](../mace_al_denoising_20260910/README.md):
30 independent melt sources at 400/450/510 K; source splits 18/6/6; anchor times
30/180/480 ps; 13,824/4,608/4,608 train/validation/test neighborhoods.
Full-cell, fixed-box relaxation uses the generating Lee2003 MEAM potential and
maximum force 0.01 eV/Å. Existing target arrays and row identities are reused.

For each anchor, preparation creates three views: the anchor center, a spatial
neighbor chosen among its nearest eight, and the same center one stored frame
earlier. Each view gets a complete nearest-80 neighborhood from the full periodic
cell and tracks those identities through its own five-frame history. Membership
can differ between views; it stays fixed within a history. Mirroring applies
consistently to all frames of a history. Positions are stored as physical float16
offsets and normalized in the loader, avoiding a second float16 quantization.

Five observations have offsets [-3, -2.25, -1.5, -0.75, 0] ps. The temporal
VICReg pair lag is **0.75 ps**, determined by the available MEAM storage cadence;
the earlier working recipe used 0.1 ps. This difference is explicit.

Only the anchor view has a prepared relaxed target, so TDA supervises its
projector output only. Spatial and earlier views contribute the unchanged VICReg
pairs; they are not assigned the anchor's relaxed label. The earlier TDA recipe
supervised the instantaneous topology of all three views. This experiment instead
tests prediction of the specified relaxed anchor.

Target transforms fit all 13,824 training anchors and no held-out rows. PCA32
retains the original whitening convention. The full-target comparison gives
H0/H1/H2 equal weight and uses the prior 5% block-scale floor. This comparison
explicitly changes target representation; it does not modify VICReg.

## Declared comparisons

The [complete plan](plan.yaml) contains eight variants, each with seeds
20260910/20260911/20260912: **24 runs**.

| Variant | Input / objective |
|---|---|
| anchor_vicreg | Original single-frame MACE and VICReg, no TDA head; evaluate a training-only ridge probe |
| anchor_pca | Single frame, original VICReg plus relaxed PCA32 TDA |
| anchor_blocks | Single frame, original VICReg plus balanced 144D relaxed TDA |
| mean_blocks | Mean of five trainable MACE frame embeddings, original VICReg plus balanced TDA |
| transformer_blocks | Two temporal transformer layers, width 128, four heads, then original projector |
| residual_blocks | Anchor embedding plus a learned temporal correction, then original projector |
| atom_anchor_blocks | Matched atom-fusion architecture with repeated anchor atom features |
| atom_temporal_blocks | One width-64, four-head transformer per atom identity, learned spatial pooling and anchor residual |

All TDA heads have coefficient 1 from epoch 1. Residual branches begin at zero;
there is no anchor-decoder warm start or extra training phase. Atom-anchor and
atom-temporal have matching trainable capacity. The pure-VICReg probe and TDA-head
scores are distinguished, and projector ridge scores are provided for every run.
The previous incomplete variance/covariance-only comparator is not called VICReg
and is not repeated as if it were the corrected objective.

## Validation and analysis

The original regression suite passes, including unchanged paired VICReg and
three-view TDA behavior. New tests cover normalization, source splits, temporal
atom alignment, shared history mirroring, the exact VICReg-plus-TDA objective,
and TDA gradients through the anchor history only. Real-GPU preflights use the
original Lightning module for three discarded updates at batch 512. MACE,
projector and TDA gradients are finite and nonzero; every input history frame
receives a gradient. The L40S anchor check uses about 5.28 GiB; the H100 atom-history
check about 6.77 GiB. Failed compilation checks remain in the output as diagnostics.

Each run automatically reloads its validation-selected original Lightning
checkpoint and evaluates the prepared source-held-out task. The existing static
pipeline cannot supply histories, so these runs use the registered post-training
hook for actual history evaluation. Reports include balanced/raw MSE, within-frame
R², homology/temperature/time breakdowns, projector ridge controls, repeated-anchor
and reversed-past interventions, saved predictions and embeddings. W&B receives
the final metrics. The existing experiment runner collects per-seed tables.
[Analysis comparisons](analysis.json) resample whole sources after averaging seeds.

The same six test trajectories were already examined in the frozen-MACE
experiment. This is a controlled follow-up on that cohort, not a new untouched
test set. No threshold or checkpoint is chosen using its test labels.

## Reproduction and outputs

```bash
conda run --no-capture-output -n pointnet python -m src.data_utils.spatiotemporal_tda \
  --config-name vicreg_mace_relaxed
conda run --no-capture-output -n pointnet python scripts/run_experiments.py \
  --plan experiments/mace_vicreg_relaxed_20260910/plan.yaml --local --nan-restart-max-retries 0
```

Detached execution splits this unchanged matrix between the existing allocations:
[L40S plan](plan_l40s.yaml), three pure-VICReg seeds; [H100 plan](plan_h100.yaml),
the other 21 runs. [L40S run spec](run_l40s.json) and [H100 run spec](run_h100.json)
invoke the maintained experiment runner, which invokes the original trainer.
Both training GPUs use the existing allocations. Output is
`output/mace_vicreg_relaxed_20260910/`; preparation is in `data/`, training plans
write below `runs/`, and each run now writes `analysis_standard/analysis_metrics.json`,
standard figures, topology predictions, original Lightning checkpoints and W&B metadata.
Earlier `analysis/` artifacts remain as historical results.

```bash
conda run --no-capture-output -n pointnet python -m src.analysis.pipeline \
  --collect-root output/mace_vicreg_relaxed_20260910/runs \
  --specification experiments/mace_vicreg_relaxed_20260910/analysis.json
```

File roles: configuration, plans, specs, this README and the discarded-update
preflight recipe are experiment records. Dataset/encoder support, shared topology
transforms/metrics and checkpoint analysis under `src/` are maintained code used
by the existing VICReg workflow. Prepared arrays, launch records, logs, checkpoints
and reports under `output/` are generated run artifacts.

## Detached launch and continuation

Initial training started at 17:29 CEST in allocations 988064 (L40S, node50) and
988041 (H100, node53). Online W&B for the first seeds:
[anchor VICReg](https://wandb.ai/teshbek/PointCloudMaterials/runs/piavjkvf) and
[anchor PCA](https://wandb.ai/teshbek/PointCloudMaterials/runs/2rsmb4t2).
The first H100 seed completed all 24 epochs and its checkpoint analysis.
One seed is insufficient to interpret the comparisons.

The original launch clients were on node50, whose allocation ends at 19:13 CEST.
A disposable heartbeat step demonstrated that losing an `srun` client kills
its remote step. A detached Slurm CPU controller, job **988568**, now prepares
the requested [full Ta/Ti run](../mace_vicreg_full_20260910/README.md) and owns the
continuation launches. It retains all three completed anchor-PCA seeds before
stopping old H100 steps 988041.5 and 988041.9. It then runs
[the remaining 18-run plan](plan_h100_remaining.yaml) through
[the continuation spec](run_h100_remaining.json), using the same existing H100.
This continuation starts untouched variants fresh from MLIP initialization;
completed anchor-PCA seeds are not retrained or warm-started into other variants.

[The detached collector](run_comparison_detached.json) requires the successful
three original PCA records, the L40S plan and the H100 continuation. Cross-node
running dependency checks verify recorded PIDs/start times through Slurm. The
earlier failed collector and replaced controller records remain as provenance.
Operational migration logs are in `output/mace_vicreg_full_20260910/slurm/`.

Migration completed at 17:58 CEST after all three anchor-PCA seeds succeeded.
The continuation is running as H100 step **988041.18**; Slurm confirms ten CPUs,
64 GiB and one GPU, with launch client `nodecpu03:2680602` owned by the detached
CPU batch job. The first continued run is
[anchor-blocks seed 20260910](https://wandb.ai/teshbek/PointCloudMaterials/runs/bmeka2nn),
with W&B online. The full Ta/Ti preparation continues concurrently on the CPU node.


## Standard analysis integration — September 10, 23:35 CEST

The existing `src.analysis.pipeline` now owns post-training analysis for these
runs. Model loading, inference caches, clustering, PCA/latent statistics, t-SNE,
PTM/CNA representatives and spatial plots use its original implementation.
`src/analysis/topology.py` is an optional stage of that pipeline, with dataset
support in `topology_dataset.py` and shared scoring in `topology_metrics.py`.
It has no standalone checkpoint loader or inference loop.

Added metrics:

- Raw and training-scale-balanced MSE, per-H0/H1/H2 R², and within-frame R².
  Within-frame R² uses variation around each source/anchor-frame target mean
  as its denominator; prediction errors are not centered away.
- A fixed-alpha ridge readout of the 128D projector, fitted and standardized
  using training sources only; trained-head scores and a training-mean baseline.
- Temperature, source and anchor-frame breakdowns, plus per-row predictions/errors.
- Real history versus repeated anchor and reversed past, scored through both
  the trained head and the same real-training ridge readout.
- Paired error reduction and 95% whole-source bootstrap intervals. Across-model
  collection averages the three seeds before resampling the six test sources,
  separately comparing trained predictions and common ridge readouts.

The canonical result is `analysis_standard/analysis_metrics.json`, with the new
metrics under `topology`. `analysis_standard/topology/` holds detailed scores,
predictions and standard inference caches. `final_metrics.json` publishes the
selected scalars to the original experiment runner. The live training hook also
publishes topology scalars to its existing W&B run. Detached retrospective
analysis produces local artifacts and does not reopen completed W&B runs.

Single-frame checkpoints use `configs/analysis/static_topology.yaml`: the
existing six static-Al snapshots (772,953 neighborhoods), plus the MEAM topology
stage. History checkpoints use `configs/analysis/relaxed_histories.yaml`:
clustering and plots use all 4,608 held-out histories and show each history's
anchor structure. The topology stage always uses the declared 13,824/4,608/4,608
train/validation/test split. Static and history clustering scores therefore
refer to different input cohorts and must not be compared directly.

Both templates select the pipeline's existing fast rendering profile: CPU/PCA
plots, six representative snapshot sets, no Blender ray tracing or equivariance
pass. Inference and clustering sample counts are unchanged. Cross-frame flicker
and transitions are disabled for the independently sampled MEAM cohort; its
sources must not be interpreted as a single continuous trajectory. Atom IDs and
physical center coordinates come from the original simulation producer.

```bash
conda run --no-capture-output -n pointnet python -m src.analysis.pipeline \
  --batch experiments/mace_vicreg_relaxed_20260910/analysis_completed.yaml
```

[The batch list](analysis_completed.yaml) explicitly names all 13 fitted
checkpoints: 12 MEAM fits and the full Al/Mg/Ta/Ti fit. The selected anchor
(`anchor_blocks__rep02`) and history (`mean_blocks__rep03`) checks completed the
full pipeline in 403 s and 108 s, respectively. The anchor's head MSE was
0.06716591 and ridge MSE 0.03195572, reproducing the previous evaluator within
floating-point inference differences. For the history checkpoint, real-history
ridge MSE was 0.02833262 versus 0.03627941 with a repeated anchor; reversing the
past gave 0.02833257, as expected for mean pooling. These are individual
checkpoints, not a completed three-seed architecture comparison.

An initial history plotting attempt failed because frame-context IDs had been
passed to the flicker stage as atom identities. The failure is retained in the
training run record. After correcting metadata and explicitly disabling
cross-source temporal metrics, [the analysis-only retry](run_standard_history.json)
succeeded using the saved checkpoint; no completed fit was repeated.

Detached CPU controller **988958** owns the analysis batch and the
[12 untouched training runs](plan_h100_unstarted.yaml) on existing H100 allocation
**988041**. It calls the original experiment runner and original VICReg trainer.
[Batch spec](run_standard_completed.json), [training spec](run_h100_unstarted.json),
and [final collection spec](run_comparison_standard.json) preserve dependencies.
The older controller's relaxed branch stopped on the reported analysis error;
its independent full-training branch was left running. New controller logs and
Slurm scripts are under `output/mace_vicreg_relaxed_20260910/standard_pipeline_migration/`.

The standalone relaxed-topology, temporal and denoising evaluators and their two
experiment review scripts were removed. Historical outputs and source provenance
archives remain. Old non-Lightning `analysis`/`all` entry points now fail explicitly
with migration guidance; they do not silently substitute the new scientific
protocol. Potential-difference physics audits remain separate and import only
shared metric functions.

Validation: 46 distinct pipeline/data/encoder/training-hook tests passed across
the focused suites, including a held-out
label perturbation test and checks of full-sequence inference, anchor identity,
interventions and paired seed/source aggregation. New `src/` modules and analysis
configurations are maintained pipeline support; batch/spec/plan files and this
README are experiment records. Generated launch scripts, logs, figures, caches
and the historical source archive are disposable diagnostics/run artifacts in
`output/`.

## UMAP, flat reports and storage — September 11

The earlier fast-profile choice overrode MD UMAP with PCA. That override is now
removed in both analysis templates, and latent UMAP comparison is enabled beside
t-SNE. Open the [flat gallery](../../output/mace/index.html), especially
[anchor blocks, seed 20260911](../../output/mace/anchor-blocks-seed20260911/index.html)
and [the full Al/Mg/Ta/Ti fit](../../output/mace/full/index.html).
Each model has one short directory with `umap.png`, `tsne.png`, `md-umap.png`,
`spatial-166ps.html`, representatives and metrics. PCA remains an additional
diagnostic. `source.json` records the checkpoint hash and artifact origins.

The original analysis pipeline produces these reports; no separate evaluator or
training loop was added. Detailed artifacts live on IDS at
`/home/ids/vmorozov/analysis/mace/artifacts/<variant>-seed<seed>/` and `full/`.
[The UMAP batch](analysis_umap_20260911.yaml) names the 13 fitted checkpoints.
Detached CPU job **989060** runs [this spec](run_umap_compact_20260911.json) on
existing H100 allocation **988041**. It skips verified completed analyses and
continues the remaining reports, independently of the IDE connection. Logs are
`output/mace/jobs/umap_compact/command.log`.

```bash
conda run --no-capture-output -n pointnet python -m src.analysis.pipeline \
  --batch experiments/mace_vicreg_relaxed_20260910/analysis_umap_20260911.yaml
```

At this storage update, the anchor-blocks seed 20260911, full-data model,
mean-blocks seed 20260912 and anchor-PCA seeds 20260910/20260911 have finished
UMAP reports; the remaining batch is running. The earlier controller **988958**
is absent from the queue and its recorded training PID is gone. Its stale
`running` files are not evidence of ongoing training. No unfinished fit was
restarted during this storage/rendering change; its recovery checkpoint remains.

[The verified relocation plan](cache_storage_20260911.json) moved 5.614 GiB of
reusable caches to `/home/ids/vmorozov/training-cache/`:

| Cache | IDS directory |
| --- | --- |
| Prepared MEAM spatial/temporal histories, relaxed targets and scaling | `mace-meam/` |
| Full Al/Mg/Ta/Ti triplets, including links to original views | `mace-full/` |
| Shared temporal neighborhood caches | `temporal/` |

Every copied file was checked by SHA-256 and the source inventory checked again
before replacing the old directory with a compatibility symlink. Current configs
use IDS directly. Historical manifest/config paths remain valid. Copy verification
is recorded in `output/mace/storage/cache-moves.json`.

Cleanup removed **413 obsolete files, including 85 superseded checkpoints**,
freeing **13.98 GiB of allocated storage**. Selected best checkpoints, unfinished
recovery states, targets/scalers, test predictions, raw trajectories and source
provenance remain. The deleted files include obsolete frozen-MACE feature caches
and seven historical standard-analysis inference caches. Their metadata was
archived and verified before deletion. Plans and archives are in
`output/mace/storage/`; each unlink is recorded in
`output/registry/cleanup_applied.jsonl`. These archives are retained provenance.

Future MACE training retains one best model and a rolling recovery checkpoint,
then deletes the recovery checkpoint only after successful analysis. The two
analysis templates discard their inference arrays after success, avoid persistent
topology embedding caches and validation prediction arrays, and disable duplicate
per-snapshot figure sets and paper-format exports. Metrics, normal spatial views,
representatives, UMAP/t-SNE and test predictions for paired comparisons remain.
No scientific objective, dataset split or score definition changed.

Validation: 22 storage, topology, fast-path, training-hook and MEAM dataset tests
passed in `pointnet`; the detached batch also exercised completed-report reuse
with cache removal. New `report.py` and `cache_storage.py` are maintained support
in `src/`; the batch/spec/relocation JSON and this section are experiment records.
Generated job scripts, logs, cleanup plans and figures belong under `output/`.
