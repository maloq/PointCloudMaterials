# Workflow details

Run commands from the repository root with the `pointnet` environment:

```bash
conda run -n pointnet python scripts/run_shooting_ablation.py --help
conda run -n pointnet python scripts/run_shooting_ablation.py geometry --help
```

Each family lists explicit workflows; `WORKFLOW --help` describes that workflow's
arguments. Changing a run's settings should use configuration or those arguments.
Scientific implementations belong in `src/`. Experiment-specific recipes live in
[`experiments/`](../experiments/README.md); new generated results stay in `output/`;
`outputs/` is retained for historical runs. See the [old-to-new command map](../docs/script_cleanup_20260905.md).

## Simulation and data

| Command | Purpose / implementation |
| --- | --- |
| `run_lammps_campaign.py WORKFLOW` | Campaign preparation, execution, continuation and summaries; `src/simulation/campaigns/` |
| `convert_trajectory.py FORMAT_OR_AUDIT` | Verified format conversion and read-only audits; [conversion guide](../docs/trajectory_conversion.md), `src/data/conversion/` |
| `inspect_temporal_lammps_dataset.py` | Inspect a temporal dump and optionally build its cache |
| `run_optimized_al_homogeneous_campaign.sh` | Resume the selected optimized Aluminum campaign; requires `PYTHON`, accepts `DEVICES` |

The `predictive-dynamics` (48 ps Langevin) and `predictive-dynamics-15ps` (CSLD,
exact continuation) workflows preserve separate protocol contracts. The shared
campaign readers retain their original 70,304-atom Aluminum input contract.
Specialized 510/520 K and Ta recipes live under `experiments/`.

The `elemental run --config CONFIG` workflow provides explicit Al/FCC and Ti/BCC
source-then-branches and Ta archived-position protocols, implemented in
`src/simulation/campaigns/elemental.py`. See the
[Ti/Ta experiment](../docs/simulations/ti_ta_crystallization/README.md).
`elemental sequence --ta-config TA --ti-config TI` finishes Ta before launching
Ti, allowing each campaign to use the full CPU allocation without overlap.
After a failed Ta sequence, add `--resume-ta` to verify completed branch/restart
checksums and run only untouched branches. Scientific settings must match the
saved campaign. `delete_verified_source_text: true` passes the converter's
`--delete-source` option after each completed trajectory.
`convert_trajectory.py elemental BRANCH` verifies this producer's completed,
sorted `id type x y z` dumps and exports float16 temporal positions by default.
Add `--delete-source` to reclaim verified source text after conversion.

`convert_trajectory.py training-cache PLAN.json` converts explicitly listed derived
neighborhood caches to float16 with range checks, verified rounding, atomic file
replacement, and per-file provenance. It leaves simulation trajectories, targets,
and IDs alone. See [cache conversion](../docs/trajectory_conversion.md#derived-training-caches).

## Training and predictive analysis

`python -m src.training_methods.embedding_forecast --config CONFIG.json --stage
prepare|train|evaluate|collect|queue|all` forecasts frozen local-structure embeddings from
their history. It supports separate (0,3], (3,6], (6,9] ps means, direct and
autoregressive future paths, and joint Gaussian path uncertainty. Autoregressive
fits declare rollout or teacher-forced training; validation/test always roll out
predictions. Preparation reuses the independent-MEAM
source selection and snapshot VICReg checkpoint; training batches memory-mapped
trajectory windows. `--variant NAME`, `--seed SEED` and `--device DEVICE` select
configured runs. See the [forecast experiment and exact input/target contracts](../experiments/embedding_forecast_20260911/README.md).
Implementation and orchestration live in `src/training_methods/embedding_forecast/`.
Training supports configured history jitter/frame dropout, explicit exact-state
`--resume`, and `--epochs-per-invocation N` for epoch-boundary continuations.
`--stage queue --queue-config PLAN.json` submits shared preparation, independent
training chains that can run concurrently, and a final comparison through native
Slurm dependencies. It freezes code/configuration and tracks each command.
Set the queue plan's `preparation_job_id` to reuse an already-submitted shared
preparation job. Setting `epochs_per_invocation` equal to the training epoch budget
submits exactly one training job per variant/seed.
See the [enlarged full-trajectory run](../experiments/embedding_forecast_20260911/SCALE_RUN.md).

`experiment_registry.py build` refreshes the shared experiment/simulation dashboard
and ideas view. `run --spec SPEC.json` tracks an explicit maintained command;
`status --record run_record.json` inspects execution state. `prune --plan PLAN.json`
previews verified cleanup, and `pack-logs --before ISO_TIMESTAMP` previews lossless
log archiving; mutations require `--apply`. Implementation: `src/experiment_runner/`.
See [the registry and run guide](../docs/output_registry.md).

`relocate-caches --plan PLAN.json` previews explicit immutable cache moves;
add `--apply` to copy, verify every file's SHA-256 and unchanged source inventory,
then replace the old directory with a compatibility symlink. The plan contains
`audit` and `moves: [{source, destination, producer}]`; no cache writer may be
active. Implementation: `src/experiment_runner/cache_storage.py`. The
[September 11 plan](../experiments/mace_vicreg_relaxed_20260910/cache_storage_20260911.json)
places shared MACE and temporal training caches in `/work/PERSO/vmorozov/training-cache/`.

`run --spec SPEC --wait-for-dependencies-until ISO_TIMESTAMP` waits for successful
completion of tracked dependencies, including their post-training analysis,
before executing the unchanged command. The timestamp must include its UTC
offset; failed/dead dependencies and expired deadlines stop with an explicit error.
Waiting status is written to the spec output's `queue_status.json`.
Running dependencies on another Slurm node are verified through the recorded
allocation and process start time using authenticated `srun` access.

| Command | Purpose / implementation |
| --- | --- |
| `run_experiments.py --plan PLAN` | Run/resume/collect local or Slurm experiment plans; `src/experiment_runner/` |
| `run_shooting_ablation.py METHOD --config CONFIG` | Distributional, spatial, geometry, multiscale, dynamical, short-horizon, encoder-finetune, temporal-pretraining methods; `src/temporal_vamp/commands/ablation_*` |
| `run_predictive_atlas.py METHOD --config CONFIG` | `frozen`, `history`, `finetune`, `temporal-encoder`; `src/temporal_vamp/commands/atlas_*` |
| `analyze_geoframe.py WORKFLOW` | `stability`, `variability`, `compare-variability`, `compare-representations`; `src/temporal_vamp/commands/geoframe_*` |
| `train_geoframe_spatiotemporal.py --config-name NAME --run-dir DIR` | Config-selected training with the repository Al/Mg/Ta stability probe; `src/training_methods/spatiotemporal.py` |
| `run_temporal_vamp.py`, `evaluate_temporal_vamp.py` | Temporal VAMP training/evaluation; `src/temporal_vamp/` |
| `run_shooting_predictor.py`, `evaluate_shooting_ablation0.py` | Baseline shooting predictor and evaluation |
| `run_nested_committor.py` | Nested shooting committor analysis |
| `run_predictability_map.py` | Configuration-driven predictability map |
| `run_temporal_encoder_pretraining.py` | Temporal encoder pretraining |
| `run_ordinary_temporal_embedding_cache.py` | Ordinary-trajectory embedding cache |
| `analyze_shooting_branch_outcomes.py` | Branch outcome statistics |

## Plots

| Command | Purpose |
| --- | --- |
| `plot_experiment_summary.py` | Plot experiment-runner summary JSON |
| `plot_grouped_metric_csv.py` | Plot grouped metric CSV |
| `plot_homogeneous_checkpoint.py` | Inspect and plot a campaign checkpoint selected by config |
| `render_shooting_dynamics_gifs.py` | Render configured shooting dynamics |

Shared metric plotting and campaign dashboards are implemented in
`src/analysis/plots/`; the experiment runner imports that implementation directly.
For full checkpoint analysis, use `python -m src.analysis.pipeline CONFIG`.
This includes encoder-only predictive-density exports (`model_type: density_encoder`),
loaded through `src/analysis/density_encoder_adapter.py`; see the
[static-Al experiment recipe](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/temporal_hypotheses_12h_20260906/README.md#standard-static-pipeline-encoder-only).
For MLIP-initialized MACE, use
the original Lightning VICReg entry point with
`python -m src.training_methods.contrastive_learning.train_contrastive --config-name vicreg_pretrained_mace_geometry`.
The [geometry-only recipe](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_original_vicreg_20260909/README.md)
uses normalized 80-point views, a fixed internal species channel, the original
projector/loss/optimizer and standard static analysis. It adds no new runner.
The [full Ta/Ti expansion](../experiments/mace_vicreg_full_20260910/README.md)
selects `vicreg_mace_full` in that same entry point. The existing
`src/research/spatiotemporal/prepare_spatiotemporal_vicreg_views.py --config JSON`
command now accepts explicit source/timeline/cutoff settings and reuses the
shared original producer in `src/data/spatiotemporal.py`. It verifies
the original cache links and completed additions, and records excluded duplicates.
The [matched TDA recipe](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_original_vicreg_tda_20260909/README.md)
adds the optional TDA head within that same module. Prepare normalized three-view
targets with `python -m src.data.topology_views --config configs/vicreg_pretrained_mace_geometry_tda.yaml`,
then select that config in the original training entry point. The producer
reuses the existing alpha-complex descriptor, preserves source row indices,
fits PCA on training views only and records cache checksums.
For prepared relaxed MEAM targets and actual temporal histories, use the
[original-VICReg MEAM comparison](../experiments/mace_vicreg_relaxed_20260910/README.md).
`python -m src.data.topology_views --config-name vicreg_mace_relaxed`
prepares the declared spatial/temporal histories and reuses converged anchor
targets. `train_contrastive --config-name vicreg_mace_relaxed` uses the same
`VICRegModule`, original projector/loss and online logger with trainable MACE.
History encoders accept normalized `(B,T,80,3)` inputs through `EncoderAdapter`.
The registered post-training hook uses the existing `src.analysis.pipeline`.
Its optional topology stage adds H0/H1/H2 errors, within-frame R², training-only
ridge probes, source/temperature/frame breakdowns and history interventions.
Use `configs/analysis/static_topology.yaml` for single-frame checkpoints and
`configs/analysis/relaxed_histories.yaml` for actual five-frame inputs. Both retain
the standard clustering, latent statistics and representative-structure analysis.
Set topology cache/manifest paths in `topology.data`; scores are written under
`analysis_standard/analysis_metrics.json["topology"]`.
Run one checkpoint with `python -m src.analysis.pipeline CONFIG --checkpoint CKPT
--output-dir OUT`. For existing checkpoints use `--batch BATCH.yaml`, with explicit
`runs: [{checkpoint, analysis_config, output_dir}]` and `cuda_device`.
Aggregate configured comparisons with `python -m src.analysis.pipeline --collect-root
RUNS --specification EXPERIMENT/analysis.json`. Dataset support is in
`src/data/relaxed_histories.py`; no separate training loop is added.

Current MACE templates publish flat, portable reports through `src/analysis/report.py`
to `output/mace/<variant>-seed<seed>/` and `output/mace/full/`; start at
[`output/mace/index.html`](../output/mace/index.html). Both UMAP and t-SNE are
enabled; the fast profile no longer replaces the configured MD UMAP with PCA.
For a batch, `--publish-only` copies completed results into those galleries;
`--rerun` recomputes completed analyses, reusing inference caches when present.
The comparison specification's `report_root` selects these flat metrics.
New run and gallery layouts use `plots/`, `tables/` and `technical/`; see [the layout guide](research_layout.md). Detailed MACE analysis artifacts now live on IDS. The two MACE templates set
`cache.retain_after_analysis: false`, suppress topology embedding caches and
validation prediction arrays, and disable duplicate figure sets/paper exports.
Test predictions remain for paired comparisons. New MACE fits retain the selected
best checkpoint and a rolling recovery checkpoint; the recovery checkpoint is
removed only after successful post-training analysis.
Static analysis of `PretrainedMACEGeometry` disables radial compilation while
retaining the trained weights and BF16 arithmetic, avoiding Dynamo's shared
recompilation limit after training. The standard trainer explicitly saves the
final optimizer checkpoint before starting analysis.
The historical explicit TDA and thermal protocols use
`python -m src.training_methods.pretrained_mace --config CONFIG --stage prepare|preflight|train|analysis|all`.
The same command supports `protocol: temporal80` for the
[five-frame transformer experiment](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_temporal_transformer_20260909/README.md).
Its distinct relaxed-anchor objective and history preparation live in
`src/training_methods/mace_temporal/train.py` and `src/data/histories.py`;
the preparation and training stages retain their historical protocol. The
standalone temporal and denoising analysis implementations were removed on
2026-09-10. Their `analysis` and `all` stages now fail explicitly with migration
instructions; historical non-Lightning checkpoint reproduction requires the
recorded source snapshot. New experiments use original VICReg and the standard
pipeline above, including real histories for temporal encoders.
The same family command accepts `protocol: denoising80` for the
[independent Al denoising comparison](../experiments/mace_al_denoising_20260910/README.md).
Its `prepare` stage creates full-cell relaxed targets from explicitly selected
independent MEAM sources and caches frozen MACE atom/frame features; `preflight`
and `train` retain their explicit meanings. Preparation/training implementation is
in `src/training_methods/mace_denoising/data.py`, `src/training_methods/mace_denoising/train.py`,
and `src/models/encoders/mace_denoising.py`.
The config fixes source splits, anchor times, temporal spacing, relaxation
potential, objective/architecture variants, seeds, training budget and deadline.
Relaxation retains CG for existing configurations; `relaxation.minimizer: fire`
requires an explicit `timestep_ps` and a LAMMPS binary supporting that minimizer.
Use a new output/cache for a changed relaxation protocol so target definitions
remain consistent across sources.
The original-VICReg pipeline compares variants on common H0/H1/H2 metrics, with
history interventions and source-level uncertainty; exported temporal encoders require
identity-aligned histories in physical Å.
`protocol: denoising80_reuse` uses an explicit list of completed `denoising80`
and `temporal80` Al shards, implemented in `src/training_methods/mace_denoising/existing_data.py`.
It retains target/potential/minimizer provenance and whole-source assignments;
it does not run new minimizations during preparation. Mixed-cadence temporal
models receive the actual `(B, T)` frame offsets in ps. The optional separate
`--stage potential-audit` performs the recorded matched-input EAM/MEAM test after
training and analysis, using `src/analysis/mace_potential_audit.py`. It requires
the completed encoder exports, an explicit potential-audit config and the
validated shared EAM/MEAM GPU binary. New relaxed artifacts use the maintained
float16 conversion command.
The [80-atom recipe](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_plain80_20260909/README.md)
uses full-graph mean pooling, uniform sampling, fixed spatial/temporal VICReg,
and TDA on the same 80 atoms beginning in epoch six. The real-GPU preflight
checks native MACE equivalence, gradient replay, TDA support, and peak-LR updates.
The encoder, dataset, objective and training implementation are respectively in
`src/models/encoders/pretrained_mace.py`, `src/training_methods/pretrained_mace/data.py`,
`src/training_methods/shared/mace_objective.py` and `src/training_methods/pretrained_mace/train.py`.
The [variant C recipe](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiments/mace_thermal80_20260909/README.md)
uses the same trainer with `protocol: thermal80`: shared hot/relaxed 80-atom
views, relaxed TDA targets, and explicit hot/relaxed consistency. Its paired
cache producer is `python -m src.data.relaxed --config CONFIG`, with
full-periodic fixed-cell minimization in `src/simulation/relaxation.py`.
`convert_trajectory.py relaxation FRAME_DIR --delete-source` verifies this
producer's converged snapshot and stores float16 positions, float32 boxes and
exact integer identities with quantization/checksum provenance.
For detached serial training, frozen probes, and the standard static analysis,
use `python -m src.training_methods.pretrained_mace_queue --plan PLAN` through
the experiment registry. The queue respects the existing allocation deadline.
Frozen topology/forecast probes use
`python -m src.analysis.pretrained_mace_ablation --plan PLAN --run NAME`
or `--collect`; they do not add predictive training losses.
Encoder exports use `model_type: pretrained_mace_encoder` and `protocol: plain80` or `thermal80`
in the existing full analysis pipeline. Older central/tapered MACE training and
its diagnostics were retired on 2026-09-09; old checkpoint inference requires
that run's recorded source snapshot. Results and checkpoint artifacts remain.
For interactive MD rendering, use `python -m src.vis_tools.md_cluster_plot ANALYSIS_DIR`.

## Temporary active-job launchers

`run_lammps_independent_meam_source_campaign.py` and
`run_lammps_independent_meam_510_520K_sources.py` retain the exact paths embedded
in already-submitted source jobs. They forward to the relocated implementation.
Remove them only after both September independent-source campaigns and their
controller chains finish; they are not entry points for new workflows.

`convert_trajectory.py temporal-storage BINARY_DIR... --delete-source` converts
verified temporal float32 artifacts to float16 positions, records periodic
rounding error and original manifests, verifies all arrays, then optionally
replaces old directories with compatibility symlinks. Implementation:
`src/data/conversion/position_storage.py`. The elemental converter defaults
to `--storage-dtype float16`; the campaign passes its configured storage dtype.

`elemental branch --config CONFIG --parents MANIFEST --index INDEX` executes one
frozen Ti position parent without launching a source. It reuses the elemental
branch protocol and converter, records provenance per branch, and obtains MPI
CPU bindings from the Slurm step affinity when submitted as a single-node job.

The [Al crystallization recipe](../docs/simulations/al_crystallization/README.md)
uses `elemental run` with `al-source-then-branches`, the Al shooting MEAM
potential, and the Ti continuous-source/position-branch design.
