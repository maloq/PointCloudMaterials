# Maintained commands

Run commands from the repository root with the `pointnet` environment:

```bash
conda run -n pointnet python scripts/run_shooting_ablation.py --help
conda run -n pointnet python scripts/run_shooting_ablation.py geometry --help
```

Each family lists explicit workflows; `WORKFLOW --help` describes that workflow's
arguments. Changing a run's settings should use configuration or those arguments.
Scientific implementations belong in `src/`. Experiment-specific recipes live in
[`experiments/`](../experiments/README.md); generated results stay in `output/` or
`outputs/`. See the [old-to-new command map](../docs/script_cleanup_20260905.md).

## Simulation and data

| Command | Purpose / implementation |
| --- | --- |
| `run_lammps_campaign.py WORKFLOW` | Campaign preparation, execution, continuation and summaries; `src/simulation/campaigns/` |
| `convert_trajectory.py FORMAT_OR_AUDIT` | Verified format conversion and read-only audits; [conversion guide](../docs/trajectory_conversion.md), `src/data_utils/conversion/` |
| `inspect_temporal_lammps_dataset.py` | Inspect a temporal dump and optionally build its cache |
| `run_optimized_al_homogeneous_campaign.sh` | Resume the selected optimized Aluminum campaign; requires `PYTHON`, accepts `DEVICES` |

The `predictive-dynamics` (48 ps Langevin) and `predictive-dynamics-15ps` (CSLD,
exact continuation) workflows preserve separate protocol contracts. The shared
campaign readers retain their original 70,304-atom Aluminum input contract.
Specialized 510/520 K and Ta recipes live under `experiments/`.

The `elemental run --config CONFIG` workflow provides explicit Ti
source-then-branches and Ta archived-position protocols, implemented in
`src/simulation/campaigns/elemental.py`. See the
[Ti/Ta experiment](../experiments/ti_ta_crystallization_20260907/README.md).
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

`experiment_registry.py build` refreshes the shared experiment/simulation dashboard
and ideas view. `run --spec SPEC.json` tracks an explicit maintained command;
`status --record run_record.json` inspects execution state. `prune --plan PLAN.json`
previews verified cleanup, and `pack-logs --before ISO_TIMESTAMP` previews lossless
log archiving; mutations require `--apply`. Implementation: `src/experiment_runner/`.
See [the registry and run guide](../docs/output_registry.md).

`run --spec SPEC --wait-for-dependencies-until ISO_TIMESTAMP` waits for successful
completion of local tracked dependencies, including their post-training analysis,
before executing the unchanged command. The timestamp must include its UTC
offset; failed/dead dependencies and expired deadlines stop with an explicit error.
Waiting status is written to the spec output's `queue_status.json`.

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
[static-Al experiment recipe](../experiments/temporal_hypotheses_12h_20260906/README.md#standard-static-pipeline-encoder-only).
For MLIP-initialized MACE, use
the original Lightning VICReg entry point with
`python -m src.training_methods.contrastive_learning.train_contrastive --config-name vicreg_pretrained_mace_geometry`.
The [geometry-only recipe](../experiments/mace_original_vicreg_20260909/README.md)
uses normalized 80-point views, a fixed internal species channel, the original
projector/loss/optimizer and standard static analysis. It adds no new runner.
The [matched TDA recipe](../experiments/mace_original_vicreg_tda_20260909/README.md)
adds the optional TDA head within that same module. Prepare normalized three-view
targets with `python -m src.data_utils.spatiotemporal_tda --config configs/vicreg_pretrained_mace_geometry_tda.yaml`,
then select that config in the original training entry point. The producer
reuses the existing alpha-complex descriptor, preserves source row indices,
fits PCA on training views only and records cache checksums.
Static analysis of `PretrainedMACEGeometry` disables radial compilation while
retaining the trained weights and BF16 arithmetic, avoiding Dynamo's shared
recompilation limit after training. The standard trainer explicitly saves the
final optimizer checkpoint before starting analysis.
The earlier explicit TDA and thermal protocols use
`python -m src.training_methods.pretrained_mace --config CONFIG --stage prepare|preflight|train|analysis|all`.
The same command supports `protocol: temporal80` for the
[five-frame transformer experiment](../experiments/mace_temporal_transformer_20260909/README.md).
Its distinct relaxed-anchor objective and history preparation live in
`src/training_methods/mace_temporal.py` and `src/data_utils/mace_history.py`;
`--stage all` runs preparation, GPU preflight, training and held-out history
analysis in sequence. `src/analysis/mace_temporal.py` produces the report,
topology probes, history interventions and sampled spatial maps. This protocol
requires genuine histories and does not feed isolated static clouds to the encoder.
The same family command accepts `protocol: denoising80` for the
[independent Al denoising comparison](../experiments/mace_al_denoising_20260910/README.md).
Its `prepare` stage creates full-cell relaxed targets from explicitly selected
independent MEAM sources and caches frozen MACE atom/frame features; `preflight`,
`train`, `analysis`, and `all` retain their explicit meanings. Implementation is
in `src/data_utils/mace_denoising.py`, `src/training_methods/mace_denoising.py`,
`src/models/encoders/mace_denoising.py`, and `src/analysis/mace_denoising.py`.
The config fixes source splits, anchor times, temporal spacing, relaxation
potential, objective/architecture variants, seeds, training budget and deadline.
Relaxation retains CG for existing configurations; `relaxation.minimizer: fire`
requires an explicit `timestep_ps` and a LAMMPS binary supporting that minimizer.
Use a new output/cache for a changed relaxation protocol so target definitions
remain consistent across sources.
Analysis compares all variants on common H0/H1/H2 metrics, with history
interventions and source-level uncertainty; exported temporal encoders require
identity-aligned histories in physical Å.
`protocol: denoising80_reuse` uses an explicit list of completed `denoising80`
and `temporal80` Al shards, implemented in `src/data_utils/mace_existing.py`.
It retains target/potential/minimizer provenance and whole-source assignments;
it does not run new minimizations during preparation. Mixed-cadence temporal
models receive the actual `(B, T)` frame offsets in ps. The optional separate
`--stage potential-audit` performs the recorded matched-input EAM/MEAM test after
training and analysis, using `src/analysis/mace_potential_audit.py`. It requires
the completed encoder exports, an explicit potential-audit config and the
validated shared EAM/MEAM GPU binary. New relaxed artifacts use the maintained
float16 conversion command.
The [80-atom recipe](../experiments/mace_plain80_20260909/README.md)
uses full-graph mean pooling, uniform sampling, fixed spatial/temporal VICReg,
and TDA on the same 80 atoms beginning in epoch six. The real-GPU preflight
checks native MACE equivalence, gradient replay, TDA support, and peak-LR updates.
The encoder, dataset, objective and training implementation are respectively in
`src/models/encoders/pretrained_mace.py`, `src/data_utils/pretrained_mace.py`,
`src/training_methods/mace_objective.py` and `src/training_methods/pretrained_mace.py`.
The [variant C recipe](../experiments/mace_thermal80_20260909/README.md)
uses the same trainer with `protocol: thermal80`: shared hot/relaxed 80-atom
views, relaxed TDA targets, and explicit hot/relaxed consistency. Its paired
cache producer is `python -m src.data_utils.mace_relaxed --config CONFIG`, with
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
`src/data_utils/conversion/position_storage.py`. The elemental converter defaults
to `--storage-dtype float16`; the campaign passes its configured storage dtype.

`elemental branch --config CONFIG --parents MANIFEST --index INDEX` executes one
frozen Ti position parent without launching a source. It reuses the elemental
branch protocol and converter, records provenance per branch, and obtains MPI
CPU bindings from the Slurm step affinity when submitted as a single-node job.
