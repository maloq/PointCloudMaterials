# Maintained commands

Run from the repository root with `conda run -n pointnet python …`.
Commands contain argument parsing and forwarding only; scientific implementation
belongs in `src/`. Run-specific settings belong in configuration, not copied runners.
See [workflow details](../docs/workflows.md), [experiment records](../experiments/README.md)
and [the output layout](../docs/research_layout.md).

| Command | Workflows / implementation |
| --- | --- |
| `experiment_registry.py build` | Refresh the searchable experiment, simulation and ideas dashboard; `src/experiment_runner/registry.py` |
| `experiment_registry.py storage` | Human-readable size report and large-file CSV; `src/experiment_runner/storage.py` |
| `experiment_registry.py clean [--root output/RUN]` | Preview removable inference caches; add `--apply --inactive` only for inactive runs. Metadata and reconstruction evidence are preserved. |
| `experiment_registry.py metrics-docs` | Check that metric descriptions match their implementing source files. |
| `experiment_registry.py run --spec SPEC` | Track an explicit command, config/source snapshots and outcome; `tracking.py` |
| `experiment_registry.py status --record RECORD` | Inspect tracked execution state. |
| `experiment_registry.py prune --plan PLAN` | Preview a hash-verified explicit deletion plan; `--apply` executes it. |
| `experiment_registry.py pack-logs --before ISO_TIME` | Preview lossless log archiving; `--apply` verifies archives before deleting originals. |
| `experiment_registry.py relocate-caches --plan PLAN` | Preview verified shared-cache relocation; details in [registry guide](../docs/output_registry.md). |
| `run_experiments.py --plan PLAN` | Local/Slurm training plans, explicit resume, collection; `src/experiment_runner/cli.py` |
| `run_lammps_campaign.py WORKFLOW` | Simulation families; `src/simulation/campaigns/` |
| `convert_trajectory.py FORMAT_OR_AUDIT` | Verified producer-specific conversions; [formats and provenance](../docs/trajectory_conversion.md) |
| `inspect_temporal_lammps_dataset.py` | Dataset inspection/cache preparation; `src/data/inspect_temporal.py` |
| `run_shooting_ablation.py METHOD --config CONFIG` | Distributional, spatial, geometry, multiscale, dynamical, short-horizon, encoder-finetune, temporal-pretraining; `src/temporal_vamp/commands/` |
| `run_predictive_atlas.py METHOD --config CONFIG` | Frozen, history, finetune, temporal-encoder protocols; `src/temporal_vamp/commands/` |
| `analyze_geoframe.py WORKFLOW` | Stability, variability and representation comparisons. |
| `train_geoframe_spatiotemporal.py` | Config-selected training; `src/training_methods/spatiotemporal.py` |
| `run_temporal_vamp.py`, `evaluate_temporal_vamp.py` | VAMP training/evaluation; `src/temporal_vamp/` |
| `run_shooting_predictor.py`, `evaluate_shooting_ablation0.py` | Baseline shooting prediction/evaluation. |
| `run_nested_committor.py`, `run_predictability_map.py` | Configured committor/predictability analysis. |
| `run_temporal_encoder_pretraining.py`, `run_ordinary_temporal_embedding_cache.py` | Temporal pretraining/embedding production. |
| `analyze_shooting_branch_outcomes.py` | Branch statistics. |
| `plot_experiment_summary.py`, `plot_grouped_metric_csv.py` | Summary/CSV plotting; `src/analysis/plots/` |
| `plot_homogeneous_checkpoint.py`, `render_shooting_dynamics_gifs.py` | Simulation visualization. |

Current training and analysis use existing module entry points:

```bash
python -m src.training_methods.contrastive_learning.train_contrastive --config-name vicreg_mace_relaxed
python -m src.analysis.pipeline configs/analysis/static_topology.yaml --checkpoint CHECKPOINT --output-dir output/QUESTION/RUN
python -m src.training_methods.embedding_forecast --config CONFIG.json --stage train
python -m src.data.topology_views --config-name vicreg_mace_relaxed
```

The MACE preparation command is
`python -m src.research.spatiotemporal.prepare_spatiotemporal_vicreg_views --config CONFIG`.
Forecast embedding storage migration uses `convert_trajectory.py embedding-cache`;
see the [conversion protocol](../docs/trajectory_conversion.md).
Forecast training accepts `--runtime-config RUNTIME.json` with
`{"loader":"resident","log_every_steps":100}` to retain stored embeddings on
the compute device and report batch progress. The default `mmap` loader remains
available. Explicit `--resume-transition TRANSITION.json` permits a reviewed,
exact old/new implementation hash transition while restoring the saved training
state and preserving the scientific configuration.
`python -m src.training_methods.embedding_forecast.handoff --plan PLAN.json
--variant NAME` replaces a tracked forecast process inside its existing Slurm
allocation. Its plan supplies exact job/node identities, the original execution
record and checkpoint directory, and the replacement run specification. See the
[GPU restart record](../experiments/embedding_forecast_20260911/GPU_RESIDENT_RESTART_20260913.md)
for the plan schema, allocation holding, and guarded collection semantics.
Observed-history sweeps use the same forecast training command with per-context
configuration files. The distinct paired comparison is
`python -m src.research.forecast_context.compare --plan PLAN.json`; it verifies
matched windows and normalization across histories and exports source-weighted
scores, paired intervals and plots. Inputs are completed forecast artifacts and
a plan listing their configs, seeds and result root. See the
[context experiment](../experiments/forecast_context_20260913/README.md).
Local crystallization assessment uses the exact tracked center atoms and existing
PTM implementation: `python -m src.research.forecast_crystallization.local_data
--config CONFIG`, then `local_predict` and `local_analyze` in the same package.
Inputs are an existing forecast cache, source trajectories and frozen checkpoints;
outputs are local physical labels, a train-only crystal readout, paired forecasts,
transition/timing tables and plots. See the
[local crystal assay](../experiments/forecast_crystallization_20260913/README.md).

Spatial/mixture forecasting uses the same training command. Runtime setting
`validation_residency: host` retains validation inputs in RAM to leave GPU space for
the training split and large batches. `validation_residency: staged_device` copies
that split to the GPU once per validation pass and releases it before backpropagation;
see [performance measurements](../docs/forecast_performance.md). Prepare periodic
observed-frame neighbor indices with
`python -m src.training_methods.embedding_forecast.spatial --config CONFIG`.
Its config declares the existing embedding cache, simulation source manifest,
neighbor count and output. Training variants select spatial inputs and a Gaussian
mixture over the complete future path; spatial training requires the resident loader.
`prepool_embeddings: true` stores hashed same-frame neighbor means in the cache
dtype, allowing broad neighborhoods to share the same GPU batch cost. Preparation
publishes `status.json` for queued consumers and accepts resolved storage tokens.
The [spatial-context recipes](../configs/embedding_forecast/spatial-context/analysis.json)
compare 32/128/512 centers against the existing 8-center control.
Learned individual-neighbor attention uses a separate geometry sidecar, prepared by
`python -m src.training_methods.embedding_forecast.spatial --stage geometry --config
configs/embedding_forecast/spatial-attention/geometry.json`. Its training configs
select `spatial_attention` and `data.geometry_cache`; the same forecast training and
physical-analysis commands apply. `training.micro_batch_size` accumulates gradients
within the unchanged effective batch. Runtime `validation_residency: swap_device`
moves attention training tensors to host during validation; `evaluation_batch_size`
sets the validation/test batch independently. See the
[attention protocol](../experiments/forecast_spatial_attention_20260914/README.md).
After collection, `python -m src.research.forecast_spatial_mixture.compare --stage
summarize --plan configs/embedding_forecast/spatial-context/report-20260914.json`
creates a fresh six-figure gallery of spatial size, paired uncertainty, timing,
history, trajectory quality and temperature strata from completed artifacts.
For completed fits, `python -m src.research.forecast_spatial_mixture.evaluate
--plan PLAN --run NAME --seed SEED` reuses the frozen local PTM assay and scores
both mean paths and probabilistic crystal readouts. Then `compare --plan PLAN` in
the same package verifies matched windows/scales and exports the full paired study.
See the [spatial/mixture experiment](../experiments/forecast_spatial_mixture_20260913/README.md).
`python -m src.research.forecast_spatial_mixture.trajectories --config CONFIG`
visualizes measured local point-cloud evolution, observed embedding trajectories,
and alternative futures from completed mixture checkpoints using one train-fitted UMAP.
The full-trajectory overview separates occupancy from time; feature heatmaps explain
changes in the frozen crystal score with training-ranked channels. The
[`trajectory-visuals.json`](../configs/embedding_forecast/trajectory-visuals.json)
recipe exports PNG/PDF figures, a GIF, offline 3D time sliders and exact plotted
arrays to a fresh output. It uses CPU inference on selected diagnostic examples.
`--stage plots` refreshes figures from that output's retained extraction without
loading checkpoints or simulation trajectories again.
`python -m src.training_methods.embedding_forecast.allocation --plan PLAN` runs
explicit training/analysis module commands serially inside an existing allocation.
Its plan specifies node/job identity, prerequisite status files, exact arguments,
completion states and an allocation deadline; it never submits or implicitly resumes
a job. An optional `status_path` publishes progress at an existing dependency's
location after its previous controller has been retired and its status archived.
The paired collector accepts explicit `comparison_pairs` and `reference_runs`
with retained `local_directory` paths for the [short-history extension](../experiments/forecast_spatial_mixture_20260913/SHORT_HISTORY.md).

Recorded older protocols live in [src/research](../src/research/README.md); their
configuration and findings stay with the dated experiment record.

The September 13 queue audit found no remaining simulation controllers. The old
spatiotemporal experiment-path forwarder was retired; use the maintained preparation
module above. Historical simulation specialization/queue code is now under
`src/simulation/campaigns/` (`independent_meam_high_temperature`, `local_source_queue`,
`ta_initial_branch`, `al_crystallization_preflight`, `recover_ta_ti_float16`).
The two existing `run_lammps_independent_meam_*` forwarding commands still import
maintained simulation implementations. Remove those two temporary launchers only
after the [post-queue cleanup conditions](../docs/src_refactor.md#post-queue-cleanup-checklist)
are satisfied. Historical exact protocols and arguments
are documented in [simulation records](../docs/simulations/README.md).
The historical Aluminum shell launcher requires `PYTHON` and an explicit
`CAMPAIGN_CONFIG`, and accepts `DEVICES`; restore its dependency tree as described
in the [config index](../configs/README.md). The FactorVAE and historical GeoFrame
objective queues require `--config-dir` for restored recipes; the spatiotemporal
trainer accepts it too. All maintained simulation recipes live in `configs/simulation/`.

`project.py` manages machine settings (`paths`, `doctor`), dataset IDs (`datasets`), JSON resolution (`resolve`), verified selected exports (`bundle`, `verify-bundle`), full checkout snapshots (`snapshot`), completed simulation publication (`publish-simulation`), and stopped failure archives (`archive-failed-simulation --inactive`). Inputs and copy semantics are documented in [portability](../docs/portability.md); implementation is in `src/project_runtime/`.

`project.py simulations --output docs/simulations` exports readable collection and
producer-outcome CSV indexes from `configs/datasets.json`; implementation is in
`src/project_runtime/simulation_inventory.py`. It preserves failed/duplicate attempt
evidence and never treats outcome records as counts of independent simulations.

Source ownership and retained historical import/command paths are documented in
[the source refactor record](../docs/src_refactor.md#completion-pass-starting-at-d119fd2).
