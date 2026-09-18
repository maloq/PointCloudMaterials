# Maintained commands

Run from the repository root with `conda run -n pointnet-torch214 python …` for new
GPU work. Exact resumes of existing jobs retain their recorded `pointnet`
environment; see [the PyTorch upgrade](../docs/pytorch214_upgrade.md).
Commands contain argument parsing and forwarding only; scientific implementation
belongs in `src/`. Run-specific settings belong in configuration, not copied runners.
See [workflow details](../docs/workflows.md), [experiment records](../experiments/README.md)
and [the output layout](../docs/research_layout.md).

`python -m src.research.local_predictability.report --config
configs/analysis/local_predictability_report.json --output output/local_predictability/NEW-SUMMARY`
collects completed H100/RTX results and scores deferred observability/readout
predictions on CPU. It verifies paired rows, exports source intervals and figures,
and refuses to overwrite an existing report. It does not fit models. See the
[summary metric definitions](../docs/metrics/local_predictability_summary.md).

| Command | Workflows / implementation |
| --- | --- |
| `project.py datasets --refresh [--output docs/datasets]` | Searchable dataset registry, metadata cards, potentials, current schemas and discovery audit; [start here](../DATASETS.md), `src/project_runtime/dataset_registry.py`. |
| `benchmark_hardware.py [storage,cpu,gpu,all]` | Defaults to all; prints/saves a standard results table. Optional `--cpu-ranks 1 8 24` runs a CPU sweep. Synthetic inputs; `src/hardware_benchmark/`; [usage](../docs/hardware_benchmark.md). |
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

`python -m src.research.gatr_equivariant --config configs/analysis/gatr_equivariant.json`
audits frozen GATr multivector directions on the requested node07/A100. Stages
`temporal`, `spatial`, `report` support resuming extraction; default `all` also
runs readout interventions. It exports source-level angular and spatial-order
metrics and an offline 3D viewer; see the
[scientific protocol](../experiments/gatr_equivariant_20260918/README.md).

`python -m src.research.trajectory_stability --config configs/analysis/trajectory_stability.json`
compares the latest selected MACE/GATr states with TDA, SOAP, bond-order, radial
and angular descriptors on matched, identity-preserving Al trajectories. Stages
`prepare`, `encode` and `report` support separate CPU/GPU execution. It exports
source intervals, full trajectories and exact metric definitions; see the
[scientific protocol](../experiments/trajectory_stability_20260918/README.md).

New geometry-protected 2× snapshot encoders use the prepared recipes in
`configs/shared_pretraining/geometry_fp32_2x/`; see
[architecture, precision and validation](../docs/shared_pretraining_geometry_fp32_2x.md).
These recipes have not been submitted. Reuse the existing `shared_pretraining`
profiler and runtime; no additional training entry point is needed.

The existing v3 VICReg restart uses its frozen code and the command `python -m src.training_methods.shared_pretraining.queue
submit --plan configs/shared_pretraining/restart_b1024_lr002_bf16/campaign.json`.
It starts only the two fresh structural fits in the recorded existing allocations;
see [the restart workflow](../docs/shared_pretraining_restart_20260918.md).
The separate `shared_pretraining.profile` command requires `--precision float32|bf16`
and measures actual batch updates after warmup. It is never called by training.
The original shared-campaign/H200 recipes below are paused historical recipes;
reproduce them with their frozen code, not the current normalized architecture.

`python -m src.training_methods.shared_pretraining.queue submit --plan
configs/shared_pretraining/campaign.json` starts the approved 12-epoch structural
fits, initialized causal continuations and frozen analyses, with detached local
workers and dependent Slurm jobs. It freezes executable code, uses online W&B,
cosine warmup and allocation-aware exact resume. Its data command is
`python -m src.training_methods.shared_pretraining.data --config
configs/shared_pretraining/data.json --workers 6`; see the
[shared campaign workflow](../docs/shared_pretraining_20260918.md).

`python -m src.training_methods.shared_pretraining.queue serial --plan
configs/shared_pretraining/h200_batch1024/campaign.json --deadline-utc ISO_TIME`
runs the batch-1,024 H200 pipelines sequentially without Slurm, checkpoints at
the deadline and resumes completed/partial stages. See the
[H200 handoff](../docs/h200_shared_pretraining_task_20260918.md). Its separate
`shared_pretraining.profile` preflight accepts `--memory-limit-gib 80`; scientific
training never runs hardware benchmarks.

`python -m src.data.structural_pretraining.prepare --config
configs/structural_pretraining/data.json --workers 6` freezes a training-only
multi-material radius calibration and prepares raw snapshot/three-frame inputs,
spatial/temporal neighbor pairs and instantaneous physical/TDA anchors. See the
[structural pretraining workflow](../docs/structural_pretraining_20260917.md).

New structural fits use `src.training_methods.shared_pretraining.queue` and
`configs/shared_pretraining/al_stable/`, with physical/instantaneous-TDA anchors,
full-batch representation statistics and training-only head calibration.
The earlier `configs/structural_pretraining/{mace_vicreg,gatr_vicreg,gatr_lejepa}.json`
are immutable historical run recipes; exact resumes require their frozen source.
The standalone trainer now also requires explicit `materials` and
`head_calibration_rows` for newly created recipes.

`python -m src.data.relaxed_targets prepare|run|status --config
configs/simulation/relaxed_tda_al.json` produces resumable full-cell relaxed-TDA
labels on denser native training windows and completed Al shooting trajectories.
CPU workers share an immutable, ancestry-aware queue; see
[target generation and resume](../docs/relaxed_tda_targets.md).

`python -m src.research.backbone_tda --config
configs/local_predictability/backbone_v2/tda_snapshot.json` compares frozen physical
MACE/GATr snapshot states with matched source-held-out instantaneous-TDA readouts.
See [execution/resume](../docs/backbone_tda.md) and the
[scientific protocol](../experiments/local_predictability_20260917/TDA.md).

`python -m src.research.local_predictability.backbone_v2 --config
configs/local_predictability/backbone_v2/rtx6000_screen.json --stage screen`
runs fresh MACE/cuEquivariance and axial GATr gates, explicit workload profiles,
and a matched one-seed physical snapshot screen. The `fit` stage uses the common
physical/onset trainer without profiling. See [execution/resume](../docs/backbone_v2.md)
and the [scientific protocol](../experiments/local_predictability_20260917/BACKBONE_V2.md).

`python -m src.research.backbone_repeats --config
configs/local_predictability/backbone_v2/h100_repeats.json --stage run` follows the
H100 screen with matched GATr onset parent/snapshot/history/repeated-frame fits,
using the existing v2 trainer and completed MACE reference. `--stage compare-screen`
only checks and exports the completed H100 speed/physical comparison. See the
[H100 extension](../docs/backbone_v2.md#h100-comparison-and-onset-repeats).

`python -m src.research.local_predictability.plan --config
configs/local_predictability/two_gpu_16h.json --output /tmp/local-predictability-queue.json`
validates the one-seed, two-worker 16-hour **planning specification** and writes
job identities. It never trains, submits or resumes jobs. Execution uses
`python -m src.research.local_predictability.data --config configs/local_predictability/h100_execution.json`
for the outcome-independent full-timeline release, followed by `audit` with the
same configuration and `baselines --config configs/local_predictability/rtx6000_baselines.json`
in that package. The maintained tracking wrapper waits for explicit stage records.
Native integration follows the H200 tested implementation; see the
[H100/H200 handoff](../docs/local_predictability_16h.md).

`python -m src.research.local_predictability.observability --config
configs/local_predictability/rtx6000_observability.json` fits current-state and
true-future packet diagnostics on the frozen native window grid. It saves model
checkpoints and predictions for later analysis; future inputs are diagnostic only.
See the [observability protocol](../experiments/local_predictability_20260917/OBSERVABILITY.md).

`python -m src.research.local_predictability.native_queue --config CONFIG
--stages snapshot history12 --worker h100 --resume` assigns disjoint frozen
continuations to workers and raises the process open-file limit for all 150
memory-mapped sources. The RTX worker uses `--stages repeat12 --worker rtx6000`.
`raw_observability --config configs/local_predictability/rtx6000_raw_observability.json`
in the same package trains the all-state raw-atom current-label check.
`native_readouts --config configs/local_predictability/rtx6000_native_readouts.json`
extracts completed onset states and fits fresh linear/MLP predictors. These
diagnostics preserve predictions for later interpretation. The
[operational queue](../docs/local_predictability_16h.md) uses explicit dependencies.

`run_lammps_campaign.py memory-sources prepare --config
configs/simulation/predictive_memory_precision.json --run-name NAME` stages fresh,
split-assigned independent Al sources. `memory-sources run-worker --campaign-root
ROOT --worker-index INDEX --workers COUNT` runs a fixed shard in a 48-rank CPU
allocation. Its paired exports use `convert_trajectory.py memory-pair RUN_DIR`.
See the [simulation record](../docs/simulations/predictive_memory_precision_20260917.md).

`python -m src.data.predictive_memory.prepare --config configs/predictive_memory/pilot.json`
audits existing trajectories and caches strictly partial observations and continuous
physical targets. `python -m src.training_methods.predictive_memory.train --config
configs/predictive_memory/pilot.json --history-ps 48 --velocity` trains the new
label-free memory pilot. `python -m src.training_methods.predictive_memory.compare
--config configs/predictive_memory/pilot.json` compares the eight completed matched
fits. See [workflow and resume](../docs/predictive_memory.md).
For new batched training, use `configs/predictive_memory/batched.json` with the
cuEquivariance backend enabled and the same CLI. It exposes effective/micro/evaluation
batch sizes and bounded GPU input
residency. Current checkpoints use format 2; earlier runs use their original commit.
The comparison's `--modalities xv` option evaluates the four-fit velocity-input
replicate from `configs/predictive_memory/replicate-xv-seed20260918.json`.
`python -m src.training_methods.predictive_memory.diagnose --config CONFIG
--modalities xv` audits frozen heads under constant-state interventions and fits
train-only ridge controls for physical future/present prediction. It writes
separate diagnostics and does not retrain or replace the native encoder.
The `configs/predictive_memory/optimization/` recipes use the same trainer for
matched 12,000-update original/stronger-present-loss fits across two seeds.

`python -m src.research.memory_report --config configs/analysis/memory_research_report.json
--output output/predictive_memory/NEW-SNAPSHOT` freezes completed causal/memory
results, reported H200 means, diagnostic tables and plots without retraining.
It refuses to overwrite a prior snapshot and does not average incomplete seed
cohorts. The dated interpretation is reviewed separately; see the
[consolidated report](../output/predictive_memory/research-summary-20260917/RESULTS.md).

`python -m src.research.mace_velocity causal-prepare --config configs/mace_causal/pilot.json`
prepares identity-preserving physical history/future examples.
`causal-train --config configs/mace_causal/pilot.json --variant D --device cuda:0`
trains the native tensor MACE state; variants A–E and `repeated_anchor` share
data and training controls. See [workflow](../docs/mace_causal.md).

Frozen-feature replacement-embedding workflows were [discarded](../docs/discarded_frozen_encoder_maps.md)
on 16 September 2026. Their source snapshot, recipes and results remain available
for historical reproduction. Embedding forecasting and native encoder training
remain active.

```bash
python -m src.training_methods.contrastive_learning.train_contrastive --config-name vicreg_mace_relaxed
python -m src.analysis.pipeline configs/analysis/static_topology.yaml --checkpoint CHECKPOINT --output-dir output/QUESTION/RUN
python -m src.training_methods.embedding_forecast --config CONFIG.json --stage train
python -m src.data.topology_views --config-name vicreg_mace_relaxed
```

Joint MACE context checkpoints use the existing `src.research.mace_context.run`
stages `static-export` and `static-verify`, followed by `src.analysis.pipeline`
with `configs/analysis/static_mace_context_al.yaml` or `static_mace_context_zr.yaml`.
See [static context analysis](../docs/mace_context_static.md).

Selected snapshot structural GATr–VICReg encoders use
`python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json
--stage export|verify`, then the same `src.analysis.pipeline` command with
`configs/analysis/static_structural_gatr_al.yaml`. The adapter preserves the
native full neighborhood, species and fixed material scale; see
[structural static analysis](../docs/structural_static_analysis.md).

The newest Al-only v6 GATr uses the same commands with
`configs/analysis/structural_gatr_v6_static.json` and
`configs/analysis/static_structural_gatr_v6_al.yaml`, in `pointnet-torch214`.
Verification includes replay of the saved compiled selection features.

The matching Al-only v6 MACE checkpoint uses those same export/verify stages
with `configs/analysis/structural_mace_v6_static.json`, then the pipeline with
`configs/analysis/static_structural_mace_v6_al.yaml`. It preserves MACE's native
per-observation graphs and cuEquivariance/BF16 execution in `pointnet-torch214`.

`python -m src.research.mace_context.cluster_diagnosis --config
configs/analysis/mace_context_clusters.json` runs saved-feature clustering
ablations and a matched GeoFrame V2 comparison. See the
[liquid-cluster diagnosis](../experiments/mace_context_clusters_20260915/README.md).
The follow-up `python -m src.research.mace_context.cluster_probe --config
configs/analysis/mace_context_clusters.json` tests frozen features with spatially
separated physical readouts using the same diagnostic samples.

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

The frozen forecast encoder assay runs with
`python -m src.research.mace_encoder_diagnostics.extract --config
configs/analysis/mace_encoder_diagnostics.json --stage all`, followed by
`python -m src.research.mace_encoder_diagnostics.analyze --config
configs/analysis/mace_encoder_readout.json`. It tests input invariance, geometric
and membership changes, storage precision, TDA generalization and physical time
dependence using retained checkpoints and trajectories. See the
[scientific protocol](../experiments/mace_encoder_diagnostics_20260914/README.md).
`python -m src.research.mace_encoder_diagnostics.verify --config
configs/analysis/mace_encoder_diagnostics.json` checks producer labels, replays
stored forecast embeddings and validates sibling velocities and timelines.

`python -m src.research.mace_tda_ridge_audit.run --config
configs/analysis/mace_tda_ridge_audit.json` re-encodes the original six single-frame
VICReg/TDA checkpoints, checks the training loss with target interventions, and
compares repository and independent ridge calculations for projector and encoder
features. `--stage summarize` rebuilds its tables and paired plot. See the
[audit protocol](../experiments/mace_tda_ridge_audit_20260914/README.md).

The same entry point with `--config configs/analysis/mace_tda_initialization.json
--stage initialization` compares frozen original MLIP and three random MACE
encoders with the audited trained encoder features. It retains the original ridge
protocol and adds a validation-only regularization sensitivity analysis.
With `--config configs/analysis/mace_tda_direct.json --stage direct`, it compares
verified random/MLIP embedding distances and top-10 neighborhoods directly with
the three TDA blocks on held-out structures. This stage fits no readout or scaling;
it reports both global and within-frame geometry, including shuffled controls.

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

`python -m src.research.mace_context.run --config configs/analysis/mace_context.json --stage STAGE` runs the [complete-context MACE pilot](../experiments/mace_context_20260914/README.md). Stages are `prepare`, `verify`, `frozen`, `train`, `labels`, `summarize`, and `smoothness`; extraction/training require an explicit `--mode`. Existing allocation plans can run these commands on assigned GPUs. The CPU-only `smoothness` stage reuses all seven completed feature extractions and writes a separate run suffixed `-smoothness`, with temporal normalization checks, paired source intervals and boundary curves.

The [information-recovery follow-up](../experiments/mace_context_recovery_20260914/README.md)
uses the same module with `--config configs/analysis/mace_context_recovery.json`
and stages `recovery-readouts`, `recovery-verify`, `recovery-train --variant
dual_ssl|dual_physics`, and `recovery-summarize`. Cached readouts compare linear
and nonlinear decoders and inner/center fusion. The distinct joint experiment
trains a shared backbone with or without encoder gradients from physical targets.
# Coordinate/velocity local-state research

The same `src.research.mace_velocity` entry point has `data-prepare`, `data-smoke`
and `data-study` stages for the distinct **end-to-end native encoder** training-data
comparison, using `configs/analysis/mace_data_amount.json`. This updates MACE itself;
it does not train a map on frozen features. See [run instructions](../docs/mace_data_amount.md).

Use `python -m src.research.mace_velocity STAGE --config configs/analysis/mace_velocity.json`
for the separate local phase-space protocol (`inventory`, `prepare`, `verify`,
`teacher`, `train`, `evaluate`). The maintained conversion dispatcher adds
`python scripts/convert_trajectory.py paired-velocity --positions P --velocities V --output O --atoms N`
for atom/time-matched legacy dumps. See [the run instructions](../docs/mace_velocity.md).

`python -m src.research.mace_causal_comparison --config configs/mace_causal/comparison.json`
collects the completed matched-seed pilot into paired physical-error tables and
plots; it verifies source/center/anchor/target identity across every readout.

`python -m src.research.mace_velocity causal-benchmark --config configs/mace_causal/runtime-benchmark.json --device cuda:0`
measures original versus packed/resident execution on verified histories.
`causal-probe --probe-modes linear nonlinear` selects cheap frozen readouts;
`--probe-modes state_constant state_history` selects the matched history-access pair.
See [execution and data handoff](../docs/mace_causal_runtime.md).

The compiled VICReg restart uses the existing shared-pretraining queue and separate profiler; see [the recipe](../docs/shared_pretraining_compiled_repair_20260918.md). No benchmark runs inside training.

The active Al-only VICReg repair uses `configs/shared_pretraining/al_stable/`
with the existing shared-pretraining queue; see [stability and calibrated
checkpoint export](../docs/shared_pretraining_al_stability_20260918.md).

The GATr continuation uses the same `shared_pretraining.queue submit --plan
configs/shared_pretraining/broad_full_tda/campaign.json`. It submits CPU expansion
through the existing `src.data.structural_pretraining.prepare --config` command,
then a GPU job dependent on successful target completion. See the
[full-TDA workflow](../docs/shared_pretraining_broad_full_tda_20260918.md).

The dynamic-only mixed-material GATr recipe reuses `shared_pretraining.queue
submit --plan configs/shared_pretraining/gatr_mixed_triplets/campaign.json`.
See [grouped normalization and detached execution](../docs/shared_pretraining_mixed_triplets_20260918.md).
