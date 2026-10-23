# Maintained commands

Run from the repository root in conda `pointnet-torch214`:

```bash
conda activate pointnet-torch214
python scripts/project.py --help
python scripts/run_experiments.py --help
python -m src.postprocessing --help
```

Tables below abbreviate `python scripts/COMMAND` to `COMMAND`. Module commands
use `python -m MODULE`. Run `COMMAND --help` or `COMMAND STAGE --help` for exact
arguments; a list of stages is not a shell pipeline.

Choose inputs through [DATASETS.md](../DATASETS.md), recipes through the
[configuration index](../configs/README.md), and scientific protocols through the
[research index](../src/research/README.md). New runs follow [AGENTS.md](../AGENTS.md).
Historical commands require their recorded code, environment, inputs and recipe;
they are not templates for new experiments. Check a campaign's launch receipt
before resuming or submitting it again.

## Environment and development

| Command | Purpose / guide |
| --- | --- |
| `project.py doctor` | Inspect machine paths and core packages; add `--profile cpu` or `--profile gpu` for scientific imports, pinned versions and tensor execution, or `--lammps` for the simulation executable. [Environments](../environments/README.md) |
| `project.py check [--output OUTPUT]` | Run lightweight repository integration checks. [Development guide](../docs/development.md) |
| `project.py example --output OUTPUT` | Run the small CPU data/train/checkpoint/resume/evaluation reference; optional `--config configs/examples/cpu_pipeline.json`. [Development guide](../docs/development.md) |
| `project.py paths` | Show resolved storage locations from machine settings. [Portability](../docs/portability.md) |
| `project.py resolve CONFIG` | Resolve dataset IDs and machine-dependent paths in a JSON recipe. [Portability](../docs/portability.md) |
| `benchmark_hardware.py all` | Local synthetic storage/CPU/GPU benchmarks; individual modes and CPU-rank sweeps are available. [Hardware guide](../docs/hardware_benchmark.md) |

Keep hardware benchmarks and development checks local; scientific training uses
online W&B with the recorded resumable run ID. Historical exact resumes retain
their original environment; see the [PyTorch upgrade](../docs/pytorch214_upgrade.md).

## Datasets and postprocessing

| Command | Purpose / guide |
| --- | --- |
| `project.py datasets --refresh` | Refresh dataset cards, schemas, provenance and availability. [Dataset registry](../DATASETS.md) |
| `project.py simulations --details --output DIRECTORY` | Export measured simulation holdings and producer outcomes. [Simulation index](../docs/simulations/README.md) |
| `project.py relaxations scan`, `status`, `plan` | Find reusable relaxed cells/neighborhoods and plan missing work. [Relaxation index](../docs/data_storage.md#unified-relaxation-index) |
| `convert_trajectory.py FORMAT_OR_AUDIT` | Convert or verify a declared repository producer. [Formats, precision and deletion rules](../docs/trajectory_conversion.md) |
| `inspect_temporal_lammps_dataset.py` | Inspect a temporal dump and prepare its existing loader cache. [Workflows](../docs/workflows.md) |
| `python -m src.data.al_pool` | Refresh, inspect or select from the shared Al inventory. [Al research pool](../docs/al_research_pool.md) |
| `python -m src.postprocessing` | Full-cell descriptors, nucleus lineages, catalogues, histories and visualization. [Postprocessing guide](../docs/postprocessing.md) |

The postprocessing dispatcher exposes these groups; use `GROUP --help` for stages:

| Groups | Purpose |
| --- | --- |
| `inventory`, `descriptors`, `gpu-extract`, `cache` | Verify inputs, prepare structural labels, and reuse extraction caches. |
| `search`, `al-gpu-search`, `ni-search`, `ni-gpu-search` | Run the declared material/cadence-specific nucleus search. |
| `dataset`, `labels` | Publish and verify birth catalogues and separate label supplements. |
| `histories`, `ni-histories`, `publish` | Extract precursor histories and publish completed packets. |
| `targets`, `spatial-dataset`, `spatial-enrich`, `spatial-relaxed` | Prepare spatial training inputs and descriptor targets. |
| `view`, `export`, `ni-preview`, `crops`, `crop-windows` | Render saved events, monitoring plots, crops and portable galleries. |
| `target-benchmark`, `gpu-benchmark` | Run explicit descriptor accuracy/performance audits. |

The [Al birth catalogue](../docs/datasets/nuclei_birth.md) keeps exact 0.10-ps and
native-only 0.30-ps protocols separate. [GPU extraction](../docs/postprocessing_gpu.md)
and [history scheduling](../docs/nuclei_search_performance.md) describe reuse and
parallel workers. The original Al 0.75-ps production histories are
[retired](../docs/storage/al075_removal_20261007/README.md).

## Simulation campaigns

Start with `run_lammps_campaign.py --help` and the
[simulation recipes](../configs/simulation/README.md). Its maintained families
include `elemental`, `homogeneous`, `birth-sources`, `memory-sources`, and the
explicit MEAM source/shooting/ensemble protocols listed by help. Commands that
have their own research protocol retain module entry points:

| Module (`python -m …`) | Purpose / operational record |
| --- | --- |
| `src.simulation.campaigns.ni_births.workflow` | Ni melt/cooling/hold preparation, validation, submission and status. [Original campaign](../docs/simulations/ni_births_010_20261007.md), [low-temperature revision and handoff](../docs/simulations/ni_births_lowT_010_20261008.md) |
| `src.simulation.campaigns.ni_ordering_branches` | Native Ni continuation pilot. [Workflow](../docs/simulations/ni_ordering_branches_20261019.md) |
| `src.simulation.campaigns.dense_al` | Exact 0.10-ps Al preparation, continuation and collection. [Campaign](../docs/simulations/al_main_010ps_20260927/README.md), [halfway stopping](../docs/simulations/al_main_half_stop_20261001/README.md) |
| `src.simulation.campaigns.dense_al_gpu` | GPU execution of declared dense-Al work. [Workflow](../docs/simulations/al_dense_gpu_20261004.md) |
| `src.simulation.campaigns.dense_al_shooting`, `src.simulation.campaigns.dense_al_birth_queue` | Dense reshoots and the separately defined birth-focused queue. [Protocol, repair and receipts](../docs/simulations/al_dense_shooting_v100_20261007.md) |
| `src.simulation.campaigns.al_million_replay` | Native-checkpoint replay for million-atom Al Q/V. [Workflow](../docs/simulations/al_million_velocity_replay_20261007.md) |
| `src.simulation.campaigns.position_shooting` | Full-cell Ta position shooting. [Workflow](../docs/simulations/ta_shooting_20260926.md) |

Use [simulation records](../docs/simulations/README.md) for existing source
assignments, duration changes, handoffs, failures and completion receipts.
Ownership-sensitive Ni `allocation`, `scale`, `handoff` and `hard_stop` commands
are documented with their campaign; copying their old invocation is not a new
submission plan. Preserve native restarts and use the matching trajectory converter.

## Training and prediction

| Command | Purpose / guide |
| --- | --- |
| `run_experiments.py --plan PLAN` | Local/Slurm plan execution; `--dry-run`, `--resume OUTPUT`, and `--collect OUTPUT` expose preparation and continuation. [Workflow](../docs/workflows.md) |
| `run_shooting_ablation.py METHOD --config CONFIG` | Distributional, spatial, geometry, multiscale, dynamical and temporal shooting protocols. [Workflow](../docs/workflows.md) |
| `run_predictive_atlas.py METHOD --config CONFIG` | Frozen/history/finetune/temporal-encoder comparisons. [Workflow](../docs/workflows.md) |
| `train_geoframe_spatiotemporal.py` | Config-selected GeoFrame training. [Workflow](../docs/workflows.md) |
| `run_temporal_vamp.py`, `evaluate_temporal_vamp.py` | Temporal VAMP training and evaluation. [Workflow](../docs/workflows.md) |
| `run_shooting_predictor.py`, `evaluate_shooting_ablation0.py` | Baseline shooting prediction and evaluation. [Workflow](../docs/workflows.md) |
| `run_nested_committor.py`, `run_predictability_map.py` | Configured committor and predictability analyses. [Workflow](../docs/workflows.md) |
| `run_temporal_encoder_pretraining.py`, `run_ordinary_temporal_embedding_cache.py` | Temporal pretraining and ordinary-trajectory embedding preparation. [Workflow](../docs/workflows.md) |

Research-specific training keeps distinct objectives and data contracts. These
are the principal navigation routes; their guides supply stages, frozen recipes
and reproduction commands:

| Module (`python -m …`) | Research workflow |
| --- | --- |
| `src.training_methods.contrastive_learning.train_contrastive` | Config-selected snapshot encoder training; [encoder handbook](../docs/encoder_research/README.md). |
| `src.training_methods.shared_pretraining.queue` | Shared structural pretraining and recorded continuations; [workflow](../docs/shared_pretraining_20260918.md). |
| `src.training_methods.embedding_forecast` | Prepare, train, resume and evaluate embedding forecasts; [workflow](../docs/workflows.md#training-and-predictive-analysis). |
| `src.training_methods.predictive_memory.train` | Label-free partial-observation memory; [workflow](../docs/predictive_memory.md). |
| `src.research.birth_prediction.queue` | Al preappearance prediction and matched readouts; [workflow](../docs/birth_prediction.md). |
| `src.research.continuous_motion.workflow` | Al fixed-step/continuous/response comparison; [workflow](../docs/continuous_motion.md). |
| `src.research.structural_response.workflow` | Structural initialization and response-supervised motion; [workflow](../docs/structural_response.md). |
| `src.research.liquid_predictability.queue` | Liquid-only descriptor/encoder controls; [workflow](../docs/liquid_predictability.md), [rich-descriptor fits](../docs/rich_descriptor_encoder.md). |
| `src.research.mace_velocity` | Local coordinate/velocity and causal-state protocols; [local state](../docs/mace_velocity.md), [causal training](../docs/mace_causal.md). |
| `src.research.ni_phase_sim.measurement_queue` | Ni measured phase-space supervision; [workflow](../docs/ni_phase_sim.md). |
| `src.research.measurement_sim.rate_campaign` | Direct Ni measurement trajectories and density rates; [workflow](../docs/measurement_sim.md). |
| `src.research.ni_kinetic_latent.queue`, `src.research.ni_spatial_scales.queue` | Ni kinetic representations and observed spatial support; [kinetic](../docs/ni_kinetic_latent.md), [spatial](../docs/ni_spatial_scales.md). |
| `src.research.ni_latent_flow.queue` | Ni native trajectory methods and validation; [methods](../docs/ni_latent_flow.md), [validation](../docs/ni_latent_validation.md). |
| `src.research.ni_ordering_flow.queue` | Ni observed-history/context comparison; [workflow](../docs/ni_ordering_flow.md). |

See the [research package index](../src/research/README.md) for additional
retained protocols, including response encoders, distance/context predictors,
structured forecasts and frozen diagnostics. The [Ni architecture map](../docs/active_ni_architecture.md)
traces the latest Ni queues through shared data, models, training and evaluation.

## Analysis and figures

| Command | Purpose / guide |
| --- | --- |
| `python -m src.analysis.pipeline CONFIG` | Standard checkpoint analysis. [Static analysis and adapters](../docs/structural_static_analysis.md) |
| `python -m src.analysis.cluster_order`, `python -m src.analysis.cluster_explorer` | Saved-cluster order summaries and linked exploration. [Visualization](../docs/analysis_visualization.md) |
| `analyze_geoframe.py WORKFLOW` | Stability, variability and representation comparisons. [Workflow](../docs/workflows.md) |
| `analyze_shooting_branch_outcomes.py` | Saved shooting-branch outcome statistics. [Workflow](../docs/workflows.md) |
| `plot_experiment_summary.py`, `plot_grouped_metric_csv.py` | Summary-JSON and grouped-CSV figures. [Workflow](../docs/workflows.md#plots) |
| `plot_homogeneous_checkpoint.py`, `render_shooting_dynamics_gifs.py` | Configured simulation checkpoint and trajectory visualization. [Workflow](../docs/workflows.md#plots) |

Use the recorded native checkpoint adapter's `export` and `verify` stages before
standard analysis. The [paired structure viewers](../docs/spatial_vicreg_bias.md)
and research guides document saved-result rendering without repeating inference.

## Results, provenance and storage

| Command | Purpose / guide |
| --- | --- |
| `experiment_registry.py build`, `encoders` | Refresh experiment/simulation and encoder catalogues. [Registry](../docs/output_registry.md), [encoder catalogue](../docs/encoder_research/README.md) |
| `experiment_registry.py run --spec SPEC`, `status --record RECORD` | Track explicit execution and inspect its receipt. [Registry](../docs/output_registry.md) |
| `experiment_registry.py publish`, `results`, `verify-results` | Publish retained analyses and verify result evidence. [Result system](../docs/research_results_system.md) |
| `experiment_registry.py metrics-docs` | Report metric documentation/source drift; optional `--strict` for an explicit gate. Training/export do not run this audit. [Development guide](../docs/development.md) |
| `experiment_registry.py storage`, `clean`, `prune`, `pack-logs`, `relocate-caches` | Inspect holdings and preview explicit cleanup/archive/relocation operations. [Registry and mutation flags](../docs/output_registry.md) |
| `project.py bundle`, `verify-bundle`, `snapshot` | Export selected verified artifacts or a checkout snapshot. [Portability](../docs/portability.md) |
| `project.py publish-simulation`, `archive-failed-simulation` | Publish completed simulations or preserve stopped failures. [Portability](../docs/portability.md) |

Outputs follow `output/<question>/<run>/`, with analyses, plots, tables and
execution evidence separated by [the result layout](../docs/research_layout.md).
Storage operations require their explicit plans and documented mutation flags;
source identities, checkpoints, predictions and scientific results are retained.

## Adding or recovering a command

Reuse an existing config/CLI option for seed, dataset, capacity and output-path
variations. Keep entry points thin and implementation in `src/`. Add a row here
for a new command family and link its operational guide; keep launch chronology
in that guide, simulation records in `docs/simulations/`, and scientific questions
and findings in `experiments/`.

[Retired architectures](../docs/architecture_retirement.md) and the
[October 23 code retirement](../docs/src_refactor.md#code-retirement--2026-10-23)
record removed commands and preserved source revisions. The retained
`run_optimized_al_homogeneous_campaign.sh` is a historical launcher: restore its
recorded dependency tree and provide explicit `PYTHON` and `CAMPAIGN_CONFIG` as
listed in [the config index](../configs/README.md). Historical results and frozen
producers remain the authority for reproduction.
