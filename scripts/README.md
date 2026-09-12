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
| `inspect_temporal_lammps_dataset.py` | Dataset inspection/cache preparation; `src/data_utils/inspect_temporal_lammps.py` |
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
python -m src.data_utils.spatiotemporal_tda --config-name vicreg_mace_relaxed
```

The MACE preparation command is
`python -m src.research.spatiotemporal.prepare_spatiotemporal_vicreg_views --config CONFIG`.
Recorded older protocols live in [src/research](../src/research/README.md); their
configuration and findings stay with the dated experiment record.

Temporary paths required by submitted jobs remain:
`run_lammps_independent_meam_source_campaign.py`,
`run_lammps_independent_meam_510_520K_sources.py`, and the old
`experiments/spatiotemporal_20260905/prepare_spatiotemporal_vicreg_views.py`.
Do not remove them until the relevant Slurm controller chains finish. The optimized
Aluminum shell launcher still requires `PYTHON` and accepts `DEVICES`.
