# Script cleanup and command map

The cleanup separates maintained commands, shared implementation, experiment
recipes and generated outputs. Existing uncommitted implementation edits were
preserved when relocating files. No dataset conversion, training or simulation
was launched. See [commands](../scripts/README.md),
[experiments](../experiments/README.md), and
[format conversion](trajectory_conversion.md).

## Design decisions

- Family commands explicitly dispatch existing scientific workflows; different
  ablation targets and the Langevin/CSLD continuation contracts remain separate.
- Ablations share config/output setup, matching cache preparation, and matching
  distributional target preparation. Model fitting/evaluation differences remain
  in their respective modules.
- Campaign I/O and constants moved to `src/simulation/campaigns/common.py` without
  widening the original 70,304-atom reader contract. Campaign implementations may
  reuse these helpers; command scripts are not libraries.
- Plot implementation lives under `src/analysis/plots/`; the experiment runner
  imports it directly. Stable plot commands remain thin wrappers.
- Experiment-specific queues, view preparation and comparisons are retained with
  their study. Hydra configs remain in their composition root under `configs/`.
- Conversion utilities remain maintained because current readers, tests and
  provenance depend on them. Their implementations now have one public CLI and
  source-preserving defaults, instead of being deleted as apparently old scripts.
- Agent instructions require existing-command discovery and documented placement
  of new code. No live script archive is maintained.

## Reproduction and jobs

Use commands below from the repository root with the `pointnet` environment.
Arguments are unchanged unless noted: atlas now requires a method name; grouped
commands require the workflow shown below; maintained spatiotemporal training
requires `--config-name`; shooting conversion uses optional `--delete-source`
instead of required `--delete-originals`. Temporal conversion also keeps sources
unless `--delete-source` is supplied.

Repository-owned imports, tests, documentation and job generators were updated.
Already-generated external Slurm scripts and saved code snapshots are historical
artifacts and were not rewritten. Before resubmitting an old generated job,
regenerate it through the updated workflow or update its old launcher path using
this map. New generated jobs refer to relocated directly executable modules.
Existing manifest/report names and their scientific contents remain compatible.

## Relocated files

| Previous path | Current command | Implementation / recipe |
| --- | --- | --- |
| `scripts/analyze_geoframe_temporal_stability.py` | `python scripts/analyze_geoframe.py stability` | `src/temporal_vamp/commands/geoframe_stability.py` |
| `scripts/analyze_spatiotemporal_vicreg_results.py` | `python experiments/spatiotemporal_20260905/analyze_spatiotemporal_vicreg_results.py` | `experiments/spatiotemporal_20260905/analyze_spatiotemporal_vicreg_results.py` |
| `scripts/audit_lammps_temporal_float32.py` | `python scripts/convert_trajectory.py audit-temporal` | `src/data_utils/conversion/audit_temporal.py` |
| `scripts/compare_geoframe_stability_representations.py` | `python scripts/analyze_geoframe.py compare-representations` | `src/temporal_vamp/commands/geoframe_compare_representations.py` |
| `scripts/compare_geoframe_temporal_variability.py` | `python scripts/analyze_geoframe.py compare-variability` | `src/temporal_vamp/commands/geoframe_compare_variability.py` |
| `scripts/compare_spatiotemporal_objectives.py` | `python experiments/spatiotemporal_20260905/compare_spatiotemporal_objectives.py` | `experiments/spatiotemporal_20260905/compare_spatiotemporal_objectives.py` |
| `scripts/compare_spatiotemporal_static_clustering.py` | `python experiments/spatiotemporal_20260905/compare_spatiotemporal_static_clustering.py` | `experiments/spatiotemporal_20260905/compare_spatiotemporal_static_clustering.py` |
| `scripts/convert_lammps_shooting_binary.py` | `python scripts/convert_trajectory.py export-shooting` | `src/data_utils/conversion/shooting_export.py` |
| `scripts/export_trajectory_npz_to_lammps_dump.py` | `python scripts/convert_trajectory.py export-npz-dump` | `src/data_utils/conversion/npz_dump.py` |
| `scripts/migrate_lammps_shooting_float32.py` | `python scripts/convert_trajectory.py shooting` | `src/data_utils/conversion/shooting.py` |
| `scripts/migrate_lammps_temporal_float32.py` | `python scripts/convert_trajectory.py temporal` | `src/data_utils/conversion/temporal.py` |
| `scripts/plot_experiment_summary.py` | `python scripts/plot_experiment_summary.py` | `src/analysis/plots/plot_experiment_summary.py` |
| `scripts/plot_grouped_metric_csv.py` | `python scripts/plot_grouped_metric_csv.py` | `src/analysis/plots/plot_grouped_metric_csv.py` |
| `scripts/plot_homogeneous_checkpoint.py` | `python scripts/plot_homogeneous_checkpoint.py` | `src/analysis/plots/plot_homogeneous_checkpoint.py` |
| `scripts/plotting_common.py` | `Shared library; no command` | `src/analysis/plots/plotting_common.py` |
| `scripts/prepare_spatiotemporal_vicreg_views.py` | `python experiments/spatiotemporal_20260905/prepare_spatiotemporal_vicreg_views.py` | `experiments/spatiotemporal_20260905/prepare_spatiotemporal_vicreg_views.py` |
| `scripts/run_geoframe_factor_vae_queue.py` | `python experiments/factor_vae_20260901/run_queue.py` | `experiments/factor_vae_20260901/run_queue.py` |
| `scripts/run_geoframe_spatiotemporal_post_analysis.py` | `python experiments/spatiotemporal_20260905/run_geoframe_spatiotemporal_post_analysis.py` | `experiments/spatiotemporal_20260905/run_geoframe_spatiotemporal_post_analysis.py` |
| `scripts/run_geoframe_temporal_variability.py` | `python scripts/analyze_geoframe.py variability` | `src/temporal_vamp/commands/geoframe_variability.py` |
| `scripts/run_lammps_homogeneous_campaign.py` | `python scripts/run_lammps_campaign.py homogeneous` | `src/simulation/campaigns/homogeneous.py` |
| `scripts/run_lammps_independent_meam_510_520K_sources.py` | `python experiments/independent_sources_20260903/independent_meam_510_520K_sources.py` | `experiments/independent_sources_20260903/independent_meam_510_520K_sources.py` |
| `scripts/run_lammps_independent_meam_source_campaign.py` | `python scripts/run_lammps_campaign.py independent-meam-source` | `src/simulation/campaigns/independent_meam_source.py` |
| `scripts/run_lammps_local_source_queue.py` | `python experiments/independent_sources_20260903/local_source_queue.py` | `experiments/independent_sources_20260903/local_source_queue.py` |
| `scripts/run_lammps_meam_nested_shooting_campaign.py` | `python scripts/run_lammps_campaign.py meam-nested-shooting` | `src/simulation/campaigns/meam_nested_shooting.py` |
| `scripts/run_lammps_meam_shooting_campaign.py` | `python scripts/run_lammps_campaign.py meam-shooting` | `src/simulation/campaigns/meam_shooting.py` |
| `scripts/run_lammps_meam_shooting_followup.py` | `python scripts/run_lammps_campaign.py meam-shooting-followup` | `src/simulation/campaigns/meam_shooting_followup.py` |
| `scripts/run_lammps_nested_fixed_horizon_compatibility.py` | `python scripts/run_lammps_campaign.py nested-fixed-horizon-compatibility` | `src/simulation/campaigns/nested_fixed_horizon_compatibility.py` |
| `scripts/run_lammps_predictive_dynamics_15ps_campaign.py` | `python scripts/run_lammps_campaign.py predictive-dynamics-15ps` | `src/simulation/campaigns/predictive_dynamics_15ps.py` |
| `scripts/run_lammps_predictive_dynamics_campaign.py` | `python scripts/run_lammps_campaign.py predictive-dynamics` | `src/simulation/campaigns/predictive_dynamics.py` |
| `scripts/run_lammps_seeded_crystallization_campaign.py` | `python scripts/run_lammps_campaign.py seeded-crystallization` | `src/simulation/campaigns/seeded_crystallization.py` |
| `scripts/run_lammps_ta_initial_branch.py` | `python experiments/ta_source_20260905/ta_initial_branch.py` | `experiments/ta_source_20260905/ta_initial_branch.py` |
| `scripts/run_lammps_unseeded_meam_crystallization.py` | `python scripts/run_lammps_campaign.py unseeded-meam-crystallization` | `src/simulation/campaigns/unseeded_meam_crystallization.py` |
| `scripts/run_lammps_unseeded_meam_ensemble.py` | `python scripts/run_lammps_campaign.py unseeded-meam-ensemble` | `src/simulation/campaigns/unseeded_meam_ensemble.py` |
| `scripts/run_lammps_unseeded_meam_source_followup.py` | `python scripts/run_lammps_campaign.py unseeded-meam-source-followup` | `src/simulation/campaigns/unseeded_meam_source_followup.py` |
| `scripts/run_lammps_unseeded_meam_temperature_campaign.py` | `python scripts/run_lammps_campaign.py unseeded-meam-temperature` | `src/simulation/campaigns/unseeded_meam_temperature.py` |
| `scripts/run_predictive_atlas.py` | `python scripts/run_predictive_atlas.py frozen` | `src/temporal_vamp/commands/atlas_frozen.py` |
| `scripts/run_predictive_atlas_expanded_finetune.py` | `python scripts/run_predictive_atlas.py finetune` | `src/temporal_vamp/commands/atlas_finetune.py` |
| `scripts/run_predictive_atlas_history.py` | `python scripts/run_predictive_atlas.py history` | `src/temporal_vamp/commands/atlas_history.py` |
| `scripts/run_predictive_atlas_temporal_encoder.py` | `python scripts/run_predictive_atlas.py temporal-encoder` | `src/temporal_vamp/commands/atlas_temporal_encoder.py` |
| `scripts/run_shooting_distributional_ablation.py` | `python scripts/run_shooting_ablation.py distributional` | `src/temporal_vamp/commands/ablation_distributional.py` |
| `scripts/run_shooting_dynamical_ablation.py` | `python scripts/run_shooting_ablation.py dynamical` | `src/temporal_vamp/commands/ablation_dynamical.py` |
| `scripts/run_shooting_encoder_finetune_ablation.py` | `python scripts/run_shooting_ablation.py encoder-finetune` | `src/temporal_vamp/commands/ablation_encoder_finetune.py` |
| `scripts/run_shooting_geometry_ablation.py` | `python scripts/run_shooting_ablation.py geometry` | `src/temporal_vamp/commands/ablation_geometry.py` |
| `scripts/run_shooting_multiscale_ablation.py` | `python scripts/run_shooting_ablation.py multiscale` | `src/temporal_vamp/commands/ablation_multiscale.py` |
| `scripts/run_shooting_short_horizon_ablation.py` | `python scripts/run_shooting_ablation.py short-horizon` | `src/temporal_vamp/commands/ablation_short_horizon.py` |
| `scripts/run_shooting_spatial_ablation.py` | `python scripts/run_shooting_ablation.py spatial` | `src/temporal_vamp/commands/ablation_spatial.py` |
| `scripts/run_shooting_temporal_pretraining_ablation.py` | `python scripts/run_shooting_ablation.py temporal-pretraining` | `src/temporal_vamp/commands/ablation_temporal_pretraining.py` |
| `scripts/run_spatiotemporal_objective_comparison.py` | `python experiments/spatiotemporal_20260905/run_spatiotemporal_objective_comparison.py` | `experiments/spatiotemporal_20260905/run_spatiotemporal_objective_comparison.py` |
| `scripts/train_geoframe_spatiotemporal.py` | `python scripts/train_geoframe_spatiotemporal.py` | `src/training_methods/spatiotemporal.py` |
| `scripts/analyze_shooting_branch_outcomes.py` | `python scripts/analyze_shooting_branch_outcomes.py` | `src/temporal_vamp/commands/analyze_shooting_branch_outcomes.py` |
| `scripts/evaluate_shooting_ablation0.py` | `python scripts/evaluate_shooting_ablation0.py` | `src/temporal_vamp/commands/evaluate_shooting_ablation0.py` |
| `scripts/render_shooting_dynamics_gifs.py` | `python scripts/render_shooting_dynamics_gifs.py` | `src/temporal_vamp/commands/render_shooting_dynamics_gifs.py` |
| `scripts/run_nested_committor.py` | `python scripts/run_nested_committor.py` | `src/temporal_vamp/commands/run_nested_committor.py` |
| `scripts/run_ordinary_temporal_embedding_cache.py` | `python scripts/run_ordinary_temporal_embedding_cache.py` | `src/temporal_vamp/commands/run_ordinary_temporal_embedding_cache.py` |
| `scripts/run_predictability_map.py` | `python scripts/run_predictability_map.py` | `src/temporal_vamp/commands/run_predictability_map.py` |
| `scripts/run_temporal_encoder_pretraining.py` | `python scripts/run_temporal_encoder_pretraining.py` | `src/temporal_vamp/commands/run_temporal_encoder_pretraining.py` |

## Validation

Validated in the `pointnet` environment: 61 targeted regression tests passed
across conversion/readers, campaign generation, plots, predictive analysis,
checkpoint handling and the multiscale extraction stage. Another 89 relocated
and grouped command/help checks passed, including direct execution of generated
job launchers outside the repository working directory. No full scientific
training or simulation campaign was rerun.

Conversion regressions exercise source retention, deferred deletion, repeated
conversion, retained-source and binary corruption, coordinate-archive checksum
failure, and existing dataset readers after source deletion. The multiscale
workflow retains its extraction stage before target preparation/training.

## Active-job correction (2026-09-05)

The subsequent live audit found that source controller `981345` failed because
its already-generated batch script referenced the removed independent-source
launcher. Two temporary forwarding launchers have now been restored at the
original paths, including the specialized 510/520 K launcher needed by pending
controller `981370`. They must remain until both source campaigns finish.
The low-temperature chain still needs recovery from source index 54; restoring
a path does not resubmit an already-failed controller. Login-node SSH access
was denied during the audit. The active jobs and completed trajectories were
not stopped or overwritten.
