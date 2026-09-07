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
For MLIP-initialized MACE with spatial/temporal VICReg, TDA and latent forecasts,
use `python -m src.training_methods.pretrained_mace --config CONFIG`; its distinct
objective and fresh fine-tuning protocol are documented in the
[pretrained-MACE experiment](../experiments/pretrained_mace_spatiotemporal_20260906/README.md).
The [80-point recipe](../experiments/pretrained_mace_spatiotemporal_80_20260906/README.md)
uses the same command with explicit compact context and online W&B settings.
The [0.1 ps continuation recipe](../experiments/pretrained_mace_80_dt01_cosine_20260906/README.md)
adds Al continuations, source-verified temporal loss selection, and per-step
warmup/cosine decay; it retains trained weights and scalers with a fresh optimizer.
For a serial MACE ablation queue after an existing training/analysis process,
use `python -m src.training_methods.pretrained_mace_queue --plan PLAN`.
It invokes the same MACE training and analysis commands inside the current
allocation; [protocol and plan](../experiments/pretrained_mace_ablations_20260907/README.md).
Matched frozen-encoder probes and incremental tables use
`python -m src.analysis.pretrained_mace_ablation --plan PLAN --run NAME`
or `--collect`.
Topology-target stability and frozen-decoder diagnostics use
`python -m src.analysis.topology_nuances --plan PLAN --stage stability|decoder`.
The subsequent existing-model analysis, larger-batch preflight, topology-aware
training and standard static analysis use
`python -m src.training_methods.topology_campaign --plan PLAN`;
[protocol](../experiments/mace_topology_nuances_20260907/README.md).
Encoder exports use `model_type: pretrained_mace_encoder` in the same full analysis
pipeline. The optional static `data.atomic_context` expands cached centers to a
verified physical receptive field without changing the sampling grid.
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
