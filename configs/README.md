# Configuration index

[`analysis/gatr_equivariant.json`](analysis/gatr_equivariant.json) pins the
frozen GATr directional trajectory and spatial-order audit on node07/A100;
see [protocol and results](../experiments/gatr_equivariant_20260918/README.md).

[`analysis/trajectory_stability.json`](analysis/trajectory_stability.json) pins
the selected MACE/GATr checkpoints and matched Al trajectory sampling for the
[temporal stability comparison](../experiments/trajectory_stability_20260918/README.md).

Keep active recipes and their dependencies here. Simulation recipes belong only in
`simulation/`; keep analysis templates in `analysis/`. Run-specific research plans
stay with their scientific record in `experiments/`.

Frozen-feature replacement-embedding recipes were [discarded](../docs/discarded_frozen_encoder_maps.md).
Exact historical copies remain with the dated scientific records; they are no
longer active recipes. Forecasting recipes are unchanged.

## Training and encoder construction

| Config name (without `.yaml`) | Why it stays |
| --- | --- |
| `vicreg_mace_full` | Current full MACE VICReg study. |
| `vicreg_mace_full_cpu` | CPU execution variant of the same full study. |
| `vicreg_mace_relaxed` | Current relaxed/history MACE VICReg study. |
| `vicreg_pretrained_mace_geometry` | Required parent of `vicreg_mace_full`. |
| `vicreg_pretrained_mace_geometry_tda` | Required parent of `vicreg_mace_relaxed`. |
| `mace_denoising_encoder`, `mace_temporal_encoder` | Maintained encoder construction/export interfaces; [input contract](../docs/mace_temporal_encoder.md). |
| `vicreg_vn_molecular_multi` | Explicitly retained by request, with `data/loaders/static_multi_material.yaml`. |

Training requires an explicit `--config-name NAME`. Choose dataset, seed, checkpoint
and output location through the existing config/CLI. Do not copy a runner.

## Supporting configs

- `analysis/structural_gatr_v6_static.json`, `analysis/static_structural_gatr_v6_al.yaml`
  and `data/loaders/static_al_structural_gatr_v6.yaml`: selected Al-only v6
  GATr–VICReg checkpoint with native precision, on the unchanged static Al grid.
- `analysis/structural_mace_v6_static.json`, `analysis/static_structural_mace_v6_al.yaml`
  and `data/loaders/static_al_structural_mace_v6.yaml`: selected Al-only v6
  MACE–VICReg checkpoint with native packed graphs and precision on the same grid.

- `shared_pretraining/geometry_fp32_2x/{mace,gatr}_vicreg_structural.json`:
  enlarged snapshot encoders with FP32 geometry and scalar BF16 computation;
  prepared, unsubmitted 12-epoch runs. See [architecture and validation](../docs/shared_pretraining_geometry_fp32_2x.md).

- `shared_pretraining/restart_b1024_lr002_bf16/campaign.json`: fresh MACE/GATr
  VICReg structural runs after the failure audit, batch 1,024, peak LR 0.002,
  BF16, normalized readouts and learning-health gates. Existing v3 jobs use their frozen code;
  [launch and precision measurements](../docs/shared_pretraining_restart_20260918.md).

- `shared_pretraining/campaign.json`: approved local 12-epoch structural,
  initialized causal and frozen-evaluation campaign, batch 512.
  `shared_pretraining/h200_batch1024/campaign.json`: separate one-H200 serial
  queue with batch 1,024 for both encoder-training stages; see the
  [H200 task](../docs/h200_shared_pretraining_task_20260918.md).

- `analysis/structural_gatr_static.json`, `analysis/static_structural_gatr_al.yaml`
  and `data/loaders/static_al_structural_gatr.yaml`: selected RTX6000 structural
  GATr–VICReg export, verification and full standard Al static analysis;
  [input contract and commands](../docs/structural_static_analysis.md).

- `local_predictability/two_gpu_16h.json`: planning-only one-seed H100/H200
  predictability queue; requires new assay/batched-training adapters before
  execution. [Protocol](../experiments/local_predictability_20260917/README.md)
  and [handoff](../docs/local_predictability_16h.md).

- `benchmarks/hardware.json`: dataset-free storage, LAMMPS CPU and GPU training
  workloads; [commands and comparison protocol](../docs/hardware_benchmark.md).

- `analysis/`: all seven analysis templates are preserved. Pass the intended
  checkpoint explicitly; historical checkpoint defaults have not been rewritten.
  `mace_encoder_diagnostics.json` and `mace_encoder_readout.json` configure the
  frozen forecast-encoder stability and source-held-out TDA assay.
  `mace_tda_initialization.json` adds frozen MLIP and random-weight controls to the
  same 256D encoder assay, including validation-selected ridge regularization.
  `mace_tda_direct.json` compares these frozen embeddings directly with TDA
  distances and neighbors, without fitting a readout or feature scaling.
  `mace_tda_ridge_audit.json` retests the six single-frame VICReg/TDA checkpoints
  with independent ridge calculations and supervision checks.
- `embedding_forecast/`: staged GPU validation and replacement allocation recipes
  for the active history/spatial/mixture study, plus matched interim-analysis cohorts;
  [execution evidence](../docs/forecast_performance.md) and [scientific findings](../experiments/forecast_spatial_mixture_20260913/INTERIM_RESULTS.md).
  `trajectory-visuals.json` links measured structures to observed embeddings and
  sampled future paths from completed models, using shared UMAP coordinates and
  interactive point-cloud time sliders. It can reuse the retained raw extraction.
  `spatial-context/` contains 32/128/512-neighbor cache recipes, a matched 3/12 ps
  history sweep, paired analysis plans and detached allocation queues.
  Its `report-20260914.json` summarizes completed cohorts without rerunning inference.
  `spatial-attention/` retains individual-neighbor geometry, learned attention fits
  at 3/12 ps, matched physical comparisons, and the active allocation recipe.
- `data/loaders/`: `static_multi_material`, `static_al_80` and
  `static_al_crystallization_step187800`, required by retained training/analysis.
- `simulation/`: current Al/Ti/Ta recipes, potential files and checkpoint producer
  compatibility records; see [simulation configs](simulation/README.md).
- `machines/`: portable profile examples; machine-specific settings belong in
  ignored `machine.local.yaml`.
- `datasets.json`: stable dataset and simulation locations.
- `experiment_registry.json`: experiment registry settings, including external roots.

See [portability](../docs/portability.md) and the
[result layout](../docs/research_layout.md) for new runs.

## Retired recipes (September 13, 2026)

The cleanup reduced this tree from 129 to 31 files (123 to 26 YAML/JSON configs).
All seven analysis configs and all eight retained root YAMLs are byte-identical to
the pre-cleanup copy; Hydra parent compositions remain intact.

The complete [old config tree](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs/) and
[SHA-256 verification](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs.verification.json) are on STORE.
The [retirement receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs.retirement.json) lists all 98 removed files.
Older GeoFrame/VN/SwAV/VAMP recipes, sweep plans and Al simulation variants are
available there; [archive restoration notes](../docs/archived_research.md) explain
where their matching repository and results live.

Restore historical recipes with their dependency tree and original code into a
separate checkout. For a restored Hydra config tree, pass
`--config-dir /absolute/path/to/restored/configs --config-name NAME`. The old
temporal SSL and descriptor entry points no longer select a retired recipe by
default. The FactorVAE and historical GeoFrame objective queues require
`--config-dir`; the spatiotemporal trainer accepts it too. The optimized Al shell
launcher requires `CAMPAIGN_CONFIG` in addition to `PYTHON`.

Fifteen old simulation configs needed by regression tests moved to
`tests/fixtures/simulation/`, with internal fixture paths updated. Two GeoFrame
regression fixtures preserve their fully composed training settings. These fixtures
are test inputs, not a second set of maintained run recipes.

The [MACE context pilot](analysis/mace_context.json) selects complete message context, smooth inner pooling and tracked-center readouts with matched VICReg continuation.
The [context recovery recipe](analysis/mace_context_recovery.json) compares cached
linear/nonlinear readouts and inner/center fusion, then a shared encoder trained
with or without physical-target gradients. Its detached node57 plans live in
`mace-context/recovery-gpu0.json` and `mace-context/recovery-gpu1.json`.

The joint checkpoint's [static export/verification recipe](analysis/mace_context_static.json)
and [Al](analysis/static_mace_context_al.yaml) / [Zr](analysis/static_mace_context_zr.yaml)
analysis recipes use complete message context with the standard static pipeline.
See [the protocol](../docs/mace_context_static.md) for Zr geometry scaling and
interior sampling. Detached node51 plans are in `mace-context/static-node51-gpu*.json`.

[Liquid-cluster diagnosis](analysis/mace_context_clusters.json) compares saved
joint features, inner/center/projector ablations and archived GeoFrame V2 labels
on matched static Al centers. It runs no encoder training.
# Local phase-space encoder

[`mace_causal/pilot.json`](mace_causal/pilot.json) selects the distinct causal native
tensor MACE architecture, fixed physical future targets, and matched A–E/history
controls. See [workflow](../docs/mace_causal.md).

`analysis/mace_velocity.json` trains the coordinate/velocity MACE extension and
its matched coordinates-only control. See [the protocol](../experiments/mace_velocity_20260915/README.md)
and [run instructions](../docs/mace_velocity.md).

The native end-to-end MACE data-size pilot uses [analysis/mace_data_amount.json](analysis/mace_data_amount.json); see [protocol](../experiments/mace_data_amount_20260916/README.md).

The two `mace_causal/pilot-seed*.json` recipes repeat the causal pilot budget with
independent initializations. `mace_causal/comparison.json` pairs completed fits and
probes on held-out physical targets with whole-source uncertainty intervals.

The three `mace_causal/pilot-gaussian-seed*.json` recipes repeat D with a diagonal
Gaussian future head under the same update budget, retaining MSE/NLL/coverage.

`mace_causal/h200/` defines the separate longer-budget C/D/repeated-anchor study
at tensor widths 16 and 32. See [H200 handoff](../docs/mace_causal_h200.md).

`predictive_memory/optimization/` contains the current partial-observation
H100 follow-up: 12,000 updates at present-loss weights 0.05 and 1.0, each with
two seeds. Use the existing immutable predictive-memory cache and H=0,12,48 plus
repeated-anchor controls. See [workflow](../docs/predictive_memory.md).

`mace_causal/runtime-benchmark.json` measures actual-graph FP32 throughput;
`mace_causal/h100-packed/` uses resident, packed batches for the longer matched
C/D/repeated-anchor cohort. H200 recipes use the same runtime.

`analysis/memory_research_report.json` selects completed causal/memory cohorts,
reported H200 summaries and simulation-status inputs for a dated cross-study
evidence snapshot. Run `python -m src.research.memory_report --config CONFIG
--output NEW_OUTPUT`; this analysis does not launch training or simulation.

`local_predictability/rtx6000_observability.json` defines the single-seed packet
recognition/onset diagnostics on the frozen broad release, with a fixed training
deadline. See [protocol](../experiments/local_predictability_20260917/OBSERVABILITY.md).

`local_predictability/rtx6000_native_onset.json` offloads the existing repeated-frame
continuation, keeping its parent and scientific identity. The adjacent
`rtx6000_raw_observability.json` and `rtx6000_native_readouts.json` recipes fill the
raw-state and frozen-predictor diagnostics, respectively, within the same allocation.
Their remaining MACE work uses CuEq and bounded parallel input lookahead; current
core optimizers retain their original backend.

`predictive_memory/batched.json` is the fresh-run optimized memory recipe:
cuEquivariance spatial kernels (`encoder.mace_backend: cueq`),
effective batch 8, microbatch 2, evaluation batch 4, bounded train-observation GPU
residency. Use the existing training CLI; see [workflow](../docs/predictive_memory.md).

Current compiled VICReg restart: [`shared_pretraining/vicreg_compiled_repair`](shared_pretraining/vicreg_compiled_repair), with the [operational recipe](../docs/shared_pretraining_compiled_repair_20260918.md).

`shared_pretraining/al_stable/` is the active Al-only MACE/GATr VICReg recipe.
It supersedes the stopped v5 compiled repair; see [scope and normalization](../docs/shared_pretraining_al_stability_20260918.md).

`shared_pretraining/broad_full_tda/` expands shooting to 75,000 anchors, restores
five-metal data with complete instantaneous TDA, and continues the Al-trained
GATr for three epochs. See [preparation and GPU dependencies](../docs/shared_pretraining_broad_full_tda_20260918.md).

`shared_pretraining/gatr_mixed_triplets/` starts fresh dynamic-only GATr on four
metals with mixed batches of 2,048 and a three-snapshot backtracking penalty.
It supersedes the failed broad continuation. See [execution and normalization](../docs/shared_pretraining_mixed_triplets_20260918.md).

`shared_pretraining/gatr_temporal_backtracking/` continues the mixed GATr at
update 250 with a training-calibrated fixed curvature weight and backtracking
on temporal updates only. It preserves optimizer/schedule progress and uses
compact W&B logging. See [transition protocol](../docs/shared_pretraining_temporal_backtracking_20260918.md).
