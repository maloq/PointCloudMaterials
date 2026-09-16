# Configuration index

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

`mace_causal/runtime-benchmark.json` measures actual-graph FP32 throughput;
`mace_causal/h100-packed/` uses resident, packed batches for the longer matched
C/D/repeated-anchor cohort. H200 recipes use the same runtime.
