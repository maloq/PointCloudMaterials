# Configuration index

Keep active recipes and their dependencies here. Simulation recipes belong only in
`simulation/`; keep analysis templates in `analysis/`. Run-specific research plans
stay with their scientific record in `experiments/`.

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

- `analysis/`: all seven analysis templates are preserved. Pass the intended
  checkpoint explicitly; historical checkpoint defaults have not been rewritten.
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
