# Current research results

Start with the [retained research and checkpoint index](../docs/research_retention.md),
[experiment records](../experiments/README.md), or [searchable run registry](registry/index.html).
Older results, including the former `outputs/` root, are in [the STORE archive](../docs/archived_research.md).

| Folder | Purpose |
| --- | --- |
| `embedding_forecast/` | Current history, local-crystallization, spatial/mixture studies and continuation evidence |
| `embedding_forecast_20260911/` | Original pilot and enlarged fits, retained at exact provenance/resume paths |
| `mace/` | MACE galleries and comparison tables |
| `mace_vicreg_relaxed_20260910/`, `mace_vicreg_full_20260910/` | Required encoders and their training/analysis state |
| `mace_al_denoising_20260910/` | Required source/target preparation and denoising comparisons |
| `pretrained_mace_spatiotemporal_20260906/` | Only the pretrained MACE model required by current preparation |
| `synthetic_data`, `temporal_cache` | Compatibility links to preserved datasets/cache; no new runs here |
| `registry/`, `maintenance/` | Generated indexes and cleanup receipts |

New research outputs use `output/<question>/<run>/plots`, `tables`, and `technical`.
See [result and metric conventions](../docs/research_layout.md). Simulation production
uses the machine simulation root, documented in [simulations](../docs/simulations/README.md).

Refresh the registry with `python scripts/experiment_registry.py build`.
