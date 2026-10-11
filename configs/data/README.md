# Dataset loading

This directory contains only configurations consumed by training, evaluation, or
analysis when loading an existing dataset.

- `loaders/` defines dataset paths, sampling, neighborhood, normalization, and cache
  settings.
- `spatial_ni_enriched.json` adds 52,838 Ni crystal-interior/interface observations
  to the completed Ni spatial dataset, plus 1,638 held-out Ni observations. The
  new fitting total is 1,373,798; validation/test merge into one 248,081-row
  evaluation set. See [phase enrichment](../../docs/datasets/spatial_ni_enriched.md).
- `spatial_ni20.json` adds 264,192 Ni observations to the existing spatial fitting
  set, for 20% Ni in 1,320,960 rows, and prepares 198 bond-order/CNA/TDA targets.
  See the [dataset guide](../../docs/datasets/spatial_ni20.md) for held-out roles,
  normalization and the generated `consumer.json` fitting configuration.
- `spatial_active_cna_targets.json` pins 40 bond-order, 45 CNA and 113 TDA columns
  from the existing 1,056,768-row training transform; no validation scores select
  outputs. `spatial_active_cna_preparation.json` is the current preparation recipe,
  writing 198 columns to an external cache. The earlier `spatial_active_targets.json`
  and `spatial_active_preparation.json` record the 153-column optimization audit
  before CNA was added. See [target preparation](../../docs/target_preparation.md).
- `nuclei_birth.json` defines the current Al birth catalogue, exact 0.10/0.30 ps
  views and compatible completed-search discovery. Update it with
  `python -m src.data.nuclei_birth refresh`; see the
  [dataset guide](../../docs/datasets/nuclei_birth.md).
- `nuclei_birth_first_crystal_20261007.json` binds the versioned supplemental
  first-appearance labels to their completed-window producer. Publication copies
  existing labels to permanent dataset storage; it does not generate simulation
  data or alter nucleus definitions. See the
  [additional-label guide](../../docs/datasets/nuclei_birth_labels.md).
- Dataset generation and molecular simulation recipes live under
  `configs/simulation/`.
- `spatial_paired_relaxed_20261008.json` binds available paired Al relaxations
  and all planned Ni birth-window relaxations to the prior Small eSEN's 198
  targets and 12-epoch VICReg recipe. Dynamic originals are audit-only;
  user-confirmed already-relaxed static Mg/Al/Zr/Ta are also included. See
  [paired relaxed data](../../docs/datasets/spatial_paired_relaxed.md).

Do not add MD potentials, integrators, temperatures, replica schedules, or simulation
output controls here.

- `nuclei_birth_histories_20261007.json`: exact-0.1-ps birth-centred precursor reconstruction and visualization; CPU collection, one final dataset publication.
- `ni_four_births_20261007.json`: four confirmed births from completed Ni histories,
  with continuous movies and property plots; no catalogue publication. See the
  [visualization workflow](../../docs/ni_birth_visualizations.md).
