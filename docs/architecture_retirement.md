# Architecture retirement — 2026-09-30

The four proposed groups were removed from live training after explicit user
approval. This removes 45 Python files and 21 training/campaign recipes,
and trims retired classes, objectives and training handoffs from retained modules.
It does not delete checkpoints, predictions, metrics, data, caches or original
run-local source snapshots. Scientific outcomes and recorded input contracts
remain historical facts, including AP selection and temperature/time conditioning.

## Removed and retained

| Removed live treatment | Retained dependency |
| --- | --- |
| `structural_state.StructuralModel`, reconstruction/relaxed-teacher/distance training, preflight and queue; dependent MACE parameter-search fits | `GeometryEncoder`, graph/data and future-target producers, physical error/calibration helpers, saved-result reports and standalone GeoFrame fitting |
| Robust-onset and spatial-hierarchy predictors, trainers and AP-selection recipes; differentiable `smooth_ap` and ranking replay | Onset horizons, physical spacing and perturbation diagnostics used by current supervised/distance studies; historical AP-trained artifacts |
| `NeighborhoodModel`, v2 and multihorizon conditional JEPA predictors, their objectives, queues, profiling and dependent context-night encoder continuation | One shared snapshot encoder, normalization controls, paired-data preparation, Epi regularizers and current label-free MACE Epi consumers |
| v2 and regularization `Model` wrappers that built then replaced an inherited encoder | One direct encoder class replaces the old model modules; no replacement constructor, task head or compatibility alias remains |

Following the user's clarification, live model APIs assume newly trained models.
`src/models/encoders/neighborhood.py` constructs the shared encoder directly,
with explicit raw/layernorm export normalization and a default width of128.
Current callers import it directly. The original, v2 and regularization model
modules are deleted, rather than retained as import aliases. Its unused
`geometry_scales` checkpoint buffer and initialization copy are removed.
`src/training_methods/regularizers.py` owns the Epi helper used by current
training; the old JEPA objective modules are deleted. No old checkpoint
migration or legacy parameter-layout guarantee is added.

The current static export adapter accepts `mace_paired_epi_v1` checkpoints and
emits `neighborhood_snapshot_encoder` / `neighborhood_snapshot_static_v1`.
MACE Epi native evaluation records the actual shared encoder's source files.
Historical exports and checkpoints remain untouched on disk; this live API does
not promise that old mutable-source recipes can load or continue them.

`relaxed_encoder.queue` now exposes only data preparation, frozen extraction,
readouts and reporting. `execute --phase` accepts `extract` or `probe`; `fit` and
the `gpu` training stage are removed. Evaluation requires existing checkpoint
files and does not wait for a removed fitter to produce them.
`relaxed_encoder.expanded` and the mixed context-night queue are removed.
`relaxed_encoder.accelerated` has no training handoff.
`relaxed_encoder.recovery` retains CPU/CUDA cell repair and data rebuilding from
an existing restart plan, without training submission or training wait stages.
The shared context descriptors/forecasters and existing saved-result readers stay.

`configs/analysis/structural_state_future_assay.json` contains the existing
fixed-data future diagnostic settings used by GeoFrame, with no encoder training
arms, optimizer settings or Slurm queue. Its recorded cache, target spacing,
readout settings and seed are copied from the historical source. Paired-relaxation
analysis configs retain their original plan/run metadata because frozen readers
verify those identities. They are not encoder training recipes.

## Historical sources and contracts

Existing `technical/code` producers are authoritative. Historical reproduction, when needed, uses recorded frozen source paths and hashes;
checkpoint state dictionaries and scientific labels are not migrated. The live API
supports newly trained models. References to retired producer paths in
`configs/datasets.json` and prediction-context audit evidence remain historical
provenance, not live launch commands.

Before removal, the repository source/recipe/metric-definition state at
`fd77bdbeee482d2acb27a6d51dd667c669f6e6c4` was copied to a separate preservation bundle:

- Durable bundle: `${storage:training_storage}/maintenance/retired-architectures-20260930-fd77bdbe/`.
- Local audit: `output/maintenance/architecture-retirement/technical/`.
- `source-bundle.json`: 1587 file checksums; receipt SHA-256 `329465b015259189335d74d81b0faf2f0265d8ccb0b97c369cc956849465da71`.

This pre-retirement bundle is not relabeled as a historical run's original
producer. The audit inventoried 21 recipe output references: ten existing outputs
and seven run-local source trees containing the relevant model implementations.
It also checked retained paired-relaxation source trees. Some older neighborhood
JEPA/context-night paths already do not exist at their recipe locations. Their
absence is recorded rather than silently substituting the preservation bundle.
No unavailable producer is claimed to have been recovered by this deletion.

Ten retired numerical metric families retain their exact original file-hash maps
and descriptions in `docs/metrics/contracts.json` and checksummed retirement
records under `docs/metrics/retired/`. Repository validation checks those records
and historical descriptions. Live numerical export rejects a retired family;
reproduction requires its frozen producer, and rendering existing results uses
the publication-only workflow. Active readout/diagnostic contracts now list their
remaining dependencies. Existing exported contracts/definitions are unchanged;
a changed active contract requires a new numerical analysis revision.

## Verification

Local receipts live in the audit directory above. Checks use conda
`pointnet-torch214` and do not create online training runs or a test suite.

- Fresh raw/layernorm shared encoders produce finite `(batch,248)` packed outputs
  and finite learning gradients on an explicit e3nn CPU graph diagnostic. Their
  current states load strictly into fresh instances and reproduce outputs.
- The shared encoder has no conditional query, projector, decoder or obsolete
  geometry-scale state; MACE Epi constructs this encoder directly.
- The current Epi helper produces a finite score and finite gradients.
- Surviving analysis, MACE Epi, structured-context, supervised/distance evaluation,
  relaxation and hardware-benchmark consumers import successfully.
- Python syntax/import-reference, retained CLI and all123 metric contract checks
  cover the final checkout. Source bundles and recorded historical source trees
  were verified before consolidating the live model API.

No new scientific training, cuEq GPU forward pass or distributed fit was run.
