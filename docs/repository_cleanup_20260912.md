# Repository cleanup — 2026-09-12

Branch: `cleanup/research-layout-20260912`. The first commit, `449f99c`, preserves all
in-progress tracked and untracked research present before cleanup. The original
`main` branch was not moved. Cleanup changes are committed separately after that
baseline. Do not reset away the baseline if you want the pre-existing work.

## Changes

- Moved 46 research Python modules into `src/research/<method>/` and moved the
  experiment-runner and temporal-dataset inspection implementations into their
  relevant `src/` packages. Script entry points remain small. No scientific protocol
  was merged or rewritten as part of the relocation.
- Moved 108 historical config/metadata files under dated experiment
  `technical/` directories. Root-level training/config YAML count fell from
  60 to 38. Hydra defaults, simulation definitions and recent queue
  paths remain intact. All moved configuration values were checked against the
  baseline after applying the recorded path replacements.
- New standard analysis, default Hydra training, forecast runs and result collection
  use readable plots/tables with machine files under `technical/`. Existing analysis
  and forecast layouts are recognized explicitly for reuse/resume.
- Reorganized 25 completed MACE galleries in place. Published figures are in `plots/`;
  scalar metrics and metric definitions in `tables/`. Existing scientific artifacts
  and paired test predictions stay at their recorded source locations. The seven-model,
  three-seed topology comparison was collected through the updated reader.
- Added frozen metric documents and source/doc fingerprints to maintained metric
  exports. Tests reject documentation drift. Interval bounds are included in the CSVs.
- Extended the existing registry command with `storage`, `clean` and `metrics-docs`.
  Registry JSON is now under `technical/`, with a readable experiment CSV and metric
  previews. The dashboard also indexes the legacy `outputs/` root.

The exact [move map](technical/repository_moves_20260912.json) is technical provenance.
Start with [the folder conventions](research_layout.md), [commands](../scripts/README.md),
[configuration index](../configs/README.md) or [registry guide](output_registry.md).

Two additional broken, unreferenced templates were retired:
`vicreg_geo_frame_multiscale.yaml` requires a missing Hydra base, and
`vicreg_vn_molecular_swav.yaml` contains conflicting Sinkhorn iteration counts (12
and 50). Both remain recoverable from `449f99c`. No numerical setting was guessed.
The training CLI now requests an explicit config instead of the missing default.

## Storage findings and retention

The initial allocated local footprint was 44.55 GiB: about 12.01 GiB NPY, 11.56 GiB
NPZ, 8.08 GiB PNG, 3.63 GiB PT and 2.57 GiB CKPT. The initial large-file report and
subsequent cleanup previews are in `output/maintenance/`. External symlink targets
are not part of those totals.

The verified quick-clean preview found 15 current-producer cache/sidecar pairs,
about 53.3 MiB, with retained results and unchanged local checkpoints. Large remaining
files are not assumed disposable: many are trajectories, paired comparison inputs,
external cache links or lack verified reconstruction prerequisites. No bulk cache,
checkpoint, simulation or dataset deletion was performed.

Local galleries reference their own technical plots with relative links to avoid
storing a second copy; separately published galleries retain independent portable
copies. Automatic inference-cache removal now archives exact reconstruction metadata.
Generic static/temporal templates still retain caches required by downstream
comparison workflows; current MACE templates retain their existing leaner settings.

## Constraints and validation

Slurm commands were unavailable locally and SSH access to the configured cluster host
was denied. Simulation launchers/controller paths and recent queue specs were therefore
preserved. The old MACE preparation command remains a small forwarding entry point for
submitted jobs. External batch files and frozen run source snapshots were not edited.

Validation includes 78 passing focused regression tests, successful composition of all
27 remaining root Hydra templates, exact forecast optimizer/sampler
resume in both layouts, a 25-job local dry-run of the maintained plan runner, all
108 moved configs' scientific values, relocated Markdown links, all links in the
25 galleries, metric-contract hashes and Python compilation. The topology collector
also succeeded on the existing seven-model/three-seed MACE results. Full GPU training
and live Slurm submission were not run for this organizational change.

## New-file classification

- Maintained implementation/tools: `src/research/`, the relocated `src/` commands,
  `src/experiment_runner/{artifacts,metric_docs,storage}.py`, focused tests and docs.
- Experiment records: the shooting-ablation README and relocated configs/metadata;
  existing scientific questions keep their own dated records.
- Disposable diagnostics: storage inventories, dry-run files and validation outputs
  under `output/maintenance/`. Retained cache provenance inside cleanup records must
  be kept if its original cache is subsequently removed.
