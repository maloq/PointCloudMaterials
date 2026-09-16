# Transfer selection for continuing encoder research

This selection covers the current coordinate/velocity MACE encoder, further training
on its prepared local atom groups, and the native encoder data-amount study.
Frozen-feature replacement-map training was [discarded](discarded_frozen_encoder_maps.md);
its outputs remain optional historical evidence, not an active training dependency.
It is a copy checklist, not a performed transfer or a tested cross-server resume.

## Code and research records

From `/home/infres/vmorozov/PointCloudMaterials/`, copy these maintained directories:

- `src/`
- `configs/`
- `scripts/`
- `tests/`
- `docs/`
- `environments/`
- `experiments/mace*/` (all matching encoder research record directories)

Also copy `AGENTS.md` and `requirements.txt`. Retain the **current working files**,
including uncommitted/untracked encoder code; a fresh clone alone misses current
work. Shared source/config directories are small, so keeping them intact avoids
breaking imports, Hydra recipes or metric-documentation contracts. `.git/` is
optional for execution but useful for development history. Do not copy a conda
environment directory; use the retained environment specifications on the new host.

## Required data and model artifacts

These are real source directories. Keep every checkpoint, hidden `.hydra/`
directory, normalization file, manifest, checksum sidecar and completed-epoch
resume state inside this selection.

| Source directory | Purpose | Approximate logical size |
| --- | --- | ---: |
| `/home/ids/vmorozov/training-cache/mace-velocity-all-20260915/` | Actual local coordinate/velocity groups, physical labels, tracked-pair metadata and source split; includes the converted legacy input | 967 MB |
| `/work/PERSO/vmorozov/analysis/mace_velocity/all-velocity-20260915/` | Current coordinate/velocity and coordinate-only checkpoints, teacher features, normalization, inventory and evaluation evidence | 446 MB |
| `/work/PERSO/vmorozov/analysis/mace_context_recovery/forecast-seed20260910-20260914/technical/train-dual_physics/` | Initialization checkpoint still required by the current velocity encoder loader; also contains the earlier physics-trained baseline | 244 MB |
| `output/mace_vicreg_relaxed_20260910/runs/mace_vicreg_relaxed_l40s_20260910_172945/default/anchor_vicreg__rep01/` relative to the checkout | Original VICReg checkpoint and its architecture config, still used by the loader | 116 MB |
| `output/pretrained_mace_spatiotemporal_20260906/` relative to the checkout | `mace_mp_0b2_small.model`, needed to construct the MACE architecture | 68 MB |

The listed model/data dependencies total approximately **1.8 GB**, excluding code
and the separate native data-amount study below. These are logical file sizes;
allocated disk usage can be smaller on this filesystem.

Dependency trace: `mace_velocity.inference.load_encoder` calls
`mace_velocity.train.backbone`, which reads the dual-physics checkpoint and
`mace_context.engine.load_model`; that loads the original VICReg checkpoint and
its `.hydra/config.yaml`, which names the pretrained `.model` file. Copying only
`coordinates_velocity/best.pt` is therefore insufficient with the current loader.
The prepared velocity cache contains the actual graph inputs and physical labels,
so the current cache-based training does not need whole source simulation campaigns.

## Add only for the corresponding next experiments

**Continuous local histories, curvature constraints, or sampling more centers/frames:**
the current velocity cache contains sampled current/previous pairs, not continuous
trajectories. Keep the trajectory paths explicitly listed under `records` in
`mace_velocity/all-velocity-20260915/technical/inventory.json`, with their binary
manifests, identity/timeline/box/position/velocity arrays and source provenance.
That exact inventory covers 1,114 selected binary trajectories, ten legacy NPZ
files and one converted input already inside the required velocity cache. The
trajectory sources span WORK and STORE. Do not substitute a whole `simulations/`
copy or include every sibling directory indiscriminately. Original campaign and
branch metadata are also needed if rerunning source discovery; discovery records
are retained in the selected velocity output. This additional raw data is **not**
included in the model/data estimate. All measured-velocity examples in this protocol are
Al; the static Zr snapshots do not supply velocities.

**Static Al/Zr spatial-cluster evaluation:** add
`/work/PERSO/vmorozov/datasets/Al/inherent_configurations_off/` and
`/work/PERSO/vmorozov/datasets/Zr/inherent_configurations/`.
For comparisons with the existing static/learned-distance results also keep
`/work/PERSO/vmorozov/analysis/mace_context_static/`,
`/work/PERSO/vmorozov/analysis/mace_context_clusters/`, and
`/work/PERSO/vmorozov/analysis/mace_local_state/`.
Static sample caches under
`/home/ids/vmorozov/training-cache/repository-datasets/static_al_context_interior_20260915/`
and `static_zr_context_interior_20260915/` preserve the existing sampled groups;
retain them for exact saved-sample comparisons, or regenerate them under an
explicitly new preparation identity.

**Earlier halo/center and dual-physics training protocols:** add
`/home/ids/vmorozov/training-cache/mace-context-pilot-20260914/`,
`/home/ids/vmorozov/training-cache/mace-meam/`, and the analysis directories
`mace_context/`, `mace_context_recovery/` (whole directory), and
`mace_encoder_diagnostics/` under `/work/PERSO/vmorozov/analysis/`.
These are not required for training the current velocity encoder from its prepared
cache; their own original preparation workflows may reference further raw inputs.

## Placement and resume details

Copy the **targets**, not just the repository's absolute `output/...` symlinks.
Preserve paths relative to the analysis/cache roots; keep the two checkout-local
model directories at their existing relative paths. Recreate the run links against
the destination analysis root. Use a destination-specific `machine.local.yaml`;
retain the old file as a record, not as the new machine's active configuration.
See [portable machine setup](portability.md).

The original velocity inference loader reads absolute paths from the embedded
checkpoint config without resolving them. Changing `machine.local.yaml` alone is
therefore **not sufficient for loading those existing checkpoints**. Preserve the original paths with compatible mounts
or links, or implement explicit, validated path remapping while keeping original
checkpoints and manifests byte unchanged. Do not bypass identity checks or rewrite
historical checkpoint/config files just to make a resume pass.

Take a final consistent copy after writers close.

## Optional discarded-map evidence (16 September)

For historical comparison, retain the already preserved outputs
`/work/PERSO/vmorozov/analysis/mace_local_smooth/`,
`/work/PERSO/vmorozov/analysis/mace_local_state/`, and
`/store/PERSO/vmorozov/analysis/mace_local_motion/`.
The smoothness feature cache is
`/home/ids/vmorozov/training-cache/mace-local-smooth-velocity-20260915/`.
These outputs and the smoothness feature cache are optional historical artifacts;
native encoder training from its prepared cache does not require them.
The [retirement record](discarded_frozen_encoder_maps.md) identifies the exact
archived source and recipes required to reproduce those historical workflows.

Retain `/home/ids/vmorozov/training-cache/mace-local-motion-sequences-20260916/`
when **repreparing the native data-amount study**: it supplies the selected source
plan and verified physical labels. Raw trajectories are also required to rebuild
local clouds. This dependency remains despite retirement of frozen-map training.
Training from the existing prepared native cache needs neither frozen embeddings
nor the original whole trajectories.

## Native encoder data-amount study (16 September)

Retain `/store/PERSO/vmorozov/analysis/mace_data_amount/independent-sources-20260916/`
and `/home/ids/vmorozov/training-cache/mace-native-data-amount-20260916/`, plus
`configs/analysis/mace_data_amount.json` and the new `mace_velocity/data_amount*.py`
implementation. This cache contains actual local positions, velocities and labels,
so the prepared study needs no whole trajectories for training. Its new checkpoint
loader needs only the resulting checkpoint and the original pretrained
`output/pretrained_mace_spatiotemporal_20260906/mace_mp_0b2_small.model`, not the
previous adapted encoder chain. Copy completed fits after their writers close.
See [execution and checkpoint details](mace_data_amount.md).
