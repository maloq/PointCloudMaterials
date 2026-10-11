# Data and result storage

The user retired the original **150 native 0.75-ps Al trajectories** on October 7.
Deleting their 1,050 production array files released **99.624 GiB** across WORK
and STORE (94.524 GiB apparent file size). Prepared liquids, native melt checkpoints,
original manifests, logs, single-frame preparation QC and analysis results remain.
The consolidated 0.10/0.30-ps birth catalogue has no input-path overlap with this
removal. No jobs were modified. The three collection IDs remain registered as
retired provenance so descendant ancestry stays resolvable. See the
[scope and verification](storage/al075_removal_20261007/README.md) and
[deletion receipt](storage/al075_removal_20261007/receipt.json).
This is additional to the 230.231 GiB of earlier cleanup described below.
The user's subsequent instruction removed the 150 production `final.restart.bin`
files, releasing another **0.713 GiB**. All 150 melt checkpoints and prepared
liquid configurations were retained and verified unchanged; see the
[restart removal receipt](storage/al075_removal_20261007/final-restarts-receipt.json).

The first October 7 user-authorized STORE cleanup released **124.090 GiB** of file
allocation: **18.660 GiB** of obsolete search payloads/redundant event exports,
**35.829 GiB** of large binary payloads from the September 13 repository snapshot,
and **69.601 GiB** of byte-identical stopped-run copies in the recovery workspace.
Snapshot figures (including all 4,092 PNGs), tables, reports, source and provenance
remain. The consolidated 0.10/0.30-ps nuclei-birth catalogue and its 17,488 upstream
label files are protected; cleanup does not change its scientific fingerprint.
The four compatible search plans still pass the dataset's refresh-contract checks.
Catalogue integrity, all upstream SHA-256 checksums, and sample event reads from
both cadences passed after cleanup; see the
[dataset verification](storage/store_cleanup_20261007/birth-dataset-verification.json).
See [payload deletion receipt](storage/store_cleanup_20261007/payload-cleanup-receipt.json),
[exact removed paths](storage/store_cleanup_20261007/deleted.jsonl.gz),
[protected labels](storage/store_cleanup_20261007/protected-birth-labels.json.gz),
and [recovery deduplication receipt](storage/store_cleanup_20261007/recovery-cleanup-receipt.json).
Affected registry entries record the partial removal; historical completion
receipts alone no longer imply that every old payload remains available.

The stopped source0891/source0892 archives did not meet the requested >50%
crystallinity criterion: PTM on their last complete frames gave **1.26% at
257.3 ps** and **1.38% at 386.3 ps**, respectively. The existing monitoring records
also stayed below 50%; this is not an exhaustive reanalysis of every saved frame.
These figures describe the interrupted attempts, not their completed successors.
Both were already resumed from native checkpoints and completed on October 4:
source0891 reached **83.88% at 456 ps**; source0892 reached its **600 ps cap at
7.20%**, right-censored for the confirmed-half-crystal event. Their completed
0.10-ps position/velocity archives retain 4,561 and 6,001 frames, respectively.
After the user's follow-up instruction, all 28 selected completed-archive files
were verified against publication hashes, alongside array shapes/dtypes and
exact timelines. The two obsolete stopped-attempt text trajectories and their
workspace symlinks were then removed, releasing another **69.605 GiB**.
Combined simulation/search/snapshot cleanup released **193.695 GiB**. Completed arrays, all native
checkpoints, logs and provenance remain. See the
[successor verification](storage/store_cleanup_20261007/completed-successors-verification.json),
[follow-up removal receipt](storage/store_cleanup_20261007/stopped-payload-removal.json),
and [earlier stopped-frame audit](storage/store_cleanup_20261007/stopped-crystallinity.json).
The recovery workspace still serves active/prepared simulations and must not be
removed wholesale. No Slurm jobs were stopped or changed by this cleanup.

The earlier "other datasets" category included **36.536 GiB** in
`/store/PERSO/vmorozov/data`: geospatial MODIS, night-light, land/road features,
saved feature tables, boundary/ecoregion files and associated models/repositories.
At the user's further request on October 7, this entire geospatial root was
deleted: 5,813 regular files, two symlinks and 183 directories, with no symlink
target traversal. File identities were checked against the inventory before
removal and the root was verified absent afterward. This released **36.536 GiB**,
bringing the combined October 7 cleanup to **230.231 GiB**. Scientific simulation,
training and result directories outside this root were untouched. See the
[geospatial removal receipt](storage/geospatial_removal_20261007/receipt.json)
and [exact inventory](storage/geospatial_removal_20261007/inventory.json.gz).
Other STORE categories include relaxation products/indexes, simulation launch
records, retained research analyses and software; see the
[October 7 storage inventory](storage/usage_20261007/summary.json).

The October 3 user-authorized relaxation cleanup removed 1,707 historical CPU
full-cell archives lacking pinned LAMMPS executable provenance, plus 3,235
associated `clouds.npz` files across the pilot, expanded and large-test caches.
Approximately 4.621 GB net file allocation was released. Logs, metadata and
completion receipts are preserved in a verified compressed provenance archive.
All 13,743 recipe-bound index entries remain available; the 4,942 removed
entries retain their provenance and now report unavailable. Original MD,
checkpoints, predictions and metrics are unchanged. See the
[receipt](storage/relaxation_unverified_removal_20261003/receipt.json) and
[exact deletion inventory](storage/relaxation_unverified_removal_20261003/deleted_files.csv).

The user-authorized Ta cleanup on 2026-10-03 deleted five earlier large-cell
trajectories and 24 shooting futures across seven folders. The original Ta
snapshot archive and earlier million-atom baseline on WORK are preserved.
Net file allocation released: 579.218 GB; 282 report/provenance files were
preserved and protected file identities were verified unchanged. See
[the receipt](storage/ta_simulation_removal_20261003/receipt.json) and
[preservation index](storage/ta_simulation_removal_20261003/preserved.json).

The October 2 Al text conversion is complete: all 1,065 planned dumps were
verified and removed, releasing a net 554.256 GB of file allocation. Native
restarts remain intact; [final summary](storage/al_float16_20261002/summary.json).
The October 3 [remaining text inventory](storage/text_simulations_20261003/files.csv)
records 372 uncompressed dumps/snapshots, about 2.795 GB allocated, including
original input snapshots, diagnostics and force-only outputs. The rerun recovery
uses a temporary STORE simulation workspace after SCRATCH quota exhaustion;
see [the recovery](simulations/al_main_half_stop_20261001/README.md#october-3-quota-recovery).

The user-authorized experiment cleanup on 2026-10-02 removed the 13 reviewed
result folders, including the remaining STORE embedding-forecast collections.
Saved 48,774 report/table/plot/provenance files under
[the results archive](storage/experiment_removal_20261002/results/), with original
paths and SHA-256 checksums in [the preservation index](storage/experiment_removal_20261002/preserved.json).
Net file allocation released after preserving results: 147.117 GB.
Checkpoints and binary arrays in those folders were deleted; historical references
to them no longer imply availability. See [the receipt](storage/experiment_removal_20261002/receipt.json).


The reviewed WORK `analysis/embedding_forecast` collection was explicitly deleted
by the user on 2026-10-02 after reviewing its contents, including saved models and
evaluation arrays: 29.458 GB allocated, 7,723 files. Historical experiment prose
remains, but its links into this deleted collection no longer represent available
artifacts. See [the removal receipt](storage/embedding_forecast_removal_20261002/receipt.json).

The user-authorized six-collection cache cleanup on 2026-10-02 removed 131,207
NPY/NPZ payload files, releasing 289.552 GB of allocated blocks after counting
hard links once: STORE 96.883 GB, SCRATCH 52.034 GB and IDS 140.635 GB.
Twenty registry entries are retired. Provenance, lock files, the JEPA reservoir
checkpoint and all six managed encoder-feature entries remain. See the
[receipt](storage/cache_cleanup_20261002/receipt.json) and
[file inventory](storage/cache_cleanup_20261002/inventory.csv).

Al text-to-float16 archival started on 2026-10-02 at the user's request. The
[main plan](storage/al_float16_20261002/plan.json) and
[historical-archive plan](storage/al_float16_20261002/additional/plan.json)
enumerate sources; `progress.json` and `actions.jsonl` beside each plan distinguish
verified/deleted originals from failures or active-job deferrals. Each replacement
is named `SOURCE_FILENAME.float16/`, with repository trajectory arrays and a
`conversion.json` checksum/quantization receipt. Native restarts are retained.
SCRATCH reached its quota during the first attempted export; those replacements
are instead published below
`/store/PERSO/vmorozov/trajectory-archives/al-float16-20261002/`, preserving the
relative source path. The original is deleted only after the STORE copy verifies.
Incomplete terminal-frame bytes, when present, are preserved separately as
`incomplete_final_frame.bin`; they are not counted as complete observations.
Scientific completion states and original campaign recipes are not rewritten.

Smoke-output cleanup on 2026-10-02 removed 65 generated directories occupying
8.608 GB allocated (6.854 GB on WORK), at the user's request. Ten dataset
identities are retained as retired entries. See the
[deletion receipt](storage/smoke_cleanup_20261002/receipt.json) and
[deleted-file inventory](storage/smoke_cleanup_20261002/deleted_files.csv).
The earlier storage audit is a pre-cleanup snapshot.

Start with [the dataset registry](../DATASETS.md) to find datasets, generating
potentials, current array schemas and provenance across these storage roles.

| Contents | Location on this cluster |
| --- | --- |
| Existing datasets and simulation inputs | `/work/PERSO/vmorozov/{datasets,simulations}` |
| Analysis products | `/work/PERSO/vmorozov/analysis` |
| Training and dataset caches | `/home/ids/vmorozov/training-cache` |
| New simulation runs | `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations` |
| Published simulations and archives | `/store/PERSO/vmorozov` |
| Current research reports and run links | Repository `experiments/` and `output/` |

Machine-specific paths belong in ignored `machine.local.yaml`. Stable dataset IDs
and dependencies are in [configs/datasets.json](../configs/datasets.json).
`python scripts/project.py paths` shows effective roots; `datasets` resolves IDs.
See [portability](portability.md) for setup, verified copies, bundles and publication.

The active [multimaterial RH2 fit](rich_multimaterial_encoder.md) writes to
`${storage:training_storage}/liquid_predictability/mm-rd-mace256-c3-l3-residual-head-20260929`
on STORE after WORK exhausted its quota on 29 September. Its original frozen
output setting is redirected by a specific alias in `machine.local.yaml`, without
changing the global analysis root or scientific identity. Checkpoints and all
other run artifacts were checksum-verified before attempted source cleanup;
`technical/storage-migration.json` records hashes. WORK refused some deletions,
so residual duplicate files there are not the active run. Use the resolved path.

SCRATCH has a 30-day inactivity purge. Publish completed simulations and preserve
stopped failures/restarts on STORE before that deadline. Conversion must verify
checksums and float16 position quantization before removing larger exports;
[trajectory conversion](trajectory_conversion.md) describes the maintained commands.

[Old research is archived on STORE](archived_research.md). The
[September 11–13 retention review](research_retention.md) identifies the live research,
checkpoints and older inputs that must remain available. A large file can be a
necessary paired test array or exact-resume checkpoint; do not classify by size alone.

[Research cache sizes, 26 September 2026](storage/cache_inventory_20260926/README.md)
lists allocated GB by cache family and physical location, including archives and
derived relaxed structures. The [cache-use audit and retirement options](storage/cache_usage_20260926/README.md)
separate recent references, active dependencies and old forecasting payloads.
The [completed September 27 cache cleanup](storage/cache_cleanup_20260927/README.md)
removed 683.461 GB of those forecasting payloads, retaining rebuild provenance.
[Simulation cleanup](simulations/cleanup_20260926/README.md)
records the separately authorized deletion of coarse Al and stopped failed MD payloads.

For disposable inference caches:

```bash
python scripts/experiment_registry.py storage
python scripts/experiment_registry.py clean --root output/QUESTION/RUN
python scripts/experiment_registry.py clean --root output/QUESTION/RUN --apply --inactive
```

Review the preview and establish that readers/writers have stopped before applying.
Keep reconstruction receipts. [The result layout](research_layout.md) separates
plots/tables from technical artifacts and documents metric exports.

Operational history: [initial migration](storage/migration/README.md),
[portable environment and storage validation](storage/portability_validation/README.md),
and [dataset inventory method](storage/dataset_inventory/README.md).

## Unified relaxation index

Use dataset ID `relaxation-index-v1`, stored at
`${storage:archive}/relaxation-index/v1`. This is the shared lookup for every
experiment requiring relaxed observations. It references existing payloads;
it does not create another sample split or change labels, ancestry, or cadence.
[Scan configuration](../configs/relaxation_index.json) lists the seven
retained Al archive families and their separately saved neighborhood caches.

In `pointnet-torch214`:

```bash
python scripts/project.py relaxations scan
python scripts/project.py relaxations status
python scripts/project.py relaxations plan --requests requests.json --output missing.json
```

`scan` is repeatable and imports completed archives only. Run it after older
frozen jobs publish more data; future producers register newly completed data
automatically. `status` refreshes `inventory.json`, `inventory.csv`, and
`recipes.json` in the shared index. Files referenced by an entry must exist;
checksums are verified before reuse. Changed payloads fail with the file path.
Failed attempts, incomplete writes and benchmark folders are excluded.

Before the subsequent user-authorized cleanup, the October 3 backfill found 9,007 full-cell archives and 9,678 saved
neighborhood files, covering 8,637 source frames from 652 trajectory manifests.
The neighborhood files contain 546,793 rows, including repeated rows across
releases. Of 18,685 product entries, 13,743 have pinned recipe provenance and
4,942 have an unverified historical engine. The latter were subsequently deleted
at user request and remain listed as unavailable provenance records. Current
retained holdings are 7,300 full-cell archives and 6,443 neighborhood files.
These are archive counts, not independent sample counts.

### Matching and precision

A request binds the source manifest checksum, frame and actual timestep to
the potential contents and pair commands, mass, minimizer, force tolerance,
iteration/evaluation limits, FIRE timestep, native restart seed when used,
LAMMPS executable hash, MPI ranks and backend flags. Timeout changes alone do
not invalidate a converged result. CPU and GPU minima remain distinct recipes.
Historical engine hashes come from frozen receipts or accelerator metadata,
never from guessing that today's executable is the historical one.

Three products are deliberately distinct:

- `saved_float32_patches`: centered offsets extracted before global conversion.
  Reuse requires the exact center atom and ordered original neighbor IDs. Row
  order is rebound to the requesting experiment; patches from different minima
  are not combined into one request.
- `archived_float16_cell`: the verified complete cell, usable when a workflow
  explicitly declares global float16 input. It cannot recover lost float32
  local precision for previously unsaved centers.
- `native_restart`: new minimizations retain one native LAMMPS restart in the
  shared store. It preserves restart precision; coordinate archives still use
  the standard verified float16 converter. New neighborhoods can be extracted
  by reading this restart, without MD steps or another minimization. Protect
  these checkpoints as source data, not disposable encoder feature caches.

### Experiment API

`src.data.relaxation_index.ensure_patches` is the common compute-only-missing
entry point. It accepts the absolute-coordinate trajectory interface used by
`relax_frame`, a frame, existing relaxation settings, center IDs `[C]`, and
original ordered neighbor IDs `[C,K]` (center first). Shooting binaries store
box-relative positions: use `src.data.relaxed_targets.worker.AbsolutePositions`
when constructing that absolute-coordinate interface.

```python
from src.data.relaxation_index import ensure_patches

positions, record = ensure_patches(
    absolute_trajectory, frame, settings, center_atom_ids, neighbor_atom_ids
)
# positions: centered float32 [C,K,3], in the requested order
# record['id']: provenance to save with this experiment
```

For lookup without computation, use `patch_lookup(request, settings, centers,
neighbors)` or `load_cell(request, settings,
precision='archived_float16_cell')`; both return `None` on a missing match.
`candidates(request)` also exposes incompatible or historically unverified
records for an explicit protocol review. Do not silently widen matching.

The planner's JSON has `settings` (the same dictionary passed to `relax_frame`)
and a `requests` list. Each request supplies `source_manifest_sha256`,
`source_frame`, `source_timestep`, and `product`. Patch requests additionally
provide `center_atom_ids` and `neighbor_atom_ids`. The output retains every
request and assigns `reuse`, `extract` (native restart, no minimization), or
`compute`; its `missing` list contains only the last category. A real example
is saved under the index's `technical/real-archive-verification-20261003/`.

Existing full-cell producers share the `relax_frame` native-checkpoint lookup
by default; performance benchmarks explicitly use `reuse=False`. Birth and
relaxed-TDA producers also query historical float32 neighborhoods; the liquid
archival-replay workflow queries historical float16 cells. Paired encoder and
TDA producers register their separate local caches. File locks ensure that
simultaneous identical full-cell requests perform one minimization. Existing
frozen queues retain their recorded code and protocols.

Verification receipts are under the index's `technical/`: a real 70,304-atom
archive lookup and missing-center plan passed, as did a small native LAMMPS
round trip, exact float32 patch comparison, concurrent requests, recipe
mismatch rejection and corruption detection. The birth producer also reused
five real neighborhoods with zero minimizations (the original quench receipt
recorded 296 seconds). The diagnostic data are isolated
from the scientific index; no scientific training or new campaign was launched.
The initial migration briefly placed caches on WORK; the later validation record
moved them back to IDS. The table above is the current placement.
