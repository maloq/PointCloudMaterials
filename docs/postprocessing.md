# Post-simulation analysis

All maintained nucleus-analysis implementations now live in `src/postprocessing`.
Use one command entry point:

```bash
conda activate pointnet-torch214
python -m src.postprocessing --help
python -m src.postprocessing search --help
```

The package owns trajectory access for analysis, full-cell descriptors, nucleus
lineages, birth catalogues and label supplements, precursor histories, plots,
movies and portable galleries. MD integration and simulation scheduling remain
in `src/simulation`; prediction training remains in `src/research`. Those callers
import this package rather than carrying their own analysis implementations.

For the October 8 completed Al backlog, `al-gpu-search prepare|submit` with
`configs/analysis/al_nuclei_gpu_additions_20261008.json` binds the two verified
inventory plans (45 main and 33 birth-focused sources). Two GPU workers run the
audited resident/Voronoi extractor; four CPU lanes track the unchanged v3
lineages. Successful collection triggers catalogue publication automatically.
Failures block dependents. Source timelines, roles, ancestry and different
observation starts remain explicit. The separate seven-source completed search
is registered for incorporation without extracting it again. This queue adds
only exact 0.10-ps views; native-only 0.30-ps source/event identities are checked
unchanged. No new simulations, relaxations or prediction fits are submitted.

## Layout and extension points

| Component | Responsibility |
|---|---|
| `io.py` | Checksum-bound manifests, bounded memory-map cache, periodic geometry |
| `descriptors/` | Full-cell PTM, coherent-bond components, local orientational order |
| `descriptors/rich.py`, `descriptors/alpha3d.*`, `targets.py`, `target_benchmark.py` | Compiled bond order, native batched alpha persistence, spatial-target preparation and timing/numerical audits |
| `spatial_dataset.py` | Source-parallel Ni extension of the spatial fitting set, compact targets and packed training arrays; see [20% Ni data](datasets/spatial_ni20.md) |
| `spatial_enrichment.py` | Explicit Ni crystal/interfacial sampling with periodic local phase checks, duplicate exclusion and the shared spatial target/packing workflow |
| `relaxed_spatial.py`, `relaxed_spatial_inventory.py` | Paired relaxed observations, source/frame/center deduplication, the existing 198 BO/CNA/TDA targets, and dependent Small eSEN training; see [paired relaxed collection](datasets/spatial_paired_relaxed.md) |
| `descriptors/gpu/`, `gpu_benchmark.py`, `gpu_extract.py` | CUDA PTM/descriptors, numerical audits and bounded streaming to separate caches |
| `extraction.py` | Locked, resumable chunks and one reader for component labels and raw PTM |
| `cache.py` | Verified, resumable copies into the selective detector layout |
| `sources.py`, `inventory.py` | Source provenance, exact native timelines, production eligibility |
| `lineage.py`, `search.py` | Current family tracking, physical-time fates, CPU queue and collection |
| `datasets/` | Stable birth catalogue, cadence policy, observation labels and published histories |
| `histories/common.py` | Shared precursor linking, periodic crops and shape measurements |
| `histories/frames.py` | Shared frame preparation and atomic per-event frame checkpoints |
| `protocols/ni_queue.py` | Bounded chunk scheduling and completed-source admission during MD |
| `histories/al.py`, `histories/ni.py` | Explicit cohort selection and observation-window protocols |
| `histories/publish.py` | Detached history queue and durable dataset publication |
| `visualization/` | Shared viewer, property plots, GIFs, crops and portable exports |
| `protocols/` | Material/study adapters and distinct historical scientific protocols |

To add a material, supply its verified source manifest and explicit physical
cutoffs in an adapter. Reuse extraction, tracking and history kernels. Ni already
does this; it remains a separate cohort and is not inserted into the Al dataset.
To add a descriptor, implement its calculation in `descriptors/`, declare which
extraction products require it, and bind it into the relevant metric contract.
The opt-in [GPU extraction workflow](postprocessing_gpu.md) adds CUDA PTM and
bounded reading/compression overlap (`gpu-extract --config CONFIG`). Its numerical audit compares
numerical outputs and synchronized timings before considering a production backend.
To add a view, consume verified saved packets; do not recompute or relabel births
inside the renderer. Register user-facing commands in `__main__.py`.

Historical threshold-episode tracking is separate from current descendant-family
tracking. Keeping their definitions explicit avoids silently turning old fates
into current labels. The removed 0.75-ps Al trajectories remain retired.

## Workflow

Use fresh output locations for a new scientific execution. Existing submitted
jobs and resumption use their recorded frozen code, configurations and receipts.

```bash
# Verify production provenance and freeze the descriptor input population.
python -m src.postprocessing inventory --config EXTRACTION_CONFIG

# The current search queue can perform pending extraction before tracking.
python -m src.postprocessing search prepare --config SEARCH_CONFIG
python -m src.postprocessing search submit --config SEARCH_CONFIG

# Publish completed compatible searches under the stable dataset IDs.
python -m src.postprocessing dataset refresh
python -m src.postprocessing dataset verify

# Separate history annotations; the existing birth labels are retained.
python -m src.postprocessing histories prepare --config HISTORY_CONFIG
python -m src.postprocessing publish launch --config HISTORY_CONFIG
python -m src.postprocessing view render --config HISTORY_CONFIG --event EVENT_ID
```

`descriptors prepare|extract|progress` exposes structural preparation and worker
stages. The `search` queue owns the dependency between extraction and current
lineage tracking. `histories assist` cooperatively claims unfinished events with
nonblocking locks. `labels` publishes observation-label supplements; `export`
packages an existing gallery. `ni-search`, `ni-histories` and `ni-preview` retain
their declared Ni selections. Every command has its own `--help`.

For an explicitly selected completed Al GPU shooting cohort, `inventory` accepts
`production_inventory.source_ids` and `campaign_manifest`. It verifies the
original campaign inputs and executed FastAl callback/NPT commands, records the
actual force/integration precision and reuses the existing passed numerical
audit. The native CPU MEAM verification remains the default without a campaign
manifest. The [seven-source recipe](../configs/analysis/al_nuclei_v100_seven_extract_20261007.json)
and [lineage recipe](../configs/analysis/al_nuclei_v100_seven_20261007.json)
use the ordinary inventory and search commands at exact 0.10 ps.

New extractions write a separate compressed array for each detector. To reuse
older extracted labels in this layout, without rerunning descriptors:

```bash
python -m src.postprocessing cache repack --plan OLD_EXTRACTION_PLAN \
  --output NEW_PREPARATION_DIRECTORY --cache NEW_CACHE_DIRECTORY
```

Use a distinct cache directory outside the repository. This verifies every
source chunk and its extraction identity, leaves the original cache intact, and
can resume completed chunks. It requires a complete extraction plan with `tasks`.
Point a fresh search configuration's `parent_output` at the new preparation and
set `parent_stage` to `extraction`. The usual `search prepare|submit` workflow
then reuses the repacked chunks. Repacking costs an extra read/write pass and
temporary disk space; it is useful for repeated searches, not mandatory for
reading old data. Existing active jobs keep their recorded inputs and code.

To resume an interrupted Ni search with a verified equivalent implementation:

```bash
python -m src.postprocessing ni-search analyse --config FROZEN_CAMPAIGN_CONFIG \
  --plan-sha256 EXISTING_PLANS_JSON_SHA256 --workers 12
```

This pins `ANALYSIS/technical/plans.json`, retains its source population and
chunk/tracking identities, and verifies the recipe and both plans. Completed
chunks and tasks are reused; unfinished extraction writes the selective layout.
Per-job/source execution receipts record the actual implementation and reused
work. Use a new frozen execution bundle and keep the earlier one. A queued
campaign with no existing search plan uses ordinary `ni-search analyse`.
The October 7 authorized relaunch and active jobs are recorded in the
[performance log](nuclei_search_performance.md#optimized-worker-relaunch).

Ni analysis now schedules chunks across sources instead of reserving a worker
for each whole trajectory. `--workers` must fit the allocated CPU count; keep
one descriptor thread per worker. To begin an unprepared campaign's analysis
while its producers are still running:

```bash
python -m src.postprocessing ni-search analyse --config FROZEN_CAMPAIGN_CONFIG \
  --workers 12 --wait-for-jobs MD_JOB_1 MD_JOB_2
```

Only completed, verified source archives are admitted. Their immutable plans
are saved in `technical/source-plans/`; new simulation completions can join
without changing previously extracted chunk identities. The final `plans.json`
references each source plan and retains failed/missing sources in coverage.
This command observes producer jobs; it never stops or modifies simulations.
The controller holds a scheduler lock and reports available/completed sources,
verified/reused chunks, in-flight tasks and waiting state in `progress.json`.

Al histories cache source-frame nearest-neighbor queries and precursor nodes
across all declared birth anchors. Shared nodes include the smallest clusters;
tracked component labels alone cannot supply them. Event-specific local arrays,
masks and scalar observations are checkpointed after each frame. Both caches
live below the configured external cache root and are bound to the complete
implementation contract and frozen population. Completed scientific packets
remain separate durable outputs. Resuming verifies saved frames before reuse.

Recipes remain in `configs/analysis`, `configs/data` and `configs/simulation`;
their location does not duplicate the implementation. See the
[birth dataset](datasets/nuclei_birth.md), [lineage protocol](al_nuclei_lineage.md),
[birth histories](datasets/nuclei_birth_histories.md), and
[performance measurements](nuclei_search_performance.md).

## Dataset API

```python
from src.postprocessing.datasets import (
    NucleiBirthDataset, NucleiBirthLabels, NucleiBirthHistories,
)

with NucleiBirthDataset('010', expected_fingerprint=YOUR_PIN) as data:
    for event in data.events(selection='established', detector='ptm'):
        source = data.source(event['source_id'])
```

The stable IDs, existing catalogue fingerprint, event IDs, ancestry and cadence
rules do not change through code relocation. Native 0.30 ps still excludes
subsampled finer histories and irregular raw prefixes. Published supplements
keep their own fingerprints. A future explicit catalogue refresh records the new
execution implementation; audited equivalent producers retain scientific event
identity. No catalogue refresh was performed as part of this refactor.

## Efficiency and provenance

Al and Ni share extraction and its chunk reader. PTM-only requests skip unused
bond-order calculations. Tracking inflates only the requested detector's array;
retained raw PTM is loaded lazily and reused by history annotation. Full-detector extraction
keeps the ordered float32 correlation kernel; histories use only the three saved
l=6 summaries. Shared geometry computes the periodic displacement field once.

Keep descriptor caches outside the repository using the configured cache root.
Source memory maps and loaded chunks are bounded. `descriptor_threads` controls
per-worker descriptor threads; launchers budget those CPUs explicitly. Cache
receipts verify input identity and content checksums. PTM labels are never inferred
from minimum-size component labels, which omit the smallest embryos.

`configs/postprocessing/producer_equivalence.json` explicitly maps a reviewed
classifier file hash to its existing cache identity. New extraction receipts also
record the actual classifier file hash. Unknown implementations get distinct
cache identities. Catalogue execution equivalences similarly require the complete
reviewed dependency set, preserving actual producer records alongside the stable
scientific identity. Frozen historical result contracts are not rewritten.

## Verification

Manual real-data checks for the October 7 consolidation are recorded in
`output/postprocessing/consolidation-20261007/technical/`:

- Exact equality for full-cell components, counts, bond histograms and atomic
  PTM on three 70,304-atom Al frames.
- Exact equality of events, interactions, trajectory summaries and decompressed
  family traces for a 3,211-frame Al lineage task.
- Exact equality of all six scientific packet files for an Al and a Ni birth,
  including movies and precursor graphs; Ni plots, GIFs and HTML render successfully.
- Catalogue readers retain 295 native 0.10-ps and 150 native 0.30-ps established
  PTM births and fingerprint
  `7720ac6b92886adb29013beb4e28b78e7def6ba16e9a175066b6574a4c27380d`.
- All 14 public command help screens and 79 relocated/calling module imports
  pass in the existing environment, including its MACE/e3nn load setting.

Five pre-existing metric families have stale dependencies in older
relaxation/appearance analyses (`liquid_controls`, `birth_prediction_relaxed`,
`birth_appearance`, `birth_appearance_cv`, `birth_shooting_appearance`). Their
mismatches were recorded, not accepted as new calculations by this refactor.
No automated test suite, training run or replacement Slurm job was created.

The subsequent acceleration pass additionally verified exact component/descriptor
values on a million-atom Al frame, all four detectors on a shooting trajectory,
and both cache layouts including raw PTM reuse and repack resumption. All 429
dataset reference lists and the 29-source preparation are unchanged. See the
[incremental measurements](nuclei_search_performance.md#second-acceleration-pass)
and `output/postprocessing/acceleration-20261007/technical/`. This pass also
validated all 15 command help screens, 94 package/consumer imports, and the
core metric contracts and PTM identity inside a frozen worker bundle.
