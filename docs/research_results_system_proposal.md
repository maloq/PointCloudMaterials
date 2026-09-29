# Research results and experiment tracking: audit and proposal

**2026-09-26 — proposal, not an implemented layout change.**

Implementation follow-up: [the adopted system and commands](research_results_system.md).
The audit and staged proposal below remain the original design record.

The repository has useful scientific analysis and substantial provenance tooling.
The main problem is that a storage convention has become the information model:
folders decide what is a run, file extensions decide what is evidence, and a
small flat gallery decides which parts of an analysis are visible.

I recommend keeping the existing research roots and workflows, making each
analysis a named, navigable result, and teaching the existing catalogue explicit
study/run/evaluation identities. Organize figures by scientific purpose and
snapshot. Separate execution records from conclusions. Recover historical
navigation before changing where producers write files.

## Audit scope and evidence

Inspected the working tree at commit
`e7ac340f7fce003bcc76c98289d46076c39aad19`, including its pre-existing uncommitted
changes; that commit alone does not identify all inspected bytes. Read the layout,
workflow, storage and encoder-research guides; traced the standard analysis
publisher, general registry, encoder catalogue, metric exporter, execution
tracker and representative recent queue/report/W&B implementations. Inspected
local result trees and existing generated inventories. Small read-only Python
checks used `pointnet-torch214`.

The initial local walk did not follow directory symlinks into WORK/IDS/STORE. It found
64 experiment directories and 226 files under `experiments/`, with no `.py`,
`.sh`, `.sbatch`, `.log`, `.out` or `.err` files. This is not an exhaustive
classification of every scientific note. External collections were assessed
through their registered metadata and existing reports, not a fresh remote
census. No inference, fitting, simulation, scheduler query, W&B request or
catalogue refresh was performed. Other work continued in this shared checkout;
the inventory figures describe observations during this audit, not a locked
filesystem snapshot.

### Findings

| Priority | Finding and repository evidence | Consequence |
| --- | --- | --- |
| High | [`analysis_artifacts()`](../src/experiment_runner/artifacts.py) sends every new standard analysis into `technical/`; [`pipeline.py`](../src/analysis/pipeline.py) passes this as its output directory. | Complete scientific subtrees are hidden together with implementation artifacts. The problem is in the producer contract, not merely directory naming. |
| High | [`publish_report()`](../src/analysis/report.py) uses a hard-coded shortlist and flattens snapshot names into filenames. In [`gatr-vicreg-step3072-al-20260918`](../output/structural_static/gatr-vicreg-step3072-al-20260918/README.md), 104 PNG/HTML/SVG/PDF artifacts are under `technical/`, while `plots/` contains 39 symlinks. | Many spatial views, crystal-like subsets, raytraces and representatives remain discoverable only through the producer tree. The flat page loses the analysis hierarchy. These counts describe files, not independent findings. |
| High | [`registry.run_id()`](../src/experiment_runner/registry.py) returns the first path component for modern paths: both `mace/mean-blocks-seed20260910/tables/metrics.csv` and `mace/residual-blocks-seed20260912/tables/metrics.csv` become `mace`. | The general registry conflates question/family containers with distinct runs. Recipe lookup then assumes the same identifier exists directly under `experiments/`, despite dated study names. |
| High | General-registry classification defaults to `experiment` with a small name-based maintenance exception list. Its saved September 18 inventory labels `hardware_benchmark`, `environment`, `relaxed_tda` and `ti_mlip_handoff_20260916` as experiments. Only 4 of its 23 local entries have a recipe link. | Operational records appear as scientific experiments. The present `experiments/` directory is cleaner than the generated dashboard suggests; the remaining mixing is particularly evident in outputs and classification. |
| High | [`check_metric_docs()`](../src/experiment_runner/metric_docs.py) only reads/filter-selects contracts. `snapshot_metric_docs()` copies declared hashes without computing them, tolerates absent descriptions/families and overwrites the destination contract. | The advertised integrity check is not implemented. A read-only independent hash comparison during this audit found **all 1,062 declared file references across 82 families matching**; this is a missing enforcement mechanism, not evidence that today's hashes are wrong. |
| High | `publish_report()` calls `write_metric_table()` on saved analysis metrics; that exporter takes descriptions/hashes from the current checkout. | Re-publication can attach today's definitions to historical numerical results. Repeated exports of different metric families into one root also share one `METRICS.md` and contract pathname. Historical render and numerical recomputation need distinct provenance. |
| Medium | [`report.py`](../src/analysis/report.py) skips absent shortlist artifacts as optional and overwrites `README.md`/`index.html`. | A reader cannot distinguish disabled, pending, failed and unavailable sections. Hand-added scientific interpretation can be lost on refresh. The observed GATr README already contains a useful results link absent from the generator template. |
| Medium | The general registry snapshot is dated September 18; the encoder catalogue snapshot is September 23. The latter records 77 studies, 150 collections, 13,086 artifacts and 2,449 CSV/result-JSON artifacts without a nearby definitions file. | Navigation freshness and provenance coverage need visible status. These are saved inventory counts, not current experiment counts or proof that all 2,449 items need the same metric glossary. |
| Medium | [`encoder_catalogue.py`](../src/experiment_runner/encoder_catalogue.py) preserves evidence well, but its normalized entities are families, studies, artifacts, raw records and curated headlines. [`database.md`](encoder_research/database.md) explicitly notes that comparison groups are not sufficient proof of comparability. | It cannot consistently join one fit, multiple checkpoints, export stages, readout fits and evaluations. Duplicate bytes and duplicate scientific results are different problems. |
| Medium | General execution attempts, specialized queue state, per-model completion receipts and W&B receipts coexist. Recent [`encoder_context/report.py`](../src/research/encoder_context/report.py) already verifies prediction hashes and paired sample identities. | Valuable checks exist, but there is no common result receipt that makes their evidence discoverable across protocols. A successful command, complete fit and complete evaluation are different facts. |
| Medium | [`.gitignore`](../.gitignore) excludes every `output/**/tables/metrics.csv` as training history, while the standard analysis exporter uses exactly that name for scientific scores. `git check-ignore` confirmed this for the GATr and MACE examples. | Compact scientific tables are excluded from normal Git publication by a filename heuristic. Training histories and evaluation scores need explicit roles and different names for future outputs. |

The restored [FactorVAE/GeoFrame gallery](../output/factor_vae_archive/geoframe-v2-vicreg-epoch034/README.md)
is a useful counterexample: it preserves historical bytes, nested plot structure,
copy receipts and the fact that restoration was not new analysis. Reuse that
care, without requiring duplicate copies for every routine local gallery.

### What to keep

- The scientific records in `experiments/`, reusable recipes in `configs/`, and
  operational/simulation documentation in `docs/`.
- Existing model-native export adapters and distinct scientific protocols.
  A common report format must not silently standardize their scientific meaning.
- Source/config snapshots, checkpoint hashes, paired prediction arrays, fixed
  cohort identities, metric definitions and historical negative findings.
- The encoder catalogue's raw evidence preservation, quotation checks, SQLite
  integrity checks, portable storage roots and explicit remote-summary labels.
- Existing immutable campaign identities, precise resume behavior, local W&B
  receipts, and source-paired comparisons where already implemented.

The [earlier improvement list](encoder_research/improvements.md) already identifies
several of these needs. This proposal makes the storage and publication boundary
concrete; it should extend that work rather than create a third independent index.

## Proposed information model

Use explicit identifiers and relations. Paths are locations, not identities.

| Entity | Meaning here | Ownership |
| --- | --- | --- |
| Study | A question, declared protocol and interpretation; may span several campaigns | Versioned scientific record under `experiments/` |
| Campaign | A planned set of treatment/seed runs and their dependencies | Existing campaign configuration and manifest |
| Run | One logical trained component/seed, or a declared analysis-only computation; resumes retain its identity | Producer-generated run record |
| Attempt | One execution of a run/stage, with environment, allocation, exit state and source snapshot | Existing execution tracking, adapted where necessary |
| Model/export | Checkpoint identity plus native loading contract; exported encoder/projector/pooled/typed feature identity | Producer receipt referencing retained artifacts |
| Evaluation | An immutable model/export + population + assay/readout + calculation contract | Evaluation receipt and its original evidence |
| Report | A presentation of one or more evaluations, with a separate rendering revision | Generated navigation and figures; authored interpretation remains separate |
| Artifact | A file/bundle with a scientific role, owner, provenance and retention class | Producer manifest, imported explicitly for legacy outputs |

Existing campaign directories frequently contain many fits. Import them as
campaign containers with child run IDs; do not call a 16-fit campaign one model
or physically split it merely to satisfy the new index. Analysis-only and
cross-model comparison results are valid without any new training run.

One trained encoder can have many checkpoints and evaluations. A frozen probe
is its own fitted component linked to the encoder and evaluation protocol.
Repeated publication, archive copies and a new figure style create no new fit.
An intentional recalculation creates a new evaluation revision with a reason
and a `supersedes` reference; it does not overwrite the old scientific evidence.

### Small records, with clear authorship

Add a short `study.yaml` beside each newly adopted study README: stable study ID,
question, protocol revision, declared comparisons, campaign/config links and
status. The person conducting the study maintains interpretation and decisions.
The producer writes run/evaluation/artifact records automatically; researchers
should not manually repeat metrics in five indexes.

A new evaluation receipt should reference, rather than duplicate, existing
manifests. Required information includes:

- Run/component IDs, checkpoint hash, native architecture/loading revision,
  encoder versus projector or typed export, scaler and precision.
- Data release, source ancestry/split and exact observation/target identities,
  including the `all64` or `legacy16` track. Record a canonical sample-key order
  and target hash; do not rely on array shape or uncanonicalized file ordering.
- Separate encoder and predictor inputs: focal support, computational halo,
  history, motion, relaxation, conditions and training-only teachers. Link to
  evidence from the actual tensor producer, not just a descriptive config tag.
- Training objective and checkpoint selector; readout objective, selector,
  calibration rule and split roles. Keep historical AP-driven or time-conditioned
  fits honestly labeled.
- Assay version, horizons/lag, target definition, weighting, censoring,
  normalization, controls, uncertainty method and independent source/seed counts.
- Requested/completed/failed/disabled stages, expected output artifacts and
  producer/metric hashes. Missing evidence has an explicit reason.

Keep a stable human-readable alias alongside the immutable identity. A file
path or display-name change must not create a new scientific result. Historical
unknowns remain unknown; new exports fail with context when required fields or
artifacts are absent. Legacy incompleteness and new-run requirements are
different policies.

## Proposed result layout

Retain `output/<question>/<run>/` for new standalone runs and result collections.
The important addition is **a named analysis bundle**. Its scientific structure
is public, with its own tables, figure groups, evidence and completion receipt.
The example below is a proposed convention, not a claim these files exist.

```text
experiments/<question>_<date>/
  README.md                         question, protocol, reproduction and result links
  study.yaml                        small authored study record
  FINDINGS.md                       optional cross-run interpretation/decision history

output/<question>/<run>/
  README.md                         short authored result card
  index.html                        generated overview and analysis navigation
  run.json                          identity, relationships, links to producer receipts
  artifacts.json                    typed artifact inventory; locations may be external
  plots/overview/                   a few selected summary figures, when useful
  tables/                           compact cross-analysis summaries, when applicable
  analyses/
    static-al-v1/
      README.md                     scope and limits of this analysis
      index.html                    grouped gallery
      analysis.json                 evaluation identities and stage coverage
      plots/
        latent/
        snapshots/166ps/            spatial views + representatives for this frame
        snapshots/170ps/
        representatives/
        transitions/
        connected-regimes/
      tables/
        scores.csv
        cluster-profiles.csv
        METRICS.md                  definitions frozen when numbers were computed
      data/                         retained scientific arrays/assignments for this assay
      technical/
        metric-contract.json        frozen calculation identity
        rendering.json              separate rendering identity
        logs/
    prediction-al64-v1/             own plots, tables, data and technical receipts
    noise-v1/
    dynamics-lag075-v1/
  artifacts/
    models/                         retained models, scalers and export contracts
  technical/
    attempts/                       execution records/source/config/environment snapshots
    logs/
    metric-contract.json            only if root summary tables are exported here
```

Create only applicable directories. A small evaluation needs a card, receipt
and table, not a full empty tree. Existing campaign containers keep their paths;
the catalogue can link their child fits and analyses without relocating them.
Cross-model comparison bundles refer to existing evaluation IDs and predictions,
and own only their additional paired comparisons and presentation.

This deliberately changes the current rule that all machine-consumed analysis
outputs go under `technical/`: scientific assignments, predictions and tables
are evidence and deserve named homes. `technical/` means execution and
reproducibility details; it is **never a deletion category**. Its source snapshots
and metric contracts remain protected. Directory role and retention policy are
separate fields.

Model and data artifact locations can be portable storage references resolved
through `machine.local.yaml`. Caches remain on IDS; existing analysis/input
storage stays on WORK; new simulations and verified publication follow the
existing SCRATCH/STORE rules. Keep the six most recently used generated encoder
feature caches across lanes, honor active leases, and preserve checkpoints,
predictions and metrics. This proposal introduces no new cleanup policy.

### Separate science from operations at registration

Use explicit record kinds: `research`, `simulation`, `dataset`, `operations`.
Use activity types within research, such as training, evaluation and comparison.
An analysis of information, noise response or stability is scientific even when
called a diagnostic. A VRAM, throughput, package-installation or storage check
is operational even when it uses the same encoder.

New operational outputs can use the established
`output/maintenance/<activity>/<run>/` namespace. Simulation and dataset records
stay in their existing systems and are linked as dependencies. Legacy operational
paths can be classified correctly without moving them. A numerical-equivalence
check may support a scientific result while retaining its operational identity.

## Presentation that serves this research

The default landing page should answer: what was asked, what completed, what the
evidence supports, and which comparison is valid. Put a few selected figures and
the main table there. Provide sections for **Structure**, **Prediction**,
**Dynamics and noise**, **Spatial/representatives**, and **Provenance**, showing
only the sections relevant to the declared protocol.

For the old static analyses, preserve the natural hierarchy: choose a snapshot,
then view its spatial cluster map, crystal-like subset, alternate view and
representatives together. A cluster selector can link the same cluster's profile
and transition evidence within that analysis. Do not align cluster numbers
across independent fits. Keep global projections and full temporal views in
their own sections. Search/filter may help large galleries, but the page should
remain useful as a plain offline HTML document.

Every figure entry needs a meaningful title/caption, analysis/model identity,
population and sampling, units/lag/horizon, transformations, source table or
array reference, and whether it is descriptive or held-out evidence. Snapshot
time is valid navigation metadata; it is not permission to feed time into a model.
State unavailable sections as `disabled`, `pending`, `failed` or
`unavailable: cache retired`, rather than silently omitting them.

Use consistent visual conventions within a matched comparison: fixed model
colors, stable physical labels, declared units, comparable axes where justified,
and paired source effects separated by training seed. Save one primary image
plus explicitly requested export formats. Do not force expensive UMAP, raytrace
or every diagnostic on every run merely to fill a standard report.

The default scientific summary should cover applicable evidence across:

| Question | Suitable evidence |
| --- | --- |
| What present information survives? | Actual training heads versus frozen probes; physical/TDA readouts, within-liquid results, neighbors and negative controls |
| Does it help crystallization prediction? | Declared predictive likelihood, raw/calibrated Brier and log loss, calibration, baseline deltas; AP at 3/6 ps as diagnostics |
| Does the representation behave sensibly? | State/movement spectra, declared-lag changes, coordinate-noise response, equivariance and numerical controls |
| Is the comparison trustworthy? | Exact cohort/support, source/seed uncertainty, all expected rows, failed stages, native versus refitted readouts and historical policy labels |

Preserve specialized forecast, conditional-information and spatial protocols.
Shared figure/table plumbing must not silently substitute targets, bootstrap
units, fitting populations or selectors. Discontinued physical-reconstruction
pretraining remains historical; physical-information readouts remain useful.

## Tracking, comparison and provenance

### One local catalogue, complementary views

Extend the existing encoder evidence catalogue for explicit runs, model exports,
evaluations and artifact relations. Make the general dashboard a view of the
same registration records, with separate research/operations/simulation/data
tabs. Retain raw imported records and historical aliases. During migration,
legacy scanners only suggest classifications; unknown items remain unclassified
until reviewed. Avoid another independent manually maintained registry.

The durable sources are versioned protocols and immutable producer receipts;
SQLite/HTML/CSV are rebuildable projections. Curated findings remain authored
and quote/link exact evidence. Register a completed immutable bundle once, then
incrementally refresh the catalogue without rereading checkpoints, training
caches or all 100,000-plus local output files. A full discovery audit remains
available separately. Record indexing failure visibly and retain the previous
index; indexing failure must not turn a completed training run into a failed fit.

A result page needs three separate status dimensions:

- Execution: planned, submitted, running, checkpointed, succeeded, failed or
  cancelled, with observation source and timestamp; remote liveness can be unknown.
- Evidence: which requested stages completed and were verified; partial is not
  complete, and execution success alone is not sufficient.
- Interpretation: exploratory, supported, inconclusive, superseded or withdrawn,
  with a human explanation and links to the evidence.

Record source availability separately from success. Distinguish a locally
verified bundle from a remote summary, an unavailable archive or an intentionally
retired cache. Show the last refresh and current collection coverage prominently.

### Comparison eligibility follows the protocol

A comparison declares the factors allowed to differ and the factors held fixed.
Require the same cohort/row-target identities, split role, horizon, target,
weighting and calibration/metric definitions when computing a paired effect.
Require matched readout/input contracts where the scientific question assumes
them; permit a declared support/history/readout ablation while displaying that
change. Model identity itself must differ in many legitimate comparisons.

Offer three explicit views: matched comparison, descriptive cross-protocol
comparison, and historical evidence with missing metadata. Preserve the full
fixed-cohort grid and show missing model rows; never silently use a
model-dependent intersection. Undefined values are not zeros. Source bootstrap
uncertainty does not substitute for training-seed replication. Static training
snapshots do not become held-out tests through better packaging.

Default ordering should follow the declared treatments or groups. No universal
encoder leaderboard, AP-driven selection or fitted ensemble weights. Preserve
historical objectives/selectors, temperature/time access and model channels as
recorded. New native MACE defaults remain width/export 128, batch/microbatch 256,
geometry-only constant atom channel and fixed material preprocessing, with
explicit deviations recorded. Enforce current input policy at the actual producer.

### W&B and local evidence have different jobs

Continue online W&B in `teshbek/PointCloudMaterials` for scientific training and
associated evaluation metrics. Reuse stable resumable run IDs and local receipts.
Link component/run IDs, parent encoder, campaign and evaluation receipts to the
existing tracker. Local fit identities should distinguish a frozen probe from
encoder training even where the current W&B convention summarizes probes under
the parent encoder. Do not change existing remote run identities during adoption.

Keep standalone rendering, index refreshes, smoke checks and hardware benchmarks
local. Reporting must work from saved evidence without a W&B connection. New
training still fails explicitly on authentication/network failures; there is no
offline fallback. Imported legacy artifacts must not create fictitious new
online training runs.

### Freeze calculations; allow presentation changes

At numerical export, verify the requested metric family's declared dependencies
against actual bytes and retain the verified hashes and description with the
evaluation. Fail on absent/mismatched required definitions. Scope each contract
to its table bundle so independent families cannot overwrite each other.

At historical publication, reuse the original calculation receipt and definitions.
If they are missing, label that absence; do not apply today's contract to old
numbers. Rendering records its own code/config identity. Style-only refreshes
must neither recalculate metrics nor rewrite authored findings. Numerical fixes
produce a new evaluation revision and correction note, retaining the original.

For new outputs, name training history `history.csv` under execution details and
evaluation scores `scores.csv` under the relevant analysis. Update Git publication
rules to include compact manifests, reports, summary tables and frozen contracts;
review size/content explicitly. Keep large per-sample exports, models and full
galleries in their documented storage. The name `metrics.csv` alone cannot
distinguish a scientific summary from a training log.

## Migration plan

Implement this incrementally. Do not bulk-move historical results, change active
jobs, rewrite frozen source/config snapshots, or require old workflows to run
under current scientific policies.

| Stage | Concrete work | Completion criterion |
| --- | --- | --- |
| 1. Recover navigation | Add an explicit legacy artifact adapter for the retained GATr static analysis; generate a grouped preview outside its source tree. Include all recognized scientific outputs, with an unclassified list for anything else. | All 104 visual artifacts are accounted for and navigable, including existing external HTML dependencies; their original bytes/paths and old links remain unchanged. No inference or metric re-export. |
| 2. Establish honest registration | Add study/run/evaluation/artifact records for that analysis, one restored GeoFrame bundle and one completed modern multi-fit comparison. Add explicit operational classification and replace path-derived identity for registered entries. | Independent MACE runs appear separately; campaign/fit/checkpoint/evaluation counts differ correctly; hardware/environment records leave the research view. Historical unknowns and partial results remain visible. |
| 3. Repair provenance publication | Implement actual family-scoped metric validation; distinguish frozen numerical contracts from render receipts and authored text; fix future score/history naming and compact publication rules. | Current 82 families validate; absent or changed required files produce contextual errors. Re-rendering a historical bundle changes no numerical bytes or original definitions. Multiple analyses retain separate contracts. |
| 4. Adopt future producers | Integrate bundle writing and artifact registration at existing stage boundaries, first in the standard analysis publisher and one maintained likelihood-based campaign. Reuse native export/row verification and online tracking. | One completed encoder has linked selected checkpoint, declared inputs, fixed-cohort prediction results, applicable structure/noise/dynamics diagnostics and readable stage coverage. Resume retains identities. |
| 5. Consolidate navigation | Generate study/result links and views from those registrations, preserving the existing raw evidence catalogue and human findings. Expand legacy adapters by demonstrated need. | One starting page reaches question → run → analysis → evidence; fresh exports appear without editing several independent lists. Missing storage/indexing is explicit. |

Stages 1 and 2 provide visible value before a broad producer refactor. Stage 3
should precede enabling any new historical re-publication that exports metric
tables. The Stage 1 preview must only index/link original artifacts.

Reuse implementation homes:

- `src/experiment_runner/artifacts.py`: small typed artifact/bundle interface.
- `src/analysis/report.py`: structured publisher and legacy adapter; use
  `src/analysis/output_layout.py`'s existing snapshot/MD semantics.
- `src/experiment_runner/metric_docs.py`: verified, scoped numerical provenance.
- `src/experiment_runner/encoder_catalogue.py` and `registry.py`: shared identity
  ingestion, derived views, explicit operational classification.
- Existing protocol queue/report modules: emit receipts at actual stage completion;
  do not replace their algorithms with a generic research runner.

Continue using existing CLI/config workflows; proposed verification and catalogue
actions belong behind maintained entry points, with their indexes updated. No
new automated tests, test-suite dependencies or `tests/` directory are proposed.
Validation can use explicit runtime audits and a manual pilot: verify hashes,
open representative links/interactive pages, inspect stage coverage and compare
the published tables to retained inputs. A full new web service, replacement
experiment platform, forced historical normalization or directory-wide cleanup
is unnecessary for the first implementation.

## Suggested first implementation scope

Deliver the GATr grouped gallery, its explicit artifact/evaluation record, correct
research/operations classification for registered entries, and the metric
provenance repair. Preserve all current outputs. Use that working example to
settle the record fields and browsing behavior before migrating more producers.
The success measure is whether an old scientific result is easy to understand,
trace and reuse without guessing its cohort, checkpoint or hidden folder layout.
