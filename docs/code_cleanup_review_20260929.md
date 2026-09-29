# Code cleanup review — 2026-09-29

The largest opportunity is to consolidate execution and evidence-handling code
that has accumulated inside individual research studies. Keep scientific
protocols separate. Address checkpoint/result correctness before splitting large
files or applying formatting.

This is a review and implementation backlog, not an applied source refactor.
The subsequent [implementation record](code_cleanup_implementation.md) describes
the shared methods now extracted and their artifact comparisons; the remaining
findings below stay on the backlog.
It covers the current working checkout, including uncommitted/new source.
HEAD was `e7ac340f`; the initial status contained 455 modified, 109 deleted and
394 untracked entries across the repository. Other work continued during the
review, so counts and line references describe an approximate live snapshot.
No production code, running jobs, model artifacts or scientific results were
changed by this review.

## Scope and evidence

I scanned all Python under `src/` and `scripts/` using AST parsing and tokenization
and inspected representative producers and callers. I also consulted the previous
[source refactor](src_refactor.md), [repository cleanup](repository_cleanup_20260912.md),
[MACE runtime refactor](encoder_research/runtime_refactor_20260925.md), command
index, result publication rules and current working instructions.

| Area | Python files | Approximate lines |
| --- | ---: | ---: |
| `src/research/` | 488 | 66,000 |
| `src/simulation/` | 65 | 47,000 |
| `src/analysis/` | 65 | 29,200 |
| `src/temporal_vamp/` | 57 | 26,200 |
| `src/training_methods/` | 142 | 22,100 |
| `src/data/` | 67 | 16,800 |
| Entire scan, including other source packages and scripts | 1,039 | 232,500 |

There are 38 Python files over 1,000 lines. Eighteen groups of function bodies
of at least nine lines are AST-identical, disregarding formatting and function
signatures. These are candidates for inspection, not proof that all callers have
the same contract. Tokenization found about 8,000 statement semicolons in 405
files, excluding semicolons inside strings/comments.

Scripts are already thin. Most of `src/data_utils/` is now small historical
import/command bridges. Repeating those earlier namespace moves would offer
little benefit. This was a static review: no complete scientific fit, live Slurm
audit or retained-checkpoint numerical comparison was performed. No new automated
tests or test dependencies were created.

## Prioritized findings

### 1. Checkpoint discovery mixes evidence with guesses — high priority

**Evidence:** `src/experiment_runner/results.py:33` calls a checkpoint "best" by
excluding filenames containing `last` and taking the alphabetically first
remaining filename. `src/experiment_runner/runner.py:65` uses that fallback to
initialize a dependent stage when the recorded checkpoint path is absent or
missing. This can affect the model used by a subsequent stage, not just a report.

At `results.py:84`, any checkpoint-loading exception is silently skipped.
At `results.py:95`, `best_model_score or current_score` treats a valid zero best
score as absent. Scalar truthiness is the wrong missing-value check.

**Change:** separate selected-checkpoint resolution from exploratory checkpoint
listing. Use the producer's recorded selected path and declared selector, or a
verified checkpoint callback receipt. Report absent/ambiguous selection with the
run and candidate paths. Preserve a zero score using an explicit `None` check;
report unreadable artifacts with their paths and exceptions. A tolerant inventory
can record unreadable entries explicitly, rather than hiding them.

**Validation:** inspect the actual callback/receipt fields from retained runs and
compare dependent-stage overrides against their recorded selections. Do not add
AP-based selection or recalculate historical selections.

### 2. Tracking policy has multiple owners — high priority

**Evidence:** `src/research/supervised_onset/tracking.py:22` already provides online
training, local evaluation, stable IDs and API-only updates to an existing run.
The scan found eight direct `wandb.init` call sites with different behavior.

`src/training_methods/trainer.py:435` takes the mode from config, sets global
environment variables and initializes without an explicit stable ID, resume
policy or entity. `src/training_methods/shared/mace_logging.py:40` uses a recorded
ID but `resume='never'`. `src/training_methods/shared_pretraining/analysis.py:173`
calls `wandb.init` while exporting frozen predictive diagnostics. That path can
create or restart an online run rather than only updating the associated
training run. These implementations do not consistently enforce the current
policy for future invocations; this review did not establish that any particular
historical run was incorrectly tracked.

**Change:** place the general tracking lifecycle under execution ownership, with
three explicit operations: online scientific training; local diagnostics; API
summary updates to a recorded training ID. Adapt Lightning to that lifecycle.
Keep study-specific metric names and scientific summaries in their study.
Preserve existing run IDs/receipts and historical input descriptions.

**Validation:** inspect the config and receipt of each supported caller; verify
failure reporting, ID preservation and diagnostic behavior with local substitutes
for network calls. Do not initialize online runs for cleanup checks.

### 3. Row-based metric exports bypass part of the evidence binding — high priority

**Evidence:** `src/experiment_runner/metric_docs.py:126` writes a table atomically
and records its CSV hash, family and definition path under
`technical/table-contracts/`. Several studies instead snapshot definitions and
write their own CSV directly:

- `src/research/liquid_predictability/control_train.py:27`;
- `src/research/liquid_predictability/descriptor_baselines.py:35`;
- `src/research/liquid_predictability/descriptor_fit.py:39`;
- `src/research/liquid_predictability/evaluate.py:14`, using
  `src/research/spatial_approach/evaluate.py:13`.

Those writers retain family definitions, but do not create the same per-table
binding as the shared exporter. They also duplicate column handling and use
`rows[0]` to determine the schema.

**Change:** extend the existing metric export module with an explicit row-table
operation. Reuse its family validation, atomic write and table receipt logic;
accept the producer's declared columns. Migrate writers without changing row
order, numeric values, undefined-value conventions or scientific calculations.
Keep an intentional policy for empty tables instead of deriving it from an
indexing error.

**Validation:** compare CSV values, order and precision on saved data and verify
the new binding hashes. Review `docs/metrics/` and `contracts.json` together with
producer changes; preserve historical exported definitions and use new revisions
where contracts change.

### 4. General helpers depend on whole research studies — medium priority

**Evidence:** newer studies import `sha`, `digest`, JSON writing and checkpoint
writing from `src/research/structural_state/common.py` or
`src/data/fixed_cohort/protocol.py`. The former imports Torch and defines the
structural-state study; the latter imports crystallization onset calculations.
For example, `src/research/spatial_vicreg_bias/queue.py:15` needs only basic
evidence helpers. `src/research/liquid_predictability/train.py:11` imports deadline
and compilation helpers from another study's full training module.

The helper definitions are similar but not interchangeable: the two `digest`
functions differ in their NaN policy, and JSON writers differ in temporary-file
naming. `src/experiment_runner/artifacts.py:29` already has a strict JSON writer
with a process-specific temporary name.

**Change:** reuse/extend neutral modules under `src/experiment_runner/` or
`src/project_runtime/` for evidence I/O, source freezing and allocation deadlines.
Place model compilation helpers beside their model runtime. Leave fixed-cohort
sampling and structural-state identity construction in their current domains.
Document serialization bytes and hash semantics before consolidating them.

**Validation:** compare existing manifest/identity digests exactly; audit new
dependencies in study identities and metric contracts. An import-only extraction
still changes source hashes, so frozen runs keep their frozen producers.

### 5. Queue plumbing is repeatedly rebuilt — medium priority

**Evidence:** `src/research/liquid_predictability/queue.py:23` and
`src/research/spatial_vicreg_bias/queue.py:19` both freeze source trees, write
sbatch files, submit jobs, persist receipts and manage GPU lanes. Their dependency
and continuation protocols differ. Deadline retrieval is also split between
`src/research/crystal_vector/train.py:57` and
`src/training_methods/shared_pretraining/queue.py:19`, with different scheduler
formats and reserve times. Existing Slurm machinery is in
`src/experiment_runner/slurm.py`.

**Change:** extract only the repeated mechanisms: frozen source bundle creation,
quoted command rendering, job submission/receipt persistence and scheduler
deadline retrieval with an explicit reserve. Keep each queue's graph, locks,
preparation gates, lane assignment and continuation choices readable in that
queue. Do not build a new universal experiment framework.

**Validation:** render commands and dependency receipts locally, compare their
resolved paths/options, and exercise existing local dry-run workflows. Scheduler
submission is unnecessary for the first extraction.

### 6. Analysis combines too many stages and side effects — medium priority

**Evidence:** `src/analysis/pipeline.py:169` is a 1,381-line function handling
config resolution, inference/caches, clustering, diagnostics and publication.
`src/analysis/real_md_qualitative.py` is 4,454 lines; its entry function at line
3370 is 1,082 lines. The latter mixes projection fitting, transitions, flicker
calculations, animation scheduling, plotting and Markdown output.

**Change:** extract coherent stages with explicit inputs/results: inference,
cluster fitting/assignment, physical and temporal readouts, rendering, and
publication. Start with one self-contained area, such as flicker rendering, then
separate its numerical producer. The pipeline should show stage order and the
actual artifacts passed between stages. Reuse existing settings dataclasses;
avoid replacing a long function with a large mutable "context" dictionary.

**Validation:** render from saved predictions/assignments where possible; verify
sample order, cache identity, selected cluster counts and figure links. Numerical
extractions require saved-output parity and reviewed metric contracts. Layout-only
changes must not refit projections or clusters.

### 7. Simulation qualification and visualization need focused splits — medium priority

**Evidence:** `src/simulation/atomistic/potential_benchmark.py:1428` contains a
1,384-line `_melting_evidence`, including a 587-line nested artifact verifier at
line 1495. `src/simulation/atomistic/config.py:1125` has an 801-line `load_config`.
`src/simulation/visualization.py` contains 4,657 lines of structural calculations,
paper panels, reference preparation and galleries.

**Change:** separate qualification evidence loading, artifact verification,
protocol validation and statistical summaries into named operations. Keep every
qualification gate and failure context. Split visualization by physical readout,
panel rendering and gallery assembly; preserve callers while migrating consumers.

There are also concrete small duplication candidates: the LAMMPS environment
builders at `src/simulation/campaigns/independent_meam_source.py:434` and
`src/simulation/atomistic/lammps_shooting.py:1008` have identical bodies. Start
there before touching the qualification engine.

**Validation:** replay recorded evidence and compare qualification decisions,
failure reasons and exported summaries. Do not merge physically different
campaigns: `src/simulation/campaigns/predictive_dynamics_15ps.py` explicitly uses
a different restart-capable thermostat from its older campaign.

### 8. Dense training code hides state and model contracts — medium priority

**Evidence:** `src/research/liquid_predictability/control_train.py` has 93 statement
semicolons in 266 lines; `src/research/spatial_vicreg_bias/evaluate.py` has 78 in
393 lines. At `liquid_predictability/train.py:196`, backward, clipping, optimizer
update and both counters occupy one line. Checkpoint loading/saving and array
assembly use similarly dense statements.

`ControlMACE` at `control_train.py:33` constructs the full `JointCrystalVector`
then deletes four heads. `DistanceMACE` at `models.py:22` inherits direction heads
that its forward path does not use. Feature layout constants such as scalar128,
vector16, 25 context patches and 80 atoms appear inside packing/unpacking code.

**Change:** expand multi-action lines in one active workflow first. Give training
progress/checkpoint payloads explicit fields. Define the exported feature layout
at its actual producer and derive slicing/reshaping from it, retaining deliberate
protocol checks. Separate the shared patch/context trunk from task heads after
the readability pass.

**Validation:** formatting can be checked through AST equality. Model extraction
needs state-dict/output/gradient checks and initialization review: removing
unused layer construction changes RNG consumption, so the same seed alone does
not preserve old initialization. Do not claim exact continuation after such a
change; retain old architecture/source for existing checkpoints.

### 9. Viewer maintenance parses and patches generated HTML — medium priority

**Evidence:** `src/research/spatial_vicreg_bias/comparison_layout.py:37` and
`checkpoint_explorer.py:87` recover scientific payloads by splitting HTML on
`const D=` and `;\nconst palette`. `checkpoint_explorer.py:31` builds an explorer
by replacing literal labels and HTML fragments. A harmless template edit can
break extraction or silently stop inserting a control. Navigation and default
run names are also hard-coded in `comparison_layout.py:11`.

**Change:** save a versioned payload JSON alongside each new page and render from
that artifact. Use explicit template slots or a separate explorer template for
controls, and make publication navigation a declared mapping. Read old HTML once
when migrating a selected historical bundle; preserve its original page/payload
and rendering history rather than spreading fallback parsing across workflows.

**Validation:** compare payloads and scientific asset hashes, then inspect source,
frame, checkpoint, representation and dataset controls in the rendered viewer.
Use existing publication-only paths; no clustering/projection recalculation.

### 10. Documentation and compatibility need clearer lifetimes — lower priority

**Evidence:** the script index is 854 lines and `configs/` has 431 files. The root
README still recommends `pointnet` at lines 54 and 63–65, while the current
instruction is `pointnet-torch214`. `configs/README.md:498` presents AP-selected
recipes as new, unsubmitted work, which conflicts with the current objective.
Historical recipe details must stay truthful, but their lifecycle is unclear.

**Change:** keep the root README focused on current setup and principal commands.
Group command navigation by workflow, link to the detailed operational docs, and
mark active, diagnostic and historical/retired recipes explicitly. Mark the old
AP-selected proposals as historical and unavailable for future queues. Migrate
live consumers to canonical imports when touching those consumers.

Do not delete the small `src/data_utils/` and SSL bridges merely because they
look redundant: retained pickle/module paths are an explicit repository contract.
The two temporary independent-MEAM launchers already have removal conditions in
[the previous cleanup checklist](src_refactor.md#post-queue-cleanup-checklist).
Their live dependencies were not checked in this review.

## Suggested implementation sequence

| Slice | Concrete outcome | Relative effort |
| --- | --- | --- |
| 1 | Correct checkpoint selection/zero handling/error evidence; align current documentation | Small to medium |
| 2 | Shared row-table export and complete per-table provenance | Small to medium |
| 3 | Move general evidence/tracking/deadline primitives to neutral owners; migrate one active study | Medium |
| 4 | Expand the liquid control trainer; make checkpoint state and feature layout explicit | Medium |
| 5 | Reuse source-freezing and submission primitives in two queues | Medium |
| 6 | Viewer payload artifacts and explicit templates | Medium |
| 7 | Extract analysis stages and simulation qualification components one at a time | Large |

Useful bounded extractions include the identical Hungarian metric-key selectors
at `src/baselines/descriptor_baselines.py:139` and
`src/training_methods/shared/supervised_cache.py:138`, and the 34-line atlas
checkpoint payloads at `src/temporal_vamp/commands/atlas_finetune.py:128` and
`atlas_temporal_encoder.py:94`. Audit signatures/callers and frozen contracts
before relocating even these identical bodies.

Success means fewer owners for execution/evidence mechanics, a visible scientific
dataflow and explicit failure causes. Splitting files alone does not achieve that.
Keep objectives/selectors, fixed-cohort rows, input covariates, physical history
spacing, geometry-only channels and frozen metric definitions intact. Preserve
active cache leases and the six retained feature caches; storage deletion is
outside this source review. Validate each source-changing slice with existing
workflows and local disposable diagnostics, without adding automated tests or
creating W&B diagnostic runs. Frozen source snapshots and existing research
identities remain the authority for historical reproduction/resume.
