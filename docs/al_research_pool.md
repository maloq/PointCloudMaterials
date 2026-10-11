# Shared Al research pool

**October 8, 17:45 Europe/Paris:** main dense simulation is now **152/152
complete**. The [new GPU birth search](al_nuclei_lineage.md#october-8-completed-data-incorporation)
is running for 45 unsearched main trajectories plus 33 completed birth-focused
branches: 78 sources, 165,828 native 0.10-ps frames. Jobs 1029088–1029091 form a
detached GPU extraction / CPU tracking / collection / publication chain.
The separately completed seven-original-branch search has been incorporated
without recomputation: `010` now has 129 views and 312 established PTM roots.
Its fingerprint is recorded in the dataset guide. These additions retain source ancestry and do not create
independent melt counts. The older review below is a dated snapshot, not the
current queue. History-supplement publication and failed velocity recovery
retain their separately recorded status.

**Queue review, October 8 late morning (Europe/Paris):** no nucleus-search or
history worker is currently running or queued. Completed labels are not the same
as complete coverage of newly collected MD:

- The published `010` catalogue retains 122 source views / 295 established
  liquid-origin PTM records; native-only `030` retains 680 views / 150 records.
  Historical Al search releases and the separate seven-original-V100 search
  are complete. The latter's 17 established roots are not yet in the catalogue.
- All **33 birth-focused V100 replacements are complete**, with 66,333 saved
  frames; their retrospective birth search has not been prepared/submitted.
- Main dense collection: **147/152 complete**, two sources in dynamics, one in
  conversion and two waiting in the worker assignments. **40 completed main
  trajectories** are absent from the available frozen extraction plans. Together
  with the 33 replacements this is a known backlog of **73 completed trajectories**,
  with preserved shared ancestry rather than 73 new independent melts.
- All **295/295 precursor-history packets** exist. Final publication job 1027852
  failed at example rendering because it attempted to write a changed metric
  contract into an existing analysis revision. The packets remain preserved;
  `technical/finish.json` records the failure, and the durable history-supplement
  manifest has not been published. This is a history annotation of existing births.
- The separate million-atom Q/V recovery stopped at segment 26 after exceeding
  its declared agreement tolerances. The failed continuation is retained and
  cannot receive historical birth labels. Original million-atom data/results
  remain separate from this incomplete recovery.

Ni is independently complete: 24 melts, 110,384 frames and 11 established
liquid-origin PTM records, all at 1100 K. See the
[verified Ni collection and storage review](simulations/ni_births_010_20261007.md#october-8-completed-collection-and-search).
No jobs were stopped or rescheduled by this status review. Earlier dated entries
below retain their historical preparation and execution context.

**Retirement, October 7:** the original 150 native 0.75-ps Al production
histories are no longer available. Their arrays were deleted at user request;
preparations, melt restarts, ancestry and scientific results remain. The production
`final.restart.bin` files were also deleted at the user's subsequent request. Historical
0.75-ps studies below describe past use and are not new experiment inputs.
The consolidated 0.10/0.30-ps birth datasets and finer-cadence simulations are
unchanged. [Scope and verification](storage/al075_removal_20261007/README.md).

[Birth-focused V100 handoff](simulations/al_dense_shooting_v100_20261007.md#birth-focused-handoff-october-7-evening)
transfers 33 unstarted branches to continuous 0.10-ps Q/V from initialization and
an early cluster-growth follow-up stop. Three completed/four active original
branches retain their protocol. Both collection IDs share the same melt ancestry;
the transferred plans are not duplicated samples. New jobs retain the original
GPU deadlines. Birth extraction/catalogue updates remain unperformed; other Al
holdings keep their own completed/running/queued/unanalyzed status.

The [October 7 launch repair](simulations/al_dense_shooting_v100_20261007.md#missing-dependency-repair-october-7-evening)
preserves the failed import-only launch and restarts its 33 unstarted branches
with the missing scheduler helper included. All seven retained original branches
are now complete. Follow the birth-focused launch's `repair.json` pointer for
live jobs and progress; no source, seed, physical protocol or deadline changed.

The [seven-original-branch birth search](al_nuclei_lineage.md#seven-original-v100-branches-october-7)
completed at 01:03 Europe/Paris on October 8: 25,867 frames, native 0.10 ps,
six melt ancestors, all 28 source/detector tasks verified. PTM finds 17 established
liquid-origin roots, 12 reaching the growth threshold, and three separately
established subcritical composites. Of 4,911 small liquid-origin embryo records,
1,144 dissolve before establishment and 3,593 retain unresolved identity.
Cumulative-dataset publication remains pending. Initial unsaved equilibration is
explicitly censored. The 33 replacement branches are outside this search.

[Million-atom production velocity replay](simulations/al_million_velocity_replay_20261007.md)
adds a gated recovery workflow for the existing 400 ps source, with exact 0.10 ps
Q/V and native 4 ps anchors. It shares the original ancestry; birth catalogue
integration remains separate. The preceding melt and other Al campaigns retain
their existing scope and status.

[Additional V100 dense Al reshoots](simulations/al_dense_shooting_v100_20261007.md)
prepare 40 velocity branches from 20 existing melt parents. The October 7
expansion uses four V100 workers: the original lane and three additional lanes
with 23-hour limits, sharing the same 40 branches and passed numerical audit.
Exact 0.10 ps Q/V and original roles are retained. This adds production MD;
birth extraction and cumulative-dataset inclusion are not yet complete.

For new nucleus-birth experiments, use the [cumulative nuclei birth dataset](datasets/nuclei_birth.md):
`al-nuclei-birth-010` or `al-nuclei-birth-030`. It indexes completed compatible
searches across dated additions, retaining small failed embryos, exact frame maps,
ancestry and original roles. `python -m src.data.nuclei_birth refresh` updates the
same dataset. This event catalogue complements the older training-ready appearance
pool below; it applies no crystal-free history or control-sampling filter.
Since the October 7 cadence-contract change, `030` includes only uniformly saved
native 0.30 ps raw trajectories. Subsampled finer views and irregular shooting
prefixes are excluded from that ID; their upstream search results remain.
`010` retains its existing exact saved-frame population. The separate
[precursor-tail supplement](datasets/nuclei_birth_histories.md) annotates existing
births and is published only after its running collection is complete.

New: [12-epoch structural starts and derivative motion](structural_response.md) reuse the broad packed multimaterial training subset and the existing verified-MEAM 0.10 ps motion release. This adds derived future neighborhoods, not trajectories or independent samples; other Al collections retain their own coverage and status.


The separate [embryo-establishment collection](continuous_motion.md#present-selected-embryo-establishment) uses current 8–63 atom embryos from the 92 pinned velocity-bearing views. Common current inputs and 1/3/6 ps labels, original ancestry, spatial exclusion of existing crystals and explicit competing-fate censoring are retained. Collection is complete: 34,060 observations, including 12,409 pooled-test observations from 1,776 embryo families and 29 ancestors. The strict pre-birth benchmark remains unchanged.

The [October7 birth-data addition](al_nuclei_lineage.md#october-7-additional-nucleus-birth-data) analyses22 newly completed native0.10ps velocity-bearing production trajectories absent from the previous three releases. Extraction, tracking and collection are queued separately; no new MD or independent melt preparation. Birth counts and strict1/3/6ps eligibility remain pending.


The [continuous-motion campaign](continuous_motion.md) uses88 verified-MEAM,
velocity-bearing exact0.1ps views from84 Al64 ancestors, without nucleus or
neighborhood filters. Four unknown-potential historical views are explicit
exclusions. All176 new100fs tangent teacher banks are complete; nine1600-update fits have separate progress
receipts;0.01ps withheld times retain training ancestry. This is one matched
subset: relaxed, velocity-free, older shooting and other Al holdings retain
their original inventory and analysis status.

The [pooled birth evaluation](response_dense_birth.md#pooled-test-and-neighborhood-audit)
is an explicit user-authorized union of original validation, calibration and
test roles for this response-transfer assay. It preserves original source
roles, refits readouts with training-only selection and separately audits
neighborhood rejections; other Al studies retain their own splits.

The [0.10 ps response-to-birth diagnostic](response_dense_birth.md) completed a separate derived cohort from all 92 velocity-bearing 0.10 ps views in the three pinned v3 releases, including older partial archives and exact views of finer histories. All three label releases and 153 separate-role readouts are complete; Al64 ancestor roles and overlapping events remain explicit. Seven million-atom views and one older 100k-atom view lack velocities and are excluded from this matched Q/QV comparison; historical pool rows remain available.

The [revised birth and fate search](al_nuclei_lineage.md) reprocesses the frozen
all-Al cadence catalogue with corrected lineage rules. It is a versioned event
audit; it does not overwrite existing predictive cohorts or their labels.
The [October 5 additions](al_nuclei_lineage.md#new-dense-observations) separately
extract newly available native 0.10 ps data and paired 0.30 ps observations.
Their ancestry and prior-release overlap remain explicit.
The [October 6 production addition](al_nuclei_lineage.md#october-6-verified-meam-production-addition)
verifies executed native MEAM commands, potential hashes and prepared-liquid
ancestry before searching new complete sources. Recovery and control simulations
remain separate holdings and are explicitly excluded from this production cohort.

An unspecified **Al** request means **all Al data and related experiments**.
This includes original trajectories, shooting branches, million-atom runs,
uniform-temperature birth campaigns, dense 0.1/0.01 ps descendants, static data
and relaxed views. A finished appearance study does not mean all Al work is
finished. This scope is also recorded in [AGENTS.md](../AGENTS.md).

The pool combines existing **training-ready observations and their labels**,
with an inventory of every registered Al collection. Collections without a
sealed appearance dataset remain visible as available inputs requiring further
eligibility work; they are not silently treated as analyzed or as negative cases.
Reported H200 copies remain marked as remote/unverified copies, not additional
independent data. Collections with unknown material metadata are listed for review.

The [new full-cell nucleus search](al_nuclei_search.md) inventories the same Al
holdings and searches exact 0.75, 0.10 and 0.30 ps views, including stride-three
0.30 ps views of 0.10 ps trajectories. Its unfiltered cluster catalogue remains
separate from these historical prediction-eligibility labels.

## Location and refresh

The [configuration](../configs/al_research_pool.json) uses
`${storage:training_storage}/al_research/pool-v1` for the release manifests,
overview and status, and `${storage:cache}/al-research-pool/v1` for pooled numeric
inputs. Both locations are outside the repository. The generated `README.md`
is the combined overview; `current.json` points to an immutable release under
`releases/`. Freeze that release identity in future experiment configurations.
Historical datasets, source roles, models, predictions and metrics are unchanged.

With `pointnet-torch214`:

```bash
python -m src.data.al_pool refresh
python -m src.data.al_pool status
```

`python -m src.data.al_pool submit-refresh` schedules two small detached CPU
publication jobs after the recorded dense-data sealing and evaluation stages.
It preserves the existing scientific queue and freezes its own publisher code.

`refresh` verifies upstream files, joins the labels, writes physical descriptor
banks and the release, and records scheduler/study status. Repeating it reuses
unchanged banks. A newly sealed native 0.1 ps dataset enters a new release;
partly written inputs are never used. `status` only updates the live status file.
The inventory uses the latest registry scan; refresh that separately when needed:

```bash
python scripts/project.py datasets --refresh
```

## One record, multiple labels and paired views

Each observation has a physical `sample_id`, `source`, `ancestor`, parent hash,
original source role, material/potential, native label cadence, exact observation
offsets, raw/relaxed domain, event stratum, matched-event and case/control-pair IDs,
descriptor-bank row, and upstream provenance. Temperature is retained **only as
audit/filter metadata**. Conditions, timestamps, source IDs and outcome metadata
are never concatenated into model inputs.

The labels are separate fields:

- `appearance`: any eligible observed crystal appearance is positive, including
  failed embryos; continuously liquid matched controls are negative.
- `fate`: the original categorical outcome, including unresolved/censored endings.
- `dissolution`: 1 for verified `dissolved` or `grew_then_dissolved`, 0 for
  `continued_growth`, and null for controls or unresolved events. Continued growth
  means documented growth under the original follow-up definition, not immortality.

Raw and relaxed views share a `sample_id` and labels. The older histories replayed
in the shooting study are included once, rather than becoming extra examples.
The event inventory also retains screened-out audited candidates, with an empty
`eligible_domains` list. An event may exist without sufficient crystal-free history
and liquid controls for this particular predictive assay.

## Select a cohort for training or evaluation

The current input banks contain the existing 442 geometry, bond-order, CNA and TDA
descriptors. A selection exports `features.npy` with shape **N × frames × 442**,
`targets.npz` with labels, event weights, ancestors and original roles,
`rows.json` with metadata, and a checksum-bound `manifest.json`.
Raw trajectory/coordinate provenance remains in the pool; this command does not
compute new MACE embeddings or turn unprocessed trajectories into descriptors.

Choose a physical observation protocol and one input domain explicitly:

| Protocol | Offsets before appearance | Eligible prepared sources |
|---|---|---|
| `legacy075` | −6 to −0.75 ps, every 0.75 ps; 8 frames | Earlier Al, raw or relaxed |
| `common150` | −6, −4.5, −3, −1.5 ps; 4 frames | Earlier + shooting Al, raw; earlier relaxed |
| `dense010` | −0.6 to −0.1 ps, every 0.1 ps; 6 frames | Native dense descendants, after sealing |

Common 1.5 ps observations are selected from existing frames exactly. Their label
detection grids remain explicitly 0.75 ps for earlier sources and 0.30 ps for
shooting. Native 0.1 ps labels are independently searched. No interpolation,
cadence relabeling or mixing histories with different physical offsets occurs.

For example, using an output directory on IDS/WORK outside the repository:

```bash
# Earlier + shooting cases and controls, from the same master pool.
python -m src.data.al_pool select --protocol common150 --domain original \
  --task appearance --output /path/outside/repo/al-common150-appearance

# The same observation protocol, resolved nuclei only; dissolution is positive.
python -m src.data.al_pool select --protocol common150 --domain original \
  --task dissolution --output /path/outside/repo/al-common150-fate

# Shooting-only transfer evaluation, or a paired relaxed cohort.
python -m src.data.al_pool select --protocol common150 --domain original \
  --where pool=shooting --output /path/outside/repo/al-shooting
python -m src.data.al_pool select --protocol legacy075 --domain relaxed \
  --where historical_role=train --output /path/outside/repo/al-relaxed-train
```

`--where FIELD=VALUE` filters any row metadata field. Repeating a field accepts
either value; different fields are combined with AND. Examples include `stratum`,
`fate`, `ancestor`, `dataset`, `pool` and `label_cadence_ps`. Numbers and booleans
use JSON syntax. An empty or unavailable selection fails explicitly. Add
`--release /absolute/path/to/releases/IDENTITY.json` to pin a past pool release.
Existing export directories are never overwritten.

## Splits, counts and interpretation

Always split/group by **ancestor**, keeping shooting branches, dense descendants
and paired coordinate views together. Existing historical roles are retained and
checked for ancestor conflicts. The old shooting-study outer CV assignment is
recorded separately; missing assignments for new ancestors are null, not invented.
This pool does not resplit the fixed Al64 release or select new hyperparameters.

Selection recomputes equal total weight per matched event within the selected
task. For fate, only resolved positive nuclei contribute. Controls inherit their
matched event's stratum for cohort filtering, but have no positive `event` label.
Raw/relaxed views, multiple centers, branches and overlapping windows are not new
independent events. Inventory collection counts also include containers and derived
views; never sum them as independent trajectories or as total unique bytes.

These remain retrospective case/control data. An apparent 1:4 balance is a sampling
design, not a natural nucleation probability. Existing metrics keep their original
cohort definitions; joining inputs does not manufacture a pooled result or imply
that a pooled model has been trained.

## Related results and coverage

- [Earlier appearance and relaxed comparisons](birth_appearance.md)
- [Shooting appearance and growth/dissolution](al_shooting_prediction.md)
- [Native 0.1 ps preparation and prediction](birth_dense010.md)
- [Dense simulation campaign](simulations/al_main_half_stop_20261001/README.md)
- [All registered datasets](datasets/index.html), including million-atom and
  other Al collections not yet converted into this appearance assay
- [Encoder research history](encoder_research/README.md)

The generated overview links final receipts for completed studies and shows missing
receipts, the live scheduler, and per-run dense simulation states. Completion is
reported per study; all-Al reporting also includes pending preparation and remaining
collections, even when the latest descriptor comparison has finished.
