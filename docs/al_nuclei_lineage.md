# Revised Al embryo birth and fate search

Current code and commands: [shared post-simulation module](postprocessing.md).
Dated execution receipts below retain their recorded implementations.

## October 8 completed-data incorporation

The main dense campaign is now complete, 152/152 trajectories. The new search
contains the 45 completed trajectories absent from all four earlier extraction
plans, plus all 33 completed birth-focused velocity branches. Production audits
verify 165,828 native 0.10-ps frames (16,575 ps summed observed span), exact
timelines, prepared-liquid and potential checksums, original roles and ancestry,
and absence of prior input identities. The main group has 99,495 frames from
45 ancestors; the focused group has 66,333 frames from 19 ancestors. These
ancestor sets overlap and are not new independent melts relative to prior data.
No neighborhood/crystallinity selection is added. The full saved histories are
searched; online simulation stopping proxies do not become birth labels.

Recipe: `configs/analysis/al_nuclei_gpu_additions_20261008.json` through
`python -m src.postprocessing al-gpu-search`. GPU array **1029088** has two
A100 workers, followed by four CPU tracking lanes **1029089**, collection
**1029090**, and automatic publication **1029091**, all with `afterok`
dependencies. The queue is detached and reuses the passed GPU numerical audit.
The GPU extractor is `resident`/`voronoi`, with separate execution provenance;
the chronological tracker and operational criteria are unchanged. There are
312 source/detector tasks, not 312 independent samples.

Output: `${storage:training_storage}/nuclei_search/al-additions-20261008-gpu-v3`.
`technical/gpu-launch.json` records this chat's jobs and frozen code;
`gpu-0.json`/`gpu-1.json` expose extraction progress, `track-*.json` tracking,
and `publication.json` the final publication outcome. A failure leaves subsequent
stages blocked. Publication verifies all previous established records and tasks,
the native-only 0.30-ps population and inclusion of every new source.

The seven-original-branch result below is separately registered as
`al-nuclei-dense-additions-v100-seven-20261007` and incorporated without rerunning
its completed CPU extraction or tracking. Precursor histories and local
first-crystal labels remain separately fingerprinted supplements; this queue
does not expand their cohorts. The failed million-atom velocity replay remains
outside historical birth-label reuse.

## Seven original V100 branches, October 7

The requested analysis is limited to the seven completed original branches of
`al-dense-shooting-v100-20261007`. It retains all 25,867 saved full-cell frames,
70,304 atoms per frame, at native 0.10 ps spacing. The seven trajectories share
six melt ancestors; the two source0979 shots are related. Other Al campaigns
and the 33 birth-focused replacement branches remain separate populations.

| Source and shot | Temperature (audit only) | Saved frames | Measured duration, ps |
|---|---:|---:|---:|
| source0887-shot00 | 400 K | 6001 | 600 |
| source0897-shot00 | 400 K | 5611 | 561 |
| source0908-shot00 | 400 K | 4111 | 411 |
| source0926-shot00 | 450 K | 1411 | 141 |
| source0933-shot00 | 450 K | 2911 | 291 |
| source0979-shot00 | 510 K | 4111 | 411 |
| source0979-shot01 | 510 K | 1711 | 171 |

The scoped inventory verifies original launch input checksums, both potential
files, prepared-liquid ancestry, executed GPU callback and native NPT commands,
the completed execution and conversion receipts, and the exact timeline. Forces
are float32, integration is float64, and stored observations are float16. The
existing passed numerical audit is referenced; no validation MD is rerun.
Initial equilibration lasts 15 ps and was not saved in these original branches.
Already-present embryos and unknown early onsets remain explicit. Ending a
trajectory before a fate is resolved does not make the embryo a failure.

The unchanged four structural definitions (PTM, loose/reference/strict coherent
bond order) feed the v3 descendant-family search. An embryo first reaches eight
connected atoms; establishment and growth require at least 64 and 128 atoms,
respectively, for 1.5 ps. The output separates liquid emergence, interface
emergence, parent-present embryos, dissolution before establishment, later
dissolution, absorption, coalescence, surviving families and unresolved fates.
Four detector views do not count as four independent physical births. There is
no model training, conditioning input, new relaxation or new simulation.

```bash
python -m src.postprocessing inventory --config configs/analysis/al_nuclei_v100_seven_extract_20261007.json
python -m src.postprocessing search prepare --config configs/analysis/al_nuclei_v100_seven_20261007.json
python -m src.postprocessing search submit --config configs/analysis/al_nuclei_v100_seven_20261007.json
```

Submitted from this chat: extraction array **1027835** (four lanes, eight
single-thread workers each), dependent tracking **1027836** (28 source/detector
tasks) and collection **1027837**. All four extraction lanes started. Execution
uses frozen code and `pointnet-torch214`; GPU simulations continue independently.
All 1,619 extraction chunks and 28 tracking tasks completed. Collection finished
at **2026-10-08 01:03 Europe/Paris**, about 1 hour 55 minutes after workers started.
On October 8 all 28 per-task receipts and their saved file hashes were checked;
the collector contains 167,795 records across four overlapping detector views.
That total includes small embryos, uncertain histories and composite families;
it is not a physical nucleus count.

PTM finds **17 liquid-origin root families** that reach at least 64 connected
atoms for 1.5 ps. Twelve reach at least 128 atoms for 1.5 ps before major merger;
14 of the 17 established roots ultimately coalesce, and three remain separate at
the endpoint. Three additional composite families first establish by joining
subcritical embryos; they remain a separate mechanism, not extra root births.
The one established parent-present family is also excluded from new liquid births.

| Source and shot | Established liquid-origin roots, PTM | Peak connected PTM cluster, atoms |
|---|---:|---:|
| source0887-shot00 | 3 | 6963 |
| source0897-shot00 | 4 | 51504 |
| source0908-shot00 | 2 | 54711 |
| source0926-shot00 | 1 | 54601 |
| source0933-shot00 | 5 | 29721 |
| source0979-shot00 | 0 | 15 |
| source0979-shot01 | 2 | 51041 |

There are 4,911 PTM liquid-origin embryo appearances (at least eight atoms).
Of these, 1,144 demonstrably dissolve before establishment, 3,593 have unresolved
identity, 115 are absorbed before establishment and 30 coalesce before establishment.
Unresolved identity and endpoint censoring are not confirmed failures. The large
unresolved fraction limits interpretation of small-embryo survival probabilities.

Reference and strict coherent-bond definitions find 19 and 16 established roots,
respectively. The loose definition finds eight roots and 21 establishments through
subcritical composites, versus PTM's 17 and three. Thus the count and attribution
of formation versus coalescence remain detector-sensitive; similar aggregate
counts do not prove event-by-event agreement.

The paired source0979 shots at 510 K are a useful distribution-of-futures example:
shot00 never establishes within its 411 ps saved window, while shot01 first
establishes a composite at 65.1 ps and a liquid-origin root at 65.6 ps, reaching a
peak 51,041-atom cluster by the end of its 171 ps window. They start from the same
prepared liquid with different velocity seeds. These times use the saved
measurement clock, after 15 ps unsaved equilibration. Two outcomes demonstrate
variability but cannot estimate a reliable nucleation probability.

Results root: `${storage:training_storage}/nuclei_search/al-v100-seven-20261007-v3`.
Its `technical/launch.json` records job ownership; `technical/extract-*.json`
records live progress. The sibling `al-v100-seven-20261007-extraction` contains
the immutable population and production audit. The collector exports the final
`analyses/catalogue-v3/tables/` event, count, source-growth and detector-comparison
tables with `METRICS.md` and implementation hashes. Per-source technical records
retain all merger interactions and family-size histories. This release has not
yet been added to the cumulative Al birth dataset.

The inventory extension only adds explicit population selection and GPU producer
provenance checks. Structural extraction and lineage calculations are unchanged;
the updated metric contracts record the new inventory and population description.

## Original revised search

For new experiments, consume the [cumulative nuclei birth dataset](datasets/nuclei_birth.md)
through `al-nuclei-birth-010` or `al-nuclei-birth-030`. Its update command collects
completed compatible searches; dated runs below remain producer provenance.

This rerun fixes the earlier catalogue's suppression of growth after any split
or merge. Birth origin is recorded once. Ordinary descendant fragmentation and
minor absorption allow growth to continue; substantial coalescence, absorption
into an existing crystal and dissolution remain separate outcomes.

The [metric definitions](metrics/al_nuclei_lineage.md) declare all numerical rules,
including how composite families preserve earlier births without creating extra
ones. The [configuration](../configs/analysis/al_nuclei_lineage_20261005.json)
reuses the entire frozen [earlier all-Al search](al_nuclei_search.md), including
all exact 0.75/0.10/0.30 ps views, million-atom data, shooting, old trajectories,
dense descendants and historical partial records. Frame classifications are reused.
New or ongoing simulations outside that release remain not yet analyzed here.
Raw MD determines outcomes; relaxed views do not become independent events.

Use the `pointnet-torch214` environment:

```bash
python -m src.research.crystallization_origin.lineage_search prepare --config configs/analysis/al_nuclei_lineage_20261005.json
python -m src.research.crystallization_origin.lineage_search submit --config configs/analysis/al_nuclei_lineage_20261005.json
```

Eight independent CPU lanes each run eight workers, using the prior measured
costs to distribute tasks. There is no barrier between groups of lanes. A final
collection job verifies all completed results, publishes cadence/detector matches
and creates a source-level growth audit. It runs after tracking terminates and
fails explicitly if any result is incomplete. No training or online W&B run is
created for this event audit.

Results live at `${storage:training_storage}/nuclei_search/al-all-20261005-v3`.
`technical/launch.json` identifies this chat's jobs, `technical/plan.json` binds
the source population and implementation, and `technical/track-*.json` records
lane progress. The final catalogue is `analyses/catalogue-v3`. The earlier release
is preserved, including its original thresholds and interpretation limitations.

The October 5 launch is tracking array **1022311** (eight jobs, 64 CPU workers)
and dependent collection job **1022312**. These job IDs are recorded in this
chat's launch receipt. All eight tracking lanes started successfully. Other
chats' simulations or helper jobs are outside this launch's ownership.

Manual verification on actual source 0863 recovers the 65,487-atom growing
lineage: the PTM 0.10 ps establishment crossing is 42.2 ps, confirmed at 43.7 ps.
Its subsequent interactions remain recorded. This is an implementation check,
not proof that every new lineage classification is physically correct.

## New dense observations

The separate October 5 addition inventories newly available native 0.10 ps Al
trajectories against the previous frozen source paths and array identities.
It includes exact stride-three 0.30 ps views. Both views use the same four
structural detectors and the revised lineage criterion described above.
Current input arrays are verified against their producer checksums before use.
Only completed production MD enters this addition. Nucleus-recovery branches,
selected-state continuation controls, seeded-growth controls and CPU/GPU engine
comparisons are explicitly excluded by scientific purpose. Atom count is not a
selection criterion. Archived partial production and incomplete runs are deferred;
existing historical releases retain their original populations. Preparation
ancestry remains explicit; these trajectories are not automatically new melts.

```bash
python scripts/project.py datasets --refresh
python -m src.research.crystallization_origin.search prepare --config configs/analysis/al_nuclei_dense_additions_extract_20261005.json
python -m src.research.crystallization_origin.lineage_search prepare --config configs/analysis/al_nuclei_dense_additions_20261005.json
python -m src.research.crystallization_origin.lineage_search submit --config configs/analysis/al_nuclei_dense_additions_20261005.json
```

The launch inventory refreshed all 28 new completed production collections and
verified their binary headers and producer metadata. It preserved the prior full
inventory date for other holdings and records the refresh scope in
`technical/inventory/combined-registry.json` under the extraction root.
The complete general inventory refresh was interrupted while scanning unrelated
archived datasets; the launch uses this explicit targeted refresh.

The addition uses twelve extraction lanes with four workers each, followed by
four logical tracking lanes and a final collector. Slurm accepted extraction
array **1022341** with 48 CPU workers. Its submission quota rejected the four-job
tracking array, so dependent job **1022354** runs those lanes sequentially with
four CPU workers; **1022355** collects the results. The receipt retains the
original rejection and this scheduling change. Scientific tasks are unchanged. Collection
checks the saved results; missing or changed inputs fail explicitly. Extraction
and tracking run detached from a frozen source copy. These are CPU analyses,
with no training or W&B run. The earlier eight-lane rerun continues independently.

Results: `${storage:training_storage}/nuclei_search/al-dense-additions-20261005-v3`.
The extraction plan is in the sibling `al-dense-additions-20261005-extraction`
directory, and component caches are under
`${storage:cache}/nuclei-search/al-dense-additions-20261005`.
The input inventory records excluded, incompatible, unavailable and unprocessed
collections. Still-running simulations need a later immutable population release;
this addition does not claim all Al work is complete.

This addition contains **28 completed production trajectories, 129,028 frames**:
source0886–source0909, source0914, source0944, source0974 and source1005.
All have the recorded `al_dense_half_stop_010ps_v1` producer protocol, complete
outcomes and exact 0.10 ps timelines. The production audit checks all 28 against
the earlier frozen input identities and confirms exclusion of
`al-nucleus-recovery-20261004`, `al-nucleus-controls-20261005`,
`meam-gpu-blackwell-20261004` and `meam-md-al16k-450K-200ps-20261004`.
Quota-recovered execution of a production run is distinguished from deliberately
restarting a selected nucleus. No diagnostic continuation enters this population.

## October 5 evening recovery and status

All 4,456 earlier-population tracking tasks now have completion receipts. One
0.30 ps shooting/reference-bond-order task failed while the live dataset catalog
was temporarily unreadable JSON. It was rerun with the exact frozen scientific
code and a machine configuration pointing at the frozen catalog. Original
failure records remain intact. Collector **1022874** replaces failed collector
1022312; ownership and recovery evidence are in
`technical/recovery-20261005-registry/receipt.json` and `technical/launch.json`.
Final event-table publication was still running at this status check.

Receipt-level counts are frozen in `technical/status-20261005-counts.json`.
At 0.10 ps, PTM/reference bond order find 82/75 established liquid-emergence
records. This includes related historical views: the 26 main completed dense
production trajectories contribute 28/25, and the million-atom trajectory
contributes 37/37. These are operational lineage records, not independent
samples or already eligible prediction examples. The final collector verifies
artifact checksums and publishes the detailed tables and comparisons.

The new 28-production-run addition was approximately 81% through structural
extraction at 20:02 Europe/Paris. Birth tracking had not started. Its pending
stages were replaced with otherwise identical jobs that use the frozen catalog:
tracking **1022875**, collection **1022876**. Prior pending jobs 1022354/1022355
were cancelled only after replacements were accepted; running extraction
1022341 was left intact. See `technical/registry-pinning-20261005/receipt.json`.

## October 6 verified MEAM production addition

New completed production sources enter a separate immutable release after both
earlier source populations are excluded. `production_inventory` verifies the
actual input deck and executed equilibration log use `pair_style meam`, with
the declared Lee–Shim–Baskes 2003 Al library/parameter files and matching SHA-256
hashes. Runs start from the original prepared liquid, whose data and native melt
restart checksums must match the saved parent provenance. The full-cell NPT
protocol uses 2 fs integration and exact 0.10 ps observations. Complete outcome,
executed completion marker, conversion receipt and integer timeline must agree.

Selected-nucleus recovery, continuation controls, seeded growth, force-response
and engine validation experiments are excluded by scientific purpose. A source's
held-out `test` split or `final_validation` parent name remains valid production
data; it is not an implementation-test simulation. Melt ancestry, temperature
and original source role remain audit metadata only. No predictive model is fit.

```bash
python -m src.research.crystallization_origin.production_inventory --config configs/analysis/al_nuclei_dense_additions_extract_20261006.json
python -m src.research.crystallization_origin.lineage_search prepare --config configs/analysis/al_nuclei_dense_additions_20261006.json
python -m src.research.crystallization_origin.lineage_search submit --config configs/analysis/al_nuclei_dense_additions_20261006.json
```

The first command refreshes producer metadata and binary headers for the newly
verified production sources through the shared dataset registry implementation.
It records all registered Al exclusions and previous-release identities without
rescanning unrelated archived arrays. Its audit lives in
`${storage:training_storage}/nuclei_search/al-dense-additions-20261006-extraction/technical/inventory/production-audit.json`.
The extraction plan binds this inventory and is checked against prior input
identities; native 0.10 ps and exact stride-three 0.30 ps views share classifications.
All four detectors and all v3 lineage/persistence thresholds are unchanged.

Results live at `${storage:training_storage}/nuclei_search/al-dense-additions-20261006-v3`,
with disposable structural caches under `${storage:cache}/nuclei-search/al-dense-additions-20261006`.
Sixteen extraction lanes use four workers each, followed by four tracking lanes
with four workers each and a collector. All stages read the frozen catalog and
machine snapshot, avoiding the mutable-registry failure seen in the prior run.
No other chat's simulations or jobs are changed. Sources finishing after this
inventory snapshot need a later release; earlier catalogues remain intact.

The sealed October 6 population has **29 production trajectories, 77,399 frames
and 7,737 ps of summed trajectory duration**: sources 0910–0913 and 0915–0939.
Four sources are at 400 K and 25 at 450 K. Original roles remain 17 train,
3 selection, 3 calibration and 6 test; these are not additional independent
ancestors relative to their older source trajectories. Both potential checksums
and liquid-start provenance passed for every source. No nucleus recovery or
diagnostic run entered this population.

Owned launch jobs are extraction **1023717** (16 lanes, 64 CPU workers),
tracking **1023718** (four lanes, 16 CPU workers) and collection **1023719**.
The launch receipt binds the frozen scientific code, catalog and machine settings.
The 232 tracking tasks apply four detectors to both cadence views of all 29 runs.

## October 7 additional nucleus-birth data

The user requested more birth examples for the1/3/6ps frozen-encoder evaluation.
The new immutable production release contains22 completed trajectories absent
from all three previous search populations: sources0940–0943,0945–0961 and0963.
There are43192 native0.10ps frames and4317ps of summed duration, with positions
and native velocities. Original roles are17 training,1 validation and4 test.
Four trajectories are450K and18 are500K; temperatures remain audit metadata.
These descendants preserve the original Al64 melt ancestry. This is additional
analysed data, not22 newly prepared independent melts.

The native-MEAM provenance, prepared-liquid and potential checksums, executed
NPT commands, completed conversion and exact timeline audit passed for all22.
Older inputs, selected-nucleus recovery/controls and still-running MD are excluded
from this release. Four detectors and the unchanged v3 lineage/establishment
rules run on exact0.10ps and matched0.30ps views. Birth counts and strict1/3/6ps
eligibility remain unknown until the search completes. No threshold is relaxed.

```bash
python -m src.research.crystallization_origin.production_inventory --config configs/analysis/al_nuclei_dense_additions_extract_20261007.json
python -m src.research.crystallization_origin.lineage_search prepare --config configs/analysis/al_nuclei_dense_additions_20261007.json
python -m src.research.crystallization_origin.lineage_search submit --config configs/analysis/al_nuclei_dense_additions_20261007.json
```

Jobs owned by the response/nucleus-recovery chat: extraction array1025453
(eight lanes,four CPU workers each), tracking array1025454 (four lanes,four
workers each), collector1025455. The frozen catalog, machine settings and launch
receipt are under `nuclei_search/al-dense-additions-20261007-v3/technical` in
training storage. The separate extraction plan and provenance audit are under
`al-dense-additions-20261007-extraction`. No existing MD or other-chat job was
stopped or changed. Based on the previous search's measured runtimes, allow
roughly5–6hours of execution plus queue time for this allocation of CPU workers.
The previous nucleus catalogue and completed prediction analysis remain intact.
