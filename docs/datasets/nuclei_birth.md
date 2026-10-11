# Al nuclei birth dataset

Use **`al-nuclei-birth-010`** for exact **0.10 ps** observations and
**`al-nuclei-birth-030`** for **native 0.30 ps saved trajectories only**. The latter
excludes subsampled 0.10 ps (and other finer) data, and mixed-cadence shooting
prefixes. Every referenced raw timestep array is checked. These stable registry
IDs combine the completed, compatible v3 searches into one cumulative catalogue.
Future experiments can request either ID without listing dated additions.

**October 8 incorporation:** the seven completed original V100 branches are now
published under fingerprint
`468d21ad04a4016be1889e709aa16c2c158765a063167354e33ea285d63d256a`.
The exact `010` catalogue has **129 source views, 110 recorded ancestry groups,
473,416 frames and 312 established liquid-origin PTM records**. The native-only
`030` population retains 680 views / 150 established PTM records. Existing event
identities are retained. The [78-source GPU addition](../al_nuclei_lineage.md#october-8-completed-data-incorporation)
is running separately and will publish only after completed extraction/tracking
checks. Its queued publication will create a new fingerprint; running studies
retain their existing pins. Supplement cohorts do not automatically expand.

The preceding October 7 native-cadence update was published with fingerprint
`7720ac6b92886adb29013beb4e28b78e7def6ba16e9a175066b6574a4c27380d`:

| Dataset | Source views | Recorded ancestors | PTM established liquid-origin birth records |
|---|---:|---:|---:|
| `010`: exact saved 0.10 ps | 122 | 109 | 295 |
| `030`: uniformly saved native 0.30 ps | 680 | 20 | 150 |

The 0.30 ps population consists of shooting descendants; source-view and birth
counts are not independent melt counts. This update excludes 147 finer-source
views and 144 nested shooting histories whose raw prefixes have mixed/finer
spacing, out of the previous 971-view `030` population. Their upstream data and
labels remain. The 0.10 ps source rows, tracking rows and established-birth rows
are unchanged, and all 110-source/295-event history bindings were verified.
The parent fingerprint changes because the published 0.30 ps population changed;
frozen running jobs retain their original pins and code.
[Verification](../../output/nucleus_detection/native030-contract-20261007/technical/verification.json)
and [raw-cadence audit](../../output/nucleus_detection/native030-contract-20261007/technical/source-audit-before.json).

The dataset contains source/frame mappings, birth and fate labels, atom identities
at appearance, ancestry, original split roles and provenance. Original MD arrays
and detailed tracking shards remain referenced in their existing storage. Small
dissolved embryos are retained. No history windows, descriptor bank, relaxed
coordinates, liquid controls or new prediction labels are fabricated by merging.

## Use in new experiments

In `pointnet-torch214`, from the repository root:

```python
from src.postprocessing.datasets.births import NucleiBirthDataset

with NucleiBirthDataset("010") as data:
    # Record the current dataset identity in the consuming experiment.
    data.pin("output/my_experiment/technical/nuclei_birth.json")
    print(data.fingerprint)
    births = list(data.events(selection="established", detector="ptm"))
    event = data.event(births[0]["event_id"])
    source = data.source(event["source_id"])
    # event["first_members"] contains actual atom IDs, not array indices.
    # source["views"]["010"] indexes source["frames"] and source["times_ps"].
    # source["frames"][event["first_frame"]] = [part_index, raw_frame_index].
```

To require that a prepared experiment still uses the same data:

```python
with NucleiBirthDataset("010", expected_fingerprint="RECORDED_FINGERPRINT") as data:
    births = list(data.events())
```

Use `"030"` for the other cadence. An open reader keeps its database snapshot;
another update cannot change its rows. Dataset paths resolve through the normal
`${dataset:al-nuclei-birth-010}` and `${dataset:al-nuclei-birth-030}` configuration
tokens. Each cadence directory contains a manifest pointing to the common SQLite
index. The dataset root also contains readable source, ancestry, established
birth and count CSVs under `analyses/catalogue/tables/`.

## Select the scientific target explicitly

**Pre-birth crystalline tails belong to existing births.** The
[`birth-tails-010-v1` history supplement](nuclei_birth_histories.md) reconstructs
up to 12 ps before each established birth's first eight-atom appearance, including
one-atom crystalline components. Shared atom IDs connect earlier components to
that birth; weak links, gaps and lookback censoring remain explicit. The original
birth event and its time are unchanged. This is retrospective history annotation,
not an additional population of births or an input feature for prediction.
The 295-record collection is still running as of this contract update; completed
working packets are not a sealed dataset. `histories/index.json` exposes it only
after final publication. There is currently no corresponding 0.30 ps tail supplement.

The separate **`first-crystal-010-v1` additional label set** records the next
first PTM-crystalline atom in prospectively sampled local observations, including
transient appearances, with a same-atom three-frame persistence variant. It is
stored permanently alongside this dataset and joined by source/frame/center
atom. It is distinct from the eight-atom `appearances` event selection below.
See [additional-label storage and API](nuclei_birth_labels.md). One publication
runs after label collection is sealed; unprocessed observations never become
negative labels. The original birth catalogue fingerprint is unchanged.

| Selection | Meaning |
|---|---|
| `established` | Liquid-origin root reaches 64 atoms for 1.5 ps |
| `appearances` | Every liquid-origin root reaching eight atoms, including failed embryos |
| `dissolved` | Confirmed dissolution before independent establishment |
| `all` | All recorded origins, composites and ambiguous events for audits |

The default is `established` with `ptm`. Other available detectors are `coherent`,
`coherent_loose` and `coherent_strict`. `source_origin="shooting"` selects shooting
descendants explicitly; `"original"`, `"historical_partial"` and
`"external_continuous"` retain their producer meanings. Source manifests include
shooting-parent selection/protocol metadata and generating-potential provenance.

`event()` returns the full checked upstream record: appearance interval and atom
IDs, establishment/growth milestones, origin, fate and censoring information.
`track`, `founder_families` and `receiver_family` refer to family IDs **within that
event's task**, not globally. The database `tasks` table records locations and
checksums of interactions, size histories and full-cell structural summaries.
The dataset leaves unresolved outcomes unresolved.

For **pre-crystal prediction**, define and record spatial support, observation
history, lead time, crystal-free eligibility and control sampling in the consuming
experiment. Birth detection itself does not enforce a crystal-free neighborhood.
Do not use future event fields, temperature, absolute time or simulation age as
predictor inputs. Timestamps organize exact observations and labels only.

## Update with new data

```bash
python -m src.postprocessing dataset refresh
python -m src.postprocessing dataset status
python -m src.postprocessing dataset verify --upstream
```

`refresh` discovers registered datasets matching the configured v3-search IDs,
verifies completion and scientific compatibility, indexes new tasks and updates
both cadences under one publication lock. Previously indexed tasks are reused.
Repeating it with unchanged searches/code/config retains the same fingerprint and
event IDs. Incomplete searches appear in status and
cannot become partial dataset rows.

**New raw MD must first complete the existing structural extraction and v3 birth
search.** Register it and use the
[verified production inventory and lineage workflow](../al_nuclei_lineage.md)
with the same criterion. Register the search as `al-nuclei-dense-additions-*`;
the next `refresh` incorporates it automatically. This command publishes existing
labels; it does not submit MD, restart searches or modify another chat's jobs.
New scientific criteria require a distinct dataset protocol, not a silent update.

The root dataset `al-nuclei-birth` lives outside the repository at
`${storage:training_storage}/nuclei_birth/al`. `README.md` gives current counts;
`status.json` lists registered Al coverage. There is one current dataset, without
an archive of earlier dataset versions or migration of previous experiments.
New experiments save their fingerprint and selected event/source IDs in their own
cohort manifests. Dated search folders remain provenance, not the user interface.
See [configuration](../../configs/data/nuclei_birth.json) and
[exact definitions](../metrics/nuclei_birth_dataset.md).

## Shared ancestry and scope

Exact duplicate input identities are indexed once. Alternative detectors,
cadences, shooting branches and historical prefixes remain distinguishable;
they must not be counted as independent nuclei or independent melt preparations.
The 0.30 ps dataset now contains only uniformly saved native shooting observations.
Finer sources remain under their own suitable protocols; their old 0.30 ps search
results remain upstream for explicit cadence-comparison studies. They are not
returned by `NucleiBirthDataset('030')`. The two published datasets are different
source populations, so their raw totals do not measure a cadence effect.
The 0.10 ps dataset still includes exact views of native 0.01 ps data, explicitly
recorded in `native_cadence_ps`; this update does not change that population.
See `analyses/catalogue/tables/source-eligibility.csv` for every selection decision.

Original source roles are preserved. Older shooting collections contain shared
ancestors under different historical role names: `role_conflict` flags these.
Do not concatenate their historical splits into a new training/test split;
freeze a grouped protocol first. The established fixed-Al benchmark remains
unchanged. This catalogue has no newly assigned benchmark split.

The original 0.75 ps campaign is retired; its retained results describe a separate
historical protocol. Unsearched
production data and paired static/relaxed holdings are visible in coverage;
publishing this dataset does not mean all Al data have been analyzed. Unknown
potential or ancestry metadata stay unknown, and missing velocity arrays do not
become inferred velocities.
