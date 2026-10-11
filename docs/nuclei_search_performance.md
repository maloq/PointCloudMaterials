# Nuclei birth and tail search performance

Current code and commands: [shared post-simulation module](postprocessing.md).
Dated execution receipts below retain their recorded implementations.

Execution audit: 2026-10-07, `pointnet-torch214`, node60, one computation thread
per process. This changes execution, not the nucleus definition or observation
cadences. The current dataset remains `al-nuclei-birth-010` / `030`.

## Changes

1. **Resolve only aliases present in the requested path.** Previously each path
   lookup expanded every dataset alias, repeatedly parsing the machine YAML and
   dataset catalogue. Birth-tail classification did this again for every frame.
   Output and cache paths are now also cached within each tail worker.
2. **Read coordinates only when a new birth record needs them.** The lineage
   tracker still processes every frame's component membership. Coordinates and
   the established-crystal neighbor tree are needed for an onset's centroid and
   distance to existing crystal. They are now constructed only on such frames.
   Established-crystal membership is captured before current-frame milestones,
   preserving the original ordering. The full-cell structural extractor still
   checks coordinates and input checksums independently.
3. **Cooperative tail workers.** `birth_histories assist` tries each unfinished
   event's existing exclusive lock without waiting. Busy and completed events
   are left to their owners. Every newly completed packet includes an execution
   receipt with actual implementation hashes and Slurm provenance. An existing
   frozen scientific plan remains the population and labeling authority.

No structural thresholds, overlap rules, PTM settings, saved observations,
precision, movie support, or descriptors changed. No approximate neighbor
search or temporal subsampling was introduced. Full-cell PTM remains necessary
when neither its per-frame cache nor verified retained extraction labels exist.

## Measurements

The primary timings below are wall times **without a profiler**, excluding Python
process startup. Reference and optimized processes run on the same node with the
same thread limits. Filesystem caches are warm; “empty PTM cache” means atomic
classifications are computed again, not that the trajectory filesystem is cold.
These are small performance samples, not a universal throughput guarantee.

| Work | Reference | Optimized | Speedup |
|---|---:|---:|---:|
| Tail, 93 frames, 70,304 atoms, PTM already cached | 134.54 s | 6.37 s | 21.1× |
| Same tail, empty per-frame PTM cache | 282.12 s | 157.32 s | 1.8× |
| Birth tracking, 3,211 frames at 0.10 ps, PTM labels already extracted | 27.00 s | 8.85 s | 3.1× |
| Shooting birth tracking, 161 frames at 0.30 ps, PTM | 1.41 s | 0.54 s | 2.6× |
| Same shooting source, coherent detector | 0.81 s | 0.67 s | 1.2× |
| Same shooting source, coherent-strict detector | 0.67 s | 0.54 s | 1.3× |
| Same shooting source, coherent-loose detector | 1.45 s | 1.68 s | No measured improvement |

The loose detector creates many candidate records, so conditional coordinate
loading saves little there. Short timings are sensitive to node and filesystem
load. Structural extraction itself is unchanged; do not apply the tracking or
cached-tail speedups to the complete extraction-plus-tracking campaign.

The profiling pass identified 285.5 of 291.5 seconds inside path resolution for
the original cached tail. Profiler instrumentation magnifies Python overhead:
use the unprofiled 21.1× figure above, not the larger profiled ratio.
Both empty-cache runs also reproduced the six scientific output files exactly.

## Equivalence and provenance

- Four real tail packets (93 + 3 × 192 frames) reproduced all six scientific
  output files byte-for-byte on the same node: birth, graph, family trace, movie,
  frame measurements and summary. Graphs include 11, 147, 181 and 14 precursor
  nodes, respectively.
- Five complete tracking tasks cover 3,211 native 0.10 ps observations and all
  four detectors on a 161-frame shooting source. Event, interaction and
  trajectory JSONs, and decompressed family-size streams match exactly. The
  tasks include establishments, dissolved embryos and interactions.
- Fifteen real path forms, including storage/dataset tokens and historical
  aliases, resolve to identical strings. The nonblocking-lock audit confirms
  that an owned packet is skipped and its completion record remains intact.
- Comparisons to packets produced on a different CPU also preserved graphs and
  movies exactly. Floating-point descriptor/eigensolver outputs differed by up
  to about 1e-6; rerunning both implementations on node60 eliminated these
  differences. Cross-machine byte equality is not assumed.

`configs/data/nuclei_birth.json` explicitly maps the reviewed optimized tracker
hash to the canonical scientific producer hash. Unknown hashes still fail.
Each search retains its real producer hashes. This allows future optimized
searches to join the existing dataset without changing its scientific protocol
or event IDs; it does not rewrite historical producers or refresh the dataset.

Raw profiles, timing JSONs, equality receipts and reference source snapshots:
`output/nucleus_detection/performance-20261007/technical/`. Derived verification
packets are outside the repository under
`${storage:training_storage}/nucleus_histories/performance-20261007/`;
temporary PTM classifications use `${storage:cache}/nucleus-histories/performance-20261007/`.

## Operation

New birth searches use the optimized tracker automatically through the existing
`lineage_search` workflow. Existing frozen jobs retain their executable code.

For tail work already prepared, use its exact recorded configuration:

```bash
python -m src.research.crystallization_origin.birth_histories assist --config /absolute/path/to/config.json
```

Freeze execution with `ExecutionBundle` and submit CPU workers through
`SlurmQueue`, as in the existing history publisher. Multiple assistants share
the existing event locks. A helper makes one pass and leaves busy events for
their original worker; the original finish stage verifies all packets and
publishes the collection. Do not cancel another chat's jobs to enable this.

Two helpers were submitted in this chat as `1027677_0` and `1027677_1`, each
with one CPU, 16 GiB RAM and an 8-hour limit. Both started on nodecpu03.
Their immutable bundle and launch receipt are under
`${storage:training_storage}/nucleus_histories/birth-tails-010-20261007-v1/technical/optimized-20261007/`.
The original five tail workers and their completion job were not stopped or
modified. The new helpers accelerate unfinished tail packets; no completed
birth search was resubmitted merely to use the optimization.

## Descriptor computation follow-up

The subsequent request extends the optimization to the structural calculations.
The code now uses freud Steinhardt harmonics and an ordered float32 scalar bond
reduction equivalent to [freud 3.4 SolidLiquid](https://github.com/glotzerlab/freud/blob/v3.4.0/freud/order/SolidLiquid.cc).
It avoids SolidLiquid's internal filtering and clustering, which the search was
discarding before independently applying its three declared thresholds. An
ordinary NumPy norm/dot implementation was rejected because its rounding differs.
The accepted reduction matches the reference correlations bit-for-bit, uses no
fastmath and requires the verified freud version. It compiles once per process;
there is no new on-disk JIT cache or training/test suite.

With **one CPU thread**, full extraction (PTM, bond order, all four component
label arrays, solid counts and correlation histogram) took:

| Input | Reference | Optimized | Speedup |
|---|---:|---:|---:|
| Al, 70,304 atoms, frame 1500 | 3.28 s | 2.49 s | 1.3× |
| Al, 70,304 atoms, frame 3000 | 3.74 s | 2.61 s | 1.4× |
| Al, 1,000,000 atoms, frame 6000 | 59.68 s | 39.99 s | 1.5× |
| Ni, 32,000 atoms, frame 840 | 1.56 s | 1.12 s | 1.4× |

These computations produced identical cluster arrays, counts and histogram bins.
The million-atom case contains 827,608 PTM-crystalline atoms, so verification
includes substantial crystallization rather than only mostly liquid frames.
The bond-order stage alone fell from roughly 1.5–1.8 seconds to about 0.65
seconds for the 70,304-atom samples. First-process imports/JIT startup are
separate from steady frame throughput.

Tail annotation now computes only the three saved l=6 descriptors: q6, qbar6
and mean coherence. It preserves the previous nearest80 neighborhood and exact
reduction order. The unused l=2/4/8 fields, Wigner invariants and quantiles are
omitted. Across 15 real patches, this part was about **4× faster**, with identical
values. All six scientific files of four full tail packets also remain identical.
Both Al and future Ni history workflows use this reduced calculation.

New extraction chunks retain the already computed uint8 atom-level PTM types,
alongside a PTM producer hash. Tails read them only when cutoff and producer
match, with the chunk checksum verified. A two-frame extraction/reuse audit
confirmed all pre-existing chunk arrays are identical and that subsequent PTM
requests avoid classification and new cache files. It also preserved 121 and 87
crystalline atoms excluded from the minimum-four-atom component labels. Older
chunks have no such types and still need their per-frame cache or fresh PTM;
they are not silently treated as all-liquid or retroactively recomputed.

Four-thread PTM and bond-order calculations were also checked against one-thread
results. PTM labels, neighbor pairs and correlations were identical for all
three 70,304-atom frames. Optimized full extraction took 1.55 and 1.47 seconds
on frames 1500 and 3000 using **four cores**. This is a wall-time gain requiring
more allocated cores, not a fourfold improvement in CPU efficiency. Independent
source workers retain the one-thread default; new extraction configurations can
set `descriptor_threads`, and launchers allocate the matching total CPU count.

Detailed receipts: `output/nucleus_detection/descriptors-performance-20261007/technical/`.
The small per-frame inputs and derived classification caches are outside the
repository at `${storage:cache}/nuclei-descriptor-performance-20261007/`.

The complete 93-frame cold-tail comparison, including fresh PTM, took **49.72 s
with four cores**, versus 157.32 s for the first optimized one-core version
(282.12 s before either optimization). All six scientific output files are
identical. This 3.2× additional wall-time gain includes the increase in allocated
cores; it is not a single-core arithmetic speedup.

Detached helpers `1027684_0` and `1027684_1` now use the verified descriptor code
and four cores each (16 GiB, up to 8 hours). They replace only this chat's earlier
`1027677` helper array, whose ownership receipt was checked before cancellation.
Completed packets and atomic per-frame PTM caches are retained. The other five
original history workers, Ni jobs and simulations are unchanged. The replacement
receipt and frozen code are in the history run's
`technical/descriptors-20261007/`; old execution receipts remain preserved.

## Second acceleration pass

These are **incremental gains over the already optimized, consolidated code**,
measured on node60 in `pointnet-torch214` with one computation thread. Preparation
uses two counterbalanced repetitions; tracking uses three and reports medians.
No profiler is active. Filesystem caches are warm and Python startup is excluded.
The overlap kernel's first compilation took 0.77 s per worker and is excluded
from tracking timings below.

| Work | Previous implementation | Current implementation | Speedup |
|---|---:|---:|---:|
| Registry + source-plan preparation, 29 sources / 77,399 frames | 3.864 s | 1.593 s | 2.42× |
| Reference discovery alone, all 429 registered dataset IDs | 27.667 s | 0.994 s | 27.8× |
| Al tracking, 3,211 frames, PTM, selective cache | 6.887 s | 3.897 s | 1.77× |
| Shooting tracking, 161 frames, coherent-loose, selective cache | 1.034 s | 0.600 s | 1.72× |
| Full descriptors, 1,000,000 Al atoms, frame 6000 | 37.332 s | 34.552 s | 1.08× |

The reference-discovery row is a single pass over the full catalogue, not an
additional factor to multiply into preparation. Preparation timings do not
include full trajectory checksums or production-log/potential audits. The
million-atom descriptor measurement is one pass per implementation, including
cold compilation. Three 70,304-atom frames show little additional extraction
gain (roughly 0–2%); PTM remains the main remaining cost. These measurements do
not imply the entire extraction-plus-search campaign is 1.8× faster.

Changes:

- Resolve machine/catalogue paths from one scoped snapshot per preparation,
  and find dataset references in one pass per file. Later operations still read
  current settings. All 429 reference lists and 219 actual path forms match.
- Store one compressed member per detector. Tracking decompresses its detector
  only and reads raw PTM only when annotation needs it. File checksum checks
  remain enabled. The two measured source caches are slightly smaller
  (78,975,080 → 78,672,185 bytes; 1,347,284 → 1,329,978 bytes).
- Accumulate lineage overlaps in reusable integer workspaces, preserving sorted
  successor IDs. Integer union-find replaces sparse-matrix component labelling
  and preserves the original component numbering. No float reductions change.

Older caches benefit only from the algorithm changes: Al tracking took 6.644 s
and shooting tracking 0.842 s (about 1.04× and 1.23×). New extractions use the
selective layout automatically. The `cache repack` command described in the
[module guide](postprocessing.md#workflow) upgrades copies of existing caches
without recalculating descriptors. The isolated benchmark conversion cost
28.56 s for the longer Al source and 0.72 s for the shooting source; this cost
must be amortized across future reads. No production cache was bulk-converted.

Events, interactions, trajectory summaries and decompressed family traces are
identical for the full Al source and all four shooting detectors. Full descriptor
arrays, counts, histograms and PTM types match exactly on the three Al frames
and the million-atom frame. New extraction and both repack input layouts also
preserve components and raw PTM on two real Ni frames; a repeated repack copies
zero completed chunks. All comparisons use the same scientific inputs.

Receipts, timings and source snapshots are in
`output/postprocessing/acceleration-20261007/technical/`; the compact durable
record is [nuclei_search_acceleration_20261007.json](nuclei_search_acceleration_20261007.json).
The stable birth catalogue is not rebuilt. No simulation or submitted worker is
stopped or changed: frozen workers retain their recorded implementation.

## Optimized worker relaunch

On October 7 the user explicitly authorized replacing the older Al and Ni
analysis jobs, including the jobs from other work. Simulations remain running.
Original launch records and frozen bundles are preserved. Replacement queues:

| Stage | New jobs | Resources and dependencies |
|---|---|---|
| Al histories | `1027795_0–3` | Four cooperative workers, four cores and 16 GiB each, 8 hours |
| Al completion/publication | `1027796` | Four cores, 16 GiB, after the new history array; fills any remaining packets |
| First Ni search | `1027797` | Four one-thread source workers, 48 GiB, 12 hours; resumes the pinned 12-source plan |
| Ni extension search | `1027798` | Eight one-thread source workers, 64 GiB, 24 hours; waits for the original simulation lanes |
| Combined Ni publication | `1027799` | One core, 16 GiB; waits for successful completion of both new Ni analyses |

Cancelled analysis jobs: `1027537`, `1027684`, `1027550`, `1026944`,
`1027525`, `1027526`. Pending publishers were cancelled first, so stopping their
prerequisites could not accidentally start an old completion stage. Replacement
workers depend on termination of their old counterparts. No MD worker was stopped.

At the pre-handoff check 127 of 295 Al packets were complete. Completed packets
and per-frame PTM caches are retained; only an interrupted unfinished packet
needs to rebuild its in-memory movie/graph. The Ni resume retains the existing
search-plan hash, completed descriptor chunks and completed tracking receipts.
New Ni chunks use the selective layout; old chunks remain readable in place.
No bulk conversion or repeated PTM computation of completed chunks is requested.
A real Ni frame reproduced all four component arrays, counts and histogram bins
exactly, and an existing chunk was reused without changing its receipt.

The old scientific plan remains the frozen population authority. Each new
execution records the actual code hashes; Ni source receipts also record how
many chunks and which detector tasks were reused. Final metric tables bind the
executing implementation. Source-selection, cadence and event definitions do not
change through this relaunch.

Master receipt: `output/postprocessing/acceleration-20261007/technical/relaunch.json`.
Al queue/bundle: the history run's `technical/acceleration-20261007/`.
Ni queue/bundles: each campaign launch's `technical/acceleration-20261007/`.
Each launch now has `active-analysis.json` pointing at its replacement receipt;
the original `launch.json`/`queue.json` remains historical.

## Chunk scheduling, streaming and frame checkpoints

The next user-authorized pass replaces whole-trajectory Ni worker assignments
with a bounded round-robin queue of the existing 16-frame extraction tasks.
Each source starts its four tracking tasks after all its extraction chunks are
verified. At most twice the worker count is submitted at a time. Existing
chunks retain their identities; the first campaign resumes its original frozen
population. The extension admits completed archives while its MD producers
continue, using immutable source-local plans. Final coverage explicitly retains
any failed/missing sources and references the per-source execution identities.

Al histories now share one periodic neighbor tree/query table and one precursor
partition per source frame across all declared birth anchors. Memberships are
stored as compressed integer arrays, with original node identities and ordering.
The cache includes clusters smaller than four atoms. Each event separately
checkpoints its local coordinates, masks and scalar descriptors after each frame;
graph linking and final shape calculations keep their original rules. Cache
identities include the complete implementation contract and frozen population.

Manual real-data validation:

- All six scientific files of a 93-frame history match the previous producer
  byte-for-byte. Fresh frame preparation took 3.46 s with PTM already cached;
  rebuilding from 93 saved observation frames took 0.66 s and wrote no new frames.
  This is a resume measurement, not a cold-PTM campaign speedup.
- For one million-atom frame, batched nearest-80 queries match individual queries
  for all 37 declared anchors. Building one tree and querying all anchors took
  0.24 s on node60; full million-atom history speedup remains to be measured.
- A two-source, 32-frame-per-source native Ni workflow reused two completed chunks
  and extracted two missing chunks, reproducing component arrays, counts and
  correlation histograms. A repeat reused all four chunks. Tracking and metric
  export completed through the same public command. These cropped real inputs
  are disposable execution diagnostics, not a new scientific cohort.
- Completed-source admission reproduces the existing full 5,201-frame Ni source
  record and validates its immutable single-source plan. Frozen worker imports
  and the affected metric contracts pass before submission.

Active replacement jobs (October 7):

| Stage | Jobs | Execution |
|---|---|---|
| Al history frames | `1027851_0–3` | Four workers, four cores / 16 GiB each, 8 hours |
| Al publication | `1027852` | Four cores, after the history array |
| First Ni search | `1027853` | 12 one-thread chunk workers, 64 GiB, 12 hours |
| Ni extension | `1027854` | 12 one-thread chunk workers, 64 GiB, 12 hours; starts immediately on completed sources |
| Ni combined publication | `1027855` | After both Ni analyses succeed |

These replace this chat's `1027795–1027799` analysis queue. Completed histories,
PTM caches and extraction chunks remain intact. An interrupted unfinished unit
from the earlier code may restart once; new Al frame checkpoints then bound
future repeated work. Simulations are unchanged. Other chats' newly submitted
analysis jobs are not modified by this replacement.

Launch receipt and manual audit records:
`output/postprocessing/pipeline-20261007/technical/`. Frozen code and queue
receipts are in each run's `technical/pipeline-20261007/`, with
`active-analysis.json` pointing to the current queue. A compact durable audit is
[nuclei_search_pipeline_20261007.json](nuclei_search_pipeline_20261007.json).
