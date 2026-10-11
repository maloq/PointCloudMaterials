# Cumulative nuclei birth dataset definitions

This publisher indexes completed `al_nucleus_lineage_v3` searches without
recomputing structural labels, event origin, onset, establishment or fate.
The scientific parameters and tracker/structural-producer hashes must match
`configs/data/nuclei_birth.json`. Frozen upstream definitions and provenance remain
in each search. This family defines the consolidation and exported count tables.
`producer_equivalences` is an explicit allowlist of verified execution-only
revisions mapped to their canonical scientific hash. Unknown hashes fail. The
October 7 lazy-coordinate tracker preserves frame labels, event memberships,
origins, fates and size histories; see the [performance audit](../nuclei_search_performance.md).
Actual producer hashes remain in the upstream records and publication provenance.
The canonical protocol and event IDs do not change for an allowed equivalent.

A completed search may supply only a subset of the supported cadences. Every
supplied cadence must match exactly; a native 0.10-ps addition does not require
an artificial 0.30-ps view. The native-only 0.30-ps inclusion rule is unchanged.
Explicitly accepted GPU descriptor backends retain their separate numerical
audit, implementation and extraction-completion records. They are not relabelled
as CPU executions or claimed globally bitwise equivalent. The same operational
thresholds and unchanged chronological tracker apply; duplicate source/task
identities still require identical saved scientific files. Publication exposes
each search's actual descriptor backend in `technical/searches.json`.

## Identity and population

`source_id` is the extraction producer's `input_identity`: SHA-256 of declared
trajectory array hashes and ordered frame mappings. Equal identities enter once.
Different histories, branches and historical prefixes remain separate records,
even when they share ancestry or overlap in physical observations. They are not
independent samples. `view_id` binds that identity, exact source-frame indices and
cadence; `task_id` additionally binds the detector and scientific protocol.
`event_id` adds the upstream integer track and episode. IDs do not depend on
publication date or row order. Different detectors/cadences have different IDs;
they must not be summed as distinct physical nuclei.

Only exact 0.10 and 0.30 ps observation intervals are accepted, checked from the
frozen source timeline with absolute tolerance 1e-8 ps. The configured
`sampling_policy` has distinct meanings:

- `010`, `exact_saved`: finer native data may enter through exact saved-frame
  selection, never interpolation. This retains the existing 0.10 ps population.
- `030`, `native_saved`: the source timeline and **every referenced raw part**
  must have uniform 0.30 ps intervals. Manifest and timestep-array hashes are
  checked, and native intervals are calculated from integer timesteps and the
  recorded integration timestep. The published frame map uses the full source
  timeline. A regular 0.30 ps selection from finer or irregular raw observations
  is excluded, including nested shooting histories with finer startup prefixes.

`source-eligibility.csv` records one decision per upstream source/view/search:
`selected`, the `sampling_policy`, declared source cadence, unique measured raw
intervals, and an inclusion/exclusion reason. An empty `raw_intervals_ps` array
means raw-part inspection was unnecessary (the exact-view policy or an already
incompatible source cadence), not missing time or zero cadence. Exclusions apply
to this published population only; upstream arrays and labels remain intact.
Source origin,
potential, original split roles and partial-observation status are retained.
The 0.75 ps historical search is preserved upstream and excluded from these two
views. Relaxed/static data are paired holdings, not independent event outcomes.

## Event selections and counts

- `established`: root record, `origin=emergence_from_liquid`, and non-null
  `establishment_confirmation_ps`: at least 64 connected atoms sustained 1.5 ps.
- `appearances`: every root `emergence_from_liquid` record, including one-frame
  appearances of at least eight atoms and failed embryos.
- `dissolved`: appearance records with `terminal_status=dissolved` and no
  independent establishment. Absorption, coalescence and unresolved endings are
  not dissolution or negative labels.
- `all`: all indexed records, including composites, parent-present crystals,
  interface-adjacent origins and ambiguous histories.

Counts group by cadence, detector, record kind and origin. `records` is the number
of indexed rows; `established` counts non-null independent establishment times;
`dissolved_before_establishment` counts confirmed dissolution without independent
establishment. `grew_to_128` counts non-null growth confirmation times (128 atoms
sustained 1.5 ps). These are operational lineages, not validated critical nuclei.
The birth time is the first eight-atom appearance, not the establishment time.
No crystal-free input eligibility filter or liquid-control sampling is applied.

The cadence table counts distinct indexed source views, distinct recorded ancestry
strings, total saved observations and the sum of last-minus-first observation
times. The latter is `observed_span_ps`, not independent simulation exposure;
historical prefixes and related branches can overlap. Sources with zero births
remain in the source/view tables. An unsearched source is never a negative example.

`ancestry.role_conflict` means more than one distinct historical role string exists
for the same recorded ancestor. This includes both potentially compatible names
and genuine cross-role overlap. No role remapping or new train/test split occurs.
Consumers must group by ancestry and declare their own permitted role handling;
missing/unknown ancestry is not evidence of independent preparation.

## Publication and verification

The dataset fingerprint binds the configuration, complete upstream search identities,
plan/completion SHA-256 values and this publisher's implementation contract.
The SQLite index and copied source metadata represent the current dataset.
Full event atom memberships and interaction/size traces reference checksum-bound
upstream shards; large trajectories are reused, not duplicated. New tasks verify
all upstream tracking artifact hashes before indexing. Unchanged tasks reuse the
current verified index after checking its database hash and upstream receipts.
`verify --upstream` rehashes all tracking artifacts; normal event-detail reads
rehash the event shard. Historical arrays retain their original precision and
declared hashes. Terabytes of MD coordinates are not rehashed by publication.

An update may add compatible completed searches; removal or mutation of upstream
search bindings and conflicting duplicate labels fail. An explicit sampling-policy
change rebuilds the published view selection and changes the fingerprint. With
unchanged schema and lineage/structure/producer contracts, existing event rows
are reused after verifying the old database hash; excluded views and their tasks
are removed only from the derived index. Frozen source payloads retain upstream
view maps as provenance; only indexed `views` belong to the published datasets.
Changing a detector or event definition requires a distinct scientific protocol.
Incomplete searches
and unsearched registered Al holdings remain in coverage. Coverage observations
can refresh without changing the dataset fingerprint. Publication holds a lock
while replacing the current index and manifests. Readers check the fingerprint
inside SQLite against the manifest, detecting interrupted publication. Existing
open readers retain their database snapshot until close. There is no historical
dataset-version archive or old-experiment compatibility layer. New experiments
record the fingerprint and selected event/source IDs in their own cohort files;
an optional expected fingerprint rejects accidental reuse after a dataset update.
