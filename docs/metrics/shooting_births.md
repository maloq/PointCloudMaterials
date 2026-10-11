# Al shooting birth and fate audit (v1)

Full-cell PTM/ancestry search on registered existing Al Lee2003 MEAM shooting
futures. This is label construction, not predictor fitting. Original observed
coordinates retain their stored precision; there is no relaxation or new MD.
The immutable plan records every branch, actual timeline, array hashes, parent
configuration hash, ancestral source, original split and dynamics protocol.
Missing outcomes in partially executed/preparation-only campaigns are enumerated.
Smoke campaigns removed by the separate October 2 cleanup are explicitly excluded
with their retirement receipt; a null branch ID denotes a whole-campaign exclusion.

## Population and counting

An exact **0.30 ps** grid is selected from actual integration steps (3 fs).
No interpolation. The nested first-passage pilot has denser early observations;
only exact 0.30 ps frames enter this audit. Its original segment takes precedence
over the overlapping fixed24 export, whose unique continuation supplies missing
later frames. Shared positions are checked after the recorded float16 conversion;
identities and boxes must agree. Original segments longer than 24 ps are retained
through their actual endpoint. A nested original and continuation constitute one
future. The continuation's thermostat reset and precision boundary are recorded.
Nested stopping/extension depends on historical basin outcomes, so these data
are not an unbiased fixed-duration ensemble.
Exact duplicate future array identities are excluded with a retained-source link.

Counts are branch-local events/episodes. `branches`, `parents` and
`ancestral_sources` count distinct trajectory IDs, parent data hashes and canonical
source runs, respectively. Sibling futures, different parents from the same source,
and multiple events within one future are correlated. These are availability counts,
not independent sample sizes, nucleation rates or unbiased birth probabilities.
Roles are historical metadata; no new split is made. Different thermostats and
durations are reported by campaign, never silently relabeled as one protocol.
Original Al64 0.75 ps results remain separate and unchanged.

## Crystal detection and establishment

Reuse full-cell PTM FCC/HCP/BCC, RMSD cutoff 0.1, periodic 3.6 Å connectivity,
minimum tracked size four atoms, strong overlap of at least four atoms and 25%
of the smaller cluster, and at most one missing-frame bridge. No missing frame
counts toward a persistence streak. These are operational crystal definitions,
not a thermodynamic critical-nucleus test. Sub-cadence events can be missed.

Establishment requires ≥64 atoms in six consecutive strong-linked observations,
spanning 1.5 ps. The original ancestry classifier supplies `establishment_kind`:
isolated, interface-associated (≤8 Å from another confirmed established crystal),
left-censored, ambiguous merge or periodic extent. Directed ancestry through
**any overlap edge** is also traced back from the birth node. If it reaches
frame zero, a later establishment is `parent_present_embryo_growth`, not a new
birth. Already established frame-zero events retain `left_censored_existing`.
`initial_cluster_ancestry` is exported separately. This can conservatively reject
weakly connected new clusters; it cannot recover unobserved pre-branch history
or ancestry of sub-four-atom motifs. A later strict crystal-free history screen
is required before any candidate becomes a classifier example.

## Failed episodes and established fates

Reuse the [original fate logic](nucleus_fates.md), adapted explicitly to physical
durations. Temporal components use all overlap links; any component connected
to a primary establishment anywhere in the branch is excluded from the failed
inventory. All remaining episodes are inventoried, including peaks of 4–7 atoms.
Only peaks ≥8 receive atom-level disappearance verification.

`brief_failed_candidate`: isolated, single temporal root, connected strong
lineage, verified disappearance. `primary_failed_candidate` adds ≥8 atoms in
four consecutive observations: **0.90 ps**, the shortest span on this grid that
meets the original ≥0.75 ps requirement. These are nested groups, not additive.

Disappearance requires no outgoing descendant and the union of all terminal
branch atom IDs to be PTM-noncrystalline at **eight** successive observations
after the final tracked frame. Actual follow-up is **2.40 ps**, the shortest
grid duration ≥2.25 ps. Residual PTM crystal is unresolved; insufficient follow-up
is right-censored. Peaks <8 are `below_verification_size`, never claimed dissolved.
Multiple-root encounters and interface-associated episodes remain explicit.

Established growth requires `max(128, ceil(2 * confirmation_size))` atoms for six
adjacent strong-linked observations strictly after establishment confirmation,
spanning 1.5 ps and preceding any merger with another established ancestry.
Growth confirmation and terminal status are separate fields. Fate labels are
`continued_growth`, `grew_then_dissolved`, `dissolved`, `unresolved_merge`,
`right_censored` or `unresolved_disappearance`; no forced binary fate is assigned.

## Tables

- `branches`: provenance keys, actual duration/cadence, frame and episode counts,
  exact realized persistence/follow-up durations.
- `established-fates`: per-event origin, parent-ancestry flag, growth and fate.
- `transient-episodes`: all unestablished encounter episodes with size, duration,
  topology, isolation, disappearance and nested candidate flags.
- `counts`: counts by campaign, population and label; all-campaign totals are
  descriptive inventories, not a pooled statistical sample.
- `unavailable-or-duplicate`: declared but unexecuted branches/preparations and
  any exact duplicate future exclusions.

`birth_time_ps`, `first_time_ps`, `last_time_ps` and confirmation times are relative
to shooting start. They organize labels, not numerical predictor inputs.
`observed_span_ps` is last minus first tracked observation. `peak_atoms` is the
largest individual spatial component, not a sum over split fragments.
