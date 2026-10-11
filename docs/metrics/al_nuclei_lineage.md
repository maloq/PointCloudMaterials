# Revised Al birth lineage definitions

This release separates embryo appearance, persistent establishment, ancestry and
terminal fate. It supersedes interpretation of the v2 clean-before-interaction
count as the number of physical births. Historical v2 files remain unchanged.
These are operational tracking definitions, not validated critical nuclei.

October 8 additions retain these same criteria at native 0.10 ps. Their
`resident`/`voronoi` GPU descriptor backend is pinned separately in the parent
plan, with the existing Al/Ni numerical audit and actual implementation hashes.
The chronological CPU tracker is unchanged. CPU/GPU equivalence is measured on
the audited windows, not asserted for every future frame; coherent-bond labels
can differ at numerical boundaries. GPU chunks and full source coverage are
verified before tracking. The 45 main trajectories omit their initial 15 ps
settling; the 33 birth-focused velocity branches retain initialization and
settling. The latter's online growth stop is not used as a retrospective label.

## Frozen populations and observations

The parent is the completed all-Al 2026-10-03 search: 1,042 source records,
179 exact 0.75 ps views, 43 exact 0.10 ps views and 892 exact 0.30 ps views.
All 0.10 ps views have matched stride-three 0.30 ps views. No interpolation,
new simulations, new quenches or new frame classifications occur. The exact
parent plan, checksummed component bank, provenance and coverage are retained.
Historical partial observations, descendants and alternative cadences are related
views, not independent melts. Ongoing/new simulations are outside this frozen
comparison and require a later population release. Relaxed and static collections
remain inventoried, not additional raw MD temporal events.

The separate October 5 dense-additions release freezes newly registered native
0.10 ps sources absent from that parent. It extracts the same four structural
definitions, then applies exactly the lineage rules below at native 0.10 ps
and stride-three 0.30 ps. Each source's integer timeline is checked; there is no
interpolation. The prior paths and input identities are excluded before task
construction. Only completed production MD (`raw_dynamics`) enters this addition. Nucleus
recovery, selected-state restarts, seeded growth controls and engine comparisons
are excluded by recorded scientific role, not atom count. Incomplete production
runs are deferred. Source state, generating potential, preparation ancestry and
precision remain explicit. The addition does not replace or expand the running frozen rerun.
`technical/plan.json` records the specific population for each release; their
counts must not be combined as independent samples.

The October 6 production addition retains exactly the same structural and
lineage rules. Selection additionally verifies native LAMMPS MEAM commands in
the input and executed equilibration log, the two Lee2003 potential-file hashes,
prepared-liquid configuration/restart hashes, completion and converted-timeline
receipts. Previously searched input identities are excluded across both earlier
releases. Recovery and control purposes are excluded regardless of their
potential. A held-out `test` source role is not a diagnostic-simulation label.
The immutable production audit and its implementation identity accompany the
fresh targeted dataset inventory. These are provenance checks, not changes to
birth thresholds or detector parameters.

PTM and loose/reference/strict bond-order definitions remain exactly those of
[the parent extraction](al_nuclei_search.md). This study fits no predictor and
feeds no conditions or timestamps to a model. The stricter crystal-free predictor
input assay remains separate from the event search.

## Component continuity and descendant families

The October 7 seven-source V100 addition uses only native 0.10 ps observations
and the same four detectors and lineage rules. Its explicit source IDs and
original campaign input checksums bind the selected velocity reshoots. The
producer audit checks the executed GPU external-force callback separately from
the intended MEAM input deck: forces are float32, native LAMMPS NPT integration
is float64, and the scalar energy reduction is float64. Six melt ancestors
underlie seven trajectories. Their initial 15 ps equilibration was not saved;
initial-frame embryos retain unknown onset, and insufficient final follow-up
remains censored. This adds provenance selection, not a new birth definition.

Track extracted connected components of at least four atoms. A temporal link
requires at least four shared atoms and at least 50 percent overlap relative to
either predecessor or successor. Both directional fractions are recorded for
interactions. Prefer adjacent observations; use historical links only when a
component has no qualifying adjacent predecessor. Within each class, maximize
shared atoms, then the smaller directional fraction, with deterministic index
ties. A component can retain identity across an absence that has not yet met
its 1.5 ps reset condition. No missing crystal labels or coordinates are invented.

Splits inherit the same family. They do not create a birth. The family's size is
the largest currently observed connected descendant, never the sum of disconnected
fragments. Thus persistence describes a connected descendant present throughout
the interval, possibly switching between related fragments. Fragmentation counts
are the number of observed frames with multiple components in a family, not an
independent count of physical splitting events.

Only distinct families with qualifying adjacent predecessors can merge.
Historical gap-overlap is not an extra merge parent. Noncrystalline atom exchange
and weak overlap with subcritical fluctuations do not veto a birth.

## Appearance and origin

Create an appearance record when a family first reaches eight connected atoms.
Retain one-frame events. The onset interval is (previous observation, first
crossing]; its lower bound is unknown for an initial-frame appearance.
A source-initial four-to-seven-atom precursor can subsequently produce a birth;
it is not automatically an already-present crystal.

Freeze origin at that first crossing:

- `emergence_from_liquid`: a root family, at least 1.5 ps of preceding source
  observations, no inherited crystal lineage and no contemporaneous established
  crystal within 8 Angstrom.
- `near_interface_emergence`: otherwise the same root criterion, with minimum
  atom-to-atom separation at most 8 Angstrom. Proximity does not prove front arrival.
- `unresolved_crystal_connection`: at root creation, weak overlap with a previously
  established family makes crystal ancestry uncertain. This criterion deliberately
  differs from harmless weak overlap with small unestablished fluctuations.
- `unresolved_history`: source observation before first crossing is shorter than
  1.5 ps. It remains a candidate record, not a confirmed birth.
- `parent_present`: already at least eight atoms at the first source observation.
- `composite`: a family created by coalescence; it is not an additional birth.

`is_birth_record` is true only for root emergence/near-interface records. All
other observations are retained. Source-initial components of at least 64 atoms
are evidence of a pre-existing crystal immediately; their persistence milestone
is still recorded separately. Later neighbors qualify only after an already
observed establishment, never through future backdating. Full periodic cells
are used. Once a family demonstrably disappears, its ancestry is not permanently
attached to the atoms and does not prohibit a later birth after melting.

## Minor absorption and coalescence

For each merger, the dominant incoming family supplies the most shared atoms.
Another family is a major contributor if it supplies at least eight atoms and
at least 20 percent of the dominant contribution, or if it is already established.
These fractions are engineering settings, not physical universal constants.

Minor unestablished donors can be absorbed while the dominant family continues
its size and persistence counters. If two or more major families join, create a
composite family with their founder IDs. Preserve all earlier birth records;
the composite is not a third birth. It has its own size and establishment history.

A fully consumed unestablished donor entering an established crystal is labelled
absorbed before establishment. It cannot inherit that crystal's size as its own
independent success. A donor with other visible fragments continues through
those fragments; only fragment transfer is recorded. A merger can therefore
both transfer a fragment and leave the original family alive.

Independent establishment milestones stop when that donor is fully consumed,
not at its first ordinary split or minor absorption. `descendant_establishment`
separately records later establishment of a composite formed from subcritical
founders. It does not count founders as separate established physical nuclei,
and does not credit mere absorption into an already established crystal.

## Persistence and terminal outcomes

Record first crossing and confirmation for thresholds 8, 32, 64 and 128 atoms,
with adjacent-observation spans of 1.5 and 3 ps. Main persistent-embryo,
establishment and documented-growth labels use 8, 64 and 128 atoms for 1.5 ps.
They are exactly representable on all three observation grids. Missing observations
break a size-persistence streak but need not reset the family.

Dissolution requires every related connected component to be absent or below
four atoms for at least 1.5 ps of observed follow-up. Earliest confirmation is
last >=4 observation plus one cadence plus 1.5 ps. Crossing below eight alone
does not create repeated births. Unresolved temporal-gap identity is not certified
dissolution. Terminal labels distinguish dissolution, absorption, coalescence,
identity uncertainty, right censoring and presence at the end. A known growth
history remains recorded if the family subsequently dissolves or interacts.

## Exports and interpretation

Events retain source/ancestry/cadence/detector, onset atom identities, size extrema,
all milestones, origin and fate. Peak component/frame references point into the
unchanged membership bank. `sizes.jsonl.gz` retains per-observation family sizes;
`interactions.json` retains directional overlap and donor/receiver records.

Counts group records by cadence, detector and origin. Parent-present and composite
records are explicit. Do not sum cadence/detector/historical-release observations
into independent physical births. Growth audits report large connected clusters,
root establishments and establishments through subcritical coalescence separately.

Paired comparisons use root appearance records, including uncertain/initial ones,
and exclude composites. Candidate matching uses shared onset atoms, overlapping
observed >=8 lifetimes with one-coarse-cadence tolerance, at least four shared atoms
and overlap >=25 percent of the smaller onset set. Greedy one-to-one matching
prioritizes overlap then shared count. The atom-index implementation avoids an
all-pairs scan without changing those declared candidate criteria. These are
matching diagnostics; unmatched does not automatically mean false detection.

## Validation and limits

The October 7 execution optimization reads coordinates and builds the periodic
established-crystal tree only on frames that create an event record. Every frame's
sealed component labels still enters tracking, resets, overlaps and milestones.
The tree's membership is captured before processing current-frame milestones,
as in the original producer. Full-cell structural extraction and input verification
are unchanged. Actual producer hashes remain in each release; the dataset admits
only explicitly verified equivalent hashes, documented in
[the performance audit](../nuclei_search_performance.md).

Audit the growing lineage in source 0863; ordinary fragmentation must not suppress
its establishment. Audit initially crystalline shooting sources and coalescence
separately, and inspect dense/downsampled comparisons. Do not select thresholds
by AP, larger event counts or forcing every crystallizing source to have an
isolated origin. Unexplained major crystallization remains a tracking uncertainty.

Prestructured liquid is physically plausible [Kawasaki and Tanaka 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922597/).
Size alone need not determine commitment [Liang et al. 2020](https://arxiv.org/abs/2004.01473).
Neither study validates the specific numerical settings above for our Al potential.
