# Cluster disappearance and post-establishment growth (v1)

Producer: `src/research/crystallization_origin/fates.py`. Label-side analysis only;
no fitting, model selection, source resplit or change to historical birth labels.

## Population and identities

Reuse all 150 original fixed-Al64 full-cell trajectories and the completed
original-coordinate PTM/ancestry audit. Saved spacing is exactly 0.75 ps; actual
timelines, graph hashes, event hashes, used PTM chunk hashes and cohort row hashes
are checked. PTM FCC/HCP/BCC with RMSD cutoff 0.1 is the operational crystal
definition. Other PTM types count as noncrystalline under this definition.
Spatial connectivity is 3.6 Å, minimum tracked size four atoms. Strong overlap
requires four shared atoms and 25% of the smaller cluster. Existing graph links
can bridge one missing frame. None of these criteria establishes a critical size.

`source,event` identifies an existing primary establishment (64 atoms for three
adjacent observations). `source,first_node` identifies a transient encounter
episode. Counts are distinct events/episodes, never the number of input centers.
Within-source events and branched/merged episodes are correlated. Roles preserve
their original assignment; the row sidecar also records the already authorized
train versus merged-test evaluation role. Temperature is not a numerical input.

## Transient inventory

Find connected components of the full temporal graph using **all** overlap edges,
including weak links and the recorded missing-frame links. Discard every component
connected to any primary establishment anywhere in the observed trajectory.
This conservative screen excludes fragments and possible ancestors of established
crystals; it can miss real failed clusters that share even a weak link with one.
Mergers among small clusters form one encounter episode, not several births.

Inventory all remaining episodes, including 4–7 atom fluctuations. Reconstruct
atom membership and verify disappearance for every episode whose peak is ≥8.
Smaller episodes are explicitly `below_verification_size`, never claimed dissolved.
The minimum verified size is an engineering choice, not a thermodynamic threshold.

For every temporal root of a verified episode, check its periodic extent and
minimum distance to an already confirmed established crystal at that frame.
Distance ≤8 Å is interface-associated. Frame-zero roots are left-censored.
Periodic extents exceeding half a box length are unresolved. A primary failed
candidate must have a single temporal root, a connected strong-overlap lineage,
isolated origin, ≥8 atoms for two adjacent strong-linked observations (0.75 ps
between them), and verified disappearance. Gap and weak-edge counts, multiple
origins and branching remain in the inventory. Bridged gaps do not count toward
the required adjacent size-qualified streak. Events below frame cadence are lost.

`peak_atoms` is the largest individual component, not a sum of fragments.
`observations` counts unique saved frames, and `observed_span_ps` is last minus
first observed time. Size sensitivity reports minimum peaks 8/16/32/64 with and
without the **same two-frame ≥8** duration requirement; it does not impose a
two-frame streak at each larger threshold. Brief ≥64 excursions that never meet
establishment can appear in the ≥64 row. No predictor-ready crystal-free histories
have yet been extracted for these new candidates.

## Disappearance verification and censoring

The entire selected episode/established lineage must have no outgoing graph
descendant. Collect the union of atom identities in **all terminal branches**.
Every such atom must be noncrystalline at each of the three observations directly
after the final tracked frame. The confirmation span is 2.25 ps after the last
tracked observation (the three confirming observations themselves span 1.5 ps).
This confirms loss of the tracked crystal and observed remelting of terminal
cores; it does not prove that every earlier member became liquid or cannot
recrystallize later. PTM cannot identify all forms of structural ordering.

Missing any required follow-up frame is `right_censored`. Residual crystalline
terminal-core atoms are `unresolved_residual_crystal`, not dissolved. An event
merging with another established ancestry is `merged`; its descendant's later
size or disappearance is not assigned to the individual pre-merge nucleus.
Single-atom or sub-four-atom residuals are checked through PTM even though they
are absent from the tracked graph. A disappearance caused solely by falling
below the tracking threshold therefore cannot automatically count as dissolution.

## Existing event fate labels

For every existing primary event, use its established ancestry and audit
unique-ancestry descendants only before the first merge with another established
event. Peak exclusive size and last exclusive frame are descriptive fields.

Growth is confirmed when a strongly supported descendant reaches
`max(128, ceil(2 * size_at_original_confirmation))` for three adjacent observations,
all strictly after original establishment confirmation and before that merge.
It is one sustained later doubling, not a claim of monotonic or indefinite growth.
Strong-supported descendants can include recorded one-frame ancestry bridges,
but the growth streak itself requires adjacent observations.

- `continued_growth`: sustained growth confirmed; inspect terminal status for a
  later merge or trajectory truncation.
- `grew_then_dissolved`: both growth and subsequent verified disappearance.
- `dissolved`: verified disappearance without the declared growth criterion.
- `unresolved_merge`: merged before growth could be confirmed.
- `right_censored`: growth unconfirmed and insufficient terminal follow-up.
- `unresolved_disappearance`: no growth confirmation and residual PTM crystal
  prevents assigning a disappearance.

`growth_confirmed` and `terminal_status` remain independent columns. No forced
binary assignment is made for ambiguous endings. Historical event kinds are
preserved; the existing classifier's retained 95 isolated events are flagged.

## Exported tables

`transient-episodes.csv` is the full candidate inventory; `established-fates.csv`
contains all primary catalogue events plus a retained-cohort flag.
`birth-row-fate-labels.csv` joins the unchanged row IDs to additional labels:
positive rows inherit their actual event fate; negative rows get
`not_applicable_liquid_control`. A negative's matched-case event ID does **not**
make it a member of that nucleus. The same sidecar applies to original and relaxed
inputs because outcome trajectories and rows are unchanged.

`fate-counts.csv` counts labels by population and original role; source counts
are distinct source IDs with that label. `failed-size-duration-sensitivity.csv`
reports candidate counts under the declared size/duration restrictions. These
are finite-observation availability counts, not nucleation rates or independent
sample sizes. Historical fitted models and numerical birth labels are untouched.
