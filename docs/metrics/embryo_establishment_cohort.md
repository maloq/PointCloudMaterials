# Present-selected embryo establishment cohort

A separate, user-authorized assay of whether a current subcritical embryo
establishes a nucleus. It does not replace or relax the strict pre-birth assay.
Reuse all 92 velocity-bearing source views from its pinned inventory and the
original Al64 melt roles. These include exact 0.1 ps views of finer histories and
historical partial archives; four historical views retain unknown potentials.
No new MD, relaxation or interpolation is involved. The pending October 7
search of 22 additional production sources requires a subsequent release.

## Observations and future labels

Read the checksum-bound v3 PTM root-family lineage traces. At fixed integer-ps
landmarks on the actual source timeline, take each family originating as
emergence_from_liquid whose current largest connected component has 8–63 atoms.
It must not already have established. Choose the current component member
closest to its periodic centroid. Neither future members nor future outcomes
choose this center. Input is one unrelaxed centered nearest-80 position patch
and native same-frame, same-atom velocities, cast to float32 in A/ps directly
from the archived velocity array. No temperature, time, age, size, future label,
species or other metadata enters these arrays. Quantized source observations
remain quantized; this extraction makes no new integration-precision claim.

Exclude centers within 8 A of any current 64+ component or the largest component
of a family known to have established by this time (including inherited crystals
and initially present 64+ bodies). If an established family is currently
fragmented, exclude all candidates for that frame: the saved trace does not
identify its smaller fragments, so their distances cannot be established.
Current embryo size and clearance are audit metadata for later explicit controls.

Let s be the first start of a connected 64+ interval lasting at least 1.5 ps in
this root family. For the same input at t, y_h = 1 exactly when t < s <= t+h,
for h = 1, 3, 6 ps. Use the v3 establishment_start_ps, not peak size or the
confirmation time. Labels must be nested. All horizons require the same source
followup through t+7.5 ps, so endpoint establishment can be confirmed.
Dissolved and non-establishing persistent embryos provide negatives. Absorbed,
coalesced or identity-unresolved families terminating by t+7.5 ps are excluded
unless their own establishment was already confirmed before that termination.
Thus this is a complete-case root-establishment assay, conditional on resolved
fate and the spatial filter, not an unconditional population nucleation rate.
Composite establishment is not silently credited to each absorbed founder.
Selection of candidate landmarks and centers uses current/past state; later
followup availability and competing-fate exclusions are reported separately.

## Identity, weights and counts

A physical-family alias hashes melt ancestor, first-observed time rounded to
8 decimals, first members, and exact decoded full-cell positions, native
velocities and box at that onset. An observation alias adds current time,
center atom ID and complete current Q/V/box. Deduplicate matching archive views
in pinned source order (complete native views before partial/finer views).
Contradictory labels or roles for identical observations are an error. Histories
whose full-state fingerprints differ remain distinct; this strict alias does
not assert that approximate or branched trajectories are independent melts.
Original roles must be constant within every melt ancestor.

An observation receives weight 1/n_f where n_f is the number of retained
landmarks for its physical family. Each family has equal total weight, not each
correlated observation. Melt ancestor remains the unit for later uncertainty.
For each original role and pooled_test = role != train, and each horizon, report:

- observations: retained rows, identical at all three horizons;
- embryo_families and ancestors: distinct physical-family aliases and melts;
- positive_observations: sum of binary labels;
- positive_families and positive_ancestors: distinct aliases/melts with any positive;
- family_weighted_prevalence: sum(weight * label) / sum(weight), undefined if empty.

Source tables are BEFORE cross-view deduplication and list eligibility counts,
accepted observations and physical-family aliases. Candidate counts are after
common source followup is checked; landmarks_without_common_followup counts
frames, not candidates. Rejection counters are sequential. Fragmented-established
frame exclusions count eligible candidates. Absent source counter fields mean
no recorded occurrences of that reason; they are not missing eligibility checks.
The duplicates table identifies every dropped identical observation. Positive
observations, source-view events and physical families are never interchangeable.

Collection does not fit an encoder or a predictor. Later evaluation must keep
training on original train ancestors, with selection confined to those ancestors,
and pool validation/selection, calibration and test only for evaluation. AP is
a diagnostic alongside predictive scores and size/structure controls, never an
objective or selector. This release is exploratory after prior related analyses.
