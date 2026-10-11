# Al cluster birth search definitions

This versioned catalogue searches full periodic original Al trajectories for
connected embryos, establishment and fate. It is independent of prediction
history selection: no zero-PTM-atom rule, future-selected center or four-liquid-control
requirement removes a detected event. No predictor is fitted and no AP criterion
selects a detector. All thresholds below are operational sensitivity settings,
not calibrated estimates of critical nucleus size.

## Population and observation protocols

The frozen plan binds every source manifest, array identity, actual integer timeline,
preparation ancestry, material/potential and original split. The all-Al inventory
includes fixed historical sources, shooting, million-atom continuous melt/source,
other branches, uniform-temperature sources, precision studies and available dense
descendants. Unknown ancestry remains explicit and is conservatively grouped by
dataset; it is not an independent-melt count. An unavailable trajectory is not liquid.

Separate protocol views use exact 0.75, 0.10 and 0.30 ps spacing. Every compatible
source enters each exactly selectable protocol. In particular 0.10 ps trajectories
also enter 0.30 ps at stride three, and 0.01 ps can support all three. No interpolation,
nearest-frame substitution or 0.70-as-0.75 relabeling is used. Native 0.15 ps data
support 0.75 and 0.30; native 0.075 supports those same grids. Incompatible sampling
remains in coverage. Irregular timelines are split at exceptional intervals with
separate left/right censoring, preserving source ancestry. Source phases are stitched
only through the previously verified continuous boundary. Existing shooting nested
prefix/continuation composition is retained exactly.

Raw/relaxed, float32/float16 and historical-prefix observations are related views,
not independent dynamics. Raw MD defines the primary event/fate labels. Available
relaxed collections and remote copies remain explicit in coverage and do not add
independent events. The inventory is frozen at preparation; live/unfinished trajectories
need a later release after archival, never mutation of this plan.

## Four structural definitions

1. PTM at RMSD cutoff 0.1; FCC/HCP/BCC are solid. Connect solid particles by
   periodic distances at most 3.6 Angstrom.
2. Coherent loose: normalized l=6 bond-vector dot product greater than 0.65,
   at least six coherent neighbors.
3. Coherent reference: dot product greater than 0.70, at least seven coherent neighbors.
4. Coherent strict: dot product greater than 0.75, at least eight coherent neighbors.

Bond order uses all neighbors within 3.6 Angstrom and freud-analysis 3.4.0.
Coherent clusters connect solid particles through the corresponding coherent
bonds. PTM and bond-order classifications are parallel definitions, not a union.
Components smaller than four atoms are untracked. Retain frame-level memberships,
solid counts and the bond-correlation histogram; share this extraction across cadences.

Execution revision (October 7): use freud's Steinhardt harmonics directly and
the same ordered float32 bond dot product and q_l normalization as freud 3.4
SolidLiquid. Its redundant internal thresholding/clustering is omitted; all three
declared coherent thresholds and component definitions are still applied here.
No fastmath or approximate reduction is used. New chunks additionally retain
uint8 atom-level PTM types and the PTM producer hash, allowing tail annotation
to reuse the classification without losing one-to-three-atom components.
Historical chunks without these types still require the original full-cell PTM.
`descriptor_threads` controls CPU execution; launchers allocate
`workers_per_lane * descriptor_threads` cores to extraction. Default is one
thread per independent frame worker. See [performance audit](../nuclei_search_performance.md).

## Temporal tracking and candidate episodes

Match components using at least four shared atom identities and overlap at least
25 percent of the smaller component. Prefer the most recent predecessor, then
largest shared membership, with deterministic tie breaking. One-to-one main tracks
retain explicit split/merge side information. Weak overlap flags uncertain ancestry
but does not remove the candidate. Maximum allowed observation-to-observation
link interval is 1.5 ps, equal across protocols: two 0.75-ps steps, fifteen 0.1-ps
steps, or five 0.3-ps steps. A gap can restore identity; it cannot establish continuous
observed persistence. Unobserved intervals contain no invented crystal labels.

Record every excursion reaching eight connected atoms. Its onset is the first
eight-atom observation in that excursion, not the first individually crystalline
atom within an unrelated spatial patch. Separate excursions after at least 1.5 ps
of observations below eight atoms. A sustained embryo remains at least eight atoms
for 1.5 ps; brief excursions are retained too.

Establishment is at least 64 atoms for 1.5 ps of adjacent observations. Record the
first crossing and later confirmation separately. Parallel 32/128-atom thresholds
are sensitivity labels on the same tracked trajectories. Documented growth is at
least 128 atoms for 1.5 ps. Growth/establishment after a split or merge is not credited
to the independent pre-interaction embryo. These criteria are not a committor or a
universal supercritical size.

All physical persistence spans are exactly representable: 3 observations at 0.75 ps,
16 at 0.1 ps, and 6 at 0.3 ps. Confirmed disappearance means the tracked connected
embryo is absent/below eight for at least 1.5 ps of subsequent observations; isolated
remaining crystal-like atoms are allowed. Earliest confirmation is one cadence plus
1.5 ps after the last high observation. A split/merge before that confirmation yields
an unresolved interaction instead of dissolution. Missing final follow-up is censored.

Origins are parent-present, split descendant, uncertain ancestry, interface-associated
or isolated. For a new track, the interface criterion is minimum atom-to-atom distance
at most 8 Angstrom to a concurrently visible track established strictly before the
current frame. Future confirmation never backdates an existing interface. This is
an operational proximity attribution, not proof that distant crystals have no influence.
Reappearance after dissolution on the same identity track is separately labeled.
Periodic extent ambiguity is retained. No uncertain origin is silently promoted.

## Exports and paired comparisons

Event rows carry source, ancestor, protocol/native cadence, detector, times, size,
origin, fate and uncertainties. `source_origin` distinguishes historical partial
trajectories from current/complete releases and shooting. Aggregate rows count
event observations, not deduplicated physical births across overlapping releases.
Onset/peak atom IDs remain in source JSON files;
complete component memberships remain in the extraction bank. Counts are per
source/protocol/detector. Never sum detector or cadence views into a physical birth count.
Histories and branches from an ancestor are not independent samples.

Pair exact 0.1 and decimated 0.3 ps events, and PTM/reference-coherence events on each
cadence. Candidate matches require observed lifetime overlap (allowing one coarser
cadence), at least four shared onset atoms and overlap at least 25 percent of the
smaller onset set. Greedily choose one-to-one pairs by overlap then shared count.
Report unmatched counts, onset shift (right minus left), origin and fate agreement.
This is a declared matching diagnostic; unmatched does not automatically mean false.

No nucleation rate is inferred from retrospective candidate counts. Thermodynamic
criticality requires separate basin/commitment validation. No new simulations or
minimizations are part of this search.

## Scientific references

The October 7 seven-source V100 extraction uses the unchanged four structural
detectors at native 0.10 ps only. An explicit source-ID selection and GPU-aware
producer audit bind campaign input hashes, executed callback/NPT commands,
potential files, precision, completion and exact timelines. This selection
extension changes no descriptor or tracking calculations. Initial equilibration
is unavailable; see the v3 lineage definitions for censoring and ancestry.

- [Lechner and Dellago 2008](https://arxiv.org/abs/0806.3345): bond-order discrimination.
- [Larsen et al. 2016](https://arxiv.org/abs/1603.05143): PTM structural classification.
- [Blow et al. 2026](https://doi.org/10.1063/5.0318093): cluster-definition sensitivity.
- [Liang et al. 2020](https://arxiv.org/abs/2004.01473): commitment validation beyond size.

The exact embryo thresholds, matching, temporal tracking, origin and fate rules
are our engineering protocol. Those papers do not validate these numerical choices
for this Al potential.
