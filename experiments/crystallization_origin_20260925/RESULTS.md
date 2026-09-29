# Completed crystallization-origin audit — 26 September 2026

The current Al64 prediction benchmark overwhelmingly measures capture by an
already established crystal. The full trajectories contain candidate births,
but the fixed prediction grid provides **zero regional-birth training windows**
at either 3 or 6 ps. A separate regional-establishment population is needed
before treating encoder performance as evidence about predicting crystal birth.

Both CPU audits completed before the requested 10:00 Paris deadline: Al64 at
**03:07**, and the external-material audit at **03:56** on 26 September.
All 150 fixed sources and 26 external trajectory records completed. No model
was trained and no new simulation was generated.

## Fixed Al64: what the existing prediction task measures

The unchanged release supplies 126,545 at-risk windows and 64 tracked centers
per source. Counts below aggregate train, selection, calibration and test;
they are label-availability counts, not model performance estimates.

| Origin-relative label | Positive windows, 3 ps | Positive windows, 6 ps |
| --- | ---: | ---: |
| Existing crystal arrival | 856 | 1,902 |
| Local isolated establishment | 0 | 2 |
| New crystal formed outside the region, then arriving | 0 | 1 |
| Unresolved unestablished ancestry | 12 | 24 |
| Unresolved ancestry | 1 | 2 |
| **All positive onset windows** | **869** | **1,931** |

Existing-crystal arrival accounts for **98.50% at both horizons**. The remaining
125,676 / 124,614 windows have no onset within 3 / 6 ps, respectively.
The audit found zero disagreements between recomputed full-cell PTM and the
historical center labels. Attribution means ancestry-based capture; it does
not measure front velocity or establish a particular microscopic mechanism.

Across full trajectories, there are **348 isolated establishment candidates**,
of which only **38 (10.9%)** are within 8 Å of any of the 64 centers at birth.
There are also 22 interface-associated establishments. An operational birth
requires 64 connected PTM-crystalline atoms for three observations spanning
1.5 ps. These are not validated physical critical nuclei.

The separate regional target asks whether a new isolated cluster establishes
within 8 Å, regardless of whether the center atom itself becomes crystalline.
On the existing center-liquid prediction grid its availability is:

| Fixed source role | Full-cell isolated candidates | Regional positive windows, 3 ps | Regional positive windows, 6 ps |
| --- | ---: | ---: | ---: |
| Train | 206 | 0 | 0 |
| Selection | 34 | 3 | 3 |
| Calibration | 41 | 0 | 0 |
| Test | 67 | 1 | 3 |

The gap between 206 births in training trajectories and zero training windows
comes from the combination of sparse spatial/temporal sampling and the current
center-liquid risk definition. It does not establish absence of nucleation
precursors in the raw trajectories. Coverage alone also does not imply that a
qualifying forecast origin exists before the covered birth.

## Larger trajectories and other materials

External histories use a separate declared 0.5 ps grid, 1.5 ps persistence and
outcome-blind 1% center samples. Counts must not be pooled with Al64 as a matched
benchmark. Material cutoffs use the existing fixed length normalization. The
regional columns count forecast windows, not distinct nuclei.

| Material / generating potential | Records | Isolated candidates | Covered by 1% centers | Covered by nested 64 centers | Regional windows, 3 ps | Regional windows, 6 ps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Al / Lee 2003 MEAM, million-atom history | 1 | 42 | 27 | 1 | 143 | 290 |
| Al / Mendelev 2008 EAM | 6 | 0 | 0 | 0 | 0 | 0 |
| Mg / Wilson 2016 EAM | 6 | 52 | 37 | 0 | 95 | 205 |
| Ti / Kavousi 2019 MEAM | 7 | 17 | 13 | 0 | 48 | 96 |
| Ta / Zhong 2014 EAM | 6 | 13 | 10 | 0 | 30 | 52 |

These are branch-local counts. Each external material/potential group currently
has one recorded preparation-ancestry group; multiple branches do not supply
the corresponding number of independent birth experiments. In particular,
overlapping branches may repeat events. Static Zr cannot supply temporal labels.

The million-atom Al trajectory uses its continuous melt history to establish
ancestry, and only its subsequent measurement phase for prediction origins.
All 42 isolated candidates occur in measurement. Its 10,000 centers cover 27
candidates (64.3%), compared with one (2.4%) for the nested 64-center subset.
There are 4,035,472 eligible measurement windows and 9,351 first center onsets
over the measurement trajectory; the latter includes onsets outside eligible
forecast windows. The large window count does not create independent nuclei.

Regional birth and first-center onset are materially different targets:

| Material / potential | Existing-arrival onset windows, 3 / 6 ps | Local-establishment onset windows, 3 / 6 ps |
| --- | ---: | ---: |
| Al / Lee MEAM | 39,945 / 89,132 | 2 / 9 |
| Al / Mendelev EAM | 12,011 / 34,488 | 0 / 0 |
| Mg / Wilson EAM | 7,181 / 18,775 | 19 / 40 |
| Ti / Kavousi MEAM | 941 / 1,788 | 0 / 0 |
| Ta / Zhong EAM | 18,474 / 43,422 | 7 / 16 |

Other onset classes and negatives remain in the complete exported tables.
Ti is a particular limitation: **7,153 of 8,104 positive 3 ps windows (88.3%)**
are unresolved ancestry or unestablished ancestry. Its primary full-cell
catalogue also contains eight geometrically unresolved establishments and one
ambiguous establishment merge. This protocol is not ready to provide clean Ti
mechanism labels without spatial and orientation-sensitive review.

## Sensitivity and next decision

The predeclared variants change size or persistence only:

| Population | 32 atoms, 1.5 ps | 64 atoms, 1.5 ps | 128 atoms, 1.5 ps | 64 atoms, 3 ps |
| --- | ---: | ---: | ---: | ---: |
| Fixed Al64 | 428 | 348 | 288 | 342 |
| Million-atom Al | 55 | 42 | 32 | 40 |
| Other Al EAM | 0 | 0 | 0 | 0 |
| Mg | 57 | 52 | 38 | 45 |
| Ti | 34 | 17 | 4 | 8 |
| Ta | 19 | 13 | 5 | 10 |

Al's establishment count is relatively stable to longer persistence; Ti is
much more sensitive to both size and duration. These checks do not validate
PTM connectivity, grain orientation, quantization or temporal resolution.

The next dataset should preserve the existing Al64 benchmark for arrival and
add a separately versioned regional-establishment task. Use dense, outcome-blind
centers and origins on existing train ancestors, with a causal risk definition
based on whether the region already contains an established lineage. Requiring
the central atom to be PTM-liquid can exclude ordered precursors. Retain all
held-out source roles; do not move the few existing selection/test positives
into training. Review candidate growth, remelting, periodic geometry and grain
coherence before treating establishments as nuclei. Splitting additional
materials must respect parent/branch ancestry; current coverage does not justify
a random branch split.

This audit establishes label availability, not predictability. No new AP,
likelihood or calibration scores are claimed, and no definition was selected
to improve those scores.

## Artifacts and reproducibility

- [Al64 report and counts](/work/PERSO/vmorozov/analysis/crystallization_origin/al64-20260925/RESULTS.md)
- [External report and counts](/work/PERSO/vmorozov/analysis/crystallization_origin/multimaterial-20260926/RESULTS.md)
- [Al64 exact definitions](../../docs/metrics/crystallization_origin.md)
- [External exact definitions](../../docs/metrics/crystallization_origin_external.md)
- [CPU execution and commands](../../docs/crystallization_origin_cpu.md)
- [Completion verification](/work/PERSO/vmorozov/analysis/crystallization_origin/completion-review-20260926.json)

The completion review verified all 176 source identities, checksums of all 502
result files, twelve frozen metric dependency entries, and the onset-label
partition against each source's denominator for every criterion/horizon.
External denominators total 20,533,887 at-risk windows. Existing CSV exports
retain `tables/METRICS.md` and implementation hashes. Frozen scientific code,
the earlier pilot, sample identities and original result arrays are preserved.
