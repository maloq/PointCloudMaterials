# Quality audit of the original 0.75-ps Al birth campaign

This is a new aggregation of frozen v3 lineage results, not a new detector,
simulation or predictor. The source plan identifies 150 `historical_fixed_cohort`
trajectories, each from a distinct melt ancestor, with native 0.75-ps observations.
Verify all manifest identities, actual saved timestep spacings, frame counts and
input-deck equilibration lengths. Velocity availability means the recorded array
exists; missing 0.10-ps velocities are not reconstructed.

The v3 growth-audit, source-results, paired-comparisons and matched-events tables
must match their frozen table-checksum receipts. Their definitions remain intact:
eight atoms begins an appearance record; isolated liquid-origin independent
establishment is 64 atoms for a sampled span of 1.5 ps. These are operational
thresholds, not a physical critical nucleus or a continuous-time certificate.
At 0.75 ps the span uses three saved observations; at 0.10 ps it uses sixteen.

`embryo_appearances` sums `liquid_emergences`. `established_liquid_origins` and
`established_births` sum `established_liquid_emergences`. Source counts with a
birth count rows where this value is positive. Initial embryo records sum
`initially_present_records`; initially embryonic sources have an initial largest
component >=8 atoms, and initially established sources use >=64. Established
composites sum `established_composites_from_subcritical`; they must not be added
to roots as extra independent births. Source-temperature tables are descriptive
and do not provide model covariates or unbiased nucleation-rate estimates.

Matched-cadence count comparisons reuse exactly the same sources within each
population: 22 uniform birth-campaign runs (native0.15 ps), two completed dense
runs (native0.01 ps), and three memory runs (native0.075 ps). Historical partial
copies are excluded. Exact075/030 grids are not nested, but both are exact views
of the same trajectory. Native0.10 ps cannot be subsampled to exact0.75 ps.
The two dense descendants share ancestors with the original campaign and do not
increase its independent-melt count. Summed counts measure cadence sensitivity;
they are not event-matched recall, precision or ground-truth accuracy.

Detector agreement reuses the frozen onset-member/time-overlap matching rule for
all root appearances, including initial and uncertain origins. Report summed
left/right/matched counts, mean matched origin/fate agreement and median absolute
matched onset-time difference. These values are conditional on that matcher,
include tiny transient records, and cannot be interpreted as established-nucleus
accuracy. Similar established totals do not prove identical detected events.

Historical prediction coverage is copied from the September30 strict preappearance
cohort, with its original target/filters. It is distinct from v3: do not divide its
birth count by the v3 established count or claim it exhausts eligible v3 histories.
The audit records missing0.75-ps per-event v3 shards explicitly. Summary tables
can still be assessed, while a new established-only event-matching analysis would
require regeneration from the retained MD. No missing labels become negatives.

The source CSVs, definitions, implementation hashes and input receipts accompany
the plot. No confidence interval or detector precision/recall is estimated without
an independently labelled reference population.


Table export: 2026-10-07T17:52:54.387878+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/al075_birth_quality.json` relative to the analysis root.
