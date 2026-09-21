# Dense fixed-grid frozen crystallization test

The test retains 30 historical independent-source roles; no source is reassigned
to increase event counts. Selection/calibration/test origins are frames 64,80,...,656
(12 ps cadence). Readout training uses the existing 15-origin training release.
Natural at-risk windows, original MD sustained-onset labels, source weighting,
hazard likelihood, horizon bins [.75,3,6,9,12] ps and 5% calibration-FPR thresholds
use crystallization_information.md. All compared inputs share the retained rows.
Timeouts remove matched rows with counts; other failures remain fatal.

Planned test coverage is 11,256 windows and 338 distinct (source, tracked atom,
first-onset-frame) events across 27 event-bearing sources. Distinct local events
are not independent nucleation events. `population_audit` reports windows,
positive windows, deduplicated local onsets, all sources and event-bearing sources
separately. At this cadence a local onset appears in at most one positive window.
Actual counts after exclusions are recorded and override planned counts.

The exported intervals use 1,000 paired whole-source bootstrap draws within the
five temperature strata. All observations from each drawn source retain the same
multiplicity. Hazard NLL, AP and classification recall retain source-equal weights;
detected timing MAE, missed-event fraction and timing-within-3-ps recall preserve
their original event-window weighting. AP groups tied scores before precision
integration. Intervals are the 2.5/97.5 percentiles of valid draws; missing-event
or missing-detection draws are undefined, not zero, and valid draw counts are saved.
These intervals do not include seed uncertainty or multiplicity correction.

Paired event-NLL gain is reference minus candidate; AP gain is candidate minus
reference, using identical bootstrap draws. References are unrelaxed hot-control,
parent with relaxed inputs, and relaxed geometry, within each readout family.
The calibration thresholds and validation-selected readout weights stay fixed in
the test bootstrap. It estimates test-source sampling uncertainty only.

All encoders are frozen. Linear and width128 neural readouts share input slots,
seed, 1,024 updates and batch512. New SIGReg/VICReg variants, earlier pilot states,
the parent under both input domains, and five historical frozen states are included.
Historical cached MACE/GATr exports are subset by exact source/anchor/atom row keys,
with checkpoint, population and per-source feature hashes verified. They use
unrelaxed inputs. Report them separately from relaxed-input encoders. No new
encoder hyperparameters are selected by this test; the source split has been used
historically and is not claimed to be an untouched confirmatory test.

Explicit user-stopped encoder runs can be excluded through a plan-bound
`technical/evaluation-exclusions.json`. Reports disclose the reason and adjust
the expected readout count; this does not filter observations or change metrics.
