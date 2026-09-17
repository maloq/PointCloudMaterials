# Cross-study memory research snapshot

This family freezes evidence from two distinct protocols. No score, interval,
or source count is pooled across them. `technical/inputs.json` records SHA256
hashes of every file read; `technical/evidence.json` retains the captured values.
The original completed runs retain their own frozen metric contracts.

## Older causal-state study

`causal-*-summary`, `paired-differences`, and `sufficiency` copy selected metrics
from completed `mace_causal_comparison` exports, including their source intervals.
Physical `block_mean` is the equal mean of six training-standardized blocks
(structure, TDA H0/H1/H2, even and signed motion), then an equal-source mean.
Seed means are formed within source before 2,000 source-bootstrap resamples.
Paired differences are left minus right: negative favors the left MSE.
`low_order` means current group qbar6 < 0.30, not an independently assigned phase.
Temporal J uses 0.75 ps differences and each source's embedding covariance;
it is dimensionless and is not an error in physical target space.

`causal-gaussian-ablation` uses the existing comparison producer's `source_values`
to form the same six-block physical means per source, then averages sources
within each seed. These are jointly trained heads, not the frozen nonlinear
readouts. Gaussian NLL and one-sigma coverage are available only for the all-source
population; deterministic heads have neither metric. No new intervals are fitted.
See the originating `mace_causal` and `mace_causal_comparison` contracts for exact
target standardization, coverage, event, and jump definitions.

## Partial-observation memory study

`memory-fits` copies completed fits' validation/test `joint_nll`, `future_mse`,
and `present_mse` means and their validation-selected update. Completion requires
both status and metric update counts to equal the recipe budget. Pending fits
appear only in `fit-progress`; no partial fit enters a matched cohort mean.

Joint NLL is the negative log density of the 640-coordinate standardized future
path divided by 640, in nats per coordinate per lag. The five lags are 0.75, 3,
12, 48, 96 ps. MSE equally weights packet coordinates and lags, unlike the older
six-block metric. Train-only normalization is common to present/future packets.
Means first average three windows within a source, then weight sources equally.
`memory-paired-gains` copies the original 500-source-bootstrap estimates per seed:
positive means snapshot/control NLL minus real-history NLL favors history.
These intervals do not measure training-seed uncertainty.

`state-use` copies frozen-head intervention and validation-selected ridge scores.
The constant-state NLL increase is intervention minus original score: positive
means replacing the state by its training mean hurts the fitted head. It is not
a full-history sufficiency test. Current-packet ridge has access to velocities,
so it is observation-matched only to xv models. Full physical-baseline metrics
are retained in `technical/evidence.json`. Exact definitions are frozen in each
originating `predictive_memory` export.

## Remote evidence, arithmetic, and progress

H200 means and qualitative interval findings are supplied by the user. Raw H200
artifacts and numerical interval endpoints were not available at capture time.
`causal-h200-reported` preserves rounded three-seed means; no intervals are
invented. `memory-width-comparison` computes local width-16 means over exactly
the two prescribed completed seeds, checks agreement with the reported rounded
reference to 0.00005, and subtracts that mean from the reported width-32 mean.
Missing/duplicate seeds are errors. No aggregate seed-confidence interval is
computed. Relative MSE reduction in the narrative is 100*(control-candidate)/control;
NLL differences are reported in their original units, not as information gain.

`data-production-status` records source completion and verified publication,
not model results. Position RMS and maximum error are the paired converter's
whole-file float16-minus-float32 componentwise coordinate errors in angstroms;
they are not target errors or minimum-image relative-position errors. Sealed-test
physical outcomes are not opened. Missing quantities remain blank, never zero.
Plots show point estimates with each protocol on its own axis; absent remote
interval endpoints cannot be reconstructed from qualitative significance claims.


Table export: 2026-09-17T11:48:20.983879+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
