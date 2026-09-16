# Paired causal MACE pilot comparison

Inputs are completed `mace_causal` test-source tables and paired physical
prediction arrays. Every encoder, readout and seed must have exactly equal
source IDs, center IDs, anchor times and present/future physical targets, in
the producer's order. Seed recipes may differ only in seed, output and the
corresponding D reference checkpoint. All other scientific settings match.
Each input table retains its original metric contract and checksum.

Individual physical errors and temporal metrics retain the definitions in
`mace_causal.md`. Additional `block_mean` metrics average the six producer
blocks equally: bond order, instantaneous TDA H0/H1/H2, even motion and signed
motion. They never average latent errors across representations. A's untrained
joint future, delta, path and hazard outputs are omitted. All variants have
matched trained linear/nonlinear frozen readouts for the future comparison.

For each model/readout/metric, average initialization seeds within each source,
then average sources equally. The 95% interval uses the configured number of
whole-source bootstrap resamples (2,000 in this study). Seed averages are held
fixed during this bootstrap, so intervals quantify source sampling uncertainty,
not uncertainty over training seeds. One-source intervals are undefined.

Paired differences are left minus right within the same seed and source,
averaged across seeds, then across sources. Bootstrap entire source differences.
Negative differences favor the left model for MSE, NLL, Brier and J; metrics
such as covariance trace do not have a universal preferred direction.
No joint future comparison against A is exported. E has additional training
updates after D and must be read as a second-stage constrained tradeoff.

Event rows are copied for each seed/readout, without averaging nonlinear ranking
or threshold metrics. Their validation-fitted thresholds and source-weighted
definitions remain those in the originating metric contract. The plot reports
low-order held-out physical errors, measured-current persistence, and declared
0.75 ps J. It is descriptive and never selects a checkpoint from test results.


Table export: 2026-09-16T19:01:12.652961+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
