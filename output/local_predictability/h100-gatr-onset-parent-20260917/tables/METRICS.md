# Native backbone v2: physical means and onset

This is a fresh protocol, not a reinterpretation of the v1 onset fits or their
preflight receipts. Each encoder returns 128 numbers. Both architectures use the
same present MLP, six linear future heads, or six-bin hazard head and the same
seven standardized condition channels.

Physical targets are immutable full-timeline 128-channel packets joined by source,
center ID and frame. All 38,400 native rows are eligible, including crystalline
states. The release's train-only mean and population standard deviation (floor
1e-4) are used unchanged. Future horizons are 0.75, 3, 9, 24, 48 and 96 ps.
Present MSE averages 128 squared standardized errors. Future MSE averages all
6 x 128 errors. The physical training and selection score is their sum with unit
coefficients. Each sampled update has eight windows, uniformly sampled by source
and then row, independent of execution microbatch size.

Evaluation gives each source equal total weight and each row within that source
equal weight. `present_blocks` and `future_blocks` average channels of the named
radial, pair, angular, speed, radial_velocity and moments blocks; future blocks
also average the six horizons. `future_by_horizon` averages 128 channels at the
specified lag. Neither endpoint weighting nor a change-target objective is added.

The 32-window gate is a separate train-only memorization diagnostic, not test
performance: the established eight-source selection, four windows per source,
has its own normalization computed across current plus six future packets. Only
present MSE is optimized in this gate. Pass requires mean present MSE <= 0.10 and
every block <= 0.25 at the same evaluation. Larger fits require a fresh passing
receipt bound to the data, implementation, dependency revision and configuration.

For onset, eligibility is the existing sustained-event risk mask. `hazard_nll`
is the sum of negative log survival probabilities before the event bin and the
negative log hazard at that bin; bin six is right censoring after all six bins.
Evaluation averages this per-window likelihood by source. Saved probabilities
are `1 - product(1 - sigmoid(logits))` through each horizon. The optional onset
export calls the existing `baselines.score_hazard` for threshold, calibration and
window-risk calculations, as defined in [local predictability](local_predictability.md)
and [native onset](local_predictability_native_onset.md). The v2 snapshot screen
does not fit onset models or report those comparisons.
Dense alarm-episode and onset-timing assays remain separate downstream analyses.

Checkpoints minimize selection score on the fixed selection subset (64 rows per
source for physical means; up to 64 eligible rows per source for onset). Full
selection, calibration and test predictions are exported from the chosen
checkpoint. Test outcomes never choose architectures or checkpoints automatically.

Runtime fields: `examples_seen = step * 8`. Training elapsed time includes input
waits, validation and checkpoint overhead within training calls, but not initial
release preparation, gates or explicit profiles. Resumed segments accumulate time.
Profile windows/second = eight divided by mean synchronized update duration after
two warmups; updates include forward, backward, clipping and AdamW. Cold input
preparation and validation have separate times. CPU function profiles distinguish
the unchanged observation/graph producer. Actual attention event names come from
the masked forward/backward CUDA profiler. Allocated and reserved peaks are bytes;
cached observations count toward both. No peak utilization claim replaces these
measurements. Profiles always use FP32; mixed precision is not enabled.

Matched-example curves use the same source/window draws and update checkpoints.
Elapsed-time curves describe these configured implementations on the same GPU;
they are not an equal-wall-clock training-budget comparison. One seed does not
estimate initialization uncertainty. Failed fitting gates establish no comparative
predictive conclusion.


Table export: 2026-09-17T16:19:26.640103+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
