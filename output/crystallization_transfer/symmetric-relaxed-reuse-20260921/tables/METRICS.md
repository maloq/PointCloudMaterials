# Symmetric structured-context forecasts

Calculations follow `crystallization_paths_refinement.md` and `context_night.md`:
source-weighted dense integrated Brier and event NLL, calibrated classification,
timing with missed-window counts, restricted mean-time error, physical/path MSE
and marginal CRPS, sampled-center spatial scores, and the ≤12 ps hazard block.
No best-of-sample metric or future teacher forcing is used during evaluation.

The population, onset labels, train/selection/calibration/test ancestry split,
origin and forecast grids are unchanged. Moments come only from training sources;
checkpoint selection uses open-loop selection Brier. Sources are the independent
unit; overlapping origins are not independent events.

Inputs now have 25 structured spatial queries on two cuboctahedral shells. Real
assigned atom offsets are retained. Spatial and causal per-slot temporal attention
precede pooling. Shared observed descriptor-history and shell inputs are additional
physical information, so this is not an embedding-only test.

Unlike the older fixed-MACE-target study, each frozen backbone supplies its own
128 latent targets. Compare event scores and the shared physical, bond-order and
crystallinity blocks across MACE/GATr. Raw latent errors or combined structural
training losses are not cross-backbone measures. Original checkpoint local supports
are preserved: 7.94 Å MACE and 16.87 Å historical step-3072 GATr.

The 36-epoch ceiling and early stopping are recorded separately from actual updates
and selected checkpoint. One seed; no claim of training-seed uncertainty. Historical
results used different context heads and initialization and are reference comparisons.

The relaxed-MACE variant uses the validation-selected cold-input/cold-target
checkpoint. Both observed context features and future latent targets use relaxed
structures. Its descriptor-history/shell auxiliaries use relaxed geometry;
velocity-independent columns only. Physical forecast labels and event onsets
remain original MD, with identical score definitions and source splits. This is
an encoder-and-observation-domain comparison, not an isolated checkpoint ablation.

For the archived-reuse comparison, both observed and relaxed arms use identical
available origins and three actual observations within 72 ps. No temporal
interpolation or duplicate-frame padding is used; attention receives actual
physical offsets. The two historical descriptor differences replace the dense
reference's three differences; its unused third difference block is zero in both
arms. Context query identities are selected in the observed geometry and retained
across quenching, with actual relaxed displacements supplied in the relaxed arm.

Both reuse arms predict the same original MACE latent timeline and original MD
physical/event targets through 96 ps. Their initial target-space state is decoded,
not copied from input embeddings. Hence no unobserved relaxed future is imputed.
Archival float16 relaxed coordinates are an explicit approximation: the precise
benchmark comparison is exported in `technical/precision.json`. Reuse is not a
replication of the dense original history protocol.

For irregular archived origins, `timing_grid_ps` is null. Alarm episodes are
computed over successive observed decisions, not continuous monitoring. The
continuous-exposure false-alarm rate is omitted and replaced by
`false_alarm_episodes_per_1000_observed_origins` = 1000 times the number of false
alarm episodes divided by evaluated forecast windows. Calibration sources still
set thresholds; test sources never select encoders or heads.


Table export: 2026-09-21T22:30:28.776186+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
