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
