# Shared structural / causal pretraining and frozen probes, version 2

Structural targets and VICReg/SIGReg definitions are those of
[structural pretraining v1](structural_pretraining.md). Each view reconstructs
its own geometry85 and instantaneous TDA144; no velocities or relaxed targets.
The new schedule uses batches of 512 anchor pairs, 12 equivalent epochs of
anchor draws, linear warmup for 10% of updates to peak 0.02 and cosine decay to
0.0002. Microbatching uses exact full-batch gradient caching, including uneven
last chunks. Epoch equivalents count anchor draws divided by training records;
partner/history frames do not increase the epoch count.

Causal targets use the original 150-source Al cohort and 38,400 fixed windows:
90/15/15/30 train/selection/calibration/test sources. Inputs are the current
snapshot or three frames at -1.5, -0.75 and 0 ps. Future geometry/TDA at 0.75,
3 and 9 ps never enters encoder inputs. Current, next and future targets are
calculated in physical Angstrom with the existing instantaneous TDA producer.
Present heads retain the structural release normalization. Forecast mean and
population standard deviation (floor 1e-4) fit only native training sources,
across current and three future target observations. Future losses average
four physical blocks and three topology blocks equally, and three horizons
equally: future = physical_MSE + 0.25*TDA_MSE. Total causal loss adds this future
term to the structural objective, retaining present reconstruction and the
parent representation regularizer. Every fourth update replays the broad
structural data without future supervision. Twelve causal epochs count only
native anchor draws; broad replay draws are reported separately.

Checkpoint selection uses current physical + 0.25*TDA for structural training.
Causal selection adds future physical + 0.25*TDA, averaged over three horizons.
Rows are averaged within each selection source, then equally over sources.
Structural selection has 480 rows; causal selection takes a seeded fixed 64
rows from each of fifteen selection sources. These scores are for selection,
not held-out test results. State_std_mean is the average coordinate population
standard deviation on the selection states, not a whitened distance metric.

Frozen analyses use all 38,400 identical rows for each backbone, separately
extracting selected structural and causal states. Target standardization uses
the causal release's train-only forecast mean/std for every model. Ridge
regularization is chosen on selection sources. The nonlinear probe is a
128→256→256→916 residual on the selected ridge, with a zero last layer and the
ridge itself as its initial candidate. Its 1,000 updates use the declared
warmup/cosine schedule, and selection labels choose its checkpoint. Target
blocks and four horizons receive the same weights as above. Neither test nor
calibration targets affect fitting or scaling.

Baselines are per-channel/per-horizon training means, a temperature-only ridge,
and persistence of each observed current physical target. Direct trained heads
are evaluated separately: structural heads only have current outputs; causal
heads also have future outputs. Missing structural future heads are not scored.
Raw latent MSE is not a cross-backbone quality metric.

Test metrics average per-row standardized MSE over channels within each of
radial32, pair32, angular16, moments5, H0-16, H1-64 and H2-64. Errors average
within source first, then equally over independent sources; physical is the
mean of its four blocks and topology the mean of its three. Results are shown
for all test rows, current noncrystalline rows (PTM class outside 1/2/3), and
separate temperatures. Noncrystalline filtering occurs only at evaluation.

Positive gain_over_persistence means lower candidate error: baseline minus
candidate standardized MSE, separately by horizon and block. The confidence
interval is the 2.5/97.5 percentile of 4,000 paired whole-source bootstrap
resamples; it does not quantify training-seed uncertainty. Test source counts
and available rows accompany each population. All models use seed 20260919.
Timing, allocated/reserved GiB, learning rate, input wait and gradient norms are
operational diagnostics, not independent research observations. Peak VRAM
varies with observation size; a 40 GiB allocator budget bounds cached memory.


Table export: 2026-09-17T22:51:40.108648+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
