# Frozen backbone instantaneous TDA readouts

Protocol `backbone_instantaneous_tda_v1` compares the **frozen physical snapshot**
MACE/cuEquivariance and axial GATr encoders selected after the matched 2,048-update
H100 screen. This is present-information retention, not future-TDA forecasting,
relaxed topology, or TDA-supervised encoder training.

## Targets and population

All 38,400 frozen native windows: 150 independent source lineages, 16 tracked
centers/source, 16 current anchors/center. Existing whole-source splits contain
90 training, 15 selection, 15 calibration and 30 test sources. PTM labels and
event risk never select the fitting population. Original coordinates retain the
existing trajectory quantization; no minimization or new simulation is performed.

For each anchor, select the 80 nearest atoms **including the center** from the
same periodic, center-relative float32 chart observed by the encoder. Sort by
squared radius, then atom ID on exact ties. `liquid_structure.persistence_image`
computes the established 144-vector: H0 = 16 bins, H1 = 64, H2 = 64. It uses a
safe alpha complex, coefficient field 2, square roots of alpha filtration
values (Angstrom radii), finite deaths <= 3.5 Angstrom, and its fixed Gaussian
grids/lifetime weights. Nearest-80 membership and death filtering are hard
boundaries: this experiment does not establish descriptor smoothness.

## Readouts and separation

The train-only descriptor mean centers all channels. For block b, let
`s_b^2 = mean_channel(var_train(y_b))` (population variance, ddof=0).
`f = 0.05 * max_b(s_b)` and `scale_b = sqrt(s_b^2 + f^2)`.
The existing topology-target producer computes these scales; its PCA outputs
are saved for provenance but **no PCA compression is used**.

For each encoder, standardize its 128 exported channels using only training
rows; fit ridge with an intercept and select a single alpha from
0.1, 1, 10, 100, 1000 by selection-source balanced TDA MSE. A second readout adds
a two-hidden-layer, 256-wide SiLU residual to that ridge prediction. The last
layer starts at zero, so checkpoint zero equals ridge. Train the residual with
equal-block standardized MSE, AdamW (lr=.001, decay=.0001), uniform training-row
batches of 1024, one seed 20260919, and at most 3,000 updates. All training
sources have 256 rows, so uniform rows also balance sources. Select every 100
updates, stop after 8 evaluations without improvement, retain the best including
step zero. Same architecture, seed, sampling and maximum budget for both states.
Calibration/test targets never affect fitting or selection. Baselines are the
training descriptor mean and a separately selected ridge from the seven frozen
condition channels. Nonlinear probes receive only z; no conditions or raw atoms.
Encoder extraction and nonlinear readouts use float32 with TF32 disabled; ridge
is solved in float64 and its saved predictions are float32.

## Exported quantities

For every reported population, first average rows within each source and then
average sources equally. `balanced_mse` averages the H0/H1/H2 block MSEs after
division by `scale_b^2`; lower is better. Each `blocks.Hb` exports raw `mse`,
`scaled_mse`, `r2 = 1 - mse / variance`, and
`within_frame_r2 = 1 - mse / within_frame_variance`. The global variance uses
the evaluated population's source-weighted descriptor mean. Within-frame
variance centers targets around the evaluated centers' mean for the same
(source, anchor), then applies the same source weights. These are denominators,
not additional fitted models. Undefined zero-variance R2 is blank/null.

Report all states, each temperature, and observed **noncrystalline** centers
(current PTM type not in {1,2,3}, matching the existing local assay). This subset
includes unknown PTM environments and must not be called a certified liquid
phase. Selection populations are diagnostic; test remains the primary endpoint.

For matched test rows, `gatr_relative_to_mace` reports
`1 - mean_source(error_gatr) / mean_source(error_mace)` for linear and nonlinear
readouts separately, with 4,000 paired whole-source bootstrap draws and the
2.5/97.5 percentiles. Positive means lower GATr error. The same masks and sources
are used on both sides. One seed means these intervals quantify source
uncertainty only, not training-seed variability.

Targets, row keys, encoder states, predictions, target scales, selected readouts,
checkpoints, source hashes and implementation hashes are retained. Parent
selection/calibration/test embeddings are independently checked against fresh
cross-device extraction (`rtol=atol=2e-4`); all row keys and artifact checksums
must match exactly.


Table export: 2026-09-17T17:11:18.277527+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
