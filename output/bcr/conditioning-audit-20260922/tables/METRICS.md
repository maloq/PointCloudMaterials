# Frozen-BCR conditioning and paired relaxed-structure audit

Denoising uses the unchanged per-environment noise-prediction NMSE in `bcr.md`:
weighted squared vector error divided by the corresponding weighted Gaussian-noise
norm. The center is uncorrupted and excluded from reconstruction. Smooth radius
weights use clean fixed membership. Reuse all five original noise/d0 levels, two
draws and 384 anchors on six development roots. Chunk size 16 is fixed because
padded shape participates in the original random-noise draw. Correct-code replay
is checked against the published per-anchor errors.

`interventions.csv`: `true` is correct-code NMSE; `other` is the named intervention's
NMSE on the same finite paired anchors. `gain=(other-true)/max(other,1e-30)` is a
fraction, not a percentage. First average draws (and valid donor assignments),
then anchors within root, then roots equally. Strict donors match temperature and
training-fitted count/density/q6 quintiles and must use another root; unrestricted
donors still use another root. Unmatched anchors are excluded from both sides;
`coverage` and `roots` identify each comparison population. Bootstrap 1,000 paired
root resamples (seed 0); intervals condition on the trained seed. Amplitudes use
z_mean_train + alpha*(z-z_mean_train), alpha=0,0.5,1,2. Optimized constant code is
one shared vector fitted with encoder and decoder frozen, using ten training roots
and selection on two other training roots. The original G1 threshold is unchanged.

`fresh_decoders.csv` uses the same paired gain with the fresh decoder on encoder
initialization as `other`, and the indicated frozen encoder's fresh decoder as
`true`. All decoders share initial parameter tensors, training root/sample/noise
streams and 10,000-update exposure budget. These are new decoder fits, not continued
coadapted decoders. An incomplete fit is not included as a completed comparison.

`probes.csv`: standardized RMSE = sqrt(mean((prediction-target)^2)) over observations
and target dimensions. Target and feature means/stds fit only the designated fitting
roots (std floor 1e-6); ridge penalty and zero-initialized residual duration use only
tuning roots. Ridge step zero remains selectable. Selected tuning MSE cannot exceed
ridge tuning MSE, but development error has no such guarantee. Original pilot:
10 fitting/2 tuning/6 development roots, balanced 384 development observations.
Paired relaxed audit: 25/5/15 roots, 1,600/320/960 observations. All roots have equal
numbers of observations. `probe_roots.csv` retains individual root RMSE;
`probe_temperatures.csv` retains temperature strata with the same fitted readout.

`probe_groups.csv` expands the original targets, without fitting additional probes:
12 smoothly weighted Gaussian radial bins, nearest distance, weighted radius,
weighted squared radius, weighted count, weighted density; q4,w4,q6,w6; and 36
upper-triangular cross-radial Gram contractions for each of l=0,2,4,6. Family-wide
RMSE is dimension-averaged and can obscure individual target changes. Degree-zero
moment-Gram components are retained. These are geometric moments, not TDA.

`ridge_tuning_mse` and `selected_tuning_mse` are mean squared standardized tuning
errors. `selected_step` is the residual-network update (zero denotes ridge).
`population=liquid` fixes observed-domain q6<0.35, including relaxed targets. Original
pilot all/liquid populations coincide; paired relaxed populations need not. Raw
prediction archives retain indices, root IDs, standardized targets and both readouts.

Paired tasks distinguish observed→observed, relaxed→relaxed and observed→relaxed;
they use same-frame full 8 Å neighborhoods around identical center IDs, with
independently extracted membership. Transfer comparisons use archived float16
coordinates, not the original high-precision melt assay. No future labels, final
test roots, or calibration roots enter these fits. Pooled cross-temperature scores
do not establish within-temperature information; consult the stratified table.


Table export: 2026-09-22T00:20:18.663037+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
