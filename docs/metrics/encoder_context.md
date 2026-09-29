# Encoder training and directional-context evaluation

The fixed Al64 all64 cohort retains 90/15/15/30 independent source roles and
43,523/20,883/16,848/45,291 prospective windows. The same immutable rows are
used for every predictor; legacy16 is a separately reported subset, using the
calibrator fitted on the complete calibration population. These are historical
test sources. No model or checkpoint is selected from test AP.

An epoch visits every training row once, in a seeded permutation, including the
last partial batch. Supervised NLL multiplies each row by N/(sources * rows in
that source). It uses no event oversampling. With batch 256 this is 171 updates
per epoch, including a three-row final batch. Supervised encoders, context
predictors and frozen readouts run 24 epochs (4,104 updates), with checkpoint
selection by source-weighted validation NLL restricted to epoch 12 onward.
Fixed epoch-12 snapshots are retained. This differs from historical replacement
sampling and early-checkpoint selection and must be reported as a new protocol.

Structural pretraining never accesses event labels or calibration/test sources.
Physical pretraining uses 1,157,760 current observed patches, 192,960 validation
patches, and 12 passes. Targets are 24 Gaussian radial weighted counts (centers
0.5..7.5 A, sigma 0.3 A), two tapered counts at 5/8 A, and l=2/4/6 powers of the
weighted mean spherical harmonics at those supports. The center is excluded.
Targets are standardized with training means/stds; radial/count/angular blocks
receive equal total MSE weight. A nonlinear decoder reads the exported state.

Paired VICReg and Epi-with-variance use 86,400 observed/relaxed same-time training
pairs and 36,480 validation pairs without an onset risk mask, for 12 passes.
VICReg = (25 alignment + 25 variance + covariance)/51 on the exported 128-D state.
Epi-with-variance = (25 alignment + 25 variance)/51 - 0.1 E/E_initial, using the
existing normalized ridge/logdet producer and a frozen initial random-MACE
128-to-64 reservoir for each view. It is a geometric adaptation, not an empirical
estimate of predictive information. All pretraining exports use fixed epoch 12;
validation structural scores are diagnostic, never onset-based selectors.
Normalization of initial pooled features uses 8,192 seeded training-only rows.
Both domains subsequently receive independent supervised fine-tuning.

The separate `multimaterial256_20260925` recipes use 14,058,484 native-unit
Al/Mg/Ti/Ta/Zr physical-training patches and 345,600 native Al relaxed pairs;
selection remains 192,960 / 36,480 patches/pairs. Their geometry-only width-128
encoder has 634,496 parameters and one constant Al channel for all materials.
Before graph construction and target calculation, coordinates are multiplied
by the fixed training-calibrated cutoff ratio `scale_Al / scale_material`.
No material ID, species input or scale feature enters the encoder or decoder;
Al coordinates are unchanged. Streaming epochs shuffle blocks of 16 shards
then rows, covering every row exactly once including the final partial batch.
For this release, both target and initial-pool normalization use the recorded
8,192 training-row sample. Epi references are computed on demand by the frozen
initial encoder, rather than held as a whole-dataset GPU reservoir. The loss
formulas, fixed epoch-12 export, downstream NLL selection and fixed evaluation
rows are unchanged. This is a broader-data comparison, not an
isolated change of pretraining loss. See the [data contract](../datasets/structural_multimaterial_256.md).

Predictive metrics use the existing first-event/survival six-category NLL and
source-weighted AP, raw/calibrated Brier, binary log loss and fixed calibration
threshold recall/FPR. See [supervised metrics](supervised_onset.md) and
[context metrics](equivariant_context.md). Every encoder is frozen before its
two context predictors are trained. Local linear and MLP onset probes and a
32-dimensional physical-feature MLP provide information/readout controls.

Physical retention uses a ridge-1e-3 linear decoder with unpenalized intercept,
training-source weighted feature/target standardization and no test tuning.
Report standardized MSE per radial/count/angular block and per-target R2.
Undefined constant-target R2 is null, never zero. This assay is on at-risk
windows, so it measures variation within the declared noncrystalline population.

Whole-population, training and test embedding spectra use centered,
source-weighted covariance. Participation/entropy ranks and d95 are linear
spectral dimensions, not nonlinear intrinsic manifold dimension estimates.
Observed-input dynamics use every stored frame of all 64 fixed centers in all
30 test sources, at exactly 0.75 ps; raw input neighborhoods are rebuilt each
frame. Normalize jumps by sqrt(2 * trace(training at-risk covariance)). Report
RMS, p95/p99 and uncentered velocity second-moment movement spectra, with
noncrystalline endpoint subsets separately. No relaxed-at-0.75-ps trajectory
exists; mark that assay unavailable instead of feeding observed coordinates to
a relaxed model. The reference covariance is from the fitting population, not
a separate dense equilibrium ensemble.

Encoder noise response uses the existing center-fixed Gaussian perturbation
assay at 0.1%, 0.5%, 1% and 3% expected 3-D RMS relative to the local mean
center-to-12-nearest-neighbor distance. Report realized relative RMS, Angstrom
RMS and response normalized by training covariance. Edges are rebuilt; selected
atom membership is retained. For relaxed inputs this perturbs supplied relaxed
coordinates; no re-quench is run. It does not establish robustness of complete
full-cell relaxation or context representative selection.

Comparisons to scratch match coordinate domain, context predictor and sample IDs.
Intervals resample whole test sources 1,000 times, candidate minus scratch:
negative favors the candidate for NLL, positive for AP. They quantify source
uncertainty conditional on one seed and are not multiple-comparison adjusted.
The different pretraining populations and total compute preclude claiming a
pure isolated regularizer effect. NLL reductions measure predictive benefit
within the fitted models, not mutual information.

Execution update (2026-09-25): shared geometry and a six-encoder LRU feature
cache change storage/scheduling only. Predictors share train-only normalization
statistics for common fields, with identical per-variant inputs. CPU calibration
and source bootstrap now read the selected-checkpoint raw probabilities in a
separate worker; formulas, populations, calibration split and selectors are
unchanged. GPU fitting completion and final metric completion have separate
receipts. Full diagnostics follow core fitting.


## Mechanism queue extension (2026-09-26)

Streaming pretraining now records configured fixed endpoints and milestone receipts, shared train-only initial states and Epi projection tensors. Historical recipes still request12 epochs.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.
