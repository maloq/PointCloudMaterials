# Sensitivity, descriptor learnability and paired relaxation

User-requested extension, 2026-09-28. Three separate questions:

1. Can MACE and boosting recover geometry-dependent information when its amount
   is known, including a genuine zero-dependence control?
2. Can an end-to-end geometry-only MACE128 and context predictor represent all
   3536 rich descriptors through a 128-D context state?
3. Does archived full-cell relaxation change predictable information, and does
   the conclusion depend on preserving original labels or recomputing relaxed ones?

Use seven generated-label datasets: zero, local bond-order 1/2/5%, and spatial
order-gradient 1/2/5% oracle RMSE gains. Calibrate only on training inputs and retain
the complete fixed original cohort. No new simulations. One fit seed and one
label realization; this establishes recovery on declared controls, not universal
power or zero physical mutual information.

Feature learning is explicitly reauthorized by the current user request. It is
a standalone target/readout experiment, not automatic pretraining of the crystal
predictor. Four descriptor families receive equal objective weight; constant
training targets remain exported. Gaussian likelihood selects checkpoints; VCReg
remains active and separate. Compare matched raw and cold learning, with a separate
full-raw reference. No species/time/temperature covariates or physical labels enter
encoder inputs. All encoders use random initial weights and shared patch encoding.

Relaxation comparison is a 2×2 factorial: raw/relaxed inputs × old/new labels.
All four fits use the same available, jointly crystal-free rows and source roles.
New labels mean instantaneous >=64-atom PTM clusters after quenching, not a claim
of sustained MD crystallization. Original labels retain their causal establishment
definition. Full-cell relaxation uses external context and archived float16
coordinates: report that observation advantage and quantization explicitly.

Reuse all eight GPU CatBoost descriptor combinations, MLP, no-input distribution,
training mean, ridge and linear-logit baseline. Selection is validation NLL;
AP is not an objective. Compare source-paired physical scores at fixed target
definition. A different-label score is not evidence of an improved predictor.

Required outputs: oracle versus recovered NLL/RMSE; null-control false gains and
source intervals; feature-level/family-level fidelity; raw/relaxed paired scores;
cohort coverage, original/new distances and visibility changes. Positive control
success strengthens a negative real-data finding only for the signal forms tested.

[Recipe](../../configs/liquid_predictability/controls_relaxed_20260928.json) ·
[Operations](../../docs/liquid_controls.md) ·
[Metrics](../../docs/metrics/liquid_controls.md).
