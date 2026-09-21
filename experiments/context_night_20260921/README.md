# Overnight: preserve local information and supply missing predictive context

Recipe `configs/context_night/night_20260921.json`; live scientific results:
`output/context_night/night-20260921/RESULTS.md`. One seed. Existing Al Lee2003-MEAM
sources only, historical90/15/15/30 train/development/calibration/test roles.
No new MD. The independently running relaxed-input/target pilot remains separate.

## Local snapshot MACE — five fits

Four matched1024-update continuations start from the completed SIGReg-direct width64
checkpoint, batch512, compiled BF16, existing VRAM-tiered execution. Encoder/head
peak LR5e-5/5e-4; warmup/cosine. Same architecture and train-only added linear heads.

| Variant | Nonlinear order weight | SIGReg weight | Linear order/angular weights |
|---|---:|---:|---|
|continued-control|0.25|0.10|0 /0|
|strong-order|1.00|0.01|0 /0|
|linear-information|1.00|0.01|1 /0.25|
|linear-no-reg|1.00|0|1 /0.25|

Physical/TDA/moment/JEPA anchors remain. Exported field of view remains local;
we do not force an80-atom input to reconstruct unobserved25Å context. Choose one
arm by development Physical +0.25 TDA +0.25 order and run2048 additional updates
with halved learning rates. This continuation has a fresh optimizer/schedule in a
new run directory; exact within-run resumes preserve optimizer/RNG state.
All five checkpoints receive the historical linear/MLP frozen crystallization
assay and short0.75–12ps probes (linear/MLP embedding-only, MLP embedding+shells).

## Trajectory prediction — twenty fits

Reuse each family's previously development-selected12-epoch configuration and
checkpoint. Direct/AR-MSE use48ps embedding histories and25Å representative context;
mixture/diffusion retain their selected12ps configurations. Shared observed additions:
current local geometry/order93; changes over3/6/12ps; and four shell count/radius
summaries at7–17Å and17–25Å. All504-slot heads have identical capacity; initial final
auxiliary projection is zero so ordinary arms start at the parent predictor.

Fourteen18-epoch screens, batch128, head LR1e-4, cosine decay, development early stop:
- Direct and AR-MSE: continued control, shells, local+history descriptors, both,
  and descriptor-only control without learned context features (five each).
- Stratified mixture and32-step diffusion: continued control and both (two each).

Four longer36-epoch fits choose one setting per family using development Brier,
within10% of the family's best development physical MSE. These start from the same
original parent with a longer schedule, rather than silently extending an optimizer.
Two final18-epoch direct/AR tests add the development-selected new encoder's current
snapshot features to existing embedding context and the combined descriptors.
These test incremental utility; they do not replace all spatial/history encodings.

All forecasts remain96ps, with dense original-MD onset evaluation and explicit
0.75/3/6/9/12ps diagnostics. Future latent/physical targets are fixed across methods.
Use physical MSE, CRPS, event Brier/NLL/AP, timing including misses, calibration and
spatial metrics. No best-of-generated-samples selection. No future values in inputs.
The dense historical cohort supplies44385 test windows; windows overlap, and test
sources have been inspected before. One-seed conclusions remain exploratory.

## Questions answered

Does linear supervision improve accessibility of local order/angular information?
Does weakening regularization help without losing physical anchors? Does explicit
wide-field density/context outperform sparse representative embedding attention?
Does observed descriptor history add beyond embedding history? Do learned features
add beyond those descriptors? Does an improved local encoder help the same fixed-
target trajectory problem? Are probabilistic predictors helped by the same inputs?
