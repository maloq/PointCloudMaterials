# Consolidated local predictability results, 17 September 2026

This export collects existing studies and evaluates saved predictions. It performs
no fitting, checkpoint selection, simulation or hardware benchmark. Older outputs
and their metric definitions remain immutable. Input SHA256 hashes are recorded
in `technical/inputs.json`; scorer and collector hashes are in the metric contract.

## Onset and observability

`onset_test.csv` retains two populations: descriptor origins every 3 ps and the
native origins every 30 ps. Descriptor predictions are restricted to the native
origins and recalibrated on the corresponding calibration subset using the
existing `baselines.score_hazard`. All native test comparisons verify exact
source/center/anchor identities and event labels. This matches evaluation, not
training budgets: descriptor hazards were trained on denser origins.

The six-bin first-event joint NLL sums negative log survival before the event and
negative log hazard at the event; censoring survives all six bins. Horizon binary
log loss instead scores cumulative event probability and clips it to
[1e-7, 1-1e-7]. They are distinct metrics; saturated predictions can make even a
one-bin logit NLL differ from clipped probability log loss. Brier averages squared
probability errors; AP is weighted noninterpolated average precision. Every source
receives equal total weight, distributed equally across its eligible windows.

Thresholds use only calibration sources, with complete ties and window FPR <=5%.
Reported test FPR need not satisfy that bound. These are window metrics, not
dense alarm episodes, fixed-lead warning or missed-event-aware timing. No test
threshold fitting or probability recalibration is introduced here.

`observability.csv` uses the same scorer with one bin and positive event_bin=0.
Current and endpoint state tasks include all 7,680 test windows. Future-sequence
onset uses the 4,691 at-risk test windows. Future inputs, including confirmation
frames, are diagnostic oracles, not forecasts. Raw current-state outputs are
mapped explicitly from the producer's binary_target/source_id/center_id fields
and paired with packet labels. There are 30 test sources in each case.

`paired_onset.csv` computes candidate minus reference (negative favors candidate)
for joint NLL and each horizon's log loss/Brier. Both models share all test rows
and labels. Differences are first averaged within source, then equally across
sources. The existing temperature-stratified bootstrap uses 1,000 resamples,
seed 20260919, and percentile 2.5/97.5 bounds. Intervals are exploratory,
unadjusted and conditional on one fitted seed; no AP difference intervals or
training-seed uncertainty are inferred. `paired_descriptor_original.csv` is an
unchanged copy of the earlier descriptor-only comparisons on both grids.

## Physical and topology outcomes

`physical.csv` uses all-state native rows, the existing train-only standardized
128-coordinate packet, and six horizons 0.75/3/9/24/48/96 ps. MSE averages
coordinates, then sources equally; future_mean additionally averages horizons.
Present MSE is separate. Ridge prediction targets and native targets are paired
by exact source/center/anchor and verified numerically (1e-5 tolerance for saved
float32 conversions). Native present/future aggregate rows retain the producer's
float32 error reduction. Per-horizon errors, ridge means and paired differences
are evaluated in float64; tiny differences between these reductions are expected.
Persistence scores are copied from the original physical producer.

`paired_physical.csv` averages paired coordinate-MSE differences within source
and uses the same temperature-stratified bootstrap. Negative favors candidate.
The ridge and backbone comparison has matched physical observations/targets and
test rows; model parameterization, fitting and selection procedures differ.
H100/RTX screens repeat one seed and sample schedule, not independent seeds.

`topology.csv` is copied unchanged from the topology-retention export. It measures
current instantaneous H0/H1/H2 persistence targets with equal block weighting,
frozen physical encoders, and separately fitted ridge/nonlinear readouts. Its
relative error-reduction intervals are defined by the original
[TDA contract](backbone_tda.md); positive reduction favors GATr. Noncrystalline
windows include unknown PTM labels and are not a pure liquid phase assignment.

`h100_speed.csv` is copied unchanged from the
[backbone comparison](backbone_comparison.md). It times resident batch-eight FP32
updates on the same H100. It is an engineering measurement, not full-run speed,
an equal-wall-clock scientific trial or a comparison with H200.

`completion.csv` records completed study receipts. Stale per-step running logs do
not override final completion records. Fit counts are not independent replications.
Earlier H200 summaries are preserved verbatim as a reported JSON source; no new
H200 physical study is inferred from its preflight/handoff records.


Table export: 2026-09-17T19:03:52.543352+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
