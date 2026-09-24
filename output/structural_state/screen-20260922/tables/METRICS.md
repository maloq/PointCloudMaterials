# Fixed structural-state objectives and evaluation

Training targets are radial17 and moment-Gram l2/l4 (36 each), produced by
src/training_methods/bcr/probes.py at 8 Å. Each block contributes its mean squared
error with equal block weight. Means/stds use the 25 fitting roots only; std floor
1e-6. C adds 0.25 times the relaxed-target error. D adds 0.1 times Huber loss
(delta1) between normalized paired distances. Teacher distance is the square root
of block-balanced geometry MSE; learned distance is RMS coordinate difference in
exported128. Teacher scale is a frozen train-pair mean, learned scale a detached
full-batch EMA0.95 initialized on the same train pairs. Head Frobenius norm cap10;
gradient norm cap5. All scientific metrics retain full precision in CSV/JSON.

The primary evaluation uses matched update2048, not the best tuning checkpoint.
Training heads receive exported z only; frozen evaluation readouts also receive
the same five temperature indicators for every encoder/control. Pooled64,
descriptor89 and PCA64 features are padded to128 readout slots. Descriptor PCA
is fit on block-scaled, train-standardized geometry. Ridge penalties are selected
from 11 logarithmic values 1e-6..1e4, with an unpenalized intercept. A zero-output
residual MLP is tuned for at most1024 updates; step0 retains the ridge baseline.

physical.csv contains component-standardized MSE and its square-root RMSE. First
average squared error over target components in an example; average examples
within an independent root, then roots uniformly in the named population. All
normalizers are fit separately per fixed physical target domain on fitting rows.
Observed and relaxed errors cannot be compared as an encoder ranking when their
target domains differ. PTM-Other includes interfaces and defects. Empty groups
have null errors and zero rows, not zero error. Future order uses the original-MD
eight-observable producer at3/9/12ps; persistence predicts that future vector by
its current value using the FUTURE target's fitted normalization.

neighbors.csv uses five fitting neighbors per development query with identical
temperature and current coarse PTM crystal/noncrystal status. All fitting roots
are disjoint from development. Distances for learned embeddings are original
Euclidean distances, without per-channel normalization. Teacher descriptors use
the declared block-scaled metric. Stable sorting resolves exact ties by training
row index. The reported error is average squared discrepancy over target
components and the five query-neighbor PAIRS, then source-weighted over queries;
it is not error to the average neighbor target. Target scales are fitting-only.

onset.csv uses first sustained original-MD PTM1/2/3 runs of three frames. Eligible
origins precede first onset and have three negative current/recent frames. All
origins have12ps plus two confirmation frames of follow-up. Five conditional
hazards describe bins ending at0.75/3/6/9/12ps. Event likelihood multiplies prior
survival and the observed bin hazard; non-events survive all five bins. Thus
right censoring beyond12ps is included without claiming knowledge of later fate.
Fits and scores weight independent sources uniformly, retaining the natural
within-source at-risk population. No transition oversampling is used.

Report joint event NLL, per-horizon binary log loss/Brier, source-weighted average
precision, and recall/false-positive rate at a tuning-negative FPR ceiling0.05.
Ties are included completely in the >= decision. The same tuning roots select
probe duration and alarm threshold; there is no separate calibration split.
Calibration reliability bins of width0.2 are retained in per-probe JSON, along
with counts, average risk and observed source-weighted frequency. Empty bins are
omitted. No-positive AP/recall and no-detection timing MAE are null.

Predicted event time at each horizon is the event-probability-weighted bin midpoint,
conditional on an event by that horizon. Detected timing MAE averages detected
event windows; missed windows and fraction of all event windows detected within
3ps are always reported. Window counts are not independent event counts or an
alarm-episode analysis. There are only18 development-positive12ps windows here.

comparisons.csv uses 100*(candidate_error/reference_error-1); negative is better.
Regression comparisons use MSE, neighbor comparisons pairwise discrepancy, and
hazard comparisons joint NLL. Root sets must match exactly. Resample complete
roots within fixed temperature strata for2000 paired draws, seed20260922, and
report percentile95% intervals. Each root has equal weight after its example
mean. Intervals condition on one encoder/probe seed and exclude training-seed
uncertainty. They are exploratory and unadjusted for multiple comparisons.
The2% proposed present-information tolerance is an engineering decision rule,
not a statistical significance threshold or an automatic model selection.

The metric contract fingerprints the implementing source files and this document.
Historical exported definitions are immutable.


Table export: 2026-09-22T09:11:53.471027+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
