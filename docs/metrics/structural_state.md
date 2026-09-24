# Fixed structural-state objectives and evaluation

This is the repaired v2 contract (23 September). Historical v1 exports retain
their own immutable tables/METRICS.md and implementation hashes.

Training targets are radial17 and moment-Gram l2/l4 (36 each), produced by
src/training_methods/bcr/probes.py at 8 Å. Each block contributes its mean squared
error with equal block weight. Means/stds use the 25 fitting roots only; std floor
1e-6. C adds 0.25 times the relaxed-target error. D adds 0.1 times Huber loss
(delta1) between normalized paired distances. Teacher distance is the square root
of block-balanced geometry MSE; learned distance is RMS coordinate difference in
exported128. Both scales are fixed initial fitting-pair means; there is no moving
denominator. D ramps relation weight0 to0.1 over512 updates. Head Frobenius norm
cap10; gradient norm cap5. All scientific metrics retain full precision in CSV/JSON.
The exported128 concatenates train-normalized pooled64 and learned64 features.
Pooled mean/std are frozen from fitting rows, std floor1e-6. Linear heads are
initialized by fitting-only ridge with an unpenalized intercept, starting alpha1
and increasing tenfold until weight norm<=10. Calibrated initial and final heads
use the same fixed target/input units; the initial backbone remains untrained.

The primary evaluation uses matched update4096, not the best tuning checkpoint.
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
Initial hazard output weights are exactly zero and biases encode source-weighted
fitting hazards, making step zero a true constant-risk control. Selection can
retain this baseline; it is not counted as successful hazard training. Per-probe
JSON/checkpoints record the chosen step.

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

training_heads.csv scores saved initial/last physical heads without refitting,
plus the constant fitting-mean control (zero in standardized target space).
For each radial17/l2-Gram36/l4-Gram36 block, report component-averaged MSE within
source then average sources uniformly, using saved fitting target normalizers.
block_mean equally averages the three blocks, not all89 components indiscriminately.
The heads receive the actual exported features, with no fresh feature rescaling.
These metrics share the same population groups as physical.csv.

Per-fit diagnostics.jsonl additionally records fitting/tuning head block MSE,
constant-mean MSE and sqrt(mean coordinate variance) of the raw exported features,
using population variance (ddof0). Complete role grids have64 observations per
source, so uniform example means also weight sources equally. Spread ratio divides
current fitting RMS spread by calibrated-initial fitting RMS spread; a ratio<0.1
stops as an optimization failure. retained_blocks tests each tuning block against
1.02 times its calibrated-initial MSE. Failing retention is recorded without
silently changing the primary final checkpoint. No future labels enter these
diagnostics or decisions.

The metric contract fingerprints the implementing source files and this document.
Historical exported definitions are immutable.

## v3 future/distance factorial extension

The future_metric_seed20260923/20260924 recipes use protocol
fixed_geometry_future_relation_v3. They retain the v2 geometry and distance
calculations above, with the following explicit changes. Current original-MD
order8 enters **every** encoder loss at weight0.25; E/F add weight0.25 times the
component-mean standardized9ps future-residual squared error. Original-MD future
order9 is therefore not a withheld encoder target in E/F. Angular/l6,3ps/12ps
future order and onset labels remain withheld from encoder losses.

The fixed future baseline is Ridge(alpha1, intercept enabled, SVD solver), from
[current order8, relaxed geometry89, one-hot temperature, frame*0.75/600,
(frame*0.75/600)^2] to unstandardized future order8. Baseline inputs are standardized
using fitting means/population stds, floor1e-6; temperature levels use fitting
rows. Current-order and future-residual outputs each use their own fitting
means/stds, floor1e-6. Baseline fitting uses all1600 fitting rows, with no onset
conditioning, oversampling or tuning. Heads receive exported z only. Its complete
coefficient/scaling receipt is stored and verified during evaluation.

future_increment_9 is the standardized future residual, not a literal time
difference or a claim of conditioning on all nonlinear present information.
auxiliary_heads.csv gives initial/final/zero-prediction current/future head MSE,
component mean within source and then equal source mean. Future-head performance
in inactive arms is diagnostic of the unchanged calibrated head plus moving
encoder. Tuning auxiliary errors are logged but never select the primary encoder.
The RMS-spread guard still uses no future labels. The native geometry best.pt
selection remains geometry-only; primary4096 is fixed in advance.

embedding_geometry.csv reports sample-covariance participation rank:
(sum eigenvalues)^2/sum eigenvalues^2, with exactly constant rank0. SVD/centering
use float64. conditional_rank averages ranks computed separately within each
source having at least two observations, weighting such sources uniformly.
Groups are all development observations and the noncrystalline subset.
In existing physical/head tables PTM_other is the complement of PTM1/2/3, not
strictly the original PTM code0. v3 neighbors.csv additionally exports that same
noncrystalline query subset, equal-source weighted after masking. Neighbor targets
add future_order_12 and future_increment_9; neighbors are still k5 current-T/phase
matched, with discrepancy averaged over individual pairs rather than neighbor
means. comparisons.csv readout neighbors_noncrystalline identifies these paired
source effects. New contrasts follow the explicit recipe contrasts list.

Encoder seeds differ; frozen probe initialization/sampling uses fixed probe_seed
20260922 in both runs. Per-seed physical source bootstraps use the encoder seed;
combined onset resampling is separately defined in structural_state_future.md.
Historical exported definitions are preserved in their run folders.

## Parameter-search training reuse

The separate `fixed_geometry_parameter_search_v4` campaign reuses the native
training kernel with an explicit per-arm encoder learning rate and stronger
distance weight. Historical v2/v3 objectives and metric formulas are unchanged.
The structural-state CLI rejects v4: its evaluation and exports use the separate
[encoder parameter search contract](encoder_parameter_search.md).
