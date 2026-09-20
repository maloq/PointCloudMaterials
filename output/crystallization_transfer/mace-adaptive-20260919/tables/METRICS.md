# Local MACE crystallization transfer metrics, version 1

The target is first local PTM crystallization sustained over three 0.75 ps frames,
on origins that are still liquid and precede first onset. Six event-time bins end
at 0.75/3/9/24/48/96 ps. Every origin has complete follow-up; event_bin=6 denotes
survival through 96 ps, not an observed event. Hazard NLL is the sum of negative
log survival for bins before the event plus negative log hazard in its event bin.
The cumulative risk is 1 minus the cumulative product of survival probabilities.
Metrics preserve equal source weights and each source's natural window prevalence.

For each horizon export source-weighted AP, AUROC, Brier error, binary log loss,
prevalence, precision, recall, F1, balanced accuracy and false-positive rate.
The threshold is selected on the separate calibration sources to allow at most
5% source-weighted negative-window FPR, respecting score ties. Undefined metrics
are JSON null / blank CSV fields. ECE sums absolute source-weighted calibration
residuals over ten fixed equal-width risk bins. Brier and log-loss intervals use
500 temperature-stratified bootstrap draws of whole test sources and describe
source uncertainty conditional on the one trained model. Test rows never select
checkpoints, thresholds, durations or hyperparameters.

Event-time estimates use the conditional mean of event-bin midpoints up to each
horizon, normalized by cumulative event probability. Report timing MAE/bias only
on detected positive windows, together with missed-window counts and the fraction
of all positive windows detected within 3 ps timing error. Discrete midpoint
timing is coarse, especially at long horizons. Repeated positive alarms collapse
into episodes with the inherited 9 ps refractory interval, on the declared 3 ps
origin grid. False alarm episodes divide by the sampled at-risk exposure
(number of eligible windows × 3 ps, converted to center-ns). Mean lead time
conditions on detected episodes. These coarse-grid metrics are not continuous
0.75 ps monitoring results, and eligibility gaps are not imputed alarms.

Spatial diagnostics use the sampled at-risk tracked centers within each source
and origin: absolute error in predicted transforming fraction, Jaccard of the
thresholded/actual transforming sets where the union is nonempty, and RMSE of
predicted minus actual pairwise risk differences for periodic center pairs within
25 Å. Spatial means weight available source/origin groups equally; pair RMSE
weights available pairs equally. Report available pair counts; these are sparse
sampled-center diagnostics, not dense front localization or propagation speed.

The maintained producers are `src/research/crystallization_transfer/metrics.py`
and the imported, hash-pinned local-predictability hazard/source weighting and
local-crystallization onset functions. Raw calibration/test logits, test row
identities, thresholds and source records are retained for paired later analysis.
The no-transition baseline uses logit -30 in every bin: effectively zero risk,
with finite NLL. It has no fitted parameters and no test-selected threshold.

Episode timing also reports the number of tracked centers with an event observable
from at least one eligible origin, the detected/missed counts among those centers,
and event recall. These raw episode counts complement source-weighted window
classification; positive windows from one event are not separate events.

## Scaling extension: population and budget accounting

The scaling queue keeps the event, calibration, test population and metric
calculations above unchanged. It uses exact shuffled passes through a declared
training subset, with per-row objective weight N/(S*n_s), where N is selected
training windows, S is selected training sources and n_s is selected windows in
that source. Thus the population objective still gives each source equal weight.
Microbatch losses divide by the actual number of examples in the complete
optimizer batch, including short final batches. Subsets never restrict selection,
calibration or test. Normalizers and initial hazard biases use the subset only.

Export `training.sources`, `eligible_windows`, `updates`, `samples`,
`complete_epochs`, `partial_epoch_updates`, and `selected_step`. An epoch is a
full permutation without replacement; the final batch can be short. Matched-data
runs use exactly the update count of three full-data epochs, cycling through their
own smaller population. Equal update counts imply approximately equal sample
exposure (short final batches are counted exactly), not equal epochs. Each fit
retains source IDs, membership hash and explicit training row indices. The
selected checkpoint may precede the final update; the stopping budget and selected
step are both reported. Historical exports retain their original definitions.

For radius sweeps the seven fixed candidates remain center plus three samples per
(0,12] and (12,25] Å annulus. Radius changes the quintic support envelope and
strictly masks zero-weight attention keys. Candidates outside the radius have
zero influence and zero feature gradient. It does not resample a dense sphere
or change the local MACE receptive field. Radius zero uses only the center.

## Adaptive trainable-encoder extension

The physical event metrics are unchanged. The adaptive protocol uses full-batch,
source-weighted differentiable normalization in training and recomputes inference
moments from fixed training-only reference windows before each validation. Saved
moments belong to the exact saved encoder/head checkpoint. No selection,
calibration or test rows update those statistics. Training diagnostics report
geometry-logit standard deviation with conditions held fixed, RMS scalar feature
standard deviation, and median normalization scale on the training references.
They are learning diagnostics, not additional held-out scores.

`training.head_warmup_updates` counts head-only updates within the total epoch
budget. Direct backward and gradient replay compute one statistical-batch loss;
replay does not count its encoder VJP as additional samples. Encoder/head gradient
norms are measured before separate clipping. The 12/24-epoch promotions are ranked
only by screen selection NLL and restart with their own budget-specific schedule.
All selected-step, population and source-uncertainty definitions above still apply.


Table export: 2026-09-19T23:23:04.793908+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
