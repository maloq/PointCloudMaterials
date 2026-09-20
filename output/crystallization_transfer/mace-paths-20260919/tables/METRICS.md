# Structural trajectory forecasting and local onset, version 1

The population, ancestry split, first sustained PTM onset, eligibility, calibration
sources and six reported horizons are inherited unchanged from
`crystallization_transfer.md`. Forecast origins stay on the original 3 ps grid.
No generated or future frame is an observation during evaluation.

Each trajectory has 32 structural states at 3, 6, ..., 96 ps. A state contains
the frozen parent MACE's 128 scalar embedding channels, the existing 128-channel
physical packet (geometry and motion), eight bond-order channels and instantaneous
PTM crystallinity. The targets are computed at the same tracked center identity.
The input is the historical MACE/geometry context and known temperature/time;
future physical targets, future geometry and future context membership never enter
that input. The physical-persistence baseline repeats the actual present state;
it additionally observes present physical targets and is explicitly a stronger-input
baseline. MACE itself sees positions, not velocities.

All target means/variances are fitted on training windows' future targets only,
weighting training sources equally and retaining each source's natural window
distribution. Scale floors at 1e-6 in original units. The input normalizer uses
eight outcome-independent reference windows per training source. No calibration,
selection or test target affects either set of moments. The encoder is frozen.

Training uses a block-balanced structural loss: 0.25 times the embedding loss plus
one times each physical-packet, bond-order and crystallinity loss, divided by 3.25.
Each block averages over its channels and 32 times. Direct and deterministic AR
use MSE; Gaussian AR uses diagonal Gaussian NLL without the constant. Both add exact
0.75 ps event-time hazard NLL through 96 ps. The mixture uses four whole-path
components and log-sum-exp of component log weights minus block-balanced Gaussian
loss and event NLL. This weighted mixture training objective is not reported as an
exact physical path log-likelihood. Predicted Gaussian log standard deviations are
bounded to [-5, 2] in standardized units. AR teacher forcing falls linearly from
one to zero over the first half of training; the entire second half trains free
rollouts. Selection, calibration and test always use free rollouts.

Diffusion predicts noise jointly for the structural path and four absorbing onset
indicators per 3 ps token (0.75 ps event resolution), mapped to -1/+1. Its loss is
block-balanced structural noise MSE plus mean event-indicator noise MSE. A cosine
64-level DDPM schedule is sampled using 16-step DDIM with eta=0. Each sampled path's
first indicator above zero defines onset; subsequent reversals cannot undo it.
The empirical CDF is mixed with one pseudo-sample of the training-only event CDF:
`(S * sampled_CDF + train_CDF) / (S+1)`. This avoids zero event masses from a finite
ensemble. It does not use held-out labels. Selection uses 16 samples, final
calibration/test 32. Direct and deterministic AR each return one path; mixture
uses its exact mixture CDF, while Gaussian AR averages conditional CDFs over sampled
rollouts. Never report best-of-sample error.

The selection criterion for every new method is source-weighted **dense integrated
Brier score**: mean `(F(k*0.75) - 1[T <= k*0.75])**2` across k=1,...,128. Checkpoints
are selected on 64 fixed windows per selection source, evaluated after every epoch.
Final evaluation uses all original calibration/test windows. Test scores never
choose a method, epoch budget or checkpoint.

For exact comparison with the original assay, sample the dense CDF at the six old
horizons, convert interval masses to conditional hazards (clip to [1e-7,1-1e-7])
and call the unchanged original evaluator. Its `timing` fields intentionally retain
the original coarse-bin midpoint estimator. The additional `fine_timing` fields use
the conditional mean over individual 0.75 ps event times through that horizon.
They report detected-positive-window MAE/bias, missed-window count, all-positive
conditional timing MAE, and the fraction of all positive windows both detected and
timed within 3 ps. These raw window scores are conditional on repeated overlapping
forecast origins, not independent events or source-weighted errors.

`restricted_mean_time_mae_ps` compares predicted E[min(T,96)] with observed min(T,96)
over **all** windows, including survivors and missed alarms. The predicted mean is
0.75 times survival summed at t=0,0.75,...,95.25. This error and dense Brier remain
defined when no positive predictions cross the calibrated threshold.

`path.standardized_mse_BLOCK` compares the sample-mean path to the measured target.
`path.standardized_crps_BLOCK` averages the empirical marginal CRPS over channels:
E|X-y| - 0.5 E|X-X'|, using all ordered sample pairs, including self pairs. For one
deterministic sample this reduces to absolute error. These are equal-source-weighted
test means at each stated horizon or averaged across all 32 times. CRPS is marginal,
not a measure of full joint temporal dependence. No exact diffusion likelihood or
full-path calibration is claimed. Raw per-window scores and dense CDFs are retained
for paired source-bootstrap follow-up; initial timing/trajectory exports have no CIs.

Source definitions/hashes accompany every exported CSV. Historical metric exports
and active hazard experiments keep their original definitions and source snapshots.


Table export: 2026-09-19T12:19:48.650508+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
