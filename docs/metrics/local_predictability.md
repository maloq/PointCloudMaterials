# Local predictability metrics — protocol v1

This family is distinct from the legacy forecast crystallization assay and the
five-horizon predictive-memory likelihood. One trained seed: 20260919. Whole
source folds are inherited and exploratory. Validation selection and calibration
use disjoint 15-source halves. Temperature-stratified paired source bootstraps
condition on the single fitted seed, not training-seed uncertainty.

## Population and labels

The 150-source cohort fixes 16 tracked centers before labels, with native origins
48,78,...,498 ps and descriptor origins every 3 ps from 48 to 498. Timesteps are
0.75 ps. A positive label is first sustained local onset in `(t,t+tau]` at the
tracked atom, three consecutive PTM crystal frames (types 1/2/3, RMSD <=0.1).
At-risk origins precede first onset and have three observed noncrystalline frames.
Five/nine-frame persistence sensitivities need 3/6 ps of confirmation. The common
origin grid leaves six ps after the 96-ps endpoint; no unavailable follow-up is
labeled negative. Explicit empty-source counts belong to the coverage report.

Horizon bins end at 0.75,3,9,24,48,96 ps. With hazard logits a_k, loss for an event
in bin j is `sum(k<j, softplus(a_k)) + softplus(-a_j)`. Survival beyond all bins is
`sum(k, softplus(a_k))`. This is a joint event-time negative log likelihood, not a
sum of six binary endpoint losses. Cumulative risk is
`1 - product(k<=j, 1-sigmoid(a_k))`. No-event risk is not a physical determinism claim.

## Aggregation and calibration

Each source with eligible windows has weight 1/S; each of its N_s windows has
weight 1/(S*N_s). Per-horizon binary log loss clips predicted risk to [1e-7,1-1e-7].
Brier is weighted squared risk error. Average precision uses sklearn's weighted
noninterpolated AP on natural eligible windows; it is undefined with no positives.
No class balancing is used in training or evaluation. Training minibatches drawn
uniformly from rows receive inverse-source-count importance weights.

Window thresholds are chosen on calibration sources to maximize permitted
negative coverage subject to weighted FPR <=0.05, including complete tied score
groups; otherwise select a threshold just above the maximum negative score.
Precision/recall/FPR use `risk >= threshold`. Empty denominators are blank/None.
This window FPR is not the dense false-alarm-episode budget. Dense episode metrics,
when produced, collapse consecutive positives and require 9 ps between episode
starts, at 0.75-ps origins. Correct alarms have onset in their next specified
horizon. Exposure ends at first onset. Timing must retain missed-event counts.

For log-loss/Brier intervals, resample sources with replacement within each
observed temperature stratum, retaining the stratum's source count, 1,000 draws.
Compute the mean source score and percentile 2.5/97.5 bounds. Paired differences
must use the same resampled sources. Neighboring centers/windows are never
independent bootstrap units.

## Physical references and features

The target is the existing center-relative physical packet, blocks radial 0:32,
pair 32:64, angular 64:80, speed 80:96, signed radial velocity 96:112, moments112:128.
Fit target mean/std on train-source native current and all six future packets;
population std has floor 1e-4. Physical MSE is mean squared standardized error
within the declared block/horizon, then equal source averaging. `all` averages
128 coordinates. Persistence repeats the current packet at each future lag.
Ridge selects alpha on selection-source MSE; test never selects it.

History descriptor summaries concatenate current, oldest, mean, population std,
least-squares physical-time slope, minimum and maximum, including every frame.
Repeated controls copy the current packet through every slot. Conditions are five
temperature indicators, time/600 ps, and its square. Inputs are normalized using
training rows only. Center order uses `bond_order` with the center and its twelve
nearest neighbors, each having twelve bonds; coordination radius3.5 A. Smooth
7–17/17–25 A annuli contribute count, weighted radius, speed squared, signed radial
velocity, radial-velocity square and squared norm of mean relative velocity.
Annulus onset/exit taper over 2 A. Crystal labels do not enter these descriptors.

Current/future observability readouts, dense alarms, native comparisons and
information-retention probes are separate named stages. An unfinished stage
provides no result and may not be inferred from another table.
