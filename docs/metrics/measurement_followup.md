# Ni measurement memory and matched physical response, 12 October 2026

This is a separate metric family from `measurement_sim`. Historical exports keep
their definitions. Source data, units, 244 columns and train-only normalization
are inherited from the pinned October 11 direct-measurement plan: BO40, TDA113,
smooth density74, velocity17. The 22 train-constant channels are frozen and excluded
from predictive scores. BO/TDA/density/velocity each receive total weight 1/4.
The cohort is outcome-enriched, not a population incidence sample.

## Two distinct tasks

Native: all 14,400 fixed-central-atom sequences, with 3,840 fitting, 960 selection
and 9,600 held-out rows from 16/4/40 ancestors. The common forecast origin moves
from old index3 to index13, exactly +1 ps. Four-frame input spans -0.3..0 ps;
fourteen-frame input spans -1.3..0 ps. The common future is 50 observations at
0.1 ps spacing, through **5 ps**. No interpolation or additional ancestors.
Nearest80 neighbors are reselected around the same center. Native normalization,
increment RMS and OU rates reuse the original training-only producer.

Physical: all 300 existing full-cell anchors, with 80/20/200 fitting/selection/held
out from the same 16/4/40 ancestors. Each has two fixed-box BAOAB shots recorded at
5..100 fs and fixed initial80 identities. Input is the current244 observation,
without historical NPT tokens. All short-task scores average the scores against
the **two individual shots**, not against their mean. Analytic91 response targets
are the two-shot mean derivatives. Shots and local centers are not independent
ancestors. Archived integration/tangents are float64, force arithmetic float32;
fixed observations/JVPs and neural response secants are float32.

Physical OU rate is fitted on the 80 training anchors, using both shots and the
common initial observation: rho=mean(y_t*y_(t+1))/max(mean(y_t^2),1e-5), clipped
to [0.01,0.9995], rate=-log(rho)/0.005 ps. Physical increment RMS uses the same
5-fs adjacent observations, floored at0.05 normalized units. This is an empirical
diagonal prior, not a physical simulator baseline. Normalization itself remains
the shared native training normalization. Native and physical scores are separate.

## Training and selection

All models evolve the full244 measurements with the existing bounded-residual
diagonal-OU stochastic transformer, at5-fs internal steps. The observed initial
state is exact. Four or fourteen native memory slots update each0.1 ps. The summary
treatment keeps the same14-state window but presents three tokens: past13 mean,
past13 population standard deviation (variance floor1e-10), and current state.
It removes order of past observations while retaining longer-window statistics.
Shared weights are initialized identically at each paired seed; ordinal slot
tables differ with history length. No explicit horizon, temperature, age, absolute
time, species or box covariate is an input. No learned spatial encoder/decoder.

Native fits: 2,400 updates, effective batch256, unroll1/3/5 ps with probabilities
0.5/0.25/0.25; four neural particles. Physical fits: 4,000 updates, batch64,
20 dense5-fs steps, four particles. AdamW, weight decay1e-5, max LR0.001,
150-update warmup, cosine to1% of max, gradient clipping10. BF16 neural matrix
work only on native training; FP32 state/loss and physical training. On96-GB GPUs
microbatch256; on64–89-GB devices128; below64 GB64, with gradient accumulation.
The same effective batch/objective is preserved, but random number partitioning
differs when microbatch changes; actual hardware/microbatch is recorded per segment.

Loss is composite marginal NLL + path energy +0.5 increment energy +0.25 lag-product
energy. NLL is the mean univariate mixture NLL over time and family-weighted
channels, using equal Gaussian kernels of fixed normalized bandwidth0.25.
It is not a joint likelihood. Kernel bandwidth is not added to plotted paths.
Energy score is mean distance to truth minus half the mean distance of all distinct
particle pairs. Joint path distance is sqrt(mean_time(sum_channels(w*delta^2))),
floored at1e-6. Increment paths start at the exact common observed initial state
and divide by train RMS. Lag products average same-particle increments separated
by1 and5 observation intervals: native0.1/0.5 ps, physical5/25 fs.

Physical-response fits add weight1 times four-group normalized response MSE,
averaged over times and Q/V directions. Groups are density74, velocity-shell9,
bond-rate4, flow4. Each denominator is the training-bank response mean square for
that group/time/direction, floored at1e-5. The physical-values control has no such
term; both have identical path supervision. Native fits have no physical auxiliary
term, so history contrasts do not confound simulator protocols. Student secants
use the actual full244 perturbed present observations, common neural draws for
plus/minus, epsilon0.25; no older history token changes. Only future smooth91 mean
response is compared. No physical gradients are claimed for BO/TDA outputs.

Checkpoint selection every200 updates uses selection NLL only: fixed128 of960
native rows, or all20 physical selection anchors, eight particles. No held-out
selection or AP objective. This queue is exploratory on already examined source
roles, not a newly independent confirmatory holdout.

## Evaluation and uncertainty

All9,600 native held rows use32 neural paths, at1/3/5 ps. All200 physical anchors
use32 paths and three independent neural noise keys, at50/100 fs for full path
scores and every5 fs for response. Native full-cohort scores use one common key;
predetermined examples additionally save three keys. Keys are paired across seeds'
treatments, and neural paths are not independent physical continuations.

`endpoint_mse` compares the neural ensemble mean with each physical endpoint,
then averages shots where available. `mean` predicts the native training mean;
`persistence` repeats the actual initial measurements. `temporal_power` averages
within-path variance across neural particles; `MD_temporal_power` uses individual
physical paths; `mean_temporal_power` uses the neural ensemble-mean trajectory.
All exclude the initial point. `measurement_90coverage` is the family-weighted
fraction of measurements within the empirical5th–95th percentiles of neural
paths, without kernel noise. These diagnose spread, not just mean accuracy.

`shuffled_particles` independently permutes particle identities at every future
time, preserving every marginal ensemble exactly while disrupting connected
paths. The observed initial state remains fixed. Marginal NLL/endpoint MSE should
agree with the model up to rounding; increment/path/lag scores may differ.

Scores average within ancestor then equally across40 ancestors. Intervals use
1,000 bootstrap resamples of ancestors, not individual overlapping centers.
Response relative MSE is ancestor-balanced error power divided by ancestor-balanced
target power, denominator floor1e-5. Bootstrap uses paired numerator/denominator
ancestor draws. `zero_relative_mse` records the actual zero baseline, which can be
less than1 where the floor dominates. Report target power and raw MSE alongside
ratios. Do not silently average away Q/V or channel-family failures. Sensitivity
evaluation also halves epsilon with the same noise keys. Half-epsilon and repeated
keys diagnose numerical/MC error; they are not new physical data. Paired treatment
contrasts resample matching ancestors and report treatment minus control; negative
error/score differences favor the treatment. Seed replications remain separate.

## Frozen readout diagnostics

Response input: the four observed244 states H and the known analytic present91
response R0 to the physical query. Each Q/V direction is fitted separately.
Three readouts: linear transport R_tau=A_tau*R0; residual linear
R_tau=R0+B_tau*R0; state-conditioned residual R_tau=R0+B_tau(H)*R0. The latter
uses a kernel linear in R0, multiplied by (1+RBF(H,H'))/2. For fixed H all three
are linear in the supplied query response, with no query bias. The identity at
time0 in residual models is **supplied**, not learned evidence or a forecast score.
Zero and unmodified R0 are explicit future-response controls.

Readout H centering/scaling uses its fitting subset (std floor0.01). R0 is not
centered; per-channel RMS uses that subset, floor1e-5. State distances divide by
H width and by median fitting pair distance (floor1e-8). RBF length0.5 or2;
ridge penalties1e-4..10, multiplied by number of fitting anchors. Output scales
are group/time fitting response RMS with squared floor1e-5. Ridge/length select
on the20 selection anchors by equal-group Gaussian NLL at those fixed variances.
Prediction uses all200 held anchors. These are predictability assays conditional
on (H,R0), not information bounds for H alone or full-Jacobian rank experiments.

Three seeded nested ancestor orders fit4/8/12/16 sources, retaining allfive anchors
per source. A complementary curve uses1/2/3/5 anchors in each of16 sources.
Identical fitting subsets are deduplicated (13 distinct subsets, 78 selected
readouts across3 families and2 directions). Fits/candidate selectors are saved.
Only the80 original fitting anchors are eligible. The curve does not create
physical independence or fit on the held sources.

Three frozen Gaussian ridge history readouts use the common shifted native origin:
four ordered states, fourteen ordered states, or present plus mean/std of past13.
They forecast244 measurements at0.1/0.5/1/3/5 ps. Fit3,840, select960, evaluate9,600.
Input train std floor0.01, train residual variance floor1e-4, six ridge penalties.
The reused readout selector is unweighted per-coordinate Gaussian NLL; exported
MSE/NLL are family weighted. Record this distinction. These Gaussian linear
readouts are separate from neural stochastic trajectory training.

Learning tables retain actual minibatch scores, derivative loss, clipped-gradient
input norm, and periodic selection scores. Missing selection fields are blank,
never zero. Source hashes accompany every CSV in `tables/METRICS.md` and
`technical/metric-contracts/measurement_followup.json`.
