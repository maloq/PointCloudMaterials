# End-to-end MD history for current crystal distance

The encoder/predictor implementation now accepts a declared history length.
This contract still describes the three-frame comparison. The separate
`distance_encoder_dense_history` contract describes six declared-cadence observations
and accumulated microbatches. Existing exports retain their frozen code.

CD-MACE128-H6 receives the same tracked atom's observed local geometry at
t-6, t-3 and t ps. The shared 128-channel MACE produces three 128-vectors.
An ordered MLP receives those vectors and their two adjacent differences,
predicts a residual on the current vector, and supplies the resulting 128-vector
to the distance head. All encoder, temporal and distance parameters are trained
together. No learned features are cached during fitting. No velocities, time
values, temperature, species, material ID, scale or surrounding patches enter
the network. Physical offsets organize observations, not input covariates.

The repeated-current control has identical parameters and ordered predictor,
but all three vectors are the current embedding. Its deterministic shared encoder
is computed once and expanded differentiably; gradients sum across all uses.
Both arms fine-tune the same completed, unregularized CD-MACE128 checkpoint with
the same sample order, single seed, optimizer and 12 complete epochs. This is
supervised fine-tuning, not training from scratch or self-supervised learning.
No variance/covariance regularization is added to either arm.

The coordinate/label release and fixed Al64 all64 source roles are retained.
Training joins exact source/center-ID slices at verified 3-ps spacing, never
adjacent unrelated dataset rows. The first two retained label frames in each
source are omitted identically in both arms because their histories are absent
from this coordinate release. They are not replaced or interpolated. Equal
material mass is recalculated on eligible training rows; equal-source weighting
is recalculated on eligible Al selection rows. Every eligible row is visited
once per epoch. External branches remain train-only, with shared ancestry
explicit. Selection begins after epoch 12 and excludes calibration/test sources.

The present-distance target and proper objective are inherited from
`distance_encoder`: distance to the nearest atom in an already confirmed
at-least-64-atom crystal component; zero-inflated lognormal NLL right-censored at
64 Al-equivalent Å, plus twice the Bernoulli log losses at 8/12/20/32 Å with
weights .05/.15/.40/.40. Labels use confirmation available by the current frame.
No future frame enters the predictor or its current-distance target. The task
estimates the present location of an existing crystal using preceding MD
observations; it does not predict future onset or unseen nucleus birth.

Training uses global batch 1024 sequences, 512 per GPU on two GPUs, an explicit
throughput deviation from the default 256. Three-frame encoding packs 1536
patches per GPU. Geometry remains resident in float32 VRAM once; histories are
integer indices into that coordinate bank. cuEquivariance, compiled MACE and
bfloat16 autocast are used; likelihoods/optimizer weights retain float32.
`validation.csv` contains the combined predictive objective, distance NLL,
early log loss, capped-mean RMSE and Brier scores at 20/32 Å. Throughput is
sequences/s, not encoded frames/s. Epoch counts complete eligible populations.

## Spatial-front evaluation

Selection/calibration/test keep every original fixed observation in their
recorded roles; train rows are not needed for this frozen evaluation. The test
population is unchanged at 45,291 rows from 30 sources. The original controlled
scan paths are also unchanged: 495 approaches and 292 far-away controls from
28 Al test sources. At each scan position, historical observations follow that
position's atom ID through the preceding MD frames. They are not earlier
positions along the spatial scan. Current geometries, distances and visibility
are checked against the historical fixed/path producer. Missing history fails
instead of dropping model-specific evaluation rows.

`distance.csv` uses the inherited source-weighted continuous-distance NLL,
64-Å capped-median MAE, capped-mean RMSE and CDF Brier scores at 4/8/12/20/32 Å.
Al distances equal physical Å. Evaluation does not establish performance on
held-out external materials. BF16 inference matches training for all three
replayed models (history, repeated-current and original CD-MACE128 snapshot).

`alarms.csv`, `paths.csv` and `confidence-reliability.csv` use the numerical
definitions of `spatial_distance.confidence`, with this changed observation
contract: visibility is the union over every local MD frame actually supplied
to the predictor, including past frames. For the repeated/current baselines it
is current-frame visibility only. Both `visible_local` and `visible_context`
refer to this actual temporal input union; there is no 25-patch spatial context.
`visible_current` and `ptm_current` preserve separate current-frame diagnostics
in raw predictions. Clear-history subsets thus differ by model and must not be
treated as matched populations. The full held-out population remains matched.

Probabilities mean P(current confirmed crystal distance <= R), separately for
R=4/8/12/20/32 Å. Strict thresholds >.5/.75/.95 are all exported, without tuning.
The primary alarm needs two consecutive spatial positions; single-position
alarms are secondary. Warning is distance at the first completed alarm, and
conditional median/p10/p90 exclude missed paths. Recall at D divides detections
at distance >=D by all approaches. Misses and far-path false alarms accompany
warning medians. Far controls remain >32 Å throughout; tangential/near-miss
paths are not covered. Visibility at alarm includes all contributing positions
and all supplied MD frames. Early-clear counts additionally require >8 Å.

Reliability gives equal mass to each source before subsetting. Thresholded
precision and mean probability normalize retained mass; coverage divides by
subset mass. Empty denominators produce blank values, not zero. A threshold
>.95 is not a guarantee of 95% accuracy. Pooled path counts and single-seed
point estimates are not source-bootstrap or training-seed uncertainty intervals.
AP, onset probes and temporal lead times are not selection criteria or primary
outputs of this experiment.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.

Implementation revision (2026-09-27): the shared trainer also supports full-history checkpoint initialization and explicit material subsets for the separate material-adaptation protocol. Historical calculations and frozen exports are unchanged.
