# Fixed structural targets, relaxed teachers and physical distances

[Completed results, 23 September 2026](../../output/structural_state/screen-20260922/RESULTS.md):
all four fits and evaluations finished. No consistent forecasting gain; the
distance-supervised arm has severe amplitude shrinkage and a nearly constant
training decoder despite improved rescaled structural probes.

Protocol fixed on 22 September 2026, following the
[literature proposal](../encoder_training_literature_20260922/README.md). Four
scratch native-MACE encoder fits, one seed (20260922), 2,048 updates each. This is
the first mechanism screen after BCR, using existing observations and relaxations.

| Arm | Input | Encoder objective |
| --- | --- | --- |
| A-observed | Observed coordinates | Observed radial and angular-density geometry |
| B-relaxed | Relaxed coordinates | Relaxed radial and angular-density geometry |
| C-relaxed-teacher | Observed coordinates | A + 0.25 × fixed relaxed-geometry prediction |
| D-physical-distance | Relaxed coordinates | B + 0.1 × physical distance matching |

Every training head reads only the exported 128-vector. The teacher in C consists
of fixed physical targets, not a simultaneously adapting neural encoder. Heads
are linear; their Frobenius norms are capped at 10 after updates. All arms share
initial encoder/head weights, a 32-channel two-layer tensor MACE, optimizer,
sampling stream, update count and float32 arithmetic. cuEquivariance is required;
there is no backend fallback or hardware benchmark inside the trainer.

## Population and support

Reuse the frozen paired BCR audit: 45 independent Lee2003 MEAM Al roots at
400/450/500/510/520 K, four original frames (64/224/368/512), and 16 tracked centers
per frame. There are 1,600 fitting observations from 25 roots, 320 tuning from
five roots, and 960 development from 15 roots. Historical calibration/test roots
are excluded. These development cohorts have been inspected previously; they are
not an untouched final test. Normalizers, PCA, encoder optimization and teachers
use fitting roots only. Tuning roots select probe regularization/duration and
alarm thresholds, never train encoder weights.

Observed and relaxed patches are independently complete inside 8 Å around the
same center identity. Keep the existing native BCR encoder's finite-patch graph
definition, 5 Å interaction cutoff and smooth boundary weights; no nearest-80
truncation. This is not a claim that pooled atoms have an infinite-bulk halo
outside 8 Å. Holding this bounded observation contract fixed avoids introducing
a context change in the objective comparison. Full-cell relaxed targets can
depend on unobserved context outside the patch.

The source archives have float16 positions, float32 boxes and exact identities.
These are structural experiments, not weak-noise denoising or tests of motions
below quantization resolution. No new simulation or quench is required.

## Fixed geometry and paired sampling

The existing physical descriptor producer supplies radial17 and 144 moment-Gram
components. Training uses three equally weighted blocks: radial17, l2 Gram36 and
l4 Gram36. Each component is standardized by fitting-domain mean/std, with a
1e-6 scale floor. Targets preserve weighted count/density and cross-shell angular
correlations. The l0 Gram block is not an additional training objective.

Reserve q4/w4/q6/w6, l6 Gram36, separately produced current order8, future order8
at 3/9/12 ps, and sustained-onset labels for evaluation. These are held-out
measurements, not a guarantee of statistical independence from training geometry.

The same paired stream supplies every arm: sample a fitting root uniformly,
then frame and center; sample a partner from a different root at the same
temperature and current coarse original-MD PTM status. Partner roots are uniform
among compatible roots. This gives a declared paired exposure distribution rather
than claiming every endpoint remains uniformly source sampled. Future labels do
not enter this sampler. Batch256 and microbatch64 preserve adjacent pairs.

For D, the teacher distance is the square root of the mean of the three block
MSEs. The learned distance is RMS coordinate difference in exported z. A Huber
loss compares them after scaling. Teacher scale is frozen on 128 train-only
pairs. Embedding scale starts on the same pairs and updates once per full batch
with EMA0.95, detached from differentiation. A scale below 1e-10 is an explicit
failure. Microbatching does not change the pair loss or scale-update frequency.

## Frozen evaluation

The primary checkpoint is the final matched update. Best physical tuning
checkpoints are saved as secondary artifacts, not substituted into the primary
comparison. Probe both pooled64 and exported128, and both corresponding untrained
initial features. Controls use fixed geometry89, its train-fitted PCA64, known
temperature alone, and physical persistence for future order. PCA64 matches the
pooled dimension; it is not described as a 128-dimensional compression of 89 inputs.

All frozen regressors have 128 feature slots plus identical temperature conditions;
smaller controls are zero padded. Ridge selects among 11 penalties on tuning
roots. A zero-initialized two-layer residual MLP may improve it, with at most
1,024 updates and tuning-selected duration including step zero. Targets/scales
are identical for each compared readout. Export each structural family,
temperature, and PTM-Other/crystalline subgroup separately. PTM-Other includes
defects and interfaces and is not synonymous with bulk liquid.

Original-space nearest-neighbor evaluation uses five fitting examples matched
to each development example's temperature/current PTM stratum. It does not
whiten individual embedding channels. Score pairwise discrepancies in withheld
structure and future order. Descriptor controls use the declared standardized,
block-balanced geometry metric. Avoid scoring only the distances taught to D.

Reuse independently produced original-MD local PTM labels, not relaxed-state labels,
for onset: first run of three crystalline frames, three preceding/current negative
frames for eligibility, and sufficient follow-up to confirm events. There are
827/231/643 at-risk fitting/tuning/development windows, containing 30/8/18 onsets
within 12 ps. Fit source-balanced linear and MLP discrete-time hazards at
0.75/3/6/9/12 ps. Report NLL, AP, Brier, calibration bins, recall at a tuning-set
5% false-alarm threshold, detected timing error and misses. The same tuning roots
select the predictor and threshold; there is no separate calibration cohort.
The four-frame sampling is sparse and event evidence will be limited.

## Decision criteria

Main comparisons are C versus A and D versus B, plus each trained representation
versus its own initialization and matched descriptor controls. Report physical
retention, original-space neighbors and future skill separately. A preliminary
engineering tolerance is at most 2% relative MSE deterioration in important
present-information blocks, alongside improved withheld or future utility. The
tolerance is not a significance level, and crossing it does not automatically
select a winner. No slowness, rank or UMAP selection criterion is used.

Use 2,000 paired bootstrap draws of entire development roots, stratified by
temperature. These intervals condition on one fitted seed, with no correction
for multiple exploratory comparisons. Preserve all comparisons and event counts.
Failure to improve on fixed relaxed descriptors is a legitimate outcome; it does
not trigger an automatic wider model or additional loss sweep.

[Recipe](../../configs/structural_state/screen_20260922.json),
[execution and resume](../../docs/structural_state.md),
[metric definitions](../../docs/metrics/structural_state.md),
[results](../../output/structural_state/screen-20260922/README.md).
