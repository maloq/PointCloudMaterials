# Repaired structural-state screen

[Completed analysis](../../output/structural_state/repaired-review-20260923/README.md):
repairs remove amplitude collapse; relaxed input improves onset ranking in this
one-seed development cohort, while teacher/distance additions offer little and
ordinary physical forecasting barely changes with encoder training.

This is a new four-arm study following the
[22 September failure analysis](../../output/structural_state/screen-20260922/RESULTS.md).
Old results/checkpoints retain their original frozen implementation. The repair
is a bundled optimization/architecture correction; comparisons to the old study
cannot attribute gains to an individual correction.

The scientific contrasts remain C versus A (fixed relaxed geometry as teacher)
and D versus B (physical-distance supervision). A/C observe original positions;
B/D observe relaxed positions. All use the same existing 45-source paired cache,
25/5/15 fitting/tuning/development roots, 8 Å observations and source sampling.
No new simulations, future encoder labels, histories or velocities are added.
One seed, 20260922; all arms receive 4,096 matched updates with batch256/micro64.

The export now contains the current trainable MACE pooled64 features directly,
after a fixed fitting-only affine normalization, followed by 64 learned features
of that same normalized vector. The added readout has small random output
weights, not a zero branch with zero downstream weights. Every output is computed
from the current trainable spatial encoder; no frozen-encoder feature bypass is
present. The change preserves pooled information through the final readout but
does not guarantee that backbone training preserves it.

Before optimization, fit bounded linear reconstruction heads on the 1,600 fitting
rows only. Start ridge penalty at1, multiply by10 until Frobenius norm is at most10;
the intercept is unpenalized. Both physical domains are initialized for every
arm, with the unused head receiving no training gradient. No tuning/development
target selects calibration. All initial encoder weights match, and A/C or B/D
calibrations match within input domain. The checkpoint called initial includes
this fitting-only affine/head calibration; its backbone is untrained.

Keep both relational distance denominators fixed at their initial fitting-pair
means. D's relation coefficient ramps linearly from0 to0.1 over512 updates.
The former detached moving scale is removed. The encoder learning rate is1e-5,
head rate3e-4; both use the original128-step warmup/cosine schedule. The original
norm10 head cap, gradient cap5 and modest weight decay remain.

Every256 updates, score the actual training decoder on fitting/tuning rows,
including radial/l2/l4 blocks and the constant fitting-mean predictor. Record
raw export RMS spread and the per-block2% retention criterion against calibrated
initial tuning errors. A spread below10% of its initial fitting value is a loud
optimization failure. Retention misses are reported; they never trigger test-set
selection or silent checkpoint replacement. The primary export is final4096;
best tuning checkpoints remain secondary artifacts.

Frozen downstream regression, nearest-neighbor, future-order and hazard tasks
remain matched across arms. Hazard step zero is now exactly the source-weighted
fitting prior, with zero output weights, so arbitrary tiny random slopes cannot
generate seemingly informative AP when training was rejected. Step-zero selection
is still allowed and must be reported as no learned hazard benefit. Native
training-head development errors are exported alongside the stronger refitted
probes. Future labels, angular/l6 tests and onset labels remain outside encoder
optimization; the sparse onset cohort remains unsuitable for a strong final claim.

Two512-update implementation pilots (A/D) used fitting and tuning diagnostics
only. They confirmed stable spread, nonconstant head predictions and retention
of all three tuning blocks before the production queue. They are disposable
debugging fits, not additional seeds or selected scientific outcomes.

[Recipe](../../configs/structural_state/repaired_20260923.json) ·
[Operations](../../docs/structural_state.md) ·
[Metrics](../../docs/metrics/structural_state.md) ·
[Run output](../../output/structural_state/repaired-20260923/README.md)
