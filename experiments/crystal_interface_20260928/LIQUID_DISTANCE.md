# Distance to an external crystal from liquid structure

[Interim training diagnosis, blocks 1–10](LIQUID_TRAINING_DRIFT.md): validation
overfitting, weak distance improvement and the inherited checkpoint-policy issue.

Question: can the geometry of liquid-only observations predict distance/direction
to an established crystal outside all observed patches? This replaces the queued
interface-only exclusion experiment, which was stopped before training began.

Use the expanded Al64 source cohort and retain every original benchmark row.
Fitting and selection require an uncrystallized query, no established crystal in
any of the 25 consumed patches, and a crystal elsewhere in the cell. Preserve
partially ordered/subcritical liquid structures. Crystal-absent cells form a separate
challenge; they must not dominate the distance objective or checkpoint selector.

One LCD-MACE128-VC fit starts from the recorded snapshot parent with a fresh context
head and optimizer. Retain joint training, vector messages, VCReg, one seed,
two GPUs and global batch 512. Train 8192 updates in 16 nominal blocks, selecting
among blocks 12–16 by predictive likelihood. No time/temperature/material inputs.
The task is snapshot localization, not prediction of a future onset time.

For all eligible rows, verify equality of cached interface distance and the saved
nearest-crystal distance before fitting. Preserve the original target records,
geometry, source roles and preparation identity; no new simulation is needed.

Primary evaluation is distance likelihood, distance error, calibrated proximity
and direction on liquid-only contexts with an external crystal. Compare against
a constant fitted on this conditional training population, and against the previous
VCReg checkpoint on exactly the same original eligible rows. Report absence cases,
feature/readout quality, normalized noise response and 0.75-ps stability separately.

The [feature audit](FEATURE_DOMINANCE.md) motivates this change: on 12,923 original
eligible test examples, the preceding model's capped-distance RMSE was 15.23 Å,
versus 10.71 Å for a training-mean constant. This restricted population is the
relevant target, rather than aggregate performance helped by visible crystals.

Prepared strict populations:

| Role | Eligible contexts | Eligible sources | Distance ≤20 Å | Distance ≤32 Å |
|---|---:|---:|---:|---:|
| Train | 183,596 | 88 | 913 | 48,458 |
| Selection | 41,418 | 15 | 231 | 11,445 |
| Calibration | 34,117 | 15 | 183 | 10,021 |
| Test | 66,839 | 28 | 307 | 17,562 |

All 150 source roles and original rows remain in the cache; sources with no
eligible external-crystal observation contribute no localization rows. No source
is reassigned. Every eligible cached distance equals nearest-crystal distance.
There are zero crystal-interior, crystal-visible or crystal-absent fitting rows.
The random batch-512 sampler expects 2.225 examples within 20 Å; its probability
of none is 10.76%, explicitly recorded rather than imposing a batch quota.

[Recipe](../../configs/crystal_vector/liquid_distance_20260928.json) ·
[Metric definitions](../../docs/metrics/crystal_liquid_distance.md) ·
[Execution](../../docs/crystal_liquid_distance.md).
