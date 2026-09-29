# Two six-observation MD-history experiments

Question: does shorter, more densely sampled MD history improve current crystal
distance information over current geometry? Both shared MACE and temporal heads
are trained end to end. Each experiment includes a matched repeated-current control.

| Experiment | Native Al interval | External interval | Training population | Evaluation |
| --- | --- | --- | --- | --- |
| CD-MACE128-D6-075nominal | 0.75 ps (3.75-ps span) | 0.70 ps (3.50-ps span) | All existing eligible distance training sources, 11,375,472 windows | Fixed Al64 source roles and unchanged spatial scans |
| CD-MACE128-D6-010ps | Excluded | 0.10 ps (0.50-ps span) | Al-million/Mg/Ti families, 3,374,496 windows | Separate Al family for selection; Ta family for transfer test |

The user explicitly accepted 0.70 ps as the external approximation to nominal
0.75 ps. Use seven saved frames per step, never alternating 7/8 strides. Log
actual per-source offsets and spans. No interpolation is used. The second
experiment uses exactly 0.10 ps at every source. Neither model receives explicit
time, temperature, material, species or motion covariates.

All-data initialization is the completed distance encoder/head. The 0.10-ps fit
starts from the earlier native-Al-only encoder with fresh temporal/distance heads,
because the distance checkpoint saw the external families now held out. All
related branches stay in one role. Selection and test each contain one ancestry
family; Ta also tests material transfer. The two experiments differ in training
population, initialization and evaluation population and do not isolate cadence.

Use one seed, 12 complete epochs, global batch 1024 (two GPUs, microbatch 256,
two accumulation steps), compiled cuEquivariance and proper distance/CDF log
losses. Select with the declared validation likelihood starting at epoch 12.
Keep the existing wider H6 queue unchanged. AP is not a selector or the primary task.

[Nominal 0.75-ps recipe](../../configs/distance_encoder/md_dense6_075nominal_20260926.json) ·
[Exact 0.10-ps recipe](../../configs/distance_encoder/md_dense6_010ps_20260926.json) ·
[Native spatial metric definitions](../../docs/metrics/distance_encoder_dense_history.md) ·
[External transfer definitions](../../docs/metrics/distance_encoder_dense_external.md) ·
[Execution](../../docs/distance_encoder_history.md)
