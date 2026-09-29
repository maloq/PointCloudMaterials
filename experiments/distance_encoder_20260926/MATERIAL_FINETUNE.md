# Does specializing the dense-history encoder help Al or Ta?

[Results at the user-requested stop on 27 September](MATERIAL_RESULTS.md).

Two independent fine-tunes start from the full completed
CD-MACE128-D6-075nominal model. Each retains its six-frame temporal head and
trainable geometry-only MACE. Twelve epochs, one seed, global batch 1024, reduced
encoder/head learning rates 1e-5/5e-5; inherited proper predictive likelihood.
No AP selection, material/species input or time/temperature covariates.

| Child | Fitting population | Evaluation | Actual MD interval |
| --- | --- | --- | --- |
| CD-MACE128-D6-Al-FT | 90 native Al sources, 4,561,920 windows | Unchanged fixed Al64 all64 test and spatial scans | 0.75 ps |
| CD-MACE128-D6-Ta-FT | Six old Ta trajectories, 3,061,560 windows | New parent00 velocity branches; shot00 selection, shots01–03 test | 0.70 ps |

Compare each child with its unchanged parent on identical held-out rows. Primary
distance measures are NLL, capped RMSE/MAE and CDF Brier scores. Al additionally
retains warning-distance recall, misses and false alarms at fixed probability
thresholds. Ta reports pointwise confidence reliability, without pretending these
are spatial-scan warning distances. This is supervised material adaptation.

All older Ta trajectories were already used to fit the parent. Its new evaluation
branches have different velocity seeds, but a known starting configuration and
shared ancestry. This measures conditional generalization within that preparation;
it cannot establish performance on independent Ta melts. New branches are not
used in training gradients, and selection/test roles are fixed before scoring.

[Exact definitions](../../docs/metrics/distance_encoder_material_finetune.md) ·
[Execution and outputs](../../docs/distance_encoder_material_finetune.md)
