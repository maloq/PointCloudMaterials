# Original VICReg: relaxed MEAM topology

Variant: `anchor_pca`; seed 20260911; selected epoch 10.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.151933 | 0.0045 | 0.028640 |
| test | 0.126988 | 0.4154 | 0.032122 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
