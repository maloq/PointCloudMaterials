# Original VICReg: relaxed MEAM topology

Variant: `anchor_pca`; seed 20260910; selected epoch 3.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.065011 | 0.5791 | 0.030974 |
| test | 0.073987 | 0.6592 | 0.034430 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
