# Original VICReg: relaxed MEAM topology

Variant: `anchor_pca`; seed 20260912; selected epoch 4.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.079153 | 0.4836 | 0.029529 |
| test | 0.074480 | 0.6592 | 0.033503 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
