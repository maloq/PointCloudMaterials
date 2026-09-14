# Original VICReg: relaxed MEAM topology

Variant: `anchor_vicreg`; seed 20260912; selected epoch 9.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.029078 | 0.8118 | 0.029078 |
| test | 0.032407 | 0.8513 | 0.032407 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
