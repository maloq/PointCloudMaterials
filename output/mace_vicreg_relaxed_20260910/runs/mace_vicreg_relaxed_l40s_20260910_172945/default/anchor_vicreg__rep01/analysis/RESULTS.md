# Original VICReg: relaxed MEAM topology

Variant: `anchor_vicreg`; seed 20260910; selected epoch 3.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.030492 | 0.8025 | 0.030492 |
| test | 0.034267 | 0.8427 | 0.034267 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
