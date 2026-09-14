# Original VICReg: relaxed MEAM topology

Variant: `anchor_blocks`; seed 20260910; selected epoch 3.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.047459 | 0.6938 | 0.030274 |
| test | 0.052934 | 0.7572 | 0.034222 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
