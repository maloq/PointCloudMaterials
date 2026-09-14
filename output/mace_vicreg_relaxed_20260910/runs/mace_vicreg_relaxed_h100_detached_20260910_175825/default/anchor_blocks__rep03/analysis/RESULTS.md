# Original VICReg: relaxed MEAM topology

Variant: `anchor_blocks`; seed 20260912; selected epoch 9.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.043201 | 0.7224 | 0.028500 |
| test | 0.050460 | 0.7696 | 0.032259 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
