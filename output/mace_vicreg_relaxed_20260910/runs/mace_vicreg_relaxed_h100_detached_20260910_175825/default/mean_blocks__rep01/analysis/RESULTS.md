# Original VICReg: relaxed MEAM topology

Variant: `mean_blocks`; seed 20260910; selected epoch 3.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.091031 | 0.4132 | 0.025816 |
| test | 0.097728 | 0.5520 | 0.030736 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
