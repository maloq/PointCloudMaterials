# Original VICReg: relaxed MEAM topology

Variant: `mean_blocks`; seed 20260911; selected epoch 5.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.074683 | 0.5193 | 0.025119 |
| test | 0.084299 | 0.6134 | 0.029826 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
