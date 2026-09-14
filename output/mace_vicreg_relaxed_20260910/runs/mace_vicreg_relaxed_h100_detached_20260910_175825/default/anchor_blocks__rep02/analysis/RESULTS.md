# Original VICReg: relaxed MEAM topology

Variant: `anchor_blocks`; seed 20260911; selected epoch 10.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.075030 | 0.5097 | 0.028322 |
| test | 0.067166 | 0.6918 | 0.031957 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
