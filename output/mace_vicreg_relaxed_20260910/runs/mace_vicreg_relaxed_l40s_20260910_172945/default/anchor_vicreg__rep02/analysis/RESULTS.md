# Original VICReg: relaxed MEAM topology

Variant: `anchor_vicreg`; seed 20260911; selected epoch 10.

| Split | Balanced TDA MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| val | 0.028159 | 0.8178 | 0.028159 |
| test | 0.032068 | 0.8529 | 0.032068 |

Independent-source splits are preserved. These six test sources were evaluated in the earlier frozen-MACE experiment; this is a controlled follow-up, not a new untouched cohort.

Full block, temperature, frame and history-intervention results: [metrics.json](metrics.json).
