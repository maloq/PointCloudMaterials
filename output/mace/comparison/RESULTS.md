# MEAM comparison in the original VICReg trainer

| Variant | Test balanced MSE | Within-frame R² | Projector ridge MSE |
|---|---:|---:|---:|
| anchor_vicreg | 0.032914 ± 0.000967 | 0.8490 | 0.032914 ± 0.000967 |
| anchor_pca | 0.091818 ± 0.024869 | 0.5779 | 0.033352 ± 0.000949 |
| anchor_blocks | 0.056853 ± 0.007362 | 0.7396 | 0.032813 ± 0.001004 |
| mean_blocks | 0.076407 ± 0.021371 | 0.6498 | 0.029632 ± 0.000991 |
| transformer_blocks | 0.359791 ± 0.449792 | -0.6520 | 0.030912 ± 0.002492 |
| atom_anchor_blocks | 0.058160 ± 0.009631 | 0.7332 | 0.032833 ± 0.000188 |
| atom_temporal_blocks | 0.073185 ± 0.013753 | 0.6632 | 0.030542 ± 0.000708 |

| Comparison | Error reduction | 95% source-bootstrap interval |
|---|---:|---:|
| anchor_blocks_versus_anchor_pca | 38.08% | [33.84%, 42.79%] |
| mean_blocks_versus_anchor_blocks | -34.39% | [-56.51%, -11.60%] |
| transformer_blocks_versus_mean_blocks | -370.89% | [-463.19%, -308.28%] |
| atom_temporal_blocks_versus_atom_anchor_blocks | -25.83% | [-41.45%, -13.30%] |

| Same ridge readout comparison | Error reduction | 95% source-bootstrap interval |
|---|---:|---:|
| anchor_blocks_versus_anchor_pca | 1.62% | [1.00%, 2.41%] |
| mean_blocks_versus_anchor_blocks | 9.69% | [1.19%, 15.43%] |
| transformer_blocks_versus_mean_blocks | -4.32% | [-8.54%, 0.45%] |
| atom_temporal_blocks_versus_atom_anchor_blocks | 6.98% | [2.60%, 10.75%] |

Intervals resample six whole source trajectories after averaging seeds. Comparisons are exploratory; this cohort was already evaluated in the frozen-MACE experiment.
